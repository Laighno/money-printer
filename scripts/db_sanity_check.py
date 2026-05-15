"""Daily DB integrity sentinel.

Runs a battery of invariant checks on market.db and alerts via Feishu if
any are violated.  Designed to catch silent data corruption (unit drift,
unit mismatch, partial-bar persistence, stale data) BEFORE the next
report run uses the bad data for predictions.

Schedule: launchd nightly (e.g. 23:00) AFTER all collector runs are done.

Exit codes:
  0 — all invariants pass
  1 — at least one invariant failed (alert sent)

Each check returns a tuple (passed: bool, name: str, detail: str).
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from typing import Callable, List, Tuple

from loguru import logger
from sqlalchemy import text

from mp.data.store import DataStore, DEFAULT_DB_URL


CheckResult = Tuple[bool, str, str]


def check_no_turnover_pollution(store: DataStore) -> CheckResult:
    """Turnover must be a decimal fraction (≤ 1.0)."""
    with store.engine.connect() as conn:
        n = conn.execute(text(
            "SELECT COUNT(*) FROM daily_bars WHERE turnover > 1.0"
        )).scalar()
    if n == 0:
        return True, "turnover_decimal_only", "0 rows with turnover > 1"
    return False, "turnover_decimal_only", (
        f"{n} rows with turnover > 1.0 — run "
        f"UPDATE daily_bars SET turnover=turnover/100 WHERE turnover > 1.0"
    )


def check_amount_volume_consistency(store: DataStore, max_bad: int = 100) -> CheckResult:
    """amount / (volume × close) ∈ [0.3, 50.0]; ratios > 50 = unit bug.

    Bound 50 (not 3) accommodates qfq-adjusted close where the historical
    raw price was higher: ratio reflects the adjustment factor.  Anything
    above 50 is the unmistakable 100× volume mismatch."""
    with store.engine.connect() as conn:
        n = conn.execute(text("""
            SELECT COUNT(*) FROM daily_bars
            WHERE volume > 0 AND close > 0 AND amount IS NOT NULL
              AND (amount / (volume * close) < 0.3
                   OR amount / (volume * close) > 50.0)
        """)).scalar()
    if n < max_bad:
        return True, "amount_volume_consistency", f"{n} bad rows (< {max_bad} threshold)"
    return False, "amount_volume_consistency", (
        f"{n} rows with amount/(volume*close) outside [0.3, 50] — likely "
        f"100× volume unit mismatch.  Backfill: UPDATE daily_bars SET "
        f"volume=volume*100 WHERE amount/(volume*close) > 50"
    )


def check_no_future_bars(store: DataStore) -> CheckResult:
    """No rows dated in the future (calendar safety)."""
    today = date.today().isoformat()
    with store.engine.connect() as conn:
        n = conn.execute(text(
            "SELECT COUNT(*) FROM daily_bars WHERE date > :t"
        ), {"t": today}).scalar()
    if n == 0:
        return True, "no_future_bars", "0 future-dated rows"
    return False, "no_future_bars", f"{n} rows dated after {today}"


def check_recent_data_freshness(store: DataStore, max_lag_days: int = 5) -> CheckResult:
    """Most recent bar must be within max_lag_days of today (allows weekends)."""
    with store.engine.connect() as conn:
        latest = conn.execute(text("SELECT MAX(date) FROM daily_bars")).scalar()
    if not latest:
        return False, "data_freshness", "daily_bars is empty"
    latest_date = datetime.strptime(latest[:10], "%Y-%m-%d").date()
    lag = (date.today() - latest_date).days
    if lag <= max_lag_days:
        return True, "data_freshness", f"latest bar = {latest} ({lag} days ago)"
    return False, "data_freshness", (
        f"latest bar = {latest} is {lag} days old (> {max_lag_days})"
    )


def check_no_extreme_returns(store: DataStore) -> CheckResult:
    """No single-day return > 22% **with active intraday movement**.

    A-share daily limits:
      - Main board (000xxx, 600xxx): ±10%
      - 创业板 (300xxx) and 科创板 (688xxx): ±20%
      - 北交所 (8xxxxx): ±30%

    Big "drops" with intraday range < 5% are almost always qfq adjustment
    artefacts (ex-dividend/split day): the open price was already at the
    new scale, but DB's prev_close is at the old scale.  Filter those out
    so the alert only fires on truly anomalous days (data corruption,
    flipped prices, etc.).
    """
    with store.engine.connect() as conn:
        rows = conn.execute(text("""
            WITH paired AS (
                SELECT a.code, a.date, a.open, a.high, a.low, a.close,
                       (SELECT close FROM daily_bars b
                        WHERE b.code = a.code AND b.date < a.date
                        ORDER BY b.date DESC LIMIT 1) AS prev_close
                FROM daily_bars a
                WHERE a.date >= date('now', '-30 days')
            )
            SELECT code, date, close, prev_close,
                   ((close / prev_close) - 1.0) AS ret,
                   ((high - low) / NULLIF(low, 0)) AS intraday_range
            FROM paired
            WHERE prev_close > 0 AND close > 0
              AND (close / prev_close > 1.22 OR close / prev_close < 0.78)
              AND substr(code, 1, 1) != '8'   -- exclude 北交所 (±30%)
              AND ((high - low) / NULLIF(low, 0)) > 0.05   -- exclude qfq artefacts
            LIMIT 5
        """)).fetchall()
    if not rows:
        return True, "no_extreme_returns", (
            "0 anomalous returns in last 30d (qfq artefacts excluded)"
        )
    sample = ", ".join(
        f"{r[0]}@{r[1]}={r[4]*100:.1f}% (intraday {r[5]*100:.1f}%)"
        for r in rows
    )
    return False, "no_extreme_returns", (
        f"{len(rows)}+ rows with |ret_1d| > 22% AND intraday > 5% in last 30d — "
        f"likely real anomaly or data corruption.  Sample: {sample}"
    )


def check_holdings_have_recent_data(store: DataStore, max_lag_days: int = 5) -> CheckResult:
    """Each portfolio holding must have a recent bar (catch silently broken stocks)."""
    import yaml
    from pathlib import Path
    p = Path("config/portfolio.yaml")
    if not p.exists():
        return True, "holdings_freshness", "no portfolio.yaml — skipped"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    holdings = [h for h in cfg.get("holdings", []) if h.get("type") == "stock"]
    if not holdings:
        return True, "holdings_freshness", "no stock holdings in portfolio"

    stale = []
    with store.engine.connect() as conn:
        for h in holdings:
            latest = conn.execute(text(
                "SELECT MAX(date) FROM daily_bars WHERE code = :code"
            ), {"code": h["code"]}).scalar()
            if not latest:
                stale.append(f"{h['name']}({h['code']})=missing")
                continue
            lag = (date.today() - datetime.strptime(latest[:10], "%Y-%m-%d").date()).days
            if lag > max_lag_days:
                stale.append(f"{h['name']}({h['code']})={lag}d")
    if not stale:
        return True, "holdings_freshness", f"{len(holdings)} holdings all fresh"
    return False, "holdings_freshness", f"stale holdings: {', '.join(stale)}"


def main() -> int:
    store = DataStore(db_url=DEFAULT_DB_URL)
    checks: List[Callable[[DataStore], CheckResult]] = [
        check_no_turnover_pollution,
        check_amount_volume_consistency,
        check_no_future_bars,
        check_recent_data_freshness,
        check_no_extreme_returns,
        check_holdings_have_recent_data,
    ]

    results = [chk(store) for chk in checks]
    n_pass = sum(1 for r in results if r[0])
    n_fail = len(results) - n_pass

    lines = [f"# DB 哨兵报告 ({date.today().isoformat()})", ""]
    lines.append(f"- 通过: {n_pass}/{len(results)}")
    lines.append(f"- 失败: {n_fail}")
    lines.append("")
    lines.append("| 检查 | 状态 | 详情 |")
    lines.append("|---|---|---|")
    for ok, name, detail in results:
        icon = "✅" if ok else "❌"
        lines.append(f"| {name} | {icon} | {detail} |")
    report = "\n".join(lines)
    print(report)

    if n_fail > 0:
        logger.error("DB sanity check FAILED: {} of {} checks failed", n_fail, len(results))
        # Best-effort Feishu notify (skip if lark-cli not installed)
        try:
            from scripts.daily_report import send_to_feishu
            send_to_feishu(f"⚠️ DB 哨兵告警 ({n_fail} 项失败)\n\n{report}")
        except Exception as e:
            logger.warning("Feishu notify failed: {}", e)
        return 1

    logger.info("DB sanity check OK: all {} checks passed", len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
