"""Weekly full-history qfq refresh.

Why this exists
---------------
Incremental fetching only pulls today's qfq-adjusted bar.  When a stock has
a corporate action (cash dividend, stock split, rights offering), the
adjustment factor changes — but the historical bars in DB were stored under
the OLD factor.  Result: a discontinuity at the ex-date.

Concretely (300033 同花顺, 2026-04-09):
    DB 4-08 close = 319.39   (stored at old scale, never re-adjusted)
    API 4-09 close = 216.41   (new qfq scale after split)
    Computed ret_1d = -32.2%   (FAKE — actually 0% real return)

The model trains on ret_1d and learns "30% drops are normal", which is
poisonous.  Daily reports also silently use these fake drops.

Solution
--------
Once a week, re-pull the FULL qfq history for the active universe and
overwrite all rows.  This re-anchors all historical prices to the current
qfq factor, eliminating the discontinuity.

Universe
--------
- ZZ500 constituents (~500 stocks)
- Portfolio holdings (typically 4-10)
- Index codes (CSI300=000300, ZZ500=000905, etc.)

Schedule
--------
Saturday 10:00 via launchd (com.moneyprinter.qfq).  Markets are closed,
no other collector running, ~30-90 min runtime depending on akshare.

Idempotent: safe to run multiple times.  Uses save_bars_upsert which
INSERT OR REPLACE — no duplicates.

Usage
-----
    python scripts/qfq_refresh.py                # full universe
    python scripts/qfq_refresh.py --codes 000539,300033   # specific
    python scripts/qfq_refresh.py --start 20240101        # custom start
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import List

import pandas as pd
import yaml
from loguru import logger
from sqlalchemy import text

from mp.data.fetcher import (
    _get_daily_bars_em,
    _get_daily_bars_etf,
    _get_daily_bars_sina,
    _is_etf_code,
    _with_retry,
    get_index_constituents,
)
from mp.data.store import DataStore, DEFAULT_DB_URL


def _force_fetch_full_history(code: str, start: str, end: str) -> pd.DataFrame | None:
    """Re-pull the FULL date range from API, bypassing DB freshness shortcuts.

    Mirrors the source-selection logic in get_daily_bars (Sina first, EM
    fallback) but without the "DB already has it" optimization.  Output is
    already normalize_bars-processed by the underlying fetcher.
    """
    if _is_etf_code(code):
        try:
            return _with_retry(lambda: _get_daily_bars_etf(code, start, end))
        except Exception as e:
            logger.debug(f"ETF bars for {code}: {e}")
            return None

    # Sina first
    try:
        df = _with_retry(lambda: _get_daily_bars_sina(code, start, end))
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.debug(f"Sina bars for {code}: {e}, trying EM...")

    # EM fallback
    try:
        return _with_retry(lambda: _get_daily_bars_em(code, start, end))
    except Exception as e:
        logger.debug(f"EM bars for {code}: {e}")
        return None


def _build_universe() -> List[str]:
    """HS300 + ZZ500 + portfolio holdings + key indices (deduplicated).

    Recommendation universe was widened from ZZ500 only to HS300 + ZZ500
    on 2026-05-14 — qfq refresh now covers both so reports never anchor
    on stale large-cap data.
    """
    codes: set[str] = set()

    # HS300 + ZZ500 (the recommendation universe)
    try:
        from mp.data.fetcher import get_recommendation_universe
        universe = get_recommendation_universe()
        codes.update(universe)
        logger.info("Universe: +{} HS300+ZZ500 constituents", len(universe))
    except Exception as e:
        logger.warning("Universe fetch failed: {}", e)

    # Portfolio holdings (stock type only)
    portfolio_path = Path("config/portfolio.yaml")
    if portfolio_path.exists():
        try:
            cfg = yaml.safe_load(portfolio_path.read_text(encoding="utf-8"))
            for h in cfg.get("holdings", []):
                if h.get("type") == "stock" and h.get("code"):
                    codes.add(str(h["code"]).zfill(6))
            logger.info("Universe: +{} portfolio holdings", len(cfg.get("holdings", [])))
        except Exception as e:
            logger.warning("portfolio.yaml read failed: {}", e)

    return sorted(codes)


def _count_diffs(
    store: DataStore, code: str, new_df: pd.DataFrame, threshold: float = 0.005,
) -> tuple[int, float]:
    """Return (n_changed, max_change_pct) compared to current DB.

    A row is "changed" if its close differs by > threshold (0.5% by default)
    from the new fetched value — strong evidence of a qfq adjustment.
    """
    if new_df is None or new_df.empty:
        return 0, 0.0
    with store.engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT date, close FROM daily_bars WHERE code = :c "
            "AND date >= :s AND date <= :e"
        ), {
            "c": code,
            "s": str(new_df["date"].min())[:10],
            "e": str(new_df["date"].max())[:10],
        }).fetchall()
    if not rows:
        return 0, 0.0
    old_map = {str(r[0])[:10]: float(r[1]) if r[1] is not None else None for r in rows}

    n_changed = 0
    max_change = 0.0
    for _, row in new_df.iterrows():
        d = str(row["date"])[:10]
        new_close = float(row["close"]) if pd.notna(row["close"]) else None
        old_close = old_map.get(d)
        if old_close is None or new_close is None or old_close <= 0:
            continue
        change = abs(new_close / old_close - 1.0)
        if change > threshold:
            n_changed += 1
            max_change = max(max_change, change)
    return n_changed, max_change


def refresh_qfq(codes: List[str], start: str, end: str) -> dict:
    """Re-fetch full qfq history for each code and overwrite DB rows.

    Returns summary stats.
    """
    store = DataStore(db_url=DEFAULT_DB_URL)

    n_total = len(codes)
    n_processed = 0
    n_failed = 0
    n_with_changes = 0
    total_rows_changed = 0
    biggest_change = ("", 0.0)

    t0 = time.time()
    logger.info("qfq_refresh: {} codes, {} → {}", n_total, start, end)

    for i, code in enumerate(codes, 1):
        try:
            # Bypass get_daily_bars freshness shortcut — we WANT to re-pull
            # everything to capture qfq adjustment changes.
            new_df = _force_fetch_full_history(code, start, end)
            if new_df is None or new_df.empty:
                n_failed += 1
                continue

            # Diagnose changes vs DB BEFORE overwriting
            n_changed, max_chg = _count_diffs(store, code, new_df)
            if n_changed > 0:
                n_with_changes += 1
                total_rows_changed += n_changed
                if max_chg > biggest_change[1]:
                    biggest_change = (code, max_chg)
                logger.info(
                    "{}: {} rows changed (max delta={:.1%}), upserting",
                    code, n_changed, max_chg,
                )

            # Overwrite — save_bars_upsert is INSERT OR REPLACE
            store.save_bars_upsert(new_df)
            n_processed += 1

        except Exception as e:
            logger.warning("{}: refresh failed: {}", code, e)
            n_failed += 1

        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_total - i) / rate if rate > 0 else 0
            logger.info(
                "Progress: {}/{} ({:.0f}/min, ETA {:.0f}s)",
                i, n_total, rate * 60, eta,
            )

    elapsed = time.time() - t0
    summary = {
        "total": n_total,
        "processed": n_processed,
        "failed": n_failed,
        "with_changes": n_with_changes,
        "rows_changed": total_rows_changed,
        "biggest_delta_code": biggest_change[0],
        "biggest_delta_pct": biggest_change[1] * 100,
        "elapsed_seconds": int(elapsed),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codes", default=None,
        help="Comma-separated codes to refresh (default: full universe)",
    )
    parser.add_argument(
        "--start", default="20230101",
        help="Start date (YYYYMMDD); default 20230101 matches build_latest_features",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date (YYYYMMDD); default = today",
    )
    parser.add_argument(
        "--feishu", action="store_true",
        help="Send summary to Feishu when done",
    )
    args = parser.parse_args()

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        codes = _build_universe()

    end = args.end or date.today().strftime("%Y%m%d")
    summary = refresh_qfq(codes, start=args.start, end=end)

    # Pretty-print summary
    lines = [
        f"# qfq 全量刷新报告 ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        "",
        f"- 处理: {summary['processed']}/{summary['total']} ({summary['failed']} 失败)",
        f"- 有调整: **{summary['with_changes']} 只股票**",
        f"- 累计修正行数: **{summary['rows_changed']}**",
        f"- 最大调整: {summary['biggest_delta_code']} 价格变动 {summary['biggest_delta_pct']:.1f}%",
        f"- 耗时: {summary['elapsed_seconds']}s",
    ]
    report = "\n".join(lines)
    print(report)

    if args.feishu:
        try:
            from scripts.daily_report import send_to_feishu
            send_to_feishu(report)
        except Exception as e:
            logger.warning("Feishu push failed: {}", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
