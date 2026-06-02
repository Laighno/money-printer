"""Monthly OOS Arm B vs EOD comparison report — round 161 guardrail (c).

Reads:
- ``data/orders/executions/exec_<DATE>_intraday_*.json``  (14:30 Arm B bucket)
- ``data/orders/executions/exec_<DATE>_*.json`` w/o intraday tag (9:25 EOD bucket)
- ``data/account_nav_history.json``                       (per-day NAV snapshots)

Writes:
- ``data/reports/oos_arm_b_<YYYYMM>.md`` (idempotent; rerun overwrites)

Usage:
    .venv/bin/python scripts/oos_arm_b_report.py [--month 202605] [--out PATH]

Default month: current month (system date). Designed to be cron-friendly:
exit 0 always (even if no Arm B executions yet — emits a placeholder section
that says "Phase 3 not yet enabled, 0 intraday executions this month").

This is the comparison harness. It does NOT compute realised PnL by
bucket — execution logs do not have fill price/quantity. What it gives:
- count of orders / total notional per bucket per month
- per-day NAV trajectory + month-over-month total return / MDD on the
  account NAV (which mixes both buckets while Arm B is small enough that
  most signal still comes from EOD)
- placeholder lines for OOS-vs-EOD cumulative ret diff (filled in once
  bucket-level NAV separation lands; see follow-up after Phase 3 monitor)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXEC_DIR = PROJECT_ROOT / "data" / "orders" / "executions"
NAV_FILE = PROJECT_ROOT / "data" / "account_nav_history.json"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


# ──────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────

def _month_str(d: date) -> str:
    return f"{d.year:04d}{d.month:02d}"


def _parse_month(s: str) -> Tuple[int, int]:
    if len(s) != 6 or not s.isdigit():
        raise ValueError(f"month must be YYYYMM (got {s!r})")
    return int(s[:4]), int(s[4:6])


def _exec_date_from_filename(name: str) -> Optional[date]:
    """exec_YYYYMMDD_*.json → date(Y, M, D)."""
    parts = name.split("_")
    if len(parts) < 2 or not parts[1].isdigit() or len(parts[1]) != 8:
        return None
    s = parts[1]
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def _is_intraday(name: str) -> bool:
    """File name contains '_intraday_' marker."""
    return "_intraday_" in name


def load_executions(year: int, month: int) -> Tuple[List[dict], List[dict]]:
    """Return (intraday_execs, eod_execs) for the given month."""
    if not EXEC_DIR.exists():
        logger.warning("No execution directory {} — returning empty", EXEC_DIR)
        return [], []
    intraday: List[dict] = []
    eod: List[dict] = []
    for p in sorted(EXEC_DIR.glob("exec_*.json")):
        d = _exec_date_from_filename(p.name)
        if d is None or d.year != year or d.month != month:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Skipping unreadable exec log {}: {}", p.name, e)
            continue
        data["_filename"] = p.name
        data["_date"] = d.isoformat()
        if _is_intraday(p.name):
            intraday.append(data)
        else:
            eod.append(data)
    return intraday, eod


def load_nav_history(year: int, month: int) -> List[dict]:
    """Return NAV snapshots for the month (sorted by date)."""
    if not NAV_FILE.exists():
        return []
    try:
        raw = json.loads(NAV_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("NAV history unreadable: {}", e)
        return []
    if not isinstance(raw, list):
        return []
    out = []
    for snap in raw:
        d = snap.get("date")
        if not isinstance(d, str) or len(d) != 10:
            continue
        try:
            dy, dm, dd = int(d[:4]), int(d[5:7]), int(d[8:10])
        except ValueError:
            continue
        if dy == year and dm == month:
            out.append(snap)
    return sorted(out, key=lambda x: x["date"])


# ──────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────

def _sum_notional(results: Iterable[dict], action: str) -> Tuple[int, float]:
    """Return (n_orders, total_notional_yuan) for SENT orders of given action."""
    n = 0
    notional = 0.0
    for r in results:
        if (r.get("status") == "sent" and r.get("action") == action):
            shares = int(r.get("shares") or 0)
            price = float(r.get("limit_price") or 0)
            n += 1
            notional += shares * price
    return n, notional


def summarize_bucket(execs: List[dict]) -> dict:
    n_days = len({e["_date"] for e in execs})
    buy_n, buy_not = 0, 0.0
    sell_n, sell_not = 0, 0.0
    skipped_n = 0
    arm_b_skipped_n = 0
    for e in execs:
        for r in (e.get("results") or []):
            if r.get("status") == "sent" and r.get("action") == "buy":
                buy_n += 1
                buy_not += int(r.get("shares") or 0) * float(r.get("limit_price") or 0)
            elif r.get("status") == "sent" and r.get("action") == "sell":
                sell_n += 1
                sell_not += int(r.get("shares") or 0) * float(r.get("limit_price") or 0)
            elif r.get("status") == "skipped":
                skipped_n += 1
                if "arm_b_cap" in (r.get("note") or ""):
                    arm_b_skipped_n += 1
    return {
        "n_days_active": n_days,
        "n_buys_sent": buy_n,
        "notional_bought_yuan": round(buy_not, 2),
        "n_sells_sent": sell_n,
        "notional_sold_yuan": round(sell_not, 2),
        "n_skipped_total": skipped_n,
        "n_skipped_arm_b_cap": arm_b_skipped_n,
    }


def nav_stats(nav_snaps: List[dict]) -> dict:
    """Compute start / end / max / min / cumulative return / MDD."""
    if not nav_snaps:
        return {"n_days": 0}
    nav_vals = [float(s.get("total_assets", 0)) for s in nav_snaps]
    nav_vals = [v for v in nav_vals if v > 0]
    if not nav_vals:
        return {"n_days": 0}
    start = nav_vals[0]
    end = nav_vals[-1]
    peak = nav_vals[0]
    max_dd = 0.0
    for v in nav_vals:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd
    cum_ret = (end - start) / start if start > 0 else 0
    return {
        "n_days": len(nav_snaps),
        "nav_start": round(start, 2),
        "nav_end": round(end, 2),
        "nav_peak": round(peak, 2),
        "nav_min": round(min(nav_vals), 2),
        "cumulative_return": round(cum_ret, 4),
        "max_drawdown": round(max_dd, 4),
        "first_date": nav_snaps[0].get("date"),
        "last_date": nav_snaps[-1].get("date"),
    }


# ──────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────

def render_report(year: int, month: int,
                  intraday_execs: List[dict],
                  eod_execs: List[dict],
                  nav_snaps: List[dict]) -> str:
    arm_b = summarize_bucket(intraday_execs)
    eod = summarize_bucket(eod_execs)
    nav = nav_stats(nav_snaps)

    month_str = f"{year:04d}-{month:02d}"

    lines = []
    lines.append(f"# OOS Arm B vs EOD 月度对比 · {month_str}")
    lines.append("")
    lines.append(f"_自动生成于 {datetime.now().isoformat(timespec='seconds')}_ · "
                  f"round 161 guardrail (c)")
    lines.append("")

    # Bucket summary
    lines.append("## 1. 执行汇总")
    lines.append("")
    lines.append("| Bucket | 当月活跃天数 | 买单(sent) | 买入名义 | 卖单(sent) | 卖出名义 | 总跳过 | 其中 Arm B cap 跳过 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| OOS Arm B (14:30) | {arm_b['n_days_active']} | "
        f"{arm_b['n_buys_sent']} | ¥{arm_b['notional_bought_yuan']:,.0f} | "
        f"{arm_b['n_sells_sent']} | ¥{arm_b['notional_sold_yuan']:,.0f} | "
        f"{arm_b['n_skipped_total']} | {arm_b['n_skipped_arm_b_cap']} |"
    )
    lines.append(
        f"| EOD (9:25) | {eod['n_days_active']} | "
        f"{eod['n_buys_sent']} | ¥{eod['notional_bought_yuan']:,.0f} | "
        f"{eod['n_sells_sent']} | ¥{eod['notional_sold_yuan']:,.0f} | "
        f"{eod['n_skipped_total']} | n/a |"
    )
    lines.append("")

    if arm_b["n_days_active"] == 0:
        lines.append("> ⚠ Arm B 当月无成交。可能原因：Phase 3 (14:30 task) 尚未启用，"
                      "或者所有候选都被 guardrail (b) 价/流动性过滤掉。")
        lines.append("")

    # NAV trajectory
    lines.append("## 2. 账户 NAV 走势 (混合 — 含两个 bucket)")
    lines.append("")
    if nav["n_days"] == 0:
        lines.append("_无 NAV 数据_")
    else:
        lines.append(f"- 区间: {nav['first_date']} → {nav['last_date']} ({nav['n_days']} 天)")
        lines.append(f"- 起始 NAV: ¥{nav['nav_start']:,.2f}")
        lines.append(f"- 期末 NAV: ¥{nav['nav_end']:,.2f}")
        lines.append(f"- 区间累计收益: **{nav['cumulative_return']*100:+.2f}%**")
        lines.append(f"- 区间 Max Drawdown: **{nav['max_drawdown']*100:+.2f}%**")
        lines.append(f"- 期间 NAV peak: ¥{nav['nav_peak']:,.2f}; 谷: ¥{nav['nav_min']:,.2f}")
    lines.append("")

    # Per-day NAV table (compact)
    if nav["n_days"]:
        lines.append("### 每日 NAV 快照")
        lines.append("")
        lines.append("| 日期 | total | cash | market_value |")
        lines.append("|---|---:|---:|---:|")
        for s in nav_snaps:
            lines.append(
                f"| {s.get('date')} | "
                f"¥{float(s.get('total_assets', 0)):,.0f} | "
                f"¥{float(s.get('cash_available', 0)):,.0f} | "
                f"¥{float(s.get('market_value', 0)):,.0f} |"
            )
        lines.append("")

    # Caveats
    lines.append("## 3. 限制与跟进")
    lines.append("")
    lines.append("- 本报告**未**按 bucket 拆分 NAV — 因执行日志没有 fill 价/量，"
                  "Arm B/EOD 的 PnL 贡献需要后续在 NAV history 里加 bucket 标记后才能精确分离。")
    lines.append("- 当前可用的对比维度：成交单量、买卖名义、跳过统计、混合 NAV。")
    lines.append("- 跟踪 guardrail (a) 触发频次：见上表 \"Arm B cap 跳过\" 列。任一月超过 N=5 次"
                  "意味着 20000 元上限对策略约束过紧，需 user 显式批准后再放大上限。")
    lines.append("- 跟踪 guardrail (d) 累计 -5pp 跑输：另开 monitor 脚本 (round 161 (d))，本报告不重复。")
    lines.append("")

    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", default=None,
                    help="YYYYMM (default: current month)")
    ap.add_argument("--out", default=None,
                    help="output path (default: data/reports/oos_arm_b_<YYYYMM>.md)")
    args = ap.parse_args(argv)

    if args.month:
        year, month = _parse_month(args.month)
    else:
        today = date.today()
        year, month = today.year, today.month

    intraday, eod = load_executions(year, month)
    nav = load_nav_history(year, month)

    md = render_report(year, month, intraday, eod, nav)

    out_path = Path(args.out) if args.out else REPORTS_DIR / f"oos_arm_b_{year:04d}{month:02d}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    logger.info("Wrote {} ({} chars, intraday={} eod={} nav_snaps={})",
                out_path, len(md), len(intraday), len(eod), len(nav))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
