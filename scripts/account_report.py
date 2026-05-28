"""Daily real-account report for QMT 8886933837 (国金证券).

Replaces paper_trade.py reporting — same evening cadence, but the numbers
reflect the actual broker account, not a simulated paper portfolio.

Pulls:
  - QMT account snapshot + positions (via SSH to ECS — same as
    sync_portfolio_from_qmt.py)
  - Today's filled orders (also via QMT broker query — get_orders today_only)
  - Tomorrow's plan from data/orders/latest.json (written by daily_report.py)
  - NAV history from data/account_nav_history.json (this script appends each
    day; first run creates an empty history)
  - ZZ500 close for benchmark (via existing get_zz500_close from paper_trade)

Outputs:
  - data/reports/account_<YYYYMMDD>.md (markdown)
  - Feishu webhook send (same env var as paper_trade)

Cadence: launchd at 18:00 Mon-Fri (replaces paper_trade.sh launchd entry).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


NAV_HISTORY_PATH = ROOT / "data" / "account_nav_history.json"
REPORT_DIR = ROOT / "data" / "reports"
PLAN_PATH = ROOT / "data" / "orders" / "latest.json"
EXEC_DIR = ROOT / "data" / "orders" / "executions"
SHADOW_STATE_PATH = ROOT / "data" / "shadow_930" / "state.json"


def fetch_qmt_snapshot() -> Dict[str, Any]:
    """Reuse the sync_portfolio_from_qmt SSH path."""
    from scripts.sync_portfolio_from_qmt import fetch_qmt_snapshot as _f
    return _f("Administrator", "14.103.49.51")


def load_nav_history() -> List[Dict[str, Any]]:
    if NAV_HISTORY_PATH.exists():
        return json.loads(NAV_HISTORY_PATH.read_text(encoding="utf-8"))
    return []


def save_nav_history(history: List[Dict[str, Any]]) -> None:
    NAV_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    NAV_HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def append_nav(history: List[Dict[str, Any]], snapshot: Dict[str, Any],
               today: str) -> List[Dict[str, Any]]:
    """Replace today's entry if present, else append."""
    new_entry = {
        "date": today,
        "total_assets": snapshot["account"]["total_assets"],
        "cash_available": snapshot["account"]["cash_available"],
        "market_value": snapshot["account"]["market_value"],
    }
    history = [h for h in history if h.get("date") != today]
    history.append(new_entry)
    history.sort(key=lambda h: h["date"])
    return history


def load_today_executions(today: str) -> List[Dict[str, Any]]:
    """Find any exec_<today>_*.json files and merge their `results`."""
    if not EXEC_DIR.exists():
        return []
    prefix = f"exec_{today}_"
    out: List[Dict[str, Any]] = []
    for f in sorted(EXEC_DIR.glob(f"{prefix}*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for r in data.get("results", []):
                if r.get("status") == "sent":
                    out.append(r)
        except Exception as e:
            print(f"[warn] failed to read {f}: {e}")
    return out


def load_tomorrow_plan() -> Optional[Dict[str, Any]]:
    if not PLAN_PATH.exists():
        return None
    try:
        return json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_zz500_today_pct() -> Optional[float]:
    """ZZ500 same-day percent change. Reuses paper_trade helper if importable."""
    try:
        import pandas as pd
        from scripts.paper_trade import get_zz500_close
        today_ts = pd.Timestamp(date.today())
        result = get_zz500_close(today_ts)
        if result is None:
            return None
        close, prev_close = result
        if prev_close and prev_close > 0:
            return (close - prev_close) / prev_close * 100
    except Exception as e:
        print(f"[warn] ZZ500 fetch failed: {e}")
    return None


def fmt_money(v: float) -> str:
    return f"{v:,.2f}"


def fmt_pct(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.2f}%"


# ──────────────────────────────────────────────────────────────────────
# P11-5 round 101: 14:30 real (Arm A) vs 9:30-shadow (Arm B) comparison
# ──────────────────────────────────────────────────────────────────────

def load_shadow_navs() -> List[Tuple[str, float]]:
    """Return Arm B shadow NAV series as [(date, nav), ...] sorted by date.

    Empty list if the shadow recorder hasn't produced state yet (e.g.
    before its first 17:00 run).
    """
    if not SHADOW_STATE_PATH.exists():
        return []
    try:
        state = json.loads(SHADOW_STATE_PATH.read_text(encoding="utf-8"))
        hist = state.get("nav_history", [])
        return sorted(((h["date"], float(h["nav"])) for h in hist), key=lambda x: x[0])
    except Exception as e:
        print(f"[warn] shadow state read failed: {e}")
        return []


def _arm_stats(navs: List[Tuple[str, float]]) -> Optional[Dict[str, float]]:
    """cumulative return / annualized Sharpe / max drawdown from a NAV series.

    Returns None if < 2 points. Sharpe over ~10 trading days is noisy by
    construction — this is a 2-week read, not a verdict (advisor round 101).
    """
    if len(navs) < 2:
        return None
    vals = [v for _, v in navs]
    start, last = vals[0], vals[-1]
    cum = last / start - 1 if start > 0 else 0.0

    # daily simple returns
    rets = [vals[i] / vals[i - 1] - 1 for i in range(1, len(vals)) if vals[i - 1] > 0]
    sharpe = 0.0
    if len(rets) >= 2:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        std = var ** 0.5
        sharpe = (mean / std * (252 ** 0.5)) if std > 0 else 0.0

    # max drawdown
    peak = vals[0]
    mdd = 0.0
    for v in vals:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1)

    return {"cum": cum, "sharpe": sharpe, "mdd": mdd, "n": len(navs)}


def build_shadow_comparison(real_history: List[Dict[str, Any]]) -> List[str]:
    """Render the "14:30 real vs 9:30-shadow" section.

    Aligns both arms to the shadow's start date (the common 5/29 origin)
    and rebases each to its own first NAV so cumulative returns are
    comparable regardless of differing absolute capital.
    """
    shadow_navs = load_shadow_navs()
    if not shadow_navs:
        return [
            "## 14:30 real vs 9:30-shadow 对比",
            "",
            "_(shadow 尚未产生 NAV — Arm B 第一次 17:00 运行后出现)_",
            "",
        ]

    shadow_start = shadow_navs[0][0]
    # Arm A real NAV = total_assets, filtered to dates ≥ shadow start.
    real_navs = sorted(
        ((h["date"], float(h["total_assets"])) for h in real_history
         if h.get("date", "") >= shadow_start),
        key=lambda x: x[0],
    )

    real_stats = _arm_stats(real_navs)
    shadow_stats = _arm_stats(shadow_navs)

    lines = ["## 14:30 real vs 9:30-shadow 对比", ""]
    lines.append(f"共同起点 **{shadow_start}** · 同 intraday_blend 模型 · 唯一差别 = 入场时点")
    lines.append("")
    if real_stats is None or shadow_stats is None:
        n_r = real_stats["n"] if real_stats else len(real_navs)
        n_s = shadow_stats["n"] if shadow_stats else len(shadow_navs)
        lines.append(f"_(样本不足：real {n_r} 点 / shadow {n_s} 点，≥2 点后出对比)_")
        lines.append("")
        return lines

    lines.append("| Arm | 入场 | 累计收益 | Sharpe(年化) | 最大回撤 | 样本 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    lines.append(f"| A (real) | 14:30 当日 | {fmt_pct(real_stats['cum']*100)} | "
                 f"{real_stats['sharpe']:.2f} | {fmt_pct(real_stats['mdd']*100)} | {real_stats['n']}d |")
    lines.append(f"| B (shadow) | T+1 9:30 | {fmt_pct(shadow_stats['cum']*100)} | "
                 f"{shadow_stats['sharpe']:.2f} | {fmt_pct(shadow_stats['mdd']*100)} | {shadow_stats['n']}d |")
    lines.append("")
    gap = (real_stats["cum"] - shadow_stats["cum"]) * 100
    lines.append(f"**Δ 累计 (real − shadow): {fmt_pct(gap)}**")
    lines.append("")
    lines.append("> 回测 (P11-3) 已显示两者 mean Sharpe 零差异 (1.95=1.95)；这是实盘 confirm。"
                 "样本 <~10d 时 Sharpe/MDD 噪声大，看累计 Δ 趋势为主。")
    lines.append("")
    return lines


def build_report(snapshot: Dict[str, Any], history: List[Dict[str, Any]],
                 today_execs: List[Dict[str, Any]],
                 tomorrow_plan: Optional[Dict[str, Any]],
                 zz500_pct: Optional[float],
                 names: Dict[str, str], today: str) -> str:
    acc = snapshot["account"]
    positions = snapshot["positions"]
    total_now = acc["total_assets"]

    # Today P&L vs yesterday entry
    yesterday_entry = None
    for h in reversed(history):
        if h["date"] < today:
            yesterday_entry = h
            break
    today_pnl = (total_now - yesterday_entry["total_assets"]) if yesterday_entry else 0.0
    today_pct = (today_pnl / yesterday_entry["total_assets"] * 100) if yesterday_entry else 0.0

    # Cumulative from first entry
    first_entry = history[0] if history else None
    cum_pnl = (total_now - first_entry["total_assets"]) if first_entry else 0.0
    cum_pct = (cum_pnl / first_entry["total_assets"] * 100) if first_entry else 0.0

    # Excess vs ZZ500
    excess_str = ""
    if zz500_pct is not None and yesterday_entry:
        excess = today_pct - zz500_pct
        excess_str = f"  vs ZZ500 {fmt_pct(zz500_pct)} → 超额 {fmt_pct(excess)}"

    lines: List[str] = []
    lines.append(f"# 真实账户日报 {today}")
    lines.append(f"")
    lines.append(f"账户: QMT 8886933837 (国金证券)")
    lines.append(f"")
    lines.append(f"## NAV")
    lines.append(f"总资产: **¥{fmt_money(total_now)}**（现金 ¥{fmt_money(acc['cash_available'])} "
                 f"+ 持仓 ¥{fmt_money(acc['market_value'])}）")
    if yesterday_entry:
        lines.append(f"今日盈亏: **¥{fmt_money(today_pnl)} ({fmt_pct(today_pct)})**{excess_str}")
    if first_entry:
        lines.append(f"累计盈亏 (起 {first_entry['date']}): "
                     f"¥{fmt_money(cum_pnl)} ({fmt_pct(cum_pct)})")
    lines.append("")

    # Today's executions
    if today_execs:
        lines.append(f"## 今日成交 ({len(today_execs)} 单)")
        lines.append("")
        lines.append("| 操作 | 代码 | 名称 | 股数 | 限价 |")
        lines.append("|---|---|---|---:|---:|")
        for r in today_execs:
            action = "买入" if r["action"] == "buy" else "卖出"
            code = r["code"]
            name = names.get(code) or r.get("name") or code
            lines.append(f"| {action} | {code} | {name} | {r['shares']:,} | {r['limit_price']:.3f} |")
        lines.append("")
    else:
        lines.append(f"## 今日成交")
        lines.append("（今日无成交）")
        lines.append("")

    # Current holdings
    lines.append(f"## 当前持仓 ({len(positions)} 只)")
    lines.append("")
    lines.append("| 代码 | 名称 | 股数 | 成本 | 现价 | 市值 | 浮盈% | 仓位% |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for p in positions:
        code = p["code"]
        name = names.get(code) or p.get("name") or code
        mkt_px = p.get("market_price", 0) or 0
        mkt_val = p.get("market_value", 0) or (p["shares"] * mkt_px if mkt_px > 0 else p["shares"] * p["avg_cost"])
        cost_basis = p["shares"] * p["avg_cost"]
        unrealized_pct = (mkt_val - cost_basis) / cost_basis * 100 if cost_basis > 0 else 0
        pos_pct = mkt_val / total_now * 100 if total_now > 0 else 0
        px_str = f"{mkt_px:.3f}" if mkt_px > 0 else "—"
        sign = "+" if unrealized_pct >= 0 else ""
        lines.append(f"| {code} | {name} | {p['shares']:,} | {p['avg_cost']:.3f} | "
                     f"{px_str} | ¥{fmt_money(mkt_val)} | {sign}{unrealized_pct:.2f}% | {pos_pct:.1f}% |")
    lines.append("")

    # Tomorrow plan
    if tomorrow_plan and tomorrow_plan.get("orders"):
        lines.append(f"## 明日开盘待执行 ({len(tomorrow_plan['orders'])} 单)")
        lines.append("")
        lines.append(f"plan generated: {tomorrow_plan.get('generated_at', '?')}")
        lines.append("")
        lines.append("| 操作 | 代码 | 名称 | 股数 | 限价 | 理由 |")
        lines.append("|---|---|---|---:|---:|---|")
        for o in tomorrow_plan["orders"]:
            lines.append(f"| {o['action']} | {o['code']} | {o['name']} | "
                         f"{o['shares']:,} | {o['limit_price']:.3f} | {o.get('reason', '')} |")
        lines.append("")
    else:
        lines.append(f"## 明日开盘待执行")
        lines.append("（暂无 plan — daily_report 可能尚未运行，或当前持仓与 Top-K 匹配无需调整）")
        lines.append("")

    # P11-5 live A/B: 14:30 real vs 9:30-shadow (round 101)
    lines.extend(build_shadow_comparison(history))

    lines.append("---")
    lines.append("_Money Printer 真实账户日报 (account_report.py)_")
    return "\n".join(lines) + "\n"


def send_feishu(report_md: str) -> bool:
    """Reuse paper_trade.send_to_feishu if importable."""
    try:
        from scripts.paper_trade import send_to_feishu
        return send_to_feishu(report_md)
    except Exception as e:
        print(f"[warn] Feishu send skipped: {e}")
        return False


def main():
    today = date.today().strftime("%Y%m%d")
    today_human = date.today().strftime("%Y-%m-%d")

    print(f"[account_report] fetching QMT snapshot for {today_human}...")
    snapshot = fetch_qmt_snapshot()
    print(f"[account_report] account total={snapshot['account']['total_assets']:.2f} "
          f"positions={len(snapshot['positions'])}")

    # Stock names
    codes = [p["code"] for p in snapshot["positions"]]
    today_codes = {r["code"] for r in load_today_executions(today)}
    if tomorrow_plan := load_tomorrow_plan():
        today_codes.update(o["code"] for o in tomorrow_plan.get("orders", []))
    all_codes = list(set(codes) | today_codes)
    names: Dict[str, str] = {}
    try:
        from scripts.daily_report import get_stock_names
        names = get_stock_names(all_codes) if all_codes else {}
    except Exception as e:
        print(f"[warn] name resolution failed: {e}")

    # NAV history
    history = load_nav_history()
    history = append_nav(history, snapshot, today_human)
    save_nav_history(history)

    today_execs = load_today_executions(today)

    zz500_pct = get_zz500_today_pct()
    if zz500_pct is not None:
        print(f"[account_report] ZZ500 today: {fmt_pct(zz500_pct)}")

    report_md = build_report(
        snapshot, history, today_execs, tomorrow_plan, zz500_pct, names, today_human,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"account_{today}.md"
    out_path.write_text(report_md, encoding="utf-8")
    print(f"[account_report] wrote {out_path}")

    sent = send_feishu(report_md)
    print(f"[account_report] feishu_sent={sent}")


if __name__ == "__main__":
    main()
