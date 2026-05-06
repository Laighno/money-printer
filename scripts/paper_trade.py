"""Paper-trading mode — autonomous BlendRanker-driven simulation.

Daily flow (16:00 cron):
  1. Load persistent state from data/paper_trade/state.json
  2. Execute yesterday's pending trades at TODAY's open price (T+1 model)
  3. Mark to market at TODAY's close → compute NAV
  4. Score ZZ500 universe with the production BlendRanker
  5. Pick Top-10 + cost-aware swap (same logic as walk_forward_backtest)
  6. Compute trades to execute TOMORROW's open
  7. Generate report (markdown + Feishu)
  8. Persist state

Decision口径完全跟 walk_forward 回测保持一致（BASELINE.md §一）：
  • Top-K = 10, equal-weight ~30k each (300k initial)
  • REBALANCE_POLICY = on_change（不在意权重漂移）
  • COST_AWARE_REBALANCE: 候选 ml_score 比当前持仓最差的 + 实际 swap cost 才换
  • 不做大盘择时 / 不做个股止损 / 不动权重再平衡（BASELINE ❌#7）
  • Universe = 中证500成分股（不交易 ETF）
  • 数据降级门：基本面数据 >50% 缺失 → 当天不换仓

State format (JSON, human-readable for debug):
  {
    "started": "2026-04-29",
    "initial_capital": 300000,
    "cash": 28341.50,
    "positions": [{"code": ..., "shares": ..., "avg_cost": ..., "entry_date": ...}],
    "pending_trades": [{"action": "buy"|"sell", "code": ..., ...}],
    "nav_history": [{"date": ..., "cash": ..., "positions_value": ..., "nav": ...}],
    "trade_log": [...]
  }
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from mp.account.broker import BrokerPosition, FeeSchedule, SimulatedBroker
from mp.backtest.ml_backtest import _cost_aware_select
from mp.data.fetcher import get_daily_bars, get_index_constituents
from mp.ml.dataset import build_latest_features
from mp.ml.model import BlendRanker

# ──────────────────────────────────────────────────────────────────────
# Configuration (mirrors BASELINE.md §一)
# ──────────────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 300_000
TOP_K = 10
SLIPPAGE_BPS = 5
COMMISSION_BPS = 3

UNIVERSE_NAME = "zz500"
MODEL_PATH_PREFIX = "data/blend"
STATE_PATH = Path("data/paper_trade/state.json")
REPORTS_DIR = Path("data/paper_trade/reports")
DATA_QUALITY_GATE = 0.5   # if <50% of universe has fundamentals → freeze换仓

# ──────────────────────────────────────────────────────────────────────
# State IO
# ──────────────────────────────────────────────────────────────────────

def _new_state(today: str) -> dict:
    return {
        "version": 1,
        "started": today,
        "initial_capital": INITIAL_CAPITAL,
        "cash": float(INITIAL_CAPITAL),
        "positions": [],
        "pending_trades": [],
        "nav_history": [],
        "trade_log": [],
    }


def load_state(today: str) -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception as e:
            logger.error("Failed to read state: {} — refusing to overwrite", e)
            raise
    logger.info("No prior state. Initializing with {:,.0f} cash on {}",
                INITIAL_CAPITAL, today)
    return _new_state(today)


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    logger.info("State saved: {} positions, cash={:,.0f}",
                len(state["positions"]), state["cash"])


# ──────────────────────────────────────────────────────────────────────
# Broker reconstruction
# ──────────────────────────────────────────────────────────────────────

def state_to_broker(state: dict) -> SimulatedBroker:
    fees = FeeSchedule(slippage_bps=SLIPPAGE_BPS, commission_bps=COMMISSION_BPS)
    broker = SimulatedBroker(initial_capital=state["initial_capital"], fees=fees, silent=True)
    broker.cash = float(state["cash"])
    broker.trade_log = list(state.get("trade_log", []))
    for p in state["positions"]:
        broker.positions[p["code"]] = BrokerPosition(
            code=p["code"],
            shares=int(p["shares"]),
            avg_cost=float(p["avg_cost"]),
            current_price=float(p.get("current_price", p["avg_cost"])),
            peak_price=float(p.get("peak_price", p["avg_cost"])),
            entry_date=p.get("entry_date", ""),
        )
    return broker


def broker_to_state(state: dict, broker: SimulatedBroker) -> None:
    """Mutate state in-place from broker (post-trades)."""
    state["cash"] = float(broker.cash)
    state["positions"] = [
        {
            "code": p.code,
            "shares": int(p.shares),
            "avg_cost": float(p.avg_cost),
            "current_price": float(p.current_price),
            "peak_price": float(p.peak_price),
            "entry_date": p.entry_date,
        }
        for p in broker.positions.values()
    ]
    state["trade_log"] = list(broker.trade_log)


# ──────────────────────────────────────────────────────────────────────
# Price lookups
# ──────────────────────────────────────────────────────────────────────

def get_today_bar(code: str, today: pd.Timestamp) -> Optional[dict]:
    """Return {open, close, amount} for *today* if available, else None."""
    df = get_daily_bars(code, today.strftime("%Y%m%d"), today.strftime("%Y%m%d"))
    if df is None or df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    row = df[df["date"] == today]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "open": float(r["open"]),
        "close": float(r["close"]),
        "high": float(r["high"]),
        "low": float(r["low"]),
        "amount": float(r.get("amount", 0)),
    }


def get_recent_amount_avg(code: str, today: pd.Timestamp, n: int = 20) -> Optional[float]:
    """20-day rolling avg amount for ADV (excluding today)."""
    start = (today - pd.Timedelta(days=n + 10)).strftime("%Y%m%d")
    end = (today - pd.Timedelta(days=1)).strftime("%Y%m%d")
    df = get_daily_bars(code, start, end)
    if df is None or df.empty or "amount" not in df.columns:
        return None
    tail = df.tail(n)
    if tail.empty:
        return None
    return float(tail["amount"].mean())


# ──────────────────────────────────────────────────────────────────────
# Trade execution
# ──────────────────────────────────────────────────────────────────────

def execute_pending(state: dict, broker: SimulatedBroker, today: pd.Timestamp,
                    bar_cache: dict) -> list[dict]:
    """Execute yesterday's pending trades at TODAY's open price.
    Returns list of executed trade summaries.
    """
    executed: list[dict] = []
    pending = list(state.get("pending_trades", []))
    if not pending:
        return executed

    today_str = today.strftime("%Y-%m-%d")
    # Sells first to free cash, then buys
    sells = [t for t in pending if t["action"] == "sell"]
    buys = [t for t in pending if t["action"] == "buy"]

    for t in sells:
        code = t["code"]
        bar = bar_cache.get(code) or get_today_bar(code, today)
        if bar is None:
            logger.warning("No today bar for {} — sell skipped, will retry tomorrow", code)
            continue
        bar_cache[code] = bar
        adv = get_recent_amount_avg(code, today)
        result = broker.sell(code=code, price=bar["open"], date=today_str,
                             action="SELL_OPEN", adv=adv)
        if result:
            executed.append({**result, "_planned_reason": t.get("reason", "")})

    for t in buys:
        code = t["code"]
        bar = bar_cache.get(code) or get_today_bar(code, today)
        if bar is None:
            logger.warning("No today bar for {} — buy skipped, candidate dropped", code)
            continue
        bar_cache[code] = bar
        adv = get_recent_amount_avg(code, today)
        target = float(t.get("target_value", broker.total_value / TOP_K))
        result = broker.buy(code=code, price=bar["open"], target_value=target,
                            date=today_str, action="BUY_OPEN", adv=adv)
        if result:
            executed.append({**result, "_planned_reason": t.get("reason", "")})

    state["pending_trades"] = []
    return executed


def mark_to_market(broker: SimulatedBroker, today: pd.Timestamp,
                   bar_cache: dict) -> None:
    """Update each position's current_price to today's close."""
    prices: dict[str, float] = {}
    for code in list(broker.positions.keys()):
        bar = bar_cache.get(code) or get_today_bar(code, today)
        if bar is None:
            logger.warning("No close price for {}", code)
            continue
        bar_cache[code] = bar
        prices[code] = bar["close"]
    if prices:
        broker.update_prices(prices)


# ──────────────────────────────────────────────────────────────────────
# Decision: pick Top-K with cost-aware swap
# ──────────────────────────────────────────────────────────────────────

def score_universe_with_blend(ranker: BlendRanker, today: pd.Timestamp,
                              held_codes: set[str]) -> tuple[pd.DataFrame, float]:
    """Return (scored_df, data_quality) for the ZZ500 universe + held codes."""
    universe = set(get_index_constituents(UNIVERSE_NAME))
    universe.update(held_codes)   # always score current holdings even if dropped from index
    codes = sorted(universe)
    logger.info("Building features for {} stocks...", len(codes))
    features = build_latest_features(codes, include_fundamentals=True)
    if features.empty:
        return pd.DataFrame(), 0.0
    dq = float(features.attrs.get("_data_quality", 1.0))
    scores = ranker.predict(features)
    raw = ranker.predict_raw(features)
    df = pd.DataFrame({
        "code": features["code"].astype(str).str.zfill(6).values,
        "ml_score": scores,
        "raw_excess": raw,
    }).sort_values("ml_score", ascending=False).reset_index(drop=True)
    df["rank_pct"] = df["ml_score"].rank(pct=True)
    return df, dq


def plan_tomorrow_trades(broker: SimulatedBroker, scored: pd.DataFrame,
                         today: pd.Timestamp, dq: float) -> tuple[list[dict], list[tuple[str, float]]]:
    """Decide what to buy/sell tomorrow at open.
    Returns (pending_trades, selected_top_k).
    """
    if dq < DATA_QUALITY_GATE:
        logger.warning("Data quality {:.0%} < {:.0%} gate → freezing换仓 today",
                       dq, DATA_QUALITY_GATE)
        return [], []

    # Build (code, score) tuples sorted by score desc
    scored_tuples = list(zip(scored["code"], scored["ml_score"]))

    # Cost-aware Top-K (mimics walk_forward exactly)
    selected = _cost_aware_select(
        scored_tuples,
        TOP_K,
        broker,
        today,
        adv_lookup={},   # empty → falls back to flat 5bps slippage; OK for our scale
    )
    selected_codes = {c for c, _ in selected}
    held_codes = set(broker.positions.keys())

    pending: list[dict] = []

    # 1. Sells: held but not in selection
    score_map = dict(scored_tuples)
    for code in held_codes:
        if code not in selected_codes:
            pending.append({
                "action": "sell",
                "code": code,
                "reason": f"跌出 Top-{TOP_K}（ml_score={score_map.get(code, float('nan')):.4f}）",
            })

    # 2. Buys: in selection but not held — conviction-weighted (BASELINE 2026-04-29).
    #    Weight ∝ predicted excess (raw_excess from BlendRanker primary), with a
    #    +0.5pp epsilon floor so even the weakest Top-K name gets a token weight.
    if held_codes != selected_codes:
        excess_map = dict(zip(scored["code"], scored["raw_excess"]))
        total_value_now = broker.total_value
        # If first-day fresh build: total_value = cash = INITIAL_CAPITAL
        # If existing portfolio: total_value reflects current NAV (cash + positions)
        # All Top-K codes get a buy entry sized by conviction
        epsilon = 0.005   # 0.5pp floor
        adj = {c: max(excess_map.get(c, 0.0), 0.0) + epsilon for c in selected_codes}
        s = sum(adj.values())
        weights = {c: (adj[c] / s if s > 0 else 1.0 / len(selected_codes))
                   for c in selected_codes}

        for code in selected_codes - held_codes:
            target_value = total_value_now * weights[code]
            ex_pct = excess_map.get(code, float('nan')) * 100
            pending.append({
                "action": "buy",
                "code": code,
                "target_value": float(target_value),
                "weight": float(weights[code]),
                "reason": f"新进 Top-{TOP_K}（excess={ex_pct:+.2f}%，仓位 {weights[code]*100:.1f}%）",
            })

    return pending, selected


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────

def append_nav(state: dict, today: pd.Timestamp, broker: SimulatedBroker) -> dict:
    positions_value = sum(p.market_value for p in broker.positions.values())
    nav = broker.cash + positions_value
    entry = {
        "date": today.strftime("%Y-%m-%d"),
        "cash": float(broker.cash),
        "positions_value": float(positions_value),
        "nav": float(nav),
    }
    # Replace if same date, else append
    history = state.get("nav_history", [])
    if history and history[-1]["date"] == entry["date"]:
        history[-1] = entry
    else:
        history.append(entry)
    state["nav_history"] = history
    return entry


def _resolve_names(codes: list[str]) -> Dict[str, str]:
    """Look up Chinese stock names for *codes* via Sina quote API
    (reuses daily_report.get_stock_names).  Returns {code: name}.  Missing
    codes default to themselves."""
    try:
        from scripts.daily_report import get_stock_names
        return get_stock_names([str(c).zfill(6) for c in codes])
    except Exception as e:
        logger.debug("Name resolution skipped: {}", e)
        return {c: c for c in codes}


def format_report(state: dict, scored: pd.DataFrame, selected: list,
                  executed: list, pending: list, today: pd.Timestamp,
                  bench_today_close: Optional[float] = None,
                  bench_yesterday_close: Optional[float] = None) -> str:
    nav_hist = state["nav_history"]
    nav_today = nav_hist[-1]["nav"] if nav_hist else state["initial_capital"]
    nav_prev = nav_hist[-2]["nav"] if len(nav_hist) >= 2 else state["initial_capital"]
    daily_ret = nav_today / nav_prev - 1 if nav_prev > 0 else 0.0
    total_ret = nav_today / state["initial_capital"] - 1

    bench_daily = (
        bench_today_close / bench_yesterday_close - 1
        if bench_today_close and bench_yesterday_close else None
    )

    # Resolve names for everything we'll reference in the report
    all_codes = set()
    for t in executed:
        all_codes.add(t.get("code", ""))
    for p in state.get("positions", []):
        all_codes.add(p["code"])
    for t in pending:
        all_codes.add(t.get("code", ""))
    for code, _ in selected:
        all_codes.add(code)
    name_map = _resolve_names([c for c in all_codes if c])

    def _fmt(code: str) -> str:
        nm = name_map.get(code, code)
        return f"{nm}({code})" if nm and nm != code else code

    # Aggregate frictions (today + cumulative).
    # Backward compatibility: trades booked before 2026-04-30 lack the
    # friction breakdown.  Fall back to |notional - value| (still correct in
    # total $, but slippage/commission/tax breakdown unavailable for those).
    def _trade_friction(t: dict) -> tuple[float, float, float, float]:
        """Return (slip, comm, tax, total) for a single trade."""
        if "total_friction" in t:
            return (t.get("slippage_cost", 0), t.get("commission", 0),
                    t.get("stamp_tax", 0), t["total_friction"])
        notional = t.get("shares", 0) * t.get("price", 0)
        value = t.get("value", notional)
        # buy: value > notional (cost + fee); sell: value < notional (proceeds - fee)
        est = abs(value - notional)
        return (float("nan"), float("nan"), float("nan"), est)

    def _agg(trades):
        slip = comm = tax = total = turnover = 0.0
        for t in trades:
            s, c, x, tot = _trade_friction(t)
            if not (s != s):  # not NaN
                slip += s; comm += c; tax += x
            total += tot
            turnover += t.get("shares", 0) * t.get("price", 0)
        return slip, comm, tax, total, turnover

    today_slip, today_comm, today_tax, today_friction, today_turnover = _agg(executed)
    cumul_slip, cumul_comm, cumul_tax, cumul_friction, cumul_turnover = _agg(state.get("trade_log", []))

    today_friction_bps = (today_friction / today_turnover * 10_000) if today_turnover > 0 else 0
    cumul_friction_bps = (cumul_friction / cumul_turnover * 10_000) if cumul_turnover > 0 else 0
    cumul_friction_nav_pct = cumul_friction / state["initial_capital"] * 100

    # Detect whether any trade lacks the breakdown
    has_legacy = any("total_friction" not in t for t in state.get("trade_log", []))

    # Excess vs benchmark today
    excess_today = (daily_ret - bench_daily) if bench_daily is not None else None

    lines = []
    lines.append(f"# 模拟仓位日报 {today.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append(f"**起始**: {state['started']} · **本金**: {state['initial_capital']:,.0f}")
    lines.append("")
    lines.append("## NAV")
    lines.append(f"- 总资产: **{nav_today:,.2f}**（现金 {state['cash']:,.2f} + 持仓 {nav_hist[-1]['positions_value']:,.2f}）")
    pnl_today = nav_today - nav_prev
    pnl_total = nav_today - state["initial_capital"]
    lines.append(f"- 今日盈亏: **{pnl_today:+,.2f}**（{daily_ret:+.2%}）" + (
        f"  vs ZZ500 {bench_daily:+.2%} → 超额 **{excess_today:+.2%}**" if excess_today is not None else ""))
    lines.append(f"- 累计盈亏: **{pnl_total:+,.2f}**（{total_ret:+.2%}）")
    lines.append("")

    # Friction summary
    if today_friction > 0 or cumul_friction > 0:
        lines.append("## 交易磨损")
        def _bd(slip, comm, tax):
            """Render breakdown; show '—' for unavailable (legacy trades)."""
            parts = []
            if slip == slip:  # not NaN
                parts.append(f"滑点 {slip:,.2f}")
            if comm == comm:
                parts.append(f"佣金 {comm:,.2f}")
            if tax == tax:
                parts.append(f"印花税 {tax:,.2f}")
            return " + ".join(parts) if parts else "（旧版本，明细不可用）"
        if today_friction > 0:
            lines.append(f"- 今日磨损: **¥{today_friction:,.2f}**（{_bd(today_slip, today_comm, today_tax)}）")
            lines.append(f"  - 今日成交额 ¥{today_turnover:,.0f}，磨损率 **{today_friction_bps:.1f} bps**")
        if cumul_friction > 0:
            lines.append(f"- 累计磨损: **¥{cumul_friction:,.2f}**（{_bd(cumul_slip, cumul_comm, cumul_tax)}）")
            lines.append(f"  - 累计成交额 ¥{cumul_turnover:,.0f}，平均磨损率 {cumul_friction_bps:.1f} bps")
            lines.append(f"  - 占本金 **{cumul_friction_nav_pct:.2f}%**")
        if has_legacy:
            lines.append("  - ⓘ 部分历史交易在新磨损追踪上线前发生，明细 (滑点/佣金/印花税) 不可拆分")
        lines.append("")

    # Today's executed trades (yesterday's plan filled at today's open)
    if executed:
        lines.append("## 今日成交（昨日计划，今日开盘价兑现）")
        lines.append("| 操作 | 股票 | 股数 | 成交价 | 名义金额 | 滑点 | 佣金 | 印花税 | 总磨损 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for t in executed:
            sign = "买入" if t["action"] in ("BUY_OPEN", "BUY") else "卖出"
            notional = t.get("shares", 0) * t.get("price", 0)
            s, c, x, fric = _trade_friction(t)
            cell = lambda v: f"{v:.2f}" if v == v else "—"  # NaN check
            lines.append(f"| {sign} | {_fmt(t['code'])} | {t['shares']:,} | {t['price']:.3f} "
                         f"| {notional:,.0f} | {cell(s)} | {cell(c)} | {cell(x)} | **{fric:.2f}** |")
        # planned reasons (if any)
        reasons = [(t.get('code', ''), t.get('_planned_reason', '')) for t in executed if t.get('_planned_reason')]
        if reasons:
            lines.append("")
            for code, r in reasons:
                lines.append(f"  - {_fmt(code)}: *{r}*")
        lines.append("")

    # Current positions
    state_positions = state["positions"]
    if state_positions:
        lines.append(f"## 当前持仓（{len(state_positions)}/{TOP_K}）")
        lines.append("")
        lines.append("| 股票 | 持仓 | 成本 | 现价 | 浮盈 | 仓位% |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        score_map = dict(zip(scored["code"], scored["ml_score"])) if not scored.empty else {}
        excess_map = dict(zip(scored["code"], scored["raw_excess"])) if not scored.empty else {}
        for p in state_positions:
            cost = p["avg_cost"]
            cur = p["current_price"]
            pnl = (cur / cost - 1) * 100 if cost > 0 else 0.0
            mv = p["shares"] * cur
            weight = mv / nav_today * 100 if nav_today > 0 else 0
            lines.append(f"| {_fmt(p['code'])} | {p['shares']:,} | {cost:.3f} | {cur:.3f} "
                         f"| {pnl:+.2f}% | {weight:.1f}% |")
        lines.append("")

    # Pending tomorrow trades
    if pending:
        lines.append("## 明日开盘待执行")
        for t in pending:
            sign = "买入" if t["action"] == "buy" else "卖出"
            extra = f"（目标 {t.get('target_value', 0):,.0f}）" if t["action"] == "buy" else ""
            lines.append(f"- {sign} **{_fmt(t['code'])}** {extra}  *{t.get('reason','')}*")
        lines.append("")
    else:
        lines.append("## 明日开盘待执行")
        lines.append("- 无（持仓与新选 Top-K 一致或 cost-aware 不换）")
        lines.append("")

    # Today's selected Top-K
    if selected:
        lines.append(f"## 今日选 Top-{TOP_K}（决策依据）")
        lines.append("| # | 股票 | ml_score | 模型超额 |")
        lines.append("|---:|---|---:|---:|")
        excess_map = dict(zip(scored["code"], scored["raw_excess"])) if not scored.empty else {}
        for i, (code, score) in enumerate(selected, 1):
            ex = excess_map.get(code, float("nan"))
            lines.append(f"| {i} | {_fmt(code)} | {score:.4f} | "
                         f"{ex*100 if pd.notna(ex) else float('nan'):+.2f}% |")
        lines.append("")

    lines.append("---")
    lines.append("*Money Printer 模拟交易自动生成*")
    return "\n".join(lines)


def save_report(report_md: str, today: pd.Timestamp) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    p = REPORTS_DIR / f"{today.strftime('%Y%m%d')}.md"
    p.write_text(report_md)
    return p


# ──────────────────────────────────────────────────────────────────────
# Feishu (reuse daily_report's sender)
# ──────────────────────────────────────────────────────────────────────

def send_to_feishu(report_md: str) -> bool:
    """Send markdown report to Feishu via daily_report's existing sender."""
    try:
        from scripts.daily_report import send_to_feishu as _send
        return _send(report_md)
    except Exception as e:
        logger.warning("Feishu send not available: {}", e)
        return False


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def get_zz500_close(today: pd.Timestamp) -> Optional[tuple[float, float]]:
    """Return (today_close, yesterday_close) for ZZ500 if available."""
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol="sh000905")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        today_row = df[df["date"] == today]
        if today_row.empty:
            return None
        idx = today_row.index[0]
        if idx == 0:
            return None
        return float(df.iloc[idx]["close"]), float(df.iloc[idx - 1]["close"])
    except Exception:
        return None


def is_trading_day(today: pd.Timestamp) -> bool:
    """Check if today is an A-share trading day using akshare's official
    trading calendar (`tool_trade_date_hist_sina`).

    Old approach (probing ZZ500 EOD bar) was too strict: it required EOD
    data to already be published, which doesn't happen until ~30-60 min
    after market close.  A 16:00 cron firing 1 min after close would see
    no data yet and incorrectly mark the day as non-trading (this exact
    bug skipped paper_trade on 2026-04-30).

    New approach: use the canonical trading calendar.  Falls back to weekday
    check + conservative skip if calendar API is unavailable.
    """
    if today.weekday() >= 5:   # Sat=5 Sun=6
        return False
    try:
        import akshare as ak
        cal = ak.tool_trade_date_hist_sina()
        cal["trade_date"] = pd.to_datetime(cal["trade_date"])
        return today.normalize() in set(cal["trade_date"])
    except Exception as e:
        logger.warning("Trade calendar fetch failed ({}); falling back to "
                       "ZZ500 data probe", e)
        bench = get_zz500_close(today)
        return bench is not None


def run() -> None:
    t0 = time.time()
    today = pd.Timestamp(date.today())
    logger.info("=" * 60)
    logger.info("Paper trade run · {}", today.strftime("%Y-%m-%d"))
    logger.info("=" * 60)

    if not is_trading_day(today):
        logger.info("Not a trading day ({}), skipping run.",
                    today.strftime("%A"))
        return

    state = load_state(today.strftime("%Y-%m-%d"))
    broker = state_to_broker(state)
    bar_cache: dict = {}

    # Step 1: Execute yesterday's pending at today's open
    executed = execute_pending(state, broker, today, bar_cache)
    logger.info("Executed {} trades from yesterday's plan", len(executed))

    # Step 2: Mark to market at today's close
    mark_to_market(broker, today, bar_cache)

    # Step 3: Score universe (load model)
    ranker = BlendRanker()
    if not ranker.load(MODEL_PATH_PREFIX):
        logger.error("Failed to load BlendRanker from {}_*.lgb — aborting",
                     MODEL_PATH_PREFIX)
        sys.exit(1)
    scored, dq = score_universe_with_blend(ranker, today, set(broker.positions.keys()))
    if scored.empty:
        logger.error("Universe scoring returned empty — aborting (no state changes)")
        sys.exit(1)
    logger.info("Scored {} stocks, data_quality={:.1%}", len(scored), dq)

    # Step 4: Plan tomorrow's trades
    pending, selected = plan_tomorrow_trades(broker, scored, today, dq)
    state["pending_trades"] = pending
    logger.info("Planned {} trades for tomorrow's open", len(pending))

    # Step 5: Persist broker state back
    broker_to_state(state, broker)
    nav_entry = append_nav(state, today, broker)
    logger.info("NAV today: {:,.2f}", nav_entry["nav"])

    # Step 6: Save state
    save_state(state)

    # Step 7: Generate report
    bench = get_zz500_close(today)
    bench_today, bench_prev = (bench or (None, None))
    report_md = format_report(state, scored, selected, executed, pending, today,
                              bench_today, bench_prev)
    p = save_report(report_md, today)
    logger.info("Report saved to {}", p)

    # Step 8: Feishu
    sent = send_to_feishu(report_md)
    logger.info("Feishu: {}", "sent" if sent else "skipped")

    elapsed = time.time() - t0
    logger.info("Done in {:.1f}s", elapsed)


if __name__ == "__main__":
    run()
