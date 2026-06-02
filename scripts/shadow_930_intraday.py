"""Arm B shadow recorder — 9:30 (T+1 open) entry with the intraday_blend model.

P11-5 live A/B (advisor round 101). Two arms, SAME intraday_blend model,
differ ONLY in entry timing:

  - Arm A (real): 14:30 same-day intraday path, real QMT money
    (scripts/intraday_plan.py + ecs_intraday_execute.ps1). NAV tracked by
    account_report.py → data/account_nav_history.json.
  - Arm B (shadow, THIS FILE): 9:30 next-day open entry, pure simulation,
    no real trades. NAV tracked here → data/shadow_930/state.json.

P11-3 backtest already isolated this variable (model held constant,
entry T+1-open) and found ZERO Sharpe difference vs the 14:30 path
(both 1.95). This shadow live-validates that null result over ~2 weeks
from a common 5/29 start.

Why a shadow (not just trust the backtest): backtest fills are idealized
(open price, modeled slippage). Real fills drift. Running both arms from
the identical 5/29 portfolio snapshot, same universe, same conviction
sizing, same model — the ONLY uncontrolled variable is entry timing —
gives a live read on whether 14:30 realtime infra is worth keeping.

DESIGN (adapts scripts/paper_trade.py, which is idle since launchd moved
to account_report.py):

| aspect              | paper_trade (old)       | shadow_930_intraday (this)              |
|---------------------|-------------------------|-----------------------------------------|
| model               | blend (64)              | intraday_blend (68, hybrid .lgb)        |
| universe            | zz500 (+chinext)        | hs300+zz500 minus 创业板/科创板          |
| sizing              | conviction-target       | conviction-target (same)                |
| entry               | T+1 open                | T+1 open (same — this IS Arm B's defn)  |
| morning features    | n/a                     | EOD-proxy (same formula as P11-3)       |
| starting cap/holds  | 300k full               | real account 5/29 snapshot (same start) |

Daily flow (run 17:00 after daily_report.py, same trading-day gate):
  1. Load shadow state. Day-1: init from real QMT snapshot (common start).
  2. Execute yesterday's pending at TODAY's open (T+1 open fill model).
  3. Mark-to-market at TODAY's close → shadow NAV.
  4. Score universe with intraday_blend (EOD-proxy morning extras).
  5. Plan tomorrow's conviction-target Top-10 (cost-aware swap).
  6. Persist state + nav_history.

NOT user-facing: no Feishu. account_report.py reads this state to render
the "14:30 real vs 9:30-shadow" comparison section.

Rule #4: uses hybrid data/intraday_blend_*.lgb. Does NOT touch production
data/blend_*.lgb.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
os.chdir(_PROJECT_ROOT)

from mp.data.fetcher import get_daily_bars
from mp.ml.dataset import build_latest_features
from mp.ml.model import BlendRanker
from mp.ml.intraday_features import INTRADAY_FEATURE_COLS, INTRADAY_EXTRA_COLUMNS

# Model-agnostic helpers reused verbatim from paper_trade (same friction
# model — SLIPPAGE_BPS=5, COMMISSION_BPS=3 — matching walk_forward).
from scripts.paper_trade import (
    state_to_broker,
    broker_to_state,
    execute_pending,
    mark_to_market,
    append_nav,
    plan_tomorrow_trades,
    get_zz500_close,
    is_trading_day,
    _resolve_names,
    TOP_K,
)

MODEL_PATH_PREFIX = "data/intraday_blend"
STATE_PATH = Path("data/shadow_930/state.json")
REPORTS_DIR = Path("data/shadow_930/reports")
DATA_QUALITY_GATE = 0.5

# ECS SSH coordinates (only used once, for day-1 real-snapshot init).
ECS_USER = "Administrator"
ECS_HOST = "14.103.49.51"


# ──────────────────────────────────────────────────────────────────────
# State IO (shadow path — do NOT reuse paper_trade's, wrong path)
# ──────────────────────────────────────────────────────────────────────

def _init_state_from_qmt(today: str) -> dict:
    """Day-1: seed the shadow portfolio from the real QMT account so both
    arms start from the identical 5/29 holdings + cash.

    Falls back to a flat-cash 300k start if the SSH snapshot is
    unreachable (logged loudly — a degraded but still-runnable start;
    advisor can re-seed by deleting state.json and re-running on a day
    ECS is reachable).
    """
    try:
        from scripts.account_report import fetch_qmt_snapshot
        snap = fetch_qmt_snapshot()
        acc = snap["account"]
        positions = []
        for p in snap["positions"]:
            px = float(p.get("market_price") or p.get("avg_cost") or 0.0)
            positions.append({
                "code": str(p["code"]).zfill(6),
                "shares": int(p["shares"]),
                "avg_cost": float(p["avg_cost"]),
                "current_price": px,
                "peak_price": px,
                "entry_date": today,
            })
        state = {
            "version": 1,
            "started": today,
            "init_source": "qmt_snapshot",
            "initial_capital": float(acc["total_assets"]),
            "cash": float(acc["cash_available"]),
            "positions": positions,
            "pending_trades": [],
            "nav_history": [],
            "trade_log": [],
        }
        logger.info("Shadow seeded from QMT snapshot: total={:,.0f} cash={:,.0f} positions={}",
                    acc["total_assets"], acc["cash_available"], len(positions))
        return state
    except Exception as e:
        logger.error("QMT snapshot init failed ({}) — falling back to flat 300k start. "
                     "Delete state.json and re-run on a day ECS is reachable to re-seed.", e)
        return {
            "version": 1,
            "started": today,
            "init_source": "fallback_flat_300k",
            "initial_capital": 300_000.0,
            "cash": 300_000.0,
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
            logger.error("Failed to read shadow state: {} — refusing to overwrite", e)
            raise
    logger.info("No prior shadow state. Day-1 init from real QMT snapshot on {}", today)
    return _init_state_from_qmt(today)


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    logger.info("Shadow state saved: {} positions, cash={:,.0f}",
                len(state["positions"]), state["cash"])


# ──────────────────────────────────────────────────────────────────────
# Universe + intraday feature panel
# ──────────────────────────────────────────────────────────────────────

def filtered_universe() -> list[str]:
    """hs300+zz500 minus 创业板/科创板 — aligns with the real account's
    tradeable universe (scripts/daily_report.py + scripts/intraday_plan.py)."""
    from mp.data.fetcher import get_recommendation_universe
    codes = [str(c).zfill(6) for c in get_recommendation_universe()]
    excluded = ("300", "301", "302", "688", "689")
    return [c for c in codes if not c.startswith(excluded)]


def attach_eod_proxy_extras(features: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Append the 4 INTRADAY_EXTRA_COLUMNS to the latest-row feature panel
    using the EOD-proxy formula (same as P11-3 training / walk_forward).

    For each code we re-fetch ~60 calendar days of EOD bars, run
    ``compute_extras_for_panel`` (which handles the 0.85 / 0.75 fudge
    factors + overnight_gap), and take the LAST row (most recent date).
    Merge on code (build_latest_features already gives one latest row per
    code, so a code-level merge is unambiguous and dodges date-dtype
    mismatch between the two fetches).
    """
    from scripts.train_intraday import compute_extras_for_panel

    features = features.copy()
    features["code"] = features["code"].astype(str).str.zfill(6)
    codes = features["code"].tolist()

    start = (today - pd.Timedelta(days=60)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    rows: list[dict] = []
    for code in codes:
        try:
            bars = get_daily_bars(code, start, end)
        except Exception:
            continue
        if bars is None or bars.empty:
            continue
        bars = bars.sort_values("date").reset_index(drop=True)
        bars["date"] = pd.to_datetime(bars["date"])
        ex = compute_extras_for_panel(bars)
        if ex.empty:
            continue
        last = ex.iloc[-1]
        rows.append({"code": code,
                     **{col: float(last[col]) for col in INTRADAY_EXTRA_COLUMNS}})

    if not rows:
        logger.warning("attach_eod_proxy_extras: no extras computed — all NaN")
        for col in INTRADAY_EXTRA_COLUMNS:
            features[col] = np.nan
        return features

    extras_df = pd.DataFrame(rows)
    merged = features.merge(extras_df, on="code", how="left")
    n_ok = int(merged[INTRADAY_EXTRA_COLUMNS[0]].notna().sum())
    logger.info("attach_eod_proxy_extras: {}/{} codes got morning extras", n_ok, len(merged))
    return merged


def score_universe_intraday(ranker: BlendRanker, today: pd.Timestamp,
                            held_codes: set[str]) -> tuple[pd.DataFrame, float]:
    """Build the 68-col intraday panel and score with intraday_blend.

    Returns (scored_df[code, ml_score, raw_excess, rank_pct], data_quality)
    — same schema paper_trade.plan_tomorrow_trades consumes.
    """
    universe = set(filtered_universe())
    universe.update(str(c).zfill(6) for c in held_codes)
    codes = sorted(universe)
    logger.info("Building EOD features for {} stocks...", len(codes))
    features = build_latest_features(codes, include_fundamentals=True)
    if features.empty:
        return pd.DataFrame(), 0.0

    dq = float(features.attrs.get("_data_quality", 1.0))
    features = attach_eod_proxy_extras(features, today)

    # Sanity: the model needs all 68 columns. Missing morning extras are
    # left as NaN (LightGBM handles NaN natively — same as training where
    # warm-up rows had NaN extras).
    missing = [c for c in INTRADAY_FEATURE_COLS if c not in features.columns]
    if missing:
        logger.error("Panel missing {} model columns: {} — aborting score", len(missing), missing[:5])
        return pd.DataFrame(), dq

    scores = ranker.predict(features)
    raw = ranker.predict_raw(features)
    df = pd.DataFrame({
        "code": features["code"].astype(str).str.zfill(6).values,
        "ml_score": scores,
        "raw_excess": raw,
    }).sort_values("ml_score", ascending=False).reset_index(drop=True)
    df["rank_pct"] = df["ml_score"].rank(pct=True)
    return df, dq


# ──────────────────────────────────────────────────────────────────────
# Lean debug report (NOT user-facing — account_report renders the compare)
# ──────────────────────────────────────────────────────────────────────

def save_debug_report(state: dict, selected: list, executed: list,
                       pending: list, today: pd.Timestamp) -> Path:
    nav_hist = state["nav_history"]
    nav_today = nav_hist[-1]["nav"] if nav_hist else state["initial_capital"]
    nav_prev = nav_hist[-2]["nav"] if len(nav_hist) >= 2 else state["initial_capital"]
    daily_ret = nav_today / nav_prev - 1 if nav_prev > 0 else 0.0
    total_ret = nav_today / state["initial_capital"] - 1 if state["initial_capital"] > 0 else 0.0

    name_codes = {p["code"] for p in state.get("positions", [])}
    name_codes.update(t.get("code", "") for t in pending)
    name_map = _resolve_names([c for c in name_codes if c])

    def _fmt(code: str) -> str:
        nm = name_map.get(code, code)
        return f"{nm}({code})" if nm and nm != code else code

    lines = [
        f"# Shadow (9:30+intraday) {today.strftime('%Y-%m-%d')}",
        "",
        f"**起始**: {state['started']} ({state.get('init_source', '?')}) · "
        f"**起始资金**: {state['initial_capital']:,.0f}",
        "",
        "## Shadow NAV",
        f"- 总资产: **{nav_today:,.2f}** (现金 {state['cash']:,.2f} + "
        f"持仓 {nav_hist[-1]['positions_value'] if nav_hist else 0:,.2f})",
        f"- 今日: {nav_today - nav_prev:+,.2f} ({daily_ret:+.2%})",
        f"- 累计: {nav_today - state['initial_capital']:+,.2f} ({total_ret:+.2%})",
        "",
        f"## 当前持仓 ({len(state['positions'])}/{TOP_K})",
    ]
    if state["positions"]:
        lines.append("| 股票 | 股数 | 成本 | 现价 | 浮盈% |")
        lines.append("|---|---:|---:|---:|---:|")
        for p in state["positions"]:
            cost, cur = p["avg_cost"], p["current_price"]
            pnl = (cur / cost - 1) * 100 if cost > 0 else 0.0
            lines.append(f"| {_fmt(p['code'])} | {p['shares']:,} | {cost:.3f} | {cur:.3f} | {pnl:+.2f}% |")
    lines.append("")
    if pending:
        lines.append("## 明日开盘待执行 (shadow)")
        for t in pending:
            sign = "买入" if t["action"] == "buy" else "卖出"
            lines.append(f"- {sign} **{_fmt(t['code'])}**  *{t.get('reason', '')}*")
    lines.append("")
    lines.append("---")
    lines.append("_Money Printer Arm B shadow recorder (shadow_930_intraday.py)_")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    p = REPORTS_DIR / f"{today.strftime('%Y%m%d')}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def run() -> None:
    t0 = time.time()
    today = pd.Timestamp(date.today())
    logger.info("=" * 60)
    logger.info("Shadow 9:30+intraday run · {}", today.strftime("%Y-%m-%d"))
    logger.info("=" * 60)

    if not is_trading_day(today):
        logger.info("Not a trading day ({}), skipping shadow run.", today.strftime("%A"))
        return

    state = load_state(today.strftime("%Y-%m-%d"))
    broker = state_to_broker(state)
    bar_cache: dict = {}

    # Step 1: fill yesterday's pending at today's open (T+1 open model)
    executed = execute_pending(state, broker, today, bar_cache)
    logger.info("Shadow executed {} trades from yesterday's plan", len(executed))

    # Step 2: load intraday_blend (Rule #4 — hybrid artifact, NOT production blend)
    ranker = BlendRanker(feature_cols=list(INTRADAY_FEATURE_COLS))
    if not ranker.load(MODEL_PATH_PREFIX):
        logger.error("Failed to load intraday_blend from {}_*.lgb — aborting", MODEL_PATH_PREFIX)
        sys.exit(1)
    logger.info("Loaded intraday_blend ({} features)", len(INTRADAY_FEATURE_COLS))

    scored, dq = score_universe_intraday(ranker, today, set(broker.positions.keys()))

    # Step 3: mark-to-market at today's close (after step 2 populated DB bars)
    mark_to_market(broker, today, bar_cache)
    if scored.empty:
        logger.error("Shadow scoring empty — aborting (no state changes)")
        sys.exit(1)
    logger.info("Shadow scored {} stocks, data_quality={:.1%}", len(scored), dq)

    # Step 4: plan tomorrow's conviction-target Top-K (cost-aware swap)
    pending, selected = plan_tomorrow_trades(broker, scored, today, dq)
    state["pending_trades"] = pending
    logger.info("Shadow planned {} trades for tomorrow's open", len(pending))

    # Step 5: persist broker state + NAV
    broker_to_state(state, broker)
    nav_entry = append_nav(state, today, broker)
    logger.info("Shadow NAV today: {:,.2f}", nav_entry["nav"])

    # Step 6: save
    save_state(state)
    rpt = save_debug_report(state, selected, executed, pending, today)
    logger.info("Shadow debug report → {}", rpt)

    logger.info("Shadow done in {:.1f}s", time.time() - t0)


if __name__ == "__main__":
    run()
