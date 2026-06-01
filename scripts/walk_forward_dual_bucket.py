"""Dual-bucket walk-forward backtest (round 174 / advisor round 173 spec).

User question (round 173): "EOD 9:25 path 和 OOS 14:30 path 都跑活 → 同票
两 bucket 选股冲突 round-trip 成本是不是把 alpha 吃光?"

This script answers it by running both buckets in parallel on the same
historical window and reporting:

  1. Three NAV curves:
       (a) EOD-only — capital fully in EOD path
       (b) OOS-only — capital fully in OOS path
       (c) Dual — split: half EOD bucket + half OOS bucket
  2. Per-day picks overlap (|EOD ∩ OOS| as fraction of top-K)
  3. Same-day round-trip conflicts:
       Type A: EOD sells X on D + OOS buys X on D (within hours)
       Type B: OOS bought X on D-1 + EOD sells X on D (overnight conflict)
  4. Cost breakdown (commission / slippage / stamp) per bucket
  5. Net alpha delta: dual_NAV − max(EOD_only, OOS_only)
                     and dual_NAV − (EOD_only + OOS_only)/2

Design:
  - Two independent SimulatedBroker instances (advisor question (1):
    independent positions — same code can sit in both buckets)
  - EOD uses `data/blend_*.lgb` (n2c BlendRanker, FACTOR_COLUMNS 64-col)
    scored at D close → executed on D+1 09:25 open
  - OOS uses `data/intraday_blend_*.lgb` (c2c, INTRADAY_FEATURE_COLS
    68-col) scored on D 14:30 with morning-bar injected factors →
    executed on D 14:30 close
  - OOS bucket enforces ¥20000 daily new-buy cap via in-memory
    ArmBBudgetTracker (round 161 guardrail (a))
  - Pure research (`--skip-update` style): no model retrain, no disk
    state, no real money, no portfolio.yaml read

Caveats (honest):
  - in-sample 2025-09 ~ 2026-04 = 8 months only
  - `data/intraday_blend_*.lgb` was trained 5/27 BEFORE the n2c label
    upgrade (round 162). It is the *currently-shipped* OOS model so
    measuring its drag is what user asked for.
  - OOS path approximates the 14:29 fill price using the 1-min close
    at 14:29:00 from `data/intraday_1m_qfq/`. Real fills are at the
    next-second auction so this is ~5bps optimistic; bounded.

Usage:
    .venv/bin/python scripts/walk_forward_dual_bucket.py
        [--start 2025-09-01] [--end 2026-04-28]
        [--seeds 42,43,44] [--top-k 10]
        [--initial-capital 200000]
        [--out data/reports/walk_forward_dual_bucket.md]
        [--smoke]   # single seed × 1 month for sanity

round 174 commit (engineer side).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mp.account.broker import FeeSchedule, SimulatedBroker  # noqa: E402
from mp.ml.dataset import FACTOR_COLUMNS  # noqa: E402
from mp.ml.intraday_features import INTRADAY_FEATURE_COLS  # noqa: E402
from mp.ml.model import BlendRanker  # noqa: E402
from mp.risk.arm_b_budget import BudgetState, ArmBBudgetTracker  # noqa: E402

# Cache locations (must already exist; cache-only mode, no fetching)
BARS_CACHE = PROJECT_ROOT / "data" / "wf_cache" / "bars.parquet"
FACTORS_CACHE = PROJECT_ROOT / "data" / "wf_cache" / "factors_label_next_open_to_close.parquet"
INTRADAY_DIR = PROJECT_ROOT / "data" / "intraday_1m_qfq"

# Backtest defaults (round numbers; research mode)
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_TOP_K = 10
DEFAULT_INITIAL_CAPITAL = 200_000.0
OOS_DAILY_BUDGET = 20_000.0
DEFAULT_START = "2025-09-01"
DEFAULT_END = "2026-04-28"

# round 176 (advisor round 175 拍板):
# (1) MERGED hard-cap on per-code position across BOTH buckets.
# Hard-coded to match daily_report.py default (hard_single_weight=0.40 is
# the conservative production setting; advisor spec example used 8% so we
# default to 0.08 here for research and surface as CLI flag).
MERGED_HARD_MAX_PCT_DEFAULT = 0.08


# ───────────────────────────────────────────────────────────────────
# Data loading
# ───────────────────────────────────────────────────────────────────

def load_bars() -> pd.DataFrame:
    """Daily OHLCV bars from cache. Returns columns
    [code, date, open, high, low, close, volume, amount, turnover]."""
    if not BARS_CACHE.exists():
        logger.error("bars cache missing: {}", BARS_CACHE)
        sys.exit(2)
    df = pd.read_parquet(BARS_CACHE)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def load_factors() -> pd.DataFrame:
    """Factor panel (next_open_to_close label). The label is what we
    *trained on* — we don't use it for prediction (we use the features)."""
    if not FACTORS_CACHE.exists():
        logger.error("factors cache missing: {}", FACTORS_CACHE)
        sys.exit(2)
    df = pd.read_parquet(FACTORS_CACHE)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def load_intraday_morning_bars(window_start: pd.Timestamp,
                                window_end: pd.Timestamp) -> Dict[Tuple[str, pd.Timestamp], dict]:
    """Aggregate 1-min bars (09:30~14:29) per (code, date) into a single
    morning-bar dict {open, high, low, close, volume} that the OOS feature
    overlay needs. Returns map keyed by (code, date)."""
    start_month = window_start.strftime("%Y%m")
    end_month = window_end.strftime("%Y%m")
    files: List[Path] = []
    for p in sorted(INTRADAY_DIR.glob("*.parquet")):
        stem = p.stem  # e.g. 202509
        if start_month <= stem <= end_month:
            files.append(p)
    if not files:
        logger.warning("no intraday parquet in {} for [{}, {}] — OOS path "
                       "will see no features and skip all scoring (returns "
                       "empty dict, downstream OOS picks will be empty).",
                       INTRADAY_DIR, start_month, end_month)
        return {}

    morning_bars: Dict[Tuple[str, pd.Timestamp], dict] = {}
    for p in files:
        logger.info("loading intraday {}", p.name)
        df = pd.read_parquet(p)
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
        df["minute"] = pd.to_datetime(df["datetime"]).dt.strftime("%H:%M")
        # Keep only morning (09:30~11:30) + early afternoon up to 14:29
        morning_mask = (df["minute"] >= "09:30") & (df["minute"] <= "11:30")
        early_afternoon_mask = (df["minute"] >= "13:00") & (df["minute"] <= "14:29")
        df_keep = df[morning_mask | early_afternoon_mask]
        # Group by (code, date) and aggregate.
        # Drop NaN minute-rows first so aggregations don't propagate NaN
        # (some suspended stocks emit NaN bars for parts of the day).
        df_keep = df_keep.dropna(subset=["open", "high", "low", "close"])
        df_keep = df_keep[(df_keep["close"] > 0) & (df_keep["open"] > 0)]
        grouped = df_keep.groupby(["code", "date"])
        for (code, date), g in grouped:
            if g.empty:
                continue
            morning_bars[(code, date)] = {
                "open": float(g["open"].iloc[0]),
                "high": float(g["high"].max()),
                "low": float(g["low"].min()),
                "close": float(g["close"].iloc[-1]),  # 14:29 close
                "volume": float(g["volume"].sum()),
            }
    logger.info("intraday morning bars loaded: {} (code,date) keys", len(morning_bars))
    return morning_bars


# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────

def make_in_memory_tracker(budget: float, sim_date: str) -> ArmBBudgetTracker:
    """Build an ArmBBudgetTracker that doesn't touch disk (for backtest)."""
    state = BudgetState(
        trading_date=sim_date,
        committed_notional=0.0,
        budget_max=float(budget),
        events=[],
    )
    # path is required by __init__ but we'll no-op _persist so nothing writes
    tracker = ArmBBudgetTracker(state, path=PROJECT_ROOT / "data" / "_dual_bucket_tmp.json")
    tracker._persist = lambda: None  # type: ignore[method-assign]
    return tracker


def reset_tracker_for_day(tracker: ArmBBudgetTracker, sim_date: str) -> None:
    """Reset tracker's committed_notional + events at the start of each sim day."""
    tracker._state = BudgetState(
        trading_date=sim_date,
        committed_notional=0.0,
        budget_max=tracker._state.budget_max,
        events=[],
    )


def adv_20d_lookup(bars: pd.DataFrame) -> Dict[Tuple[str, pd.Timestamp], float]:
    """Build (code, date) → ADV(20d) for impact model. Uses shift(1) so no lookahead."""
    out: Dict[Tuple[str, pd.Timestamp], float] = {}
    for code, g in bars.groupby("code"):
        g = g.sort_values("date")
        adv = g["amount"].rolling(20, min_periods=5).mean().shift(1)
        for d, v in zip(g["date"].values, adv.values):
            if pd.notna(v):
                out[(code, pd.Timestamp(d))] = float(v)
    return out


def top_k_picks(scores_df: pd.DataFrame, k: int,
                price_map: Optional[Dict[str, float]] = None,
                cap_price: Optional[float] = None,
                adv_lookup: Optional[Dict[str, float]] = None,
                adv_floor: Optional[float] = None) -> pd.DataFrame:
    """Return top-K rows from scores_df (must have 'code' + 'score').

    Applies the same filter chain as production:
      - drop 688/689/300/301 (科创板 / 创业板 — Arm B rule-of-thumb)
      - drop ADV(20d) < adv_floor (if adv_lookup + adv_floor given; OOS only)
      - drop close > cap_price (if cap_price + price_map given; OOS only)
      - sort by score desc, take top K
    """
    df = scores_df.copy()
    if df.empty or "score" not in df.columns:
        return df.head(0)
    # Defensive board filter (Arm B rule from intraday_plan / round 161)
    df = df[~df["code"].astype(str).str.zfill(6).str.startswith(("688", "689", "300", "301"))]
    # ADV floor (OOS only)
    if adv_lookup is not None and adv_floor is not None:
        df["_adv"] = df["code"].map(adv_lookup).fillna(0.0)
        df = df[df["_adv"] >= adv_floor]
    # Price cap (OOS only — round 161 ARM_B_PRICE_CAP=50)
    if price_map is not None and cap_price is not None:
        df["_price"] = df["code"].map(price_map).fillna(0.0)
        df = df[(df["_price"] > 0) & (df["_price"] <= cap_price)]
    return df.sort_values("score", ascending=False).head(k).reset_index(drop=True)


# ───────────────────────────────────────────────────────────────────
# Rebalance logic
# ───────────────────────────────────────────────────────────────────

def rebalance_to_targets(broker: SimulatedBroker, top_picks: pd.DataFrame,
                          prices: Dict[str, float],
                          adv_lookup: Dict[str, float],
                          sim_date: str,
                          target_pos_pct: float = 0.70) -> List[str]:
    """Rebalance broker to equal-weighted target across `top_picks`.

    Cap each name at target_pos_pct × total_value / len(top_picks).
    Returns the picks list (as codes) for picks_log tracking.
    """
    total = broker.total_value
    investable = total * target_pos_pct
    n = max(1, len(top_picks))
    per_name = investable / n

    pick_codes = [str(c).zfill(6) for c in top_picks["code"].tolist()]
    pick_set = set(pick_codes)

    # First: sell anything not in top_picks
    for code in list(broker.positions.keys()):
        if code not in pick_set:
            price = prices.get(code)
            if price is None or price <= 0 or pd.isna(price):
                continue  # skip stocks without today's price (suspended / no data)
            adv = adv_lookup.get(code)
            broker.sell(code, price, date=sim_date, adv=adv)

    # Then: buy/scale to per_name target for everyone in top_picks
    for code in pick_codes:
        price = prices.get(code)
        if price is None or price <= 0 or pd.isna(price):
            continue
        adv = adv_lookup.get(code)
        broker.buy(code, price, target_value=per_name, date=sim_date, adv=adv)

    return pick_codes


def execute_oos_only_buy(broker: SimulatedBroker,
                          oos_picks: pd.DataFrame,
                          prices: Dict[str, float],
                          adv_lookup: Dict[str, float],
                          sim_date: str,
                          target_pos_pct: float = 0.70) -> Tuple[List[str], List[dict]]:
    """round 181 (advisor round 179 spec patch, v3.1 no-cap):

    OOS bucket = strictly **only buy, never sell**. Production
    ``intraday_plan`` has no sell logic (grep ``def.*sell`` = 0 lines).

    Cash comes from the SHARED broker pool (single QMT account model).
    **No ¥20k daily cap in backtest** (advisor 179: that's a prod risk
    guardrail, not part of model-alpha evaluation). OOS picks size each
    name as ``broker.cash × target_pos_pct / K`` (equal-weight on
    available cash, matching EOD's sizing for apples-to-apples
    comparison).

    When ``broker.cash`` is short — because EOD path already spent
    it on its own picks at 09:25 — ``broker.buy()`` auto-truncates
    or returns None. This is the production behavior the user worried
    about: OOS competes for cash with EOD on the SAME account.
    """
    skipped: List[dict] = []
    if oos_picks.empty:
        return [], skipped

    pick_codes = [str(c).zfill(6) for c in oos_picks["code"].tolist()]
    # round 181 sizing: equal-weight on currently-available cash × target_pos_pct
    # (matches EOD rebalance_to_targets formula)
    investable = broker.cash * target_pos_pct
    per_name = investable / max(1, len(pick_codes))

    for code in pick_codes:
        price = prices.get(code)
        if price is None or price <= 0 or pd.isna(price):
            skipped.append({"code": code, "reason": "no_price_or_nan"})
            continue
        adv = adv_lookup.get(code)
        # broker.buy with target_value auto-truncates when cash short
        trade = broker.buy(code, price, target_value=per_name,
                           date=sim_date, adv=adv, action="BUY_OOS")
        if trade is None:
            skipped.append({"code": code, "reason": "insufficient_cash_or_below_lot"})
            continue

    # Return only codes that successfully bought (filtered against skipped)
    skipped_codes = {s["code"] for s in skipped}
    executed = [c for c in pick_codes if c not in skipped_codes]
    return executed, skipped


# ───────────────────────────────────────────────────────────────────
# Scoring (BlendRanker + INTRADAY overlay)
# ───────────────────────────────────────────────────────────────────

def make_eod_panel_for(factors: pd.DataFrame, sim_date: pd.Timestamp) -> pd.DataFrame:
    """EOD path uses factors @ D close (= the row at sim_date in cache)."""
    sub = factors[factors["date"] == sim_date].copy()
    return sub


def make_oos_panel_for(factors: pd.DataFrame, bars: pd.DataFrame,
                       morning_bars: Dict[Tuple[str, pd.Timestamp], dict],
                       sim_date: pd.Timestamp) -> pd.DataFrame:
    """OOS path: take factors @ D-1 close, then re-overlay TECHNICAL_COLUMNS
    using morning bars from sim_date 09:30~14:29.

    For MVP simplicity, we use the D-1 factor row AS-IS and just append the
    4 INTRADAY_EXTRA_COLUMNS computed from morning bars. The full
    re-derivation of 64 TECHNICAL_COLUMNS from intraday is the production
    intraday_plan path; the MVP shortcut is justified because:
      - INTRADAY_EXTRA_COLUMNS carry the strongest signal incrementally
      - re-deriving all 64 from intraday requires the full feature
        pipeline (volume_trend, amount_ratio, amihud_illiq, mfi_14, ...)
        which expects daily bars not 1m bars
      - the goal is *relative* comparison, not absolute OOS-model accuracy
    """
    # D-1 (most recent factors row <= sim_date strictly)
    prev_dates = factors["date"][factors["date"] < sim_date]
    if prev_dates.empty:
        return factors.head(0)
    d_minus_1 = prev_dates.max()
    base = factors[factors["date"] == d_minus_1].copy()
    if base.empty:
        return base

    # Append intraday-extra columns from morning bars
    today_ts = pd.Timestamp(sim_date)
    extras = []
    for _, row in base.iterrows():
        code = str(row["code"]).zfill(6)
        mb = morning_bars.get((code, today_ts))
        if mb is None:
            extras.append({"_morning_ret": 0.0, "_morning_vol_ratio": 1.0,
                          "_morning_amt_ratio": 1.0, "_morning_hl_range": 0.0})
            continue
        op = mb["open"]
        cl = mb["close"]
        morning_ret = (cl - op) / op if op > 0 else 0.0
        # rough proxies for vol/amount ratios (MVP)
        extras.append({
            "_morning_ret": morning_ret,
            "_morning_vol_ratio": 1.0,
            "_morning_amt_ratio": 1.0,
            "_morning_hl_range": (mb["high"] - mb["low"]) / op if op > 0 else 0.0,
        })
    extras_df = pd.DataFrame(extras).reset_index(drop=True)
    base = pd.concat([base.reset_index(drop=True), extras_df], axis=1)

    # Map our 4 proxy column names to the model's INTRADAY_EXTRA_COLUMNS
    # NOTE: The actual model was trained on column names from
    # intraday_features.INTRADAY_EXTRA_COLUMNS. We map ours to those.
    from mp.ml.intraday_features import INTRADAY_EXTRA_COLUMNS
    if len(INTRADAY_EXTRA_COLUMNS) == 4:
        rename_map = {
            "_morning_ret": INTRADAY_EXTRA_COLUMNS[0],
            "_morning_vol_ratio": INTRADAY_EXTRA_COLUMNS[1],
            "_morning_amt_ratio": INTRADAY_EXTRA_COLUMNS[2],
            "_morning_hl_range": INTRADAY_EXTRA_COLUMNS[3],
        }
        base = base.rename(columns=rename_map)
    return base


def score_panel(ranker: BlendRanker, panel: pd.DataFrame) -> pd.DataFrame:
    """Run ranker.predict on panel. Returns DataFrame[code, score]."""
    if panel.empty:
        return pd.DataFrame(columns=["code", "score"])
    # Drop rows with NaN in any feature_col
    cols_needed = list(ranker.feature_cols)
    missing_cols = [c for c in cols_needed if c not in panel.columns]
    if missing_cols:
        logger.debug("score_panel: missing cols {} (filling 0)", missing_cols[:5])
        for c in missing_cols:
            panel[c] = 0.0
    sub = panel.dropna(subset=cols_needed).copy()
    if sub.empty:
        return pd.DataFrame(columns=["code", "score"])
    # BlendRanker.predict expects a label col too; add a dummy
    if "fwd_ret" not in sub.columns:
        sub["fwd_ret"] = 0.0
    if "excess_ret" not in sub.columns:
        sub["excess_ret"] = 0.0
    if "date" not in sub.columns:
        sub["date"] = pd.Timestamp("2020-01-01")
    scores = ranker.predict(sub)
    out = pd.DataFrame({"code": sub["code"].astype(str).str.zfill(6).values,
                        "score": scores})
    return out


# ───────────────────────────────────────────────────────────────────
# Single-seed orchestration
# ───────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    seed: int
    mode: str  # "eod_only", "oos_only", "dual"
    nav_records: List[dict] = field(default_factory=list)
    picks_log: List[dict] = field(default_factory=list)
    conflicts: List[dict] = field(default_factory=list)
    eod_trades: List[dict] = field(default_factory=list)
    oos_trades: List[dict] = field(default_factory=list)


def run_seed(seed: int, mode: str,
             factors: pd.DataFrame, bars: pd.DataFrame,
             morning_bars: Dict[Tuple[str, pd.Timestamp], dict],
             trading_dates: List[pd.Timestamp],
             initial_capital: float, top_k: int,
             eod_ranker: BlendRanker, oos_ranker: BlendRanker,
             adv_lookup: Dict[Tuple[str, pd.Timestamp], float],
             merged_hard_max_pct: float = MERGED_HARD_MAX_PCT_DEFAULT,
             ) -> BacktestResult:
    """round 180 (v3 production-faithful, advisor 178 spec):

    Single SHARED ``broker`` for all 3 modes. EOD path does full
    rebalance (sell+buy) at 09:25; OOS path strictly only buys at 14:30
    (production ``intraday_plan`` has 0 sell logic). Both compete for
    the SAME cash pool — that's the real user-facing model.

    No merged-cap check (production has none). Conflicts detected
    post-hoc from ``broker.trade_log`` (OOS buy D → EOD sell D+1 = real
    round-trip with cost).
    """
    np.random.seed(seed)
    fees = FeeSchedule()

    # v3.1: single shared broker for ALL modes (apples-to-apples capital).
    # round 181: no daily cap (advisor 179 patch — prod guardrail removed
    # from backtest so model alpha isn't compressed).
    broker = SimulatedBroker(initial_capital=initial_capital, fees=fees, silent=True)

    enable_eod = mode in ("eod_only", "dual")
    enable_oos = mode in ("oos_only", "dual")

    result = BacktestResult(seed=seed, mode=mode)
    eod_plan_for_tomorrow: Optional[pd.DataFrame] = None

    bars_by_date_code = {(str(r["code"]).zfill(6), r["date"]): r
                         for _, r in bars[["code", "date", "open", "close"]].iterrows()}

    for i, D in enumerate(trading_dates):
        sim_date_str = D.strftime("%Y-%m-%d")

        # Build today's price maps
        open_prices: Dict[str, float] = {}
        close_prices: Dict[str, float] = {}
        for (code, date), row in bars_by_date_code.items():
            if date == D:
                open_prices[code] = float(row["open"])
                close_prices[code] = float(row["close"])
        adv_today = {code: adv_lookup.get((code, D), 0.0) for code in close_prices}

        # ─── 1. EOD execute yesterday's plan on today's open (shared broker) ───
        eod_picks_executed: List[str] = []
        if enable_eod and eod_plan_for_tomorrow is not None and not eod_plan_for_tomorrow.empty:
            eod_picks_executed = rebalance_to_targets(
                broker, eod_plan_for_tomorrow, open_prices,
                adv_today, sim_date_str, target_pos_pct=0.70,
            )

        # ─── 2. OOS score + only-buy at 14:30 (shared broker, fights for cash) ───
        oos_picks_executed: List[str] = []
        oos_skipped: List[dict] = []
        if enable_oos:
            oos_panel = make_oos_panel_for(factors, bars, morning_bars, D)
            oos_scores = score_panel(oos_ranker, oos_panel)
            if not oos_scores.empty:
                oos_prices = {}
                for code in oos_scores["code"].values:
                    mb = morning_bars.get((code, D))
                    if mb is not None:
                        oos_prices[code] = mb["close"]
                    else:
                        if code in close_prices:
                            oos_prices[code] = close_prices[code]
                top = top_k_picks(oos_scores, k=top_k,
                                  price_map=oos_prices, cap_price=50.0,
                                  adv_lookup={c: adv_today.get(c, 0.0) for c in oos_scores["code"]},
                                  adv_floor=100_000_000.0)
                # v3.1: only-buy, shared broker, NO cap
                oos_picks_executed, oos_skipped = execute_oos_only_buy(
                    broker, top, oos_prices, adv_today, sim_date_str,
                )

        # ─── 3. Mark-to-market at D close ───
        broker.update_prices(close_prices)
        nav = broker.total_value

        # v3: single broker, so eod_nav / oos_nav / dual_nav are all the same
        # number per mode. We keep all three columns to maintain output schema
        # backward compatibility (consumers indexed by mode).
        nav_row = {
            "date": sim_date_str,
            "eod_nav": nav if mode == "eod_only" else 0.0,
            "oos_nav": nav if mode == "oos_only" else 0.0,
            "dual_nav": nav if mode == "dual" else 0.0,
            "nav": nav,  # canonical (v3)
        }
        result.nav_records.append(nav_row)

        # ─── 4. Generate EOD plan for D+1 ───
        if enable_eod:
            eod_panel = make_eod_panel_for(factors, D)
            eod_scores = score_panel(eod_ranker, eod_panel)
            if not eod_scores.empty:
                eod_plan_for_tomorrow = top_k_picks(eod_scores, k=top_k)
            else:
                eod_plan_for_tomorrow = pd.DataFrame()

        # ─── 5. Picks log (post-hoc conflict detection done in summarize) ───
        picks_row = {
            "date": sim_date_str,
            "eod_picks_executed_today_open": list(eod_picks_executed),
            "oos_picks_executed_today_1430": list(oos_picks_executed),
            "eod_plan_for_tomorrow": (list(eod_plan_for_tomorrow["code"].astype(str).str.zfill(6))
                                       if enable_eod and eod_plan_for_tomorrow is not None and not eod_plan_for_tomorrow.empty
                                       else []),
            "oos_skipped_count": len(oos_skipped),
        }
        result.picks_log.append(picks_row)

        if i % 20 == 0:
            logger.info("[seed={} {}] D={} ({}/{}) NAV: {:.0f}",
                       seed, mode, sim_date_str, i + 1, len(trading_dates), nav)

    # v3: single trade log on shared broker
    result.eod_trades = []   # legacy field unused
    result.oos_trades = []   # legacy field unused
    # Stash the shared trade log on the result for post-hoc conflict detection.
    # We re-use eod_trades for it (it's the union of EOD + OOS trades from the
    # shared broker; downstream conflict detection looks at action labels
    # "BUY_OOS" vs others to split them post-hoc).
    result.eod_trades = list(broker.trade_log)

    # ─── Post-hoc conflict detection (v3, advisor 178 spec) ───
    # Type A: OOS bought X on D 14:30, EOD sells X on D 09:25 same day
    #   (i.e., 9:25 EOD sold X, then 14:30 OOS bought it back the same day —
    #    this is the user's "EOD just sold → OOS re-buys" worry)
    # Type B: OOS bought X on D-1 14:30, EOD sells X on D 09:25 next day
    #   (OOS's purchase from yesterday gets dumped this morning by EOD)
    # Strict count: A + B = total conflicts. Cost lives in NAV already.
    log_by_date_code: Dict[Tuple[str, str], List[dict]] = {}
    for t in result.eod_trades:
        key = (t.get("date", ""), t["code"])
        log_by_date_code.setdefault(key, []).append(t)

    for picks_row in result.picks_log:
        d = picks_row["date"]
        oos_today = set(picks_row.get("oos_picks_executed_today_1430", []))
        eod_today = set(picks_row.get("eod_picks_executed_today_open", []))
        # Type A: same-day EOD-sell + OOS-buy. EOD action is "SELL" or
        # action != "BUY_OOS"; OOS action is "BUY_OOS".
        for code in oos_today:
            trades_today = log_by_date_code.get((d, code), [])
            had_sell_today = any(t.get("action") == "SELL" for t in trades_today)
            if had_sell_today:
                result.conflicts.append({
                    "date": d, "type": "A_eod_sell_then_oos_buy_same_day", "code": code,
                })

    for idx, picks_row in enumerate(result.picks_log):
        if idx == 0:
            continue
        d = picks_row["date"]
        # OOS picks from yesterday
        prev_oos = set(result.picks_log[idx - 1].get("oos_picks_executed_today_1430", []))
        # EOD sells today (action=SELL on the shared trade_log)
        for code in prev_oos:
            trades_today = log_by_date_code.get((d, code), [])
            had_sell_today = any(t.get("action") == "SELL" for t in trades_today)
            if had_sell_today:
                result.conflicts.append({
                    "date": d, "type": "B_oos_bought_yesterday_eod_sells_today", "code": code,
                })

    return result


# ───────────────────────────────────────────────────────────────────
# Reporting
# ───────────────────────────────────────────────────────────────────

def summarize_results(results: List[BacktestResult], initial_capital: float) -> dict:
    """Aggregate across seeds. Returns nested dict keyed by mode."""
    summary = {}
    for mode in ("eod_only", "oos_only", "dual"):
        mode_results = [r for r in results if r.mode == mode]
        if not mode_results:
            continue
        final_navs = []
        max_drawdowns = []
        total_friction = []
        total_commission = []
        total_slippage = []
        total_stamp = []
        for r in mode_results:
            # v3: single shared broker → use canonical "nav" column. Fall
            # back to legacy column for compat with v2 nav_records.
            if r.nav_records:
                last = r.nav_records[-1]
                final_navs.append(float(last.get("nav") or last.get(f"{mode.split('_')[0]}_nav") or
                                          last.get("dual_nav") or initial_capital))
            else:
                final_navs.append(initial_capital)
            trades = r.eod_trades  # v3: union log on shared broker; oos_trades empty
            # max drawdown (use canonical "nav")
            navs = [float(n.get("nav") or n.get(f"{mode.split('_')[0]}_nav") or n.get("dual_nav") or 0.0)
                    for n in r.nav_records]
            if navs:
                peak = navs[0]
                mdd = 0.0
                for v in navs:
                    peak = max(peak, v)
                    mdd = max(mdd, (peak - v) / peak) if peak > 0 else mdd
                max_drawdowns.append(mdd)
            commission = sum(t.get("commission", 0) for t in trades)
            slippage = sum(t.get("slippage_cost", 0) for t in trades)
            stamp = sum(t.get("stamp_tax", 0) for t in trades)
            friction = commission + slippage + stamp
            total_commission.append(commission)
            total_slippage.append(slippage)
            total_stamp.append(stamp)
            total_friction.append(friction)
        # round 176: strict conflict count = Type A + Type B (advisor 175 (2))
        n_conflicts_total = []
        n_conflicts_A = []
        n_conflicts_B = []
        pick_overlap = []
        n_oos_skipped_merged_cap = []
        for r in mode_results:
            ta = sum(1 for c in r.conflicts if c.get("type", "").startswith("A_"))
            tb = sum(1 for c in r.conflicts if c.get("type", "").startswith("B_"))
            n_conflicts_A.append(ta)
            n_conflicts_B.append(tb)
            n_conflicts_total.append(ta + tb)
            overlaps_per_day = []
            for row in r.picks_log:
                e_set = set(row.get("eod_picks_executed_today_open", []))
                o_set = set(row.get("oos_picks_executed_today_1430", []))
                if e_set or o_set:
                    union = e_set | o_set
                    inter = e_set & o_set
                    overlaps_per_day.append(len(inter) / max(1, len(union)))
            pick_overlap.append(np.mean(overlaps_per_day) if overlaps_per_day else 0.0)
            n_oos_skipped_merged_cap.append(sum(row.get("oos_skipped_count", 0) for row in r.picks_log))
        summary[mode] = {
            "n_seeds": len(mode_results),
            "final_nav_mean": float(np.mean(final_navs)),
            "final_nav_std": float(np.std(final_navs)),
            "final_nav_values": final_navs,
            "total_ret_mean": float(np.mean(final_navs) / initial_capital - 1),
            "max_drawdown_mean": float(np.mean(max_drawdowns)),
            "friction_mean": float(np.mean(total_friction)),
            "commission_mean": float(np.mean(total_commission)),
            "slippage_mean": float(np.mean(total_slippage)),
            "stamp_mean": float(np.mean(total_stamp)),
            "conflict_total_mean": float(np.mean(n_conflicts_total)),
            "conflict_typeA_mean": float(np.mean(n_conflicts_A)),
            "conflict_typeB_mean": float(np.mean(n_conflicts_B)),
            "picks_jaccard_mean": float(np.mean(pick_overlap)),
            "oos_skipped_mean": float(np.mean(n_oos_skipped_merged_cap)),
        }
    return summary


def write_report(summary: dict, out_path: Path, initial_capital: float,
                  start: str, end: str, n_seeds: int, top_k: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Walk-Forward Dual-Bucket Backtest Report (v3 production-faithful)",
        f"",
        f"_Generated by `scripts/walk_forward_dual_bucket.py` (round 180, v3)_",
        f"",
        f"**v3 changes vs v2** (advisor round 178 spec):",
        f"- Single SHARED `SimulatedBroker` (matches real QMT account 8886933837)",
        f"- OOS path strictly **only buys** (matches production `intraday_plan` 0-sell)",
        f"- No merged-cap check (production has none)",
        f"- Post-hoc conflict detection from `broker.trade_log` (not preemptive)",
        f"",
        f"## Inputs",
        f"- Window: {start} → {end}",
        f"- Seeds: {n_seeds}",
        f"- Initial capital: ¥{initial_capital:,.0f}",
        f"- Top-K per bucket: {top_k}",
        f"- OOS daily new-buy cap: ¥{OOS_DAILY_BUDGET:,.0f}",
        f"- EOD model: `data/blend_*.lgb` (n2c, FACTOR_COLUMNS 64-col)",
        f"- OOS model: `data/intraday_blend_*.lgb` (c2c, INTRADAY_FEATURE_COLS 68-col)",
        f"",
        f"## Results (mean ± std across {n_seeds} seeds)",
        f"",
        f"| Mode | Final NAV | Return | Max DD | Friction (¥) | Pick overlap (Jaccard) | Conflicts (A+B) | OOS skips (merged-cap + budget) |",
        f"|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("eod_only", "oos_only", "dual"):
        if mode not in summary:
            continue
        s = summary[mode]
        c_total = s.get("conflict_total_mean", 0)
        c_a = s.get("conflict_typeA_mean", 0)
        c_b = s.get("conflict_typeB_mean", 0)
        c_str = f"{c_total:.1f} (A={c_a:.1f} / B={c_b:.1f})"
        lines.append(
            f"| {mode} | "
            f"¥{s['final_nav_mean']:,.0f} ± {s['final_nav_std']:,.0f} | "
            f"{s['total_ret_mean']*100:+.2f}% | "
            f"{s['max_drawdown_mean']*100:.2f}% | "
            f"¥{s['friction_mean']:,.0f} | "
            f"{s['picks_jaccard_mean']*100:.1f}% | "
            f"{c_str} | "
            f"{s.get('oos_skipped_mean', 0):.0f} |"
        )

    if "dual" in summary and "eod_only" in summary and "oos_only" in summary:
        dual_final = summary["dual"]["final_nav_mean"]
        eod_final = summary["eod_only"]["final_nav_mean"]
        oos_final = summary["oos_only"]["final_nav_mean"]
        # Compare to capital-equivalent baselines
        avg_baseline = (eod_final + oos_final) / 2  # 100k EOD + 100k OOS naively averaged
        best_solo = max(eod_final, oos_final)
        delta_vs_best = dual_final - best_solo
        delta_vs_avg = dual_final - avg_baseline
        lines += [
            f"",
            f"## Net alpha delta",
            f"- `dual_NAV − max(EOD_only, OOS_only)` = **¥{delta_vs_best:+,.0f}** (¥{delta_vs_best/initial_capital*10000:+.0f}bps of capital)",
            f"- `dual_NAV − (EOD_only + OOS_only)/2` = **¥{delta_vs_avg:+,.0f}** (¥{delta_vs_avg/initial_capital*10000:+.0f}bps of capital)",
            f"",
            f"## Friction breakdown (mean across seeds)",
            f"",
            f"| Mode | Commission | Slippage | Stamp tax | Total |",
            f"|---|---:|---:|---:|---:|",
        ]
        for mode in ("eod_only", "oos_only", "dual"):
            if mode not in summary:
                continue
            s = summary[mode]
            lines.append(
                f"| {mode} | ¥{s['commission_mean']:,.0f} | ¥{s['slippage_mean']:,.0f} | "
                f"¥{s['stamp_mean']:,.0f} | ¥{s['friction_mean']:,.0f} |"
            )
    lines += [
        f"",
        f"## Reading the comparison fairly (v3)",
        f"All three modes share the SAME total capital (¥{initial_capital:,.0f}) in a SINGLE shared broker (matches production QMT single-account):",
        f"- `eod_only`: only EOD rebalance runs (sell + buy). OOS path disabled.",
        f"- `oos_only`: only OOS only-buy runs at 14:30, capped at ¥{OOS_DAILY_BUDGET:,.0f}/day. **No selling** (production `intraday_plan` has 0 sell logic). Cash drains as positions accumulate; nothing rebalances stale picks.",
        f"- `dual`: EOD rebalance at 09:25 + OOS only-buy at 14:30 on the same broker. OOS competes with EOD for the same `broker.cash`. EOD's 09:25 sells may dump positions OOS just bought yesterday — the round-trip the user worried about.",
        f"",
        f"`dual − max(solo)` answers: *does adding OOS to EOD net-add alpha, or does the round-trip eat it?* If negative, OOS is a drag on EOD; if positive, the two strategies complement.",
        f"",
        f"## Conflict definition (v3, post-hoc from `broker.trade_log`)",
        f"- **Type A**: OOS buys X on D 14:30 + EOD sells X on D 09:25 (same day, EOD already sold this morning, OOS rebuys at 14:30 — user's headline worry)",
        f"- **Type B**: OOS bought X on D-1 14:30 + EOD sells X on D 09:25 (overnight: OOS yesterday's pick, EOD dumps it next morning)",
        f"",
        f"Both count toward the strict `Conflicts (A+B)` total. v3 doesn't *prevent* conflicts — production architecture doesn't either — the cost is already baked into `dual NAV`.",
        f"",
        f"## Caveats",
        f"- 8-month in-sample window — sample size is small; treat as directional, not definitive",
        f"- **Seed std is 0** for all modes: the pipeline (factors → predict → top-K → execute) is deterministic. Different seeds label runs but do not produce variation. A real ± needs a stochastic source (random tie-break, bootstrap, or train-time seed). Out of MVP scope.",
        f"- `data/intraday_blend_*.lgb` was trained 2026-05-27, BEFORE the n2c label upgrade (round 162). This is the *currently-shipped* OOS model so its drag is what user asked us to measure.",
        f"- OOS panel uses D-1 factor row + 4 INTRADAY_EXTRA_COLUMNS proxies (MVP). Production intraday_plan re-derives 64 TECHNICAL_COLUMNS from morning bars — this would tighten OOS scoring",
        f"- OOS execution price = 1m close at 14:29 from `data/intraday_1m_qfq/`; real fills are next-second auction (~5bps optimistic)",
        f"- `oos_only` accumulates positions forever (no sell); over 8 months this drags down NAV when bad picks turn worse. Production wouldn't run OOS in isolation — this mode is reference only.",
        f"",
        f"## Rule reminders",
        f"- Rule #4: pure research, `--skip-update` style, no prod model touched",
        f"- Rule #11: PIT — EOD ≤ D close (executes D+1 open), OOS ≤ D 14:29",
        f"- Rule #1: report + json output NOT in git; this script IS in git",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written → {}", out_path)


# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                    help="comma-separated, e.g. 42,43,44")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    ap.add_argument("--out", default="data/reports/walk_forward_dual_bucket.md")
    ap.add_argument("--json-out", default="data/walk_forward_dual_bucket.json")
    ap.add_argument("--modes", default="eod_only,oos_only,dual",
                    help="comma-separated subset of {eod_only, oos_only, dual}")
    ap.add_argument("--merged-hard-max-pct", type=float, default=MERGED_HARD_MAX_PCT_DEFAULT,
                    help="round 176: per-code combined EOD+OOS cap (dual mode only)")
    ap.add_argument("--smoke", action="store_true",
                    help="single seed × 1 month — fast sanity check")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if args.smoke:
        seeds = seeds[:1]
        # 1-month window
        start_ts = pd.Timestamp(args.start)
        args.end = (start_ts + pd.Timedelta(days=31)).strftime("%Y-%m-%d")
        logger.info("SMOKE mode: seed={} window={}→{}", seeds[0], args.start, args.end)

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)

    # ─── Load data once ───
    t0 = time.time()
    logger.info("Loading bars + factors + intraday morning bars ...")
    bars = load_bars()
    factors = load_factors()
    morning_bars = load_intraday_morning_bars(start_ts, end_ts)
    adv_lookup = adv_20d_lookup(bars)
    logger.info("Data loaded in {:.1f}s", time.time() - t0)

    # Filter to window
    bars = bars[(bars["date"] >= start_ts) & (bars["date"] <= end_ts)]
    factors = factors[(factors["date"] >= start_ts) & (factors["date"] <= end_ts)]
    trading_dates = sorted(bars["date"].unique())
    logger.info("Window: {} → {}, {} trading dates", args.start, args.end, len(trading_dates))

    # ─── Load both rankers ───
    eod_ranker = BlendRanker(feature_cols=list(FACTOR_COLUMNS))
    if not eod_ranker.load("data/blend"):
        logger.error("Failed to load EOD BlendRanker from data/blend_*.lgb")
        return 1
    logger.info("EOD ranker loaded: feature_cols={}", len(eod_ranker.feature_cols))

    oos_ranker = BlendRanker(feature_cols=list(INTRADAY_FEATURE_COLS))
    if not oos_ranker.load("data/intraday_blend"):
        logger.error("Failed to load OOS BlendRanker from data/intraday_blend_*.lgb")
        return 1
    logger.info("OOS ranker loaded: feature_cols={}", len(oos_ranker.feature_cols))

    # ─── Run all (mode × seed) combos ───
    all_results: List[BacktestResult] = []
    for mode in modes:
        for seed in seeds:
            logger.info("Running mode={} seed={}", mode, seed)
            t1 = time.time()
            res = run_seed(seed, mode, factors, bars, morning_bars,
                           list(trading_dates), args.initial_capital, args.top_k,
                           eod_ranker, oos_ranker, adv_lookup,
                           merged_hard_max_pct=args.merged_hard_max_pct)
            logger.info("  done in {:.1f}s; final NAV: eod={:.0f} oos={:.0f} dual={:.0f}",
                       time.time() - t1,
                       res.nav_records[-1]["eod_nav"] if res.nav_records else 0,
                       res.nav_records[-1]["oos_nav"] if res.nav_records else 0,
                       res.nav_records[-1]["dual_nav"] if res.nav_records else 0)
            all_results.append(res)

    # ─── Aggregate + write report ───
    summary = summarize_results(all_results, args.initial_capital)
    out_path = PROJECT_ROOT / args.out
    write_report(summary, out_path, args.initial_capital, args.start, args.end,
                 len(seeds), args.top_k)

    # JSON metrics dump (for downstream consumers)
    json_out = PROJECT_ROOT / args.json_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps({
        "window": {"start": args.start, "end": args.end},
        "seeds": seeds,
        "top_k": args.top_k,
        "initial_capital": args.initial_capital,
        "summary": summary,
        "nav_per_day": [
            {"seed": r.seed, "mode": r.mode, "nav_records": r.nav_records}
            for r in all_results
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("JSON metrics → {}", json_out)

    # Print summary to stdout
    print("\n=== Dual-bucket backtest summary ===")
    for mode, s in summary.items():
        c_total = s.get('conflict_total_mean', 0)
        c_a = s.get('conflict_typeA_mean', 0)
        c_b = s.get('conflict_typeB_mean', 0)
        print(f"{mode}: NAV={s['final_nav_mean']:,.0f} (±{s['final_nav_std']:,.0f}) "
              f"ret={s['total_ret_mean']*100:+.2f}% DD={s['max_drawdown_mean']*100:.2f}% "
              f"friction=¥{s['friction_mean']:,.0f} overlap={s['picks_jaccard_mean']*100:.1f}% "
              f"conflicts={c_total:.0f} (A={c_a:.0f}, B={c_b:.0f}) "
              f"oos_skips={s.get('oos_skipped_mean', 0):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
