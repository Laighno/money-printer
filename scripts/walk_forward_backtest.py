"""Walk-Forward Backtest with monthly model retraining and daily rebalancing.

Monthly:
  1. Retrain LightGBM on expanding window (TRAIN_START → current month)
     Training labels (fwd_ret) are excluded for the last HORIZON trading days
     to prevent forward-return label leakage.

Daily:
  2. Score all universe stocks with current model at close, store as pending signal
  3. Next day: execute pending trades at T+1 open prices (no look-ahead)
  4. Track daily NAV

After the backtest, retrain the production model on the full 2015-2026 dataset.

Usage:
    python scripts/walk_forward_backtest.py
"""

from __future__ import annotations

import bisect
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

# P7-γ (docs/dialog/ rounds 50-51, rule #7): PYTHONHASHSEED must be 0 for
# deterministic universe iteration / set ordering across processes. It is
# environment-only (cannot be set after Python starts), so we warn loudly
# instead of failing — the run will still produce reasonable numbers but
# may not be byte-perfect reproducible across processes if hashseed varies.
if os.environ.get("PYTHONHASHSEED") != "0":
    sys.stderr.write(
        "WARNING: PYTHONHASHSEED != 0 → universe iteration / set ordering "
        "may be nondeterministic across processes. For byte-perfect "
        "reproducible backtest, run as:\n"
        "  PYTHONHASHSEED=0 LGBM_SEED=42 WF_FEATURE_PRESET=W_BASELINE "
        "python scripts/walk_forward_backtest.py [...]\n"
        "See P7-3 docs/dialog/ rounds 50-51 + rule #7 in docs/TODO.md.\n"
    )

from mp.account.broker import FeeSchedule, SimulatedBroker
from mp.backtest.engine import calc_performance
from mp.backtest.ml_backtest import _build_factor_panel, _prefetch_bars
from mp.data.fetcher import get_daily_bars, get_index_constituents, get_index_constituents_at
from mp.ml.dataset import (
    CURATED_COLUMNS,
    FACTOR_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    FUNDAMENTAL_TREND_COLUMNS,
    INDUSTRY_RANK_COLUMNS,
    TECHNICAL_COLUMNS,
    _fetch_financial_history,
    _align_fundamentals_to_dates,
    _add_industry_relative_features,
)
from mp.ml.model import FEATURE_COLS, StockRanker
from mp.ml.regime_features import REGIME_COLUMNS, add_regime_features

# ──────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────

# env-overridable for the round-133 AUM-scaling sweep (sqrt-impact scales with
# notional = INITIAL_CAPITAL/TOP_K, so larger AUM → larger per-stock impact).
INITIAL_CAPITAL = int(os.environ.get("INITIAL_CAPITAL", "100000"))   # default 10万元
# Universe widened 2026-05-14 from ZZ500-only to HS300+ZZ500 (~800 unique).
# Both indices have ≥120 PIT snapshots in DB so training preserves
# point-in-time integrity (no survivorship bias from today's HS300 list).
UNIVERSES: tuple[str, ...] = ("hs300", "zz500")
UNIVERSE = "+".join(UNIVERSES)  # for logging / model_path conventions
TRAIN_START = "20160501"        # expanding window start (Baidu 近十年 covers from 2016-04)


def _merged_pit_at(date_str: str) -> list[str] | None:
    """PIT constituent union across UNIVERSES at a given date.  Returns
    None only if no member index has a snapshot for that date."""
    from mp.data.fetcher import get_index_constituents_at as _at
    merged: set[str] = set()
    any_found = False
    for idx in UNIVERSES:
        snap = _at(idx, date_str)
        if snap:
            merged.update(snap)
            any_found = True
    return sorted(merged) if any_found else None


def _merged_all_snapshots(store) -> tuple[list[str], list[str]]:
    """Return (sorted_union_codes, all_snapshot_dates) across UNIVERSES."""
    union: set[str] = set()
    dates: set[str] = set()
    for idx in UNIVERSES:
        ds = store.list_constituent_snapshot_dates(idx)
        dates.update(ds)
        for d in ds:
            c = store.load_constituent_snapshot_at(idx, d)
            if c:
                union.update(c)
    return sorted(union), sorted(dates)


def _merged_current() -> list[str]:
    """Merged current constituents across UNIVERSES (today's list)."""
    from mp.data.fetcher import get_recommendation_universe
    return get_recommendation_universe(UNIVERSES)
BT_START = os.environ.get("BT_START", "20200101")   # first trading month (env-overridable for windowed A/B, P11 round 117)
BT_END = os.environ.get("BT_END", "20260401")       # last month (exclusive)
TOP_K = int(os.environ.get("TOP_K", "10"))
HORIZON = 20                    # forward return horizon in trading days

# Round 155 (label-alignment research): which label to train on.
#   "close_to_close" (default, current production): fwd_ret[i] = close[i+H]/close[i]−1
#       → assumes entry at close[i]; rewards stocks that gap UP overnight before
#       T+1 open since the gap is captured in the label but NOT capturable when
#       you actually execute at open[i+1] (train/serve mismatch).
#   "next_open_to_close" (round 155 fix): fwd_ret[i] = close[i+H]/open[i+1]−1
#       → label entry = open[i+1] = real T+1 open execution price; eliminates
#       the overnight-gap windfall and the model no longer prefers gap-up names
#       whose alpha you can't capture.
# Cache key (factors.parquet path) is branched by LABEL_KIND so the legacy
# close_to_close cache stays untouched.
LABEL_KIND = os.environ.get("LABEL_KIND", "close_to_close")
assert LABEL_KIND in ("close_to_close", "next_open_to_close"), \
    f"Unknown LABEL_KIND={LABEL_KIND}"
SLIPPAGE_BPS = int(os.environ.get("SLIPPAGE_BPS", "5"))
COMMISSION_BPS = int(os.environ.get("COMMISSION_BPS", "3"))
COST_AWARE_REBALANCE = True      # skip swaps where score gap < round-trip cost

# Rebalance policy — controls when to rebalance positions that are still
# selected (same Top-K names as yesterday).  Positions that drop out of the
# selection are always sold regardless of policy.
#
#   "on_change"        : only rebalance when selection changes (low turnover).
#                        Weights drift with price movement — default, preserves
#                        historical behavior.
#   "drift_threshold"  : additionally rebalance on days where any held position's
#                        weight deviates from equal-weight target by more than
#                        ``MAX_WEIGHT_DRIFT`` (e.g. 0.10 = 10 percentage points).
#                        Enforces stricter equal-weight at the cost of turnover.
#
# Override with env var REBALANCE_POLICY for A/B runs.
REBALANCE_POLICY = os.environ.get("REBALANCE_POLICY", "on_change")
MAX_WEIGHT_DRIFT = float(os.environ.get("MAX_WEIGHT_DRIFT", "0.10"))
assert REBALANCE_POLICY in ("on_change", "drift_threshold"), \
    f"Unknown REBALANCE_POLICY={REBALANCE_POLICY}"

# Which ranker to use for monthly retrain.  "stock" = StockRanker on fwd_ret
# (the historical default, validated at 1.81 Sharpe / 57% annual after the
# 2026-04-28 data fix).  "blend" = BlendRanker (0.8 primary on excess_ret +
# 0.2 extreme), which is what daily_report.py and paper_trade.py actually use.
# Validating "blend" here is the missing piece — until this completes we
# don't know if BlendRanker matches/beats StockRanker on the same backtest.
RANKER_KIND = os.environ.get("RANKER_KIND", "blend")  # P10-2: default to production path (BlendRanker) — see Rule #11
# "intraday_blend" added P11-3 (round 79 spec): BlendRanker trained on
# INTRADAY_FEATURE_COLS (FACTOR_COLUMNS + 4 morning extras = 68 cols),
# panel augmented per-row via EOD-proxy. Option C entry: walk_forward
# execution timing UNCHANGED (T+1 open), only the model variable
# differs — clean A/B vs `blend` per Rule #10.
assert RANKER_KIND in ("stock", "blend", "intraday_blend"), \
    f"Unknown RANKER_KIND={RANKER_KIND}"

# P11-4 Phase C (round 91): ENTRY_TIME selects execution price source.
#   "t_plus_1_open" (default): use next-day open (matches production EOD blend
#                              path; what P10-CLOSE baseline + P11-3 used)
#   "14_30":                   use T 14:30 close from data/intraday_1m parquet
#                              for dates ≥ 2025-09-01 where available, else T
#                              close fallback. Models the 14:30 production
#                              entry path; pair with RANKER_KIND=intraday_blend
#                              for Rule #11 alignment.
ENTRY_TIME = os.environ.get("ENTRY_TIME", "t_plus_1_open")
assert ENTRY_TIME in ("t_plus_1_open", "14_30"), \
    f"Unknown ENTRY_TIME={ENTRY_TIME}"

# Round 147 (Design 2 fix-and-rerun): which intraday-1m dir to read for the
# 14:30 entry price + Arm B injection. Default = the legacy NON-adjusted
# `data/intraday_1m`; the round-147 clean rerun points this at the
# front-adjusted (qfq) re-fetch so the 14:30 bar shares the qfq scale of the
# EOD history + model training (fixes defect ① 复权尺度混用).
INTRADAY_DIR = os.environ.get("INTRADAY_DIR", "data/intraday_1m")

# Round 147 (fix ② 时序错位): ENTRY_TIME=14_30 now means TRUE SAME-DAY —
# decide on D's ≤14:29 injected factors and EXECUTE the same day D at the
# 14:29 close (≈14:30 fill), instead of the legacy pending-mechanism lag that
# scored D and filled D+1 14:30 (隔日). Set INTRADAY_NEXT_DAY=1 to restore the
# legacy next-day fill (for the A/B that isolates the timing effect from ①).
# Only affects ENTRY_TIME=14_30; baseline + ③(t_plus_1_open) are untouched.
INTRADAY_NEXT_DAY = os.environ.get("INTRADAY_NEXT_DAY", "0") == "1"
SAME_DAY_14_30 = (ENTRY_TIME == "14_30") and not INTRADAY_NEXT_DAY

# P11-4 Phase C (round 91): INTRADAY_HYBRID enables hybrid feature compute
# when RANKER_KIND=intraday_blend. 1 → 2025-09+ dates use real intraday
# from data/intraday_1m/*.parquet, else EOD-proxy (matches Phase B
# training-time hybrid panel). 0 → 100% EOD-proxy (matches P11-3 / P11-2).
INTRADAY_HYBRID = os.environ.get("INTRADAY_HYBRID", "0") == "1"

# Position-sizing scheme for the Top-K selection.
#   "equal"      : 1/N each (BASELINE-validated, default).
#   "inverse_vol": weight ∝ 1 / volatility_20d, normalized.  Lower-vol stocks
#                  get bigger positions, with the goal of reducing portfolio
#                  vol & DD.  Falls back to equal-weight if any vol is missing.
# Position-sizing default switched 2026-04-29 from "equal" to "conviction":
# conviction (weight ∝ model's predicted excess) outperforms equal-weight on
# Sharpe 2.01 vs 1.88, annual 69.84% vs 54.25%, Calmar 3.07 vs 2.21.
# Leak audit (POSITION_SIZING=conviction_oracle, using realized fwd_ret as
# weights) gave 366% annual / Sharpe 6.77 — 5x higher than real conviction —
# proving the model's edge is real, not future-leaked.
POSITION_SIZING = os.environ.get("POSITION_SIZING", "conviction")
assert POSITION_SIZING in ("equal", "inverse_vol", "conviction", "vol_target",
                           "conviction_oracle"), f"Unknown POSITION_SIZING={POSITION_SIZING}"
# "conviction_oracle" = LEAK-CHECK ONLY.  Uses REALIZED fwd_ret as conviction,
# which is information the model could not have at decision time.  Result is
# the upper bound of "perfect-conviction" performance.  If real conviction is
# anywhere close to oracle, real conviction is leaking.  If oracle is much
# higher, real conviction is just exploiting genuine model edge.

# Regime-proxy features: zz500 momentum/volatility/drawdown + cross-sectional
# breadth, added as per-date features (same value for all stocks on a date).
# LGBM can learn regime↔factor interactions via tree splits.  Toggle via env
# var USE_REGIME_FEATURES=1 for A/B testing.
USE_REGIME_FEATURES = os.environ.get("USE_REGIME_FEATURES", "0") == "1"

CACHE_DIR = Path("data/wf_cache")
REPORT_PATH = Path(os.environ.get(
    "WF_REPORT_PATH", "data/reports/walk_forward_result.md"))

# ──────────────────────────────────────────────────────────────────────
# Data Preparation
# ──────────────────────────────────────────────────────────────────────

def _load_or_fetch_bars(codes: List[str]) -> Dict[str, pd.DataFrame]:
    """Pre-fetch all daily bars with parquet cache."""
    cache_path = CACHE_DIR / "bars.parquet"
    if cache_path.exists():
        logger.info("Loading cached bars from {}", cache_path)
        big = pd.read_parquet(cache_path)
        big["date"] = pd.to_datetime(big["date"])
        bars_map: Dict[str, pd.DataFrame] = {}
        for code, grp in big.groupby("code"):
            bars_map[str(code)] = grp.sort_values("date").reset_index(drop=True)
        logger.info("Loaded bars for {} stocks from cache", len(bars_map))
        return bars_map

    # Fetch from network — need extra warmup before TRAIN_START for factors
    warmup_start = "20140601"
    logger.info("Fetching bars for {} stocks from {}...", len(codes), warmup_start)
    bars_map = _prefetch_bars(codes, warmup_start, date.today().strftime("%Y%m%d"))

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for code, df in bars_map.items():
        df = df.copy()
        df["code"] = code
        frames.append(df)
    if frames:
        big = pd.concat(frames, ignore_index=True)
        big.to_parquet(cache_path, index=False)
        logger.info("Cached bars for {} stocks to {}", len(bars_map), cache_path)
    return bars_map


def _load_or_build_factors(
    bars_map: Dict[str, pd.DataFrame],
    codes: List[str],
) -> pd.DataFrame:
    """Build full factor panel with forward returns, with parquet cache."""
    # Round 155: cache key branches on LABEL_KIND so the new next_open_to_close
    # label gets its own file and never collides with the legacy close_to_close
    # cache. Default (close_to_close) keeps the original `factors.parquet` path
    # — no rebuild needed for existing runs.
    cache_path = CACHE_DIR / ("factors.parquet" if LABEL_KIND == "close_to_close"
                              else f"factors_label_{LABEL_KIND}.parquet")
    if cache_path.exists():
        logger.info("Loading cached factor panel from {}", cache_path)
        panel = pd.read_parquet(cache_path)
        panel["date"] = pd.to_datetime(panel["date"])
        logger.info("Loaded factor panel: {} rows", len(panel))
        return panel

    # Build technical factors only; fundamentals are aligned below in this
    # script (WF owns its own richer merge path — keep _build_factor_panel
    # from doing the same work twice).
    logger.info("Building factor panel for {} stocks...", len(bars_map))
    factor_map = _build_factor_panel(bars_map, include_fundamentals=False)

    # Merge into one big DataFrame
    frames = []
    for code, fdf in factor_map.items():
        frames.append(fdf)
    if not frames:
        raise RuntimeError("Factor computation failed for all stocks")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])

    # Add fundamental factors (time-aligned)
    logger.info("Fetching and aligning fundamental data...")

    # Pre-load all historical PE/PB/total_mv once; split per-code below.
    val_hist_map: Dict[str, pd.DataFrame] = {}
    try:
        from mp.data.store import DataStore as _DS
        _date_min = panel["date"].min().strftime("%Y-%m-%d")
        _date_max = panel["date"].max().strftime("%Y-%m-%d")
        vh_all = _DS().load_valuation_history(codes=codes, start=_date_min, end=_date_max)
        if not vh_all.empty:
            for c, grp in vh_all.groupby("code"):
                val_hist_map[c] = grp[["date", "pe_ttm", "pb", "total_mv"]].copy()
            logger.info("Loaded historical valuation for {}/{} stocks",
                        len(val_hist_map), len(codes))
        else:
            logger.warning("Valuation table empty — PE/PB/total_mv will be NaN")
    except Exception as e:
        logger.warning("Failed to load historical valuation: {}", e)

    for code in codes:
        mask = panel["code"] == code
        if mask.sum() == 0:
            continue
        fin_hist = _fetch_financial_history(code)
        code_dates = panel.loc[mask, "date"]
        val_hist = val_hist_map.get(code)
        fund = _align_fundamentals_to_dates(
            code_dates, fin_hist, valuation_row=None, valuation_hist=val_hist,
        )
        for col in FUNDAMENTAL_COLUMNS + FUNDAMENTAL_TREND_COLUMNS:
            new_vals = fund[col]
            # For total_mv_log, fall back to the derived float-mv proxy
            # (computed in _build_factor_panel) when historical valuation is
            # missing (e.g. pre-2016 dates beyond Baidu's 近十年 lookback).
            if col == "total_mv_log":
                existing = panel.loc[mask, col].to_numpy()
                merged_vals = np.where(
                    np.isnan(new_vals), existing, new_vals,
                )
                panel.loc[mask, col] = merged_vals
            else:
                panel.loc[mask, col] = new_vals

    # Compute forward returns for each stock
    logger.info("Computing {}-day forward returns...", HORIZON)
    fwd_parts = []
    for code, grp in panel.groupby("code"):
        grp = grp.sort_values("date").copy()
        close_vals = grp.merge(
            bars_map.get(str(code), pd.DataFrame()),
            on="date", how="left",
        )
        if "close" in close_vals.columns:
            close_arr = close_vals["close"].values.astype(float)
            n = len(close_arr)
            fwd = np.full(n, np.nan)
            if LABEL_KIND == "next_open_to_close":
                # Round 155: entry = open[i+1] (real T+1 fill); exit unchanged
                # at close[i+HORIZON]. Both indexes are bounded by the loop's
                # range(n - HORIZON) since HORIZON ≥ 1 ⇒ i+1 ≤ n-HORIZON ≤ n-1.
                open_arr = (close_vals["open"].values.astype(float)
                            if "open" in close_vals.columns else None)
                if open_arr is None:
                    grp["fwd_ret"] = np.nan
                    fwd_parts.append(grp)
                    continue
                for i in range(n - HORIZON):
                    entry = open_arr[i + 1] if i + 1 < n else np.nan
                    if entry > 0 and not np.isnan(entry) and close_arr[i + HORIZON] > 0:
                        fwd[i] = close_arr[i + HORIZON] / entry - 1.0
            else:
                for i in range(n - HORIZON):
                    if close_arr[i] > 0:
                        fwd[i] = close_arr[i + HORIZON] / close_arr[i] - 1.0
            grp["fwd_ret"] = fwd
        else:
            grp["fwd_ret"] = np.nan
        fwd_parts.append(grp)
    panel = pd.concat(fwd_parts, ignore_index=True)

    # Drop warm-up NaN rows
    core_cols = TECHNICAL_COLUMNS[:13]
    panel = panel.dropna(subset=core_cols)

    # Industry-relative ranking features
    logger.info("Computing industry-relative ranking features...")
    try:
        # Point-in-time industry assignment: for each (code, date) use the
        # SW classification in effect on that date (merge_asof backward).
        # Avoids look-ahead from stocks that switched industries mid-panel
        # (~67% of A-share stocks have >=1 historical reclassification).
        from mp.data.fetcher import get_industry_history
        ind_hist = get_industry_history(universe=codes)
        panel = _add_industry_relative_features(panel, ind_hist)
        n_ranked = panel["pe_ind_rank"].notna().sum()
        logger.info("Industry rank features (PIT): {}/{} rows", n_ranked, len(panel))
    except Exception as e:
        logger.warning("Industry rank features failed, skipping: {}", e)
        for col in INDUSTRY_RANK_COLUMNS:
            if col not in panel.columns:
                panel[col] = np.nan

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache_path, index=False)
    logger.info("Cached factor panel: {} rows to {}", len(panel), cache_path)
    return panel


# ──────────────────────────────────────────────────────────────────────
# Walk-Forward Backtest
# ──────────────────────────────────────────────────────────────────────

def _get_monthly_retrain_dates(panel: pd.DataFrame) -> List[pd.Timestamp]:
    """Get first trading day of each month in [BT_START, BT_END) for model retraining."""
    bt_start = pd.Timestamp(BT_START)
    bt_end = pd.Timestamp(BT_END)
    dates = panel["date"].drop_duplicates().sort_values()
    dates = dates[(dates >= bt_start) & (dates < bt_end)]
    # First trading day of each month
    monthly = dates.groupby(dates.dt.to_period("M")).first()
    return monthly.tolist()


_ADV_PERIOD = 20   # days for rolling average daily value


_VOL_TARGET_ANNUAL = 0.25   # target annualized portfolio vol for "vol_target"


def _compute_position_weights(
    codes: list,
    dt: pd.Timestamp,
    panel_by_date: Dict[pd.Timestamp, pd.DataFrame],
    sizing: str,
    raw_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Return {code: weight}.

    For "equal" / "inverse_vol" / "conviction": weights sum to 1 (fully invested).
    For "vol_target": weights may sum to <1 (rest stays in cash) — this is the
    only scheme that scales total leverage.

    Schemes:
      - "equal":       1/N each.
      - "inverse_vol": weight_i ∝ 1 / vol_i, normalized.  Lower-vol stocks bigger.
      - "conviction":  weight_i ∝ max(raw_excess_i - floor, 0) + epsilon, normalized.
                       Higher predicted excess = bigger position.  Stocks below
                       a small floor get a token (epsilon) weight to maintain
                       diversification.  Falls back to equal if all raws are
                       missing or non-positive.
      - "vol_target":  scale total exposure so estimated portfolio vol ≈
                       _VOL_TARGET_ANNUAL (currently 25%).  Within the
                       portfolio still equal-weight.  Excess cash stays at 0%.
    """
    n = len(codes)
    if n == 0:
        return {}

    if sizing == "equal":
        w = 1.0 / n
        return {c: w for c in codes}

    if sizing == "inverse_vol":
        today_df = panel_by_date.get(dt)
        if today_df is None:
            return {c: 1.0 / n for c in codes}
        vol_map = {}
        sub = today_df[today_df["code"].isin(codes)]
        for _, row in sub.iterrows():
            v = row.get("volatility_20d")
            if pd.notna(v) and v > 0:
                vol_map[row["code"]] = float(v)
        if len(vol_map) < n:
            return {c: 1.0 / n for c in codes}
        inv = {c: 1.0 / vol_map[c] for c in codes}
        s = sum(inv.values())
        if s <= 0:
            return {c: 1.0 / n for c in codes}
        return {c: inv[c] / s for c in codes}

    if sizing in ("conviction", "conviction_oracle"):
        # Use raw excess predictions; floor at 0, add small epsilon for tail
        # diversification (so even the weakest Top-K name gets a token weight).
        # conviction_oracle uses realized fwd_ret instead of model prediction
        # (intentional leak, for upper-bound diagnostics only).
        if not raw_scores:
            return {c: 1.0 / n for c in codes}
        epsilon = 0.005   # 0.5pp floor as min "conviction"
        floor = 0.0
        adj = {c: max(raw_scores.get(c, 0.0) - floor, 0.0) + epsilon for c in codes}
        s = sum(adj.values())
        if s <= 0:
            return {c: 1.0 / n for c in codes}
        return {c: adj[c] / s for c in codes}

    if sizing == "vol_target":
        # Estimate portfolio vol assuming average pairwise correlation ρ.
        # var_p = avg(var) / N + (N-1)/N * avg(cov) ≈ avg(var)/N + (N-1)/N * ρ * avg(var)
        # For a long-only equity portfolio in same universe, ρ ≈ 0.4-0.6.
        # We use ρ = 0.5 as a reasonable mid-point, no need to estimate.
        today_df = panel_by_date.get(dt)
        if today_df is None:
            return {c: 1.0 / n for c in codes}
        sub = today_df[today_df["code"].isin(codes)]
        vols = []
        for _, row in sub.iterrows():
            v = row.get("volatility_20d")
            if pd.notna(v) and v > 0:
                vols.append(float(v))
        if len(vols) < n:
            return {c: 1.0 / n for c in codes}
        avg_var = sum(v ** 2 for v in vols) / len(vols)
        rho = 0.5
        port_var = avg_var / n + (n - 1) / n * rho * avg_var
        port_vol = port_var ** 0.5
        if port_vol <= 0:
            return {c: 1.0 / n for c in codes}
        leverage = min(1.0, _VOL_TARGET_ANNUAL / port_vol)   # never >1 (no margin)
        w = leverage / n
        return {c: w for c in codes}

    raise ValueError(f"Unknown sizing scheme: {sizing}")


def _build_price_adv_lookup(
    bars_map: Dict[str, pd.DataFrame],
) -> tuple[Dict, Dict, Dict]:
    """Build (code,date)->{close,open,adv} lookup dicts.

    ADV (average daily value) is the 20-day rolling mean of ``amount``
    ending the day BEFORE the trade date (no look-ahead).
    """
    close_lk: Dict[tuple, float] = {}
    open_lk: Dict[tuple, float] = {}
    adv_lk: Dict[tuple, float] = {}
    for code, df in bars_map.items():
        df = df.sort_values("date").reset_index(drop=True)
        for _, row in df.iterrows():
            d = row["date"]
            close_lk[(code, d)] = float(row["close"])
            open_lk[(code, d)] = float(row["open"])
        if "amount" in df.columns:
            # shift(1) so that ADV on day D uses amounts from D-20..D-1
            rolling_adv = df["amount"].rolling(_ADV_PERIOD, min_periods=5).mean().shift(1)
            for i, row in df.iterrows():
                v = rolling_adv.iloc[i]
                if not pd.isna(v) and v > 0:
                    adv_lk[(code, row["date"])] = float(v)
    return close_lk, open_lk, adv_lk


def _build_entry_lk_14_30(close_lk: Dict[tuple, float]) -> Dict[tuple, float]:
    """P11-4 Phase C: per (code, date) lookup for T 14:30 entry price.

    Strategy:
      - For dates in data/intraday_1m/*.parquet (2025-09 ~ 2026-04 by
        round-89 fetch scope): close of the 14:29:00 bar (last bar
        before 14:30, per PIT exclude-14:30 rule).
      - Otherwise: T close (close_lk fallback). This models "if we had
        intraday data we'd use 14:29 price; absent it, use the EOD close
        as best proxy for the price the model would see at decision time".

    Returns a dict that overlays close_lk where intraday available, else
    deferring to close_lk via .get fallback at call sites.

    NOTE Rule #11: production 14:30 entry would use the LIVE 14:30 price
    at decision time; this backtest approximates with the 14:29 bar's
    close (1m precision; live snapshot precision is ~1s). This is the
    same approximation P11-4 Step 1 fetch made.
    """
    out: Dict[tuple, float] = {}
    intra_dir = Path(INTRADAY_DIR)
    if not intra_dir.exists():
        logger.warning("ENTRY_TIME=14_30 but {} missing — full fallback to T close", INTRADAY_DIR)
        return out
    parts = sorted(intra_dir.glob("*.parquet"))
    if not parts:
        logger.warning("ENTRY_TIME=14_30 but {} empty — full fallback to T close", INTRADAY_DIR)
        return out
    frames = [pd.read_parquet(p) for p in parts]
    intra = pd.concat(frames, ignore_index=True)
    intra["datetime"] = pd.to_datetime(intra["datetime"])
    # The fetcher excluded 14:30 bar (PIT); the last bar each day is 14:29.
    # Take last bar per (code, date) — its close is the 14:29 close.
    intra["date"] = intra["datetime"].dt.normalize()
    intra = intra.sort_values(["code", "datetime"])
    last_bars = intra.groupby(["code", "date"]).tail(1)
    skipped_nan = 0
    for _, row in last_bars.iterrows():
        v = row["close"]
        if pd.isna(v):
            skipped_nan += 1
            continue
        out[(str(row["code"]), pd.Timestamp(row["date"]))] = float(v)
    logger.info("ENTRY_TIME=14_30: built entry_lk for {} (code, date) pairs from intraday_1m "
                "({} bars skipped due to NaN close)", len(out), skipped_nan)
    return out


# ── P11 Design 2 (round 117): Arm B real-14:30 factor injection ──────────
# INTRADAY_INJECT=1 overlays, onto panel_by_date for window dates only, the
# 64 factors recomputed with the REAL 14:30(T) morning bar injected as the
# latest bar — i.e. the historical replay of production's
# build_latest_features(intraday_bars=...). This is SCORING-only: training
# reads `panel` (untouched), so the model stays the standard non-injected
# EOD blend; at inference it sees 14:30-as-today's-close (Design 2's clean
# reframe). Rule #11: intraday_1m already ends 14:29 (no 14:30 leak).
INTRADAY_INJECT = os.environ.get("INTRADAY_INJECT", "0") == "1"
# Round 121 negative control: when > 0, inject the WRONG day's morning bar
# (shifted N positions in the sorted morning-date list, same code) instead of
# day T's own. If Arm B's alpha genuinely comes from T's 14:30 bar it MUST
# collapse to ~baseline here (garbage input → no signal); if it survives, the
# excess is structural leakage, not the injected signal. Decisive test.
INJECT_PLACEBO_SHIFT = int(os.environ.get("INJECT_PLACEBO_SHIFT", "0"))


def _load_morning_bars(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Dict[tuple, dict]:
    """Aggregate data/intraday_1m/*.parquet into a per-(code, date) morning
    bar (09:30–14:29). amount is synthesized as Σ(close_i × volume_i) since
    the 1m parquet has no amount column — that IS the morning turnover."""
    intra_dir = Path(INTRADAY_DIR)
    parts = sorted(intra_dir.glob("*.parquet"))
    if not parts:
        logger.warning("INTRADAY_INJECT=1 but {} empty — no Arm B injection", INTRADAY_DIR)
        return {}
    intra = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    intra["datetime"] = pd.to_datetime(intra["datetime"])
    intra["date"] = intra["datetime"].dt.normalize()
    intra = intra[(intra["date"] >= start_ts) & (intra["date"] <= end_ts)].copy()
    if intra.empty:
        return {}
    intra["code"] = intra["code"].astype(str).str.zfill(6)
    intra["_amt"] = intra["close"].astype(float) * intra["volume"].astype(float)
    intra = intra.sort_values(["code", "datetime"])
    agg = intra.groupby(["code", "date"]).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), amount=("_amt", "sum"),
    )
    morning: Dict[tuple, dict] = {}
    for (code, date), r in agg.iterrows():
        if pd.isna(r["close"]) or pd.isna(r["open"]):
            continue
        morning[(code, pd.Timestamp(date))] = {
            "open": float(r["open"]), "high": float(r["high"]), "low": float(r["low"]),
            "close": float(r["close"]), "volume": float(r["volume"]), "amount": float(r["amount"]),
        }
    logger.info("Arm B: loaded {} (code, date) morning bars from intraday_1m", len(morning))
    return morning


def _overlay_injected_factors(
    panel_by_date: Dict[pd.Timestamp, pd.DataFrame],
    bars_map: Dict[str, pd.DataFrame],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Return a NEW panel_by_date where, for dates in [window_start, window_end]
    that have intraday_1m data, each (code, T) row's 51 TECHNICAL_COLUMNS are
    recomputed on [EOD bars < T] + [real 14:30(T) morning bar]. The 13
    fundamental/trend/industry columns are kept from the baseline row
    (injection-invariant: publish-date aligned + cross-sectional). Dates
    outside the window, and codes without a morning bar, are passed through
    unchanged so the backtest degrades gracefully."""
    from mp.ml.dataset import _compute_technical_factors, TECHNICAL_COLUMNS

    morning = _load_morning_bars(window_start, window_end)
    if not morning:
        logger.warning("Arm B: no morning bars in window — returning panel unchanged")
        return panel_by_date

    # Negative-control remap (round 121): decision day dt → a WRONG morning
    # date (shifted in the sorted morning-date list). Same code, wrong day.
    morning_dates = sorted({d for (_c, d) in morning})
    wrong_day: Dict[pd.Timestamp, pd.Timestamp] = {}
    if INJECT_PLACEBO_SHIFT and morning_dates:
        n_md = len(morning_dates)
        wrong_day = {d: morning_dates[(i + INJECT_PLACEBO_SHIFT) % n_md]
                     for i, d in enumerate(morning_dates)}
        logger.warning("Arm B NEGATIVE CONTROL: INJECT_PLACEBO_SHIFT={} — injecting "
                       "WRONG-day morning bars; alpha MUST collapse to ~baseline if "
                       "the signal is real", INJECT_PLACEBO_SHIFT)
    _audit_done = False

    # Pre-index bars_map by code → sorted (date, ohlcv arrays) for fast slicing.
    out: Dict[pd.Timestamp, pd.DataFrame] = dict(panel_by_date)
    window_dates = [d for d in panel_by_date if window_start <= d <= window_end]
    n_rows_injected = 0
    n_codes_injected = 0
    proof_samples: List[dict] = []  # injection-proof: baseline vs injected factor values

    _sorted_window = sorted(window_dates)
    for _di, dt in enumerate(_sorted_window):
        if _di % 20 == 0:
            logger.info("Arm B inject progress: {}/{} window dates ({} rows so far)",
                        _di, len(_sorted_window), n_rows_injected)
        base = panel_by_date[dt]
        if base is None or base.empty:
            continue
        base = base.copy()
        base["code"] = base["code"].astype(str).str.zfill(6)
        tech_updates: Dict[int, Dict[str, float]] = {}  # row_idx → {tech_col: val}
        for idx, code in zip(base.index, base["code"]):
            src_dt = (wrong_day.get(pd.Timestamp(dt), pd.Timestamp(dt))
                      if INJECT_PLACEBO_SHIFT else pd.Timestamp(dt))
            mb = morning.get((code, src_dt))
            if mb is None:
                continue
            bdf = bars_map.get(code)
            if bdf is None or bdf.empty:
                continue
            hist = bdf[bdf["date"] < dt]
            if len(hist) < 80:   # _process_single_stock minimum
                continue
            # Cap to the last 300 bars before computing factors. Every
            # TECHNICAL_COLUMNS factor uses a rolling window <= 60, and MACD's
            # EMA26 fully converges in < 300 bars, so the LAST row's factor
            # values are identical (to ~1e-8) to the full-history computation
            # — but ~9x cheaper: hist otherwise grows to ~2700 bars by 2025,
            # which was the real Arm B bottleneck (~2h). Correctness-preserving.
            if len(hist) > 300:
                hist = hist.tail(300)
            # Build injected series arrays: [EOD bars < T] + [morning(T)].
            close = np.append(hist["close"].to_numpy(float), mb["close"])
            high = np.append(hist["high"].to_numpy(float), mb["high"])
            low = np.append(hist["low"].to_numpy(float), mb["low"])
            volume = np.append(hist["volume"].to_numpy(float), mb["volume"])
            amount = np.append(
                (hist["amount"].to_numpy(float) if "amount" in hist.columns
                 else hist["close"].to_numpy(float) * hist["volume"].to_numpy(float)),
                mb["amount"])
            open_arr = np.append(hist["open"].to_numpy(float), mb["open"])
            turnover = np.append(
                (hist["turnover"].to_numpy(float) if "turnover" in hist.columns
                 else np.full(len(hist), np.nan)), np.nan)
            try:
                fac = _compute_technical_factors(close, high, low, volume, amount, turnover, open_arr)
            except Exception as e:
                logger.debug("Arm B inject factor calc failed {} {}: {}", code, dt, e)
                continue
            row_vals = {c: float(fac[c][-1]) for c in TECHNICAL_COLUMNS if c in fac}
            tech_updates[idx] = row_vals
            n_rows_injected += 1
            if not _audit_done:
                # Round 121 ② panel audit: prove the injected series ends
                # [..., T-1 real, T synthetic-14:30] with NO T real EOD bar, the
                # synth close == 14:29 (≠ full-day close), and that ALL 51
                # technical cols are replaced (missing → kept-baseline = leak).
                real_t = bdf.loc[bdf["date"] == dt, "close"]
                missing = [c for c in TECHNICAL_COLUMNS if c not in fac]
                logger.info(
                    "Arm B PANEL AUDIT {} T={} (src={}): hist last2={} | T-real-bar "
                    "in hist? {} | synth(14:29) close={:.4f} vs real-T close={} | "
                    "tech cols updated {}/{}, missing(kept-baseline)={}",
                    code, str(pd.Timestamp(dt).date()), str(src_dt.date()),
                    [str(x.date()) for x in hist['date'].iloc[-2:]],
                    bool((hist['date'] == dt).any()), mb['close'],
                    (f"{float(real_t.iloc[0]):.4f}" if len(real_t) else "NA(T not in DB)"),
                    len(row_vals), len(TECHNICAL_COLUMNS), missing)
                _audit_done = True
            if len(proof_samples) < 6:
                proof_samples.append({
                    "code": code, "date": str(pd.Timestamp(dt).date()),
                    "rsi_14_base": float(base.at[idx, "rsi_14"]) if "rsi_14" in base.columns else None,
                    "rsi_14_inj": row_vals.get("rsi_14"),
                    "close_ma5_dev_base": float(base.at[idx, "close_ma5_dev"]) if "close_ma5_dev" in base.columns else None,
                    "close_ma5_dev_inj": row_vals.get("close_ma5_dev"),
                })
        if tech_updates:
            for col in TECHNICAL_COLUMNS:
                if col not in base.columns:
                    continue
                col_map = {idx: vals[col] for idx, vals in tech_updates.items() if col in vals}
                if col_map:
                    base.loc[list(col_map.keys()), col] = list(col_map.values())
            out[pd.Timestamp(dt)] = base
            n_codes_injected = max(n_codes_injected, len(tech_updates))

    logger.info("Arm B: injected 14:30 factors into {} (code,date) rows across {} window dates",
                n_rows_injected, len(window_dates))
    for s in proof_samples:
        logger.info("  Arm B inject-proof {} {}: rsi_14 {:.3f}→{:.3f} | close_ma5_dev {:.5f}→{:.5f}",
                    s["code"], s["date"], s["rsi_14_base"] or float("nan"), s["rsi_14_inj"] or float("nan"),
                    s["close_ma5_dev_base"] or float("nan"), s["close_ma5_dev_inj"] or float("nan"))
    return out


def run_walk_forward():
    """Main walk-forward backtest."""
    t0 = time.time()

    # 1. Universe
    logger.info("=" * 60)
    logger.info("Walk-Forward Backtest: {} -> {}", BT_START, BT_END)
    logger.info("Capital: {:,.0f} | Universe: {} | Top-K: {}", INITIAL_CAPITAL, UNIVERSE, TOP_K)
    logger.info("=" * 60)

    # Universe: union of all point-in-time constituents across every stored
    # snapshot (so bars / factors are fetched for stocks that were ever in
    # the index, not just today's members).  Fixes the half-baked
    # survivorship mitigation that fetched only current members.
    try:
        from mp.data.store import DataStore as _DS
        _store = _DS()
        _union, _snap_dates = _merged_all_snapshots(_store)
        if len(_snap_dates) >= 2 and _union:
            codes = _union
            logger.info("Universe: {} unique stocks across {} historical {} snapshots "
                        "({} → {})", len(codes), len(_snap_dates), UNIVERSE,
                        _snap_dates[0], _snap_dates[-1])
        else:
            codes = _merged_current()
            logger.warning("Only {} snapshot(s) for {}; falling back to current "
                           "constituents ({} stocks) — SURVIVORSHIP BIAS PRESENT.  "
                           "Run scripts/backfill_constituents.py to fix.",
                           len(_snap_dates), UNIVERSE, len(codes))
    except Exception as _e:
        codes = _merged_current()
        logger.warning("Historical snapshot lookup failed ({}); using current "
                       "{} constituents", _e, UNIVERSE)

    # 2. Data
    bars_map = _load_or_fetch_bars(codes)
    panel = _load_or_build_factors(bars_map, codes)

    # BlendRanker primary trains on excess_ret; if the cached panel lacks it,
    # add it now (one akshare fetch + simple subtraction).  StockRanker uses
    # fwd_ret directly and doesn't need this, but adding the column is cheap
    # and keeps the panel ready for either ranker.
    if "excess_ret" not in panel.columns and "fwd_ret" in panel.columns:
        from mp.ml.dataset import add_excess_ret
        logger.info("Computing excess_ret column for BlendRanker compatibility...")
        panel = add_excess_ret(panel, horizon=HORIZON)

    # Optional: augment panel with per-date regime features (PIT-safe,
    # computed from ZZ500 + cross-sectional breadth).  LGBM can learn
    # regime↔factor interactions via tree splits.
    # 2026-05-23: switched from FACTOR_COLUMNS (47) to CURATED_COLUMNS (28)
    # — IC analysis showed the dropped 19 features had |IR| < 0.15 (noise);
    # the kept set adds 3 strong industry-rank signals that were previously
    # masked by data issues (now fixed).
    # 2026-05-24: env override for W0/W1/W2 walk-forward comparison after
    # P0 ICIR fix. See mp/ml/feature_presets.py for frozen preset lists.
    _wf_preset = os.environ.get("WF_FEATURE_PRESET")
    if _wf_preset:
        from mp.ml.feature_presets import PRESETS, preset_signature
        if _wf_preset not in PRESETS:
            raise SystemExit(
                f"WF_FEATURE_PRESET={_wf_preset!r} invalid; "
                f"choose one of {sorted(PRESETS)}"
            )
        feature_cols = list(PRESETS[_wf_preset])
        logger.info(
            "WF_FEATURE_PRESET={} ACTIVE: {} features, sig={}",
            _wf_preset, len(feature_cols), preset_signature(_wf_preset),
        )
    else:
        feature_cols = list(CURATED_COLUMNS)
    if USE_REGIME_FEATURES:
        logger.info("Regime features: ENABLED ({} cols added)", len(REGIME_COLUMNS))
        panel = add_regime_features(panel)
        feature_cols = feature_cols + [c for c in REGIME_COLUMNS if c in panel.columns]
    else:
        logger.info("Regime features: disabled (set USE_REGIME_FEATURES=1 to enable)")

    # ── P11-3 (round 79): RANKER_KIND=intraday_blend augments panel ────────
    # with 4 EOD-proxy intraday extras + switches feature_cols to
    # INTRADAY_FEATURE_COLS (68). Same EOD-proxy formula as
    # scripts/train_intraday.py — keeps training and walk-forward in the
    # same distribution. Execution timing UNCHANGED (T+1 open simulator
    # path), per Option C — only the model variable differs vs RANKER_KIND=blend.
    if RANKER_KIND == "intraday_blend":
        from mp.ml.intraday_features import INTRADAY_FEATURE_COLS, INTRADAY_EXTRA_COLUMNS
        from scripts.train_intraday import (
            compute_extras_for_panel,
            compute_extras_for_panel_hybrid,
            load_intraday_1m,
            _real_morning_extras_per_code,
        )

        # P11-4 Phase C: INTRADAY_HYBRID=1 overlays real intraday extras where available.
        real_extras_by_code: Dict[str, pd.DataFrame] = {}
        if INTRADAY_HYBRID:
            intra_long = load_intraday_1m()
            if not intra_long.empty:
                for code_xt, code_df in intra_long.groupby("code"):
                    code = str(code_xt)
                    bars = bars_map.get(code)
                    if bars is None or bars.empty:
                        continue
                    eod = bars.sort_values("date").reset_index(drop=True)
                    eod["date"] = pd.to_datetime(eod["date"]).dt.normalize()
                    eod_vol_ma20 = eod.set_index("date")["volume"].rolling(window=20, min_periods=20).mean().shift(1)
                    real_extras_by_code[code] = _real_morning_extras_per_code(code_df.assign(code=code), eod_vol_ma20)
            logger.info("INTRADAY_HYBRID=1 — real intraday extras prepared for {} codes",
                        len(real_extras_by_code))
            logger.info("RANKER_KIND=intraday_blend — augmenting panel (hybrid: real where available, else EOD-proxy)")
        else:
            logger.info("RANKER_KIND=intraday_blend — augmenting panel with EOD-proxy extras")

        extras_frames: List[pd.DataFrame] = []
        for code, bars in bars_map.items():
            if INTRADAY_HYBRID and code in real_extras_by_code:
                extras = compute_extras_for_panel_hybrid(bars, real_extras_by_code[code])
            else:
                extras = compute_extras_for_panel(bars)
            extras["code"] = code
            extras_frames.append(extras[["code", "date"] + list(INTRADAY_EXTRA_COLUMNS)])
        extras_all = pd.concat(extras_frames, ignore_index=True)
        extras_all["date"] = pd.to_datetime(extras_all["date"])
        panel = panel.merge(extras_all, on=["code", "date"], how="left")
        # Sanity: every panel row should now have all 4 extras (some NaN
        # at warm-up boundaries is acceptable; LightGBM handles them).
        for col in INTRADAY_EXTRA_COLUMNS:
            n_nan = int(panel[col].isna().sum())
            logger.info("  {}: {:.1f}% non-null ({} NaN of {})",
                        col, 100.0 * (1 - n_nan / max(len(panel), 1)),
                        n_nan, len(panel))
        feature_cols = list(INTRADAY_FEATURE_COLS)
        logger.info("intraday_blend feature_cols: {} cols (FACTOR_COLUMNS + 4 extras)",
                    len(feature_cols))

    close_lk, open_lk, adv_lk = _build_price_adv_lookup(bars_map)

    # P11-4 Phase C (round 91): ENTRY_TIME=14_30 swaps the execution price
    # source. Build entry_lk once: T 14:30 close (from intraday_1m for
    # 2025-09+, T close fallback otherwise). When ENTRY_TIME=t_plus_1_open
    # (default), entry_lk is empty and code paths read open_lk as before.
    if ENTRY_TIME == "14_30":
        entry_lk_14_30 = _build_entry_lk_14_30(close_lk)
        # Build the effective entry lookup: 14:29 close where available,
        # else T close (NOT T+1 open — this is the semantic shift for 14:30
        # entry: decision & execution same day at 14:30 = ~ T close).
        # Bind to a name so later code can use the same `entry_lk` variable.
        def _entry_price(code, dt):
            return entry_lk_14_30.get((code, dt), close_lk.get((code, dt)))
    else:
        def _entry_price(code, dt):
            return open_lk.get((code, dt))
    logger.info("ENTRY_TIME={} — entry price source: {}",
                ENTRY_TIME,
                "{} 14:29 close + T close fallback".format(INTRADAY_DIR)
                if ENTRY_TIME == "14_30" else "T+1 open")
    if ENTRY_TIME == "14_30":
        logger.info("14_30 execution timing: {} (round 147 fix ②)",
                    "SAME-DAY D 14:30 (decision ≤14:29 → fill D 14:29-close)"
                    if SAME_DAY_14_30 else "legacy NEXT-DAY D+1 14:30 (INTRADAY_NEXT_DAY=1)")

    # All trading dates in the BT window
    bt_start_ts = pd.Timestamp(BT_START)
    bt_end_ts = pd.Timestamp(BT_END)
    all_dates_set: set[pd.Timestamp] = set()
    for df in bars_map.values():
        mask = (df["date"] >= bt_start_ts) & (df["date"] < bt_end_ts)
        all_dates_set.update(df.loc[mask, "date"].tolist())
    trading_dates = sorted(all_dates_set)

    retrain_dates = _get_monthly_retrain_dates(panel)
    retrain_set = set(retrain_dates)
    logger.info("{} trading days, {} retrain months", len(trading_dates), len(retrain_dates))

    # Pre-group panel by date for fast daily feature lookup
    core = TECHNICAL_COLUMNS[:13]
    panel_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    for dt_key, grp in panel.groupby("date"):
        panel_by_date[pd.Timestamp(dt_key)] = grp

    # Arm B (Design 2): overlay real-14:30-injected factors onto the SCORING
    # panel for window dates with intraday_1m data. Training (`panel`) is
    # untouched. The window is the full BT span; injection only lands where
    # morning bars exist (2025-09+), so older dates pass through unchanged.
    if INTRADAY_INJECT:
        logger.info("INTRADAY_INJECT=1 (Arm B / Design 2) — overlaying real-14:30 "
                    "injected factors onto scoring panel (training untouched)")
        _all_pbd_dates = sorted(panel_by_date.keys())
        if _all_pbd_dates:
            # Round 127 sweep: the injected panel is SEED- and COST-independent
            # (factors only; LGBM_SEED/cost act downstream). INJECT_CACHE lets the
            # cost sweep reuse one overlay instead of re-running the ~33min build
            # per (seed,cost). Cache only the traded window [BT_START,BT_END] rows
            # (scoring uses panel_by_date only there; retrain uses `panel`).
            _cache = os.environ.get("INJECT_CACHE", "")
            if _cache and Path(_cache).exists():
                cdf = pd.read_parquet(_cache)
                cdf["date"] = pd.to_datetime(cdf["date"])
                for dkey, grp in cdf.groupby("date"):
                    panel_by_date[pd.Timestamp(dkey)] = grp.reset_index(drop=True)
                logger.info("INJECT_CACHE hit: loaded {} overlaid rows / {} dates from {} "
                            "— skipped ~33min overlay", len(cdf), cdf["date"].nunique(), _cache)
            else:
                panel_by_date = _overlay_injected_factors(
                    panel_by_date, bars_map, _all_pbd_dates[0], _all_pbd_dates[-1])
                if _cache:
                    win = [d for d in panel_by_date if bt_start_ts <= d <= bt_end_ts]
                    cat = pd.concat([panel_by_date[d] for d in win], ignore_index=True)
                    cat.to_parquet(_cache)
                    logger.info("INJECT_CACHE saved: {} rows / {} dates → {}",
                                len(cat), len(win), _cache)

    # Trading calendar from panel for label_cutoff (Fix #3: avoid calendar-day approximation)
    _all_trade_dates = sorted(panel["date"].unique())

    # ── Survivorship-bias mitigation: time-varying scoring universe ──────────
    # Build a mapping from each retrain month to the index constituents that
    # were known at that time (from accumulated constituent_snapshots in the DB).
    # On the first ever run only today's snapshot exists, so all historical
    # months fall back to the current list.  Over quarterly runs the DB
    # accumulates snapshots and the mapping becomes increasingly accurate.
    _current_codes_set = frozenset(codes)
    _month_universe: Dict[pd.Period, frozenset] = {}
    _snapshot_dates = []
    try:
        from mp.data.store import DataStore as _DS
        _, _snapshot_dates = _merged_all_snapshots(_DS())
    except Exception:
        pass
    n_snapshots = len(_snapshot_dates)
    if n_snapshots > 1:
        logger.info("Found {} historical {} snapshots — using time-varying universe",
                    n_snapshots, UNIVERSE)
    else:
        logger.warning(
            "Only {} constituent snapshot(s) available for {} — "
            "scoring universe uses current members throughout.  "
            "Survivorship bias is present: stocks that left the index "
            "before today are excluded from all historical periods.  "
            "Re-run quarterly to accumulate snapshots.",
            n_snapshots, UNIVERSE,
        )
    for rd in retrain_dates:
        rd_str = rd.strftime("%Y-%m-%d")
        snap = _merged_pit_at(rd_str)
        _month_universe[rd.to_period("M")] = frozenset(snap) if snap else _current_codes_set

    # ── PIT training membership lookup ────────────────────────────────────
    # We also need the PIT universe at every *training* row, not just at
    # scoring/retrain dates.  Pre-compute a {(code, month_period): True}
    # membership set so filtering the training panel is O(1) per row.
    _membership: dict[tuple[str, pd.Period], bool] = {}
    if n_snapshots >= 2:
        # Build month-indexed snapshots across the full training span
        _panel_months = pd.period_range(
            start=pd.Timestamp(TRAIN_START).to_period("M"),
            end=pd.Timestamp.today().to_period("M"),
            freq="M",
        )
        for _pm in _panel_months:
            _asof = (_pm.to_timestamp(how="end")).strftime("%Y-%m-%d")
            _snap = _merged_pit_at(_asof)
            if _snap:
                for _c in _snap:
                    _membership[(_c, _pm)] = True
        logger.info("PIT training membership: {} (code, month) pairs across {} months",
                    len(_membership), len(_panel_months))
    else:
        logger.warning("Fewer than 2 snapshots — cannot filter training rows PIT; "
                       "training will use all ever-members (selection bias).")
    # ────────────────────────────────────────────────────────────────────────

    # 3. Simulation state
    # Round 127 cost sweep: cost model is env-driven so Arm B can be re-costed
    # cheaply (with INJECT_CACHE) without rebuilding the injected panel.
    #   USE_SQRT_IMPACT=0 → flat per-trade slippage = SLIPPAGE_BPS (clean "bps"
    #     sweep); =1 (default) keeps the realistic sqrt-impact (Phase-1 config).
    #   IMPACT_ALPHA_BPS overrides the sqrt α (default 150).
    #   STAMP_TAX_BPS overrides sell-side stamp (default FeeSchedule new=5);
    #     set 0 (with slippage/commission 0) for a true zero-cost gross run.
    _use_sqrt = os.environ.get("USE_SQRT_IMPACT", "1") == "1"
    _impact_alpha = float(os.environ.get("IMPACT_ALPHA_BPS", "150.0"))
    _fee_kw = dict(slippage_bps=SLIPPAGE_BPS, commission_bps=COMMISSION_BPS,
                   use_sqrt_impact=_use_sqrt, impact_alpha_bps=_impact_alpha,
                   min_slippage_bps=3.0)
    _stamp = os.environ.get("STAMP_TAX_BPS")
    if _stamp is not None:
        _fee_kw["stamp_tax_bps_old"] = float(_stamp)
        _fee_kw["stamp_tax_bps_new"] = float(_stamp)
    fees = FeeSchedule(**_fee_kw)
    logger.info("FeeSchedule: slippage={} commission={} sqrt_impact={} alpha={} stamp={}",
                SLIPPAGE_BPS, COMMISSION_BPS, _use_sqrt, _impact_alpha,
                _stamp if _stamp is not None else "default(5)")
    logger.info("Slippage model: sqrt-impact α={} bps, floor={} bps (fallback linear={} bps)",
                fees.impact_alpha_bps, fees.min_slippage_bps, fees.slippage_bps)
    broker = SimulatedBroker(INITIAL_CAPITAL, fees, silent=True)
    nav_records: List[dict] = []
    monthly_returns: List[dict] = []
    prev_total_value = float(INITIAL_CAPITAL)
    month_start_nav = 1.0
    current_month = None
    current_ranker: Optional[StockRanker] = None
    pending_selection: Optional[List[tuple]] = None  # signal from previous close
    pending_raw_scores: Dict[str, float] = {}        # raw excess pred by code, populated in Step B
    current_universe: frozenset = _current_codes_set  # updated at each retrain

    # Tail-quality records — for each scoring day, capture:
    #   predicted top-K codes, actual fwd_ret of all valid codes
    # Used after backtest to compute Hit Rate@K and NDCG@K, which measure
    # how well the model identifies the extreme tail (top performers).
    # IC measures mean predictive power; these measure top-K precision —
    # what actually matters for a Top-K strategy.
    tail_quality_records: List[dict] = []

    # Round 147 (fix ②): extract Step B's scoring so SAME_DAY_14_30 can run it
    # BEFORE Step A (same-day fill), while the default path keeps running it
    # after (next-day fill). Returns (selection, raw_scores) or (None, {})
    # when the ranker isn't ready / today's panel is empty — callers leave
    # the prior `pending_selection` untouched in that case.
    def _score_today(dt):
        ranker_ready = current_ranker is not None and (
            getattr(current_ranker, "model", None) is not None
            or (
                hasattr(current_ranker, "primary")
                and getattr(current_ranker.primary, "model", None) is not None
            )
        )
        if not ranker_ready:
            return None, {}
        today_df = panel_by_date.get(dt)
        if today_df is None:
            return None, {}
        today_df = today_df[today_df["code"].isin(current_universe)]
        today_valid = today_df.dropna(subset=core)
        if today_valid.empty:
            return None, {}
        codes_in = today_valid["code"].tolist()
        scores = current_ranker.predict(today_valid)
        if hasattr(current_ranker, "predict_raw"):
            raws = current_ranker.predict_raw(today_valid)
        else:
            raws = scores
        raw_score_map = dict(zip(codes_in, raws))
        scored = sorted(zip(codes_in, scores), key=lambda x: -x[1])
        if COST_AWARE_REBALANCE:
            from mp.backtest.ml_backtest import _cost_aware_select
            selection = _cost_aware_select(scored, TOP_K, broker, dt, adv_lk)
        else:
            selection = scored[:TOP_K]
        try:
            pred_top_k = [c for c, _ in scored[:TOP_K]]
            fwd_ret_map = dict(zip(today_valid["code"], today_valid["fwd_ret"]))
            valid_fwd = {c: r for c, r in fwd_ret_map.items() if pd.notna(r)}
            if len(valid_fwd) >= TOP_K * 5:
                tail_quality_records.append({
                    "date": dt,
                    "pred_top_k": pred_top_k,
                    "fwd_ret_map": valid_fwd,
                })
        except Exception as e:
            logger.debug("tail_quality record skip {}: {}", dt, e)
        if POSITION_SIZING == "conviction_oracle":
            oracle_map = {}
            for _, row in today_valid.iterrows():
                fr = row.get("fwd_ret")
                oracle_map[row["code"]] = float(fr) if pd.notna(fr) else 0.0
            raw_scores = oracle_map
        else:
            raw_scores = raw_score_map
        return selection, raw_scores

    # 4. Main loop — retrain monthly, score & rebalance daily
    for step, dt in enumerate(trading_dates):
        # Track month transitions for monthly return logging
        dt_period = dt.to_period("M")
        if current_month is not None and dt_period != current_month:
            current_nav = nav_records[-1]["nav"] if nav_records else 1.0
            monthly_returns.append({
                "month": str(current_month),
                "return": current_nav / month_start_nav - 1,
                "nav": current_nav,
            })
            month_start_nav = current_nav
        current_month = dt_period

        # --- Monthly: retrain model on first trading day of each month ---
        if dt in retrain_set:
            logger.info("── Retrain: {} ──", dt.strftime("%Y-%m-%d"))
            # Update scoring universe for this month.
            current_universe = _month_universe.get(dt_period, _current_codes_set)

            # ── Label-cutoff logic (prevents fwd_ret leakage into test period) ──
            # fwd_ret[t] = close[t + HORIZON] / close[t] - 1.
            # A row at date D is safe to train on iff D + HORIZON < dt
            # (i.e. the label uses only prices that pre-date the test period start).
            # label_cutoff is the trading date at position (idx_of_dt - HORIZON),
            # so every row with date < label_cutoff satisfies D + HORIZON < dt.
            # The `< label_cutoff` strict inequality ensures the cutoff date
            # itself (whose fwd_ret uses close[dt]) is also excluded.
            _dt_idx = bisect.bisect_right(_all_trade_dates, dt) - 1
            _cutoff_idx = max(0, _dt_idx - HORIZON)
            label_cutoff = _all_trade_dates[_cutoff_idx]
            train_mask = (panel["date"] < label_cutoff) & panel["fwd_ret"].notna()
            train_df = panel.loc[train_mask].copy()

            # PIT training universe: for each training row, require that the
            # stock was actually in the index at that row's month.  Removes the
            # "ever-in-ZZ500" selection bias (training on stocks that future
            # rebalances would promote into the index = mild look-ahead).
            if _membership:
                _periods = train_df["date"].dt.to_period("M")
                _keep = [
                    (c, p) in _membership
                    for c, p in zip(train_df["code"].to_numpy(), _periods.to_numpy())
                ]
                n_before = len(train_df)
                train_df = train_df.loc[_keep].copy()
                logger.debug("PIT training filter: {} → {} rows ({:.0f}% kept)",
                             n_before, len(train_df),
                             100.0 * len(train_df) / max(n_before, 1))

            if len(train_df) < 500:
                logger.warning("Too few training rows ({}), skipping retrain", len(train_df))
            else:
                try:
                    if RANKER_KIND in ("blend", "intraday_blend"):
                        from mp.ml.model import BlendRanker
                        ranker = BlendRanker(feature_cols=feature_cols)
                        metrics = ranker.train_fast(train_df)
                        # BlendRanker returns nested {"primary": {...}, "extreme": {...}}
                        m_p = metrics.get("primary", {})
                        m_e = metrics.get("extreme", {})
                        logger.info("  Train: {} rows | primary: MAE={:.4f} IC={:.3f} rounds={} | extreme: IC={:.3f} rounds={}",
                                    len(train_df),
                                    m_p.get("mae", float("nan")), m_p.get("ic", float("nan")),
                                    m_p.get("best_rounds", -1),
                                    m_e.get("ic", float("nan")), m_e.get("best_rounds", -1))
                    else:
                        ranker = StockRanker(feature_cols=feature_cols)
                        metrics = ranker.train_fast(train_df)
                        logger.info("  Train: {} rows, MAE={:.4f}, IC={:.3f}, HitRate@10={:.2f}, Precision@10={:.2f}, rounds={}",
                                    len(train_df), metrics["mae"], metrics["ic"],
                                    metrics.get("hit_rate_at_k", float("nan")),
                                    metrics.get("precision_at_k", float("nan")),
                                    metrics["best_rounds"])
                    current_ranker = ranker
                except Exception as e:
                    logger.error("  Training failed: {}", e)

        # Round 147 (fix ② same-day timing): for SAME_DAY_14_30 the decision
        # uses today's ≤14:29 injected factors and fills at TODAY's 14:29
        # close — so we score BEFORE Step A and feed today's selection into
        # the same pending mechanism, which Step A then executes at dt.
        # Other modes (baseline / ③ next-day open / 14_30 legacy next-day
        # with INTRADAY_NEXT_DAY=1) keep scoring at Step B → pending for
        # tomorrow (unchanged).
        if SAME_DAY_14_30:
            _sel, _raws = _score_today(dt)
            if _sel is not None:
                pending_selection = _sel
                pending_raw_scores = _raws

        # --- Step A: Execute pending signal from previous close at today's open ---
        if pending_selection is not None:
            sel_codes = {c for c, _ in pending_selection}
            held_codes = set(broker.positions.keys())
            selection_changed = sel_codes != held_codes

            # Drift-triggered rebalance: even when names are unchanged, if any
            # held position's weight has drifted beyond threshold, re-equalize.
            drift_triggered = False
            if (not selection_changed and REBALANCE_POLICY == "drift_threshold"
                    and held_codes):
                # Compute current weights using today's open (execution price).
                pos_values = {}
                for c in held_codes:
                    p = _entry_price(c, dt) or broker.positions[c].current_price
                    pos_values[c] = broker.positions[c].shares * p
                total_equity = broker.cash + sum(pos_values.values())
                target_w = 1.0 / TOP_K
                max_drift = max(
                    abs(v / total_equity - target_w) for v in pos_values.values()
                ) if total_equity > 0 else 0
                drift_triggered = max_drift > MAX_WEIGHT_DRIFT

            if selection_changed or drift_triggered:
                # Sizing uses previous close (already in broker from yesterday's NAV step).
                # Using today's close here would be look-ahead bias.

                dt_str = dt.strftime("%Y-%m-%d")

                # (a) For drift rebalance (names unchanged): trim overweights
                # first so broker.buy() has cash to top up underweights.
                if drift_triggered and not selection_changed:
                    total_value_now = broker.total_value
                    target_per_stock = total_value_now / TOP_K
                    for code in list(broker.positions.keys()):
                        price_raw = _entry_price(code, dt)
                        if price_raw is None or pd.isna(price_raw) or price_raw <= 0:
                            continue
                        current_value = broker.positions[code].shares * price_raw
                        excess = current_value - target_per_stock
                        if excess > target_per_stock * 0.01:  # >1pp over target
                            sell_shares = int(excess / price_raw)
                            if sell_shares > 0:
                                broker.sell(code, price_raw, shares=sell_shares,
                                            date=dt_str, action="SELL (drift)",
                                            adv=adv_lk.get((code, dt)))

                # (b) Sell positions not in new selection at today's open
                for code in list(broker.positions.keys()):
                    if code not in sel_codes:
                        sell_price = _entry_price(code, dt) or broker.positions[code].current_price
                        broker.sell(code, sell_price, date=dt_str,
                                    adv=adv_lk.get((code, dt)))

                # (c) Buy new / adjust positions at today's open.
                #     Position weights per POSITION_SIZING setting.
                total_value_now = broker.total_value
                weights = _compute_position_weights(
                    [c for c, _ in pending_selection], dt, panel_by_date,
                    POSITION_SIZING,
                    raw_scores=pending_raw_scores,
                )
                for code, score in pending_selection:
                    price_raw = _entry_price(code, dt)
                    if price_raw is None or pd.isna(price_raw) or price_raw <= 0:
                        continue
                    target_value = total_value_now * weights.get(code, 1.0 / TOP_K)
                    broker.buy(code, price_raw, target_value=target_value,
                               date=dt_str,
                               action="BUY (add)" if code in broker.positions else "BUY",
                               adv=adv_lk.get((code, dt)))

                reason = "drift" if drift_triggered and not selection_changed else "sel"
                logger.info("  Day {}: rebalanced ({}) → {} stocks, cash: {:,.0f}",
                            dt.strftime("%Y-%m-%d"), reason, len(broker.positions), broker.cash)
            pending_selection = None  # consumed

        # --- Step B: Score stocks at today's close, store as pending for tomorrow ---
        # Round 147: SAME_DAY_14_30 already scored & consumed above (fix ②);
        # the default path (baseline / ③ / 14_30 legacy next-day) scores here
        # and stores pending for tomorrow's open execution in Step A.
        if not SAME_DAY_14_30:
            _sel, _raws = _score_today(dt)
            if _sel is not None:
                pending_selection = _sel
                pending_raw_scores = _raws

        # --- Daily NAV ---
        close_snap = {}
        for code in broker.positions:
            price = close_lk.get((code, dt), broker.positions[code].current_price)
            if price > 0:
                close_snap[code] = price
        broker.update_prices(close_snap)
        total_value = broker.total_value

        daily_ret = (total_value / prev_total_value - 1) if prev_total_value > 0 else 0.0
        nav = (nav_records[-1]["nav"] * (1 + daily_ret)) if nav_records else (1 + daily_ret)
        nav_records.append({"date": dt, "daily_return": daily_ret, "nav": nav})
        prev_total_value = total_value

        if (step + 1) % 100 == 0:
            logger.info("  Day {}/{}: NAV={:.4f}, value={:,.0f}",
                        step + 1, len(trading_dates), nav, total_value)

    # Log last month
    if nav_records and current_month is not None:
        monthly_returns.append({
            "month": str(current_month),
            "return": nav_records[-1]["nav"] / month_start_nav - 1,
            "nav": nav_records[-1]["nav"],
        })

    # 5. Performance metrics
    nav_df = pd.DataFrame(nav_records)
    # Round 135: NAV_DUMP writes the daily (date, daily_return, nav) series for
    # regime bucketing (Arm B vs A daily excess by ZZ500 up/flat/down day).
    _nav_dump = os.environ.get("NAV_DUMP")
    if _nav_dump and not nav_df.empty:
        nav_df.to_csv(_nav_dump, index=False)
        logger.info("NAV_DUMP: wrote {} daily rows → {}", len(nav_df), _nav_dump)
    metrics = calc_performance(nav_df)

    # 5b. Tail-quality metrics — Hit Rate@K and NDCG@K averaged over
    # all scoring days.  Measures how good the model is at the TAIL
    # (top-K picks), which is what actually drives a Top-K strategy.
    tail_q = _compute_tail_quality_metrics(tail_quality_records, k=TOP_K)
    metrics["hit_rate_at_k"] = tail_q["hit_rate"]
    metrics["ndcg_at_k"] = tail_q["ndcg"]
    metrics["actual_topk_alpha"] = tail_q["actual_topk_fwd_mean"]
    metrics["selected_topk_alpha"] = tail_q["selected_topk_fwd_mean"]
    metrics["topk_alpha_capture_rate"] = tail_q["alpha_capture_rate"]
    metrics["tail_q_n_days"] = tail_q["n_days"]
    logger.info("Tail quality ({} days @ K={}):  Hit Rate={:.1%}  NDCG={:.3f}  "
                "Selected alpha={:.3%}  Realistic top-K alpha={:.3%}  Capture={:.1%}",
                tail_q["n_days"], TOP_K, tail_q["hit_rate"], tail_q["ndcg"],
                tail_q["selected_topk_fwd_mean"], tail_q["actual_topk_fwd_mean"],
                tail_q["alpha_capture_rate"])

    # Fetch ZZ500 benchmark
    benchmark_ret = _calc_benchmark_return(trading_dates, close_lk, bars_map)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Walk-Forward Backtest Complete ({:.1f} min)", elapsed / 60)
    logger.info("=" * 60)

    # 6. Print results
    _print_results(metrics, monthly_returns, benchmark_ret, broker.trade_log, elapsed)

    # 7. Save report
    _save_report(metrics, monthly_returns, benchmark_ret, broker.trade_log, elapsed)

    return metrics, monthly_returns, current_ranker, benchmark_ret


def _compute_tail_quality_metrics(records: List[dict], k: int = 10) -> Dict[str, float]:
    """Compute Hit Rate@K, NDCG@K, and alpha capture rate over the daily
    prediction records.

    For each scoring day:
      - actual top-K = the K codes with highest realized fwd_ret that day
      - selected top-K = the K codes the model picked
      - hit_rate@K = |selected ∩ actual| / K
      - NDCG@K = sum(realized_fwd_ret[selected[i]] / log2(i+2)) /
                 sum(realized_fwd_ret[actual[i]]    / log2(i+2))
      - alpha capture rate = mean(selected fwd_ret) / mean(actual top-K fwd_ret)
    """
    if not records:
        return {"hit_rate": float("nan"), "ndcg": float("nan"),
                "actual_topk_fwd_mean": float("nan"),
                "selected_topk_fwd_mean": float("nan"),
                "alpha_capture_rate": float("nan"), "n_days": 0}

    hits, ndcgs, sel_means, act_means = [], [], [], []

    for rec in records:
        pred = rec["pred_top_k"]
        fwd_map = rec["fwd_ret_map"]
        if not pred or not fwd_map:
            continue
        # Filter selected to those with valid fwd_ret
        sel_valid = [c for c in pred if c in fwd_map]
        if not sel_valid:
            continue

        # Actual top-K by realized fwd_ret
        sorted_actual = sorted(fwd_map.items(), key=lambda x: -x[1])[:k]
        actual_codes = [c for c, _ in sorted_actual]
        actual_rets = [r for _, r in sorted_actual]

        # Hit rate
        hit = len(set(sel_valid) & set(actual_codes)) / len(sel_valid)
        hits.append(hit)

        # NDCG@K (normalized by ideal DCG of actual top-K)
        sel_rets = [fwd_map[c] for c in sel_valid][:k]
        dcg = sum(r / np.log2(i + 2) for i, r in enumerate(sel_rets))
        ideal_dcg = sum(r / np.log2(i + 2) for i, r in enumerate(actual_rets))
        if ideal_dcg != 0:
            ndcgs.append(dcg / ideal_dcg)

        sel_means.append(float(np.mean(sel_rets)))
        act_means.append(float(np.mean(actual_rets)))

    return {
        "hit_rate": float(np.mean(hits)) if hits else float("nan"),
        "ndcg": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "actual_topk_fwd_mean": float(np.mean(act_means)) if act_means else float("nan"),
        "selected_topk_fwd_mean": float(np.mean(sel_means)) if sel_means else float("nan"),
        "alpha_capture_rate": (float(np.mean(sel_means)) / float(np.mean(act_means))
                                if act_means and np.mean(act_means) != 0 else float("nan")),
        "n_days": len(hits),
    }


def _calc_benchmark_return(
    trading_dates: List[pd.Timestamp],
    close_lk: Dict,
    bars_map: Dict[str, pd.DataFrame],
) -> Optional[float]:
    """Approximate ZZ500 index return over the backtest period.

    Uses akshare to fetch index data directly.
    """
    try:
        import akshare as ak
        idx_df = ak.stock_zh_index_daily(symbol="sh000905")
        idx_df["date"] = pd.to_datetime(idx_df["date"])
        start_dt = trading_dates[0]
        end_dt = trading_dates[-1]
        mask = (idx_df["date"] >= start_dt) & (idx_df["date"] <= end_dt)
        idx_sub = idx_df.loc[mask].sort_values("date")
        if len(idx_sub) >= 2:
            ret = float(idx_sub["close"].iloc[-1] / idx_sub["close"].iloc[0] - 1)
            n_days = len(idx_sub)
            annual_ret = (1 + ret) ** (252 / max(n_days, 1)) - 1
            logger.info("ZZ500 benchmark: total={:.2%}, annual={:.2%}", ret, annual_ret)
            return {"total_return": ret, "annual_return": annual_ret, "days": n_days}
    except Exception as e:
        logger.warning("Failed to fetch ZZ500 benchmark: {}", e)
    return None


def _print_results(metrics, monthly_returns, benchmark_ret, trade_log, elapsed):
    """Print results to console."""
    print("\n" + "=" * 60)
    print("  WALK-FORWARD BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Period:          {BT_START[:4]}-{BT_START[4:6]} ~ {BT_END[:4]}-{BT_END[4:6]}")
    print(f"  Initial Capital: {INITIAL_CAPITAL:,.0f}")
    print(f"  Universe:        {UNIVERSE} | Top-K: {TOP_K}")
    print(f"  Runtime:         {elapsed/60:.1f} min")
    print("-" * 60)
    for k, v in metrics.items():
        if k != "trading_days":
            print(f"  {k:20s}: {v}")
    print(f"  {'trading_days':20s}: {metrics.get('trading_days', 'N/A')}")

    if benchmark_ret:
        print("-" * 60)
        print(f"  ZZ500 Total Return:  {benchmark_ret['total_return']:.2%}")
        print(f"  ZZ500 Annual Return: {benchmark_ret['annual_return']:.2%}")
        excess = float(metrics.get('annual_return', '0%').rstrip('%')) / 100 - benchmark_ret['annual_return']
        print(f"  Excess (alpha):      {excess:.2%}")

    print("-" * 60)
    print(f"  Total trades: {len(trade_log)}")
    print()

    # Monthly returns table
    print("  Monthly Returns:")
    print("  " + "-" * 50)
    for mr in monthly_returns:
        bar = "+" * int(max(0, mr["return"] * 100)) + "-" * int(max(0, -mr["return"] * 100))
        print(f"  {mr['month']:8s}  {mr['return']:+7.2%}  NAV={mr['nav']:.4f}  {bar}")
    print("=" * 60 + "\n")


def _save_report(metrics, monthly_returns, benchmark_ret, trade_log, elapsed):
    """Save markdown report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Walk-Forward Backtest Report",
        f"**Period**: {BT_START[:4]}-{BT_START[4:6]} ~ {BT_END[:4]}-{BT_END[4:6]}",
        f"**Initial Capital**: {INITIAL_CAPITAL:,.0f} | **Universe**: {UNIVERSE} | **Top-K**: {TOP_K}",
        f"**Model**: LightGBM monthly retrain, daily rebalance | **Horizon**: {HORIZON}d",
        f"**Runtime**: {elapsed/60:.1f} min",
        "",
        "## Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")

    if benchmark_ret:
        lines.extend([
            "",
            "## Benchmark (ZZ500)",
            "",
            f"| Total Return | {benchmark_ret['total_return']:.2%} |",
            f"| Annual Return | {benchmark_ret['annual_return']:.2%} |",
        ])

    lines.extend(["", "## Monthly Returns", "", "| Month | Return | NAV |", "|-------|--------|-----|"])
    for mr in monthly_returns:
        lines.append(f"| {mr['month']} | {mr['return']:+.2%} | {mr['nav']:.4f} |")

    lines.extend([
        "",
        f"## Trades ({len(trade_log)} total)",
        "",
    ])
    # Show first 20 and last 20 trades
    if len(trade_log) <= 40:
        show_trades = trade_log
    else:
        show_trades = trade_log[:20] + [{"date": "...", "code": "...", "action": "...", "shares": "...", "price": "...", "value": "..."}] + trade_log[-20:]

    lines.append("| Date | Code | Action | Shares | Price | Value |")
    lines.append("|------|------|--------|--------|-------|-------|")
    for t in show_trades:
        d = t["date"].strftime("%Y-%m-%d") if hasattr(t["date"], "strftime") else str(t["date"])
        lines.append(f"| {d} | {t['code']} | {t['action']} | {t['shares']} | {t.get('price', ''):.2f} | {t.get('value', ''):.0f} |" if isinstance(t.get("value"), (int, float)) else f"| {d} | {t['code']} | {t['action']} | ... | ... | ... |")

    lines.extend(["", "---", f"*Generated by Walk-Forward Backtest | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*"])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved to {}", REPORT_PATH)


# ──────────────────────────────────────────────────────────────────────
# Production Model Update
# ──────────────────────────────────────────────────────────────────────

def update_production_models(
    codes: List[str],
    ranker_20d=None,  # StockRanker or BlendRanker from walk-forward
) -> dict:
    """Save/retrain production models.

    - ranker_20d (BlendRanker or StockRanker, from walk-forward): if provided
      AND already trained, save it directly.  BlendRanker is the production
      default (saved to data/blend_*.lgb).  StockRanker is saved to data/model.lgb
      as a fallback.
    - 60d StockRanker: always retrain (walk-forward doesn't train 60d).

    Returns a dict with training metrics for all models.
    """
    from mp.ml.dataset import build_dataset
    from mp.ml.model import BlendRanker

    logger.info("=" * 60)
    logger.info("Updating production models...")
    logger.info("=" * 60)

    results: dict = {}
    ds_20 = None  # built lazily if needed

    # Detect what kind of ranker_20d we got
    ranker_is_blend = (
        ranker_20d is not None
        and hasattr(ranker_20d, "primary")
        and hasattr(ranker_20d, "extreme")
    )
    ranker_is_stock = (
        ranker_20d is not None
        and hasattr(ranker_20d, "model")
        and getattr(ranker_20d, "model", None) is not None
        and not ranker_is_blend
    )

    # ── Production BlendRanker (data/blend_*.lgb) — what daily_report/paper_trade use ──
    if ranker_is_blend and getattr(ranker_20d.primary, "model", None) is not None:
        ranker_20d.save("data/blend")
        logger.info("Production BlendRanker: saved from walk-forward (no retrain)")
        results["blend"] = {"source": "walk-forward", "saved": True}
    elif ranker_20d is None:
        # --update-only path: legitimate refresh from scratch
        logger.info("Training production BlendRanker (0.8 primary + 0.2 extreme)...")
        if ds_20 is None:
            ds_20 = build_dataset(codes, TRAIN_START, horizon=20)
        if ds_20 is not None and not ds_20.empty:
            blend = BlendRanker()
            try:
                blend_metrics = blend.train_fast(ds_20)
                blend.save("data/blend")
                results["blend"] = {
                    "saved": True,
                    "source": "retrained",
                    "n_train_rows": len(ds_20),
                    "primary_ic": blend_metrics.get("primary", {}).get("ic"),
                    "extreme_ic": blend_metrics.get("extreme", {}).get("ic"),
                }
                logger.info("BlendRanker: saved (rows={}, primary IC={:.3f}, extreme IC={:.3f})",
                            len(ds_20),
                            results["blend"].get("primary_ic") or float("nan"),
                            results["blend"].get("extreme_ic") or float("nan"))
            except Exception as e:
                logger.error("BlendRanker training failed: {}", e)
                results["blend"] = {"error": str(e)}
        else:
            results["blend"] = {"error": "no dataset"}
    else:
        # P3-1c: caller passed a non-blend ranker (typically RANKER_KIND=stock).
        # Skip retrain to avoid clobbering production blend (see commit 37ebfa8).
        logger.warning(
            "update_production_models: caller's ranker_20d is {} (not "
            "BlendRanker); SKIPPING blend_*.lgb retrain. Use --update-only "
            "to explicitly refresh blend.",
            type(ranker_20d).__name__,
        )
        results["blend"] = {"skipped": True,
                            "reason": f"non-blend ranker ({type(ranker_20d).__name__})"}

    # ── StockRanker fallback (data/model.lgb) ──
    if ranker_is_stock:
        ranker_20d.model_path = Path("data/model.lgb")
        ranker_20d.save()
        logger.info("20d StockRanker fallback: saved from walk-forward")
        results["20d"] = {"source": "walk-forward", "saved": True}
    else:
        logger.info("Building 20d dataset for StockRanker fallback (if not already)...")
        if ds_20 is None:
            ds_20 = build_dataset(codes, TRAIN_START, horizon=20)
        if ds_20 is not None and not ds_20.empty:
            ranker = StockRanker(model_path="data/model.lgb")
            m = ranker.train(ds_20)
            logger.info("20d StockRanker fallback: MAE={:.4f}±{:.4f}, IC={:.3f}±{:.3f}, rounds={}",
                         m["cv_mae_mean"], m["cv_mae_std"],
                         m["cv_ic_mean"], m["cv_ic_std"], m["best_rounds"])
            results["20d"] = m
        else:
            results["20d"] = {"error": "dataset build failed"}

    # 60-day StockRanker — always retrain
    logger.info("Building 60d dataset...")
    ds_60 = build_dataset(codes, TRAIN_START, horizon=60)
    if not ds_60.empty:
        ranker_60 = StockRanker(model_path="data/model_60d.lgb")
        m = ranker_60.train(ds_60)
        logger.info("60d StockRanker: MAE={:.4f}±{:.4f}, IC={:.3f}±{:.3f}, rounds={}",
                     m["cv_mae_mean"], m["cv_mae_std"],
                     m["cv_ic_mean"], m["cv_ic_std"], m["best_rounds"])
        results["60d"] = m
    else:
        logger.error("Failed to build 60d dataset")
        results["60d"] = {"error": "dataset build failed"}

    return results


# ──────────────────────────────────────────────────────────────────────
# Backtest history (for run-over-run comparison)
# ──────────────────────────────────────────────────────────────────────

BACKTEST_HISTORY_PATH = Path("data/reports/backtest_history.json")
_MAX_HISTORY = 10  # keep at most this many snapshots


def _save_backtest_snapshot(
    bt_metrics: dict,
    bt_benchmark: Optional[dict],
    model_results: dict,
) -> None:
    """Append current run metrics to the history file (max _MAX_HISTORY entries)."""
    import json

    snapshot = {
        "date": date.today().isoformat(),
        "bt_metrics": bt_metrics,
        "bt_benchmark": bt_benchmark,
        "model_ic": {
            h: model_results.get(h, {}).get("cv_ic_mean")
            for h in ("20d", "60d")
        },
        "model_hit_rate": {
            h: model_results.get(h, {}).get("cv_hit_rate_mean")
            for h in ("20d", "60d")
        },
    }

    history: list = []
    if BACKTEST_HISTORY_PATH.exists():
        try:
            history = json.loads(BACKTEST_HISTORY_PATH.read_text())
        except Exception:
            history = []

    history.append(snapshot)
    history = history[-_MAX_HISTORY:]  # trim oldest

    BACKTEST_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    BACKTEST_HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False))
    logger.info("Backtest snapshot saved to {}", BACKTEST_HISTORY_PATH)


def _load_prev_backtest_snapshot() -> Optional[dict]:
    """Return the second-to-last snapshot (i.e. the previous run), or None."""
    import json

    if not BACKTEST_HISTORY_PATH.exists():
        return None
    try:
        history = json.loads(BACKTEST_HISTORY_PATH.read_text())
        # last entry is current run (already saved), second-to-last is previous
        if len(history) >= 2:
            return history[-2]
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────
# Feishu notification
# ──────────────────────────────────────────────────────────────────────

def send_model_update_report(
    model_results: dict,
    bt_metrics: Optional[dict] = None,
    bt_benchmark: Optional[dict] = None,
    elapsed_min: float = 0.0,
    update_only: bool = False,
) -> bool:
    """Send a model update summary to Feishu via lark-cli.

    Parameters
    ----------
    model_results:
        Return value of ``update_production_models()``.
    bt_metrics:
        Walk-forward backtest performance metrics (optional, from full run).
    bt_benchmark:
        ZZ500 benchmark return dict (optional).
    elapsed_min:
        Total runtime in minutes.
    update_only:
        True if this was a ``--update-only`` run (no backtest).
    """
    import subprocess
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    today = date.today().strftime("%Y-%m-%d")
    mode = "仅模型更新" if update_only else "完整回测 + 模型更新"

    lines = [
        f"# 📊 模型更新完成",
        f"**{today}**  |  模式: {mode}  |  耗时: {elapsed_min:.1f} 分钟",
        "",
    ]

    # --- Model training summary ---
    lines.append("## 模型训练结果")
    lines.append("")

    for horizon, label in [("20d", "20日预测模型"), ("60d", "60日预测模型")]:
        m = model_results.get(horizon, {})
        if "error" in m:
            lines.append(f"**{label}**: ❌ {m['error']}")
        elif m.get("source") == "walk-forward":
            lines.append(f"**{label}**: ✅ 沿用回测末期模型（无需重训）")
        else:
            mae  = m.get("cv_mae_mean", "N/A")
            ic   = m.get("cv_ic_mean",  "N/A")
            hr   = m.get("cv_hit_rate_mean")
            prec = m.get("cv_precision_mean")
            rds  = m.get("best_rounds", "N/A")
            mae_str = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
            ic_str  = f"{ic:.3f}"  if isinstance(ic,  float) else str(ic)
            hr_str  = f"{hr:.2f}"  if isinstance(hr,  float) else "N/A"
            prec_str = f"{prec:.2f}" if isinstance(prec, float) else "N/A"
            lines.append(f"**{label}**: ✅  MAE={mae_str}  IC={ic_str}  HitRate@10={hr_str}  Precision@10={prec_str}  迭代={rds}")
        lines.append("")

    # --- Backtest summary (only for full runs) ---
    if not update_only and bt_metrics:
        lines.append("## 回测绩效摘要")
        lines.append("")
        key_map = {
            "total_return":  "总收益",
            "annual_return": "年化收益",
            "sharpe_ratio":  "夏普比率",
            "max_drawdown":  "最大回撤",
            "win_rate":      "胜率",
        }
        for k, label in key_map.items():
            v = bt_metrics.get(k)
            if v is not None:
                lines.append(f"- **{label}**: {v}")
        if bt_benchmark:
            excess = None
            ar_str = bt_metrics.get("annual_return", "")
            try:
                ar = float(str(ar_str).rstrip("%")) / 100
                excess = ar - bt_benchmark["annual_return"]
            except Exception:
                pass
            lines.append(f"- **ZZ500基准年化**: {bt_benchmark['annual_return']:.2%}")
            if excess is not None:
                lines.append(f"- **超额收益(alpha)**: {excess:+.2%}")
        lines.append("")

        # --- P4-1C threshold-breach alerts (BASELINE §4.1) ---
        # Inline path: append the markdown block to the weekly report so
        # the operator reading the Feishu weekly summary sees the breach.
        # P8-α-3 (docs/dialog/ round 53): ALSO dispatch the breach event
        # through alert_dispatch (Feishu + JSONL audit + stderr) so the
        # breach record survives even if the weekly report's Feishu send
        # is silent (e.g. lark-cli broken / webhook muted). Belt-and-
        # suspenders — the weekly report is for visibility, alert_dispatch
        # is for audit + SPOF mitigation.
        try:
            from mp.monitor.threshold_alert import (
                check_thresholds,
                format_for_feishu,
            )
            _alerts = check_thresholds(bt_metrics)
            if _alerts:
                _alert_block = format_for_feishu(_alerts)
                if _alert_block:
                    lines.append(_alert_block)
                    lines.append("")
                # Parallel multi-channel dispatch (P8-α-3)
                try:
                    from mp.monitor.alert_dispatch import dispatch_alert
                    _max_level = "RED" if any(a["level"] == "RED" for a in _alerts) else "YELLOW"
                    dispatch_alert(
                        level=_max_level,
                        title=f"{_max_level}: walk_forward threshold breach ({len(_alerts)} indicator(s))",
                        body=_alert_block,
                        source="threshold_alert",
                    )
                except Exception as _de:
                    logger.warning("alert_dispatch failed (inline weekly path still ran): {}", _de)
        except Exception as _e:
            logger.warning("threshold_alert dispatch failed: {}", _e)

        # --- Comparison with previous run ---
        prev = _load_prev_backtest_snapshot()
        if prev:
            prev_date = prev.get("date", "上次")
            lines.append(f"## 与上次回测对比（{prev_date}）")
            lines.append("")

            def _delta_str(cur_raw, prev_raw, is_pct: bool = False) -> str:
                """Format current value and delta vs previous."""
                try:
                    cur = float(str(cur_raw).rstrip("%")) / (100 if is_pct else 1)
                    prv = float(str(prev_raw).rstrip("%")) / (100 if is_pct else 1)
                    delta = cur - prv
                    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "↔")
                    cur_str = f"{cur:.2%}" if is_pct else f"{cur:.2f}"
                    prv_str = f"{prv:.2%}" if is_pct else f"{prv:.2f}"
                    delta_str = f"{delta:+.2%}" if is_pct else f"{delta:+.2f}"
                    return f"{prv_str} → {cur_str} ({arrow} {delta_str})"
                except Exception:
                    return str(cur_raw)

            prev_bt = prev.get("bt_metrics", {})
            cmp_map = [
                ("annual_return", "年化收益", True),
                ("sharpe_ratio",  "夏普比率", False),
                ("max_drawdown",  "最大回撤", True),
                ("win_rate",      "胜率",     True),
            ]
            for k, label, is_pct in cmp_map:
                cur_v = bt_metrics.get(k)
                prv_v = prev_bt.get(k)
                if cur_v is not None and prv_v is not None:
                    lines.append(f"- **{label}**: {_delta_str(cur_v, prv_v, is_pct)}")

            # Model IC & HitRate comparison
            for horizon, label in [("20d", "20日模型"), ("60d", "60日模型")]:
                m = model_results.get(horizon, {})
                cur_ic = m.get("cv_ic_mean")
                cur_hr = m.get("cv_hit_rate_mean")
                prv_ic = prev.get("model_ic", {}).get(horizon)
                prv_hr = prev.get("model_hit_rate", {}).get(horizon)
                parts = []
                if cur_ic is not None and prv_ic is not None:
                    parts.append(f"IC {_delta_str(cur_ic, prv_ic, False)}")
                if cur_hr is not None and prv_hr is not None:
                    parts.append(f"HitRate@10 {_delta_str(cur_hr, prv_hr, False)}")
                if parts:
                    lines.append(f"- **{label}**: {' | '.join(parts)}")
            lines.append("")

    # --- Footer ---
    lines.append("---")
    lines.append(f"*Money Printer 自动生成 | {now}*")

    markdown = "\n".join(lines)

    # Read default user ID from daily_report constants
    USER_ID = "ou_da792f0119461fb14c41b21b40834b09"

    # round 174 fix: resolve lark-cli via shutil.which (Windows ECS sometimes
    # lacks PATH inheritance into subprocess → WinError 2).
    import shutil
    binary = shutil.which("lark-cli")
    if binary is None:
        for cand in ("lark-cli.exe", "lark-cli.cmd", "lark-cli.bat"):
            binary = shutil.which(cand)
            if binary is not None:
                break
    if binary is None:
        logger.warning("lark-cli not found on PATH — model update notification skipped")
        return False

    cmd = [binary, "im", "+messages-send", "--as", "bot",
           "--user-id", USER_ID, "--markdown", markdown]

    logger.info("Sending model update report to Feishu...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("Model update report sent to Feishu")
            return True
        else:
            logger.error("Feishu send failed: {}", result.stderr or result.stdout)
            return False
    except FileNotFoundError as e:
        logger.warning("lark-cli binary at {} not executable ({}) — notification skipped", binary, e)
        return False
    except Exception as e:
        logger.error("Feishu send error: {}", e)
        return False


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Walk-Forward Backtest (monthly retrain, daily rebalance)")
    parser.add_argument("--skip-update", action="store_true",
                        help="Skip production model update after backtest")
    parser.add_argument("--cache-only", action="store_true",
                        help="Only build data cache, don't run backtest")
    parser.add_argument("--update-only", action="store_true",
                        help="[DEPRECATED 2026-05-24 P6-A2] DO NOT USE. "
                             "Triggered the P3-1c residual train_fast bug "
                             "(IC=-0.005 weak blend overwrites production). "
                             "Now raises SystemExit. See docs/dialog/ round 47.")
    args = parser.parse_args()

    if args.update_only:
        raise SystemExit(
            "ERROR: --update-only is deprecated (P3-1c residual bug — it would\n"
            "       train_fast on full panel and overwrite production blend\n"
            "       models with a weaker non-walk-forward fit; IC=-0.005 vs\n"
            "       baseline +0.038).\n"
            "\n"
            "       Use full walk-forward retrain instead:\n"
            "         python scripts/walk_forward_backtest.py [other args]\n"
            "\n"
            "       Production crontab was migrated 2026-05-24 (commit f5b5255,\n"
            "       see docs/cron_setup.md). If you're seeing this from your own\n"
            "       script/alias, update it too."
        )
    elif args.cache_only:
        codes = _merged_current()
        bars_map = _load_or_fetch_bars(codes)
        _load_or_build_factors(bars_map, codes)
        logger.info("Cache built successfully. Run again without --cache-only to start backtest.")
    else:
        metrics, monthly, last_ranker, bt_benchmark = run_walk_forward()

        if not args.skip_update:
            t1 = time.time()
            codes = _merged_current()
            model_results = update_production_models(codes, ranker_20d=last_ranker)
            elapsed_min = (time.time() - t1) / 60
            logger.info("Production models updated! ({:.1f} min)", elapsed_min)
            _save_backtest_snapshot(metrics, bt_benchmark, model_results)
            send_model_update_report(
                model_results=model_results,
                bt_metrics=metrics,
                bt_benchmark=bt_benchmark,
                elapsed_min=elapsed_min,
                update_only=False,
            )
        else:
            logger.info("Skipping production model update (--skip-update)")
