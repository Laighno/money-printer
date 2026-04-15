"""Prediction Diagnostics — validate model ranking ability.

Reuses the walk-forward cached factor panel to score ALL stocks (not just top-K),
then runs four diagnostic analyses:
  1. Decile returns (分层收益)
  2. Top-K hit rate (命中率)
  3. Rolling IC by market regime (滚动稳定性)
  4. Long-short spread (多空价差)

Usage:
    python scripts/prediction_diagnostics.py [--force] [--analysis-only] [--top-k 10]
"""

from __future__ import annotations

import argparse
import bisect
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from mp.ml.dataset import FACTOR_COLUMNS, TECHNICAL_COLUMNS, EXCESS_LABEL, add_excess_ret, filter_universe
from mp.ml.model import StockRanker, TwoStageRanker, BlendRanker

# ──────────────────────────────────────────────────────────────────────
# Parameters (mirror walk_forward_backtest.py)
# ──────────────────────────────────────────────────────────────────────

TRAIN_START = "20150101"
BT_START = "20200101"
BT_END = "20260401"
HORIZON = 20
TOP_K = 10

CACHE_DIR = Path("data/wf_cache")
FACTORS_CACHE = CACHE_DIR / "factors.parquet"
DIAG_CACHE = CACHE_DIR / "diagnostics.parquet"
REPORT_PATH = Path("data/reports/prediction_diagnostics.md")

# Market regimes for IC analysis
REGIMES: Dict[str, List[Tuple[str, str]]] = {
    "牛市 (Bull)": [
        ("2020-07-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
        ("2024-09-01", "2024-12-31"),
    ],
    "熊市 (Bear)": [
        ("2022-01-01", "2022-06-30"),
        ("2024-01-01", "2024-06-30"),
    ],
    "震荡 (Sideways)": [
        ("2023-01-01", "2023-12-31"),
        ("2025-01-01", "2025-06-30"),
    ],
}


def _classify_regime(dt: pd.Timestamp) -> str:
    """Classify a date into a market regime."""
    for regime, periods in REGIMES.items():
        for start, end in periods:
            if pd.Timestamp(start) <= dt <= pd.Timestamp(end):
                return regime
    return "其他 (Other)"


# ──────────────────────────────────────────────────────────────────────
# Label cutoff helper (trading-day based, not calendar-day approximation)
# ──────────────────────────────────────────────────────────────────────

_TRADE_DATES_CACHE: dict[int, list] = {}


def _label_cutoff(panel: pd.DataFrame, dt, horizon: int = HORIZON):
    """Go back `horizon` actual trading days for label cutoff."""
    pid = id(panel)
    if pid not in _TRADE_DATES_CACHE:
        _TRADE_DATES_CACHE[pid] = sorted(panel["date"].unique())
    all_dates = _TRADE_DATES_CACHE[pid]
    idx = bisect.bisect_right(all_dates, dt) - 1
    return all_dates[max(0, idx - horizon)]


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Scoring Loop
# ──────────────────────────────────────────────────────────────────────

def _get_monthly_retrain_dates(panel: pd.DataFrame) -> List[pd.Timestamp]:
    """First trading day of each month in [BT_START, BT_END)."""
    bt_start = pd.Timestamp(BT_START)
    bt_end = pd.Timestamp(BT_END)
    dates = panel["date"].drop_duplicates().sort_values()
    dates = dates[(dates >= bt_start) & (dates < bt_end)]
    monthly = dates.groupby(dates.dt.to_period("M")).first()
    return monthly.tolist()


def run_scoring_loop(force: bool = False) -> pd.DataFrame:
    """Monthly retrain + full-universe scoring → diagnostics.parquet.

    Returns DataFrame with columns: score_date, code, pred_score, fwd_ret.
    """
    # Check cache
    if not force and DIAG_CACHE.exists():
        logger.info("Loading cached diagnostics from {}", DIAG_CACHE)
        return pd.read_parquet(DIAG_CACHE)

    # Load factor panel
    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found at {}. Run walk_forward_backtest.py first.", FACTORS_CACHE)
        sys.exit(1)

    logger.info("Loading factor panel...")
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])
    logger.info("Factor panel: {} rows, {} stocks, {} ~ {}",
                len(panel), panel["code"].nunique(),
                panel["date"].min().strftime("%Y-%m-%d"),
                panel["date"].max().strftime("%Y-%m-%d"))

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("{} monthly retrain dates", len(retrain_dates))

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []
    current_ranker: Optional[StockRanker] = None

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] Retrain & score: {}", i + 1, len(retrain_dates),
                    dt.strftime("%Y-%m-%d"))

        # --- Retrain ---
        label_cutoff = _label_cutoff(panel, dt)
        train_mask = (panel["date"] < label_cutoff) & panel["fwd_ret"].notna()
        train_df = panel.loc[train_mask]

        if len(train_df) < 500:
            logger.warning("  Too few training rows ({}), skipping", len(train_df))
            continue

        ranker = StockRanker(feature_cols=FACTOR_COLUMNS)
        try:
            metrics = ranker.train_fast(train_df)
            logger.info("  Train: {} rows, MAE={:.4f}, IC={:.3f}",
                        len(train_df), metrics["mae"], metrics["ic"])
            current_ranker = ranker
        except Exception as e:
            logger.error("  Training failed: {}", e)
            continue

        # --- Score all stocks on this date ---
        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)

        if today_valid.empty:
            logger.warning("  No valid stocks on {}", dt.strftime("%Y-%m-%d"))
            continue

        scores = current_ranker.predict(today_valid)
        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
            })

    diag_df = pd.DataFrame(records)

    # Drop rows without forward return (can't evaluate)
    before = len(diag_df)
    diag_df = diag_df.dropna(subset=["fwd_ret"])
    logger.info("Diagnostics: {} scored rows ({} dropped for missing fwd_ret)",
                len(diag_df), before - len(diag_df))

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(DIAG_CACHE, index=False)
    logger.info("Cached to {}", DIAG_CACHE)

    return diag_df


def run_scoring_loop_experiment(
    force: bool = False,
    objective: str = "regression",
    label_col: str = "fwd_ret",
    use_filter: bool = False,
    train_cap_pctile: float = 0.0,
    post_cap_pctile: float = 0.0,
    cap_blend_alpha: float = 0.0,
    tag: str = "exp",
) -> pd.DataFrame:
    """Parameterized single-stage scoring for A/B experiments.

    Parameters
    ----------
    train_cap_pctile : float
        If > 0, only train on stocks above this percentile of total_mv_log
        per date (e.g. 0.3 = exclude bottom 30% by market cap from training).
    post_cap_pctile : float
        If > 0, remove stocks below this percentile of total_mv_log from
        predictions before analysis (e.g. 0.1 = exclude bottom 10%).
    cap_blend_alpha : float
        If > 0, blend cap rank into score: final = pred_score + alpha * cap_pct_rank.
        Higher alpha pushes the model toward larger-cap picks.

    Saves to diagnostics_{tag}.parquet.
    """
    cache_path = CACHE_DIR / f"diagnostics_{tag}.parquet"
    if not force and cache_path.exists():
        logger.info("Loading cached {} diagnostics", tag)
        return pd.read_parquet(cache_path)

    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found.")
        sys.exit(1)

    logger.info("[{}] Loading factor panel...", tag)
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    # Add excess return if needed
    if label_col == EXCESS_LABEL:
        panel = add_excess_ret(panel, horizon=HORIZON)

    if use_filter:
        panel = filter_universe(panel)

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("[{}] {} retrain dates, objective={}, label={}, filter={}, "
                "train_cap_pctile={}, post_cap_pctile={}",
                tag, len(retrain_dates), objective, label_col, use_filter,
                train_cap_pctile, post_cap_pctile)

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []
    current_ranker: Optional[StockRanker] = None

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] {}: retrain {}", i + 1, len(retrain_dates), tag,
                    dt.strftime("%Y-%m-%d"))

        label_cutoff = _label_cutoff(panel, dt)
        train_mask = (panel["date"] < label_cutoff) & panel[label_col].notna()
        train_df = panel.loc[train_mask]

        # Filter training data to mid/large caps if requested
        if train_cap_pctile > 0 and "total_mv_log" in train_df.columns:
            cap_thresh = train_df.groupby("date")["total_mv_log"].transform(
                lambda s: s.quantile(train_cap_pctile))
            train_df = train_df[train_df["total_mv_log"] >= cap_thresh]

        if len(train_df) < 500:
            continue

        ranker = StockRanker(
            feature_cols=FACTOR_COLUMNS,
            objective=objective,
            label_col=label_col,
        )
        try:
            metrics = ranker.train_fast(train_df)
            logger.info("  Train: {} rows, IC={:.3f}, HitRate@K={:.3f}",
                        len(train_df), metrics.get("ic", 0), metrics.get("hit_rate_at_k", 0))
            current_ranker = ranker
        except Exception as e:
            logger.error("  Training failed: {}", e)
            continue

        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)
        if today_valid.empty:
            continue

        scores = current_ranker.predict(today_valid)

        # Score blending: add cap rank component
        if cap_blend_alpha > 0 and "total_mv_log" in today_valid.columns:
            cap_vals = today_valid["total_mv_log"].values
            cap_rank = pd.Series(cap_vals).rank(pct=True).values
            # Normalize pred_score to [0,1] range for fair blending
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                norm_scores = (scores - s_min) / (s_max - s_min)
            else:
                norm_scores = np.full_like(scores, 0.5)
            scores = norm_scores + cap_blend_alpha * cap_rank

        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            cap = row.get("total_mv_log", np.nan)
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
                "total_mv_log": float(cap) if pd.notna(cap) else np.nan,
            })

    diag_df = pd.DataFrame(records)
    diag_df = diag_df.dropna(subset=["fwd_ret"])

    # Post-prediction cap filter: remove smallest stocks from consideration
    if post_cap_pctile > 0 and "total_mv_log" in diag_df.columns:
        n_before = len(diag_df)
        cap_thresh = diag_df.groupby("score_date")["total_mv_log"].transform(
            lambda s: s.quantile(post_cap_pctile))
        diag_df = diag_df[diag_df["total_mv_log"] >= cap_thresh].reset_index(drop=True)
        logger.info("[{}] Post-cap filter: {} → {} rows (removed bottom {:.0%})",
                    tag, n_before, len(diag_df), post_cap_pctile)

    logger.info("[{}] {} scored rows", tag, len(diag_df))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(cache_path, index=False)
    return diag_df


def _add_market_context_features(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add ZZ500 momentum + per-stock beta as market-context features.

    Returns (panel_with_features, list_of_new_column_names).
    """
    import akshare as ak

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    # 1. Fetch ZZ500 daily close
    idx_df = ak.stock_zh_index_daily(symbol="sh000905")
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_df = idx_df.sort_values("date").set_index("date")
    idx_close = idx_df["close"]

    # ZZ500 20d momentum at each date
    idx_mom = idx_close.pct_change(20)
    # ZZ500 5d momentum
    idx_mom5 = idx_close.pct_change(5)
    # ZZ500 daily returns for beta calculation
    idx_ret = idx_close.pct_change()

    # Map to panel dates
    panel_dates = panel["date"].drop_duplicates().sort_values()
    zz500_mom_map = idx_mom.reindex(panel_dates, method="ffill").to_dict()
    zz500_mom5_map = idx_mom5.reindex(panel_dates, method="ffill").to_dict()

    panel["zz500_mom_20d"] = panel["date"].map(zz500_mom_map).astype(float)
    panel["zz500_mom_5d"] = panel["date"].map(zz500_mom5_map).astype(float)

    # 2. Per-stock rolling 60d beta to ZZ500
    # Need bars data — load from cache
    bars_cache = CACHE_DIR / "bars.parquet"
    if bars_cache.exists():
        bars = pd.read_parquet(bars_cache)
        bars["date"] = pd.to_datetime(bars["date"])

        # Compute stock daily return
        stock_rets = bars.sort_values(["code", "date"]).groupby("code")["close"].pct_change()
        bars["stock_ret"] = stock_rets.values

        # Map index return to bars
        bars["idx_ret"] = bars["date"].map(idx_ret.to_dict()).astype(float)

        # Rolling 60d beta per stock
        def _rolling_beta(g, window=60):
            sr = g["stock_ret"]
            ir = g["idx_ret"]
            cov = sr.rolling(window).cov(ir)
            var = ir.rolling(window).var()
            return cov / var.replace(0, np.nan)

        beta_parts = []
        for code, g in bars.groupby("code"):
            g = g.sort_values("date").copy()
            g["beta_60d"] = _rolling_beta(g)
            beta_parts.append(g[["date", "code", "beta_60d"]])
        beta_df = pd.concat(beta_parts, ignore_index=True)

        panel = panel.merge(beta_df, on=["date", "code"], how="left")
    else:
        panel["beta_60d"] = np.nan
        logger.warning("No bars cache found; beta_60d will be NaN")

    new_cols = ["zz500_mom_20d", "zz500_mom_5d", "beta_60d"]
    logger.info("Added market context features: {}", new_cols)
    return panel, new_cols


def _add_residual_alpha_features(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add per-stock residual alpha features that vary cross-sectionally.

    Unlike raw ZZ500 momentum (constant per date), these capture
    stock-specific outperformance relative to market exposure.

    Features added:
      - residual_mom_20d: stock mom_20d minus beta * zz500_mom_20d
      - residual_mom_5d:  short-term residual momentum
      - idio_vol_60d:     idiosyncratic volatility (std of residual returns)
      - alpha_20d:        rolling 20d cumulative alpha (sum of daily residuals)

    Returns (panel_with_features, list_of_new_column_names).
    """
    import akshare as ak

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    # 1. Fetch ZZ500 daily close
    idx_df = ak.stock_zh_index_daily(symbol="sh000905")
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_df = idx_df.sort_values("date").set_index("date")
    idx_close = idx_df["close"]

    # ZZ500 momentum
    idx_mom20 = idx_close.pct_change(20)
    idx_mom5 = idx_close.pct_change(5)
    idx_ret = idx_close.pct_change()

    panel_dates = panel["date"].drop_duplicates().sort_values()
    zz500_mom20_map = idx_mom20.reindex(panel_dates, method="ffill").to_dict()
    zz500_mom5_map = idx_mom5.reindex(panel_dates, method="ffill").to_dict()

    panel["_zz500_mom_20d"] = panel["date"].map(zz500_mom20_map).astype(float)
    panel["_zz500_mom_5d"] = panel["date"].map(zz500_mom5_map).astype(float)

    # 2. Per-stock rolling beta + residual stats from bars
    bars_cache = CACHE_DIR / "bars.parquet"
    if not bars_cache.exists():
        logger.warning("No bars cache; residual alpha features will be NaN")
        for col in ["residual_mom_20d", "residual_mom_5d", "idio_vol_60d", "alpha_20d"]:
            panel[col] = np.nan
        panel.drop(columns=["_zz500_mom_20d", "_zz500_mom_5d"], inplace=True)
        new_cols = ["residual_mom_20d", "residual_mom_5d", "idio_vol_60d", "alpha_20d"]
        return panel, new_cols

    bars = pd.read_parquet(bars_cache)
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.sort_values(["code", "date"])

    # Stock daily returns
    bars["stock_ret"] = bars.groupby("code")["close"].pct_change()
    # Map index return
    bars["idx_ret"] = bars["date"].map(idx_ret.to_dict()).astype(float)

    # Per-stock rolling stats
    residual_parts = []
    for code, g in bars.groupby("code"):
        g = g.sort_values("date").copy()
        sr = g["stock_ret"]
        ir = g["idx_ret"]

        # Rolling 60d beta
        cov_60 = sr.rolling(60).cov(ir)
        var_60 = ir.rolling(60).var()
        beta = cov_60 / var_60.replace(0, np.nan)

        # Daily residual return: stock_ret - beta * mkt_ret
        daily_resid = sr - beta * ir

        # Idiosyncratic vol: std of residuals over 60d
        idio_vol = daily_resid.rolling(60).std()

        # Rolling 20d cumulative alpha (sum of daily residuals)
        alpha_20d = daily_resid.rolling(20).sum()

        g["beta_60d"] = beta
        g["idio_vol_60d"] = idio_vol
        g["alpha_20d"] = alpha_20d
        residual_parts.append(g[["date", "code", "beta_60d", "idio_vol_60d", "alpha_20d"]])

    resid_df = pd.concat(residual_parts, ignore_index=True)
    panel = panel.merge(resid_df, on=["date", "code"], how="left")

    # Compute residual momentum: stock_mom - beta * zz500_mom
    # mom_20d and mom_5d should already be in panel from FACTOR_COLUMNS
    if "mom_20d" in panel.columns:
        panel["residual_mom_20d"] = panel["mom_20d"] - panel["beta_60d"] * panel["_zz500_mom_20d"]
    else:
        panel["residual_mom_20d"] = np.nan

    # Short-term residual momentum from bars isn't directly available as mom_5d
    # in panel, so approximate using alpha_20d for short-term and a scaled version
    # Use a 5d version: we have mom_5d? Let me check...
    if "mom_5d" in panel.columns:
        panel["residual_mom_5d"] = panel["mom_5d"] - panel["beta_60d"] * panel["_zz500_mom_5d"]
    else:
        panel["residual_mom_5d"] = np.nan

    # Clean up temp columns
    panel.drop(columns=["_zz500_mom_20d", "_zz500_mom_5d"], inplace=True)

    new_cols = ["residual_mom_20d", "residual_mom_5d", "idio_vol_60d", "alpha_20d"]
    n_valid = panel[new_cols].notna().all(axis=1).sum()
    logger.info("Added residual alpha features: {}, {}/{} rows fully valid",
                new_cols, n_valid, len(panel))
    return panel, new_cols


def run_scoring_loop_residual(
    force: bool = False,
    tag: str = "residual",
    lgb_params: Optional[Dict] = None,
    label_col: str = "fwd_ret",
    n_models: int = 1,
) -> pd.DataFrame:
    """Scoring loop with per-stock residual alpha features.

    These features vary cross-sectionally (unlike ZZ500 momentum) and
    capture stock-specific alpha after removing market exposure.
    """
    cache_path = CACHE_DIR / f"diagnostics_{tag}.parquet"
    if not force and cache_path.exists():
        logger.info("Loading cached {} diagnostics", tag)
        return pd.read_parquet(cache_path)

    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found.")
        sys.exit(1)

    logger.info("[{}] Loading factor panel + computing residual alpha features...", tag)
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    # Add excess return if needed
    if label_col == EXCESS_LABEL:
        panel = add_excess_ret(panel, horizon=HORIZON)

    panel, alpha_cols = _add_residual_alpha_features(panel)
    extended_features = FACTOR_COLUMNS + alpha_cols

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("[{}] {} retrain dates, {} features (base {} + {} alpha), label={}, n_models={}",
                tag, len(retrain_dates), len(extended_features),
                len(FACTOR_COLUMNS), len(alpha_cols), label_col, n_models)

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] {}: retrain {}", i + 1, len(retrain_dates), tag,
                    dt.strftime("%Y-%m-%d"))

        label_cutoff = _label_cutoff(panel, dt)
        train_mask = (panel["date"] < label_cutoff) & panel[label_col].notna()
        train_df = panel.loc[train_mask]

        if len(train_df) < 500:
            continue

        # Train ensemble of models
        rankers = []
        for seed in range(n_models):
            params = {"seed": seed + 42, "bagging_seed": seed + 42,
                      "feature_fraction_seed": seed + 42}
            if lgb_params:
                params.update(lgb_params)
            ranker = StockRanker(
                feature_cols=extended_features,
                objective="regression",
                label_col=label_col,
                lgb_params=params,
            )
            try:
                metrics = ranker.train_fast(train_df)
                rankers.append(ranker)
            except Exception as e:
                logger.error("  Model {} failed: {}", seed, e)

        if not rankers:
            continue

        if i % 10 == 0:
            logger.info("  {} models trained, IC={:.3f}",
                        len(rankers), metrics.get("ic", 0))

        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)
        if today_valid.empty:
            continue

        # Average scores across models
        if len(rankers) == 1:
            scores = rankers[0].predict(today_valid)
        else:
            all_scores = np.stack([r.predict(today_valid) for r in rankers])
            scores = all_scores.mean(axis=0)

        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
            })

    diag_df = pd.DataFrame(records)
    diag_df = diag_df.dropna(subset=["fwd_ret"])
    logger.info("[{}] {} scored rows", tag, len(diag_df))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(cache_path, index=False)
    return diag_df


def run_scoring_loop_multi_horizon(
    force: bool = False,
    tag: str = "multi_horizon",
    horizons: Tuple[int, ...] = (5, 10, 20),
    lgb_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """Multi-horizon ensemble: train separate models on different forward-return horizons,
    then average their ranking scores.

    Stocks that look good across all horizons are more robust picks.
    Evaluation is still done against the 20d fwd_ret for apples-to-apples comparison.
    """
    cache_path = CACHE_DIR / f"diagnostics_{tag}.parquet"
    if not force and cache_path.exists():
        logger.info("Loading cached {} diagnostics", tag)
        return pd.read_parquet(cache_path)

    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found.")
        sys.exit(1)

    logger.info("[{}] Loading factor panel...", tag)
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    # Compute forward returns for each horizon from bars
    bars_cache = CACHE_DIR / "bars.parquet"
    if not bars_cache.exists():
        logger.error("Bars cache required for multi-horizon. Run walk_forward_backtest.py first.")
        sys.exit(1)

    bars = pd.read_parquet(bars_cache)
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.sort_values(["code", "date"])

    for h in horizons:
        col_name = f"fwd_ret_{h}d"
        if col_name not in panel.columns:
            # Compute from bars
            fwd_parts = []
            for code, g in bars.groupby("code"):
                g = g.sort_values("date").copy()
                g[col_name] = g["close"].shift(-h) / g["close"] - 1.0
                fwd_parts.append(g[["date", "code", col_name]])
            fwd_df = pd.concat(fwd_parts, ignore_index=True)
            panel = panel.merge(fwd_df, on=["date", "code"], how="left")
            logger.info("[{}] Added {} from bars ({} valid)", tag, col_name,
                        panel[col_name].notna().sum())

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("[{}] {} retrain dates, horizons={}", tag, len(retrain_dates), horizons)

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] {}: retrain {}", i + 1, len(retrain_dates), tag,
                    dt.strftime("%Y-%m-%d"))

        label_cutoff = _label_cutoff(panel, dt, horizon=max(horizons))
        train_mask = (panel["date"] < label_cutoff) & panel["fwd_ret"].notna()
        train_df = panel.loc[train_mask]

        if len(train_df) < 500:
            continue

        # Train one model per horizon
        rankers = []
        for h in horizons:
            label_col = f"fwd_ret_{h}d" if h != HORIZON else "fwd_ret"
            sub_train = train_df.dropna(subset=[label_col])
            if len(sub_train) < 500:
                continue
            ranker = StockRanker(
                feature_cols=FACTOR_COLUMNS,
                objective="regression",
                label_col=label_col,
                lgb_params=lgb_params,
            )
            try:
                metrics = ranker.train_fast(sub_train)
                rankers.append(ranker)
            except Exception as e:
                logger.error("  Horizon {}d failed: {}", h, e)

        if not rankers:
            continue

        if i % 10 == 0:
            logger.info("  {} horizon models trained", len(rankers))

        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)
        if today_valid.empty:
            continue

        # Average scores across horizon models
        all_scores = np.stack([r.predict(today_valid) for r in rankers])
        scores = all_scores.mean(axis=0)

        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
            })

    diag_df = pd.DataFrame(records)
    diag_df = diag_df.dropna(subset=["fwd_ret"])
    logger.info("[{}] {} scored rows", tag, len(diag_df))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(cache_path, index=False)
    return diag_df


def run_scoring_loop_beta_tilt(
    force: bool = False,
    tag: str = "beta_tilt",
    lgb_params: Optional[Dict] = None,
    tilt_alpha: float = 0.3,
) -> pd.DataFrame:
    """Score with deep model, then apply beta-tilt in rising markets.

    The model has r=-0.51 anti-ZZ500 bias: 72% hit in bear, 38% in bull.
    This adds a post-scoring adjustment: when recent market momentum is positive,
    boost high-beta stocks' scores. When negative, boost low-beta.

    tilt_alpha controls the strength of the adjustment (0 = no tilt, 1 = strong).
    """
    cache_path = CACHE_DIR / f"diagnostics_{tag}.parquet"
    if not force and cache_path.exists():
        logger.info("Loading cached {} diagnostics", tag)
        return pd.read_parquet(cache_path)

    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found.")
        sys.exit(1)

    logger.info("[{}] Loading data + computing per-stock beta...", tag)
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    # Get ZZ500 data for momentum signal and per-stock beta
    import akshare as ak
    idx_df = ak.stock_zh_index_daily(symbol="sh000905")
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_df = idx_df.sort_values("date").set_index("date")
    idx_close = idx_df["close"]
    idx_ret = idx_close.pct_change()
    idx_mom20 = idx_close.pct_change(20)  # 20d momentum signal

    # Build per-date ZZ500 momentum map
    panel_dates = panel["date"].drop_duplicates().sort_values()
    zz500_mom_map = idx_mom20.reindex(panel_dates, method="ffill").to_dict()
    panel["_zz500_mom"] = panel["date"].map(zz500_mom_map).astype(float)

    # Compute per-stock rolling 60d beta from bars
    bars_cache = CACHE_DIR / "bars.parquet"
    bars = pd.read_parquet(bars_cache)
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.sort_values(["code", "date"])
    bars["stock_ret"] = bars.groupby("code")["close"].pct_change()
    bars["idx_ret"] = bars["date"].map(idx_ret.to_dict()).astype(float)

    beta_parts = []
    for code, g in bars.groupby("code"):
        g = g.sort_values("date").copy()
        cov_60 = g["stock_ret"].rolling(60).cov(g["idx_ret"])
        var_60 = g["idx_ret"].rolling(60).var()
        g["beta_60d"] = cov_60 / var_60.replace(0, np.nan)
        beta_parts.append(g[["date", "code", "beta_60d"]])
    beta_df = pd.concat(beta_parts, ignore_index=True)
    panel = panel.merge(beta_df, on=["date", "code"], how="left")
    panel["beta_60d"] = panel["beta_60d"].fillna(1.0)  # assume beta=1 for missing

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("[{}] {} retrain dates, tilt_alpha={}", tag, len(retrain_dates), tilt_alpha)

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []
    current_ranker: Optional[StockRanker] = None

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] {}: retrain {}", i + 1, len(retrain_dates), tag,
                    dt.strftime("%Y-%m-%d"))

        label_cutoff = _label_cutoff(panel, dt)
        train_mask = (panel["date"] < label_cutoff) & panel["fwd_ret"].notna()
        train_df = panel.loc[train_mask]
        if len(train_df) < 500:
            continue

        ranker = StockRanker(
            feature_cols=FACTOR_COLUMNS,
            lgb_params=lgb_params,
        )
        try:
            metrics = ranker.train_fast(train_df)
            current_ranker = ranker
        except Exception as e:
            logger.error("  Training failed: {}", e)
            continue

        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)
        if today_valid.empty:
            continue

        raw_scores = current_ranker.predict(today_valid)

        # Beta-tilt post-processing:
        # If market has been rising (positive ZZ500 20d mom), boost high-beta stocks
        # If falling, boost low-beta stocks (which tend to be defensive)
        market_mom = today_valid["_zz500_mom"].iloc[0] if len(today_valid) > 0 else 0.0
        betas = today_valid["beta_60d"].values
        # Normalize beta to z-score for fair blending
        beta_z = (betas - np.nanmean(betas)) / max(np.nanstd(betas), 1e-8)
        # Normalize raw scores to z-score
        score_z = (raw_scores - raw_scores.mean()) / max(raw_scores.std(), 1e-8)
        # Tilt: positive momentum × positive beta = higher score
        # The sign of market_mom handles the direction automatically
        tilt_signal = np.sign(market_mom) * np.clip(np.abs(market_mom) * 10, 0, 1)
        scores = score_z + tilt_alpha * tilt_signal * beta_z

        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
            })

    diag_df = pd.DataFrame(records)
    diag_df = diag_df.dropna(subset=["fwd_ret"])
    logger.info("[{}] {} scored rows", tag, len(diag_df))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(cache_path, index=False)
    return diag_df


def _add_regime_interaction_features(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add features that capture stock×market-regime interactions.

    The model's key failure: high-momentum picks in bull markets mean-revert.
    Raw momentum is cross-sectional but lacks market context.
    Raw ZZ500 momentum is per-date constant (useless for ranking).
    Their PRODUCT varies cross-sectionally AND encodes regime.

    Features added:
      - mom_x_mkt20:  mom_20d * zz500_mom_20d  (momentum × market direction)
      - mom5_x_mkt20: mom_5d * zz500_mom_20d   (short-term momentum × market)
      - vol_x_mkt20:  volatility * zz500_mom_20d (risk × market direction)
      - mom_20d_rank:  cross-sectional percentile rank of mom_20d per date
      - mom_5d_rank:   cross-sectional percentile rank of mom_5d per date
      - mom_spread:    mom_5d - mom_20d (short-long momentum divergence)
      - turn_x_mkt20: turnover_5d * zz500_mom_20d (activity × market)
    """
    import akshare as ak

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    # Fetch ZZ500 momentum
    idx_df = ak.stock_zh_index_daily(symbol="sh000905")
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_df = idx_df.sort_values("date").set_index("date")
    idx_close = idx_df["close"]
    idx_mom20 = idx_close.pct_change(20)
    idx_mom5 = idx_close.pct_change(5)

    panel_dates = panel["date"].drop_duplicates().sort_values()
    mkt20_map = idx_mom20.reindex(panel_dates, method="ffill").to_dict()
    mkt5_map = idx_mom5.reindex(panel_dates, method="ffill").to_dict()

    panel["_mkt20"] = panel["date"].map(mkt20_map).astype(float)
    panel["_mkt5"] = panel["date"].map(mkt5_map).astype(float)

    # Interaction features (cross-sectional × market-state)
    new_cols = []

    if "mom_20d" in panel.columns:
        panel["mom_x_mkt20"] = panel["mom_20d"] * panel["_mkt20"]
        panel["mom_20d_rank"] = panel.groupby("date")["mom_20d"].rank(pct=True)
        new_cols += ["mom_x_mkt20", "mom_20d_rank"]

    if "mom_5d" in panel.columns:
        panel["mom5_x_mkt20"] = panel["mom_5d"] * panel["_mkt20"]
        panel["mom_5d_rank"] = panel.groupby("date")["mom_5d"].rank(pct=True)
        new_cols += ["mom5_x_mkt20", "mom_5d_rank"]

    if "mom_5d" in panel.columns and "mom_20d" in panel.columns:
        panel["mom_spread"] = panel["mom_5d"] - panel["mom_20d"]
        new_cols.append("mom_spread")

    if "volatility" in panel.columns:
        panel["vol_x_mkt20"] = panel["volatility"] * panel["_mkt20"]
        new_cols.append("vol_x_mkt20")

    if "turnover_5d" in panel.columns:
        panel["turn_x_mkt20"] = panel["turnover_5d"] * panel["_mkt20"]
        new_cols.append("turn_x_mkt20")

    # Clean up temp columns
    panel.drop(columns=["_mkt20", "_mkt5"], inplace=True)

    n_valid = panel[new_cols].notna().all(axis=1).sum()
    logger.info("Added {} regime-interaction features: {}, {}/{} rows valid",
                len(new_cols), new_cols, n_valid, len(panel))
    return panel, new_cols


def run_scoring_loop_regime_interact(
    force: bool = False,
    tag: str = "regime_interact",
    lgb_params: Optional[Dict] = None,
    label_col: str = "fwd_ret",
    n_models: int = 1,
) -> pd.DataFrame:
    """Scoring loop with regime-interaction features.

    Targets the anti-ZZ500 bias (r=-0.51): model picks high-momentum stocks
    in bull markets that mean-revert. Interaction features let the model learn
    "high momentum in bull market = bad for relative performance."
    """
    cache_path = CACHE_DIR / f"diagnostics_{tag}.parquet"
    if not force and cache_path.exists():
        logger.info("Loading cached {} diagnostics", tag)
        return pd.read_parquet(cache_path)

    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found.")
        sys.exit(1)

    logger.info("[{}] Loading factor panel + computing regime-interaction features...", tag)
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    # Add excess return if needed
    if label_col == EXCESS_LABEL:
        panel = add_excess_ret(panel, horizon=HORIZON)

    panel, interact_cols = _add_regime_interaction_features(panel)
    extended_features = FACTOR_COLUMNS + interact_cols

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("[{}] {} retrain dates, {} features (base {} + {} interact), label={}, n_models={}",
                tag, len(retrain_dates), len(extended_features),
                len(FACTOR_COLUMNS), len(interact_cols), label_col, n_models)

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] {}: retrain {}", i + 1, len(retrain_dates), tag,
                    dt.strftime("%Y-%m-%d"))

        label_cutoff = _label_cutoff(panel, dt)
        train_mask = (panel["date"] < label_cutoff) & panel[label_col].notna()
        train_df = panel.loc[train_mask]

        if len(train_df) < 500:
            continue

        # Train ensemble of models
        rankers = []
        for seed in range(n_models):
            params = {"seed": seed + 42, "bagging_seed": seed + 42,
                      "feature_fraction_seed": seed + 42}
            if lgb_params:
                params.update(lgb_params)
            ranker = StockRanker(
                feature_cols=extended_features,
                objective="regression",
                label_col=label_col,
                lgb_params=params,
            )
            try:
                metrics = ranker.train_fast(train_df)
                rankers.append(ranker)
            except Exception as e:
                logger.error("  Model {} failed: {}", seed, e)

        if not rankers:
            continue

        if i % 10 == 0:
            logger.info("  {} models trained, IC={:.3f}",
                        len(rankers), metrics.get("ic", 0))

        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)
        if today_valid.empty:
            continue

        # Average scores across models
        if len(rankers) == 1:
            scores = rankers[0].predict(today_valid)
        else:
            all_scores = np.stack([r.predict(today_valid) for r in rankers])
            scores = all_scores.mean(axis=0)

        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
            })

    diag_df = pd.DataFrame(records)
    diag_df = diag_df.dropna(subset=["fwd_ret"])
    logger.info("[{}] {} scored rows", tag, len(diag_df))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(cache_path, index=False)
    return diag_df


def run_scoring_loop_regime_combo(
    force: bool = False,
    tag: str = "regime_combo",
    lgb_params: Optional[Dict] = None,
    n_models: int = 1,
) -> pd.DataFrame:
    """Combine regime-interaction AND residual-alpha features.

    Interaction features: capture stock×market-state interactions.
    Residual features: capture stock-specific alpha after removing market exposure.
    Together: the model can condition on market regime while also seeing
    individual stock quality beyond market moves.
    """
    cache_path = CACHE_DIR / f"diagnostics_{tag}.parquet"
    if not force and cache_path.exists():
        logger.info("Loading cached {} diagnostics", tag)
        return pd.read_parquet(cache_path)

    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found.")
        sys.exit(1)

    logger.info("[{}] Loading factor panel + interaction + residual features...", tag)
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    panel, interact_cols = _add_regime_interaction_features(panel)
    panel, alpha_cols = _add_residual_alpha_features(panel)
    extended_features = FACTOR_COLUMNS + interact_cols + alpha_cols

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("[{}] {} retrain dates, {} features (base {} + {} interact + {} alpha)",
                tag, len(retrain_dates), len(extended_features),
                len(FACTOR_COLUMNS), len(interact_cols), len(alpha_cols))

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] {}: retrain {}", i + 1, len(retrain_dates), tag,
                    dt.strftime("%Y-%m-%d"))

        label_cutoff = _label_cutoff(panel, dt)
        train_mask = (panel["date"] < label_cutoff) & panel["fwd_ret"].notna()
        train_df = panel.loc[train_mask]

        if len(train_df) < 500:
            continue

        rankers = []
        for seed in range(n_models):
            params = {"seed": seed + 42, "bagging_seed": seed + 42,
                      "feature_fraction_seed": seed + 42}
            if lgb_params:
                params.update(lgb_params)
            ranker = StockRanker(
                feature_cols=extended_features,
                lgb_params=params,
            )
            try:
                metrics = ranker.train_fast(train_df)
                rankers.append(ranker)
            except Exception as e:
                logger.error("  Model {} failed: {}", seed, e)

        if not rankers:
            continue

        if i % 10 == 0:
            logger.info("  {} models trained, IC={:.3f}",
                        len(rankers), metrics.get("ic", 0))

        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)
        if today_valid.empty:
            continue

        if len(rankers) == 1:
            scores = rankers[0].predict(today_valid)
        else:
            all_scores = np.stack([r.predict(today_valid) for r in rankers])
            scores = all_scores.mean(axis=0)

        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
            })

    diag_df = pd.DataFrame(records)
    diag_df = diag_df.dropna(subset=["fwd_ret"])
    logger.info("[{}] {} scored rows", tag, len(diag_df))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(cache_path, index=False)
    return diag_df


def run_scoring_loop_regime_split(
    force: bool = False,
    tag: str = "regime_split",
    lgb_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """Train separate models for bull/bear regimes and dispatch at scoring time.

    The anti-ZZ500 bias (r=-0.509) suggests the model learns the wrong patterns
    when trained on mixed-regime data. By splitting, each model can specialize:
    - Bull model: learn what outperforms when market is rising
    - Bear model: learn what outperforms when market is falling
    """
    cache_path = CACHE_DIR / f"diagnostics_{tag}.parquet"
    if not force and cache_path.exists():
        logger.info("Loading cached {} diagnostics", tag)
        return pd.read_parquet(cache_path)

    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found.")
        sys.exit(1)

    import akshare as ak

    logger.info("[{}] Loading factor panel + computing market regime...", tag)
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    # Compute ZZ500 20d momentum for regime detection
    idx_df = ak.stock_zh_index_daily(symbol="sh000905")
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_df = idx_df.sort_values("date").set_index("date")
    idx_mom20 = idx_df["close"].pct_change(20)

    panel_dates = panel["date"].drop_duplicates().sort_values()
    mkt_mom_map = idx_mom20.reindex(panel_dates, method="ffill").to_dict()
    panel["_mkt_mom20"] = panel["date"].map(mkt_mom_map).astype(float)

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("[{}] {} retrain dates", tag, len(retrain_dates))

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []
    bull_ranker: Optional[StockRanker] = None
    bear_ranker: Optional[StockRanker] = None
    neutral_ranker: Optional[StockRanker] = None

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] {}: retrain {}", i + 1, len(retrain_dates), tag,
                    dt.strftime("%Y-%m-%d"))

        label_cutoff = _label_cutoff(panel, dt)
        train_mask = (panel["date"] < label_cutoff) & panel["fwd_ret"].notna()
        train_all = panel.loc[train_mask]

        if len(train_all) < 500:
            continue

        # Split training data by regime
        bull_mask = train_all["_mkt_mom20"] > 0.02   # market up >2%
        bear_mask = train_all["_mkt_mom20"] < -0.02  # market down >2%
        neutral_mask = ~bull_mask & ~bear_mask

        params = lgb_params or {}

        # Train bull model (on bull-market data only)
        train_bull = train_all.loc[bull_mask]
        if len(train_bull) >= 500:
            r = StockRanker(feature_cols=FACTOR_COLUMNS, lgb_params=params)
            try:
                r.train_fast(train_bull)
                bull_ranker = r
            except Exception:
                pass

        # Train bear model
        train_bear = train_all.loc[bear_mask]
        if len(train_bear) >= 500:
            r = StockRanker(feature_cols=FACTOR_COLUMNS, lgb_params=params)
            try:
                r.train_fast(train_bear)
                bear_ranker = r
            except Exception:
                pass

        # Train neutral model (fallback)
        r = StockRanker(feature_cols=FACTOR_COLUMNS, lgb_params=params)
        try:
            r.train_fast(train_all)
            neutral_ranker = r
        except Exception:
            continue

        # Score current date
        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)
        if today_valid.empty:
            continue

        # Detect current regime
        current_mom = today_valid["_mkt_mom20"].iloc[0] if len(today_valid) > 0 else 0.0

        if current_mom > 0.02 and bull_ranker is not None:
            scores = bull_ranker.predict(today_valid)
            regime = "bull"
        elif current_mom < -0.02 and bear_ranker is not None:
            scores = bear_ranker.predict(today_valid)
            regime = "bear"
        else:
            scores = neutral_ranker.predict(today_valid)
            regime = "neutral"

        if i % 10 == 0:
            logger.info("  Regime={}, mkt_mom={:.3f}", regime, current_mom)

        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
            })

    diag_df = pd.DataFrame(records)
    diag_df = diag_df.dropna(subset=["fwd_ret"])
    logger.info("[{}] {} scored rows", tag, len(diag_df))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(cache_path, index=False)
    return diag_df


def run_scoring_loop_with_context(
    force: bool = False,
    tag: str = "context",
) -> pd.DataFrame:
    """Scoring loop with market-context features (ZZ500 momentum + stock beta)."""
    cache_path = CACHE_DIR / f"diagnostics_{tag}.parquet"
    if not force and cache_path.exists():
        logger.info("Loading cached {} diagnostics", tag)
        return pd.read_parquet(cache_path)

    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found.")
        sys.exit(1)

    logger.info("[{}] Loading factor panel + computing market context...", tag)
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    panel, ctx_cols = _add_market_context_features(panel)
    extended_features = FACTOR_COLUMNS + ctx_cols

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("[{}] {} retrain dates, {} features (base {} + {} context)",
                tag, len(retrain_dates), len(extended_features),
                len(FACTOR_COLUMNS), len(ctx_cols))

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []
    current_ranker: Optional[StockRanker] = None

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] {}: retrain {}", i + 1, len(retrain_dates), tag,
                    dt.strftime("%Y-%m-%d"))

        label_cutoff = _label_cutoff(panel, dt)
        train_mask = (panel["date"] < label_cutoff) & panel["fwd_ret"].notna()
        train_df = panel.loc[train_mask]

        if len(train_df) < 500:
            continue

        ranker = StockRanker(
            feature_cols=extended_features,
            objective="regression",
            label_col="fwd_ret",
        )
        try:
            metrics = ranker.train_fast(train_df)
            logger.info("  Train: {} rows, IC={:.3f}",
                        len(train_df), metrics.get("ic", 0))
            current_ranker = ranker
        except Exception as e:
            logger.error("  Training failed: {}", e)
            continue

        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)
        if today_valid.empty:
            continue

        scores = current_ranker.predict(today_valid)
        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
            })

    diag_df = pd.DataFrame(records)
    diag_df = diag_df.dropna(subset=["fwd_ret"])
    logger.info("[{}] {} scored rows", tag, len(diag_df))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(cache_path, index=False)
    return diag_df


def run_scoring_loop_tuned(
    force: bool = False,
    tag: str = "tuned",
    lgb_params: Optional[Dict] = None,
    extreme_pctile: float = 0.0,
    label_col: str = "fwd_ret",
) -> pd.DataFrame:
    """Scoring loop with custom LGB params and optional extreme-training.

    Parameters
    ----------
    lgb_params : dict
        Override LightGBM hyperparameters (merged into defaults).
    extreme_pctile : float
        If > 0, train on only the top/bottom this fraction per date.
        E.g. 0.3 = keep top-30% and bottom-30% by fwd_ret, discard middle 40%.
    """
    cache_path = CACHE_DIR / f"diagnostics_{tag}.parquet"
    if not force and cache_path.exists():
        logger.info("Loading cached {} diagnostics", tag)
        return pd.read_parquet(cache_path)

    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found.")
        sys.exit(1)

    logger.info("[{}] Loading factor panel...", tag)
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    # Add excess return if needed
    if label_col == EXCESS_LABEL:
        panel = add_excess_ret(panel, horizon=HORIZON)

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("[{}] {} retrain dates, lgb_params={}, extreme_pctile={}, label={}",
                tag, len(retrain_dates), lgb_params, extreme_pctile, label_col)

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []
    current_ranker: Optional[StockRanker] = None

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] {}: retrain {}", i + 1, len(retrain_dates), tag,
                    dt.strftime("%Y-%m-%d"))

        label_cutoff = _label_cutoff(panel, dt)
        train_mask = (panel["date"] < label_cutoff) & panel[label_col].notna()
        train_df = panel.loc[train_mask]

        # Extreme training: keep only top/bottom pctile per date
        if extreme_pctile > 0:
            def _keep_extremes(g):
                lo = g[label_col].quantile(extreme_pctile)
                hi = g[label_col].quantile(1 - extreme_pctile)
                return g[(g[label_col] <= lo) | (g[label_col] >= hi)]
            train_df = train_df.groupby("date", group_keys=False).apply(_keep_extremes)

        if len(train_df) < 500:
            continue

        ranker = StockRanker(
            feature_cols=FACTOR_COLUMNS,
            objective="regression",
            label_col=label_col,
            lgb_params=lgb_params,
        )
        try:
            metrics = ranker.train_fast(train_df)
            logger.info("  Train: {} rows, IC={:.3f}",
                        len(train_df), metrics.get("ic", 0))
            current_ranker = ranker
        except Exception as e:
            logger.error("  Training failed: {}", e)
            continue

        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)
        if today_valid.empty:
            continue

        scores = current_ranker.predict(today_valid)
        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
            })

    diag_df = pd.DataFrame(records)
    diag_df = diag_df.dropna(subset=["fwd_ret"])
    logger.info("[{}] {} scored rows", tag, len(diag_df))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(cache_path, index=False)
    return diag_df


def run_scoring_loop_ensemble(
    force: bool = False,
    tag: str = "ensemble",
    n_models: int = 5,
    lgb_params: Optional[Dict] = None,
    winsorize_pct: float = 0.0,
    top_n_features: int = 0,
    label_col: str = "fwd_ret",
) -> pd.DataFrame:
    """Score ensemble: average multiple models with different seeds.

    Parameters
    ----------
    n_models : int
        Number of models to average.
    winsorize_pct : float
        If > 0, clip fwd_ret to [pct, 1-pct] percentiles per date before training.
    top_n_features : int
        If > 0, first train a baseline model, then retrain using only
        the top-N most important features.
    """
    cache_path = CACHE_DIR / f"diagnostics_{tag}.parquet"
    if not force and cache_path.exists():
        logger.info("Loading cached {} diagnostics", tag)
        return pd.read_parquet(cache_path)

    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found.")
        sys.exit(1)

    logger.info("[{}] Loading factor panel...", tag)
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    # Add excess return if needed
    if label_col == EXCESS_LABEL:
        panel = add_excess_ret(panel, horizon=HORIZON)

    feature_cols = list(FACTOR_COLUMNS)

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("[{}] {} retrain dates, n_models={}, winsorize={}, top_n_features={}, label={}",
                tag, len(retrain_dates), n_models, winsorize_pct, top_n_features, label_col)

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] {}: retrain {}", i + 1, len(retrain_dates), tag,
                    dt.strftime("%Y-%m-%d"))

        label_cutoff = _label_cutoff(panel, dt)
        train_mask = (panel["date"] < label_cutoff) & panel[label_col].notna()
        train_df = panel.loc[train_mask].copy()

        if len(train_df) < 500:
            continue

        # Winsorize target per date
        if winsorize_pct > 0:
            def _winsorize(g):
                lo = g[label_col].quantile(winsorize_pct)
                hi = g[label_col].quantile(1 - winsorize_pct)
                g[label_col] = g[label_col].clip(lo, hi)
                return g
            train_df = train_df.groupby("date", group_keys=False).apply(_winsorize)

        # Feature pruning: train baseline, get top-N features
        use_features = feature_cols
        if top_n_features > 0 and i == 0:
            base_ranker = StockRanker(feature_cols=feature_cols, lgb_params=lgb_params)
            base_ranker.train_fast(train_df)
            if base_ranker.model:
                imp = base_ranker.model.feature_importance(importance_type="gain")
                feat_names = base_ranker.model.feature_name()
                top_idx = np.argsort(imp)[-top_n_features:]
                use_features = [feat_names[j] for j in top_idx]
                logger.info("[{}] Feature pruning: {} → {}", tag, len(feature_cols), len(use_features))

        # Train ensemble
        rankers = []
        for seed in range(n_models):
            params = {"seed": seed + 42, "bagging_seed": seed + 42,
                      "feature_fraction_seed": seed + 42}
            if lgb_params:
                params.update(lgb_params)
            ranker = StockRanker(
                feature_cols=use_features,
                objective="regression",
                label_col=label_col,
                lgb_params=params,
            )
            try:
                metrics = ranker.train_fast(train_df)
                rankers.append(ranker)
            except Exception as e:
                logger.error("  Model {} failed: {}", seed, e)

        if not rankers:
            continue

        if i % 10 == 0:
            logger.info("  Ensemble: {} models trained", len(rankers))

        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)
        if today_valid.empty:
            continue

        # Average scores across models
        all_scores = np.stack([r.predict(today_valid) for r in rankers])
        scores = all_scores.mean(axis=0)

        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
            })

    diag_df = pd.DataFrame(records)
    diag_df = diag_df.dropna(subset=["fwd_ret"])
    logger.info("[{}] {} scored rows", tag, len(diag_df))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(cache_path, index=False)
    return diag_df


def run_scoring_loop_xscore(
    force: bool = False,
    tag: str = "xscore",
    lgb_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """Scoring loop with per-date cross-sectional z-score normalization.

    Each feature is standardized within each date (z-score) so the model
    sees purely relative-to-peers values, removing market-regime confound.
    """
    cache_path = CACHE_DIR / f"diagnostics_{tag}.parquet"
    if not force and cache_path.exists():
        logger.info("Loading cached {} diagnostics", tag)
        return pd.read_parquet(cache_path)

    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found.")
        sys.exit(1)

    logger.info("[{}] Loading factor panel + cross-sectional z-scoring...", tag)
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    # Z-score normalize each factor within each date
    for col in FACTOR_COLUMNS:
        if col in panel.columns:
            panel[col] = panel.groupby("date")[col].transform(
                lambda s: (s - s.mean()) / max(s.std(), 1e-8)
            )

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("[{}] {} retrain dates, z-scored features", tag, len(retrain_dates))

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []
    current_ranker: Optional[StockRanker] = None

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] {}: retrain {}", i + 1, len(retrain_dates), tag,
                    dt.strftime("%Y-%m-%d"))

        label_cutoff = _label_cutoff(panel, dt)
        train_mask = (panel["date"] < label_cutoff) & panel["fwd_ret"].notna()
        train_df = panel.loc[train_mask]

        if len(train_df) < 500:
            continue

        ranker = StockRanker(
            feature_cols=FACTOR_COLUMNS,
            objective="regression",
            label_col="fwd_ret",
            lgb_params=lgb_params,
        )
        try:
            metrics = ranker.train_fast(train_df)
            logger.info("  Train: {} rows, IC={:.3f}",
                        len(train_df), metrics.get("ic", 0))
            current_ranker = ranker
        except Exception as e:
            logger.error("  Training failed: {}", e)
            continue

        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)
        if today_valid.empty:
            continue

        scores = current_ranker.predict(today_valid)
        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
            })

    diag_df = pd.DataFrame(records)
    diag_df = diag_df.dropna(subset=["fwd_ret"])
    logger.info("[{}] {} scored rows", tag, len(diag_df))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(cache_path, index=False)
    return diag_df


DIAG_CACHE_2S = CACHE_DIR / "diagnostics_twostage.parquet"


def run_scoring_loop_twostage(force: bool = False) -> pd.DataFrame:
    """Two-stage scoring loop: Stage1 (regression/fwd_ret) → Stage2 (excess_ret).

    Same structure as run_scoring_loop() but uses TwoStageRanker.
    """
    if not force and DIAG_CACHE_2S.exists():
        logger.info("Loading cached two-stage diagnostics from {}", DIAG_CACHE_2S)
        return pd.read_parquet(DIAG_CACHE_2S)

    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found. Run walk_forward_backtest.py first.")
        sys.exit(1)

    logger.info("Loading factor panel for two-stage scoring...")
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    # Add excess return and filter universe
    panel = add_excess_ret(panel, horizon=HORIZON)
    panel = filter_universe(panel)

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("{} retrain dates (two-stage mode)", len(retrain_dates))

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []
    current_ranker: Optional[TwoStageRanker] = None

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] Two-stage retrain & score: {}", i + 1, len(retrain_dates),
                    dt.strftime("%Y-%m-%d"))

        # --- Retrain ---
        label_cutoff = _label_cutoff(panel, dt)
        train_mask = (
            (panel["date"] < label_cutoff)
            & panel["fwd_ret"].notna()
            & panel[EXCESS_LABEL].notna()
        )
        train_df = panel.loc[train_mask]

        if len(train_df) < 500:
            logger.warning("  Too few rows ({}), skipping", len(train_df))
            continue

        ranker = TwoStageRanker(top_pct=0.2, stage2_objective="lambdarank",
                                        stage2_label=EXCESS_LABEL)
        try:
            metrics = ranker.train_fast(train_df)
            m1, m2 = metrics["stage1"], metrics["stage2"]
            logger.info("  S1: IC={:.3f}  S2: IC={:.3f}, HitRate@K={:.3f}",
                        m1.get("ic", 0), m2.get("ic", 0), m2.get("hit_rate_at_k", 0))
            current_ranker = ranker
        except Exception as e:
            logger.error("  Two-stage training failed: {}", e)
            continue

        # --- Score all stocks on this date ---
        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)

        if today_valid.empty:
            continue

        scores = current_ranker.predict(today_valid)
        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
            })

    diag_df = pd.DataFrame(records)
    before = len(diag_df)
    diag_df = diag_df.dropna(subset=["fwd_ret"])
    # Remove -inf scores (stocks that didn't pass Stage-1)
    diag_df = diag_df[np.isfinite(diag_df["pred_score"])].reset_index(drop=True)
    logger.info("Two-stage diagnostics: {} scored rows (top 20%% only)",
                len(diag_df))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(DIAG_CACHE_2S, index=False)
    logger.info("Cached to {}", DIAG_CACHE_2S)

    return diag_df


# ──────────────────────────────────────────────────────────────────────
# Blend scoring loop (walk-forward with BlendRanker)
# ──────────────────────────────────────────────────────────────────────

def run_scoring_loop_blend(
    force: bool = False,
    tag: str = "blend_best",
    weight_primary: float = 0.80,
    extreme_pctile: float = 0.30,
    primary_label: str = EXCESS_LABEL,
) -> pd.DataFrame:
    """Walk-forward scoring with BlendRanker (excess_ret + extreme30).

    Trains two models per rebalance date:
      1. Primary: regression on excess_ret (all samples)
      2. Extreme: regression on fwd_ret (top/bottom extreme_pctile only)

    Blends rank-normalized scores: weight_primary * R(primary) + (1 - weight_primary) * R(extreme).
    """
    from mp.ml.model import BlendRanker

    cache_path = CACHE_DIR / f"diagnostics_{tag}.parquet"
    if not force and cache_path.exists():
        logger.info("Loading cached {} diagnostics", tag)
        return pd.read_parquet(cache_path)

    if not FACTORS_CACHE.exists():
        logger.error("Factor cache not found.")
        sys.exit(1)

    logger.info("[{}] Loading factor panel...", tag)
    panel = pd.read_parquet(FACTORS_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    # Add excess return
    if primary_label == EXCESS_LABEL:
        panel = add_excess_ret(panel, horizon=HORIZON)

    retrain_dates = _get_monthly_retrain_dates(panel)
    logger.info("[{}] {} retrain dates, weight_primary={}, extreme_pctile={}",
                tag, len(retrain_dates), weight_primary, extreme_pctile)

    core_cols = TECHNICAL_COLUMNS[:13]
    records: List[dict] = []

    for i, dt in enumerate(retrain_dates):
        logger.info("[{}/{}] {}: retrain {}", i + 1, len(retrain_dates), tag,
                    dt.strftime("%Y-%m-%d"))

        label_cutoff = _label_cutoff(panel, dt)
        train_mask = (
            (panel["date"] < label_cutoff)
            & panel["fwd_ret"].notna()
            & panel[primary_label].notna()
        )
        train_df = panel.loc[train_mask]

        if len(train_df) < 500:
            continue

        ranker = BlendRanker(
            weight_primary=weight_primary,
            extreme_pctile=extreme_pctile,
            primary_label=primary_label,
        )
        try:
            metrics = ranker.train_fast(train_df)
            logger.info("  Primary IC={:.3f}, Extreme IC={:.3f}",
                        metrics["primary"].get("ic", 0),
                        metrics["extreme"].get("ic", 0))
        except Exception as e:
            logger.error("  Training failed: {}", e)
            continue

        today_df = panel.loc[panel["date"] == dt].copy()
        today_valid = today_df.dropna(subset=core_cols)
        if today_valid.empty:
            continue

        scores = ranker.predict(today_valid)
        for idx_row, (_, row) in enumerate(today_valid.iterrows()):
            fwd = row["fwd_ret"] if pd.notna(row.get("fwd_ret")) else np.nan
            records.append({
                "score_date": dt,
                "code": row["code"],
                "pred_score": float(scores[idx_row]),
                "fwd_ret": float(fwd),
            })

    diag_df = pd.DataFrame(records)
    diag_df = diag_df.dropna(subset=["fwd_ret"])
    logger.info("[{}] {} scored rows", tag, len(diag_df))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_parquet(cache_path, index=False)
    return diag_df


def run_posthoc_blend(
    tag_a: str = "excess_ret",
    tag_b: str = "extreme30",
    weight_a: float = 0.80,
    output_tag: str = "blend_posthoc",
    force: bool = False,
) -> pd.DataFrame:
    """Post-hoc rank-blend of two cached diagnostics files.

    Loads diagnostics_{tag_a}.parquet and diagnostics_{tag_b}.parquet,
    rank-normalizes pred_score per date for each, then blends.
    """
    cache_path = CACHE_DIR / f"diagnostics_{output_tag}.parquet"
    if not force and cache_path.exists():
        logger.info("Loading cached {} blend", output_tag)
        return pd.read_parquet(cache_path)

    path_a = CACHE_DIR / f"diagnostics_{tag_a}.parquet"
    path_b = CACHE_DIR / f"diagnostics_{tag_b}.parquet"
    if not path_a.exists() or not path_b.exists():
        logger.error("Need both {} and {} to exist. Run those experiments first.",
                      path_a, path_b)
        sys.exit(1)

    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)

    # Merge on (score_date, code)
    merged = df_a.merge(
        df_b[["score_date", "code", "pred_score"]],
        on=["score_date", "code"], suffixes=("_a", "_b"),
    )
    logger.info("[{}] Merged {}/{} rows from {} and {}",
                output_tag, len(merged), len(df_a), tag_a, tag_b)

    # Rank-normalize per date and blend
    merged["rank_a"] = merged.groupby("score_date")["pred_score_a"].rank(pct=True)
    merged["rank_b"] = merged.groupby("score_date")["pred_score_b"].rank(pct=True)
    merged["pred_score"] = weight_a * merged["rank_a"] + (1 - weight_a) * merged["rank_b"]

    result = merged[["score_date", "code", "pred_score", "fwd_ret"]].copy()
    result.to_parquet(cache_path, index=False)
    logger.info("[{}] Saved {} rows", output_tag, len(result))
    return result


# ──────────────────────────────────────────────────────────────────────
# Analysis 1: Decile Returns (分层收益)
# ──────────────────────────────────────────────────────────────────────

def analyze_deciles(diag_df: pd.DataFrame, n_deciles: int = 10
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-month decile analysis on prediction scores.

    Returns (per_month_df, aggregate_df).
    """
    logger.info("── Analysis 1: Decile Returns ──")

    diag_df = diag_df.copy()
    diag_df["month"] = diag_df["score_date"].dt.to_period("M")

    per_month_rows = []
    for month, grp in diag_df.groupby("month"):
        if len(grp) < n_deciles * 2:
            continue
        try:
            grp = grp.copy()
            # Labels: D1 = highest score (long group), D10 = lowest score (short group)
            grp["decile"] = pd.qcut(grp["pred_score"], n_deciles,
                                     labels=range(n_deciles, 0, -1),
                                     duplicates="drop")
        except ValueError:
            continue

        for d, dgrp in grp.groupby("decile", observed=True):
            per_month_rows.append({
                "month": str(month),
                "decile": int(d),
                "mean_ret": dgrp["fwd_ret"].mean(),
                "n_stocks": len(dgrp),
            })

    per_month_df = pd.DataFrame(per_month_rows)
    if per_month_df.empty:
        logger.warning("No decile data produced")
        return per_month_df, pd.DataFrame()

    # Aggregate: average return per decile across all months
    agg = per_month_df.groupby("decile").agg(
        mean_ret=("mean_ret", "mean"),
        std_ret=("mean_ret", "std"),
        n_months=("mean_ret", "count"),
    ).reset_index()

    # Monotonicity check per month: D1 (best) should beat D10 (worst)
    months = per_month_df["month"].unique()
    monotone_count = 0
    for m in months:
        m_data = per_month_df[per_month_df["month"] == m].sort_values("decile")
        rets = m_data["mean_ret"].values
        if len(rets) >= n_deciles:
            # D1 is first row (sorted ascending), should have highest return
            if rets[0] > rets[-1]:
                monotone_count += 1

    monotone_pct = monotone_count / len(months) * 100 if len(months) > 0 else 0
    agg.attrs["monotone_pct"] = monotone_pct
    agg.attrs["n_months"] = len(months)

    # Long-short spread
    d1_mean = agg.loc[agg["decile"] == 1, "mean_ret"].values
    d10_mean = agg.loc[agg["decile"] == n_deciles, "mean_ret"].values
    spread = (d1_mean[0] - d10_mean[0]) if len(d1_mean) > 0 and len(d10_mean) > 0 else 0
    agg.attrs["ls_spread"] = spread

    logger.info("Decile analysis: {} months, D1-D10 monotone {:.1f}%, spread {:.2%}",
                len(months), monotone_pct, spread)
    return per_month_df, agg


# ──────────────────────────────────────────────────────────────────────
# Analysis 2: Top-K Hit Rate (命中率)
# ──────────────────────────────────────────────────────────────────────

def _get_zz500_monthly_returns() -> Dict[str, float]:
    """Fetch ZZ500 index returns aligned to scoring months."""
    try:
        import akshare as ak
        idx_df = ak.stock_zh_index_daily(symbol="sh000905")
        idx_df["date"] = pd.to_datetime(idx_df["date"])
        idx_df = idx_df.sort_values("date")
        idx_df["ret_20d"] = idx_df["close"].pct_change(HORIZON)
        # Map each month's first trading day to its 20d forward return
        idx_df["month"] = idx_df["date"].dt.to_period("M")
        monthly = idx_df.groupby("month").first()
        return {str(m): r for m, r in monthly["ret_20d"].items() if pd.notna(r)}
    except Exception as e:
        logger.warning("Failed to fetch ZZ500 index: {}. Using 0 as benchmark.", e)
        return {}


def analyze_hit_rate(diag_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Top-K hit rate: fraction of top-K predictions that beat benchmark."""
    logger.info("── Analysis 2: Top-K Hit Rate (K={}) ──", top_k)

    zz500_rets = _get_zz500_monthly_returns()

    diag_df = diag_df.copy()
    diag_df["month"] = diag_df["score_date"].dt.to_period("M")

    rows = []
    for month, grp in diag_df.groupby("month"):
        top = grp.nlargest(top_k, "pred_score")
        if len(top) < top_k:
            continue

        month_str = str(month)
        bench_ret = zz500_rets.get(month_str, 0.0)
        abs_hits = (top["fwd_ret"] > 0).sum()
        rel_hits = (top["fwd_ret"] > bench_ret).sum()
        mean_ret = top["fwd_ret"].mean()

        rows.append({
            "month": month_str,
            "mean_ret": mean_ret,
            "bench_ret": bench_ret,
            "abs_hit_rate": abs_hits / len(top),
            "rel_hit_rate": rel_hits / len(top),
            "n": len(top),
        })

    hit_df = pd.DataFrame(rows)
    if not hit_df.empty:
        hit_df.attrs["abs_hit_mean"] = hit_df["abs_hit_rate"].mean()
        hit_df.attrs["rel_hit_mean"] = hit_df["rel_hit_rate"].mean()
        hit_df.attrs["mean_excess"] = (hit_df["mean_ret"] - hit_df["bench_ret"]).mean()
        logger.info("Hit rate: abs={:.1%}, rel={:.1%}, mean excess={:.2%}",
                    hit_df.attrs["abs_hit_mean"],
                    hit_df.attrs["rel_hit_mean"],
                    hit_df.attrs["mean_excess"])
    return hit_df


# ──────────────────────────────────────────────────────────────────────
# Analysis 3: Rolling IC by Regime (滚动稳定性)
# ──────────────────────────────────────────────────────────────────────

def analyze_rolling_ic(diag_df: pd.DataFrame
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-sectional Rank IC per scoring date, summarized by regime.

    Returns (monthly_ic_df, regime_summary_df).
    """
    logger.info("── Analysis 3: Rolling IC by Regime ──")

    ic_rows = []
    for dt, grp in diag_df.groupby("score_date"):
        if len(grp) < 30:
            continue
        ic, _ = spearmanr(grp["pred_score"], grp["fwd_ret"])
        if np.isnan(ic):
            continue
        ic_rows.append({
            "score_date": dt,
            "ic": float(ic),
            "n_stocks": len(grp),
            "regime": _classify_regime(pd.Timestamp(dt)),
        })

    ic_df = pd.DataFrame(ic_rows)
    if ic_df.empty:
        return ic_df, pd.DataFrame()

    # Overall
    ic_df.attrs["ic_mean"] = ic_df["ic"].mean()
    ic_df.attrs["ic_std"] = ic_df["ic"].std()
    ic_df.attrs["icir"] = ic_df["ic"].mean() / ic_df["ic"].std() if ic_df["ic"].std() > 0 else 0
    ic_df.attrs["ic_pos_pct"] = (ic_df["ic"] > 0).mean()

    # By regime
    regime_rows = []
    for regime, rgrp in ic_df.groupby("regime"):
        ic_mean = rgrp["ic"].mean()
        ic_std = rgrp["ic"].std()
        regime_rows.append({
            "regime": regime,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": ic_mean / ic_std if ic_std > 0 else 0,
            "ic_pos_pct": (rgrp["ic"] > 0).mean(),
            "n_dates": len(rgrp),
        })

    regime_df = pd.DataFrame(regime_rows)

    logger.info("IC: mean={:.4f}, std={:.4f}, ICIR={:.2f}, IC>0={:.1%}",
                ic_df.attrs["ic_mean"], ic_df.attrs["ic_std"],
                ic_df.attrs["icir"], ic_df.attrs["ic_pos_pct"])
    for _, row in regime_df.iterrows():
        logger.info("  {}: IC={:.4f}, ICIR={:.2f}, IC>0={:.1%}, n={}",
                    row["regime"], row["ic_mean"], row["icir"],
                    row["ic_pos_pct"], row["n_dates"])

    return ic_df, regime_df


# ──────────────────────────────────────────────────────────────────────
# Analysis 4: Long-Short Spread (多空价差)
# ──────────────────────────────────────────────────────────────────────

def analyze_long_short(diag_df: pd.DataFrame, n_deciles: int = 10
                       ) -> pd.DataFrame:
    """Monthly long-short spread: D1 mean return - D(last) mean return."""
    logger.info("── Analysis 4: Long-Short Spread ──")

    diag_df = diag_df.copy()
    diag_df["month"] = diag_df["score_date"].dt.to_period("M")

    rows = []
    for month, grp in diag_df.groupby("month"):
        if len(grp) < n_deciles * 2:
            continue
        try:
            grp = grp.copy()
            # D1 = highest score (long), D10 = lowest score (short)
            grp["decile"] = pd.qcut(grp["pred_score"], n_deciles,
                                     labels=range(n_deciles, 0, -1),
                                     duplicates="drop")
        except ValueError:
            continue

        d1 = grp[grp["decile"] == 1]["fwd_ret"].mean()  # highest scores
        d_last = grp[grp["decile"] == n_deciles]["fwd_ret"].mean()  # lowest scores
        spread = d1 - d_last

        rows.append({
            "month": str(month),
            "d1_ret": d1,
            "d10_ret": d_last,
            "spread": spread,
            "regime": _classify_regime(pd.Timestamp(grp["score_date"].iloc[0])),
        })

    ls_df = pd.DataFrame(rows)
    if ls_df.empty:
        return ls_df

    # Cumulative spread
    ls_df["cum_spread"] = ls_df["spread"].cumsum()
    ls_df.attrs["mean_spread"] = ls_df["spread"].mean()
    ls_df.attrs["spread_pos_pct"] = (ls_df["spread"] > 0).mean()
    ls_df.attrs["annual_spread"] = ls_df["spread"].mean() * 12

    # By regime
    regime_spreads = {}
    for regime, rgrp in ls_df.groupby("regime"):
        regime_spreads[regime] = {
            "mean_spread": rgrp["spread"].mean(),
            "pos_pct": (rgrp["spread"] > 0).mean(),
            "n": len(rgrp),
        }
    ls_df.attrs["regime_spreads"] = regime_spreads

    logger.info("L/S spread: mean={:.2%}, positive={:.1%}, annualized={:.2%}",
                ls_df.attrs["mean_spread"],
                ls_df.attrs["spread_pos_pct"],
                ls_df.attrs["annual_spread"])
    for regime, stats in regime_spreads.items():
        logger.info("  {}: spread={:.2%}, pos={:.1%}, n={}",
                    regime, stats["mean_spread"], stats["pos_pct"], stats["n"])

    return ls_df


# ──────────────────────────────────────────────────────────────────────
# Health Check Verdict
# ──────────────────────────────────────────────────────────────────────

def _grade(value: float, pass_th: float, warn_th: float, higher_is_better: bool = True) -> str:
    """Return PASS / WARN / FAIL based on thresholds."""
    if higher_is_better:
        if value >= pass_th:
            return "PASS"
        elif value >= warn_th:
            return "WARN"
        return "FAIL"
    else:
        if value <= pass_th:
            return "PASS"
        elif value <= warn_th:
            return "WARN"
        return "FAIL"


def health_check(decile_agg: pd.DataFrame, hit_df: pd.DataFrame,
                 ic_df: pd.DataFrame, regime_df: pd.DataFrame,
                 ls_df: pd.DataFrame) -> List[dict]:
    """Automated health check across all analyses."""
    checks = []

    # 1. Decile monotonicity
    mono_pct = decile_agg.attrs.get("monotone_pct", 0)
    checks.append({
        "metric": "分层单调性 (D1>D10 months)",
        "value": f"{mono_pct:.1f}%",
        "grade": _grade(mono_pct, 60, 40),
    })

    # 2. Top-K hit rate
    abs_hit = hit_df.attrs.get("abs_hit_mean", 0) * 100
    rel_hit = hit_df.attrs.get("rel_hit_mean", 0) * 100
    checks.append({
        "metric": f"Top-K 绝对命中率 (fwd_ret>0)",
        "value": f"{abs_hit:.1f}%",
        "grade": _grade(abs_hit, 55, 50),
    })
    checks.append({
        "metric": f"Top-K 相对命中率 (>ZZ500)",
        "value": f"{rel_hit:.1f}%",
        "grade": _grade(rel_hit, 55, 50),
    })

    # 3. Cross-sectional IC
    ic_mean = ic_df.attrs.get("ic_mean", 0)
    checks.append({
        "metric": "截面 IC 均值",
        "value": f"{ic_mean:.4f}",
        "grade": _grade(ic_mean, 0.03, 0.02),
    })

    # 4. IC in all regimes > 0.03
    if not regime_df.empty:
        bear_rows = regime_df[regime_df["regime"].str.contains("熊市")]
        bear_ic = bear_rows["ic_mean"].values[0] if len(bear_rows) > 0 else 0
        checks.append({
            "metric": "熊市 IC",
            "value": f"{bear_ic:.4f}",
            "grade": _grade(bear_ic, 0.01, 0, higher_is_better=True),
        })

        min_ic = regime_df["ic_mean"].min()
        checks.append({
            "metric": "最弱市场状态 IC",
            "value": f"{min_ic:.4f}",
            "grade": _grade(min_ic, 0.03, 0.02),
        })

    # 5. Long-short spread positive rate
    spread_pos = ls_df.attrs.get("spread_pos_pct", 0) * 100
    checks.append({
        "metric": "多空价差 >0 月份占比",
        "value": f"{spread_pos:.1f}%",
        "grade": _grade(spread_pos, 65, 50),
    })

    return checks


# ──────────────────────────────────────────────────────────────────────
# Report Generation
# ──────────────────────────────────────────────────────────────────────

def _ascii_bar(value: float, max_width: int = 30) -> str:
    """Simple ASCII bar for inline display."""
    if np.isnan(value):
        return ""
    # Normalize to [-1, 1] range approximately
    clamped = max(-0.15, min(0.15, value))
    width = int(abs(clamped) / 0.15 * max_width)
    if clamped >= 0:
        return "+" + "█" * width
    return "-" + "█" * width


def generate_report(
    decile_per_month: pd.DataFrame,
    decile_agg: pd.DataFrame,
    hit_df: pd.DataFrame,
    ic_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    ls_df: pd.DataFrame,
    checks: List[dict],
    top_k: int,
) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# Prediction Diagnostics Report")
    lines.append(f"**Period**: {BT_START[:4]}-{BT_START[4:6]} ~ {BT_END[:4]}-{BT_END[4:6]}")
    lines.append(f"**Universe**: ZZ500 | **Horizon**: {HORIZON}d | **Top-K**: {top_k}")
    lines.append("")

    # ── Health Check ──
    lines.append("## Health Check Verdict")
    lines.append("")
    lines.append("| Metric | Value | Grade |")
    lines.append("|--------|-------|-------|")
    pass_count = sum(1 for c in checks if c["grade"] == "PASS")
    for c in checks:
        emoji = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[c["grade"]]
        lines.append(f"| {c['metric']} | {c['value']} | **{emoji}** |")
    lines.append("")
    lines.append(f"**Overall: {pass_count}/{len(checks)} PASS**")
    lines.append("")

    # ── Decile Analysis ──
    lines.append("## 1. Decile Returns (分层收益)")
    lines.append("")
    if not decile_agg.empty:
        mono_pct = decile_agg.attrs.get("monotone_pct", 0)
        n_months = decile_agg.attrs.get("n_months", 0)
        ls_spread = decile_agg.attrs.get("ls_spread", 0)
        lines.append(f"- **{n_months}** months analyzed")
        lines.append(f"- D1>D10 monotonicity: **{mono_pct:.1f}%** of months")
        lines.append(f"- D1-D10 spread: **{ls_spread:.2%}** (avg monthly)")
        lines.append("")
        lines.append("| Decile | Mean Return | Std | Bar |")
        lines.append("|--------|------------|-----|-----|")
        for _, row in decile_agg.iterrows():
            bar = _ascii_bar(row["mean_ret"])
            lines.append(f"| D{int(row['decile']):d} | {row['mean_ret']:.2%} | "
                         f"{row['std_ret']:.2%} | {bar} |")
    lines.append("")

    # ── Hit Rate ──
    lines.append(f"## 2. Top-{top_k} Hit Rate (命中率)")
    lines.append("")
    if not hit_df.empty:
        abs_hit = hit_df.attrs.get("abs_hit_mean", 0)
        rel_hit = hit_df.attrs.get("rel_hit_mean", 0)
        mean_excess = hit_df.attrs.get("mean_excess", 0)
        lines.append(f"- Absolute hit rate (fwd_ret > 0): **{abs_hit:.1%}**")
        lines.append(f"- Relative hit rate (> ZZ500): **{rel_hit:.1%}**")
        lines.append(f"- Mean monthly excess return: **{mean_excess:.2%}**")
        lines.append("")
        lines.append("| Month | Top-K Ret | Bench | Excess | Abs Hit | Rel Hit |")
        lines.append("|-------|-----------|-------|--------|---------|---------|")
        for _, row in hit_df.iterrows():
            excess = row["mean_ret"] - row["bench_ret"]
            lines.append(f"| {row['month']} | {row['mean_ret']:.2%} | "
                         f"{row['bench_ret']:.2%} | {excess:+.2%} | "
                         f"{row['abs_hit_rate']:.0%} | {row['rel_hit_rate']:.0%} |")
    lines.append("")

    # ── Rolling IC ──
    lines.append("## 3. Rolling IC by Regime (滚动稳定性)")
    lines.append("")
    if not ic_df.empty:
        lines.append(f"- Overall IC: **{ic_df.attrs.get('ic_mean', 0):.4f}** "
                     f"(std={ic_df.attrs.get('ic_std', 0):.4f})")
        lines.append(f"- ICIR: **{ic_df.attrs.get('icir', 0):.2f}**")
        lines.append(f"- IC > 0: **{ic_df.attrs.get('ic_pos_pct', 0):.1%}**")
        lines.append("")
        if not regime_df.empty:
            lines.append("| Regime | IC Mean | IC Std | ICIR | IC>0 | N |")
            lines.append("|--------|---------|--------|------|------|---|")
            for _, row in regime_df.iterrows():
                lines.append(f"| {row['regime']} | {row['ic_mean']:.4f} | "
                             f"{row['ic_std']:.4f} | {row['icir']:.2f} | "
                             f"{row['ic_pos_pct']:.1%} | {row['n_dates']} |")
        lines.append("")
        lines.append("### Monthly IC Series")
        lines.append("")
        lines.append("| Date | IC | Regime |")
        lines.append("|------|----|--------|")
        for _, row in ic_df.iterrows():
            d = pd.Timestamp(row["score_date"]).strftime("%Y-%m")
            lines.append(f"| {d} | {row['ic']:.4f} | {row['regime']} |")
    lines.append("")

    # ── Long-Short ──
    lines.append("## 4. Long-Short Spread (多空价差)")
    lines.append("")
    if not ls_df.empty:
        lines.append(f"- Mean monthly spread: **{ls_df.attrs.get('mean_spread', 0):.2%}**")
        lines.append(f"- Annualized spread: **{ls_df.attrs.get('annual_spread', 0):.2%}**")
        lines.append(f"- Spread > 0: **{ls_df.attrs.get('spread_pos_pct', 0):.1%}** of months")
        lines.append("")

        regime_spreads = ls_df.attrs.get("regime_spreads", {})
        if regime_spreads:
            lines.append("| Regime | Mean Spread | Positive% | N |")
            lines.append("|--------|-------------|-----------|---|")
            for regime, stats in regime_spreads.items():
                lines.append(f"| {regime} | {stats['mean_spread']:.2%} | "
                             f"{stats['pos_pct']:.1%} | {stats['n']} |")
            lines.append("")

        lines.append("| Month | D1 | D10 | Spread | Cumulative |")
        lines.append("|-------|-----|-----|--------|------------|")
        for _, row in ls_df.iterrows():
            lines.append(f"| {row['month']} | {row['d1_ret']:.2%} | "
                         f"{row['d10_ret']:.2%} | {row['spread']:+.2%} | "
                         f"{row['cum_spread']:.2%} |")
    lines.append("")

    from datetime import datetime
    lines.append("---")
    lines.append(f"*Generated by Prediction Diagnostics | {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Console Output
# ──────────────────────────────────────────────────────────────────────

def print_summary(checks: List[dict], decile_agg: pd.DataFrame,
                  hit_df: pd.DataFrame, ic_df: pd.DataFrame,
                  regime_df: pd.DataFrame, ls_df: pd.DataFrame,
                  top_k: int):
    """Print concise summary to console."""
    print("\n" + "=" * 60)
    print("  PREDICTION DIAGNOSTICS SUMMARY")
    print("=" * 60)

    # Health check
    print("\n📋 Health Check:")
    for c in checks:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[c["grade"]]
        print(f"  {icon} {c['metric']}: {c['value']}")

    pass_count = sum(1 for c in checks if c["grade"] == "PASS")
    print(f"\n  Overall: {pass_count}/{len(checks)} PASS")

    # Decile returns
    if not decile_agg.empty:
        print("\n📊 Decile Returns:")
        for _, row in decile_agg.iterrows():
            bar = _ascii_bar(row["mean_ret"], max_width=20)
            print(f"  D{int(row['decile']):2d}: {row['mean_ret']:+.2%}  {bar}")

    # Hit rate
    if not hit_df.empty:
        print(f"\n🎯 Top-{top_k} Hit Rate:")
        print(f"  Absolute: {hit_df.attrs.get('abs_hit_mean', 0):.1%}")
        print(f"  Relative: {hit_df.attrs.get('rel_hit_mean', 0):.1%}")
        print(f"  Excess:   {hit_df.attrs.get('mean_excess', 0):.2%}/month")

    # IC
    if not ic_df.empty:
        print(f"\n📈 Cross-sectional IC:")
        print(f"  Mean: {ic_df.attrs.get('ic_mean', 0):.4f}")
        print(f"  ICIR: {ic_df.attrs.get('icir', 0):.2f}")
        print(f"  IC>0: {ic_df.attrs.get('ic_pos_pct', 0):.1%}")
        if not regime_df.empty:
            for _, row in regime_df.iterrows():
                print(f"    {row['regime']}: IC={row['ic_mean']:.4f}, ICIR={row['icir']:.2f}")

    # Long-short
    if not ls_df.empty:
        print(f"\n📉 Long-Short Spread:")
        print(f"  Monthly: {ls_df.attrs.get('mean_spread', 0):.2%}")
        print(f"  Annual:  {ls_df.attrs.get('annual_spread', 0):.2%}")
        print(f"  Positive: {ls_df.attrs.get('spread_pos_pct', 0):.1%}")

    print("\n" + "=" * 60)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prediction diagnostics for walk-forward model")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run scoring loop (ignore cache)")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Skip scoring loop, use cached diagnostics.parquet")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Top-K for hit rate analysis (default: {TOP_K})")
    parser.add_argument("--two-stage", action="store_true",
                        help="Use TwoStageRanker (excess_ret + Stage2 features)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run experiment: excess_ret | lambdarank | filter | lambdarank_excess | "
                             "midcap_train | midcap_post | midcap_both | midcap_50_post")
    args = parser.parse_args()

    t0 = time.time()

    # Phase 1: Scoring
    EXP_CONFIGS = {
        "excess_ret": dict(objective="regression", label_col=EXCESS_LABEL,
                           use_filter=False, tag="excess_ret"),
        "lambdarank": dict(objective="lambdarank", label_col="fwd_ret",
                           use_filter=False, tag="lambdarank"),
        "filter": dict(objective="regression", label_col="fwd_ret",
                       use_filter=True, tag="filter"),
        "lambdarank_excess": dict(objective="lambdarank", label_col=EXCESS_LABEL,
                                  use_filter=False, tag="lambdarank_excess"),
        # --- New cap-aware experiments ---
        "midcap_train": dict(objective="regression", label_col="fwd_ret",
                             use_filter=False, train_cap_pctile=0.3,
                             tag="midcap_train"),
        "midcap_post": dict(objective="regression", label_col="fwd_ret",
                            use_filter=False, post_cap_pctile=0.1,
                            tag="midcap_post"),
        "midcap_both": dict(objective="regression", label_col="fwd_ret",
                            use_filter=False, train_cap_pctile=0.3,
                            post_cap_pctile=0.1, tag="midcap_both"),
        "midcap_50_post": dict(objective="regression", label_col="fwd_ret",
                               use_filter=False, train_cap_pctile=0.5,
                               post_cap_pctile=0.1, tag="midcap_50_post"),
        "post20": dict(objective="regression", label_col="fwd_ret",
                       use_filter=False, post_cap_pctile=0.2, tag="post20"),
        "post30": dict(objective="regression", label_col="fwd_ret",
                       use_filter=False, post_cap_pctile=0.3, tag="post30"),
        # --- Score blending experiments ---
        "blend_0.3": dict(objective="regression", label_col="fwd_ret",
                          use_filter=False, cap_blend_alpha=0.3, tag="blend_03"),
        "blend_0.5": dict(objective="regression", label_col="fwd_ret",
                          use_filter=False, cap_blend_alpha=0.5, tag="blend_05"),
        "blend_1.0": dict(objective="regression", label_col="fwd_ret",
                          use_filter=False, cap_blend_alpha=1.0, tag="blend_10"),
        "blend_2.0": dict(objective="regression", label_col="fwd_ret",
                          use_filter=False, cap_blend_alpha=2.0, tag="blend_20"),
        # --- Market context experiment (dispatched separately) ---
        "context": None,  # sentinel — handled specially
        # --- Hyperparameter tuning experiments ---
        "deep64": None,       # num_leaves=64, min_child=30
        "deep128": None,      # num_leaves=128, min_child=20
        "slow_lr": None,      # learning_rate=0.02
        "less_reg": None,     # reg_alpha=0.01, reg_lambda=0.01
        "extreme30": None,    # train on top/bottom 30% only
        "extreme20": None,    # train on top/bottom 20% only
        "deep64_extreme30": None,  # combo: deep + extreme
        "xscore": None,           # cross-sectional z-score normalization
        "xscore_deep64": None,    # z-score + deep trees
        # --- Ensemble + target engineering ---
        "ensemble5": None,        # 5-model ensemble
        "ensemble5_deep": None,   # 5-model ensemble + deep64
        "winsor5": None,          # winsorize top/bottom 5%
        "winsor2": None,          # winsorize top/bottom 2%
        "prune25": None,          # top-25 features only
        "combo_best": None,       # deep64 + winsor5 + ensemble3
        "residual_excess": None,      # residual features + excess_ret label
        "residual_excess_deep": None,  # above + deep64
        "residual_ens3_deep": None,    # residual + 3-model ensemble + deep64
        "residual_excess_ens3": None,  # residual + excess_ret + ensemble3 + deep64
        "beta_tilt_03": None,          # deep64 + beta-tilt alpha=0.3
        "beta_tilt_05": None,          # deep64 + beta-tilt alpha=0.5
        "beta_tilt_01": None,          # deep64 + beta-tilt alpha=0.1
        # --- Residual alpha features ---
        "residual": None,         # per-stock alpha after removing beta*mkt
        "residual_deep64": None,  # residual + deeper trees
        # --- Multi-horizon ensemble ---
        "multi_horizon": None,    # 5d+10d+20d horizon ensemble
        "multi_horizon_deep": None,  # multi-horizon + deep64
        # --- Regime-interaction features ---
        # --- Excess-ret + hyperparameter combos ---
        "excess_deep64": None,          # excess_ret label + deep64
        "excess_ens3": None,            # excess_ret label + 3-model ensemble
        "excess_deep64_ens3": None,     # excess_ret + deep64 + 3-model ensemble
        "excess_shallow15": None,       # excess_ret + shallow (num_leaves=15)
        "excess_shallow20": None,       # excess_ret + shallow (num_leaves=20)
        "excess_reg": None,             # excess_ret + strong regularization
        "excess_filter": dict(objective="regression", label_col=EXCESS_LABEL,
                              use_filter=True, tag="excess_filter"),
        "excess_winsor5": None,         # excess_ret + winsorize 5%
        "excess_prune30": None,         # excess_ret + top 30 features only
        # --- Regime-interaction features ---
        "regime_interact": None,        # interaction features + base model
        "regime_deep64": None,          # interaction + deep64
        "regime_ens3_deep": None,       # interaction + deep64 + 3-model ensemble
        "regime_excess_deep": None,     # interaction + excess_ret + deep64
        "regime_combo_deep": None,      # interaction + residual + deep64
        "regime_split": None,           # separate bull/bear models
        "regime_split_deep": None,      # regime split + deep64
        # --- Blend experiments ---
        "blend_best": None,             # walk-forward blend: excess_ret(0.80) + extreme30(0.20)
        "blend_posthoc": None,          # post-hoc blend from cached diagnostics
        "blend_82_18": None,            # walk-forward blend: excess_ret(0.82) + extreme30(0.18)
    }

    # Tuned experiment configs: tag -> (lgb_params, extreme_pctile)
    TUNED_CONFIGS = {
        "deep64": ({"num_leaves": 64, "min_child_samples": 30}, 0.0),
        "deep128": ({"num_leaves": 128, "min_child_samples": 20}, 0.0),
        "slow_lr": ({"learning_rate": 0.02}, 0.0),
        "less_reg": ({"reg_alpha": 0.01, "reg_lambda": 0.01}, 0.0),
        "extreme30": ({}, 0.3),
        "extreme20": ({}, 0.2),
        "deep64_extreme30": ({"num_leaves": 64, "min_child_samples": 30}, 0.3),
    }

    if args.experiment:
        if args.experiment not in EXP_CONFIGS:
            logger.error("Unknown experiment: {}. Choose: {}", args.experiment,
                         list(EXP_CONFIGS.keys()))
            sys.exit(1)
        cfg = EXP_CONFIGS[args.experiment]
        logger.info("=== EXPERIMENT: {} ===", args.experiment)
        if args.experiment == "context":
            diag_df = run_scoring_loop_with_context(force=args.force)
        elif args.experiment in TUNED_CONFIGS:
            lgb_p, ext_pct = TUNED_CONFIGS[args.experiment]
            diag_df = run_scoring_loop_tuned(
                force=args.force, tag=args.experiment,
                lgb_params=lgb_p or None, extreme_pctile=ext_pct)
        elif args.experiment == "xscore":
            diag_df = run_scoring_loop_xscore(force=args.force)
        elif args.experiment == "xscore_deep64":
            diag_df = run_scoring_loop_xscore(
                force=args.force, tag="xscore_deep64",
                lgb_params={"num_leaves": 64, "min_child_samples": 30})
        elif args.experiment == "ensemble5":
            diag_df = run_scoring_loop_ensemble(force=args.force, n_models=5)
        elif args.experiment == "ensemble5_deep":
            diag_df = run_scoring_loop_ensemble(
                force=args.force, tag="ensemble5_deep", n_models=5,
                lgb_params={"num_leaves": 64, "min_child_samples": 30})
        elif args.experiment == "winsor5":
            diag_df = run_scoring_loop_ensemble(
                force=args.force, tag="winsor5", n_models=1, winsorize_pct=0.05)
        elif args.experiment == "winsor2":
            diag_df = run_scoring_loop_ensemble(
                force=args.force, tag="winsor2", n_models=1, winsorize_pct=0.02)
        elif args.experiment == "prune25":
            diag_df = run_scoring_loop_ensemble(
                force=args.force, tag="prune25", n_models=1, top_n_features=25)
        elif args.experiment == "combo_best":
            diag_df = run_scoring_loop_ensemble(
                force=args.force, tag="combo_best", n_models=3,
                lgb_params={"num_leaves": 64, "min_child_samples": 30},
                winsorize_pct=0.05)
        elif args.experiment == "residual":
            diag_df = run_scoring_loop_residual(force=args.force)
        elif args.experiment == "residual_deep64":
            diag_df = run_scoring_loop_residual(
                force=args.force, tag="residual_deep64",
                lgb_params={"num_leaves": 64, "min_child_samples": 30})
        elif args.experiment == "multi_horizon":
            diag_df = run_scoring_loop_multi_horizon(force=args.force)
        elif args.experiment == "multi_horizon_deep":
            diag_df = run_scoring_loop_multi_horizon(
                force=args.force, tag="multi_horizon_deep",
                lgb_params={"num_leaves": 64, "min_child_samples": 30})
        elif args.experiment == "residual_excess":
            diag_df = run_scoring_loop_residual(
                force=args.force, tag="residual_excess",
                label_col=EXCESS_LABEL)
        elif args.experiment == "residual_excess_deep":
            diag_df = run_scoring_loop_residual(
                force=args.force, tag="residual_excess_deep",
                label_col=EXCESS_LABEL,
                lgb_params={"num_leaves": 64, "min_child_samples": 30})
        elif args.experiment == "residual_ens3_deep":
            diag_df = run_scoring_loop_residual(
                force=args.force, tag="residual_ens3_deep",
                lgb_params={"num_leaves": 64, "min_child_samples": 30},
                n_models=3)
        elif args.experiment == "residual_excess_ens3":
            diag_df = run_scoring_loop_residual(
                force=args.force, tag="residual_excess_ens3",
                label_col=EXCESS_LABEL,
                lgb_params={"num_leaves": 64, "min_child_samples": 30},
                n_models=3)
        elif args.experiment == "beta_tilt_03":
            diag_df = run_scoring_loop_beta_tilt(
                force=args.force, tag="beta_tilt_03",
                lgb_params={"num_leaves": 64, "min_child_samples": 30},
                tilt_alpha=0.3)
        elif args.experiment == "beta_tilt_05":
            diag_df = run_scoring_loop_beta_tilt(
                force=args.force, tag="beta_tilt_05",
                lgb_params={"num_leaves": 64, "min_child_samples": 30},
                tilt_alpha=0.5)
        elif args.experiment == "beta_tilt_01":
            diag_df = run_scoring_loop_beta_tilt(
                force=args.force, tag="beta_tilt_01",
                lgb_params={"num_leaves": 64, "min_child_samples": 30},
                tilt_alpha=0.1)
        elif args.experiment == "excess_deep64":
            diag_df = run_scoring_loop_tuned(
                force=args.force, tag="excess_deep64",
                lgb_params={"num_leaves": 64, "min_child_samples": 30},
                label_col=EXCESS_LABEL)
        elif args.experiment == "excess_ens3":
            diag_df = run_scoring_loop_ensemble(
                force=args.force, tag="excess_ens3", n_models=3,
                label_col=EXCESS_LABEL)
        elif args.experiment == "excess_deep64_ens3":
            diag_df = run_scoring_loop_ensemble(
                force=args.force, tag="excess_deep64_ens3", n_models=3,
                lgb_params={"num_leaves": 64, "min_child_samples": 30},
                label_col=EXCESS_LABEL)
        elif args.experiment == "excess_shallow15":
            diag_df = run_scoring_loop_tuned(
                force=args.force, tag="excess_shallow15",
                lgb_params={"num_leaves": 15, "min_child_samples": 50},
                label_col=EXCESS_LABEL)
        elif args.experiment == "excess_shallow20":
            diag_df = run_scoring_loop_tuned(
                force=args.force, tag="excess_shallow20",
                lgb_params={"num_leaves": 20, "min_child_samples": 40},
                label_col=EXCESS_LABEL)
        elif args.experiment == "excess_reg":
            diag_df = run_scoring_loop_tuned(
                force=args.force, tag="excess_reg",
                lgb_params={"num_leaves": 31, "reg_alpha": 1.0, "reg_lambda": 5.0,
                            "min_child_samples": 50, "feature_fraction": 0.7},
                label_col=EXCESS_LABEL)
        elif args.experiment == "excess_winsor5":
            diag_df = run_scoring_loop_ensemble(
                force=args.force, tag="excess_winsor5", n_models=1,
                winsorize_pct=0.05, label_col=EXCESS_LABEL)
        elif args.experiment == "excess_prune30":
            diag_df = run_scoring_loop_ensemble(
                force=args.force, tag="excess_prune30", n_models=1,
                top_n_features=30, label_col=EXCESS_LABEL)
        elif args.experiment == "regime_interact":
            diag_df = run_scoring_loop_regime_interact(force=args.force)
        elif args.experiment == "regime_deep64":
            diag_df = run_scoring_loop_regime_interact(
                force=args.force, tag="regime_deep64",
                lgb_params={"num_leaves": 64, "min_child_samples": 30})
        elif args.experiment == "regime_ens3_deep":
            diag_df = run_scoring_loop_regime_interact(
                force=args.force, tag="regime_ens3_deep",
                lgb_params={"num_leaves": 64, "min_child_samples": 30},
                n_models=3)
        elif args.experiment == "regime_excess_deep":
            diag_df = run_scoring_loop_regime_interact(
                force=args.force, tag="regime_excess_deep",
                label_col=EXCESS_LABEL,
                lgb_params={"num_leaves": 64, "min_child_samples": 30})
        elif args.experiment == "regime_combo_deep":
            diag_df = run_scoring_loop_regime_combo(
                force=args.force, tag="regime_combo_deep",
                lgb_params={"num_leaves": 64, "min_child_samples": 30})
        elif args.experiment == "regime_split":
            diag_df = run_scoring_loop_regime_split(force=args.force)
        elif args.experiment == "regime_split_deep":
            diag_df = run_scoring_loop_regime_split(
                force=args.force, tag="regime_split_deep",
                lgb_params={"num_leaves": 64, "min_child_samples": 30})
        elif args.experiment == "blend_best":
            diag_df = run_scoring_loop_blend(
                force=args.force, tag="blend_best",
                weight_primary=0.80, extreme_pctile=0.30)
        elif args.experiment == "blend_82_18":
            diag_df = run_scoring_loop_blend(
                force=args.force, tag="blend_82_18",
                weight_primary=0.82, extreme_pctile=0.30)
        elif args.experiment == "blend_posthoc":
            diag_df = run_posthoc_blend(
                tag_a="excess_ret", tag_b="extreme30",
                weight_a=0.80, output_tag="blend_posthoc",
                force=args.force)
        else:
            diag_df = run_scoring_loop_experiment(force=args.force, **cfg)
        diag_df["score_date"] = pd.to_datetime(diag_df["score_date"])
    elif args.two_stage:
        logger.info("=== TWO-STAGE MODE ===")
        diag_df = run_scoring_loop_twostage(force=args.force)
        diag_df["score_date"] = pd.to_datetime(diag_df["score_date"])
    elif args.analysis_only:
        if not DIAG_CACHE.exists():
            logger.error("No cached diagnostics found. Run without --analysis-only first.")
            sys.exit(1)
        logger.info("Loading cached diagnostics (--analysis-only)")
        diag_df = pd.read_parquet(DIAG_CACHE)
        diag_df["score_date"] = pd.to_datetime(diag_df["score_date"])
    else:
        diag_df = run_scoring_loop(force=args.force)
        diag_df["score_date"] = pd.to_datetime(diag_df["score_date"])

    logger.info("Diagnostics data: {} rows, {} months, {} unique stocks",
                len(diag_df), diag_df["score_date"].nunique(),
                diag_df["code"].nunique())

    # Phase 2: Analyses
    decile_per_month, decile_agg = analyze_deciles(diag_df)
    hit_df = analyze_hit_rate(diag_df, top_k=args.top_k)
    ic_df, regime_df = analyze_rolling_ic(diag_df)
    ls_df = analyze_long_short(diag_df)

    # Health check
    checks = health_check(decile_agg, hit_df, ic_df, regime_df, ls_df)

    # Console output
    print_summary(checks, decile_agg, hit_df, ic_df, regime_df, ls_df, args.top_k)

    # Markdown report
    report = generate_report(decile_per_month, decile_agg, hit_df,
                             ic_df, regime_df, ls_df, checks, args.top_k)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    elapsed = time.time() - t0
    logger.info("Done in {:.1f}s. Report: {}", elapsed, REPORT_PATH)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
