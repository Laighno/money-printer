"""Quantitative stock screening with IC-weighted signal scoring.

Screens candidate stocks through the full signal system:
  1. Fetch OHLCV data
  2. Compute 7 technical signals
  3. (Optional) Fetch external data signals
  4. Scoring — LightGBM model if available, else IC-weighted formula fallback
  5. Rating assignment

Usage:
    from mp.screener.signal_screener import screen_stocks
    result = screen_stocks(["603799", "002466", "002460"])
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from mp.data.fetcher import get_daily_bars
from mp.indicators.technical import compute_all_technical_signals
from mp.indicators.external import fetch_all_external_signals
from mp.backtest.ic_analysis import (
    rolling_rsi, rolling_macd_hist, rolling_bollinger_pctb,
    rolling_kdj_j, rolling_vol_price_ratio, rolling_momentum,
    rolling_volatility, rolling_momentum_accel,
    rolling_rsi_delta, rolling_macd_hist_delta, rolling_volume_trend,
)


# === IC weights from ZZ500 cross-sectional analysis (N=100, 20d horizon, 2026-04-08) ===
# Weights proportional to |IR| = |IC_mean / IC_std|, normalized to sum=0.73.
# Source: scripts/batch_ic_analysis.py on 100 random ZZ500 stocks, start=20230101.
#
# ic_sign: -1 = mean-reversion, +1 = trend-following
# Factors with |IR| < 0.3 excluded (量能趋势: 0.239, 量价比: 0.232)

_IC_FACTOR_CONFIG = {
    # Level factors — top 4 all have |IR| > 1.0, extremely consistent across stocks
    "RSI(14)":   (0.127, -1),  # |IR|=1.319, IC>0=0%, N=100
    "动量60d":   (0.119, -1),  # |IR|=1.235, IC>0=0%, N=100
    "动量20d":   (0.108, -1),  # |IR|=1.125, IC>0=0%, N=100
    "布林%B":    (0.101, -1),  # |IR|=1.052, IC>0=0%, N=100
    "KDJ-J":     (0.068, -1),  # |IR|=0.706, IC>0=1%, N=100
    "MACD柱":    (0.058, -1),  # |IR|=0.598, IC>0=2%, N=100
    "波动率20d":  (0.032, -1),  # |IR|=0.328, IC>0=12%, N=100
    # Derivative factors
    "动量加速度":  (0.066, +1),  # |IR|=0.685, IC>0=100%, N=100 — only trend-following factor
    "RSI变化":   (0.047, -1),  # |IR|=0.484, IC>0=2%, N=100
    "布林%B变化": (0.030, -1),  # |IR|=0.316, IC>0=21%, N=100 — newly included
    "MACD柱变化": (0.029, -1),  # |IR|=0.300, IC>0=24%, N=100
}
# Sum of weights: 0.785, remaining ~0.215 for external signal vote


def _compute_factor_values(close: np.ndarray, high: np.ndarray,
                           low: np.ndarray, volume: np.ndarray
                           ) -> Dict[str, float]:
    """Compute latest factor values for IC scoring."""
    n = len(close)
    values = {}

    if n >= 15:
        rsi = rolling_rsi(close)
        values["RSI(14)"] = float(rsi[-1]) if not np.isnan(rsi[-1]) else np.nan

    if n >= 35:
        macd = rolling_macd_hist(close)
        values["MACD柱"] = float(macd[-1]) if not np.isnan(macd[-1]) else np.nan

    if n >= 20:
        boll = rolling_bollinger_pctb(close)
        values["布林%B"] = float(boll[-1]) if not np.isnan(boll[-1]) else np.nan

    if n >= 9:
        kdj = rolling_kdj_j(high, low, close)
        values["KDJ-J"] = float(kdj[-1]) if not np.isnan(kdj[-1]) else np.nan

    if n >= 10:
        vpr = rolling_vol_price_ratio(close, volume)
        values["量价比"] = float(vpr[-1]) if not np.isnan(vpr[-1]) else np.nan

    if n >= 21:
        mom20 = rolling_momentum(close, 20)
        values["动量20d"] = float(mom20[-1]) if not np.isnan(mom20[-1]) else np.nan

    if n >= 61:
        mom60 = rolling_momentum(close, 60)
        values["动量60d"] = float(mom60[-1]) if not np.isnan(mom60[-1]) else np.nan

    if n >= 21:
        vol20 = rolling_volatility(close, 20)
        values["波动率20d"] = float(vol20[-1]) if not np.isnan(vol20[-1]) else np.nan

    if n >= 21:
        accel = rolling_momentum_accel(close)
        values["动量加速度"] = float(accel[-1]) if not np.isnan(accel[-1]) else np.nan

    if n >= 20:
        rsi_d = rolling_rsi_delta(close)
        values["RSI变化"] = float(rsi_d[-1]) if not np.isnan(rsi_d[-1]) else np.nan

    if n >= 25:
        boll_d = rolling_bollinger_pctb(close)
        if len(boll_d) >= 6:
            delta = boll_d[-1] - boll_d[-6]
            values["布林%B变化"] = float(delta) if not np.isnan(delta) else np.nan

    if n >= 40:
        macd_d = rolling_macd_hist_delta(close)
        values["MACD柱变化"] = float(macd_d[-1]) if not np.isnan(macd_d[-1]) else np.nan

    return values


def _normalize_factor(name: str, value: float) -> float:
    """Normalize a factor value to roughly [-1, 1] using known typical ranges.

    Convention: positive = bullish signal direction BEFORE applying IC sign.
    """
    ranges = {
        "RSI(14)":  (50.0, 30.0),   # center=50, half-range=30 → RSI 20→-1, RSI 80→+1
        "MACD柱":   (0.0, 1.0),     # center=0, scale=1
        "布林%B":   (0.5, 0.5),     # center=0.5, half-range=0.5 → %B 0→-1, %B 1→+1
        "KDJ-J":    (50.0, 50.0),   # center=50, half-range=50
        "动量20d":  (0.0, 0.15),    # center=0, half-range=15%
        "动量60d":  (0.0, 0.25),    # center=0, half-range=25%
        "波动率20d": (0.3, 0.2),    # center=0.3, half-range=0.2
        "动量加速度": (0.0, 0.10),   # center=0, half-range=10% → positive = recovering
        "RSI变化":  (0.0, 15.0),   # center=0, half-range=15 RSI points
        "MACD柱变化": (0.0, 0.5),   # center=0, half-range=0.5
        "布林%B变化": (0.0, 0.5),   # center=0, half-range=0.5 → delta over 5 days
    }
    center, scale = ranges.get(name, (0.0, 1.0))
    return max(-1.0, min(1.0, (value - center) / scale)) if scale > 0 else 0.0


def _ic_score_single(factor_values: Dict[str, float]) -> float:
    """Compute IC-weighted score for one stock from its normalized factor values.

    Score > 0 = bullish, < 0 = bearish. Magnitude indicates strength.
    """
    score = 0.0
    total_weight = 0.0

    for fname, (weight, ic_sign) in _IC_FACTOR_CONFIG.items():
        val = factor_values.get(fname)
        if val is None or np.isnan(val):
            continue
        normalized = _normalize_factor(fname, val)
        # ic_sign flips the meaning: if IC<0 (mean reversion), low value = bullish
        score += weight * ic_sign * normalized
        total_weight += weight

    if total_weight > 0:
        score /= total_weight
    return score


def _vote_score(signals: list) -> float:
    """Simple bullish/bearish vote score: (bull - bear) / total."""
    if not signals:
        return 0.0
    n_bull = sum(1 for s in signals if s["signal"] == "bullish")
    n_bear = sum(1 for s in signals if s["signal"] == "bearish")
    total = len(signals)
    return (n_bull - n_bear) / total


def _rating(signal_score: float) -> str:
    """Assign rating based on signal score."""
    if signal_score > 0.5:
        return "★★★"
    elif signal_score > 0.2:
        return "★★☆"
    elif signal_score > 0:
        return "★☆☆"
    else:
        return "⚠️"


def screen_single_stock(
    code: str,
    start: str = "20230101",
    include_external: bool = False,
) -> Optional[Dict]:
    """Screen a single stock. Returns dict with scores or None on failure."""
    try:
        df = get_daily_bars(code, start)
        if df is None or len(df) < 30:
            return None

        df = df.sort_values("date")
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        open_arr = df["open"].values.astype(float)
        volume = df["volume"].values.astype(float)

        # Technical signals (for vote score + display)
        tech_signals = compute_all_technical_signals(close, high, low, open_arr, volume)

        # IC-weighted factor score
        factor_values = _compute_factor_values(close, high, low, volume)
        ic_score = _ic_score_single(factor_values)

        # External signals (optional)
        ext_signals = []
        ext_vote = 0.0
        if include_external:
            ext_signals = fetch_all_external_signals(code)
            ext_vote = _vote_score(ext_signals)

        all_signals = tech_signals + ext_signals

        # Combined score
        tech_vote = _vote_score(tech_signals)
        if include_external and ext_signals:
            vote = 0.7 * tech_vote + 0.3 * ext_vote
        else:
            vote = tech_vote

        # ic_score is already in [-1, 1] from normalized factor values
        signal_score = 0.7 * ic_score + 0.3 * vote

        n_bull = sum(1 for s in all_signals if s["signal"] == "bullish")
        n_bear = sum(1 for s in all_signals if s["signal"] == "bearish")
        n_neut = sum(1 for s in all_signals if s["signal"] == "neutral")

        return {
            "code": code,
            "price": float(close[-1]),
            "chg_5d": float((close[-1] / close[-6] - 1) * 100) if len(close) > 6 else 0,
            "chg_20d": float((close[-1] / close[-21] - 1) * 100) if len(close) > 21 else 0,
            "bull": n_bull,
            "bear": n_bear,
            "neutral": n_neut,
            "ic_score": round(ic_score, 3),
            "vote_score": round(vote, 3),
            "signal_score": round(signal_score, 3),
            "rating": _rating(signal_score),
            "signals": all_signals,
            "factor_values": factor_values,
        }
    except Exception as e:
        logger.warning(f"Screen failed for {code}: {e}")
        return None


def screen_stocks(
    codes: List[str],
    names: Optional[Dict[str, str]] = None,
    start: str = "20230101",
    include_external: bool = False,
    use_ml: bool = False,
    progress_callback=None,
) -> pd.DataFrame:
    """Screen multiple stocks and return ranked DataFrame.

    Args:
        codes: list of 6-digit A-share stock codes
        names: optional code->name mapping
        start: start date for historical data
        include_external: whether to fetch external API signals (slow)
        use_ml: if True, use LightGBM model for scoring (falls back to formula)
        progress_callback: optional callable(current, total) for progress updates

    Returns:
        DataFrame sorted by signal_score descending, with columns:
        code, name, price, chg_5d, chg_20d, bull, bear, neutral,
        ic_score, vote_score, signal_score, rating
    """
    if names is None:
        names = {}

    results = []
    for i, code in enumerate(codes):
        result = screen_single_stock(code, start, include_external)
        if result is not None:
            result["name"] = names.get(code, code)
            results.append(result)
        if progress_callback:
            progress_callback(i + 1, len(codes))

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # --- ML scoring overlay ---
    if use_ml:
        try:
            from mp.ml.model import StockRanker, FEATURE_COLS
            from mp.ml.dataset import build_latest_features

            ranker = StockRanker()
            if ranker.load():
                features = build_latest_features(df["code"].tolist())
                if not features.empty:
                    ml_scores = ranker.predict(features)
                    # Map ml_score back to result rows by code
                    ml_map = dict(zip(features["code"], ml_scores))
                    df["ml_score"] = df["code"].map(ml_map)
                    # Replace signal_score with ML-based score where available
                    has_ml = df["ml_score"].notna()
                    if has_ml.any():
                        # Normalize ML scores to roughly [-1, 1] range
                        ml_vals = df.loc[has_ml, "ml_score"]
                        ml_mean, ml_std = ml_vals.mean(), ml_vals.std()
                        if ml_std > 0:
                            df.loc[has_ml, "signal_score"] = ((ml_vals - ml_mean) / ml_std).clip(-1, 1).round(3)
                        df.loc[has_ml, "rating"] = df.loc[has_ml, "signal_score"].apply(_rating)
                        logger.info("ML scoring applied to {} stocks", has_ml.sum())
            else:
                logger.info("No ML model found, using formula scoring")
        except Exception as e:
            logger.warning("ML scoring failed, using formula fallback: {}", e)

    # Drop internal detail columns for the summary view
    display_cols = ["code", "name", "price", "chg_5d", "chg_20d",
                    "bull", "bear", "neutral", "ic_score", "vote_score",
                    "signal_score", "rating"]
    df = df[[c for c in display_cols if c in df.columns]].sort_values(
        "signal_score", ascending=False).reset_index(drop=True)
    return df
