"""Factor library — all 51 technical factors declared as expression trees.

Every entry in ``FACTOR_LIBRARY`` corresponds 1-to-1 with a column in
``mp.ml.dataset.TECHNICAL_COLUMNS``.  The key is the column name, and
the value is a :class:`FactorDef` whose expression, when evaluated,
produces the same time series as the hand-written ``rolling_*``
functions in ``mp.backtest.ic_analysis``.

Usage::

    from mp.factor.library import FACTOR_LIBRARY, compute_factor, compute_all_factors

    # single factor
    series = compute_factor("rsi_14", df)

    # all factors at once
    factor_df = compute_all_factors(df)
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .expr import (
    BinaryOp,
    Const,
    Custom,
    Delta,
    Expr,
    FactorDef,
    Field,
    Lag,
    Rolling,
    UnaryOp,
    # helpers
    add,
    amount,
    close,
    delta,
    div,
    evaluate,
    high,
    low,
    mul,
    open_,
    pct_change,
    roll_mean,
    roll_std,
    roll_max,
    roll_min,
    roll_sum,
    sub,
    turnover,
    volume,
)


# ---------------------------------------------------------------------------
# Factor definitions — grouped to match TECHNICAL_COLUMNS in dataset.py
# ---------------------------------------------------------------------------

def _build_library() -> Dict[str, FactorDef]:
    """Construct and return the full factor library."""
    lib: Dict[str, FactorDef] = {}

    def _add(name: str, expr: Expr, desc: str, cat: str = "technical") -> None:
        lib[name] = FactorDef(name=name, expr=expr, description=desc, category=cat)

    # =======================================================================
    # --- Original 13 factors ---
    # =======================================================================

    # 1. RSI(14) — Wilder smoothing
    _add("rsi_14",
         Custom("rsi", (14,)),
         "RSI with Wilder smoothing, period 14",
         "mean_reversion")

    # 2. MACD histogram
    _add("macd_hist",
         Custom("macd_hist", (12, 26, 9)),
         "MACD histogram (2 * (DIF - DEA)), fast=12, slow=26, signal=9",
         "momentum")

    # 3. Bollinger %B
    _add("boll_pctb",
         Custom("bollinger_pctb", (20, 2.0)),
         "Bollinger %B: (close - lower) / (upper - lower), period=20, 2 std",
         "mean_reversion")

    # 4. KDJ J-value
    _add("kdj_j",
         Custom("kdj_j", (9, 3, 3)),
         "KDJ J-value (3K - 2D), n=9, m1=3, m2=3",
         "mean_reversion")

    # 5. Volume/price ratio: recent_5d_vol / prev_5d_vol
    _add("vol_price_ratio",
         Custom("vol_price_ratio", (5,)),
         "Volume ratio: avg volume(5d recent) / avg volume(5d previous)",
         "volume")

    # 6. 20-day momentum
    _add("mom_20d",
         pct_change(close(), 20),
         "20-day price momentum: close/close[20] - 1",
         "momentum")

    # 7. 60-day momentum
    _add("mom_60d",
         pct_change(close(), 60),
         "60-day price momentum: close/close[60] - 1",
         "momentum")

    # 8. 20-day volatility (annualized)
    # rolling_volatility: std(daily_returns, 20) * sqrt(252)
    _add("volatility_20d",
         mul(roll_std(pct_change(close(), 1), 20), Const(252 ** 0.5)),
         "Annualized 20-day rolling volatility of daily returns",
         "volatility")

    # 9. RSI delta (change over 5 bars)
    _add("rsi_delta",
         Delta(Custom("rsi", (14,)), 5),
         "RSI(14) change over 5 bars — positive means RSI recovering",
         "mean_reversion")

    # 10. MACD histogram delta
    _add("macd_hist_delta",
         Delta(Custom("macd_hist", (12, 26, 9)), 5),
         "MACD histogram change over 5 bars — positive means momentum improving",
         "momentum")

    # 11. Bollinger %B delta
    _add("boll_pctb_delta",
         Delta(Custom("bollinger_pctb", (20, 2.0)), 5),
         "Bollinger %B change over 5 bars",
         "mean_reversion")

    # 12. Momentum acceleration: mom_5d - mom_20d
    _add("mom_accel",
         sub(pct_change(close(), 5), pct_change(close(), 20)),
         "Momentum acceleration: 5d momentum - 20d momentum",
         "momentum")

    # 13. Volume trend: avg_vol(5) / avg_vol(20)
    _add("volume_trend",
         div(roll_mean(volume(), 5), roll_mean(volume(), 20)),
         "Volume trend: 5-day avg volume / 20-day avg volume (>1 = expanding)",
         "volume")

    # =======================================================================
    # --- MA factors ---
    # =======================================================================

    # 14. Close / MA(5) - 1
    _add("close_ma5_dev",
         sub(div(close(), roll_mean(close(), 5)), Const(1.0)),
         "Close deviation from 5-day MA: close/MA(5) - 1",
         "trend")

    # 15. Close / MA(20) - 1
    _add("close_ma20_dev",
         sub(div(close(), roll_mean(close(), 20)), Const(1.0)),
         "Close deviation from 20-day MA: close/MA(20) - 1",
         "trend")

    # 16. Close / MA(60) - 1
    _add("close_ma60_dev",
         sub(div(close(), roll_mean(close(), 60)), Const(1.0)),
         "Close deviation from 60-day MA: close/MA(60) - 1",
         "trend")

    # 17. MA alignment: (MA5 - MA60) / MA60
    _add("ma_alignment",
         div(sub(roll_mean(close(), 5), roll_mean(close(), 60)),
             roll_mean(close(), 60)),
         "Multi-MA alignment: (MA5 - MA60) / MA60 — positive = bullish",
         "trend")

    # =======================================================================
    # --- ATR / range ---
    # =======================================================================

    # 18. ATR(14) / close — normalized
    _add("atr_14",
         Custom("atr", (14,)),
         "Normalized ATR: ATR(14) / close — Wilder smoothing",
         "volatility")

    # 19. Average (high - low) / close over 10d
    _add("price_range_10d",
         roll_mean(div(sub(high(), low()), close()), 10),
         "Average intraday range: mean((high-low)/close, 10)",
         "volatility")

    # =======================================================================
    # --- Oscillators ---
    # =======================================================================

    # 20. Williams %R
    _add("williams_r",
         Custom("williams_r", (14,)),
         "Williams %R (14): -100 (oversold) to 0 (overbought)",
         "mean_reversion")

    # 21. ADX(14)
    _add("adx_14",
         Custom("adx", (14,)),
         "ADX(14): average directional index, 0-100 trend strength",
         "trend")

    # =======================================================================
    # --- Volume / amount ---
    # =======================================================================

    # 22. OBV slope (normalized)
    _add("obv_slope",
         Custom("obv_slope", (20,)),
         "OBV regression slope over 20d, normalized by avg volume",
         "volume")

    # 23. Amount ratio: today's amount / MA(20) amount
    _add("amount_ratio",
         div(amount(), roll_mean(amount(), 20)),
         "Amount ratio: current amount / MA(20) amount",
         "volume")

    # =======================================================================
    # --- Turnover ---
    # =======================================================================

    # 24. Turnover MA(5)
    _add("turnover_5d",
         roll_mean(turnover(), 5),
         "5-day moving average of turnover rate",
         "volume")

    # 25. Turnover percentile vs 60d
    _add("turnover_pctile",
         Custom("turnover_percentile", (60,)),
         "Current turnover percentile within trailing 60 days (0-1)",
         "volume")

    # =======================================================================
    # --- Return distribution ---
    # =======================================================================

    # 26. Return skewness (20d)
    _add("return_skew_20d",
         Custom("return_skew", (20,)),
         "Skewness of 20-day daily returns — positive = right tail (bullish)",
         "volatility")

    # 27. Return kurtosis (20d)
    _add("return_kurtosis_20d",
         Custom("return_kurtosis", (20,)),
         "Excess kurtosis of 20-day daily returns — high = fat tails",
         "volatility")

    # 28. Upside/downside vol ratio
    _add("updown_vol_ratio",
         Custom("updown_vol_ratio", (20,)),
         "Upside vol / downside vol (20d) — >1 means up moves bigger",
         "volatility")

    # 29. Max drawdown (20d)
    _add("max_drawdown_20d",
         Custom("max_drawdown", (20,)),
         "Maximum drawdown over trailing 20 days (always <= 0)",
         "volatility")

    # 30. Return autocorrelation (lag-1, 20d window)
    _add("return_autocorr",
         Custom("return_autocorr", (20,)),
         "Lag-1 autocorrelation of returns (20d) — +trend, -reversal",
         "momentum")

    # =======================================================================
    # --- K-line shape ---
    # =======================================================================

    # 31. Close position: avg((close-low)/(high-low), 5d)
    # daily_pos = (close - low) / (high - low), then rolling mean
    _add("close_position",
         roll_mean(
             div(sub(close(), low()),
                 sub(high(), low())),
             5),
         "Avg close position within daily range over 5d (1=top, 0=bottom)",
         "mean_reversion")

    # 32. Upper shadow: avg((high - max(open,close))/close, 5d)
    _add("upper_shadow",
         roll_mean(
             div(sub(high(), BinaryOp("max", open_(), close())),
                 close()),
             5),
         "Avg upper shadow ratio over 5d",
         "mean_reversion")

    # 33. Lower shadow: avg((min(open,close) - low)/close, 5d)
    _add("lower_shadow",
         roll_mean(
             div(sub(BinaryOp("min", open_(), close()), low()),
                 close()),
             5),
         "Avg lower shadow ratio over 5d",
         "mean_reversion")

    # 34. Body ratio: avg(|close-open|/(high-low), 5d)
    _add("body_ratio",
         roll_mean(
             div(UnaryOp("abs", sub(close(), open_())),
                 sub(high(), low())),
             5),
         "Avg body/range ratio over 5d (1 = no shadow)",
         "mean_reversion")

    # 35. Gap: avg((open - prev_close) / prev_close, 5d)
    _add("gap_5d",
         roll_mean(
             div(sub(open_(), Lag(close(), 1)),
                 Lag(close(), 1)),
             5),
         "Average gap ratio (open vs prev close) over 5d",
         "momentum")

    # =======================================================================
    # --- Liquidity ---
    # =======================================================================

    # 36. Amihud illiquidity: avg(|ret| / amount, 20d)
    _add("amihud_illiq",
         roll_mean(
             div(UnaryOp("abs", pct_change(close(), 1)),
                 amount()),
             20),
         "Amihud illiquidity: avg(|return|/amount, 20d) — higher = less liquid",
         "volume")

    # 37. Volume volatility: CV of volume over 20d
    _add("volume_volatility",
         div(roll_std(volume(), 20), roll_mean(volume(), 20)),
         "Volume CV: std(volume,20) / mean(volume,20) — irregular trading",
         "volume")

    # =======================================================================
    # --- Short momentum ---
    # =======================================================================

    # 38. 5-day momentum
    _add("mom_5d",
         pct_change(close(), 5),
         "5-day price momentum: close/close[5] - 1",
         "momentum")

    # 39. 10-day momentum
    _add("mom_10d",
         pct_change(close(), 10),
         "10-day price momentum: close/close[10] - 1",
         "momentum")

    # =======================================================================
    # --- Cross-field combinations ---
    # =======================================================================

    # 40. VWAP deviation
    _add("vwap_dev",
         Custom("vwap_deviation", (20,)),
         "Close / VWAP(20) - 1 where VWAP = sum(amount)/sum(volume)",
         "mean_reversion")

    # 41. Bollinger bandwidth: 2 * num_std * std / mean
    _add("boll_bandwidth",
         div(mul(Const(4.0), roll_std(close(), 20)),
             roll_mean(close(), 20)),
         "Bollinger bandwidth: 4 * std(20) / mean(20) — volatility squeeze",
         "volatility")

    # 42. Volume-price correlation (20d)
    _add("vol_price_corr",
         Custom("vol_price_corr", (20,)),
         "Rolling corr(returns, volume, 20d) — +1 bullish, -1 bearish vol",
         "volume")

    # 43. Consecutive up/down days
    _add("consecutive_days",
         Custom("consecutive_days"),
         "Consecutive up(+)/down(-) day count",
         "momentum")

    # 44. Distance from 60-day high: close / max(high, 60) - 1
    _add("high_distance_60d",
         sub(div(close(), roll_max(high(), 60)), Const(1.0)),
         "Distance from 60-day high: close/max(high,60) - 1 (always <= 0)",
         "momentum")

    # 45. Distance from 60-day low: close / min(low, 60) - 1
    _add("low_distance_60d",
         sub(div(close(), roll_min(low(), 60)), Const(1.0)),
         "Distance from 60-day low: close/min(low,60) - 1 (always >= 0)",
         "momentum")

    # 46. Volatility ratio: short-term vol(5) / long-term vol(60)
    _add("vol_ratio_5_60",
         div(roll_std(pct_change(close(), 1), 5),
             roll_std(pct_change(close(), 1), 60)),
         "Vol ratio: std(ret,5) / std(ret,60) — >1 = vol expanding",
         "volatility")

    # 47. MFI(14) — Money Flow Index
    _add("mfi_14",
         Custom("mfi", (14,)),
         "Money Flow Index(14): RSI weighted by volume, 0-100",
         "volume")

    # 48. Intraday intensity
    _add("intraday_intensity",
         Custom("intraday_intensity", (20,)),
         "Intraday intensity: avg((2c-h-l)/(h-l)*vol) / avg_vol, 20d",
         "volume")

    # 49. CCI(20)
    _add("cci_20",
         Custom("cci", (20,)),
         "Commodity Channel Index(20): (tp - sma(tp)) / (0.015 * mad(tp))",
         "mean_reversion")

    # 50. Return extremes ratio
    _add("return_extremes_ratio",
         Custom("return_extremes_ratio", (20,)),
         "max(ret) / |min(ret)| over 20d — >1 = up extremes dominate",
         "momentum")

    # 51. Amount volatility: CV of trading amount
    _add("amount_volatility",
         div(roll_std(amount(), 20), roll_mean(amount(), 20)),
         "Amount CV: std(amount,20) / mean(amount,20) — erratic activity",
         "volume")

    return lib


# ---------------------------------------------------------------------------
# Module-level library
# ---------------------------------------------------------------------------

FACTOR_LIBRARY: Dict[str, FactorDef] = _build_library()


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def compute_factor(name: str, df: pd.DataFrame) -> pd.Series:
    """Evaluate a single named factor against an OHLCV DataFrame.

    Parameters
    ----------
    name : str
        Factor name — must be a key in ``FACTOR_LIBRARY``.
    df : pd.DataFrame
        DataFrame with at least: open, high, low, close, volume.

    Returns
    -------
    pd.Series
    """
    if name not in FACTOR_LIBRARY:
        available = ", ".join(sorted(FACTOR_LIBRARY.keys()))
        raise KeyError(f"Factor '{name}' not in library. Available: {available}")
    return evaluate(FACTOR_LIBRARY[name].expr, df)


def compute_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 51 technical factors and return as a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (must have: open, high, low, close, volume;
        optionally: amount, turnover).

    Returns
    -------
    pd.DataFrame
        Same index as *df*, one column per factor.
    """
    results: Dict[str, pd.Series] = {}
    for name, fdef in FACTOR_LIBRARY.items():
        try:
            results[name] = evaluate(fdef.expr, df)
        except Exception as exc:
            import warnings
            warnings.warn(f"Failed to compute factor '{name}': {exc}")
            import numpy as np
            results[name] = pd.Series(np.nan, index=df.index)
    return pd.DataFrame(results, index=df.index)
