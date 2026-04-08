"""Build cross-sectional training datasets for LightGBM.

Fetches daily OHLCV for a universe of stocks, computes ~30 rolling technical
factors via ``mp.backtest.ic_analysis``, optionally fetches fundamental data,
aligns them with forward returns, and returns a single DataFrame ready for
model training / prediction.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from mp.data.fetcher import get_daily_bars
from mp.backtest.ic_analysis import (
    # Original 8 level factors
    rolling_rsi,
    rolling_macd_hist,
    rolling_bollinger_pctb,
    rolling_kdj_j,
    rolling_vol_price_ratio,
    rolling_momentum,
    rolling_volatility,
    # Original 5 inflection factors
    rolling_rsi_delta,
    rolling_macd_hist_delta,
    rolling_bollinger_pctb_delta,
    rolling_momentum_accel,
    rolling_volume_trend,
    # New MA factors
    rolling_ma_deviation,
    rolling_ma_alignment,
    # New ATR / range
    rolling_atr,
    rolling_price_range,
    # New oscillators
    rolling_williams_r,
    rolling_adx,
    # New volume factors
    rolling_obv_slope,
    rolling_amount_ratio,
    # Turnover factors
    rolling_turnover_ma,
    rolling_turnover_percentile,
    # Return distribution
    rolling_return_skew,
    rolling_return_kurtosis,
    rolling_updown_vol_ratio,
    rolling_max_drawdown,
    rolling_return_autocorr,
    # K-line shape
    rolling_close_position,
    rolling_upper_shadow,
    rolling_lower_shadow,
    rolling_body_ratio,
    rolling_gap,
    # Liquidity
    rolling_amihud_illiquidity,
    rolling_volume_volatility,
    # Short momentum
    rolling_momentum_5d,
    rolling_momentum_10d,
    # Cross-field combinations
    rolling_vwap_deviation,
    rolling_bollinger_bandwidth,
    rolling_vol_price_corr,
    rolling_consecutive_days,
    rolling_high_distance,
    rolling_low_distance,
    rolling_vol_ratio,
    rolling_mfi,
    rolling_intraday_intensity,
    rolling_cci,
    rolling_return_extremes_ratio,
    rolling_amount_volatility,
)

# -----------------------------------------------------------------------
# Stable feature column names expected by downstream models.
# -----------------------------------------------------------------------

# Technical factors (computed from price/volume/amount/turnover)
TECHNICAL_COLUMNS: List[str] = [
    # --- original 13 ---
    "rsi_14",
    "macd_hist",
    "boll_pctb",
    "kdj_j",
    "vol_price_ratio",
    "mom_20d",
    "mom_60d",
    "volatility_20d",
    "rsi_delta",
    "macd_hist_delta",
    "boll_pctb_delta",
    "mom_accel",
    "volume_trend",
    # --- new MA factors ---
    "close_ma5_dev",
    "close_ma20_dev",
    "close_ma60_dev",
    "ma_alignment",
    # --- new ATR / range ---
    "atr_14",
    "price_range_10d",
    # --- new oscillators ---
    "williams_r",
    "adx_14",
    # --- new volume/amount ---
    "obv_slope",
    "amount_ratio",
    # --- turnover ---
    "turnover_5d",
    "turnover_pctile",
    # --- return distribution ---
    "return_skew_20d",
    "return_kurtosis_20d",
    "updown_vol_ratio",
    "max_drawdown_20d",
    "return_autocorr",
    # --- K-line shape ---
    "close_position",
    "upper_shadow",
    "lower_shadow",
    "body_ratio",
    "gap_5d",
    # --- liquidity ---
    "amihud_illiq",
    "volume_volatility",
    # --- short momentum ---
    "mom_5d",
    "mom_10d",
    # --- cross-field combinations ---
    "vwap_dev",
    "boll_bandwidth",
    "vol_price_corr",
    "consecutive_days",
    "high_distance_60d",
    "low_distance_60d",
    "vol_ratio_5_60",
    "mfi_14",
    "intraday_intensity",
    "cci_20",
    "return_extremes_ratio",
    "amount_volatility",
]

# Fundamental factors (from financial reports, quasi-static)
FUNDAMENTAL_COLUMNS: List[str] = [
    "pe_ttm",
    "pb",
    "total_mv_log",
    "roe",
    "revenue_growth",
    "profit_growth",
]

# All factor columns
FACTOR_COLUMNS: List[str] = TECHNICAL_COLUMNS + FUNDAMENTAL_COLUMNS


def _compute_technical_factors(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    amount: np.ndarray,
    turnover: np.ndarray,
    open_arr: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute all technical rolling factors and return as {col_name: array}."""
    return {
        # --- original 13 ---
        "rsi_14": rolling_rsi(close, period=14),
        "macd_hist": rolling_macd_hist(close, fast=12, slow=26, sig=9),
        "boll_pctb": rolling_bollinger_pctb(close, period=20, num_std=2.0),
        "kdj_j": rolling_kdj_j(high, low, close, n_period=9, m1=3, m2=3),
        "vol_price_ratio": rolling_vol_price_ratio(close, volume, window=5),
        "mom_20d": rolling_momentum(close, period=20),
        "mom_60d": rolling_momentum(close, period=60),
        "volatility_20d": rolling_volatility(close, period=20),
        "rsi_delta": rolling_rsi_delta(close, period=14, lag=5),
        "macd_hist_delta": rolling_macd_hist_delta(close, lag=5),
        "boll_pctb_delta": rolling_bollinger_pctb_delta(close, lag=5),
        "mom_accel": rolling_momentum_accel(close),
        "volume_trend": rolling_volume_trend(volume, short=5, long=20),
        # --- new MA factors ---
        "close_ma5_dev": rolling_ma_deviation(close, 5),
        "close_ma20_dev": rolling_ma_deviation(close, 20),
        "close_ma60_dev": rolling_ma_deviation(close, 60),
        "ma_alignment": rolling_ma_alignment(close),
        # --- new ATR / range ---
        "atr_14": rolling_atr(high, low, close, period=14),
        "price_range_10d": rolling_price_range(high, low, close, period=10),
        # --- new oscillators ---
        "williams_r": rolling_williams_r(high, low, close, period=14),
        "adx_14": rolling_adx(high, low, close, period=14),
        # --- new volume/amount ---
        "obv_slope": rolling_obv_slope(close, volume, period=20),
        "amount_ratio": rolling_amount_ratio(amount, period=20),
        # --- turnover ---
        "turnover_5d": rolling_turnover_ma(turnover, period=5),
        "turnover_pctile": rolling_turnover_percentile(turnover, period=60),
        # --- return distribution ---
        "return_skew_20d": rolling_return_skew(close, period=20),
        "return_kurtosis_20d": rolling_return_kurtosis(close, period=20),
        "updown_vol_ratio": rolling_updown_vol_ratio(close, period=20),
        "max_drawdown_20d": rolling_max_drawdown(close, period=20),
        "return_autocorr": rolling_return_autocorr(close, period=20),
        # --- K-line shape ---
        "close_position": rolling_close_position(high, low, close, period=5),
        "upper_shadow": rolling_upper_shadow(high, open_arr, close, period=5),
        "lower_shadow": rolling_lower_shadow(low, open_arr, close, period=5),
        "body_ratio": rolling_body_ratio(high, low, open_arr, close, period=5),
        "gap_5d": rolling_gap(open_arr, close, period=5),
        # --- liquidity ---
        "amihud_illiq": rolling_amihud_illiquidity(close, amount, period=20),
        "volume_volatility": rolling_volume_volatility(volume, period=20),
        # --- short momentum ---
        "mom_5d": rolling_momentum_5d(close),
        "mom_10d": rolling_momentum_10d(close),
        # --- cross-field combinations ---
        "vwap_dev": rolling_vwap_deviation(close, amount, volume, period=20),
        "boll_bandwidth": rolling_bollinger_bandwidth(close, period=20),
        "vol_price_corr": rolling_vol_price_corr(close, volume, period=20),
        "consecutive_days": rolling_consecutive_days(close),
        "high_distance_60d": rolling_high_distance(close, high, period=60),
        "low_distance_60d": rolling_low_distance(close, low, period=60),
        "vol_ratio_5_60": rolling_vol_ratio(close, short=5, long=60),
        "mfi_14": rolling_mfi(high, low, close, volume, period=14),
        "intraday_intensity": rolling_intraday_intensity(high, low, close, volume, period=20),
        "cci_20": rolling_cci(high, low, close, period=20),
        "return_extremes_ratio": rolling_return_extremes_ratio(close, period=20),
        "amount_volatility": rolling_amount_volatility(amount, period=20),
    }


def _fetch_financial_history(code: str) -> Optional[pd.DataFrame]:
    """Fetch historical financial data with publish dates for time-alignment.

    Returns DataFrame with columns: publish_date, roe, revenue_growth, profit_growth.
    Sorted by publish_date ascending.
    """
    try:
        from mp.data.fetcher import get_financial_data
        fin = get_financial_data(code)
        if fin.empty:
            return None
        fin = fin.dropna(subset=["publish_date"]).sort_values("publish_date")
        if fin.empty:
            return None
        return fin[["publish_date", "roe", "revenue_growth", "profit_growth"]].copy()
    except Exception:
        return None


def _align_fundamentals_to_dates(
    dates: pd.Series,
    fin_hist: Optional[pd.DataFrame],
    valuation_row: Optional[Dict[str, float]],
) -> Dict[str, np.ndarray]:
    """Align fundamental data to a date series using publish_date (no look-ahead).

    For ROE/growth: uses merge_asof on publish_date — each trading day sees
    only the most recent report published BEFORE that date.

    For PE/PB/market_cap: these are price-derived and change daily. Currently
    we only have today's snapshot, so we fill them only for the latest row
    (used by build_latest_features). For training, these columns will be NaN.
    """
    n = len(dates)
    result: Dict[str, np.ndarray] = {col: np.full(n, np.nan) for col in FUNDAMENTAL_COLUMNS}

    # --- ROE / growth: time-aligned via publish_date ---
    if fin_hist is not None and not fin_hist.empty:
        dates_df = pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date")
        fin_hist = fin_hist.copy()
        fin_hist["publish_date"] = pd.to_datetime(fin_hist["publish_date"])

        merged = pd.merge_asof(
            dates_df, fin_hist,
            left_on="date", right_on="publish_date",
            direction="backward",
        )
        # Map back to original order
        date_to_idx = {d: i for i, d in enumerate(dates)}
        for _, row in merged.iterrows():
            idx = date_to_idx.get(row["date"])
            if idx is not None:
                for col in ["roe", "revenue_growth", "profit_growth"]:
                    if pd.notna(row.get(col)):
                        result[col][idx] = float(row[col])

    # --- PE / PB / market cap: snapshot only (for latest features) ---
    if valuation_row:
        # Only set the last row (latest date) to avoid broadcasting future info
        for col in ["pe_ttm", "pb", "total_mv_log"]:
            if col in valuation_row and pd.notna(valuation_row.get(col)):
                result[col][-1] = valuation_row[col]

    return result


def _fetch_valuation_snapshot_map(codes: List[str]) -> Dict[str, Dict[str, float]]:
    """Fetch today's PE/PB/market_cap for live prediction (not training).

    Tries EM first (fast, has PE/PB), falls back to Sina (slow, no PE/PB).
    For small code lists, skips the full snapshot and fetches per-stock.
    """
    result: Dict[str, Dict[str, float]] = {}

    # For small lists (<20 stocks), skip the 5500-stock full snapshot
    if len(codes) < 20:
        logger.info("Small code list ({}), fetching per-stock financials only", len(codes))
        return result  # PE/PB will be NaN, but ROE/growth are more important anyway

    try:
        from mp.data.fetcher import get_valuation_snapshot
        snap = get_valuation_snapshot()
        if not snap.empty:
            snap["code"] = snap["code"].astype(str).str.zfill(6)
            code_set = set(codes)
            for _, row in snap.iterrows():
                code = row["code"]
                if code in code_set:
                    mv = row.get("total_mv")
                    result[code] = {
                        "pe_ttm": float(row["pe_ttm"]) if pd.notna(row.get("pe_ttm")) else np.nan,
                        "pb": float(row["pb"]) if pd.notna(row.get("pb")) else np.nan,
                        "total_mv_log": float(np.log(mv)) if pd.notna(mv) and mv > 0 else np.nan,
                    }
    except Exception as e:
        logger.warning("Valuation snapshot failed: {}", e)
    return result


def _process_single_stock(
    code: str,
    start: str,
    end: Optional[str],
    horizon: Optional[int],
    fin_hist: Optional[pd.DataFrame] = None,
    valuation_row: Optional[Dict[str, float]] = None,
) -> Optional[pd.DataFrame]:
    """Fetch bars and build factor rows for one stock.

    Parameters
    ----------
    code : str
        6-digit A-share stock code.
    start, end : str
        Date range in ``YYYYMMDD`` format.
    horizon : int or None
        If given, compute ``fwd_ret`` (forward return over *horizon* trading
        days).  ``None`` means skip forward-return computation.
    fin_hist : DataFrame or None
        Historical financial reports with publish_date for time-alignment.
    valuation_row : dict or None
        Today's PE/PB/market_cap snapshot (only used for latest features).

    Returns
    -------
    DataFrame or None
    """
    df = get_daily_bars(code, start, end)
    if df is None or df.empty:
        logger.warning("No data returned for {}", code)
        return None

    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    open_arr = df["open"].values.astype(float)
    amount = df["amount"].values.astype(float) if "amount" in df.columns else volume * close
    turnover = df["turnover"].values.astype(float) if "turnover" in df.columns else np.full(len(close), np.nan)

    if len(close) < 80:
        logger.warning("Insufficient bars for {} ({} bars)", code, len(close))
        return None

    # --- technical factors ---
    factors = _compute_technical_factors(close, high, low, volume, amount, turnover, open_arr)

    result = pd.DataFrame({"date": df["date"], "code": code})
    for col in TECHNICAL_COLUMNS:
        result[col] = factors[col]

    # --- fundamental factors (time-aligned via publish_date) ---
    fund_aligned = _align_fundamentals_to_dates(df["date"], fin_hist, valuation_row)
    for col in FUNDAMENTAL_COLUMNS:
        result[col] = fund_aligned[col]

    # --- forward return (optional) ---
    if horizon is not None:
        n = len(close)
        fwd = np.full(n, np.nan)
        for i in range(n - horizon):
            if close[i] > 0:
                fwd[i] = close[i + horizon] / close[i] - 1.0
        result["fwd_ret"] = fwd

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset(
    codes: List[str],
    start: str,
    end: Optional[str] = None,
    horizon: int = 20,
    include_fundamentals: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Build a cross-sectional training dataset for LightGBM.

    Parameters
    ----------
    codes : list[str]
        Universe of 6-digit stock codes.
    start, end : str
        Date range in ``YYYYMMDD`` format.  *end* defaults to today.
    horizon : int
        Number of trading days for the forward-return label.
    include_fundamentals : bool
        If True, fetch PE/PB/ROE etc. Adds ~2 min for 500 stocks.
    progress_callback : callable, optional
        ``callback(current, total)`` invoked after each stock is processed.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``code``, factor columns, ``fwd_ret``.
    """
    total = len(codes)
    logger.info("build_dataset: {} codes, start={}, end={}, horizon={}, fundamentals={}",
                total, start, end, horizon, include_fundamentals)

    # Pre-fetch financial histories (per-stock, with publish_date for time-alignment)
    fin_hist_map: Dict[str, Optional[pd.DataFrame]] = {}
    if include_fundamentals:
        logger.info("Fetching financial report histories...")
        for i, code in enumerate(codes):
            fin_hist_map[code] = _fetch_financial_history(code)
            if (i + 1) % 50 == 0:
                logger.info("Financial data progress: {}/{}", i + 1, total)
        n_ok = sum(1 for v in fin_hist_map.values() if v is not None)
        logger.info("Got financial history for {}/{} stocks", n_ok, total)

    frames: List[pd.DataFrame] = []
    for idx, code in enumerate(codes):
        try:
            fin_hist = fin_hist_map.get(code) if include_fundamentals else None
            part = _process_single_stock(code, start, end, horizon,
                                         fin_hist=fin_hist, valuation_row=None)
            if part is not None:
                frames.append(part)
        except Exception:
            logger.warning("Failed to process {}, skipping", code)

        if progress_callback is not None:
            progress_callback(idx + 1, total)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            logger.info("build_dataset progress: {}/{}", idx + 1, total)

    if not frames:
        logger.error("build_dataset: no valid data produced")
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)

    # Drop rows where ALL technical factors are NaN (warm-up period).
    # Allow some NaN in fundamentals and turnover (optional data).
    before = len(dataset)
    # Use original 13 core factors as required columns
    core_cols = TECHNICAL_COLUMNS[:13]
    dataset = dataset.dropna(subset=core_cols)
    logger.debug("Dropped {} warm-up rows (core factor NaN)", before - len(dataset))

    # Drop rows where fwd_ret is NaN (last horizon rows of each stock).
    if "fwd_ret" in dataset.columns:
        before = len(dataset)
        dataset = dataset.dropna(subset=["fwd_ret"])
        logger.debug("Dropped {} tail rows (fwd_ret NaN)", before - len(dataset))

    logger.info(
        "build_dataset complete: {} rows, {} stocks, {} factors, date range {} ~ {}",
        len(dataset),
        dataset["code"].nunique(),
        len(FACTOR_COLUMNS),
        dataset["date"].min().strftime("%Y-%m-%d") if len(dataset) else "N/A",
        dataset["date"].max().strftime("%Y-%m-%d") if len(dataset) else "N/A",
    )
    return dataset


def build_latest_features(
    codes: List[str],
    start: str = "20230101",
    include_fundamentals: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Return the latest feature row per stock for live prediction.

    Same factor computation as :func:`build_dataset` but:
    - No ``fwd_ret`` column (future is unknown).
    - Only the most recent valid row per stock is kept.
    - Fundamentals use latest financial report + today's PE/PB snapshot.
    """
    total = len(codes)
    logger.info("build_latest_features: {} codes, start={}", total, start)

    # For live prediction: get today's valuation snapshot + financial history
    valuation_map: Dict[str, Dict[str, float]] = {}
    fin_hist_map: Dict[str, Optional[pd.DataFrame]] = {}
    if include_fundamentals:
        valuation_map = _fetch_valuation_snapshot_map(codes)
        logger.info("Got valuation snapshot for {} stocks", len(valuation_map))
        for code in codes:
            fin_hist_map[code] = _fetch_financial_history(code)
        n_ok = sum(1 for v in fin_hist_map.values() if v is not None)
        logger.info("Got financial history for {}/{} stocks", n_ok, total)

    rows: List[pd.DataFrame] = []
    core_cols = TECHNICAL_COLUMNS[:13]

    for idx, code in enumerate(codes):
        try:
            fin_hist = fin_hist_map.get(code) if include_fundamentals else None
            val_row = valuation_map.get(code) if include_fundamentals else None
            part = _process_single_stock(code, start, None, horizon=None,
                                         fin_hist=fin_hist, valuation_row=val_row)
            if part is not None:
                clean = part.dropna(subset=core_cols)
                if not clean.empty:
                    rows.append(clean.iloc[[-1]])
        except Exception:
            logger.warning("Failed to process {}, skipping", code)

        if progress_callback is not None:
            progress_callback(idx + 1, total)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            logger.info("build_latest_features progress: {}/{}", idx + 1, total)

    if not rows:
        logger.error("build_latest_features: no valid data produced")
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    logger.info("build_latest_features complete: {} stocks, {} factors", len(result), len(FACTOR_COLUMNS))
    return result
