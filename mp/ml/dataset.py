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

# Fundamental trend factors: quarter-over-quarter changes in financial metrics.
# Captures whether fundamentals are improving or deteriorating.
FUNDAMENTAL_TREND_COLUMNS: List[str] = [
    "roe_qoq",              # ROE this period - ROE last period
    "profit_growth_accel",  # profit_growth change vs prior period
    "revenue_growth_accel", # revenue_growth change vs prior period
]

# Industry-relative ranking factors: percentile rank within the stock's
# industry peer group on each date.  Computed as a cross-sectional
# post-processing step after all stocks are assembled.
INDUSTRY_RANK_COLUMNS: List[str] = [
    "pe_ind_rank",      # PE percentile within industry (0=cheapest, 1=most expensive)
    "pb_ind_rank",
    "roe_ind_rank",     # ROE percentile within industry (0=lowest, 1=highest)
    "mom_20d_ind_rank", # 20-day momentum percentile within industry
]

# All factor columns
FACTOR_COLUMNS: List[str] = (
    TECHNICAL_COLUMNS
    + FUNDAMENTAL_COLUMNS
    + FUNDAMENTAL_TREND_COLUMNS
    + INDUSTRY_RANK_COLUMNS
)

# Label column names
EXCESS_LABEL = "excess_ret"

# Curated factors selected by cross-sectional IC analysis (|ICIR| >= 0.15).
# 34 noise factors removed — improves out-of-sample generalization.
CURATED_COLUMNS: List[str] = [
    # STRONG (|ICIR| >= 0.5)
    "amihud_illiq",
    # MODERATE (|ICIR| >= 0.3)
    "vwap_dev",
    # WEAK (0.15 <= |ICIR| < 0.3)
    "amount_volatility",
    "ma_alignment",
    "close_ma60_dev",
    "low_distance_60d",
    "volume_volatility",
    "mom_20d",
    "volatility_20d",
    "price_range_10d",
    "upper_shadow",
    "mom_accel",
    "lower_shadow",
    "turnover_5d",
    "mom_60d",
    "atr_14",
    "obv_slope",
    "gap_5d",
    "rsi_14",
    "close_ma20_dev",
    "turnover_pctile",
    "boll_bandwidth",
    "intraday_intensity",
]


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

    Returns DataFrame with columns:
        publish_date, roe, revenue_growth, profit_growth,
        roe_qoq, profit_growth_accel, revenue_growth_accel
    Sorted by publish_date ascending.  The *_qoq / *_accel columns are
    quarter-over-quarter differences computed from consecutive reports.
    """
    try:
        from mp.data.fetcher import get_financial_data
        fin = get_financial_data(code)
        if fin.empty:
            return None
        fin = fin.dropna(subset=["publish_date"]).sort_values("publish_date").reset_index(drop=True)
        if fin.empty:
            return None
        fin = fin[["publish_date", "roe", "revenue_growth", "profit_growth"]].copy()
        # Quarter-over-quarter deltas: positive = improving, negative = deteriorating
        fin["roe_qoq"] = fin["roe"].diff()
        fin["profit_growth_accel"] = fin["profit_growth"].diff()
        fin["revenue_growth_accel"] = fin["revenue_growth"].diff()
        return fin
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
    _fund_cols = FUNDAMENTAL_COLUMNS + FUNDAMENTAL_TREND_COLUMNS
    result: Dict[str, np.ndarray] = {col: np.full(n, np.nan) for col in _fund_cols}

    # --- ROE / growth + trend diffs: time-aligned via publish_date ---
    _align_cols = ["roe", "revenue_growth", "profit_growth",
                   "roe_qoq", "profit_growth_accel", "revenue_growth_accel"]
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
                for col in _align_cols:
                    if col in row and pd.notna(row.get(col)):
                        result[col][idx] = float(row[col])

    # --- PE / PB / market cap: snapshot only (for latest features) ---
    if valuation_row:
        # Only set the last row (latest date) to avoid broadcasting future info
        for col in ["pe_ttm", "pb", "total_mv_log"]:
            if col in valuation_row and pd.notna(valuation_row.get(col)):
                result[col][-1] = valuation_row[col]

    return result


def _fetch_pe_pb_baidu(code: str) -> Dict[str, float]:
    """Fetch latest PE(TTM) and PB from Baidu Finance (per-stock, ~0.5s each)."""
    import akshare as ak
    result: Dict[str, float] = {}
    try:
        pe_df = ak.stock_zh_valuation_baidu(symbol=code, indicator="市盈率(TTM)", period="近一年")
        if not pe_df.empty:
            result["pe_ttm"] = float(pe_df.iloc[-1]["value"])
    except Exception:
        pass
    try:
        pb_df = ak.stock_zh_valuation_baidu(symbol=code, indicator="市净率", period="近一年")
        if not pb_df.empty:
            result["pb"] = float(pb_df.iloc[-1]["value"])
    except Exception:
        pass
    return result


def _fetch_valuation_snapshot_map(codes: List[str]) -> Dict[str, Dict[str, float]]:
    """Fetch today's PE/PB/market_cap for live prediction (not training).

    Uses SQLite cache (valuation table) for today's data first, then falls
    back to API (EM → Sina → Baidu) for codes not yet cached. Successful
    API results are written back to cache for subsequent calls.
    """
    from datetime import date as _date
    from mp.data.store import DataStore

    today_str = _date.today().strftime("%Y-%m-%d")
    store = DataStore()
    result: Dict[str, Dict[str, float]] = {}

    # --- Step 1: load from SQLite cache ---
    try:
        cached = store.load_valuation(codes=codes, date_str=today_str)
        if not cached.empty:
            cached["code"] = cached["code"].astype(str).str.zfill(6)
            for _, row in cached.iterrows():
                code = row["code"]
                mv = row.get("total_mv")
                entry = {
                    "pe_ttm": float(row["pe_ttm"]) if pd.notna(row.get("pe_ttm")) else np.nan,
                    "pb": float(row["pb"]) if pd.notna(row.get("pb")) else np.nan,
                    "total_mv_log": float(np.log(mv)) if pd.notna(mv) and mv > 0 else np.nan,
                }
                # Only use cache if PE/PB are actually present
                if not np.isnan(entry["pe_ttm"]) or not np.isnan(entry["pb"]):
                    result[code] = entry
            if result:
                logger.info("Valuation cache hit: {}/{} stocks from SQLite", len(result), len(codes))
    except Exception as e:
        logger.debug("Valuation cache read failed: {}", e)

    # --- Step 2: API fetch for codes not in cache ---
    uncached = [c for c in codes if c not in result]
    api_fetched: Dict[str, Dict[str, float]] = {}

    if uncached:
        try:
            from mp.data.fetcher import get_valuation_snapshot
            snap = get_valuation_snapshot()
            if not snap.empty:
                snap["code"] = snap["code"].astype(str).str.zfill(6)
                uncached_set = set(uncached)
                for _, row in snap.iterrows():
                    code = row["code"]
                    if code in uncached_set:
                        mv = row.get("total_mv")
                        entry = {
                            "pe_ttm": float(row["pe_ttm"]) if pd.notna(row.get("pe_ttm")) else np.nan,
                            "pb": float(row["pb"]) if pd.notna(row.get("pb")) else np.nan,
                            "total_mv_log": float(np.log(mv)) if pd.notna(mv) and mv > 0 else np.nan,
                        }
                        result[code] = entry
                        api_fetched[code] = {
                            "pe_ttm": entry["pe_ttm"],
                            "pb": entry["pb"],
                            "total_mv": float(mv) if pd.notna(mv) else np.nan,
                        }
        except Exception as e:
            logger.warning("Valuation snapshot failed: {}", e)

    # --- Step 3: Baidu fallback for codes missing PE/PB ---
    # Raised limit: use concurrent fetch for large batches (no longer cap at 20)
    missing_pepb = [c for c in codes if
                    c not in result or
                    np.isnan(result[c].get("pe_ttm", np.nan)) or
                    np.isnan(result[c].get("pb", np.nan))]
    if missing_pepb:
        logger.info("Baidu fallback: fetching PE/PB for {} stocks (concurrent)", len(missing_pepb))
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        def _baidu_one(code):
            return code, _fetch_pe_pb_baidu(code)
        workers = min(15, len(missing_pepb))
        with ThreadPoolExecutor(max_workers=workers) as _pool:
            futs = {_pool.submit(_baidu_one, c): c for c in missing_pepb}
            for fut in _as_completed(futs):
                code, baidu = fut.result()
                if not baidu:
                    continue
                if code not in result:
                    result[code] = {"pe_ttm": np.nan, "pb": np.nan, "total_mv_log": np.nan}
                if "pe_ttm" in baidu:
                    result[code]["pe_ttm"] = baidu["pe_ttm"]
                if "pb" in baidu:
                    result[code]["pb"] = baidu["pb"]
                if code not in api_fetched:
                    api_fetched[code] = {"pe_ttm": np.nan, "pb": np.nan, "total_mv": np.nan}
                if "pe_ttm" in baidu:
                    api_fetched[code]["pe_ttm"] = baidu["pe_ttm"]
                if "pb" in baidu:
                    api_fetched[code]["pb"] = baidu["pb"]
        baidu_ok = sum(1 for c in missing_pepb if c in result and not np.isnan(result[c].get("pe_ttm", np.nan)))
        logger.info("Baidu PE/PB: {}/{} recovered", baidu_ok, len(missing_pepb))

    # --- Step 4: write back newly fetched data to SQLite cache ---
    # Only persist entries that have at least PE or PB — never cache NULL PE/PB rows,
    # as that would poison the cache and block future real-data fetches.
    if api_fetched:
        try:
            rows = []
            for code, vals in api_fetched.items():
                pe = vals.get("pe_ttm", np.nan)
                pb = vals.get("pb", np.nan)
                if np.isnan(pe) and np.isnan(pb):
                    continue  # skip: no useful valuation data to cache
                rows.append({
                    "code": code,
                    "date": today_str,
                    "name": "",
                    "close": np.nan,
                    "pe_ttm": pe,
                    "pb": pb,
                    "total_mv": vals.get("total_mv", np.nan),
                })
            if rows:
                cache_df = pd.DataFrame(rows)
                store.save_valuation(cache_df)
                logger.info("Valuation cache saved: {} stocks for {}", len(rows), today_str)
        except Exception as e:
            logger.warning("Valuation cache write failed: {}", e)

    return result


def _process_single_stock(
    code: str,
    start: str,
    end: Optional[str],
    horizon: Optional[int],
    fin_hist: Optional[pd.DataFrame] = None,
    valuation_row: Optional[Dict[str, float]] = None,
    intraday_bar: Optional[Dict] = None,
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

    # Inject intraday partial bar (e.g. morning session data for midday report)
    if intraday_bar is not None:
        today_dt = pd.Timestamp(intraday_bar["date"])
        df = df[df["date"] < today_dt]  # remove any stale today row
        today_row = pd.DataFrame([{
            "code": code,
            "date": today_dt,
            "open": intraday_bar["open"],
            "high": intraday_bar["high"],
            "low": intraday_bar["low"],
            "close": intraday_bar["close"],
            "volume": intraday_bar["volume"],
            "amount": intraday_bar["amount"],
            "turnover": float("nan"),
        }])
        df = pd.concat([df, today_row], ignore_index=True)

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
    for col in FUNDAMENTAL_COLUMNS + FUNDAMENTAL_TREND_COLUMNS:
        result[col] = fund_aligned[col]

    # Derive total_mv_log from daily bars (close * volume / turnover).
    # Only overwrite the valuation-cache value when the bar-derived estimate is valid;
    # Sina daily bars lack turnover, so we fall back to the cached total_mv_log
    # rather than replacing it with NaN.
    valid_t = turnover > 0
    float_mv = np.where(valid_t, close * volume / turnover, np.nan)
    bars_mv_log = np.where(float_mv > 0, np.log(float_mv), np.nan)
    cached_mv_log = result["total_mv_log"].values.copy()
    result["total_mv_log"] = np.where(~np.isnan(bars_mv_log), bars_mv_log, cached_mv_log)

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
# Benchmark (ZZ500) forward returns
# ---------------------------------------------------------------------------

def _compute_benchmark_fwd_ret(
    dates: pd.Series, horizon: int = 20
) -> pd.DataFrame:
    """Compute ZZ500 forward return for each trading date.

    Returns DataFrame with columns ``date`` and ``bench_fwd_ret``, one row per
    unique date in *dates*.
    """
    import akshare as ak

    idx = ak.stock_zh_index_daily(symbol="sh000905")
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    close = idx.set_index("date")["close"]

    unique_dates = pd.Series(sorted(dates.unique()))
    bench = []
    for dt in unique_dates:
        # Find the closest date <= dt in the index
        mask = close.index <= dt
        if not mask.any():
            bench.append({"date": dt, "bench_fwd_ret": np.nan})
            continue
        t_close = close.loc[mask].iloc[-1]

        # Forward date: find the trading date ~horizon days ahead
        future_mask = close.index > dt
        future_closes = close.loc[future_mask]
        if len(future_closes) < horizon:
            bench.append({"date": dt, "bench_fwd_ret": np.nan})
            continue
        t_fwd_close = future_closes.iloc[horizon - 1]
        bench.append({"date": dt, "bench_fwd_ret": t_fwd_close / t_close - 1.0})

    return pd.DataFrame(bench)


# ---------------------------------------------------------------------------
# Industry-relative cross-sectional ranking (post-processing)
# ---------------------------------------------------------------------------

def _add_industry_relative_features(
    df: pd.DataFrame,
    code_to_industry: Dict[str, str],
) -> pd.DataFrame:
    """Add industry-relative percentile rank columns to a cross-sectional panel.

    For each (date, industry) group, computes the within-group percentile rank
    for PE, PB, ROE, and 20-day momentum.  Stocks not in *code_to_industry*
    get NaN for all rank columns (LightGBM handles NaN natively).

    Parameters
    ----------
    df : pd.DataFrame
        Panel with columns ``code``, ``date``, ``pe_ttm``, ``pb``,
        ``roe``, ``mom_20d``.
    code_to_industry : dict
        {code: industry_name} mapping from ``get_industry_mapping()``.

    Returns
    -------
    The same DataFrame with INDUSTRY_RANK_COLUMNS added in-place.
    """
    if df.empty or not code_to_industry:
        for col in INDUSTRY_RANK_COLUMNS:
            df[col] = np.nan
        return df

    df = df.copy()
    df["_industry"] = df["code"].map(code_to_industry)

    _rank_map = {
        "pe_ind_rank": "pe_ttm",
        "pb_ind_rank": "pb",
        "roe_ind_rank": "roe",
        "mom_20d_ind_rank": "mom_20d",
    }

    for rank_col, src_col in _rank_map.items():
        if src_col not in df.columns:
            df[rank_col] = np.nan
            continue
        # Within each (date, industry) group, compute 0-1 percentile rank.
        # min_periods=1 allows groups with a single stock (rank=0.5 for lone member).
        df[rank_col] = df.groupby(["date", "_industry"], observed=True)[src_col].transform(
            lambda x: x.rank(pct=True, na_option="keep")
        )

    df.drop(columns=["_industry"], inplace=True)
    n_mapped = df["pe_ind_rank"].notna().sum()
    logger.debug(
        "Industry rank features computed: {}/{} rows have industry assignment",
        n_mapped, len(df),
    )
    return df


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

    # Compute excess return (fwd_ret - ZZ500 fwd_ret)
    if "fwd_ret" in dataset.columns and len(dataset) > 0:
        try:
            bench = _compute_benchmark_fwd_ret(dataset["date"], horizon=horizon)
            dataset = dataset.merge(bench, on="date", how="left")
            dataset[EXCESS_LABEL] = dataset["fwd_ret"] - dataset["bench_fwd_ret"]
            dataset.drop(columns=["bench_fwd_ret"], inplace=True)
            n_valid = dataset[EXCESS_LABEL].notna().sum()
            logger.info("Excess return computed: {}/{} rows with valid bench", n_valid, len(dataset))
        except Exception as e:
            logger.warning("Failed to compute benchmark fwd_ret, excess_ret unavailable: {}", e)

    # --- Industry-relative ranking features (cross-sectional post-processing) ---
    if include_fundamentals and len(dataset) > 0:
        try:
            from mp.data.fetcher import get_industry_mapping
            code_to_industry = get_industry_mapping(universe=codes)
            dataset = _add_industry_relative_features(dataset, code_to_industry)
            n_ranked = dataset["pe_ind_rank"].notna().sum()
            logger.info("Industry rank features added: {}/{} rows", n_ranked, len(dataset))
        except Exception as e:
            logger.warning("Industry rank features failed, skipping: {}", e)
            for col in INDUSTRY_RANK_COLUMNS:
                if col not in dataset.columns:
                    dataset[col] = np.nan

    logger.info(
        "build_dataset complete: {} rows, {} stocks, {} factors, date range {} ~ {}",
        len(dataset),
        dataset["code"].nunique(),
        len(FACTOR_COLUMNS),
        dataset["date"].min().strftime("%Y-%m-%d") if len(dataset) else "N/A",
        dataset["date"].max().strftime("%Y-%m-%d") if len(dataset) else "N/A",
    )
    return dataset


def add_excess_ret(df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Add ``excess_ret`` column to a panel that already has ``fwd_ret``.

    Useful for cached datasets (e.g. ``data/wf_cache/factors.parquet``) that
    were built before benchmark subtraction was implemented.
    """
    if EXCESS_LABEL in df.columns:
        logger.debug("excess_ret already present, skipping")
        return df
    if "fwd_ret" not in df.columns:
        logger.warning("No fwd_ret column, cannot compute excess_ret")
        return df
    try:
        bench = _compute_benchmark_fwd_ret(df["date"], horizon=horizon)
        df = df.merge(bench, on="date", how="left")
        df[EXCESS_LABEL] = df["fwd_ret"] - df["bench_fwd_ret"]
        df.drop(columns=["bench_fwd_ret"], inplace=True)
        logger.info("Added excess_ret to {} rows", df[EXCESS_LABEL].notna().sum())
    except Exception as e:
        logger.warning("Failed to add excess_ret: {}", e)
    return df


def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove noisy samples: extreme illiquidity and micro-cap.

    Drops the top 5 % by ``amihud_illiq`` and bottom 5 % by ``total_mv_log``
    within each date cross-section.  These are per-date cutoffs so the filter
    adapts to changing market conditions.
    """
    n_before = len(df)

    # 1. Extreme illiquidity (top 5 % per date)
    if "amihud_illiq" in df.columns and df["amihud_illiq"].notna().any():
        thr = df.groupby("date")["amihud_illiq"].transform(lambda x: x.quantile(0.95))
        df = df[df["amihud_illiq"] <= thr].reset_index(drop=True)

    # 2. Micro-cap (bottom 5 % per date)
    if "total_mv_log" in df.columns and df["total_mv_log"].notna().any():
        thr = df.groupby("date")["total_mv_log"].transform(lambda x: x.quantile(0.05))
        df = df[df["total_mv_log"] >= thr].reset_index(drop=True)

    logger.info("filter_universe: {} → {} rows ({} removed)",
                n_before, len(df), n_before - len(df))
    return df


def build_latest_features(
    codes: List[str],
    start: str = "20230101",
    include_fundamentals: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    intraday_bars: Optional[Dict[str, Dict]] = None,
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
            bar = intraday_bars.get(code) if intraday_bars else None
            part = _process_single_stock(code, start, None, horizon=None,
                                         fin_hist=fin_hist, valuation_row=val_row,
                                         intraday_bar=bar)
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

    # --- Data quality warnings: flag missing fundamental columns per row ---
    def _make_warning(row):
        missing = [c for c in FUNDAMENTAL_COLUMNS if pd.isna(row.get(c))]
        return ",".join(missing) if missing else ""
    result["_data_warnings"] = result.apply(_make_warning, axis=1)
    n_warn = (result["_data_warnings"] != "").sum()
    # Store global data quality ratio in DataFrame attrs
    result.attrs["_data_quality"] = 1.0 - (n_warn / len(result)) if len(result) > 0 else 0.0
    if n_warn > 0:
        logger.warning("⚠ {}/{} 股票存在基本面数据缺失，预测可能不准", n_warn, len(result))
    if n_warn > len(result) * 0.5:
        logger.warning("⚠ 超过50%股票缺少基本面数据(PE/PB等)，数据源可能异常")

    # --- Industry-relative ranking features ---
    if include_fundamentals and len(result) > 0:
        try:
            from mp.data.fetcher import get_industry_mapping
            code_to_industry = get_industry_mapping(universe=codes)
            result = _add_industry_relative_features(result, code_to_industry)
        except Exception as e:
            logger.warning("Industry rank features failed for live prediction, skipping: {}", e)
            for col in INDUSTRY_RANK_COLUMNS:
                if col not in result.columns:
                    result[col] = np.nan

    logger.info("build_latest_features complete: {} stocks, {} factors", len(result), len(FACTOR_COLUMNS))
    return result
