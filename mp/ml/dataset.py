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

# Winsorization cap for excess_ret training label, applied symmetrically in
# build_dataset() and add_excess_ret(). Recovered from prior-session WIP via
# docs/dialog/ round 20-22 (Claude Code transcript). Value 0.50 was chosen
# empirically by prior session without documented justification; preserved
# here because round-11 walk-forward (with winsorize active) reproduced
# Sharpe 1.90 at 64-feature scale vs 1.53 without it. See P2-fix-1 commit
# message for full caveats. Tuning is a P3 question.
EXCESS_CAP = 0.50

# Curated factors selected by cross-sectional IC analysis (|ICIR| >= 0.15).
# 34 noise factors removed — improves out-of-sample generalization.
#
# DEPRECATED 2026-05-24: walk-forward A/B (docs/dialog/ on branch
# collab/advisor-dialog) shows FACTOR_COLUMNS full set beats any precomputed
# subset by 0.56–0.74 Sharpe on the hs300+zz500 universe with BlendRanker
# + conviction sizing. Ranker defaults now resolve to FACTOR_COLUMNS
# (see mp/ml/model.py and commit a3cb98c). This list is kept ONLY so
# legacy callers passing `feature_cols=CURATED_COLUMNS` explicitly still
# work; new code should NOT reference it.
#
# Historical snapshots of expanded CURATED variants (32/28/30 features)
# are frozen as W0/W1/W2 in mp/ml/feature_presets.py.
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
    # ETFs (fund codes) have no financial statements — skip entirely
    from mp.data.fetcher import _is_etf_code, get_financial_data
    if _is_etf_code(code):
        return None
    try:
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
    valuation_row: Optional[Dict[str, float]] = None,
    valuation_hist: Optional[pd.DataFrame] = None,
) -> Dict[str, np.ndarray]:
    """Align fundamental data to a date series using publish_date (no look-ahead).

    Parameters
    ----------
    dates : pd.Series
        Trading dates for this stock (any order; index is used for mapping).
    fin_hist : DataFrame or None
        Quarterly financial reports with ``publish_date``. Aligned via
        ``merge_asof(direction="backward")`` so each trading day only sees
        reports already announced.
    valuation_row : dict or None
        Latest PE/PB/market-cap snapshot, used **only** for live prediction
        (``build_latest_features``).  Written to the last row of the panel.
    valuation_hist : DataFrame or None
        Historical daily PE/PB/market-cap for this stock
        (columns: ``date``, ``pe_ttm``, ``pb``, ``total_mv``).  When provided,
        PE/PB/total_mv_log are aligned day-by-day (training path).  This takes
        precedence over ``valuation_row`` for the last row as well.

    Returns a dict of column -> np.ndarray aligned to *dates*.
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

    # --- PE / PB / market cap: historical alignment (training) ---
    # valuation_hist may be sparse (e.g. Baidu returns ~biweekly samples).
    # Use merge_asof(backward) to forward-fill without look-ahead: each
    # trading day sees only the most recent observation published on or
    # before it.
    if valuation_hist is not None and not valuation_hist.empty:
        vh = valuation_hist.copy()
        vh["date"] = pd.to_datetime(vh["date"])
        vh = vh.sort_values("date").reset_index(drop=True)

        dates_df = pd.DataFrame({
            "_orig_idx": np.arange(len(dates)),
            "date": pd.to_datetime(dates),
        }).sort_values("date").reset_index(drop=True)

        merged = pd.merge_asof(
            dates_df, vh, on="date", direction="backward",
        )

        if "pe_ttm" in merged.columns:
            pe_vals = merged["pe_ttm"].to_numpy(dtype=float)
            result["pe_ttm"][merged["_orig_idx"].to_numpy()] = pe_vals
        if "pb" in merged.columns:
            pb_vals = merged["pb"].to_numpy(dtype=float)
            result["pb"][merged["_orig_idx"].to_numpy()] = pb_vals
        if "total_mv" in merged.columns:
            mv_vals = merged["total_mv"].to_numpy(dtype=float)
            safe = np.where((~np.isnan(mv_vals)) & (mv_vals > 0), mv_vals, np.nan)
            result["total_mv_log"][merged["_orig_idx"].to_numpy()] = np.log(safe)

    # --- PE / PB / market cap: snapshot only (live prediction path) ---
    # ``valuation_hist`` has already populated historical rows; the snapshot is
    # still useful to cover "today" when the historical table hasn't caught up
    # yet. We only write it when the last row is still NaN to avoid clobbering
    # authoritative historical data.
    if valuation_row:
        for col in ["pe_ttm", "pb", "total_mv_log"]:
            if np.isnan(result[col][-1]) and col in valuation_row and pd.notna(valuation_row.get(col)):
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
    from mp.data.fetcher import _is_etf_code

    today_str = _date.today().strftime("%Y-%m-%d")
    store = DataStore()
    result: Dict[str, Dict[str, float]] = {}

    # ETFs have no PE/PB — exclude from all valuation lookups upfront
    codes = [c for c in codes if not _is_etf_code(c)]

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

    # --- Step 3.5: SQLite historical fallback for any missing valuation field ---
    # Covers three failure modes:
    #   A) pe_ttm missing  — EM down AND Baidu failed
    #   B) total_mv missing — EM down (only source); Sina/Baidu don't carry market cap
    #      *** this is the common midday case: Baidu fills pe/pb but total_mv stays NaN,
    #          causing _data_quality=0 and荐股降级 even though pe/pb are fine ***
    #   C) pb missing  — rare fallback
    #
    # IMPORTANT: We do NOT hardcode yesterday — EM can be down multiple consecutive days.
    # Instead, for each code we pick its most recent row where total_mv IS NOT NULL,
    # going back up to 30 calendar days.  Market cap changes slowly; a few-day-old
    # value is almost always accurate enough to serve as a control variable.
    still_missing_any = [c for c in codes if
                         c not in result
                         or np.isnan(result[c].get("pe_ttm", np.nan))
                         or np.isnan(result[c].get("total_mv_log", np.nan))]
    if still_missing_any:
        try:
            from datetime import timedelta, date as _date_cls
            from sqlalchemy import text as _text
            cutoff_str = (_date_cls.today() - timedelta(days=30)).strftime("%Y-%m-%d")
            # Per-code: find the most recent row where total_mv IS NOT NULL AND total_mv > 0
            placeholders = ",".join(f":c{i}" for i in range(len(still_missing_any)))
            sql = f"""
                SELECT v.*
                FROM valuation v
                INNER JOIN (
                    SELECT code, MAX(date) AS best_date
                    FROM valuation
                    WHERE total_mv IS NOT NULL AND total_mv > 0
                      AND date >= :cutoff
                      AND code IN ({placeholders})
                    GROUP BY code
                ) best ON v.code = best.code AND v.date = best.best_date
            """
            params: dict = {"cutoff": cutoff_str}
            for i, c in enumerate(still_missing_any):
                params[f"c{i}"] = c
            with store.engine.connect() as _conn:
                hist_df = pd.read_sql(_text(sql), _conn, params=params)

            if not hist_df.empty:
                hist_df["code"] = hist_df["code"].astype(str).str.zfill(6)
                mv_recovered = pe_recovered = 0
                best_dates: dict = {}
                for _, row in hist_df.iterrows():
                    code = row["code"]
                    pe = float(row["pe_ttm"]) if pd.notna(row.get("pe_ttm")) else np.nan
                    pb_val = float(row["pb"]) if pd.notna(row.get("pb")) else np.nan
                    mv = row.get("total_mv")
                    if code not in result:
                        result[code] = {"pe_ttm": np.nan, "pb": np.nan, "total_mv_log": np.nan}
                    if np.isnan(result[code].get("pe_ttm", np.nan)) and not np.isnan(pe):
                        result[code]["pe_ttm"] = pe
                        pe_recovered += 1
                        best_dates[code] = row.get("date", "?")
                    if np.isnan(result[code].get("pb", np.nan)) and not np.isnan(pb_val):
                        result[code]["pb"] = pb_val
                    if np.isnan(result[code].get("total_mv_log", np.nan)) and pd.notna(mv) and float(mv) > 0:
                        result[code]["total_mv_log"] = float(np.log(float(mv)))
                        mv_recovered += 1
                        best_dates[code] = row.get("date", "?")
                if pe_recovered or mv_recovered:
                    # Log the actual dates used so we can verify the fallback worked
                    date_range = sorted(set(str(v) for v in best_dates.values()))
                    logger.info("SQLite历史估值兜底(最近有效日期): PE恢复{}只, total_mv恢复{}只, 数据日期={}",
                                pe_recovered, mv_recovered, date_range)
        except Exception as e:
            logger.debug("SQLite历史估值兜底失败: {}", e)

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
    valuation_hist: Optional[pd.DataFrame] = None,
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
    valuation_hist : DataFrame or None
        Pre-filtered (single-stock) daily PE/PB/market_cap history
        (columns: ``date``, ``pe_ttm``, ``pb``, ``total_mv``).  When provided,
        used to populate training-set fundamentals day by day.

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
    fund_aligned = _align_fundamentals_to_dates(
        df["date"], fin_hist, valuation_row=valuation_row, valuation_hist=valuation_hist,
    )
    for col in FUNDAMENTAL_COLUMNS + FUNDAMENTAL_TREND_COLUMNS:
        result[col] = fund_aligned[col]

    # Derive total_mv_log from daily bars (close * volume / turnover).
    #
    # Two value SPACES used to collide here:
    #   (A) bars-derived  = close × volume / turnover_rate = 流通市值 float-cap
    #   (B) cached value  = EM/Sina valuation snapshot total_mv = 总市值 total-cap
    # For most stocks A ≠ B by a factor of 1.5-3× (e.g. 粤电力A: 流通 ~196亿 vs
    # 总 ~374亿 → log diff 0.64).  When the intraday_bar injection path has
    # no turnover, bars_mv_log goes NaN and we used to fall back to the cached
    # value — silently switching the feature's value space mid-pipeline.
    # Since training data uses one space per stock-day, the model sees the
    # midday cross-space jump as "a different-sized company" and predictions
    # break (verified 2026-05-13: 粤电力A 14:00 −0.71% vs EOD +3.02%, the
    # entire +3.73pp swing came from this one feature).
    #
    # Fix: forward-fill bars_mv_log to today using yesterday's bars-derived
    # value × (today_close / prev_close).  Shares outstanding barely moves
    # intra-day; market-cap change ≈ close change.  This keeps the
    # intraday row in the SAME value space as the rest of the history.
    # Only when no prior bars-derived value exists at all do we fall back
    # to the cached snapshot value (training-time Sina-only path).
    valid_t = turnover > 0
    float_mv = np.where(valid_t, close * volume / turnover, np.nan)
    bars_mv_log = np.where(float_mv > 0, np.log(float_mv), np.nan)
    cached_mv_log = result["total_mv_log"].values.copy()

    # Forward-fill bars_mv_log, adjusting for close changes (keeps value space)
    filled = bars_mv_log.copy()
    last_valid = -1
    for i in range(len(filled)):
        if not np.isnan(filled[i]):
            last_valid = i
        elif last_valid >= 0 and close[i] > 0 and close[last_valid] > 0:
            # log(today_mv) ≈ log(prev_mv) + log(today_close / prev_close)
            filled[i] = bars_mv_log[last_valid] + np.log(close[i] / close[last_valid])

    result["total_mv_log"] = np.where(~np.isnan(filled), filled, cached_mv_log)

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
    code_to_industry,
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
    code_to_industry : dict[str, str] | pd.DataFrame
        Either:
        - ``{code: industry_name}`` dict from :func:`get_industry_mapping` —
          every date uses the same (current) label.  **Introduces mild
          look-ahead** for stocks that changed industry; fine for live
          inference, unsafe for training.
        - DataFrame ``(code, start_date, board_name)`` from
          :func:`get_industry_history` — for each (code, date) we use the
          label in effect on that date via ``merge_asof(backward)``.
          Look-ahead-free; preferred for training / backtest.

    Returns
    -------
    The same DataFrame with INDUSTRY_RANK_COLUMNS added in-place.
    """
    if df.empty or code_to_industry is None or (
        hasattr(code_to_industry, "__len__") and len(code_to_industry) == 0
    ):
        for col in INDUSTRY_RANK_COLUMNS:
            df[col] = np.nan
        return df

    df = df.copy()

    if isinstance(code_to_industry, pd.DataFrame):
        # Point-in-time: merge_asof(backward) picks the industry in effect as of `date`.
        hist = code_to_industry[["code", "start_date", "board_name"]].copy()
        hist["start_date"] = pd.to_datetime(hist["start_date"])
        hist = hist.sort_values(["code", "start_date"]).reset_index(drop=True)

        left = df[["code", "date"]].copy()
        left["date"] = pd.to_datetime(left["date"])
        left["_row"] = np.arange(len(left))

        # merge_asof requires both sides sorted on the `on` key.
        left_sorted = left.sort_values(["date"]).reset_index(drop=True)
        merged = pd.merge_asof(
            left_sorted,
            hist.rename(columns={"start_date": "date"}).sort_values(["date"]).reset_index(drop=True),
            on="date",
            by="code",
            direction="backward",
        )
        merged = merged.sort_values("_row").reset_index(drop=True)
        df["_industry"] = merged["board_name"].to_numpy()
    else:
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

    # Pre-load historical PE/PB/market-cap in one SQL query, then split per code.
    # Falls back silently when the valuation table is empty (legacy behaviour).
    val_hist_map: Dict[str, pd.DataFrame] = {}
    if include_fundamentals:
        try:
            from mp.data.store import DataStore as _DS
            _start_iso = pd.to_datetime(str(start)).strftime("%Y-%m-%d")
            _end_iso = (
                pd.to_datetime(str(end)).strftime("%Y-%m-%d")
                if end else None
            )
            vh_all = _DS().load_valuation_history(codes=codes, start=_start_iso, end=_end_iso)
            if not vh_all.empty:
                for code, grp in vh_all.groupby("code"):
                    val_hist_map[code] = grp[["date", "pe_ttm", "pb", "total_mv"]].copy()
                logger.info("Loaded historical valuation for {}/{} stocks", len(val_hist_map), total)
            else:
                logger.info("Valuation table is empty; PE/PB/market_cap will be NaN in training")
        except Exception as e:
            logger.warning("Failed to load historical valuation: {}", e)

    frames: List[pd.DataFrame] = []
    for idx, code in enumerate(codes):
        try:
            fin_hist = fin_hist_map.get(code) if include_fundamentals else None
            val_hist = val_hist_map.get(code) if include_fundamentals else None
            part = _process_single_stock(code, start, end, horizon,
                                         fin_hist=fin_hist,
                                         valuation_row=None,
                                         valuation_hist=val_hist)
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

            # Winsorize extreme excess_ret outliers — usually qfq adjustment
            # artefacts (splits/rights offerings that DB hasn't fully reconciled).
            # MSE-trained LGBM is sensitive to tail values: |excess| > 50% in
            # 20 days is almost always a data error, not a real signal.
            # Cap at ±EXCESS_CAP (≈ 4.4σ given std ~0.114) to keep training
            # stable without losing real extreme moves. See module-level
            # constant for full caveats.
            n_clipped = (dataset[EXCESS_LABEL].abs() > EXCESS_CAP).sum()
            if n_clipped > 0:
                dataset[EXCESS_LABEL] = dataset[EXCESS_LABEL].clip(
                    lower=-EXCESS_CAP, upper=EXCESS_CAP,
                )
                logger.info("Winsorized {} excess_ret outliers at ±{:.0%} ({:.3f}% of rows)",
                            n_clipped, EXCESS_CAP, n_clipped/len(dataset)*100)
        except Exception as e:
            logger.warning("Failed to compute benchmark fwd_ret, excess_ret unavailable: {}", e)

    # --- Industry-relative ranking features (cross-sectional post-processing) ---
    if include_fundamentals and len(dataset) > 0:
        try:
            from mp.data.fetcher import get_industry_history
            ind_hist = get_industry_history(universe=codes)
            dataset = _add_industry_relative_features(dataset, ind_hist)
            n_ranked = dataset["pe_ind_rank"].notna().sum()
            logger.info("Industry rank features added (PIT): {}/{} rows", n_ranked, len(dataset))
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
        # Same winsorize as build_dataset — see EXCESS_CAP module constant.
        n_clipped = (df[EXCESS_LABEL].abs() > EXCESS_CAP).sum()
        if n_clipped > 0:
            df[EXCESS_LABEL] = df[EXCESS_LABEL].clip(-EXCESS_CAP, EXCESS_CAP)
            logger.info("Winsorized {} excess_ret outliers at ±{:.0%}", n_clipped, EXCESS_CAP)
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
    # _data_quality drives the荐股降级 gate, so only count columns that truly
    # signal a broken data source.  total_mv_log is a control variable (size
    # factor) fetched only from EM; its absence during EM outages is expected
    # and handled by yesterday's SQLite fallback.  Excluding it from the gate
    # prevents false degradation when EM is down but PE/PB are fine via Baidu.
    _QUALITY_GATE_COLS = [c for c in FUNDAMENTAL_COLUMNS if c != "total_mv_log"]

    # Both the display warning [缺] and the quality gate use the same col set —
    # total_mv_log is excluded from both so EM outages don't produce noisy [缺]
    # tags on otherwise well-predicted stocks.
    # ETFs legitimately lack PE/PB/ROE (they track an index, not a company) —
    # exclude them from the quality gate and per-row warning, otherwise e.g.
    # holding 军工ETF (512660) alongside normal stocks always trips the
    # "数据源异常" banner even when every real stock has full fundamentals.
    from mp.data.fetcher import _is_etf_code

    def _make_warning(row):
        if _is_etf_code(str(row.get("code", ""))):
            return ""
        missing = [c for c in _QUALITY_GATE_COLS if pd.isna(row.get(c))]
        return ",".join(missing) if missing else ""
    result["_data_warnings"] = result.apply(_make_warning, axis=1)

    existing_gate = [c for c in _QUALITY_GATE_COLS if c in result.columns]
    _nonetf_mask = ~result["code"].astype(str).map(_is_etf_code)
    n_gate = int(result.loc[_nonetf_mask, existing_gate].isna().any(axis=1).sum()) if existing_gate else 0
    n_warn = n_gate  # same set now
    result.attrs["_data_quality"] = 1.0 - (n_gate / len(result)) if len(result) > 0 else 0.0
    if n_warn > 0:
        logger.warning("⚠ {}/{} 股票存在基本面数据缺失，预测可能不准", n_warn, len(result))
    if n_gate > len(result) * 0.5:
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
