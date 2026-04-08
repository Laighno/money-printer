"""Factor IC (Information Coefficient) backtesting for individual stocks.

Computes time-series Spearman rank correlation between factor values at time t
and forward returns from t to t+n, for each technical factor.

Usage:
    from mp.backtest.ic_analysis import run_ic_analysis
    ic_df = run_ic_analysis("603799", start="20230101")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# === EMA helper (duplicated from technical.py to keep module self-contained) ===

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    k = 2.0 / (period + 1)
    out = np.empty_like(data, dtype=float)
    out[:period] = np.nan
    out[period - 1] = data[:period].mean()
    for i in range(period, len(data)):
        out[i] = data[i] * k + out[i - 1] * (1 - k)
    return out


# === Spearman helpers (pure numpy, no scipy) ===

def _rank(arr: np.ndarray) -> np.ndarray:
    """Rank values (1-based). NaN positions remain NaN."""
    out = np.full_like(arr, np.nan, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return out
    order = arr[mask].argsort().argsort() + 1.0
    out[mask] = order
    return out


def _spearman_corr(x: np.ndarray, y: np.ndarray, min_obs: int = 20) -> float:
    """Spearman rank correlation between x and y, ignoring NaN pairs."""
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < min_obs:
        return np.nan
    rx = _rank(x[valid])
    ry = _rank(y[valid])
    # Pearson on ranks = Spearman
    cc = np.corrcoef(rx, ry)
    return float(cc[0, 1])


# === Rolling technical indicators (full time-series output) ===

def rolling_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI at each bar. Wilder's smoothing. First valid at index `period`."""
    n = len(close)
    out = np.full(n, np.nan)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    if len(gain) < period:
        return out

    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


def rolling_macd_hist(close: np.ndarray, fast: int = 12, slow: int = 26,
                      sig: int = 9) -> np.ndarray:
    """MACD histogram at each bar. First valid at ~bar 34."""
    n = len(close)
    out = np.full(n, np.nan)
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    dif = ema_f - ema_s
    start = slow - 1
    dea = _ema(dif[start:], sig)
    hist = 2.0 * (dif[start:] - dea)
    out[start:] = hist
    return out


def rolling_bollinger_pctb(close: np.ndarray, period: int = 20,
                           num_std: float = 2.0) -> np.ndarray:
    """Bollinger %B at each bar. First valid at index `period - 1`."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = close[i - period + 1: i + 1]
        mid = window.mean()
        std = window.std(ddof=0)
        upper = mid + num_std * std
        lower = mid - num_std * std
        width = upper - lower
        out[i] = (close[i] - lower) / width if width > 0 else 0.5
    return out


def rolling_kdj_j(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  n_period: int = 9, m1: int = 3, m2: int = 3) -> np.ndarray:
    """KDJ J-value at each bar. First valid at index `n_period - 1`."""
    length = len(close)
    out = np.full(length, np.nan)
    k_val, d_val = 50.0, 50.0
    for i in range(n_period - 1, length):
        h = high[i - n_period + 1: i + 1].max()
        l = low[i - n_period + 1: i + 1].min()
        rsv = (close[i] - l) / (h - l) * 100 if h != l else 50.0
        k_val = (m1 - 1) / m1 * k_val + 1 / m1 * rsv
        d_val = (m2 - 1) / m2 * d_val + 1 / m2 * k_val
        out[i] = 3 * k_val - 2 * d_val
    return out


def rolling_vol_price_ratio(close: np.ndarray, volume: np.ndarray,
                            window: int = 5) -> np.ndarray:
    """Volume ratio (recent/previous window). First valid at `2*window - 1`."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(2 * window - 1, n):
        recent = volume[i - window + 1: i + 1].mean()
        prev = volume[i - 2 * window + 1: i - window + 1].mean()
        out[i] = recent / prev if prev > 0 else 1.0
    return out


def rolling_momentum(close: np.ndarray, period: int) -> np.ndarray:
    """Price momentum: close[i]/close[i-period] - 1. First valid at `period`."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period, n):
        if close[i - period] > 0:
            out[i] = close[i] / close[i - period] - 1.0
    return out


def rolling_volatility(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Annualized rolling volatility. First valid at `period`."""
    n = len(close)
    out = np.full(n, np.nan)
    rets = np.diff(close) / close[:-1]  # daily returns
    for i in range(period, n):
        # rets[i-period:i] corresponds to returns ending at bar i
        window_rets = rets[i - period: i]
        out[i] = window_rets.std(ddof=1) * np.sqrt(252)
    return out


# === Inflection point factors (二阶导/拐点因子) ===

def _delta(arr: np.ndarray, lag: int = 5) -> np.ndarray:
    """Change in a rolling indicator over `lag` bars: arr[i] - arr[i-lag]."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(lag, n):
        if not np.isnan(arr[i]) and not np.isnan(arr[i - lag]):
            out[i] = arr[i] - arr[i - lag]
    return out


def rolling_rsi_delta(close: np.ndarray, period: int = 14, lag: int = 5) -> np.ndarray:
    """RSI change over `lag` bars. Positive = RSI recovering."""
    return _delta(rolling_rsi(close, period), lag)


def rolling_macd_hist_delta(close: np.ndarray, lag: int = 5) -> np.ndarray:
    """MACD histogram change. Positive = momentum improving."""
    return _delta(rolling_macd_hist(close), lag)


def rolling_bollinger_pctb_delta(close: np.ndarray, lag: int = 5) -> np.ndarray:
    """%B change. Positive = moving away from lower band."""
    return _delta(rolling_bollinger_pctb(close), lag)


def rolling_momentum_accel(close: np.ndarray) -> np.ndarray:
    """Momentum acceleration: mom_5d - mom_20d. Positive = short-term catching up."""
    mom5 = rolling_momentum(close, 5)
    mom20 = rolling_momentum(close, 20)
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(mom5[i]) and not np.isnan(mom20[i]):
            out[i] = mom5[i] - mom20[i]
    return out


def rolling_volume_trend(volume: np.ndarray, short: int = 5, long: int = 20) -> np.ndarray:
    """Volume trend: short-term avg / long-term avg. >1 = volume increasing."""
    n = len(volume)
    out = np.full(n, np.nan)
    for i in range(long - 1, n):
        avg_short = volume[max(0, i - short + 1): i + 1].mean()
        avg_long = volume[i - long + 1: i + 1].mean()
        if avg_long > 0:
            out[i] = avg_short / avg_long
    return out


# === Moving average factors ===

def rolling_ma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = close[i - period + 1: i + 1].mean()
    return out


def rolling_ma_deviation(close: np.ndarray, period: int) -> np.ndarray:
    """Close / MA(period) - 1. Positive = above MA."""
    ma = rolling_ma(close, period)
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(ma[i]) and ma[i] > 0:
            out[i] = close[i] / ma[i] - 1.0
    return out


def rolling_ma_alignment(close: np.ndarray) -> np.ndarray:
    """Multi-MA alignment score: (MA5 - MA60) / MA60.
    Positive = bullish alignment (short MA above long MA)."""
    ma5 = rolling_ma(close, 5)
    ma60 = rolling_ma(close, 60)
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(ma5[i]) and not np.isnan(ma60[i]) and ma60[i] > 0:
            out[i] = (ma5[i] - ma60[i]) / ma60[i]
    return out


# === ATR / Range factors ===

def rolling_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    """ATR(period) / close — normalized average true range."""
    n = len(close)
    out = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    # Wilder smoothing
    if n < period + 1:
        return out
    atr_val = tr[1: period + 1].mean()
    if close[period] > 0:
        out[period] = atr_val / close[period]
    for i in range(period + 1, n):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        if close[i] > 0:
            out[i] = atr_val / close[i]
    return out


def rolling_price_range(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        period: int = 10) -> np.ndarray:
    """Average (high-low)/close over period — intraday range."""
    n = len(close)
    out = np.full(n, np.nan)
    daily_range = np.where(close > 0, (high - low) / close, 0.0)
    for i in range(period - 1, n):
        out[i] = daily_range[i - period + 1: i + 1].mean()
    return out


# === Williams %R ===

def rolling_williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       period: int = 14) -> np.ndarray:
    """Williams %R. Range: -100 (oversold) to 0 (overbought)."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh = high[i - period + 1: i + 1].max()
        ll = low[i - period + 1: i + 1].min()
        if hh != ll:
            out[i] = (hh - close[i]) / (hh - ll) * -100.0
        else:
            out[i] = -50.0
    return out


# === ADX (Average Directional Index) ===

def rolling_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    """ADX(period). Measures trend strength, 0-100."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < 2 * period + 1:
        return out

    # True range, +DM, -DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    # Wilder smoothing
    atr14 = tr[1: period + 1].sum()
    pdm14 = plus_dm[1: period + 1].sum()
    mdm14 = minus_dm[1: period + 1].sum()

    dx_values = []
    for i in range(period, n):
        if i > period:
            atr14 = atr14 - atr14 / period + tr[i]
            pdm14 = pdm14 - pdm14 / period + plus_dm[i]
            mdm14 = mdm14 - mdm14 / period + minus_dm[i]

        pdi = 100.0 * pdm14 / atr14 if atr14 > 0 else 0.0
        mdi = 100.0 * mdm14 / atr14 if atr14 > 0 else 0.0
        di_sum = pdi + mdi
        dx = 100.0 * abs(pdi - mdi) / di_sum if di_sum > 0 else 0.0
        dx_values.append(dx)

        if len(dx_values) == period:
            adx = np.mean(dx_values)
            out[i] = adx
        elif len(dx_values) > period:
            adx = (out[i - 1] * (period - 1) + dx) / period
            out[i] = adx

    return out


# === OBV (On-Balance Volume) ===

def rolling_obv_slope(close: np.ndarray, volume: np.ndarray,
                      period: int = 20) -> np.ndarray:
    """OBV regression slope over period, normalized by average volume."""
    n = len(close)
    out = np.full(n, np.nan)

    # Compute OBV
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    # Rolling slope via linear regression
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    for i in range(period - 1, n):
        y = obv[i - period + 1: i + 1]
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var if x_var > 0 else 0.0
        avg_vol = volume[i - period + 1: i + 1].mean()
        out[i] = slope / avg_vol if avg_vol > 0 else 0.0
    return out


# === Amount (成交额) factors ===

def rolling_amount_ratio(amount: np.ndarray, period: int = 20) -> np.ndarray:
    """Recent amount / MA(period) amount — relative trading activity."""
    n = len(amount)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        ma_amt = amount[i - period + 1: i + 1].mean()
        if ma_amt > 0:
            out[i] = amount[i] / ma_amt
    return out


# === Turnover factors ===

def rolling_turnover_ma(turnover: np.ndarray, period: int = 5) -> np.ndarray:
    """Moving average of turnover rate."""
    n = len(turnover)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = turnover[i - period + 1: i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            out[i] = valid.mean()
    return out


def rolling_turnover_percentile(turnover: np.ndarray, period: int = 60) -> np.ndarray:
    """Turnover percentile vs trailing period. 0-1 range."""
    n = len(turnover)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = turnover[i - period + 1: i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10 and not np.isnan(turnover[i]):
            out[i] = float((valid < turnover[i]).sum()) / len(valid)
    return out


# === Return distribution factors ===

def rolling_return_skew(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Skewness of daily returns over period. Positive = right tail (bullish)."""
    n = len(close)
    out = np.full(n, np.nan)
    rets = np.diff(close) / close[:-1]
    rets = np.insert(rets, 0, 0.0)
    for i in range(period, n):
        r = rets[i - period + 1: i + 1]
        valid = r[~np.isnan(r)]
        if len(valid) >= 10:
            m = valid.mean()
            s = valid.std()
            if s > 1e-10:
                out[i] = ((valid - m) ** 3).mean() / (s ** 3)
    return out


def rolling_return_kurtosis(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Excess kurtosis of daily returns. High = fat tails / tail risk."""
    n = len(close)
    out = np.full(n, np.nan)
    rets = np.diff(close) / close[:-1]
    rets = np.insert(rets, 0, 0.0)
    for i in range(period, n):
        r = rets[i - period + 1: i + 1]
        valid = r[~np.isnan(r)]
        if len(valid) >= 10:
            m = valid.mean()
            s = valid.std()
            if s > 1e-10:
                out[i] = ((valid - m) ** 4).mean() / (s ** 4) - 3.0
    return out


def rolling_updown_vol_ratio(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Upside vol / downside vol. >1 = up moves bigger than down moves."""
    n = len(close)
    out = np.full(n, np.nan)
    rets = np.diff(close) / close[:-1]
    rets = np.insert(rets, 0, 0.0)
    for i in range(period, n):
        r = rets[i - period + 1: i + 1]
        up = r[r > 0]
        down = r[r < 0]
        if len(up) >= 3 and len(down) >= 3:
            up_std = up.std()
            dn_std = down.std()
            if dn_std > 1e-10:
                out[i] = up_std / dn_std
    return out


def rolling_max_drawdown(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Max drawdown over trailing period. Always negative or zero."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = close[i - period + 1: i + 1]
        peak = np.maximum.accumulate(window)
        dd = (window - peak) / peak
        out[i] = dd.min()
    return out


def rolling_return_autocorr(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Lag-1 autocorrelation of returns. Positive = trend, Negative = reversal."""
    n = len(close)
    out = np.full(n, np.nan)
    rets = np.diff(close) / close[:-1]
    rets = np.insert(rets, 0, 0.0)
    for i in range(period + 1, n):
        r = rets[i - period: i + 1]
        r1 = r[:-1]
        r2 = r[1:]
        valid_mask = ~(np.isnan(r1) | np.isnan(r2))
        if valid_mask.sum() >= 10:
            r1v = r1[valid_mask]
            r2v = r2[valid_mask]
            m1, m2 = r1v.mean(), r2v.mean()
            s1, s2 = r1v.std(), r2v.std()
            if s1 > 1e-10 and s2 > 1e-10:
                out[i] = ((r1v - m1) * (r2v - m2)).mean() / (s1 * s2)
    return out


# === K-line shape factors ===

def rolling_close_position(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                           period: int = 5) -> np.ndarray:
    """Average (close-low)/(high-low) over period. 1 = closes at top, 0 = at bottom."""
    n = len(close)
    out = np.full(n, np.nan)
    daily_pos = np.where(high - low > 0, (close - low) / (high - low), 0.5)
    for i in range(period - 1, n):
        out[i] = daily_pos[i - period + 1: i + 1].mean()
    return out


def rolling_upper_shadow(high: np.ndarray, open_arr: np.ndarray, close: np.ndarray,
                         period: int = 5) -> np.ndarray:
    """Average upper shadow ratio: (high - max(open,close)) / close."""
    n = len(close)
    out = np.full(n, np.nan)
    body_top = np.maximum(open_arr, close)
    shadow = np.where(close > 0, (high - body_top) / close, 0.0)
    for i in range(period - 1, n):
        out[i] = shadow[i - period + 1: i + 1].mean()
    return out


def rolling_lower_shadow(low: np.ndarray, open_arr: np.ndarray, close: np.ndarray,
                         period: int = 5) -> np.ndarray:
    """Average lower shadow ratio: (min(open,close) - low) / close."""
    n = len(close)
    out = np.full(n, np.nan)
    body_bot = np.minimum(open_arr, close)
    shadow = np.where(close > 0, (body_bot - low) / close, 0.0)
    for i in range(period - 1, n):
        out[i] = shadow[i - period + 1: i + 1].mean()
    return out


def rolling_body_ratio(high: np.ndarray, low: np.ndarray, open_arr: np.ndarray,
                       close: np.ndarray, period: int = 5) -> np.ndarray:
    """Average body/range ratio: |close-open|/(high-low). 1 = no shadow."""
    n = len(close)
    out = np.full(n, np.nan)
    rng = high - low
    body = np.abs(close - open_arr)
    ratio = np.where(rng > 0, body / rng, 0.0)
    for i in range(period - 1, n):
        out[i] = ratio[i - period + 1: i + 1].mean()
    return out


def rolling_gap(open_arr: np.ndarray, close: np.ndarray, period: int = 5) -> np.ndarray:
    """Average gap ratio: (open - prev_close) / prev_close."""
    n = len(close)
    out = np.full(n, np.nan)
    gap = np.full(n, 0.0)
    for i in range(1, n):
        if close[i - 1] > 0:
            gap[i] = (open_arr[i] - close[i - 1]) / close[i - 1]
    for i in range(period, n):
        out[i] = gap[i - period + 1: i + 1].mean()
    return out


# === Liquidity / Volume distribution ===

def rolling_amihud_illiquidity(close: np.ndarray, amount: np.ndarray,
                               period: int = 20) -> np.ndarray:
    """Amihud illiquidity: avg(|ret|/amount). Higher = less liquid."""
    n = len(close)
    out = np.full(n, np.nan)
    rets = np.full(n, 0.0)
    for i in range(1, n):
        if close[i - 1] > 0:
            rets[i] = abs(close[i] / close[i - 1] - 1.0)
    ratio = np.where(amount > 0, rets / amount, 0.0)
    for i in range(period, n):
        out[i] = ratio[i - period + 1: i + 1].mean()
    return out


def rolling_volume_volatility(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Coefficient of variation of volume: std/mean. Higher = irregular trading."""
    n = len(volume)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        w = volume[i - period + 1: i + 1]
        m = w.mean()
        if m > 0:
            out[i] = w.std() / m
    return out


def rolling_momentum_5d(close: np.ndarray) -> np.ndarray:
    """5-day momentum (percentage return)."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(5, n):
        if close[i - 5] > 0:
            out[i] = close[i] / close[i - 5] - 1.0
    return out


def rolling_momentum_10d(close: np.ndarray) -> np.ndarray:
    """10-day momentum (percentage return)."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(10, n):
        if close[i - 10] > 0:
            out[i] = close[i] / close[i - 10] - 1.0
    return out


# === VWAP / Cross-field combination factors ===

def rolling_vwap_deviation(close: np.ndarray, amount: np.ndarray, volume: np.ndarray,
                           period: int = 20) -> np.ndarray:
    """Close / VWAP(period) - 1. VWAP = sum(amount)/sum(volume)."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_sum = volume[i - period + 1: i + 1].sum()
        amt_sum = amount[i - period + 1: i + 1].sum()
        if vol_sum > 0 and close[i] > 0:
            vwap = amt_sum / vol_sum
            out[i] = close[i] / vwap - 1.0
    return out


def rolling_bollinger_bandwidth(close: np.ndarray, period: int = 20,
                                num_std: float = 2.0) -> np.ndarray:
    """Bollinger bandwidth: (upper - lower) / middle. Measures volatility squeeze."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        w = close[i - period + 1: i + 1]
        m = w.mean()
        s = w.std()
        if m > 0:
            out[i] = 2.0 * num_std * s / m
    return out


def rolling_vol_price_corr(close: np.ndarray, volume: np.ndarray,
                           period: int = 20) -> np.ndarray:
    """Rolling correlation between returns and volume. +1 = bullish vol, -1 = bearish vol."""
    n = len(close)
    out = np.full(n, np.nan)
    rets = np.full(n, 0.0)
    for i in range(1, n):
        if close[i - 1] > 0:
            rets[i] = close[i] / close[i - 1] - 1.0
    for i in range(period, n):
        r = rets[i - period + 1: i + 1]
        v = volume[i - period + 1: i + 1].astype(float)
        r_std = r.std()
        v_std = v.std()
        if r_std > 1e-10 and v_std > 1e-10:
            out[i] = ((r - r.mean()) * (v - v.mean())).mean() / (r_std * v_std)
    return out


def rolling_consecutive_days(close: np.ndarray) -> np.ndarray:
    """Count of consecutive up/down days. Positive = up streak, negative = down streak."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    streak = 0
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak = streak + 1 if streak > 0 else 1
        elif close[i] < close[i - 1]:
            streak = streak - 1 if streak < 0 else -1
        else:
            streak = 0
        out[i] = float(streak)
    return out


def rolling_high_distance(close: np.ndarray, high: np.ndarray,
                          period: int = 60) -> np.ndarray:
    """close / max(high, period) - 1. Always <= 0. How far from recent high."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh = high[i - period + 1: i + 1].max()
        if hh > 0:
            out[i] = close[i] / hh - 1.0
    return out


def rolling_low_distance(close: np.ndarray, low: np.ndarray,
                         period: int = 60) -> np.ndarray:
    """close / min(low, period) - 1. Always >= 0. How far from recent low."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        ll = low[i - period + 1: i + 1].min()
        if ll > 0:
            out[i] = close[i] / ll - 1.0
    return out


def rolling_vol_ratio(close: np.ndarray, short: int = 5, long: int = 60) -> np.ndarray:
    """Short-term volatility / long-term volatility. >1 = vol expanding."""
    n = len(close)
    out = np.full(n, np.nan)
    rets = np.full(n, 0.0)
    for i in range(1, n):
        if close[i - 1] > 0:
            rets[i] = close[i] / close[i - 1] - 1.0
    for i in range(long, n):
        s_short = rets[i - short + 1: i + 1].std()
        s_long = rets[i - long + 1: i + 1].std()
        if s_long > 1e-10:
            out[i] = s_short / s_long
    return out


def rolling_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                volume: np.ndarray, period: int = 14) -> np.ndarray:
    """Money Flow Index. RSI weighted by volume. 0-100."""
    n = len(close)
    out = np.full(n, np.nan)
    tp = (high + low + close) / 3.0
    mf = tp * volume
    for i in range(period, n):
        pos_mf = 0.0
        neg_mf = 0.0
        for j in range(i - period + 1, i + 1):
            if tp[j] > tp[j - 1]:
                pos_mf += mf[j]
            elif tp[j] < tp[j - 1]:
                neg_mf += mf[j]
        if neg_mf > 0:
            ratio = pos_mf / neg_mf
            out[i] = 100.0 - 100.0 / (1.0 + ratio)
        elif pos_mf > 0:
            out[i] = 100.0
        else:
            out[i] = 50.0
    return out


def rolling_intraday_intensity(high: np.ndarray, low: np.ndarray,
                               close: np.ndarray, volume: np.ndarray,
                               period: int = 20) -> np.ndarray:
    """Intraday intensity: avg((2*close-high-low)/(high-low)*volume) normalized."""
    n = len(close)
    out = np.full(n, np.nan)
    rng = high - low
    ii = np.where(rng > 0, (2.0 * close - high - low) / rng * volume, 0.0)
    for i in range(period - 1, n):
        avg_ii = ii[i - period + 1: i + 1].mean()
        avg_vol = volume[i - period + 1: i + 1].mean()
        if avg_vol > 0:
            out[i] = avg_ii / avg_vol
    return out


def rolling_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 20) -> np.ndarray:
    """Commodity Channel Index: (tp - sma(tp)) / (0.015 * mad(tp))."""
    n = len(close)
    out = np.full(n, np.nan)
    tp = (high + low + close) / 3.0
    for i in range(period - 1, n):
        w = tp[i - period + 1: i + 1]
        m = w.mean()
        mad = np.abs(w - m).mean()
        if mad > 1e-10:
            out[i] = (tp[i] - m) / (0.015 * mad)
    return out


def rolling_return_extremes_ratio(close: np.ndarray, period: int = 20) -> np.ndarray:
    """max(ret)/|min(ret)| over period. >1 = up extremes dominate."""
    n = len(close)
    out = np.full(n, np.nan)
    rets = np.full(n, 0.0)
    for i in range(1, n):
        if close[i - 1] > 0:
            rets[i] = close[i] / close[i - 1] - 1.0
    for i in range(period, n):
        r = rets[i - period + 1: i + 1]
        mx = r.max()
        mn = r.min()
        if mn < -1e-10:
            out[i] = mx / abs(mn)
        elif mx > 1e-10:
            out[i] = 2.0
        else:
            out[i] = 1.0
    return out


def rolling_amount_volatility(amount: np.ndarray, period: int = 20) -> np.ndarray:
    """CV of trading amount: std/mean. Higher = erratic activity."""
    n = len(amount)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        w = amount[i - period + 1: i + 1]
        m = w.mean()
        if m > 0:
            out[i] = w.std() / m
    return out


# === Forward returns ===

def forward_returns(close: np.ndarray,
                    horizons: tuple = (5, 10, 20)) -> dict:
    """Forward returns for each horizon. Last `h` elements are NaN."""
    result = {}
    n = len(close)
    for h in horizons:
        fwd = np.full(n, np.nan)
        for i in range(n - h):
            if close[i] > 0:
                fwd[i] = close[i + h] / close[i] - 1.0
        result[h] = fwd
    return result


# === IC computation ===

def calc_ic_series(factor_values: np.ndarray, fwd_rets: np.ndarray,
                   window: int = 60) -> np.ndarray:
    """Rolling Spearman IC between factor values and forward returns."""
    n = len(factor_values)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        f_win = factor_values[i - window + 1: i + 1]
        r_win = fwd_rets[i - window + 1: i + 1]
        out[i] = _spearman_corr(f_win, r_win)
    return out


def calc_ic_summary(ic_series: np.ndarray) -> dict:
    """Summarize an IC time series into key metrics."""
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) < 10:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ir": np.nan,
                "ic_positive_ratio": np.nan}
    ic_mean = float(valid.mean())
    ic_std = float(valid.std())
    ir = ic_mean / ic_std if ic_std > 0 else 0.0
    pos_ratio = float((valid > 0).sum() / len(valid))
    return {"ic_mean": ic_mean, "ic_std": ic_std, "ir": ir,
            "ic_positive_ratio": pos_ratio}


# === Orchestrator ===

def run_ic_analysis(
    code: str,
    start: str = "20230101",
    horizons: tuple = (5, 10, 20),
    ic_window: int = 60,
) -> pd.DataFrame:
    """Run IC analysis for a stock. Returns DataFrame with IC metrics per factor."""
    from mp.data.fetcher import get_daily_bars

    df = get_daily_bars(code, start)
    if df is None or len(df) < ic_window + max(horizons):
        logger.warning(f"IC analysis: insufficient data for {code} ({len(df) if df is not None else 0} bars)")
        return pd.DataFrame()

    df = df.sort_values("date")
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)

    # Compute all rolling factors — level (水平值) + inflection (拐点)
    factors = {
        # Level factors (IC typically negative — mean reversion)
        "RSI(14)": rolling_rsi(close),
        "MACD柱": rolling_macd_hist(close),
        "布林%B": rolling_bollinger_pctb(close),
        "KDJ-J": rolling_kdj_j(high, low, close),
        "量价比": rolling_vol_price_ratio(close, volume),
        "动量20d": rolling_momentum(close, 20),
        "动量60d": rolling_momentum(close, 60),
        "波动率20d": rolling_volatility(close, 20),
        # Inflection factors (IC expected positive — trend change)
        "RSI变化": rolling_rsi_delta(close),
        "MACD柱变化": rolling_macd_hist_delta(close),
        "布林%B变化": rolling_bollinger_pctb_delta(close),
        "动量加速度": rolling_momentum_accel(close),
        "量能趋势": rolling_volume_trend(volume),
    }

    # Compute forward returns
    fwd = forward_returns(close, horizons)

    # Compute IC for each factor x horizon
    rows = []
    for fname, fvals in factors.items():
        row = {"因子": fname}
        effective_count = 0
        for h in horizons:
            ic_ser = calc_ic_series(fvals, fwd[h], ic_window)
            summary = calc_ic_summary(ic_ser)
            row[f"IC({h}d)"] = summary["ic_mean"]
            row[f"IR({h}d)"] = summary["ir"]
            row[f"IC>0({h}d)"] = summary["ic_positive_ratio"]
            # Count effective horizons (negative IC is also predictive)
            if not np.isnan(summary["ic_mean"]):
                if abs(summary["ic_mean"]) > 0.03 and abs(summary["ir"]) > 0.3:
                    effective_count += 1
        # Rating
        if effective_count >= 3:
            row["有效性"] = "★★★"
        elif effective_count >= 2:
            row["有效性"] = "★★☆"
        elif effective_count >= 1:
            row["有效性"] = "★☆☆"
        else:
            row["有效性"] = "噪音"
        rows.append(row)

    result = pd.DataFrame(rows).set_index("因子")
    return result
