"""Factor expression DSL — declarative expression trees for technical factors.

Provides a set of expression node classes that describe factor computations
as data rather than code.  An ``evaluate()`` function walks the tree and
produces a pandas Series from an OHLCV DataFrame.

Usage::

    from mp.factor.expr import evaluate, Field, Rolling, BinaryOp
    expr = BinaryOp('/', Rolling('mean', Field('volume'), 5),
                         Rolling('mean', Field('volume'), 20))
    result = evaluate(expr, df)  # returns pd.Series
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Expression tree nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Expr:
    """Base class for factor expressions."""
    pass


@dataclass(frozen=True)
class Field(Expr):
    """Reference to a raw OHLCV column: close, open, high, low, volume, amount, turnover."""
    name: str


@dataclass(frozen=True)
class Const(Expr):
    """Constant value."""
    value: float


@dataclass(frozen=True)
class Rolling(Expr):
    """Rolling window operation.

    ``op`` is one of: mean, std, max, min, sum, rank, corr, cov, ema,
    skew, kurtosis, percentile.
    For corr/cov ``expr2`` is the second series.
    """
    op: str
    expr: Expr
    window: int
    expr2: Optional[Expr] = None


@dataclass(frozen=True)
class BinaryOp(Expr):
    """Binary arithmetic: +, -, *, /."""
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class UnaryOp(Expr):
    """Unary operation: neg, abs, log, sign, sqrt."""
    op: str
    expr: Expr


@dataclass(frozen=True)
class Delta(Expr):
    """Lagged difference: value(t) - value(t - lag)."""
    expr: Expr
    lag: int


@dataclass(frozen=True)
class Lag(Expr):
    """Lagged value: value(t - lag)."""
    expr: Expr
    lag: int


@dataclass(frozen=True)
class Rank(Expr):
    """Cross-sectional rank (percentile). Not commonly used in time-series mode."""
    expr: Expr


@dataclass(frozen=True)
class Condition(Expr):
    """Conditional: where(cond > 0, true_expr, false_expr)."""
    cond: Expr
    true_expr: Expr
    false_expr: Expr


@dataclass(frozen=True)
class Custom(Expr):
    """Escape hatch — a named custom operation with arbitrary args.

    Used for complex indicators (RSI, MACD, KDJ, ADX, OBV, ...) that
    cannot be neatly decomposed into primitive Rolling/BinaryOp nodes
    while staying numerically equivalent to the reference implementation.
    """
    name: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FactorDef
# ---------------------------------------------------------------------------

@dataclass
class FactorDef:
    """A named factor backed by an expression tree."""
    name: str
    expr: Expr
    description: str = ""
    category: str = "technical"


# ---------------------------------------------------------------------------
# Helper constructors (ergonomic shortcuts)
# ---------------------------------------------------------------------------

def close() -> Field:
    return Field("close")

def open_() -> Field:
    return Field("open")

def high() -> Field:
    return Field("high")

def low() -> Field:
    return Field("low")

def volume() -> Field:
    return Field("volume")

def amount() -> Field:
    return Field("amount")

def turnover() -> Field:
    return Field("turnover")

# Rolling helpers
def roll_mean(expr: Expr, w: int) -> Rolling:
    return Rolling("mean", expr, w)

def roll_std(expr: Expr, w: int) -> Rolling:
    return Rolling("std", expr, w)

def roll_max(expr: Expr, w: int) -> Rolling:
    return Rolling("max", expr, w)

def roll_min(expr: Expr, w: int) -> Rolling:
    return Rolling("min", expr, w)

def roll_sum(expr: Expr, w: int) -> Rolling:
    return Rolling("sum", expr, w)

def roll_ema(expr: Expr, w: int) -> Rolling:
    return Rolling("ema", expr, w)

def roll_corr(a: Expr, b: Expr, w: int) -> Rolling:
    return Rolling("corr", a, w, expr2=b)

def roll_skew(expr: Expr, w: int) -> Rolling:
    return Rolling("skew", expr, w)

def roll_kurtosis(expr: Expr, w: int) -> Rolling:
    return Rolling("kurtosis", expr, w)

def roll_percentile(expr: Expr, w: int) -> Rolling:
    return Rolling("percentile", expr, w)

# Binary helpers
def div(a: Expr, b: Expr) -> BinaryOp:
    return BinaryOp("/", a, b)

def sub(a: Expr, b: Expr) -> BinaryOp:
    return BinaryOp("-", a, b)

def add(a: Expr, b: Expr) -> BinaryOp:
    return BinaryOp("+", a, b)

def mul(a: Expr, b: Expr) -> BinaryOp:
    return BinaryOp("*", a, b)

# Composite helpers
def pct_change(expr: Expr, n: int) -> BinaryOp:
    """(value - lag(value, n)) / lag(value, n)"""
    return div(sub(expr, Lag(expr, n)), Lag(expr, n))

def daily_return(expr: Expr) -> BinaryOp:
    """One-period percentage return."""
    return pct_change(expr, 1)

def delta(expr: Expr, lag: int) -> Delta:
    return Delta(expr, lag)


# ---------------------------------------------------------------------------
# EMA helper (numpy, matches ic_analysis._ema exactly)
# ---------------------------------------------------------------------------

def _ema_array(data: np.ndarray, period: int) -> np.ndarray:
    """EMA matching the Wilder-style initialization in ic_analysis."""
    k = 2.0 / (period + 1)
    out = np.empty_like(data, dtype=float)
    out[:period] = np.nan
    if len(data) < period:
        return out
    out[period - 1] = np.nanmean(data[:period])
    for i in range(period, len(data)):
        out[i] = data[i] * k + out[i - 1] * (1 - k)
    return out


# ---------------------------------------------------------------------------
# Custom evaluator functions (complex indicators)
# ---------------------------------------------------------------------------

def _eval_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI — Wilder smoothing, numerically matches rolling_rsi."""
    close_arr = df["close"].values.astype(float)
    n = len(close_arr)
    out = np.full(n, np.nan)
    d = np.diff(close_arr)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    if len(gain) < period:
        return pd.Series(out, index=df.index)
    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    if avg_loss == 0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            out[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return pd.Series(out, index=df.index)


def _eval_macd_hist(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                    sig: int = 9) -> pd.Series:
    """MACD histogram — matches rolling_macd_hist."""
    close_arr = df["close"].values.astype(float)
    n = len(close_arr)
    out = np.full(n, np.nan)
    ema_f = _ema_array(close_arr, fast)
    ema_s = _ema_array(close_arr, slow)
    dif = ema_f - ema_s
    start = slow - 1
    dea = _ema_array(dif[start:], sig)
    hist = 2.0 * (dif[start:] - dea)
    out[start:] = hist
    return pd.Series(out, index=df.index)


def _eval_bollinger_pctb(df: pd.DataFrame, period: int = 20,
                         num_std: float = 2.0) -> pd.Series:
    """Bollinger %B — matches rolling_bollinger_pctb."""
    close_arr = df["close"].values.astype(float)
    n = len(close_arr)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = close_arr[i - period + 1: i + 1]
        mid = window.mean()
        std = window.std(ddof=0)
        upper = mid + num_std * std
        lower = mid - num_std * std
        width = upper - lower
        out[i] = (close_arr[i] - lower) / width if width > 0 else 0.5
    return pd.Series(out, index=df.index)


def _eval_kdj_j(df: pd.DataFrame, n_period: int = 9, m1: int = 3,
                m2: int = 3) -> pd.Series:
    """KDJ J-value — matches rolling_kdj_j."""
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    k_val, d_val = 50.0, 50.0
    for i in range(n_period - 1, n):
        hh = h[i - n_period + 1: i + 1].max()
        ll = l[i - n_period + 1: i + 1].min()
        rsv = (c[i] - ll) / (hh - ll) * 100 if hh != ll else 50.0
        k_val = (m1 - 1) / m1 * k_val + 1 / m1 * rsv
        d_val = (m2 - 1) / m2 * d_val + 1 / m2 * k_val
        out[i] = 3 * k_val - 2 * d_val
    return pd.Series(out, index=df.index)


def _eval_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR / close — matches rolling_atr."""
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    if n < period + 1:
        return pd.Series(out, index=df.index)
    atr_val = tr[1: period + 1].mean()
    if c[period] > 0:
        out[period] = atr_val / c[period]
    for i in range(period + 1, n):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        if c[i] > 0:
            out[i] = atr_val / c[i]
    return pd.Series(out, index=df.index)


def _eval_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R — matches rolling_williams_r."""
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh = h[i - period + 1: i + 1].max()
        ll = l[i - period + 1: i + 1].min()
        if hh != ll:
            out[i] = (hh - c[i]) / (hh - ll) * -100.0
        else:
            out[i] = -50.0
    return pd.Series(out, index=df.index)


def _eval_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX — matches rolling_adx."""
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    if n < 2 * period + 1:
        return pd.Series(out, index=df.index)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
    atr14 = tr[1: period + 1].sum()
    pdm14 = plus_dm[1: period + 1].sum()
    mdm14 = minus_dm[1: period + 1].sum()
    dx_values: list[float] = []
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
            out[i] = np.mean(dx_values)
        elif len(dx_values) > period:
            out[i] = (out[i - 1] * (period - 1) + dx) / period
    return pd.Series(out, index=df.index)


def _eval_obv_slope(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """OBV slope — matches rolling_obv_slope."""
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    obv = np.zeros(n)
    for i in range(1, n):
        if c[i] > c[i - 1]:
            obv[i] = obv[i - 1] + v[i]
        elif c[i] < c[i - 1]:
            obv[i] = obv[i - 1] - v[i]
        else:
            obv[i] = obv[i - 1]
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    for i in range(period - 1, n):
        y = obv[i - period + 1: i + 1]
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var if x_var > 0 else 0.0
        avg_vol = v[i - period + 1: i + 1].mean()
        out[i] = slope / avg_vol if avg_vol > 0 else 0.0
    return pd.Series(out, index=df.index)


def _eval_vol_price_ratio(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Volume ratio (recent / previous window) — matches rolling_vol_price_ratio."""
    v = df["volume"].values.astype(float)
    n = len(v)
    out = np.full(n, np.nan)
    for i in range(2 * window - 1, n):
        recent = v[i - window + 1: i + 1].mean()
        prev = v[i - 2 * window + 1: i - window + 1].mean()
        out[i] = recent / prev if prev > 0 else 1.0
    return pd.Series(out, index=df.index)


def _eval_turnover_percentile(df: pd.DataFrame, period: int = 60) -> pd.Series:
    """Turnover percentile — matches rolling_turnover_percentile."""
    t = df["turnover"].values.astype(float) if "turnover" in df.columns else np.full(len(df), np.nan)
    n = len(t)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = t[i - period + 1: i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10 and not np.isnan(t[i]):
            out[i] = float((valid < t[i]).sum()) / len(valid)
    return pd.Series(out, index=df.index)


def _eval_return_skew(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Return skewness — matches rolling_return_skew."""
    c = df["close"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    rets = np.diff(c) / c[:-1]
    rets = np.insert(rets, 0, 0.0)
    for i in range(period, n):
        r = rets[i - period + 1: i + 1]
        valid = r[~np.isnan(r)]
        if len(valid) >= 10:
            m = valid.mean()
            s = valid.std()
            if s > 1e-10:
                out[i] = ((valid - m) ** 3).mean() / (s ** 3)
    return pd.Series(out, index=df.index)


def _eval_return_kurtosis(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Return kurtosis — matches rolling_return_kurtosis."""
    c = df["close"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    rets = np.diff(c) / c[:-1]
    rets = np.insert(rets, 0, 0.0)
    for i in range(period, n):
        r = rets[i - period + 1: i + 1]
        valid = r[~np.isnan(r)]
        if len(valid) >= 10:
            m = valid.mean()
            s = valid.std()
            if s > 1e-10:
                out[i] = ((valid - m) ** 4).mean() / (s ** 4) - 3.0
    return pd.Series(out, index=df.index)


def _eval_updown_vol_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Upside/downside vol ratio — matches rolling_updown_vol_ratio."""
    c = df["close"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    rets = np.diff(c) / c[:-1]
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
    return pd.Series(out, index=df.index)


def _eval_max_drawdown(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Max drawdown — matches rolling_max_drawdown."""
    c = df["close"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = c[i - period + 1: i + 1]
        peak = np.maximum.accumulate(window)
        dd = (window - peak) / peak
        out[i] = dd.min()
    return pd.Series(out, index=df.index)


def _eval_return_autocorr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Return autocorrelation — matches rolling_return_autocorr."""
    c = df["close"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    rets = np.diff(c) / c[:-1]
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
    return pd.Series(out, index=df.index)


def _eval_consecutive_days(df: pd.DataFrame) -> pd.Series:
    """Consecutive up/down days — matches rolling_consecutive_days."""
    c = df["close"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    if n < 2:
        return pd.Series(out, index=df.index)
    streak = 0
    for i in range(1, n):
        if c[i] > c[i - 1]:
            streak = streak + 1 if streak > 0 else 1
        elif c[i] < c[i - 1]:
            streak = streak - 1 if streak < 0 else -1
        else:
            streak = 0
        out[i] = float(streak)
    return pd.Series(out, index=df.index)


def _eval_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — matches rolling_mfi."""
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    tp = (h + l + c) / 3.0
    mf = tp * v
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
    return pd.Series(out, index=df.index)


def _eval_intraday_intensity(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Intraday intensity — matches rolling_intraday_intensity."""
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    rng = h - l
    ii = np.where(rng > 0, (2.0 * c - h - l) / rng * v, 0.0)
    for i in range(period - 1, n):
        avg_ii = ii[i - period + 1: i + 1].mean()
        avg_vol = v[i - period + 1: i + 1].mean()
        if avg_vol > 0:
            out[i] = avg_ii / avg_vol
    return pd.Series(out, index=df.index)


def _eval_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """CCI — matches rolling_cci."""
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    tp = (h + l + c) / 3.0
    for i in range(period - 1, n):
        w = tp[i - period + 1: i + 1]
        m = w.mean()
        mad = np.abs(w - m).mean()
        if mad > 1e-10:
            out[i] = (tp[i] - m) / (0.015 * mad)
    return pd.Series(out, index=df.index)


def _eval_vwap_deviation(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """VWAP deviation — matches rolling_vwap_deviation."""
    c = df["close"].values.astype(float)
    a = df["amount"].values.astype(float) if "amount" in df.columns else (
        df["volume"].values.astype(float) * c)
    v = df["volume"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_sum = v[i - period + 1: i + 1].sum()
        amt_sum = a[i - period + 1: i + 1].sum()
        if vol_sum > 0 and c[i] > 0:
            vwap = amt_sum / vol_sum
            out[i] = c[i] / vwap - 1.0
    return pd.Series(out, index=df.index)


def _eval_vol_price_corr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Vol-price correlation — matches rolling_vol_price_corr."""
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    rets = np.full(n, 0.0)
    for i in range(1, n):
        if c[i - 1] > 0:
            rets[i] = c[i] / c[i - 1] - 1.0
    for i in range(period, n):
        r = rets[i - period + 1: i + 1]
        vol = v[i - period + 1: i + 1].astype(float)
        r_std = r.std()
        v_std = vol.std()
        if r_std > 1e-10 and v_std > 1e-10:
            out[i] = ((r - r.mean()) * (vol - vol.mean())).mean() / (r_std * v_std)
    return pd.Series(out, index=df.index)


def _eval_return_extremes_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Return extremes ratio — matches rolling_return_extremes_ratio."""
    c = df["close"].values.astype(float)
    n = len(c)
    out = np.full(n, np.nan)
    rets = np.full(n, 0.0)
    for i in range(1, n):
        if c[i - 1] > 0:
            rets[i] = c[i] / c[i - 1] - 1.0
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
    return pd.Series(out, index=df.index)


# Dispatch table for Custom nodes
_CUSTOM_EVALUATORS: dict[str, callable] = {
    "rsi": _eval_rsi,
    "macd_hist": _eval_macd_hist,
    "bollinger_pctb": _eval_bollinger_pctb,
    "kdj_j": _eval_kdj_j,
    "atr": _eval_atr,
    "williams_r": _eval_williams_r,
    "adx": _eval_adx,
    "obv_slope": _eval_obv_slope,
    "vol_price_ratio": _eval_vol_price_ratio,
    "turnover_percentile": _eval_turnover_percentile,
    "return_skew": _eval_return_skew,
    "return_kurtosis": _eval_return_kurtosis,
    "updown_vol_ratio": _eval_updown_vol_ratio,
    "max_drawdown": _eval_max_drawdown,
    "return_autocorr": _eval_return_autocorr,
    "consecutive_days": _eval_consecutive_days,
    "mfi": _eval_mfi,
    "intraday_intensity": _eval_intraday_intensity,
    "cci": _eval_cci,
    "vwap_deviation": _eval_vwap_deviation,
    "vol_price_corr": _eval_vol_price_corr,
    "return_extremes_ratio": _eval_return_extremes_ratio,
}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate(expr: Expr, df: pd.DataFrame) -> pd.Series:
    """Evaluate an expression tree against an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume.
        May optionally have: amount, turnover.
    expr : Expr
        The expression tree to evaluate.

    Returns
    -------
    pd.Series
        Computed factor values, same index as *df*.
    """
    if isinstance(expr, Field):
        col = expr.name
        if col in df.columns:
            return df[col].astype(float)
        # Synthesize amount = close * volume if missing
        if col == "amount" and "volume" in df.columns and "close" in df.columns:
            return (df["close"] * df["volume"]).astype(float)
        # Return NaN series for optional columns (e.g. turnover)
        return pd.Series(np.nan, index=df.index, dtype=float)

    if isinstance(expr, Const):
        return pd.Series(expr.value, index=df.index, dtype=float)

    if isinstance(expr, BinaryOp):
        left = evaluate(expr.left, df)
        right = evaluate(expr.right, df)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right.replace(0, np.nan)
        if expr.op == "max":
            return pd.Series(np.maximum(left.values, right.values), index=df.index)
        if expr.op == "min":
            return pd.Series(np.minimum(left.values, right.values), index=df.index)
        raise ValueError(f"Unknown binary op: {expr.op}")

    if isinstance(expr, UnaryOp):
        inner = evaluate(expr.expr, df)
        if expr.op == "neg":
            return -inner
        if expr.op == "abs":
            return inner.abs()
        if expr.op == "log":
            return np.log(inner.replace(0, np.nan).clip(lower=1e-10))
        if expr.op == "sign":
            return np.sign(inner)
        if expr.op == "sqrt":
            return np.sqrt(inner.clip(lower=0))
        raise ValueError(f"Unknown unary op: {expr.op}")

    if isinstance(expr, Rolling):
        inner = evaluate(expr.expr, df)
        w = expr.window
        if expr.op == "mean":
            return inner.rolling(w, min_periods=w).mean()
        if expr.op == "std":
            return inner.rolling(w, min_periods=w).std(ddof=1)
        if expr.op == "std0":
            # ddof=0 variant used by Bollinger
            return inner.rolling(w, min_periods=w).std(ddof=0)
        if expr.op == "max":
            return inner.rolling(w, min_periods=w).max()
        if expr.op == "min":
            return inner.rolling(w, min_periods=w).min()
        if expr.op == "sum":
            return inner.rolling(w, min_periods=w).sum()
        if expr.op == "ema":
            return pd.Series(_ema_array(inner.values, w), index=df.index)
        if expr.op == "corr":
            if expr.expr2 is None:
                raise ValueError("Rolling corr requires expr2")
            other = evaluate(expr.expr2, df)
            return inner.rolling(w, min_periods=w).corr(other)
        if expr.op == "cov":
            if expr.expr2 is None:
                raise ValueError("Rolling cov requires expr2")
            other = evaluate(expr.expr2, df)
            return inner.rolling(w, min_periods=w).cov(other)
        if expr.op == "skew":
            return inner.rolling(w, min_periods=w).skew()
        if expr.op == "kurtosis":
            return inner.rolling(w, min_periods=w).kurt()
        if expr.op == "rank":
            return inner.rolling(w, min_periods=w).rank(pct=True)
        if expr.op == "percentile":
            # Current value's percentile within trailing window
            def _pctile(arr):
                if np.isnan(arr.iloc[-1]):
                    return np.nan
                valid = arr.dropna()
                if len(valid) < 10:
                    return np.nan
                return float((valid < arr.iloc[-1]).sum()) / len(valid)
            return inner.rolling(w, min_periods=10).apply(_pctile, raw=False)
        raise ValueError(f"Unknown rolling op: {expr.op}")

    if isinstance(expr, Lag):
        inner = evaluate(expr.expr, df)
        return inner.shift(expr.lag)

    if isinstance(expr, Delta):
        inner = evaluate(expr.expr, df)
        return inner - inner.shift(expr.lag)

    if isinstance(expr, Rank):
        inner = evaluate(expr.expr, df)
        return inner.rank(pct=True)

    if isinstance(expr, Condition):
        cond = evaluate(expr.cond, df)
        true_val = evaluate(expr.true_expr, df)
        false_val = evaluate(expr.false_expr, df)
        return pd.Series(np.where(cond > 0, true_val, false_val), index=df.index)

    if isinstance(expr, Custom):
        func = _CUSTOM_EVALUATORS.get(expr.name)
        if func is None:
            raise ValueError(f"Unknown custom evaluator: {expr.name}")
        return func(df, *expr.args, **expr.kwargs)

    raise TypeError(f"Cannot evaluate expression type: {type(expr).__name__}")
