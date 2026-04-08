"""Technical indicators computed from OHLCV data. No external dependencies beyond numpy."""

import numpy as np
from . import Signal


# === EMA helper ===

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. Seed with SMA of first `period` values."""
    k = 2.0 / (period + 1)
    out = np.empty_like(data, dtype=float)
    out[:period] = np.nan
    out[period - 1] = data[:period].mean()
    for i in range(period, len(data)):
        out[i] = data[i] * k + out[i - 1] * (1 - k)
    return out


# === RSI (Wilder's smoothing) ===

def calc_rsi(close: np.ndarray, period: int = 14) -> float:
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _interpret_rsi(rsi: float) -> Signal:
    if rsi < 20:
        return Signal(dimension="技术指标", name="RSI(14)", value=round(rsi, 1),
                      signal="bullish", detail=f"RSI={rsi:.1f}, 极度超卖")
    if rsi < 30:
        return Signal(dimension="技术指标", name="RSI(14)", value=round(rsi, 1),
                      signal="bullish", detail=f"RSI={rsi:.1f}, 超卖区间")
    if rsi > 80:
        return Signal(dimension="技术指标", name="RSI(14)", value=round(rsi, 1),
                      signal="bearish", detail=f"RSI={rsi:.1f}, 极度超买")
    if rsi > 70:
        return Signal(dimension="技术指标", name="RSI(14)", value=round(rsi, 1),
                      signal="bearish", detail=f"RSI={rsi:.1f}, 超买区间")
    return Signal(dimension="技术指标", name="RSI(14)", value=round(rsi, 1),
                  signal="neutral", detail=f"RSI={rsi:.1f}, 中性区间")


# === MACD ===

def calc_macd(close: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9
              ) -> tuple[float, float, float]:
    """Returns (DIF, DEA, histogram) for latest bar."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    dif = ema_fast - ema_slow
    # DEA = EMA of DIF, but only from where DIF is valid
    start = slow - 1
    dea = _ema(dif[start:], sig)
    hist = 2 * (dif[start:] - dea)
    return float(dif[-1]), float(dea[-1]), float(hist[-1])


def _interpret_macd(dif: float, dea: float, hist: float, prev_hist: float) -> Signal:
    if prev_hist <= 0 < hist:
        return Signal(dimension="技术指标", name="MACD", value=round(hist, 3),
                      signal="bullish", detail="MACD金叉 (柱线翻红)")
    if prev_hist >= 0 > hist:
        return Signal(dimension="技术指标", name="MACD", value=round(hist, 3),
                      signal="bearish", detail="MACD死叉 (柱线翻绿)")
    if hist > 0 and hist > prev_hist:
        return Signal(dimension="技术指标", name="MACD", value=round(hist, 3),
                      signal="bullish", detail=f"MACD红柱放大 ({hist:.3f})")
    if hist < 0 and hist < prev_hist:
        return Signal(dimension="技术指标", name="MACD", value=round(hist, 3),
                      signal="bearish", detail=f"MACD绿柱放大 ({hist:.3f})")
    if hist < 0 and hist > prev_hist:
        return Signal(dimension="技术指标", name="MACD", value=round(hist, 3),
                      signal="neutral", detail=f"MACD绿柱收窄, 空方衰减")
    return Signal(dimension="技术指标", name="MACD", value=round(hist, 3),
                  signal="neutral", detail=f"MACD柱={hist:.3f}")


# === Bollinger Bands ===

def calc_bollinger(close: np.ndarray, period: int = 20, num_std: float = 2.0
                   ) -> tuple[float, float, float, float]:
    """Returns (upper, middle, lower, %B)."""
    mid = close[-period:].mean()
    std = close[-period:].std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = upper - lower
    pct_b = (close[-1] - lower) / width if width > 0 else 0.5
    return float(upper), float(mid), float(lower), float(pct_b)


def _interpret_bollinger(pct_b: float, bandwidth: float) -> Signal:
    if pct_b < 0:
        return Signal(dimension="技术指标", name="布林带", value=round(pct_b, 2),
                      signal="bullish", detail=f"%B={pct_b:.2f}, 跌破下轨, 极度超卖")
    if pct_b < 0.2:
        return Signal(dimension="技术指标", name="布林带", value=round(pct_b, 2),
                      signal="bullish", detail=f"%B={pct_b:.2f}, 接近下轨")
    if pct_b > 1.0:
        return Signal(dimension="技术指标", name="布林带", value=round(pct_b, 2),
                      signal="bearish", detail=f"%B={pct_b:.2f}, 突破上轨, 极度超买")
    if pct_b > 0.8:
        return Signal(dimension="技术指标", name="布林带", value=round(pct_b, 2),
                      signal="bearish", detail=f"%B={pct_b:.2f}, 接近上轨")
    if bandwidth < 0.05:
        return Signal(dimension="技术指标", name="布林带", value=round(pct_b, 2),
                      signal="neutral", detail=f"%B={pct_b:.2f}, 布林收窄, 变盘在即")
    return Signal(dimension="技术指标", name="布林带", value=round(pct_b, 2),
                  signal="neutral", detail=f"%B={pct_b:.2f}, 中轨附近")


# === KDJ (Chinese standard: 9,3,3) ===

def calc_kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             n: int = 9, m1: int = 3, m2: int = 3) -> tuple[float, float, float]:
    """Returns (K, D, J)."""
    length = len(close)
    k_val, d_val = 50.0, 50.0
    for i in range(n - 1, length):
        h = high[i - n + 1: i + 1].max()
        l = low[i - n + 1: i + 1].min()
        rsv = (close[i] - l) / (h - l) * 100 if h != l else 50.0
        k_val = (m1 - 1) / m1 * k_val + 1 / m1 * rsv
        d_val = (m2 - 1) / m2 * d_val + 1 / m2 * k_val
    j_val = 3 * k_val - 2 * d_val
    return float(k_val), float(d_val), float(j_val)


def _interpret_kdj(k: float, d: float, j: float) -> Signal:
    if j < 0:
        return Signal(dimension="技术指标", name="KDJ", value=round(j, 1),
                      signal="bullish", detail=f"J={j:.1f}, 极度超卖")
    if k < 20 and k > d:
        return Signal(dimension="技术指标", name="KDJ", value=round(j, 1),
                      signal="bullish", detail=f"K={k:.0f} D={d:.0f} J={j:.0f}, 低位金叉")
    if j > 100:
        return Signal(dimension="技术指标", name="KDJ", value=round(j, 1),
                      signal="bearish", detail=f"J={j:.1f}, 极度超买")
    if k > 80 and k < d:
        return Signal(dimension="技术指标", name="KDJ", value=round(j, 1),
                      signal="bearish", detail=f"K={k:.0f} D={d:.0f} J={j:.0f}, 高位死叉")
    return Signal(dimension="技术指标", name="KDJ", value=round(j, 1),
                  signal="neutral", detail=f"K={k:.0f} D={d:.0f} J={j:.0f}")


# === Volume-Price Divergence ===

def _interpret_vol_price_div(close: np.ndarray, volume: np.ndarray, window: int = 5) -> Signal:
    p_chg = (close[-1] / close[-window - 1] - 1) * 100
    v_chg = volume[-window:].mean() / volume[-2 * window: -window].mean() if len(volume) >= 2 * window else 1.0

    if p_chg > 2 and v_chg < 0.8:
        return Signal(dimension="技术指标", name="量价配合", value=round(v_chg, 2),
                      signal="bearish", detail=f"价涨{p_chg:+.1f}%但量缩{v_chg:.2f}x, 上涨乏力")
    if p_chg < -2 and v_chg > 1.3:
        return Signal(dimension="技术指标", name="量价配合", value=round(v_chg, 2),
                      signal="bullish", detail=f"价跌{p_chg:+.1f}%放量{v_chg:.2f}x, 可能放量见底")
    if p_chg > 2 and v_chg > 1.2:
        return Signal(dimension="技术指标", name="量价配合", value=round(v_chg, 2),
                      signal="bullish", detail=f"价升{p_chg:+.1f}%量增{v_chg:.2f}x, 量价齐升")
    if p_chg < -2 and v_chg < 0.8:
        return Signal(dimension="技术指标", name="量价配合", value=round(v_chg, 2),
                      signal="neutral", detail=f"价跌{p_chg:+.1f}%缩量{v_chg:.2f}x, 下跌动能减弱")
    return Signal(dimension="技术指标", name="量价配合", value=round(v_chg, 2),
                  signal="neutral", detail=f"量价{p_chg:+.1f}%/{v_chg:.2f}x, 无明显背离")


# === N-day New High / New Low ===

def _interpret_new_high_low(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Signal:
    n = len(close)
    for window in [250, 120, 60, 20]:
        if n < window:
            continue
        if close[-1] >= high[-window:].max():
            if window >= 120:
                return Signal(dimension="技术指标", name=f"{window}日新高", value=float(close[-1]),
                              signal="bearish", detail=f"创{window}日新高, 注意追高风险")
            return Signal(dimension="技术指标", name=f"{window}日新高", value=float(close[-1]),
                          signal="bullish", detail=f"创{window}日新高, 趋势强劲")
        if close[-1] <= low[-window:].min():
            if window >= 60:
                return Signal(dimension="技术指标", name=f"{window}日新低", value=float(close[-1]),
                              signal="bullish", detail=f"创{window}日新低, 极度超跌区间")
            return Signal(dimension="技术指标", name=f"{window}日新低", value=float(close[-1]),
                          signal="bearish", detail=f"创{window}日新低, 弱势延续")
    return Signal(dimension="技术指标", name="高低点", value=None,
                  signal="neutral", detail="未创近期新高或新低")


# === Gap Detection (跳空缺口) ===

def _interpret_gaps(open_arr: np.ndarray, high: np.ndarray, low: np.ndarray,
                    close: np.ndarray, lookback: int = 10) -> Signal:
    n = len(close)
    start = max(1, n - lookback)
    latest_gap = None
    for i in range(n - 1, start - 1, -1):
        if low[i] > high[i - 1]:  # up gap
            # Check if filled
            filled = any(low[j] <= high[i - 1] for j in range(i + 1, n))
            if not filled:
                latest_gap = ("up", i, float(high[i - 1]), float(low[i]))
                break
        elif high[i] < low[i - 1]:  # down gap
            filled = any(high[j] >= low[i - 1] for j in range(i + 1, n))
            if not filled:
                latest_gap = ("down", i, float(high[i]), float(low[i - 1]))
                break

    if latest_gap is None:
        return Signal(dimension="技术指标", name="跳空缺口", value=None,
                      signal="neutral", detail="近期无未回补缺口")
    gap_type, idx, gap_lo, gap_hi = latest_gap
    days_ago = n - 1 - idx
    if gap_type == "up":
        return Signal(dimension="技术指标", name="跳空缺口", value=gap_hi,
                      signal="bullish", detail=f"{days_ago}日前向上跳空, 支撑位{gap_hi:.2f}")
    return Signal(dimension="技术指标", name="跳空缺口", value=gap_lo,
                  signal="bearish", detail=f"{days_ago}日前向下跳空, 压力位{gap_lo:.2f}")


# === Orchestrator ===

def compute_all_technical_signals(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_arr: np.ndarray,
    volume: np.ndarray,
) -> list[Signal]:
    """Compute all technical indicator signals from OHLCV arrays (sorted ascending by date)."""
    signals: list[Signal] = []
    n = len(close)

    if n >= 15:
        signals.append(_interpret_rsi(calc_rsi(close)))

    if n >= 35:
        dif, dea, hist = calc_macd(close)
        _, _, prev_hist = calc_macd(close[:-1])
        signals.append(_interpret_macd(dif, dea, hist, prev_hist))

    if n >= 20:
        upper, mid, lower, pct_b = calc_bollinger(close)
        bw = (upper - lower) / mid if mid > 0 else 0
        signals.append(_interpret_bollinger(pct_b, bw))

    if n >= 9:
        k, d, j = calc_kdj(high, low, close)
        signals.append(_interpret_kdj(k, d, j))

    if n >= 10:
        signals.append(_interpret_vol_price_div(close, volume))

    if n >= 20:
        signals.append(_interpret_new_high_low(close, high, low))

    if n >= 10:
        signals.append(_interpret_gaps(open_arr, high, low, close, lookback=10))

    return signals
