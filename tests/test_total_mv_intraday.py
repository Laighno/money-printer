"""Regression test for 2026-05-13 bug: ``total_mv_log`` value-space jump
between intraday and EOD predictions.

Root cause
----------
``_process_single_stock`` derives ``total_mv_log`` from daily bars via
``close * volume / turnover_rate`` (= 流通市值, float-cap).  When the
``intraday_bar`` injection path lacks ``turnover`` (sina realtime endpoint
doesn't carry it), the bars-derived value is NaN and the code used to
fall back to the cached valuation-snapshot ``total_mv`` (= 总市值,
total-cap).  The two spaces differ by 1.5-3× for most A-shares (e.g.
粤电力A 流通 ~196亿 vs 总 ~374亿 → log diff ≈ 0.64).

Training data uses one value space per stock-day, so the model has never
seen a within-stock cross-space jump.  Result: intraday predictions
diverged from EOD predictions by 3-4pp for the same data day.

Fix verified by 粤电力A on 2026-05-13: 14:00 intraday and EOD
predictions become identical (+3.02%) after fix; previously +3.73pp gap.

This test pins the invariant: ``total_mv_log`` for an intraday row
**must inherit the previous day's bars-derived value** (scaled by close
ratio), not the cached snapshot value.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _fake_history_df(code: str, n_days: int = 100, *, base_close: float = 7.0,
                     turnover_rate: float = 0.05, float_shares: float = 2.5e9) -> pd.DataFrame:
    """Build a synthetic daily-bars history where every row has valid turnover.
    Float-cap = base_close × float_shares ≈ 17.5e9 → log ≈ 23.6.
    """
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_days):
        d = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
        close = base_close * (1 + rng.normal(0, 0.01))
        volume = turnover_rate * float_shares
        rows.append({
            "code": code, "date": d,
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": volume, "amount": close * volume,
            "turnover": turnover_rate,
        })
    return pd.DataFrame(rows)


def test_intraday_total_mv_inherits_bars_space_not_cache(monkeypatch):
    """When intraday_bar lacks turnover, total_mv_log must inherit the
    previous day's bars-derived value (scaled by close ratio), NOT the
    cached snapshot total_mv value."""
    from mp.ml import dataset as ds

    code = "000539"
    hist = _fake_history_df(code, n_days=100, base_close=7.0,
                            turnover_rate=0.05, float_shares=2.5e9)
    # Bars-derived float-cap (constant ish): close × volume / turnover
    #   = 7.0 × (0.05 × 2.5e9) / 0.05 = 7.0 × 2.5e9 = 1.75e10 → log ≈ 23.59
    expected_float_mv_log = np.log(7.0 * 2.5e9)   # ≈ 23.5876

    # Cached snapshot returns TOTAL-cap (5× float-cap), simulating the
    # value-space mismatch.
    cached_total_mv = 5 * 7.0 * 2.5e9   # ≈ 8.75e10 → log ≈ 25.20
    cached_total_mv_log = np.log(cached_total_mv)

    def _fake_align(dates, fin_hist, valuation_row=None, valuation_hist=None):
        n = len(dates)
        return pd.DataFrame({
            "pe_ttm": [np.nan] * n, "pb": [np.nan] * n,
            "total_mv_log": [cached_total_mv_log] * n,
            "roe": [np.nan]*n, "revenue_growth": [np.nan]*n,
            "profit_growth": [np.nan]*n, "roe_qoq": [np.nan]*n,
            "profit_growth_accel": [np.nan]*n, "revenue_growth_accel": [np.nan]*n,
        })

    monkeypatch.setattr(ds, "_align_fundamentals_to_dates", _fake_align)
    monkeypatch.setattr(ds, "get_daily_bars",
                        lambda code, start, end=None: hist)

    # Inject intraday_bar with NO turnover (simulating sina realtime path)
    intraday_close = 7.50   # +7% above history average
    intraday_bar = {
        "date": pd.Timestamp("2026-04-15"),   # past end of hist to force append
        "open": intraday_close, "high": intraday_close, "low": intraday_close,
        "close": intraday_close,
        "volume": 1.0e8, "amount": intraday_close * 1.0e8,
        # NB: no "turnover" key — this is the bug condition
    }

    part = ds._process_single_stock(
        code, start="20260101", end=None, horizon=None,
        fin_hist=None, valuation_row=None,
        intraday_bar=intraday_bar,
    )
    assert part is not None and not part.empty

    last_row = part.iloc[-1]
    mv_log_today = float(last_row["total_mv_log"])

    # Expected: inherit prev-day bars-derived + log(today_close / prev_close)
    # Prev close ≈ 7.0 (within ±1% noise), so expected ≈ 23.59 + log(7.50/7.0) ≈ 23.66
    prev_close = float(hist.iloc[-1]["close"])
    expected_today = expected_float_mv_log + np.log(intraday_close / prev_close)

    # Must be in the bars-derived (float-cap) space, NOT the cache space
    assert abs(mv_log_today - expected_today) < 0.05, (
        f"total_mv_log {mv_log_today:.4f} should ≈ {expected_today:.4f} "
        f"(inherited from bars-derived), not the cached {cached_total_mv_log:.4f}"
    )
    # Strong assertion: must be much closer to bars-derived than cache
    assert abs(mv_log_today - expected_float_mv_log) < abs(mv_log_today - cached_total_mv_log), (
        f"total_mv_log {mv_log_today:.4f} fell back to cache "
        f"({cached_total_mv_log:.4f}) instead of bars-derived ({expected_float_mv_log:.4f})"
    )


def test_eod_path_unchanged_when_all_turnover_valid(monkeypatch):
    """For training-time / EOD path where every bar has valid turnover,
    total_mv_log must be bars-derived as before (no behavior change)."""
    from mp.ml import dataset as ds

    code = "000539"
    hist = _fake_history_df(code, n_days=100, base_close=7.0,
                            turnover_rate=0.05, float_shares=2.5e9)
    expected = np.log(7.0 * 2.5e9)

    def _fake_align(dates, fin_hist, valuation_row=None, valuation_hist=None):
        n = len(dates)
        # Cache holds different (total-cap) value — should be ignored entirely
        return pd.DataFrame({
            "pe_ttm": [np.nan] * n, "pb": [np.nan] * n,
            "total_mv_log": [99.0] * n,    # absurd value to make sure it's not used
            "roe": [np.nan]*n, "revenue_growth": [np.nan]*n,
            "profit_growth": [np.nan]*n, "roe_qoq": [np.nan]*n,
            "profit_growth_accel": [np.nan]*n, "revenue_growth_accel": [np.nan]*n,
        })

    monkeypatch.setattr(ds, "_align_fundamentals_to_dates", _fake_align)
    monkeypatch.setattr(ds, "get_daily_bars",
                        lambda code, start, end=None: hist)

    part = ds._process_single_stock(
        code, start="20260101", end=None, horizon=None,
        fin_hist=None, valuation_row=None, intraday_bar=None,
    )
    assert part is not None and not part.empty
    # Every row's total_mv_log should be ≈ bars-derived, not 99
    mv_logs = part["total_mv_log"].dropna().to_numpy()
    assert (mv_logs < 25).all(), \
        f"Some rows still use cached value (99.0): max={mv_logs.max():.2f}"
    assert abs(mv_logs.mean() - expected) < 0.05, \
        f"Mean total_mv_log {mv_logs.mean():.4f} ≠ expected bars-derived {expected:.4f}"


def test_full_sina_history_falls_back_to_cache(monkeypatch):
    """Edge case: when NO row in history has valid turnover (sina-only data
    source from training era), fall back to cache value as before."""
    from mp.ml import dataset as ds

    code = "000539"
    hist = _fake_history_df(code, n_days=100, base_close=7.0,
                            turnover_rate=0.05, float_shares=2.5e9)
    hist["turnover"] = np.nan   # kill all turnover values

    cache_value = 25.20

    def _fake_align(dates, fin_hist, valuation_row=None, valuation_hist=None):
        n = len(dates)
        return pd.DataFrame({
            "pe_ttm": [np.nan] * n, "pb": [np.nan] * n,
            "total_mv_log": [cache_value] * n,
            "roe": [np.nan]*n, "revenue_growth": [np.nan]*n,
            "profit_growth": [np.nan]*n, "roe_qoq": [np.nan]*n,
            "profit_growth_accel": [np.nan]*n, "revenue_growth_accel": [np.nan]*n,
        })

    monkeypatch.setattr(ds, "_align_fundamentals_to_dates", _fake_align)
    monkeypatch.setattr(ds, "get_daily_bars",
                        lambda code, start, end=None: hist)

    part = ds._process_single_stock(
        code, start="20260101", end=None, horizon=None,
        fin_hist=None, valuation_row=None, intraday_bar=None,
    )
    assert part is not None and not part.empty
    # All rows should use cached value
    mv_logs = part["total_mv_log"].dropna().to_numpy()
    assert all(abs(v - cache_value) < 0.01 for v in mv_logs), \
        f"Expected all rows = cache {cache_value}, got range " \
        f"[{mv_logs.min():.4f}, {mv_logs.max():.4f}]"
