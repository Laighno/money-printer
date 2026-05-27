"""P11-1 tests for mp.ml.intraday_features.

Pure feature math — no database, no model load, no I/O.  Covers:

- INTRADAY_EXTRA_COLUMNS / INTRADAY_FEATURE_COLS shape contract
- compute_intraday_extras correctness on known numeric inputs
- NaN/zero/missing-key handling (does not raise)
- morning_vol_ratio fallback when eod_history is None / too short
- Idempotence: calling compute_intraday_extras twice returns identical dicts

P11-2 walk-forward will exercise build_intraday_panel against real data.
For P11-1 we verify only the schema/math contract — build_intraday_panel
itself is thin orchestration and is covered by the dataset.py tests of
build_latest_features (intraday_bars injection already validated by the
midday-report tests).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mp.ml.dataset import FACTOR_COLUMNS
from mp.ml.intraday_features import (
    INTRADAY_EXTRA_COLUMNS,
    INTRADAY_FEATURE_COLS,
    compute_intraday_extras,
)


# ─────────────────────────────────────────────────────────────────────
# Schema contract (locks the column shape — P11-2 walk_forward will
# train against exactly this; any change here cascades to retraining)
# ─────────────────────────────────────────────────────────────────────


def test_intraday_extra_columns_are_exactly_three():
    """Advisor (round 73) asked for 1-3 intraday-specific features. Lock at 3."""
    assert len(INTRADAY_EXTRA_COLUMNS) == 3, (
        f"INTRADAY_EXTRA_COLUMNS should be 3, got {len(INTRADAY_EXTRA_COLUMNS)}: "
        f"{INTRADAY_EXTRA_COLUMNS}"
    )


def test_intraday_extra_columns_names():
    """Lock the exact names — model artifacts (P11-2) will refer to these by name."""
    assert INTRADAY_EXTRA_COLUMNS == [
        "morning_return",
        "morning_vwap_dev",
        "morning_vol_ratio",
    ]


def test_intraday_feature_cols_extends_factor_columns():
    """INTRADAY_FEATURE_COLS = FACTOR_COLUMNS + INTRADAY_EXTRA_COLUMNS (append, no removal)."""
    assert INTRADAY_FEATURE_COLS[: len(FACTOR_COLUMNS)] == FACTOR_COLUMNS, (
        "INTRADAY_FEATURE_COLS prefix must equal FACTOR_COLUMNS — "
        "P11-1 is APPEND-only; do not reorder or drop EOD features."
    )
    assert INTRADAY_FEATURE_COLS[len(FACTOR_COLUMNS):] == INTRADAY_EXTRA_COLUMNS


def test_intraday_feature_cols_length():
    """Length = base FACTOR_COLUMNS + 3 extras."""
    assert len(INTRADAY_FEATURE_COLS) == len(FACTOR_COLUMNS) + 3


def test_intraday_feature_cols_unique():
    """No accidental duplicates from the append."""
    assert len(set(INTRADAY_FEATURE_COLS)) == len(INTRADAY_FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────────
# compute_intraday_extras — happy path
# ─────────────────────────────────────────────────────────────────────


def test_morning_return_basic():
    """Open 10.00, 14:30 close 10.50 → +5.0% morning return."""
    bar = {"open": 10.0, "high": 10.6, "low": 9.9, "close": 10.5,
           "volume": 1_000_000, "amount": 10_300_000}
    out = compute_intraday_extras(bar)
    assert out["morning_return"] == pytest.approx(0.05)


def test_morning_return_negative():
    """Open 20.00, close 19.00 → -5.0% morning return."""
    bar = {"open": 20.0, "high": 20.1, "low": 18.9, "close": 19.0,
           "volume": 500_000, "amount": 9_700_000}
    out = compute_intraday_extras(bar)
    assert out["morning_return"] == pytest.approx(-0.05)


def test_morning_vwap_dev_positive():
    """VWAP = 10.0 (amount 10M / vol 1M), close 10.5 → +5% above VWAP (late buying)."""
    bar = {"open": 9.8, "high": 10.5, "low": 9.7, "close": 10.5,
           "volume": 1_000_000, "amount": 10_000_000}
    out = compute_intraday_extras(bar)
    assert out["morning_vwap_dev"] == pytest.approx(0.05)


def test_morning_vwap_dev_close_at_vwap():
    """Close == VWAP → zero deviation."""
    bar = {"open": 9.8, "high": 10.2, "low": 9.7, "close": 10.0,
           "volume": 1_000_000, "amount": 10_000_000}
    out = compute_intraday_extras(bar)
    assert out["morning_vwap_dev"] == pytest.approx(0.0)


def test_morning_vol_ratio_with_history():
    """Morning vol 2M vs 20d EOD vol MA 1M → ratio = 2.0."""
    hist = pd.DataFrame({"volume": [1_000_000] * 25})  # >= 20
    bar = {"open": 10.0, "high": 10.5, "low": 9.9, "close": 10.3,
           "volume": 2_000_000, "amount": 20_300_000}
    out = compute_intraday_extras(bar, eod_history=hist)
    assert out["morning_vol_ratio"] == pytest.approx(2.0)


def test_morning_vol_ratio_uses_only_tail_20():
    """Only the last 20 EOD bars participate in the MA."""
    # First 10 entries are wild outliers; last 20 are a steady 500k.
    vols = [10_000_000] * 10 + [500_000] * 20
    hist = pd.DataFrame({"volume": vols})
    bar = {"open": 10.0, "high": 10.5, "low": 9.9, "close": 10.3,
           "volume": 1_000_000, "amount": 10_300_000}
    out = compute_intraday_extras(bar, eod_history=hist)
    # MA of last 20 = 500k; ratio = 1M / 500k = 2.0
    assert out["morning_vol_ratio"] == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────────
# compute_intraday_extras — graceful degradation (no raises)
# ─────────────────────────────────────────────────────────────────────


def test_empty_bar_returns_nan_triple():
    """Empty dict → all three extras NaN, no raise."""
    out = compute_intraday_extras({})
    for col in INTRADAY_EXTRA_COLUMNS:
        assert math.isnan(out[col]), f"{col} expected NaN, got {out[col]}"


def test_zero_open_returns_nan_morning_return():
    """Open == 0 (data glitch) → morning_return NaN, others may still compute."""
    bar = {"open": 0.0, "high": 10.0, "low": 9.9, "close": 10.0,
           "volume": 1_000_000, "amount": 10_000_000}
    out = compute_intraday_extras(bar)
    assert math.isnan(out["morning_return"])
    # VWAP is still computable
    assert not math.isnan(out["morning_vwap_dev"])


def test_zero_volume_returns_nan_vwap_and_vol_ratio():
    """Volume == 0 → morning_vwap_dev NaN AND morning_vol_ratio NaN."""
    hist = pd.DataFrame({"volume": [1_000_000] * 25})
    bar = {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
           "volume": 0, "amount": 0}
    out = compute_intraday_extras(bar, eod_history=hist)
    assert math.isnan(out["morning_vwap_dev"])
    assert math.isnan(out["morning_vol_ratio"])
    assert out["morning_return"] == pytest.approx(0.0)


def test_short_history_returns_nan_vol_ratio():
    """eod_history shorter than 20 rows → morning_vol_ratio NaN, others OK."""
    hist = pd.DataFrame({"volume": [1_000_000] * 10})  # < 20
    bar = {"open": 10.0, "high": 10.5, "low": 9.9, "close": 10.3,
           "volume": 2_000_000, "amount": 20_300_000}
    out = compute_intraday_extras(bar, eod_history=hist)
    assert math.isnan(out["morning_vol_ratio"])
    assert not math.isnan(out["morning_return"])
    assert not math.isnan(out["morning_vwap_dev"])


def test_no_history_returns_nan_vol_ratio():
    """eod_history=None → morning_vol_ratio NaN, others OK."""
    bar = {"open": 10.0, "high": 10.5, "low": 9.9, "close": 10.3,
           "volume": 2_000_000, "amount": 20_300_000}
    out = compute_intraday_extras(bar, eod_history=None)
    assert math.isnan(out["morning_vol_ratio"])
    assert not math.isnan(out["morning_return"])
    assert not math.isnan(out["morning_vwap_dev"])


def test_missing_volume_column_in_history():
    """eod_history without 'volume' column → morning_vol_ratio NaN, no raise."""
    hist = pd.DataFrame({"close": [10.0] * 25})  # no volume column
    bar = {"open": 10.0, "high": 10.5, "low": 9.9, "close": 10.3,
           "volume": 2_000_000, "amount": 20_300_000}
    out = compute_intraday_extras(bar, eod_history=hist)
    assert math.isnan(out["morning_vol_ratio"])


def test_string_inputs_coerced():
    """OHLCV may arrive as strings from JSON sources — coerce, don't raise."""
    bar = {"open": "10.0", "high": "10.5", "low": "9.9", "close": "10.5",
           "volume": "1000000", "amount": "10300000"}
    out = compute_intraday_extras(bar)
    assert out["morning_return"] == pytest.approx(0.05)


# ─────────────────────────────────────────────────────────────────────
# Idempotence + determinism
# ─────────────────────────────────────────────────────────────────────


def test_idempotent_no_history():
    """Calling twice produces identical dicts (no hidden state)."""
    bar = {"open": 10.0, "high": 10.5, "low": 9.9, "close": 10.3,
           "volume": 1_000_000, "amount": 10_200_000}
    out1 = compute_intraday_extras(bar)
    out2 = compute_intraday_extras(bar)
    assert out1 == out2 or all(
        (math.isnan(a) and math.isnan(b)) or a == b
        for a, b in zip(out1.values(), out2.values())
    )


def test_idempotent_with_history():
    """Same with history — function is pure."""
    hist = pd.DataFrame({"volume": np.linspace(800_000, 1_200_000, 25)})
    bar = {"open": 10.0, "high": 10.5, "low": 9.9, "close": 10.3,
           "volume": 1_500_000, "amount": 15_400_000}
    out1 = compute_intraday_extras(bar, eod_history=hist)
    out2 = compute_intraday_extras(bar, eod_history=hist)
    assert out1 == out2
