"""Regression tests for the 2026-05-23 excess_ret winsorization.

Caps |excess_ret| at ±50% so MSE-trained LGBM isn't pulled by qfq-artefact
tail values (411 such rows existed before fix, max 4.49 = +449%).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_add_excess_ret_clips_outliers(monkeypatch):
    """add_excess_ret caps |excess| at ±50%."""
    from mp.ml import dataset as ds

    # Mock benchmark fwd_ret = 0 for all dates → excess_ret = fwd_ret
    def _fake_bench(dates, horizon=20):
        unique = pd.Series(dates).drop_duplicates()
        return pd.DataFrame({"date": unique, "bench_fwd_ret": 0.0})
    monkeypatch.setattr(ds, "_compute_benchmark_fwd_ret", _fake_bench)

    df = pd.DataFrame({
        "code": ["A", "B", "C", "D", "E"],
        "date": pd.to_datetime(["2025-01-01"] * 5),
        "fwd_ret": [0.10, -0.30, 4.49, -0.55, 0.01],   # 4.49 + 0.55 should clip
    })
    out = ds.add_excess_ret(df.copy())
    excess = out["excess_ret"].tolist()
    assert excess[0] == pytest.approx(0.10)
    assert excess[1] == pytest.approx(-0.30)
    assert excess[2] == pytest.approx(0.50)    # clipped
    assert excess[3] == pytest.approx(-0.50)   # clipped
    assert excess[4] == pytest.approx(0.01)


def test_winsorize_threshold_is_50pct(monkeypatch):
    """Sanity: default threshold = 50% (EXCESS_CAP is env-overridable, so
    check the resolved value rather than the source text)."""
    import importlib

    monkeypatch.delenv("EXCESS_CAP", raising=False)
    from mp.ml import dataset as ds
    ds = importlib.reload(ds)
    assert ds.EXCESS_CAP == pytest.approx(0.50), "default threshold should be 50%"


def test_no_winsorize_below_threshold(monkeypatch):
    """Values within ±50% should be untouched."""
    from mp.ml import dataset as ds
    monkeypatch.setattr(ds, "_compute_benchmark_fwd_ret",
                        lambda dates, horizon=20: pd.DataFrame({
                            "date": pd.Series(dates).drop_duplicates(),
                            "bench_fwd_ret": 0.0,
                        }))
    df = pd.DataFrame({
        "code": ["A", "B", "C"],
        "date": pd.to_datetime(["2025-01-01"] * 3),
        "fwd_ret": [0.40, -0.49, 0.30],   # all within ±50%
    })
    out = ds.add_excess_ret(df.copy())
    assert out["excess_ret"].tolist() == pytest.approx([0.40, -0.49, 0.30])
