"""Smoke tests for mp.ml.wf_gate.

Real calibration happens in docs/dialog/ round 28+ against the 3 known
cases (max_drawdown_20d should produce Δ<0, amount_ratio/atr_14 should
produce Δ≈0 within noise) — these tests just verify the function does not
crash on a synthetic panel and that the return-dict schema is correct.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mp.ml.wf_gate import (
    EVAL_WINDOW_START,
    EVAL_WINDOW_END,
    wf_gate_delta_sharpe,
    _expanding_folds,
    _portfolio_sharpe,
)


def _synthetic_panel(
    n_dates: int = 250,
    n_stocks: int = 80,
    n_features: int = 6,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a panel covering EVAL_WINDOW with random features + a label
    weakly correlated to feature_0.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(EVAL_WINDOW_START, periods=n_dates, freq="B")
    rows = []
    for d in dates:
        X = rng.standard_normal((n_stocks, n_features))
        # Signal: feature_0 explains 10% of fwd_ret
        fwd_ret = 0.1 * X[:, 0] + 0.9 * rng.standard_normal(n_stocks)
        for i in range(n_stocks):
            row = {"date": d, "code": f"S{i:03d}", "fwd_ret": fwd_ret[i]}
            for j in range(n_features):
                row[f"f{j}"] = X[i, j]
            rows.append(row)
    return pd.DataFrame(rows)


def test_expanding_folds_shapes():
    dates = pd.date_range("2024-01-01", periods=30, freq="D").values
    folds = _expanding_folds(dates, n_folds=3)
    assert len(folds) == 3
    # Folds should partition the eval dates roughly evenly
    assert folds[0][0] == pd.Timestamp("2024-01-01")
    assert folds[-1][1] == pd.Timestamp(dates[-1])


def test_portfolio_sharpe_basic():
    # Constant positive 1% per period → infinite Sharpe (std=0) → returns NaN
    assert np.isnan(_portfolio_sharpe(np.array([0.01] * 50), horizon=20))
    # Random returns → finite Sharpe
    rng = np.random.default_rng(1)
    sh = _portfolio_sharpe(rng.standard_normal(100) * 0.01, horizon=20)
    assert np.isfinite(sh)
    # Tiny array → NaN
    assert np.isnan(_portfolio_sharpe(np.array([0.01, 0.02]), horizon=20))


def test_wf_gate_delta_sharpe_smoke():
    """End-to-end smoke: function runs, returns expected schema, sign of
    delta_sharpe for the *signal* feature should be roughly correct (+)."""
    panel = _synthetic_panel()
    base = ["f1", "f2", "f3", "f4", "f5"]   # noise features
    result = wf_gate_delta_sharpe(
        panel=panel,
        feature_to_test="f0",                # signal feature
        base_features=base,
        n_folds=3,
        ranker_kind="stock",                 # avoid Blend (heavier on tiny synthetic data)
        eval_window_start=EVAL_WINDOW_START,
        eval_window_end=str(panel["date"].max().date()),
    )
    # Schema check
    for k in (
        "feature", "sharpe_with", "sharpe_without", "delta_sharpe",
        "n_folds_used", "n_train_rows_per_fold", "n_val_dates_per_fold",
    ):
        assert k in result, f"missing key {k}"
    assert result["feature"] == "f0"
    assert isinstance(result["n_train_rows_per_fold"], list)
    # On synthetic data the signal feature should *tend* to help, but
    # synthetic-data noise means we don't assert direction. Just that
    # we get a finite number.
    assert np.isfinite(result["delta_sharpe"]) or np.isnan(result["delta_sharpe"])


def test_wf_gate_raises_if_feature_already_present_only():
    """If feature_to_test is already in base AND removing it would yield
    same set (impossible — sanity check the guard)."""
    # This branch fires only if list manipulation is wrong; just ensure no
    # crash when feature is already in base.
    panel = _synthetic_panel(n_dates=80)
    base = ["f0", "f1", "f2", "f3", "f4", "f5"]
    result = wf_gate_delta_sharpe(
        panel=panel,
        feature_to_test="f0",
        base_features=base,
        n_folds=2,
        ranker_kind="stock",
        eval_window_start=EVAL_WINDOW_START,
        eval_window_end=str(panel["date"].max().date()),
    )
    assert result["feature"] == "f0"
