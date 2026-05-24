"""Regression test for the silent-failure mode where StockRanker.train_fast
did not populate self.feature_importance, so feature_importance_audit.py
got gain_pct=0 across the board.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mp.ml.model import StockRanker


def _make_panel(n_dates: int = 30, n_stocks: int = 60, n_features: int = 3,
                seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rows = []
    for d in dates:
        X = rng.standard_normal((n_stocks, n_features))
        # Signal: first feature explains 30% of variance in label
        y = 0.3 * X[:, 0] + 0.7 * rng.standard_normal(n_stocks)
        for i in range(n_stocks):
            rows.append({"date": d, **{f"f{j}": X[i, j] for j in range(n_features)},
                         "fwd_ret": y[i]})
    return pd.DataFrame(rows)


def test_train_fast_populates_feature_importance():
    panel = _make_panel()
    feats = ["f0", "f1", "f2"]
    ranker = StockRanker(feature_cols=feats, label_col="fwd_ret")
    ranker.train_fast(panel)
    assert ranker.feature_importance, "train_fast must populate feature_importance"
    assert set(ranker.feature_importance) == set(feats)
    assert any(v > 0 for v in ranker.feature_importance.values()), \
        "at least one feature should have non-zero gain"
