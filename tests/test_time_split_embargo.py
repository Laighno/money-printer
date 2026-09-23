"""Lock-in tests for the date-level train/val split + label-horizon embargo
used by ``StockRanker.train_fast`` (``mp.ml.model._time_split``).

Bug being prevented: the old split was by *row position*
(``split_idx = int(n * (1 - val_frac))``) on a date-sorted panel, so the
~800-stock cross-section of one trading day was cut in half between train
and val.  Both halves share the same forward-return window, so early
stopping picked ``best_rounds`` on a leaked validation set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mp.ml.model import DEFAULT_HORIZON, StockRanker, _time_split


def _make_panel(n_stocks: int = 3, n_dates: int = 100, seed: int = 0,
                shuffle: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rows = []
    for d in dates:
        for s in range(n_stocks):
            x = rng.standard_normal(3)
            rows.append({"date": d, "code": f"{s:06d}",
                         "f0": x[0], "f1": x[1], "f2": x[2],
                         "fwd_ret": 0.3 * x[0] + 0.7 * rng.standard_normal()})
    df = pd.DataFrame(rows)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def _date_positions(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Positions (0..n_dates-1) of the unique dates selected by *mask*."""
    unique = np.sort(df["date"].unique())
    pos = {d: i for i, d in enumerate(unique)}
    return np.array(sorted(pos[d] for d in df.loc[mask, "date"].unique()))


@pytest.mark.parametrize("shuffle", [False, True])
def test_no_date_straddles_train_and_val(shuffle):
    df = _make_panel(n_stocks=3, n_dates=100, shuffle=shuffle)
    train_mask, val_mask = _time_split(df, val_frac=0.15, horizon=DEFAULT_HORIZON)

    train_dates = set(df.loc[train_mask, "date"])
    val_dates = set(df.loc[val_mask, "date"])
    assert train_dates, "train must not be empty"
    assert val_dates, "val must not be empty"
    assert not (train_dates & val_dates), "a trading day must never be in both sets"
    assert not (train_mask & val_mask).any()

    # Every row of a selected date is selected (whole cross-section moves together)
    for d in val_dates:
        assert val_mask[(df["date"] == d).values].all()
    for d in train_dates:
        assert train_mask[(df["date"] == d).values].all()


def test_embargo_gap_is_at_least_horizon_trading_days():
    df = _make_panel(n_stocks=3, n_dates=100)
    horizon = DEFAULT_HORIZON
    train_mask, val_mask = _time_split(df, val_frac=0.15, horizon=horizon)

    train_pos = _date_positions(df, train_mask)
    val_pos = _date_positions(df, val_mask)

    # 100 dates * 0.85 -> val starts at position 85, train ends before 85-20=65
    assert val_pos.min() == 85
    assert val_pos.max() == 99
    assert train_pos.max() == 64
    gap = val_pos.min() - train_pos.max() - 1   # trading days strictly between
    assert gap >= horizon, f"embargo gap {gap} < horizon {horizon}"

    # The label of the last train row (close[D+horizon]) must not reach val
    assert train_pos.max() + horizon < val_pos.min()


def test_val_is_the_most_recent_dates_and_contiguous():
    df = _make_panel(n_stocks=3, n_dates=100)
    train_mask, val_mask = _time_split(df, val_frac=0.15, horizon=DEFAULT_HORIZON)
    train_pos = _date_positions(df, train_mask)
    val_pos = _date_positions(df, val_mask)
    assert val_pos.tolist() == list(range(85, 100))
    assert train_pos.tolist() == list(range(0, 65))
    assert df.loc[train_mask, "date"].max() < df.loc[val_mask, "date"].min()


def test_horizon_zero_means_no_gap():
    df = _make_panel(n_stocks=3, n_dates=100)
    train_mask, val_mask = _time_split(df, val_frac=0.15, horizon=0)
    assert (train_mask | val_mask).all(), "with horizon=0 every row is used"
    assert not (train_mask & val_mask).any()
    assert _date_positions(df, train_mask).max() == 84
    assert _date_positions(df, val_mask).min() == 85


def test_too_few_dates_falls_back_to_no_embargo_with_warning():
    # 20 dates, val_frac 0.15 -> split_pos = 17 <= horizon 20 -> no embargo
    df = _make_panel(n_stocks=3, n_dates=20)
    warnings: list[str] = []
    from loguru import logger
    sink_id = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
    try:
        train_mask, val_mask = _time_split(df, val_frac=0.15, horizon=DEFAULT_HORIZON)
    finally:
        logger.remove(sink_id)
    assert (train_mask | val_mask).all()
    assert not (train_mask & val_mask).any()
    assert any("embargo" in w for w in warnings), warnings


def test_train_fast_uses_date_split_end_to_end():
    """Integration: train_fast on 3 stocks x 100 days must run the early-
    stopping path (train >= 100 rows, val >= 20 rows) and evaluate only on
    whole trailing dates."""
    df = _make_panel(n_stocks=40, n_dates=100)
    ranker = StockRanker(feature_cols=["f0", "f1", "f2"], label_col="fwd_ret")
    metrics = ranker.train_fast(df, val_frac=0.15)
    assert ranker.model is not None
    assert set(metrics) >= {"mae", "ic", "best_rounds"}
    assert not np.isnan(metrics["mae"]), "fallback path must not trigger on this panel"
    assert ranker.model.best_iteration >= 1
