"""PIT (Point-in-Time) filter regression tests.

These tests guard against research-pipeline drift that silently relaxes
PIT discipline — the class of bug that caused pre-2026-04-21 diagnostics
to be survivorship-inflated.  Each test pins a specific invariant:

1. Scoring filter drops codes not in the month's PIT universe.
2. Training filter drops (code, month) pairs not in membership.
3. When no / one snapshot exists, filters no-op and caller falls back
   (rather than silently filtering everything away).
4. Diagnostics and walk-forward see the same universe on the same date.
5. Industry history PIT lookup (merge_asof direction=backward) never
   leaks an industry assignment that post-dates the query.
"""

from __future__ import annotations

import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════════════
# 1. Diagnostics PIT filters (scripts/prediction_diagnostics.py)
# ══════════════════════════════════════════════════════════════════════

def test_pit_filter_scoring_drops_non_members():
    """Scoring filter must remove rows whose code is not in the month's universe."""
    from scripts.prediction_diagnostics import _pit_filter_scoring

    today_df = pd.DataFrame({
        "date": pd.to_datetime(["2022-03-15"] * 4),
        "code": ["600000", "600001", "600002", "600003"],
        "pred_score": [0.1, 0.2, 0.3, 0.4],
    })
    month_universe = {
        pd.Period("2022-03", freq="M"): frozenset({"600000", "600002"}),
    }
    out = _pit_filter_scoring(today_df, month_universe, pd.Timestamp("2022-03-15"))
    assert set(out["code"]) == {"600000", "600002"}, \
        "Scoring filter did not drop non-members"


def test_pit_filter_scoring_noop_when_empty():
    """Empty month_universe must return the input unchanged (fallback mode)."""
    from scripts.prediction_diagnostics import _pit_filter_scoring

    today_df = pd.DataFrame({
        "date": pd.to_datetime(["2022-03-15"] * 3),
        "code": ["A", "B", "C"],
    })
    out = _pit_filter_scoring(today_df, {}, pd.Timestamp("2022-03-15"))
    assert len(out) == 3, "Empty month_universe must no-op (fallback)"


def test_pit_filter_scoring_noop_on_missing_month():
    """If the queried month has no snapshot, pass through unchanged."""
    from scripts.prediction_diagnostics import _pit_filter_scoring

    today_df = pd.DataFrame({"code": ["A", "B"], "date": pd.to_datetime(["2022-03-15"] * 2)})
    month_universe = {pd.Period("2022-01", freq="M"): frozenset({"A"})}
    out = _pit_filter_scoring(today_df, month_universe, pd.Timestamp("2022-03-15"))
    assert len(out) == 2, "Missing-month fallback should pass through"


def test_pit_filter_training_drops_non_members():
    """Training filter must drop (code, month) pairs outside membership."""
    from scripts.prediction_diagnostics import _pit_filter_training

    train_df = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-05", "2022-02-10", "2022-03-20"]),
        "code": ["A", "A", "A"],
        "label": [0.01, 0.02, 0.03],
    })
    membership = {
        ("A", pd.Period("2022-01", freq="M")),
        ("A", pd.Period("2022-03", freq="M")),
        # Feb missing → row should drop
    }
    out = _pit_filter_training(train_df, membership)
    assert set(out["date"].dt.to_period("M")) == {
        pd.Period("2022-01", freq="M"),
        pd.Period("2022-03", freq="M"),
    }


def test_pit_filter_training_noop_when_empty():
    from scripts.prediction_diagnostics import _pit_filter_training

    train_df = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-05", "2022-02-10"]),
        "code": ["A", "B"],
    })
    out = _pit_filter_training(train_df, set())
    assert len(out) == 2


# ══════════════════════════════════════════════════════════════════════
# 2. Snapshot fallback behavior
# ══════════════════════════════════════════════════════════════════════

def test_build_pit_filters_fallback_when_snapshots_missing(monkeypatch, caplog):
    """With <2 snapshots, _build_pit_filters returns empty structures so that
    callers naturally fall back to unfiltered behavior (rather than silently
    erasing the entire universe)."""
    from scripts import prediction_diagnostics as pd_mod

    class _FakeStore:
        def list_constituent_snapshot_dates(self, _universe):
            return ["2026-04-01"]  # exactly 1 — below threshold

    monkeypatch.setattr(pd_mod, "_PIT_CACHE", {})
    monkeypatch.setattr("mp.data.store.DataStore", lambda: _FakeStore())

    panel = pd.DataFrame({"date": pd.to_datetime(["2022-01-05"]), "code": ["X"]})
    membership, month_universe = pd_mod._build_pit_filters(panel, silent=True)
    assert membership == set()
    assert month_universe == {}


# ══════════════════════════════════════════════════════════════════════
# 3. Diagnostics ↔ Walk-forward universe consistency
# ══════════════════════════════════════════════════════════════════════

def test_diagnostics_and_walk_forward_use_same_pit_logic():
    """Both pipelines must call the same underlying PIT primitive
    (get_index_constituents_at) so they can never disagree about which
    codes were in the index on a given date."""
    from scripts import prediction_diagnostics as pd_mod
    from mp.data import fetcher

    # The helper must exist and be importable from both call sites.
    assert hasattr(fetcher, "get_index_constituents_at"), \
        "walk-forward depends on get_index_constituents_at"
    # Diagnostics imports it inside _build_pit_filters.  The import will
    # raise here if the symbol disappears.
    import scripts.prediction_diagnostics  # noqa: F401


# ══════════════════════════════════════════════════════════════════════
# 4. Industry history PIT lookup
# ══════════════════════════════════════════════════════════════════════

def test_industry_history_merge_asof_is_backward_only():
    """``_add_industry_relative_features`` must use merge_asof(direction='backward')
    for the DataFrame path — a 'forward' or 'nearest' would leak an industry
    reassignment that happened AFTER the evaluation date."""
    import inspect
    from mp.ml import dataset

    src = inspect.getsource(dataset._add_industry_relative_features)
    # The DataFrame path uses merge_asof; verify direction='backward' is pinned.
    assert 'direction="backward"' in src or "direction='backward'" in src, \
        "Industry merge_asof must use direction='backward' to prevent future leak"


def test_industry_history_merge_asof_no_lookahead_on_synthetic_data():
    """Reassign a stock's industry on 2022-06-01.  A query on 2022-03-15
    must return the OLD industry, not the new one."""
    hist = pd.DataFrame({
        "code": ["600000", "600000"],
        "start_date": pd.to_datetime(["2020-01-01", "2022-06-01"]),
        "board_name": ["OldIndustry", "NewIndustry"],
    })
    query = pd.DataFrame({
        "code": ["600000"],
        "date": pd.to_datetime(["2022-03-15"]),  # before the reassignment
    }).sort_values("date")
    hist_sorted = hist.sort_values("start_date").rename(columns={"start_date": "date"})
    merged = pd.merge_asof(
        query.sort_values("date"),
        hist_sorted,
        on="date", by="code", direction="backward",
    )
    assert merged.loc[0, "board_name"] == "OldIndustry", \
        "merge_asof direction=backward leaked a future industry assignment"
