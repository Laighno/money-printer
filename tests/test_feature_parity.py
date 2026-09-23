"""Train/serve parity for the industry-relative rank features (audit 2026-09-23).

Two skews were found between ``build_dataset`` / walk-forward (training) and
``build_latest_features`` (live inference):

1. Industry source: training feeds ``_add_industry_relative_features`` the
   point-in-time ``get_industry_history`` DataFrame (merge_asof on the row
   date); live used the ``get_industry_mapping`` *current* snapshot dict, so a
   stock that changed industry was ranked among today's peers for every date.
2. Rank pool: training pct-ranks within (date, industry) over the whole panel;
   live ranked only within the codes passed by the caller.  Same stock, same
   day → different ``pe_ind_rank`` depending on who else happened to be scored.

These tests build a synthetic panel (3 industries × 4 stocks × 60 days, one
stock switching industry mid-way) and assert the live path — via the real
production helpers, with the industry source injected — reproduces the
training path value-for-value.  No network, no production DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mp.ml.dataset import (  # noqa: E402
    INDUSTRY_RANK_COLUMNS,
    TECHNICAL_COLUMNS,
    _add_industry_relative_features,
    _attach_live_industry_ranks,
    _resolve_live_industry_source,
)

TOL = 1e-9
INDUSTRIES = {"A": ["000101", "000102", "000103", "000104"],
              "B": ["000201", "000202", "000203", "000204"],
              "C": ["000301", "000302", "000303", "000304"]}
SWITCHER = "000204"          # moves B → C on SWITCH_DAY
SWITCH_DAY = 30
N_DAYS = 60
ALL_CODES = sorted(c for cs in INDUSTRIES.values() for c in cs)


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=N_DAYS)


@pytest.fixture(scope="module")
def panel(dates) -> pd.DataFrame:
    """Full training-style panel: one row per (code, date) with rank inputs."""
    rng = np.random.default_rng(20260923)
    rows = []
    for code in ALL_CODES:
        for d in dates:
            rows.append({
                "date": d,
                "code": code,
                "pe_ttm": float(rng.uniform(5, 80)),
                "pb": float(rng.uniform(0.5, 10)),
                "roe": float(rng.normal(0.08, 0.05)),
                "mom_20d": float(rng.normal(0.0, 0.1)),
                "close": float(rng.uniform(5, 50)),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def ind_hist(dates) -> pd.DataFrame:
    """Synthetic ``get_industry_history`` output (code, start_date, board_name)."""
    rows = []
    for ind, codes in INDUSTRIES.items():
        for c in codes:
            rows.append({"code": c, "start_date": pd.Timestamp("2000-01-01"), "board_name": ind})
    rows.append({"code": SWITCHER, "start_date": dates[SWITCH_DAY], "board_name": "C"})
    hist = pd.DataFrame(rows)
    hist["start_date"] = pd.to_datetime(hist["start_date"])
    return hist.sort_values(["code", "start_date"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def train_ranked(panel, ind_hist) -> pd.DataFrame:
    """Exactly what build_dataset / walk-forward do to the full panel."""
    return _add_industry_relative_features(panel.copy(), ind_hist)


def _latest_rows(panel: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Shape of build_latest_features' pre-rank `result`: one row per code on `day`."""
    return panel[panel["date"] == day].reset_index(drop=True)


def _rank_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("code")[INDUSTRY_RANK_COLUMNS].sort_index()


def _capture_warnings():
    """Context manager collecting loguru WARNING messages."""
    from loguru import logger

    class _Cap:
        def __init__(self):
            self.messages: list[str] = []
            self._id = None

        def __enter__(self):
            self._id = logger.add(lambda m: self.messages.append(m.record["message"]),
                                  level="WARNING")
            return self

        def __exit__(self, *exc):
            logger.remove(self._id)
            return False

    return _Cap()


# ---------------------------------------------------------------------------
# 1. live path == training path, value for value (PIT industry included)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("day_idx", [10, 45])
def test_live_industry_ranks_match_training(panel, ind_hist, train_ranked, dates, day_idx):
    day = dates[day_idx]
    live_in = _latest_rows(panel, day)

    with _capture_warnings() as cap:
        live = _attach_live_industry_ranks(live_in, codes=ALL_CODES,
                                           rank_universe=ALL_CODES,
                                           industry_source=ind_hist)
    assert not any("rank_universe" in m for m in cap.messages), cap.messages

    expected = _rank_frame(train_ranked[train_ranked["date"] == day])
    got = _rank_frame(live)
    assert list(got.index) == list(expected.index)
    assert not got.isna().any().any()
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), atol=TOL, rtol=0)


def test_pit_switcher_is_ranked_in_industry_in_effect(panel, ind_hist, dates):
    """Before SWITCH_DAY the switcher ranks among B (4 peers); after, among C (5)."""
    before, after = dates[SWITCH_DAY - 1], dates[SWITCH_DAY + 5]

    for day, group_codes in ((before, INDUSTRIES["B"]),
                             (after, INDUSTRIES["C"] + [SWITCHER])):
        live_in = _latest_rows(panel, day)
        live = _attach_live_industry_ranks(live_in, codes=ALL_CODES,
                                           rank_universe=ALL_CODES,
                                           industry_source=ind_hist)
        grp = live_in[live_in["code"].isin(group_codes)].set_index("code")
        expected = grp["pe_ttm"].rank(pct=True).loc[SWITCHER]
        got = live.set_index("code").loc[SWITCHER, "pe_ind_rank"]
        assert abs(got - expected) < TOL, (day, got, expected)

    # And the snapshot-dict (legacy live) branch would have got `after`'s label
    # for the `before` date — the skew we are eliminating.
    snapshot = {c: ind for ind, cs in INDUSTRIES.items() for c in cs}
    snapshot[SWITCHER] = "C"
    legacy = _add_industry_relative_features(_latest_rows(panel, before), snapshot)
    pit = _add_industry_relative_features(_latest_rows(panel, before), ind_hist)
    assert legacy.set_index("code").loc[SWITCHER, "pe_ind_rank"] != \
        pytest.approx(pit.set_index("code").loc[SWITCHER, "pe_ind_rank"], abs=TOL)


# ---------------------------------------------------------------------------
# 2. rank pool: subset codes + full rank_universe == full-panel values;
#    no rank_universe → different values + WARNING
# ---------------------------------------------------------------------------

SUBSET = ["000101", "000201", "000301", SWITCHER]


def test_subset_with_rank_universe_matches_full_panel(panel, ind_hist, train_ranked, dates):
    day = dates[45]
    live_in = _latest_rows(panel, day)          # codes ∪ rank_universe rows

    with _capture_warnings() as cap:
        live = _attach_live_industry_ranks(live_in, codes=SUBSET,
                                           rank_universe=ALL_CODES,
                                           industry_source=ind_hist)
    assert not any("rank_universe" in m for m in cap.messages)

    # Peer-only codes never leak into the caller's frame.
    assert sorted(live["code"]) == sorted(SUBSET)

    expected = _rank_frame(train_ranked[(train_ranked["date"] == day)
                                        & train_ranked["code"].isin(SUBSET)])
    np.testing.assert_allclose(_rank_frame(live).to_numpy(), expected.to_numpy(),
                               atol=TOL, rtol=0)


def test_subset_without_rank_universe_is_skewed_and_warns(panel, ind_hist, train_ranked, dates):
    day = dates[45]
    live_in = _latest_rows(panel, day)
    live_in = live_in[live_in["code"].isin(SUBSET)].reset_index(drop=True)

    with _capture_warnings() as cap:
        legacy = _attach_live_industry_ranks(live_in, codes=SUBSET,
                                             rank_universe=None,
                                             industry_source=ind_hist)
    assert any("rank_universe" in m for m in cap.messages), cap.messages

    expected = _rank_frame(train_ranked[(train_ranked["date"] == day)
                                        & train_ranked["code"].isin(SUBSET)])
    got = _rank_frame(legacy)
    # Same stock, same day, different pool → different feature values.
    assert not np.allclose(got.to_numpy(), expected.to_numpy(), atol=TOL, rtol=0)


# ---------------------------------------------------------------------------
# 3. industry source resolution: PIT history preferred, snapshot only as a
#    loud fallback
# ---------------------------------------------------------------------------

def test_resolve_live_industry_source_prefers_pit_history(monkeypatch, ind_hist):
    from mp.data import fetcher

    monkeypatch.setattr(fetcher, "get_industry_history", lambda universe=None: ind_hist)
    monkeypatch.setattr(fetcher, "get_industry_mapping",
                        lambda universe=None: pytest.fail("snapshot must not be consulted"))
    with _capture_warnings() as cap:
        src = _resolve_live_industry_source(ALL_CODES)
    assert isinstance(src, pd.DataFrame)
    assert cap.messages == []


def test_resolve_live_industry_source_falls_back_with_warning(monkeypatch):
    from mp.data import fetcher

    empty = pd.DataFrame(columns=["code", "start_date", "board_name"])
    monkeypatch.setattr(fetcher, "get_industry_history", lambda universe=None: empty)
    monkeypatch.setattr(fetcher, "get_industry_mapping",
                        lambda universe=None: {c: "A" for c in universe})
    with _capture_warnings() as cap:
        src = _resolve_live_industry_source(["000101", "000102"])
    assert src == {"000101": "A", "000102": "A"}
    assert any("falling back" in m and "get_industry_mapping" in m for m in cap.messages), \
        cap.messages


# ---------------------------------------------------------------------------
# 4. build_latest_features end-to-end (pool + worker stubbed): peer codes are
#    built, ranked, and dropped; output == training values for `codes`
# ---------------------------------------------------------------------------

class _SyncPool:
    """Stand-in for ProcessPoolExecutor that runs map() in-process."""

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, it, chunksize=1):
        return map(fn, it)


@pytest.fixture
def stubbed_live(monkeypatch, panel, ind_hist, dates):
    """Stub everything in build_latest_features that touches DB/network."""
    import concurrent.futures
    import mp.ml.dataset as ds
    from mp.data import fetcher

    day = dates[45]
    rows_by_code = {r["code"]: r for _, r in _latest_rows(panel, day).iterrows()}

    def _fake_worker(args):
        code = args[0]
        r = rows_by_code[code]
        row = {"date": day, "code": code}
        for c in TECHNICAL_COLUMNS:
            row[c] = 0.0
        row.update({"pe_ttm": r["pe_ttm"], "pb": r["pb"], "roe": r["roe"],
                    "mom_20d": r["mom_20d"], "total_mv_log": 10.0,
                    "revenue_growth": 0.1, "profit_growth": 0.1})
        return code, pd.DataFrame([row])

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _SyncPool)
    monkeypatch.setattr(ds, "_build_latest_worker", _fake_worker)
    monkeypatch.setattr(ds, "_fetch_valuation_snapshot_map", lambda codes: {})
    monkeypatch.setattr(ds, "_fetch_financial_history", lambda code: None)
    monkeypatch.setattr(fetcher, "get_industry_history", lambda universe=None: ind_hist)
    monkeypatch.setattr(fetcher, "get_industry_mapping",
                        lambda universe=None: pytest.fail("snapshot must not be consulted"))
    return day


def test_build_latest_features_ranks_over_universe_and_slices_back(stubbed_live, train_ranked):
    from mp.ml.dataset import build_latest_features

    day = stubbed_live
    seen: list[tuple[int, int]] = []
    with _capture_warnings() as cap:
        out = build_latest_features(SUBSET, include_fundamentals=True,
                                    rank_universe=ALL_CODES,
                                    progress_callback=lambda i, n: seen.append((i, n)))
    assert not any("rank_universe" in m for m in cap.messages), cap.messages

    # All 12 codes were built (peers included) but only SUBSET is returned.
    assert seen[-1] == (len(ALL_CODES), len(ALL_CODES))
    assert list(out["code"]) == SUBSET
    assert out.attrs.get("_data_quality") == pytest.approx(1.0)
    assert "_data_warnings" in out.columns

    expected = _rank_frame(train_ranked[(train_ranked["date"] == day)
                                        & train_ranked["code"].isin(SUBSET)])
    np.testing.assert_allclose(_rank_frame(out).to_numpy(), expected.to_numpy(),
                               atol=TOL, rtol=0)


def test_build_latest_features_without_rank_universe_warns(stubbed_live, train_ranked):
    from mp.ml.dataset import build_latest_features

    day = stubbed_live
    with _capture_warnings() as cap:
        out = build_latest_features(SUBSET, include_fundamentals=True)
    assert any("rank_universe" in m for m in cap.messages), cap.messages
    assert list(out["code"]) == SUBSET

    expected = _rank_frame(train_ranked[(train_ranked["date"] == day)
                                        & train_ranked["code"].isin(SUBSET)])
    assert not np.allclose(_rank_frame(out).to_numpy(), expected.to_numpy(), atol=TOL, rtol=0)
