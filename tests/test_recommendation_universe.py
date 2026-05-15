"""Regression tests for the 2026-05-14 universe widening.

Recommendation universe was widened from ZZ500-only to HS300 + ZZ500 (~800
unique stocks) so reports cover both large-caps (e.g. 卓胜微 / 招商银行
in HS300) and mid-caps (粤电力A / 启明星辰 in ZZ500).  Previously stocks
outside ZZ500 never appeared in recommendations even when the model would
have ranked them highly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_recommendation_universe_merges_hs300_and_zz500(monkeypatch):
    """Default merge: HS300 + ZZ500, deduplicated."""
    from mp.data import fetcher

    def _fake(idx, **kw):
        if idx == "hs300":
            return [f"30{i:04d}" for i in range(300)]
        if idx == "zz500":
            return [f"50{i:04d}" for i in range(500)]
        raise ValueError(idx)

    monkeypatch.setattr(fetcher, "get_index_constituents", _fake)
    uni = fetcher.get_recommendation_universe()
    assert len(uni) == 800
    # Sorted
    assert uni == sorted(uni)


def test_recommendation_universe_dedupes_overlap(monkeypatch):
    """If two indices have overlapping codes, the result is deduplicated."""
    from mp.data import fetcher

    def _fake(idx, **kw):
        if idx == "hs300":
            return ["000001", "000002", "000003"]
        if idx == "zz500":
            return ["000002", "000003", "000004", "000005"]
        raise ValueError(idx)

    monkeypatch.setattr(fetcher, "get_index_constituents", _fake)
    uni = fetcher.get_recommendation_universe()
    assert uni == ["000001", "000002", "000003", "000004", "000005"]


def test_recommendation_universe_tolerates_partial_failure(monkeypatch):
    """If one index fetch fails, return whatever the others gave (graceful)."""
    from mp.data import fetcher

    def _fake(idx, **kw):
        if idx == "hs300":
            raise RuntimeError("hs300 unreachable")
        if idx == "zz500":
            return ["000001", "000002"]
        raise ValueError(idx)

    monkeypatch.setattr(fetcher, "get_index_constituents", _fake)
    uni = fetcher.get_recommendation_universe()
    assert uni == ["000001", "000002"]


def test_recommendation_universe_custom_indices(monkeypatch):
    """Callers can request a different mix (e.g. HS300-only or +ZZ1000)."""
    from mp.data import fetcher

    def _fake(idx, **kw):
        return {
            "hs300": ["A", "B"],
            "zz500": ["C", "D"],
            "zz1000": ["E", "F"],
        }[idx]

    monkeypatch.setattr(fetcher, "get_index_constituents", _fake)
    assert fetcher.get_recommendation_universe(("hs300",)) == ["A", "B"]
    assert fetcher.get_recommendation_universe(("zz500", "zz1000")) == ["C", "D", "E", "F"]
