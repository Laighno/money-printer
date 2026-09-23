"""Regression tests for the 2026-05-21 low-liquidity guard.

Two thresholds:
- LOW_LIQUIDITY_FILTER_AMOUNT (¥1亿): drop from top-N entirely
- LOW_LIQUIDITY_WARN_AMOUNT  (¥3亿): keep, but flag _low_liquidity=True

Reason: low-liquidity stocks suffer the largest midday→EOD prediction
swings (early-session volume × scale 2 over-estimates the day's true
volume).  Verified: 神州细胞 −8.46pp, 中交设计 −2.14pp, both ¥<100M daily.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _MockBlendRanker:
    score_type = "rank_percentile"

    def predict(self, features):
        # All stocks get same blend score; rank determined by predict_raw
        return pd.Series([0.99] * len(features)).values

    def predict_raw(self, features):
        # Order codes by raw excess; first in list gets highest
        n = len(features)
        return pd.Series([(n - i) / 100.0 for i in range(n)]).values

    def feature_importance_report(self):
        return pd.DataFrame({"feature": ["mom_5d"], "importance": [1.0]})


def test_hard_filter_drops_below_1yi(monkeypatch):
    """Stocks with 20d avg amount < ¥1亿 are dropped from top-N entirely."""
    import scripts.daily_report as dr

    codes = ["000001", "000002", "000003", "000004", "000005"]
    feat = pd.DataFrame({"code": codes, "mom_5d": [0.0] * 5})
    feat.attrs["_data_quality"] = 1.0

    # 000003 is very illiquid; rest are fine
    monkeypatch.setattr(dr, "_recent_amount_avg",
                        lambda codes, days=20: {
                            "000001": 1.0e9, "000002": 5.0e8,
                            "000003": 5.0e7,   # ¥0.5亿 — should DROP
                            "000004": 4.0e8, "000005": 6.0e8,
                        })
    monkeypatch.setattr(dr, "get_stock_names", lambda c: {x: x for x in c})
    monkeypatch.setattr(dr, "evaluate_holdings_60d", lambda c: [])

    top_df, _, _, full_scored = dr.recommend_stocks(
        _MockBlendRanker(), n_recommend=5, precomputed_features=feat,
    )
    assert "000003" not in set(top_df["code"]), "Hard filter should drop low-liquidity"
    # Full scored still has it
    assert "000003" in set(full_scored["code"]), "full_scored keeps for ranking"


def test_soft_warning_flag_set(monkeypatch):
    """Stocks with ¥1亿 ≤ amount < ¥3亿 stay in top-N but flagged."""
    import scripts.daily_report as dr

    codes = ["000001", "000002", "000003", "000004", "000005"]
    feat = pd.DataFrame({"code": codes, "mom_5d": [0.0] * 5})
    feat.attrs["_data_quality"] = 1.0

    # 000003 is borderline (¥2亿 — above hard floor, below soft warn)
    monkeypatch.setattr(dr, "_recent_amount_avg",
                        lambda codes, days=20: {
                            "000001": 5.0e8, "000002": 4.0e8,
                            "000003": 2.0e8,   # WARN, not DROP
                            "000004": 4.0e8, "000005": 6.0e8,
                        })
    monkeypatch.setattr(dr, "get_stock_names", lambda c: {x: x for x in c})
    monkeypatch.setattr(dr, "evaluate_holdings_60d", lambda c: [])

    top_df, _, _, _ = dr.recommend_stocks(
        _MockBlendRanker(), n_recommend=5, precomputed_features=feat,
    )
    assert "000003" in set(top_df["code"]), "Borderline must NOT be dropped"
    flagged = top_df[top_df["_low_liquidity"]]["code"].tolist()
    assert "000003" in flagged, "Borderline must be flagged"
    # 000005 (¥6亿) above warn threshold → NOT flagged
    assert "000005" not in flagged


def test_no_drop_when_all_liquid(monkeypatch):
    """If all 5 are healthy liquidity, no filter, no flags."""
    import scripts.daily_report as dr

    codes = ["000001", "000002", "000003", "000004", "000005"]
    feat = pd.DataFrame({"code": codes, "mom_5d": [0.0] * 5})
    feat.attrs["_data_quality"] = 1.0

    monkeypatch.setattr(dr, "_recent_amount_avg",
                        lambda codes, days=20: {c: 1.0e9 for c in codes})
    monkeypatch.setattr(dr, "get_stock_names", lambda c: {x: x for x in c})
    monkeypatch.setattr(dr, "evaluate_holdings_60d", lambda c: [])

    top_df, _, _, _ = dr.recommend_stocks(
        _MockBlendRanker(), n_recommend=5, precomputed_features=feat,
    )
    assert len(top_df) == 5
    assert not top_df["_low_liquidity"].any()


def test_missing_amount_data_treated_as_illiquid(monkeypatch):
    """If DB returns no amount data for a code, treat it as illiquid (drop)."""
    import scripts.daily_report as dr

    codes = ["000001", "000002", "000003"]
    feat = pd.DataFrame({"code": codes, "mom_5d": [0.0] * 3})
    feat.attrs["_data_quality"] = 1.0

    # 000003 missing entirely
    monkeypatch.setattr(dr, "_recent_amount_avg",
                        lambda codes, days=20: {"000001": 5.0e8, "000002": 4.0e8})
    monkeypatch.setattr(dr, "get_stock_names", lambda c: {x: x for x in c})
    monkeypatch.setattr(dr, "evaluate_holdings_60d", lambda c: [])

    top_df, _, _, _ = dr.recommend_stocks(
        _MockBlendRanker(), n_recommend=5, precomputed_features=feat,
    )
    assert "000003" not in set(top_df["code"])
