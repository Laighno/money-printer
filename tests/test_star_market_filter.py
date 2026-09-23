"""Regression tests for the 2026-05-19 科创板 (STAR market) filter.

Only daily_report's top-N recommendations filter out STAR codes
(688xxx / 689xxx).  paper_trade and backtests keep the full universe
unchanged — the filter is purely a display/recommendation preference,
not a strategy change.

Full universe scoring (full_scored) still contains STAR codes for
watchlist tracking and rank diagnostics — only the *displayed picks*
in daily/midday reports skip STAR.

Background: model learned a "buy the dip on unprofitable biotech"
pattern during walk-forward training that produced picks like 神州细胞
(ROE -434%) with excess +8% predictions.  STAR has 20% daily limits
and thin liquidity, making this style riskier in practice than the
model prices in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────────
# daily_report.recommend_stocks: STAR excluded from top-N
# ──────────────────────────────────────────────────────────────────

class _MockBlendRanker:
    """Minimal stand-in for BlendRanker matching the rank_percentile path."""
    score_type = "rank_percentile"

    def predict(self, features):
        # Higher ml_score for STAR so they'd dominate if not filtered
        return pd.Series([0.99 if str(c).startswith("688") else 0.50
                          for c in features["code"]]).values

    def predict_raw(self, features):
        return pd.Series([0.08 if str(c).startswith("688") else 0.02
                          for c in features["code"]]).values

    def feature_importance_report(self):
        return pd.DataFrame({"feature": ["mom_5d"], "importance": [1.0]})


def test_recommend_stocks_filters_star_market(monkeypatch):
    """Top-N must NOT contain 科创板; full_scored MUST still contain them."""
    import scripts.daily_report as dr

    # Build a fake feature panel: 3 STAR + 5 non-STAR
    fake_codes = ["688520", "688538", "688709", "000539", "002439",
                  "002385", "600673", "002131"]
    fake_features = pd.DataFrame({"code": fake_codes, "mom_5d": [0.0] * 8})
    fake_features.attrs["_data_quality"] = 1.0

    monkeypatch.setattr(dr, "get_stock_names",
                        lambda codes: {c: f"NAME_{c}" for c in codes})
    monkeypatch.setattr(dr, "evaluate_holdings_60d", lambda codes: [])
    monkeypatch.setattr(dr, "_recent_amount_avg",
                        lambda codes, days=20: {c: 5.0e8 for c in codes})

    top_df, _rec60d, _name_map, full_scored = dr.recommend_stocks(
        _MockBlendRanker(),
        n_recommend=5,
        precomputed_features=fake_features,
    )

    top_codes = set(top_df["code"].tolist())
    star_in_top = {c for c in top_codes if c.startswith(("688", "689"))}
    assert not star_in_top, f"Top-N must not contain STAR codes, got {star_in_top}"

    full_codes = set(full_scored["code"].tolist())
    # Every input code must still appear in full_scored
    assert set(fake_codes) == full_codes, (
        "full_scored must retain STAR codes for watchlist tracking; "
        f"missing {set(fake_codes) - full_codes}"
    )
    # And rank column intact
    assert "_rank" in full_scored.columns
    assert full_scored["_rank"].min() == 1


def test_recommend_stocks_top_n_respects_filter_count(monkeypatch):
    """If we ask for 5 picks and only 3 non-STAR exist, get 3 (no padding)."""
    import scripts.daily_report as dr

    # 5 STAR (would dominate if scored) + 3 non-STAR
    fake_codes = ["688001", "688002", "688003", "688004", "688005",
                  "000001", "000002", "000003"]
    fake_features = pd.DataFrame({"code": fake_codes, "mom_5d": [0.0] * 8})
    fake_features.attrs["_data_quality"] = 1.0

    monkeypatch.setattr(dr, "get_stock_names",
                        lambda codes: {c: f"NAME_{c}" for c in codes})
    monkeypatch.setattr(dr, "evaluate_holdings_60d", lambda codes: [])
    # Skip liquidity filter — return generous amounts for all codes
    monkeypatch.setattr(dr, "_recent_amount_avg",
                        lambda codes, days=20: {c: 5.0e8 for c in codes})

    top_df, _, _, _ = dr.recommend_stocks(
        _MockBlendRanker(), n_recommend=5, precomputed_features=fake_features,
    )
    assert len(top_df) == 3
    assert all(not c.startswith(("688", "689")) for c in top_df["code"])


# Note: paper_trade and backtests intentionally do NOT filter STAR —
# the filter is display-only.  No tests here for those code paths.
