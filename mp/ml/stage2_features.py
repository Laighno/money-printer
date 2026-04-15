"""Cross-sectional features for Stage-2 fine ranking.

Stage-1 uses per-stock time-series factors (momentum, volatility, etc.).
Stage-2 adds *relative-to-peers* features so the model can distinguish
"good" from "better" within the already-filtered top candidates.

Usage:
    from mp.ml.stage2_features import compute_stage2_features, STAGE2_COLUMNS
    df = compute_stage2_features(df)  # adds STAGE2_COLUMNS in-place
"""

from __future__ import annotations

from typing import List

import pandas as pd

# Existing columns from which cross-sectional ranks are derived.
_RANK_SOURCES = [
    "mom_5d",           # short-term momentum
    "mom_10d",          # short-term momentum (slightly longer)
    "turnover_5d",      # recent trading activity
    "amount_ratio",     # volume surge vs average
    "vol_price_corr",   # volume-price coordination
]

# Stage-2 feature columns (cross-sectional percentile ranks).
STAGE2_COLUMNS: List[str] = [
    "mom_5d_rank",
    "mom_10d_rank",
    "turnover_rank",
    "amount_ratio_rank",
    "vol_price_corr_rank",
    "stage1_score_rank",
]


def compute_stage2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional rank features to *df*.

    If a ``date`` column exists, ranks are computed within each date
    (true cross-sectional). Otherwise ranks are computed over the full
    DataFrame (useful for single-day prediction).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns listed in ``_RANK_SOURCES``.
        Optionally contains ``stage1_score`` (Stage-1 prediction).

    Returns
    -------
    pd.DataFrame with additional columns from ``STAGE2_COLUMNS``.
    """
    out = df.copy()

    has_date = "date" in out.columns

    def _rank_col(series: pd.Series) -> pd.Series:
        return series.rank(pct=True)

    # Cross-sectional ranks for existing factors
    source_to_target = {
        "mom_5d": "mom_5d_rank",
        "mom_10d": "mom_10d_rank",
        "turnover_5d": "turnover_rank",
        "amount_ratio": "amount_ratio_rank",
        "vol_price_corr": "vol_price_corr_rank",
    }

    for src, tgt in source_to_target.items():
        if src not in out.columns:
            out[tgt] = 0.5  # neutral if source missing
        elif has_date:
            out[tgt] = out.groupby("date")[src].transform(_rank_col)
        else:
            out[tgt] = _rank_col(out[src])

    # Stage-1 score rank
    if "stage1_score" in out.columns:
        if has_date:
            out["stage1_score_rank"] = out.groupby("date")["stage1_score"].transform(_rank_col)
        else:
            out["stage1_score_rank"] = _rank_col(out["stage1_score"])
    elif "stage1_score_rank" not in out.columns:
        out["stage1_score_rank"] = 0.5  # neutral placeholder

    return out
