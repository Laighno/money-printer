"""Factor scoring engine - calculate, normalize and combine factors."""

import pandas as pd
from loguru import logger

from ..config import FactorConfig
from .registry import get_factor_func

# Ensure built-in factors are registered
from . import builtin as _  # noqa: F401


def calculate_factors(
    bars: pd.DataFrame,
    factor_configs: list[FactorConfig],
    valuation: pd.DataFrame | None = None,
    financial: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate all configured factors and return a score DataFrame.

    Args:
        bars: Daily bar data for all stocks
        factor_configs: List of factor configurations
        valuation: Real-time PE/PB/MV data (from stock_zh_a_spot_em)
        financial: Financial statement data (ROE, EPS, etc.)

    Returns:
        DataFrame indexed by code with columns: factor values, z-scores, and final_score
    """
    if valuation is None:
        valuation = pd.DataFrame()
    if financial is None:
        financial = pd.DataFrame()

    results = {}
    for fc in factor_configs:
        func = get_factor_func(fc.name)
        raw = func(bars, valuation, financial)
        results[fc.name] = raw
        logger.info(f"Calculated factor '{fc.name}': {raw.notna().sum()} valid values")

    score_df = pd.DataFrame(results)

    # Z-score normalization (cross-sectional)
    for fc in factor_configs:
        col = fc.name
        if col not in score_df.columns:
            continue
        zscore = (score_df[col] - score_df[col].mean()) / (score_df[col].std() + 1e-8)
        # Flip sign if ascending (lower is better)
        if fc.direction == "asc":
            zscore = -zscore
        score_df[f"{col}_zscore"] = zscore

    # Weighted composite score
    zscore_cols = [f"{fc.name}_zscore" for fc in factor_configs if f"{fc.name}_zscore" in score_df.columns]
    if zscore_cols:
        score_df["final_score"] = sum(
            score_df[f"{fc.name}_zscore"] * fc.weight
            for fc in factor_configs
            if f"{fc.name}_zscore" in score_df.columns
        )
    else:
        score_df["final_score"] = 0.0

    score_df = score_df.sort_values("final_score", ascending=False)
    return score_df
