"""Portfolio construction - select top stocks and assign weights."""

import pandas as pd
from loguru import logger


def select_top_n(score_df: pd.DataFrame, top_n: int = 30, max_position_pct: float = 0.1) -> pd.DataFrame:
    """Select top N stocks by score and assign weights.

    Args:
        score_df: Factor score DataFrame with 'final_score' column, indexed by code
        top_n: Number of stocks to hold
        max_position_pct: Maximum weight per stock

    Returns:
        DataFrame with columns: code, weight, final_score
    """
    # Drop stocks with NaN scores
    valid = score_df.dropna(subset=["final_score"])
    selected = valid.head(top_n).copy()

    if selected.empty:
        logger.warning("No valid stocks to select")
        return pd.DataFrame(columns=["code", "weight", "final_score"])

    # Equal weight, capped at max_position_pct
    n = len(selected)
    base_weight = 1.0 / n
    selected["weight"] = min(base_weight, max_position_pct)

    # Re-normalize so weights sum to 1
    selected["weight"] = selected["weight"] / selected["weight"].sum()

    result = selected[["final_score"]].copy()
    result["weight"] = selected["weight"]
    result.index.name = "code"
    result = result.reset_index()

    logger.info(f"Selected {len(result)} stocks, weight range: [{result['weight'].min():.4f}, {result['weight'].max():.4f}]")
    return result
