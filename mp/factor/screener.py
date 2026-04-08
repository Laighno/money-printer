"""Stock screener - fundamental factor scoring for individual stocks within a sector.

Two-level screening:
  Level 1: Rotation signals select target sectors (板块)
  Level 2: This module scores stocks within those sectors using fundamentals

Factors used:
  - PE_TTM (lower is better, but exclude negative)
  - PB (lower is better)
  - ROE (higher is better)
  - Revenue growth (higher is better)
  - Profit growth (higher is better)
  - Gross margin (higher is better)
  - Debt ratio (lower is better)
"""

import numpy as np
import pandas as pd
from loguru import logger


# Default weights for fundamental scoring
DEFAULT_WEIGHTS = {
    "pe_ttm": -0.20,        # lower PE = cheaper (negative weight)
    "pb": -0.10,            # lower PB = cheaper
    "roe": 0.25,            # higher ROE = more profitable
    "revenue_growth": 0.15, # higher growth = better
    "profit_growth": 0.15,  # higher growth = better
    "gross_margin": 0.10,   # higher margin = better moat
    "debt_ratio": -0.05,    # lower debt = safer
}


def score_stocks(constituents: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Score constituent stocks using fundamental factors from real-time data.

    Args:
        constituents: DataFrame from get_industry_constituents() with columns
                     code, name, close, change_pct, pe_ttm, pb, turnover, ...
        weights: factor weight overrides (default: DEFAULT_WEIGHTS)

    Returns:
        DataFrame with original columns + z-scores + fundamental_score, sorted desc.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    df = constituents.copy()

    # Filter: must have valid price
    df = df[df["close"] > 0].copy()

    # Available fundamental columns from constituents (pe_ttm, pb come from EM spot data)
    available = {}
    for col in ["pe_ttm", "pb"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            # For PE/PB: filter out negative (loss-making) and extreme values
            if col == "pe_ttm":
                series = series.where((series > 0) & (series < 500))
            elif col == "pb":
                series = series.where((series > 0) & (series < 50))
            available[col] = series

    # Compute z-scores and weighted sum
    for col, weight in weights.items():
        if col not in available:
            continue
        series = available[col]
        mean = series.mean()
        std = series.std()
        if std > 0:
            df[f"_z_{col}"] = (series - mean) / std * weight
        else:
            df[f"_z_{col}"] = 0.0

    z_cols = [c for c in df.columns if c.startswith("_z_")]
    if z_cols:
        df["fundamental_score"] = df[z_cols].sum(axis=1)
    else:
        df["fundamental_score"] = 0.0

    df = df.drop(columns=z_cols)
    df = df.sort_values("fundamental_score", ascending=False)
    return df


def score_stocks_with_financials(
    constituents: pd.DataFrame,
    financial_data: pd.DataFrame | None = None,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Enhanced scoring that also uses financial statement data (ROE, growth, margins).

    Args:
        constituents: from get_industry_constituents()
        financial_data: from get_financial_data_batch() with columns
                       code, roe, revenue_growth, profit_growth, gross_margin,
                       net_margin, debt_ratio
        weights: factor weight overrides
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    df = constituents.copy()
    df = df[df["close"] > 0].copy()

    # Merge financial data if available
    if financial_data is not None and not financial_data.empty:
        fin = financial_data.copy()
        # Keep latest report per stock
        if "report_date" in fin.columns:
            fin = fin.sort_values("report_date").groupby("code").tail(1)
        fin_cols = ["code"] + [c for c in ["roe", "revenue_growth", "profit_growth",
                                            "gross_margin", "net_margin", "debt_ratio"]
                               if c in fin.columns]
        fin = fin[fin_cols]
        df = df.merge(fin, on="code", how="left")

    # Collect all scoreable columns
    available = {}
    for col in weights:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if col == "pe_ttm":
                series = series.where((series > 0) & (series < 500))
            elif col == "pb":
                series = series.where((series > 0) & (series < 50))
            elif col == "debt_ratio":
                series = series.where((series > 0) & (series < 100))
            available[col] = series

    # Z-score weighted sum
    for col, weight in weights.items():
        if col not in available:
            continue
        series = available[col]
        mean = series.mean()
        std = series.std()
        if std > 0:
            df[f"_z_{col}"] = (series - mean) / std * weight
        else:
            df[f"_z_{col}"] = 0.0

    z_cols = [c for c in df.columns if c.startswith("_z_")]
    if z_cols:
        df["fundamental_score"] = df[z_cols].sum(axis=1)
    else:
        df["fundamental_score"] = 0.0

    df = df.drop(columns=z_cols)
    df = df.sort_values("fundamental_score", ascending=False)

    n_scored = df["fundamental_score"].notna().sum()
    factors_used = [c for c in weights if c in available]
    logger.info(f"Scored {n_scored} stocks using factors: {factors_used}")
    return df
