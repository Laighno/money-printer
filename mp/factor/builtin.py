"""Built-in factor implementations using real financial data."""

import numpy as np
import pandas as pd

from .registry import register


# === Price/Technical factors (from bars) ===

@register("momentum_20d")
def momentum_20d(bars: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
    """20-day momentum: (close_today / close_20d_ago) - 1."""
    latest = bars.groupby("code").tail(1).set_index("code")["close"]
    past = bars.groupby("code").nth(-21).set_index("code")["close"]
    return (latest / past - 1).rename("momentum_20d")


@register("momentum_60d")
def momentum_60d(bars: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
    """60-day momentum."""
    latest = bars.groupby("code").tail(1).set_index("code")["close"]
    past = bars.groupby("code").nth(-61).set_index("code")["close"]
    return (latest / past - 1).rename("momentum_60d")


@register("volatility_20d")
def volatility_20d(bars: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
    """20-day realized volatility (annualized)."""
    def calc(group):
        if len(group) < 21:
            return np.nan
        returns = group["close"].pct_change().dropna().tail(20)
        return returns.std() * np.sqrt(252)

    return bars.groupby("code").apply(calc, include_groups=False).rename("volatility_20d")


@register("turnover_20d")
def turnover_20d(bars: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
    """Average daily turnover (amount) over last 20 days."""
    def calc(group):
        return group["amount"].tail(20).mean()

    return bars.groupby("code").apply(calc, include_groups=False).rename("turnover_20d")


# === Valuation factors (from real-time PE/PB data) ===

@register("pe_ttm")
def pe_ttm_factor(bars: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
    """PE TTM (dynamic) from eastmoney real-time data."""
    if valuation.empty or "pe_ttm" not in valuation.columns:
        return pd.Series(dtype=float, name="pe_ttm")
    df = valuation[["code", "pe_ttm"]].dropna(subset=["pe_ttm"]).copy()
    # Filter out negative PE (loss-making companies)
    df = df[df["pe_ttm"] > 0]
    return df.set_index("code")["pe_ttm"]


@register("pb")
def pb_factor(bars: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
    """PB (Price-to-Book) from eastmoney real-time data."""
    if valuation.empty or "pb" not in valuation.columns:
        return pd.Series(dtype=float, name="pb")
    df = valuation[["code", "pb"]].dropna(subset=["pb"]).copy()
    df = df[df["pb"] > 0]
    return df.set_index("code")["pb"]


@register("market_cap")
def market_cap_factor(bars: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
    """Total market capitalization (log scale)."""
    if valuation.empty or "total_mv" not in valuation.columns:
        return pd.Series(dtype=float, name="market_cap")
    df = valuation[["code", "total_mv"]].dropna(subset=["total_mv"]).copy()
    df = df[df["total_mv"] > 0]
    df["market_cap"] = np.log(df["total_mv"])
    return df.set_index("code")["market_cap"]


# === Financial factors (from financial statements via EM) ===

@register("roe")
def roe_factor(bars: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
    """ROE (Return on Equity) from latest financial report."""
    if financial.empty or "roe" not in financial.columns:
        return pd.Series(dtype=float, name="roe")
    df = financial[["code", "roe"]].dropna(subset=["roe"]).copy()
    return df.set_index("code")["roe"]


@register("gross_margin")
def gross_margin_factor(bars: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
    """Gross profit margin from latest financial report."""
    if financial.empty or "gross_margin" not in financial.columns:
        return pd.Series(dtype=float, name="gross_margin")
    df = financial[["code", "gross_margin"]].dropna(subset=["gross_margin"]).copy()
    return df.set_index("code")["gross_margin"]


@register("net_margin")
def net_margin_factor(bars: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
    """Net profit margin from latest financial report."""
    if financial.empty or "net_margin" not in financial.columns:
        return pd.Series(dtype=float, name="net_margin")
    df = financial[["code", "net_margin"]].dropna(subset=["net_margin"]).copy()
    return df.set_index("code")["net_margin"]


@register("debt_ratio")
def debt_ratio_factor(bars: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
    """Asset-liability ratio from latest financial report."""
    if financial.empty or "debt_ratio" not in financial.columns:
        return pd.Series(dtype=float, name="debt_ratio")
    df = financial[["code", "debt_ratio"]].dropna(subset=["debt_ratio"]).copy()
    return df.set_index("code")["debt_ratio"]


@register("revenue_growth")
def revenue_growth_factor(bars: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
    """Revenue YoY growth rate."""
    if financial.empty or "revenue_growth" not in financial.columns:
        return pd.Series(dtype=float, name="revenue_growth")
    df = financial[["code", "revenue_growth"]].dropna(subset=["revenue_growth"]).copy()
    return df.set_index("code")["revenue_growth"]


@register("profit_growth")
def profit_growth_factor(bars: pd.DataFrame, valuation: pd.DataFrame, financial: pd.DataFrame) -> pd.Series:
    """Net profit YoY growth rate."""
    if financial.empty or "profit_growth" not in financial.columns:
        return pd.Series(dtype=float, name="profit_growth")
    df = financial[["code", "profit_growth"]].dropna(subset=["profit_growth"]).copy()
    return df.set_index("code")["profit_growth"]
