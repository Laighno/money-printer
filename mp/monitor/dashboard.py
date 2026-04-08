"""Streamlit dashboard for strategy monitoring."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_nav(result: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> go.Figure:
    """Plot net asset value curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["date"], y=result["nav"], name="Strategy", line=dict(color="#2196F3", width=2)))
    if benchmark is not None and not benchmark.empty:
        fig.add_trace(go.Scatter(x=benchmark["date"], y=benchmark["nav"], name="Benchmark", line=dict(color="#9E9E9E", width=1.5, dash="dash")))
    fig.update_layout(title="Net Asset Value", xaxis_title="Date", yaxis_title="NAV", template="plotly_white", height=450)
    return fig


def plot_drawdown(result: pd.DataFrame) -> go.Figure:
    """Plot drawdown chart."""
    nav = result["nav"]
    peak = nav.cummax()
    drawdown = (nav - peak) / peak

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["date"], y=drawdown, fill="tozeroy", name="Drawdown", line=dict(color="#F44336", width=1)))
    fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown", template="plotly_white", height=300)
    return fig


def plot_monthly_returns(result: pd.DataFrame) -> go.Figure:
    """Plot monthly returns heatmap."""
    df = result.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly = df.groupby(["year", "month"])["daily_return"].apply(lambda x: (1 + x).prod() - 1).reset_index()
    monthly.columns = ["year", "month", "return"]

    pivot = monthly.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="RdYlGn",
        text=[[f"{v:.1%}" if pd.notna(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig.update_layout(title="Monthly Returns", template="plotly_white", height=300)
    return fig


def render_performance_table(metrics: dict) -> str:
    """Render performance metrics as markdown table."""
    labels = {
        "total_return": "Total Return",
        "annual_return": "Annual Return",
        "annual_volatility": "Annual Volatility",
        "sharpe_ratio": "Sharpe Ratio",
        "max_drawdown": "Max Drawdown",
        "win_rate": "Win Rate",
        "trading_days": "Trading Days",
    }
    lines = ["| Metric | Value |", "|--------|-------|"]
    for key, label in labels.items():
        if key in metrics:
            lines.append(f"| {label} | {metrics[key]} |")
    return "\n".join(lines)
