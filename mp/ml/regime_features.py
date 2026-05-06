"""Market-regime proxy features (panel-level).

Builds a per-date DataFrame of regime-indicative signals that can be joined
onto a cross-sectional training panel. The feature value is the SAME for all
stocks on a given date; LightGBM can still learn interactions between regime
and stock-level factors via tree splits.

All features are strictly PIT: computed from prices ≤ date *t* and used as
inputs at *t*'s close (never peeks at t+1).

Columns produced:
    regime_mom_20d    : ZZ500 20-day log return
    regime_mom_60d    : ZZ500 60-day log return
    regime_vol_20d    : ZZ500 realized volatility (20d stdev of log returns)
    regime_dd_60d     : distance from 60d high  (0 = at high, negative = below)
    regime_breadth_20d: cross-sectional share of stocks with positive 20d ret

The first four come from a single benchmark series (ZZ500).  Breadth is
derived from whatever panel is passed in — it's stock-universe-specific and
needs the training panel to compute.

Cached on disk at ``data/wf_cache/regime_features.parquet`` (index features
only; breadth depends on the panel and is recomputed per call).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

_CACHE_PATH = Path("data/wf_cache/regime_features.parquet")

REGIME_INDEX_COLUMNS = [
    "regime_mom_20d",
    "regime_mom_60d",
    "regime_vol_20d",
    "regime_dd_60d",
]
REGIME_BREADTH_COLUMNS = ["regime_breadth_20d"]
REGIME_COLUMNS = REGIME_INDEX_COLUMNS + REGIME_BREADTH_COLUMNS


def _fetch_zz500_close() -> pd.Series:
    """Load full ZZ500 daily close series, date-indexed."""
    import akshare as ak

    idx = ak.stock_zh_index_daily(symbol="sh000905")
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").drop_duplicates("date")
    return idx.set_index("date")["close"].astype(float)


def _compute_index_regime(close: pd.Series) -> pd.DataFrame:
    """Compute index-only regime features. All quantities are PIT (lag-safe)."""
    log_px = np.log(close)
    ret = log_px.diff()

    mom_20 = log_px - log_px.shift(20)
    mom_60 = log_px - log_px.shift(60)
    vol_20 = ret.rolling(20).std() * np.sqrt(252)
    roll_max_60 = close.rolling(60).max()
    dd_60 = close / roll_max_60 - 1.0  # 0 = at high, -0.15 = 15% off high

    out = pd.DataFrame({
        "date": close.index,
        "regime_mom_20d": mom_20.values,
        "regime_mom_60d": mom_60.values,
        "regime_vol_20d": vol_20.values,
        "regime_dd_60d": dd_60.values,
    }).reset_index(drop=True)
    return out


def build_index_regime_features(refresh: bool = False) -> pd.DataFrame:
    """Return per-date index regime features, cached on disk.

    Parameters
    ----------
    refresh : bool
        If True, fetch fresh index data and overwrite cache.
    """
    if not refresh and _CACHE_PATH.exists():
        try:
            df = pd.read_parquet(_CACHE_PATH)
            df["date"] = pd.to_datetime(df["date"])
            logger.debug("regime cache hit: {} rows", len(df))
            return df
        except Exception as e:
            logger.warning("regime cache read failed ({}), recomputing", e)

    close = _fetch_zz500_close()
    df = _compute_index_regime(close)
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_CACHE_PATH)
        logger.info("regime cache saved: {} ({} rows)", _CACHE_PATH, len(df))
    except Exception as e:
        logger.warning("regime cache save failed: {}", e)
    return df


def compute_breadth(panel: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Compute per-date breadth = share of stocks with positive N-day return.

    Requires panel with columns ``date``, ``code``, and ``mom_20d`` (or the
    N-day momentum column matching ``period``).  Returns DataFrame with
    ``date`` and ``regime_breadth_{period}d``.
    """
    col = f"mom_{period}d"
    if col not in panel.columns:
        logger.warning("compute_breadth: {} not in panel, skipping", col)
        return pd.DataFrame(columns=["date", f"regime_breadth_{period}d"])
    mask = panel[[col, "date"]].dropna()
    mask["pos"] = (mask[col] > 0).astype(float)
    out = mask.groupby("date")["pos"].mean().reset_index()
    out.columns = ["date", f"regime_breadth_{period}d"]
    return out


def add_regime_features(
    panel: pd.DataFrame,
    refresh_index: bool = False,
    include_breadth: bool = True,
) -> pd.DataFrame:
    """Join regime features onto a training panel (in-place merge).

    Missing values left as NaN; LightGBM handles NaN natively.
    """
    if "date" not in panel.columns:
        raise ValueError("panel must have 'date' column")

    idx = build_index_regime_features(refresh=refresh_index)
    panel = panel.merge(idx, on="date", how="left")

    if include_breadth:
        breadth = compute_breadth(panel, period=20)
        if not breadth.empty:
            panel = panel.merge(breadth, on="date", how="left")
        else:
            panel["regime_breadth_20d"] = np.nan

    # Report coverage
    for col in REGIME_COLUMNS:
        if col in panel.columns:
            n_ok = panel[col].notna().sum()
            logger.debug("regime {}: {}/{} rows non-null", col, n_ok, len(panel))

    return panel


__all__ = [
    "REGIME_COLUMNS",
    "REGIME_INDEX_COLUMNS",
    "REGIME_BREADTH_COLUMNS",
    "build_index_regime_features",
    "compute_breadth",
    "add_regime_features",
]
