"""Data unit contract & normalization for daily bars and valuation rows.

This module is the **single source of truth** for "what units does our system
expect" and "how do we convert each data source into those units".

## Why this exists

We've been bitten twice by silent unit-mismatch bugs:

1. 2026-04-28: Eastmoney returns 成交量 in 手 (lots, ×100 shares) while Sina
   returns it in 股 (shares).  EM-fallback days had 100× wrong volume,
   silently corrupting volume-based factors (vwap_dev, obv_slope, mfi_14, …).

2. 2026-05-07: Eastmoney returns 换手率 in 百分数 (e.g. 7.14% as the value
   7.14) while Sina returns it in 小数 (0.0714).  EM-fallback days had 100×
   wrong turnover, causing 粤电力A's predict_raw to swing from +11.85% in
   the afternoon report to +2.94% later the same evening as more bars came in.

Both bugs share a structure: **a data source quietly differs from another
source in unit convention, and the fetcher path didn't normalize**.  The
fix-as-we-go approach won't scale — every new column or new source is a
trap.

## How this protects

Three layers, all enforced by this module:

1. **Declarative contract** (:class:`BarSchema`): every numeric column has a
   documented expected range and unit.  This is the spec.

2. **Source-aware normalization** (:func:`normalize_bars`): every data source
   declares its raw conventions, and is converted to canonical units in
   exactly one place.  **Adding a new source raises an error if it's not
   registered**, making it impossible to silently introduce a third unit
   mismatch.

3. **Validation** (:func:`validate_bars`): after normalization, the result
   must satisfy schema invariants.  Run on every save_bars_upsert call as
   a backstop.

If you find yourself writing ad-hoc unit conversion outside this file —
stop, register the source here, and call :func:`normalize_bars` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# =====================================================================
# Layer 1: Declarative contract
# =====================================================================


@dataclass(frozen=True)
class ColumnSpec:
    """The unit and range contract for one numeric column."""
    unit: str                            # canonical unit, e.g. "shares", "CNY", "decimal_fraction"
    hard_min: float                      # below this = physically impossible
    hard_max: float                      # above this = physically impossible
    expected_median_min: float           # if batch median below, suspect unit error
    expected_median_max: float           # if batch median above, suspect unit error
    description: str                     # human-readable explanation


class BarSchema:
    """The canonical contract for ``daily_bars`` rows.

    Adding a new column?  Register its ``ColumnSpec`` here.  Adding a new
    data source?  Register its conventions in :data:`SOURCE_CONVENTIONS`.
    """

    COLUMNS: Dict[str, ColumnSpec] = {
        "open": ColumnSpec(
            unit="CNY", hard_min=0.001, hard_max=1e5,
            expected_median_min=0.1, expected_median_max=1e4,
            description="opening price in 元",
        ),
        "high": ColumnSpec(
            unit="CNY", hard_min=0.001, hard_max=1e5,
            expected_median_min=0.1, expected_median_max=1e4,
            description="daily high in 元",
        ),
        "low": ColumnSpec(
            unit="CNY", hard_min=0.001, hard_max=1e5,
            expected_median_min=0.1, expected_median_max=1e4,
            description="daily low in 元",
        ),
        "close": ColumnSpec(
            unit="CNY", hard_min=0.001, hard_max=1e5,
            expected_median_min=0.1, expected_median_max=1e4,
            description="closing price in 元",
        ),
        "volume": ColumnSpec(
            # NOT 手 (lots).  EM returns 手, must ×100 to canonical.
            unit="shares", hard_min=0.0, hard_max=1e12,
            expected_median_min=1e3, expected_median_max=1e10,
            description="daily volume in 股 (NOT 手 — EM returns 手, must ×100)",
        ),
        "amount": ColumnSpec(
            unit="CNY", hard_min=0.0, hard_max=1e14,
            expected_median_min=1e4, expected_median_max=1e12,
            description="daily turnover value in 元 (NOT 万元 NOT 百万元)",
        ),
        "turnover": ColumnSpec(
            # NOT 百分数 (percent).  EM returns 7.14 for "7.14%", must /100.
            unit="decimal_fraction", hard_min=0.0, hard_max=1.0,
            expected_median_min=1e-4, expected_median_max=0.5,
            description="换手率 as decimal fraction (0.0714 = 7.14%)",
        ),
    }

    # Cross-column invariant: amount ≈ volume × close (within reasonable slack).
    # Bounds relaxed (2026-05-07) to (0.3, 50.0) because qfq-adjusted prices
    # cause amount/(volume*close) to grow up to ~5× legitimately (forward-
    # adjusted close is lower than the close that produced the historical
    # amount).  Anything > 50 is almost certainly a 100× volume unit bug.
    AMOUNT_VOLUME_RATIO_BOUNDS: Tuple[float, float] = (0.3, 50.0)


# =====================================================================
# Layer 2: Source-aware normalization
# =====================================================================
#
# Each data source must declare which columns deviate from canonical units.
# Calling normalize_bars() with an unregistered source raises ValueError —
# making it impossible to silently introduce a third unit mismatch.
#
# Convention values:
#   "shares" / "lots"               for volume
#   "decimal" / "percent"           for turnover
#   "CNY" / "万元" / "百万元"         for amount

SOURCE_CONVENTIONS: Dict[str, Dict[str, str]] = {
    "eastmoney": {
        "volume": "lots",       # 手 → ×100
        "turnover": "percent",  # 7.14 → /100
        "amount": "CNY",
    },
    "eastmoney_etf": {
        "volume": "lots",
        "turnover": "percent",
        "amount": "CNY",
    },
    "sina": {
        "volume": "shares",
        # Sina sometimes omits turnover entirely — handle nan-safely
        "turnover": "decimal",
        "amount": "CNY",
    },
    "intraday_sina": {
        # Real-time spot from Sina (volume already converted to shares in caller).
        # Turnover not provided.
        "volume": "shares",
        "turnover": None,
        "amount": "CNY",
    },
    "test_canonical": {
        # Used by tests — input already in canonical units, no conversion.
        "volume": "shares",
        "turnover": "decimal",
        "amount": "CNY",
    },
}


def normalize_bars(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Convert raw bars from ``source`` into the canonical schema.

    This is the **only** place unit conversion should happen.  Every
    ``_get_daily_bars_*`` fetcher should end with::

        return normalize_bars(df, source="eastmoney")

    If you're tempted to do ``df["volume"] * 100`` somewhere else, stop and
    add the source here instead.

    Raises
    ------
    ValueError
        If ``source`` is not registered, refusing to silently store rows
        with unknown unit conventions.
    """
    if source not in SOURCE_CONVENTIONS:
        raise ValueError(
            f"Unknown data source {source!r}.  Register its unit conventions "
            f"in mp.data.schema.SOURCE_CONVENTIONS before calling normalize_bars. "
            f"Known: {list(SOURCE_CONVENTIONS.keys())}"
        )
    conv = SOURCE_CONVENTIONS[source]
    df = df.copy()

    # Volume: lots → shares
    if "volume" in df.columns and conv.get("volume") == "lots":
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100.0

    # Turnover: percent → decimal
    if "turnover" in df.columns and conv.get("turnover") == "percent":
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce") / 100.0

    # Amount: 万元/百万元 → CNY (defensive — no current source uses this, but
    # registering "amount" as 万元 in SOURCE_CONVENTIONS would auto-multiply)
    if "amount" in df.columns:
        amt_unit = conv.get("amount", "CNY")
        if amt_unit == "万元":
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1e4
        elif amt_unit == "百万元":
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1e6
        # CNY: no-op

    return df


# =====================================================================
# Layer 3: Validation (run on every write)
# =====================================================================


def _check_hard_bounds(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """Drop rows that violate hard physical bounds.  Returns (clean_df, warnings)."""
    warnings: list[str] = []
    n_before = len(df)
    keep_mask = pd.Series(True, index=df.index)
    for col, spec in BarSchema.COLUMNS.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        # Allow NaN (some sources don't provide all columns)
        bad = v.notna() & ((v < spec.hard_min) | (v > spec.hard_max))
        n_bad_col = int(bad.sum())
        if n_bad_col > 0:
            keep_mask &= ~bad
            warnings.append(
                f"{col}: {n_bad_col} rows out of [{spec.hard_min}, {spec.hard_max}] "
                f"({spec.description})"
            )
    return df[keep_mask].reset_index(drop=True), warnings


def _check_unit_drift(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """Detect batch-level unit-mismatch heuristics; auto-fix and warn.

    For ``turnover`` specifically: if the batch median > 1.0 (clearly not
    decimal), divide by 100.  This is the back-stop that catches a future
    new source whose conventions weren't declared.
    """
    warnings: list[str] = []
    df = df.copy()

    # Turnover: any value > 1 is impossible in decimal convention
    if "turnover" in df.columns:
        t = pd.to_numeric(df["turnover"], errors="coerce")
        bad_mask = t > 1.0
        n_bad = int(bad_mask.sum())
        if n_bad > 0:
            df.loc[bad_mask, "turnover"] = t[bad_mask] / 100.0
            warnings.append(
                f"turnover: auto-normalized {n_bad} rows with value > 1.0 "
                f"(suspected 百分数 instead of 小数)"
            )
    return df, warnings


def _check_amount_consistency(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """Drop rows where amount ≠ volume × close (within VWAP slack).

    This is the strongest cross-column unit check: if amount/(volume*close)
    is far from 1.0, either volume or amount has wrong unit.
    """
    warnings: list[str] = []
    needed = {"volume", "amount", "close"}
    if not needed.issubset(df.columns):
        return df, warnings

    v = pd.to_numeric(df["volume"], errors="coerce")
    a = pd.to_numeric(df["amount"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")

    valid_inputs = v.notna() & a.notna() & c.notna() & (v > 0) & (c > 0)
    if not valid_inputs.any():
        return df, warnings

    ratio = pd.Series(np.nan, index=df.index)
    ratio[valid_inputs] = a[valid_inputs] / (v[valid_inputs] * c[valid_inputs])

    lo, hi = BarSchema.AMOUNT_VOLUME_RATIO_BOUNDS
    bad = valid_inputs & ((ratio < lo) | (ratio > hi))
    n_bad = int(bad.sum())
    if n_bad > 0:
        warnings.append(
            f"amount/(volume*close) outside [{lo}, {hi}] for {n_bad} rows "
            f"(volume or amount unit mismatch)"
        )
        df = df[~bad].reset_index(drop=True)
    return df, warnings


def validate_bars(df: pd.DataFrame, *, raise_on_error: bool = False) -> pd.DataFrame:
    """Apply all schema invariants; return clean rows.

    Operations:
    1. Auto-normalize obvious unit drift (turnover > 1 → /100)
    2. Drop rows violating hard physical bounds
    3. Drop rows violating cross-column consistency

    All actions are logged at warning level so silent regressions are visible.
    """
    if df is None or df.empty:
        return df

    all_warnings: list[str] = []

    df, w1 = _check_unit_drift(df)
    all_warnings.extend(w1)

    df, w2 = _check_hard_bounds(df)
    all_warnings.extend(w2)

    df, w3 = _check_amount_consistency(df)
    all_warnings.extend(w3)

    if all_warnings:
        for w in all_warnings:
            logger.warning("BarSchema validation: {}", w)
        if raise_on_error:
            raise ValueError(f"BarSchema validation failed: {'; '.join(all_warnings)}")

    return df


# =====================================================================
# Layer 4: Per-stock drift detector (catches "single bad row")
# =====================================================================


def detect_per_stock_drift(
    new_rows: pd.DataFrame,
    historical_median_lookup: Callable[[str, str], float | None],
    threshold: float = 50.0,
) -> pd.DataFrame:
    """Flag rows whose values jumped >threshold× their historical median.

    This catches the case where a single source returns one bad row (e.g.
    ETF on a single day with wrong unit) — hard bounds wouldn't catch a
    100× spike if the canonical value is normal-sized, but per-stock drift
    will.

    Parameters
    ----------
    new_rows
        Bars about to be written, must have columns code, date, volume,
        amount, turnover.
    historical_median_lookup
        Callable ``f(code, col) -> float | None`` returning the trailing
        median for that stock+column, or None if no history exists.
    threshold
        Multiplier above which to flag.  Default 50× is permissive enough
        for legitimate volume spikes but catches unit errors.
    """
    if new_rows is None or new_rows.empty:
        return new_rows

    flagged_indices = []
    for col in ["volume", "amount", "turnover"]:
        if col not in new_rows.columns:
            continue
        for idx, row in new_rows.iterrows():
            v = row.get(col)
            if pd.isna(v) or v <= 0:
                continue
            code = str(row.get("code", ""))
            if not code:
                continue
            historical = historical_median_lookup(code, col)
            if historical is None or historical <= 0:
                continue
            ratio = v / historical
            if ratio > threshold:
                flagged_indices.append((idx, code, col, v, historical, ratio))

    if flagged_indices:
        for idx, code, col, v, h, r in flagged_indices[:5]:  # log first 5
            logger.warning(
                "Per-stock drift: {} {}={:.4g} is {:.1f}× its 30d median {:.4g} "
                "— possible unit error",
                code, col, v, r, h,
            )
    return new_rows
