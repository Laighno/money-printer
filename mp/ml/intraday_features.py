"""Intraday (T 14:30 snapshot) feature pipeline for the P11 entry path.

P11 chain (see docs/dialog/ round 73 P11-START + docs/decision_log P11 entry):
re-predict the universe at T 14:30 using intraday-aware features and execute
via 14:55-15:00 集合竞价收盘 撮合, capturing additional alpha by reducing
20d→19d prediction noise AND removing the 1% limit-buffer cost.

This module is the **P11-1 foundation layer**: it defines the canonical
feature contract for the 14:30 entry path and provides pure feature math.
It does NOT yet train a model, run walk-forward, or fetch real data — those
are P11-2, P11-3, P11-4 respectively.

Feature contract
----------------
``INTRADAY_FEATURE_COLS = FACTOR_COLUMNS + INTRADAY_EXTRA_COLUMNS``

- ``FACTOR_COLUMNS`` (64): the existing technical/fundamental/industry-rank
  features defined in :mod:`mp.ml.dataset`, computed on a synthetic EOD
  panel where today's T 14:30 OHLCV is appended as a synthetic bar via
  the existing :func:`mp.ml.dataset._process_single_stock` injection hook
  (this hook was built for the midday-report path and is battle-tested).
- ``INTRADAY_EXTRA_COLUMNS`` (3): morning-session-specific features that
  do not exist in the EOD-only schema. Names are stable — model artifacts
  trained on this schema (``data/intraday_blend_*.lgb``, P11-2) will refer
  to these columns by name.

Why three and not zero (or twenty)
----------------------------------
The advisor (round 73) asked for "1-3 intraday-specific" features and to
"keep schema small". Three captures the most information-dense slices of
the morning session without overfitting risk:

  1. ``morning_return`` — directional signal (was the stock up or down by
     14:30?). Cleanest possible intraday momentum feature.
  2. ``morning_vwap_dev`` — close-vs-VWAP gap, a classic intraday strength
     signal (close > VWAP = late-session buying; close < VWAP = distribution).
  3. ``morning_vol_ratio`` — today-morning volume vs 20d EOD daily volume.
     Captures unusual-attention spikes. Imperfect (true comparison would
     be 20d morning-session totals) but informative; P11-4 will replace
     the EOD-daily denominator with historical morning-session totals once
     intraday history is reliably available.

Schema decisions documented here for P11-2 review:

- We DO NOT remove any EOD features. The 64-feature core is preserved so
  P11-2 retraining starts from a known-good baseline.
- We DO NOT modify ``FACTOR_COLUMNS`` itself; this module's constant is an
  *append*. Existing 9:30-entry production code is untouched.
- We DO NOT touch ``data/blend_*.lgb``. P11-2 will write
  ``data/intraday_blend_*.lgb`` as a parallel artifact (Rule #4).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from mp.ml.dataset import FACTOR_COLUMNS, build_latest_features

# 3 intraday-specific features. Names are part of the model contract once
# P11-2 trains against this schema — do not rename without retraining.
INTRADAY_EXTRA_COLUMNS: List[str] = [
    "morning_return",     # (T 14:30 close - T open) / T open
    "morning_vwap_dev",   # (T 14:30 close - morning_VWAP) / morning_VWAP
    "morning_vol_ratio",  # T morning volume / 20-day EOD volume MA (proxy)
]

# Full feature schema for the 14:30 entry path. Order matters — keep
# INTRADAY_EXTRA_COLUMNS at the tail so existing column-index code can
# treat the first len(FACTOR_COLUMNS) slots as the legacy schema.
INTRADAY_FEATURE_COLS: List[str] = FACTOR_COLUMNS + INTRADAY_EXTRA_COLUMNS


def compute_intraday_extras(
    intraday_bar: Dict,
    eod_history: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Compute the 3 intraday-specific features for one ``(code, T 14:30)`` point.

    Pure function — no I/O, safe to call from training loops.

    Parameters
    ----------
    intraday_bar : dict
        Today's morning-session OHLCV. Same shape consumed by
        :func:`mp.ml.dataset._process_single_stock`'s ``intraday_bar`` arg::

            {"date": <Timestamp>, "open": float, "high": float, "low": float,
             "close": float, "volume": float, "amount": float}

        ``high`` and ``low`` are accepted but not used by this function (the
        existing technical-factor pipeline consumes them via the synthetic bar).
    eod_history : DataFrame or None
        Optional EOD bar history (rows ending at T-1) with at least a
        ``volume`` column. Used to normalize ``morning_vol_ratio`` against
        a 20-day EOD-volume moving average. When ``None`` or shorter than
        20 rows, ``morning_vol_ratio`` is NaN (``morning_return`` and
        ``morning_vwap_dev`` remain computable).

    Returns
    -------
    dict[str, float]
        ``{col: value}`` for every entry of :data:`INTRADAY_EXTRA_COLUMNS`.
        NaN where inputs are zero/missing — does NOT raise.
    """
    open_p = float(intraday_bar.get("open") or 0.0)
    close_p = float(intraday_bar.get("close") or 0.0)
    volume = float(intraday_bar.get("volume") or 0.0)
    amount = float(intraday_bar.get("amount") or 0.0)

    if open_p > 0 and close_p > 0:
        morning_return = (close_p - open_p) / open_p
    else:
        morning_return = float("nan")

    if volume > 0 and amount > 0 and close_p > 0:
        morning_vwap = amount / volume
        if morning_vwap > 0:
            morning_vwap_dev = (close_p - morning_vwap) / morning_vwap
        else:
            morning_vwap_dev = float("nan")
    else:
        morning_vwap_dev = float("nan")

    if (
        eod_history is not None
        and "volume" in eod_history.columns
        and len(eod_history) >= 20
        and volume > 0
    ):
        recent20 = eod_history["volume"].tail(20).astype(float)
        ma20 = float(recent20.mean())
        morning_vol_ratio = (volume / ma20) if ma20 > 0 else float("nan")
    else:
        morning_vol_ratio = float("nan")

    return {
        "morning_return": morning_return,
        "morning_vwap_dev": morning_vwap_dev,
        "morning_vol_ratio": morning_vol_ratio,
    }


def build_intraday_panel(
    codes: List[str],
    asof_dt: pd.Timestamp,
    intraday_bars: Dict[str, Dict],
    start: str = "20230101",
    include_fundamentals: bool = True,
    eod_history_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Build the cross-sectional feature panel for prediction at T 14:30.

    Wraps :func:`mp.ml.dataset.build_latest_features` (which already supports
    ``intraday_bars`` injection — the synthetic-bar mechanism is in
    ``_process_single_stock``) and appends :data:`INTRADAY_EXTRA_COLUMNS`.

    Parameters
    ----------
    codes : list[str]
        Stocks to score.
    asof_dt : pd.Timestamp
        Trading-day timestamp for the T 14:30 snapshot. Advisory only —
        consistency against ``intraday_bars[c]["date"]`` is the caller's
        responsibility.
    intraday_bars : dict[code → dict]
        Per-stock morning-session OHLCV. Same dict shape consumed by the
        existing midday-report path.
    start : str
        EOD history start date in ``YYYYMMDD`` form.
    include_fundamentals : bool
        Whether to merge in fundamentals (PE/PB/ROE etc).
    eod_history_map : dict[code → DataFrame] or None
        Optional per-code EOD bar history used to fill ``morning_vol_ratio``.
        When omitted, ``morning_vol_ratio`` is NaN for all rows (the other
        two extras are still populated). P11-4 will wire this up against
        the proper historical morning-session totals.

    Returns
    -------
    DataFrame
        One row per stock with columns ``code``, ``date``, and
        :data:`INTRADAY_FEATURE_COLS`. Stocks dropped by
        ``build_latest_features`` (insufficient history, no fundamentals,
        etc.) are not added back.
    """
    panel = build_latest_features(
        codes,
        start=start,
        include_fundamentals=include_fundamentals,
        intraday_bars=intraday_bars,
    )
    if panel.empty:
        return panel

    extras_rows: List[Dict[str, float]] = []
    for c in panel["code"].astype(str):
        bar = intraday_bars.get(c, {})
        hist = eod_history_map.get(c) if eod_history_map else None
        extras_rows.append(compute_intraday_extras(bar, eod_history=hist))

    extras_df = pd.DataFrame(extras_rows, index=panel.index)
    for col in INTRADAY_EXTRA_COLUMNS:
        panel[col] = extras_df[col].to_numpy()

    return panel
