"""Shared IC / ICIR computation helpers.

History note: an earlier copy in ``scripts/cross_sectional_ic.py`` defined
``ICIR = mean / std * sqrt(N)``, which is actually the t-statistic of the IC
mean (mean / SE). With N ~ 800 trading days, sqrt(N) ~ 28x, so every factor
appeared "strong" — see ``docs/dialog/`` for the audit. This module is the
single source of truth; other scripts must import from here.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def icir(ic: Iterable[float]) -> float:
    """Standard ICIR = mean(IC) / std(IC).

    Returns 0.0 if std is zero or the series is empty.
    """
    arr = np.asarray(list(ic), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if std <= 0:
        return 0.0
    return float(arr.mean()) / std


def t_stat(ic: Iterable[float]) -> float:
    """t-statistic of IC mean = mean(IC) / std(IC) * sqrt(N).

    Use this when you want to test "is mean IC significantly != 0", NOT
    as a substitute for ICIR.
    """
    arr = np.asarray(list(ic), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 2:
        return 0.0
    std = float(arr.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(arr.mean()) / std * np.sqrt(n)


def summarize_ic(ic_series: pd.Series | Iterable[float]) -> dict:
    """Return ``{n, mean, std, icir, t_stat, pos_pct, abs_mean}``.

    ICIR uses the standard ``mean / std`` definition; ``t_stat`` is the
    significance-of-mean version (``mean / std * sqrt(N)``).
    """
    if isinstance(ic_series, pd.Series):
        arr = ic_series.dropna().to_numpy(dtype=float)
    else:
        arr = np.asarray(list(ic_series), dtype=float)
        arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "icir": float("nan"), "t_stat": float("nan"),
                "pos_pct": float("nan"), "abs_mean": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    ir = mean / std if std > 0 else 0.0
    ts = ir * np.sqrt(n) if std > 0 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "icir": ir,
        "t_stat": ts,
        "pos_pct": float((arr > 0).sum() / n),
        "abs_mean": abs(mean),
    }
