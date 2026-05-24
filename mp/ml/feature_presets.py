"""Walk-forward feature-set presets.

Frozen snapshots used to compare W0 (production baseline) vs W1/W2
(audit-driven candidates) after the P0 ICIR fix (commit b023ba4).

These lists are **the single source of truth** for the W0/W1/W2 experiments
documented in `docs/dialog/`. Don't mutate `CURATED_COLUMNS` in
`mp/ml/dataset.py` for these experiments — pass `WF_FEATURE_PRESET=W0|W1|W2`
env var to `scripts/walk_forward_backtest.py` instead. That way the
preset names appear in logs/reports and remain auditable in git history.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List


# 32-feature snapshot from working-tree mp/ml/dataset.py:CURATED_COLUMNS as
# of 2026-05-24 (post prior-session WIP, pre-P0 audit verdict). This is the
# feature set that trained data/model.lgb and produced BASELINE.md
# Sharpe 1.88-2.01 / Calmar ~3.07 / annual ~67%. W0 must reproduce those
# numbers within ±0.1 Sharpe; otherwise W0 ground-truth has drifted.
W0_PRESET: List[str] = [
    # MODERATE univariate (0.30 <= |IR| < 0.50)
    "pb_ind_rank",
    "pe_ind_rank",
    "amihud_illiq",
    "vwap_dev",
    "total_mv_log",
    "ma_alignment",
    # WEAK univariate but kept (0.15 <= |IR| < 0.30)
    "close_ma60_dev",
    "volume_volatility",
    "amount_volatility",
    "low_distance_60d",
    "mom_60d",
    "mom_20d",
    "lower_shadow",
    "pb",
    "mom_20d_ind_rank",
    "mom_accel",
    "price_range_10d",
    "upper_shadow",
    "close_ma20_dev",
    "rsi_14",
    "obv_slope",
    "intraday_intensity",
    "return_extremes_ratio",
    "vol_price_corr",
    "mfi_14",
    "mom_10d",
    "return_kurtosis_20d",
    "vol_ratio_5_60",
    # Added 2026-05-23 after permutation audit — Bug 2 era; verdicts now
    # re-validated under fixed audit (max_drawdown_20d / roe_qoq still
    # REAL CONTRIBUTOR, amount_ratio / atr_14 noise). Kept in W0 because
    # W0 = "snapshot of what production was actually trained with".
    "max_drawdown_20d",
    "roe_qoq",
    "amount_ratio",
    "atr_14",
]

# Features the new IC analysis (post-P0) drops from W0:
#   amount_ratio / atr_14         — noise in both IC and audit (gain=0, perm ΔIC=0)
#   max_drawdown_20d / roe_qoq    — fail IC threshold but REAL CONTRIBUTOR in audit
_BUG2_AUDIT_DROPPED = frozenset(
    {"amount_ratio", "atr_14", "max_drawdown_20d", "roe_qoq"}
)
_NOISE_ONLY = frozenset({"amount_ratio", "atr_14"})

# 28 = W0 minus all 4 IC-failing features (pure new-CURATED hypothesis)
W1_PRESET: List[str] = [f for f in W0_PRESET if f not in _BUG2_AUDIT_DROPPED]

# 30 = W1 plus the two REAL CONTRIBUTOR adds (audit-hybrid hypothesis)
W2_PRESET: List[str] = [f for f in W0_PRESET if f not in _NOISE_ONLY]


# 64-feature snapshot of mp.ml.dataset.FACTOR_COLUMNS taken on 2026-05-24.
# This is the feature set BASELINE.md L25 cites — running W_BASELINE here
# is the reproduction gate before comparing W0/W1/W2. The list is HARDCODED
# (not `list(FACTOR_COLUMNS)`) so it stays frozen even if FACTOR_COLUMNS
# is mutated later. If a future contributor edits FACTOR_COLUMNS, the
# default walk_forward path (no env) will drift but W_BASELINE will not —
# that's the entire point of pinning it here.
W_BASELINE_PRESET: List[str] = [
    "rsi_14", "macd_hist", "boll_pctb", "kdj_j", "vol_price_ratio",
    "mom_20d", "mom_60d", "volatility_20d", "rsi_delta", "macd_hist_delta",
    "boll_pctb_delta", "mom_accel", "volume_trend", "close_ma5_dev",
    "close_ma20_dev", "close_ma60_dev", "ma_alignment", "atr_14",
    "price_range_10d", "williams_r", "adx_14", "obv_slope", "amount_ratio",
    "turnover_5d", "turnover_pctile", "return_skew_20d", "return_kurtosis_20d",
    "updown_vol_ratio", "max_drawdown_20d", "return_autocorr", "close_position",
    "upper_shadow", "lower_shadow", "body_ratio", "gap_5d", "amihud_illiq",
    "volume_volatility", "mom_5d", "mom_10d", "vwap_dev", "boll_bandwidth",
    "vol_price_corr", "consecutive_days", "high_distance_60d",
    "low_distance_60d", "vol_ratio_5_60", "mfi_14", "intraday_intensity",
    "cci_20", "return_extremes_ratio", "amount_volatility", "pe_ttm", "pb",
    "total_mv_log", "roe", "revenue_growth", "profit_growth", "roe_qoq",
    "profit_growth_accel", "revenue_growth_accel", "pe_ind_rank",
    "pb_ind_rank", "roe_ind_rank", "mom_20d_ind_rank",
]


PRESETS: Dict[str, List[str]] = {
    "W_BASELINE": W_BASELINE_PRESET,
    "W0": W0_PRESET,
    "W1": W1_PRESET,
    "W2": W2_PRESET,
}


def preset_signature(name: str) -> str:
    """Return short SHA1 of the joined preset list, for log/report provenance.

    If someone silently edits a preset later, the signature changes and the
    historical report still names the old SHA — making the drift obvious.
    """
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; valid: {sorted(PRESETS)}")
    payload = "\n".join(PRESETS[name]).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]
