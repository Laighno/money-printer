"""Mini walk-forward Δ Sharpe gate — standalone ad-hoc tool (P2-#1).

Computes Δ Sharpe = Sharpe(with feature) - Sharpe(without feature) on a
recent K-fold expanding-window walk-forward restricted to a fixed
evaluation window.

⚠ KNOWN CALIBRATION FAILURE 2026-05-24 (docs/dialog/ round 30): when
``base_features=FACTOR_COLUMNS`` this gate does NOT reproduce the
W2-vs-W1 small-CURATED experiment Δ Sharpe values. Direction of effect
flipped (max_drawdown_20d: mini WF +0.11 vs full WF -0.18). Root cause:
the gate tests 64-feature leave-one-out counterfactual while W1/W2
ground truth is 28/30-feature add/remove — different baselines, conditional
on feature-set capacity (see round-21 winsorize reversal for the same
pattern). This is NOT a bug in this module; it is a property of feature
contribution being conditional on baseline.

**Therefore**:
  - Δ Sharpe here is meaningful WITHIN this gate run only (64-feature LOO).
  - Do NOT use this gate's verdict as a binding "REAL CONTRIBUTOR"
    confirmation for CURATED decisions — that requires either explicit
    same-baseline ground truth or a different design.
  - audit script (scripts/feature_importance_audit.py) intentionally
    does NOT integrate this gate as a verdict-promoter (reverted in
    P2-#1-fix-3 after calibration failed). Use this module for ad-hoc
    per-feature LOO experiments where you control the counterfactual.

See docs/TODO.md P2 ("multi-counterfactual problem"), docs/dialog/
rounds 25-30 for the full design + calibration history.

Used by `scripts/wf_gate_calibrate.py` (the calibration script that
caught the failure) and available for any future caller that needs an
ad-hoc per-feature Δ Sharpe measurement.

Why a separate "mini" walk-forward
----------------------------------
Production `walk_forward_backtest.py` does monthly retrain + daily
rebalance with SimulatedBroker / cost-aware logic for the full 2020-2026
period — ~8 minutes per run. Per-feature leave-one-out would multiply
that into days. Mini WF strips out the production-only pieces and
restricts the evaluation window so per-feature cost drops to ~3-5 min.

What this is NOT
----------------
- NOT a substitute for the production walk_forward report. Numbers here
  are calibration-scale (no transaction costs, no cost-aware rebalance,
  no PIT universe filter); compare WITHIN a wf_gate run, not against
  production Sharpe.
- NOT bit-perfect deterministic across runs unless caller passes a
  fully-deterministic ``panel`` and sets ``seed`` consistently.

Fixed configuration (do not parameterise without advisor sign-off)
------------------------------------------------------------------
- ``EVAL_WINDOW_START`` = "2024-01-01" — recent enough to capture the
  hs300+zz500 universe (widened 2026-05-14), long enough to span at
  least one drawdown.
- ``EVAL_WINDOW_END``   = "2025-12-31" — leave 2026 out as future hold-
  out for separate validation.
- ``TRAIN_START``       = "2016-05-01" — matches
  ``scripts/walk_forward_backtest.py::TRAIN_START`` so train window is
  identical to production.
- ``TOP_K``             = 10 — matches production.
- ``RANKER_KIND``       = "blend" — production default.

See docs/dialog/ round 27 for the full design rationale and round 28+
for the calibration scaling-factor derivation against known cases
(max_drawdown_20d / amount_ratio / atr_14).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger


# Fixed evaluation configuration — do not change without recalibrating
# the wf_gate Δ Sharpe threshold against known cases.
EVAL_WINDOW_START = "2024-01-01"
EVAL_WINDOW_END = "2025-12-31"
TRAIN_START = "2016-05-01"
TOP_K = 10
HORIZON_DEFAULT = 20


def _expanding_folds(
    eval_dates: np.ndarray,
    n_folds: int,
) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split ``eval_dates`` (sorted unique dates inside EVAL_WINDOW) into
    ``n_folds`` consecutive validation windows. Each fold's training set
    is everything before the validation window's start date.

    Returns list of (val_start, val_end) tuples (inclusive bounds).
    """
    eval_dates = np.asarray(eval_dates)
    if len(eval_dates) < n_folds:
        raise ValueError(
            f"only {len(eval_dates)} eval dates, cannot make {n_folds} folds"
        )
    chunk = len(eval_dates) // n_folds
    folds = []
    for i in range(n_folds):
        start_idx = i * chunk
        end_idx = (i + 1) * chunk - 1 if i < n_folds - 1 else len(eval_dates) - 1
        folds.append((pd.Timestamp(eval_dates[start_idx]),
                      pd.Timestamp(eval_dates[end_idx])))
    return folds


def _topk_excess_returns(
    panel_val: pd.DataFrame,
    scores: np.ndarray,
    label_col: str,
    top_k: int = TOP_K,
) -> np.ndarray:
    """For each date in ``panel_val``, take Top-K stocks by ``scores`` and
    compute (mean Top-K fwd_ret) - (median all-stocks fwd_ret). Returns
    one excess-return number per date.
    """
    df = panel_val.copy()
    df["__score"] = scores
    out = []
    for _, g in df.groupby("date"):
        valid = g.dropna(subset=[label_col, "__score"])
        if len(valid) < top_k * 2:
            continue
        top = valid.nlargest(top_k, "__score")
        out.append(float(top[label_col].mean() - valid[label_col].median()))
    return np.asarray(out, dtype=float)


def _portfolio_sharpe(daily_excess: np.ndarray, horizon: int) -> float:
    """Annualised Sharpe from daily Top-K excess returns. Strips NaN/inf.

    Since each daily 'excess' here is a forward-``horizon``-day return
    realised at decision time, we approximate annualisation as
    ``mean / std * sqrt(252 / horizon)`` — matches production
    ``walk_forward_backtest.py::calc_performance`` convention.
    """
    arr = daily_excess[np.isfinite(daily_excess)]
    if arr.size < 5:
        return float("nan")
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return mu / sd * np.sqrt(252.0 / horizon)


def wf_gate_delta_sharpe(
    panel: pd.DataFrame,
    feature_to_test: str,
    base_features: List[str],
    *,
    n_folds: int = 3,
    horizon: int = HORIZON_DEFAULT,
    seed: int = 42,
    ranker_kind: str = "blend",
    label_col: str = "fwd_ret",
    eval_window_start: str = EVAL_WINDOW_START,
    eval_window_end: str = EVAL_WINDOW_END,
) -> dict:
    """Compute Δ Sharpe for ``feature_to_test`` via leave-one-out
    mini walk-forward on the fixed evaluation window.

    Parameters
    ----------
    panel
        Must contain ``date``, all ``base_features``, ``feature_to_test``,
        and ``label_col``. Built by caller via ``build_dataset`` so PIT
        / winsorize / etc. happen once.
    feature_to_test
        Feature name to test. If already in ``base_features``, ``feats_with``
        equals ``base_features`` (no-op) and we still compare to
        ``feats_without = base \\ {feature}``.
    base_features
        Baseline feature set (typically ``FACTOR_COLUMNS`` full 64).
    n_folds
        Expanding-window mini WF fold count. Default 3 (~30 min total per
        feature). Higher = lower noise but linear cost.
    horizon, seed, ranker_kind, label_col
        See `walk_forward_backtest.py` for semantics.
    eval_window_start, eval_window_end
        Override only if recalibrating the gate's Sharpe threshold.

    Returns
    -------
    dict::

        {"feature": str,
         "sharpe_with": float,
         "sharpe_without": float,
         "delta_sharpe": float,   # positive = feature helps
         "n_folds_used": int,
         "n_train_rows_per_fold": list[int],
         "n_val_dates_per_fold": list[int]}

    Notes
    -----
    Per-fold training uses ``BlendRanker.train_fast`` (matches production
    monthly retrain calls). LGBM seed passed via ``LGBM_SEED`` env at
    function entry; do not rely on a thread-local override.
    """
    import os
    os.environ["LGBM_SEED"] = str(seed)

    # Import lazily so `from mp.ml.wf_gate import ...` is cheap.
    from mp.ml.model import BlendRanker, StockRanker

    feats_with = list(base_features)
    if feature_to_test not in feats_with:
        feats_with = feats_with + [feature_to_test]
    feats_without = [f for f in feats_with if f != feature_to_test]
    if len(feats_with) == len(feats_without):
        raise ValueError(
            f"feature_to_test={feature_to_test!r} not in base_features and "
            f"removing it yields the same set; nothing to compare"
        )

    # Restrict panel to eval window upper bound (training is everything
    # at or before each fold's val_start - 1 day).
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)

    eval_mask = (panel["date"] >= pd.Timestamp(eval_window_start)) & (
        panel["date"] <= pd.Timestamp(eval_window_end)
    )
    eval_dates = np.sort(panel.loc[eval_mask, "date"].unique())
    if len(eval_dates) < n_folds * 5:
        raise ValueError(
            f"only {len(eval_dates)} eval dates in [{eval_window_start},"
            f"{eval_window_end}], need at least {n_folds * 5}"
        )

    folds = _expanding_folds(eval_dates, n_folds)

    def _run_one_feature_set(feats: List[str]) -> tuple[list[float], list[int], list[int]]:
        per_date_excess: list[float] = []
        train_rows_per_fold: list[int] = []
        val_dates_per_fold: list[int] = []
        for val_start, val_end in folds:
            train_mask = panel["date"] < val_start
            val_mask = (panel["date"] >= val_start) & (panel["date"] <= val_end)
            train = panel.loc[train_mask].copy()
            val = panel.loc[val_mask].copy()
            if len(train) < 100 or len(val) < TOP_K * 5:
                logger.debug("wf_gate fold {}-{} too small: train={} val={}",
                             val_start.date(), val_end.date(), len(train), len(val))
                continue

            # Train on this expanding window with ``feats``
            if ranker_kind == "blend":
                ranker = BlendRanker(feature_cols=feats)
            else:
                ranker = StockRanker(feature_cols=feats, label_col=label_col)
            try:
                ranker.train_fast(train, val_frac=0.10)
            except Exception as e:
                logger.warning("wf_gate train_fast failed on fold {}-{}: {}",
                               val_start.date(), val_end.date(), e)
                continue

            scores = ranker.predict(val)
            fold_excess = _topk_excess_returns(val, scores, label_col=label_col)
            per_date_excess.extend(fold_excess.tolist())
            train_rows_per_fold.append(int(len(train)))
            val_dates_per_fold.append(int(val["date"].nunique()))
        return per_date_excess, train_rows_per_fold, val_dates_per_fold

    logger.info("wf_gate '{}': training {} folds × 2 feature sets ({} vs {})",
                feature_to_test, len(folds), len(feats_with), len(feats_without))

    ex_with, tr_with, vd_with = _run_one_feature_set(feats_with)
    ex_without, tr_without, vd_without = _run_one_feature_set(feats_without)

    sh_with = _portfolio_sharpe(np.asarray(ex_with), horizon)
    sh_without = _portfolio_sharpe(np.asarray(ex_without), horizon)
    delta = sh_with - sh_without if np.isfinite(sh_with) and np.isfinite(sh_without) else float("nan")

    return {
        "feature": feature_to_test,
        "sharpe_with": sh_with,
        "sharpe_without": sh_without,
        "delta_sharpe": delta,
        "n_folds_used": min(len(tr_with), len(tr_without)),
        "n_train_rows_per_fold": tr_with,
        "n_val_dates_per_fold": vd_with,
    }
