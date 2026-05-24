"""Compare 3 feature-importance metrics to audit CURATED_COLUMNS,
optionally with a mini walk-forward second-stage gate.

The cross-sectional IR test (|IR| >= 0.15) is *univariate* — it can't see
non-linear interactions or feature substitution.  This script trains a
LightGBM model on FACTOR_COLUMNS (all 47 features) and reports three
complementary importance metrics:

  1. **IR** (existing) — univariate cross-sectional rank correlation
  2. **LightGBM gain** — total split improvement (intrinsic to GBT)
  3. **Permutation IC drop** — shuffle each feature, measure how much
     val-set IC drops — gold standard for multi-variate importance

If a feature has low IR but high gain + permutation drop, it's a real
multi-variate contributor and we may have wrongly dropped it from CURATED.

⚠ in-sample only by default. The "in-sample positive" verdict requires
walk-forward validation before being used for CURATED decisions —
P2 2026-05-24 W1/W2 walk-forward A/B showed audit's "REAL CONTRIBUTOR"
classification disagreed with out-of-sample evidence (max_drawdown_20d /
roe_qoq added Sharpe -0.18 / Max DD -3.36pp, opposite of in-sample
prediction). Run with ``--wf-gate`` to add walk-forward Δ Sharpe as
a conditional second-stage gate. See docs/TODO.md P2 / docs/dialog/
rounds 12-14 (origin) and 25-28 (--wf-gate design).

Output: ranked feature table with all three metrics + recommendation
(and, when ``--wf-gate``, a fourth column ``wf_gate_delta_sharpe``).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mp.data.fetcher import get_recommendation_universe
from mp.ml.dataset import (
    CURATED_COLUMNS,
    EXCESS_LABEL,
    FACTOR_COLUMNS,
    build_dataset,
)
from mp.ml.model import StockRanker
from mp.ml.wf_gate import TRAIN_START as WF_GATE_TRAIN_START
from mp.ml.wf_gate import wf_gate_delta_sharpe


def _ir_from_panel(panel: pd.DataFrame, feature: str) -> float:
    """Univariate cross-sectional IR (mean/std of per-date Spearman corr)."""
    ic_per_date = []
    for _, group in panel.groupby("date"):
        x = group[feature].to_numpy(dtype=float)
        y = group[EXCESS_LABEL].to_numpy(dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 30:
            continue
        try:
            ic, _ = spearmanr(x[mask], y[mask])
            if not np.isnan(ic):
                ic_per_date.append(ic)
        except Exception:
            pass
    if not ic_per_date:
        return float("nan")
    arr = np.array(ic_per_date)
    return float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else 0.0


def _val_ic(model, X: pd.DataFrame, y: pd.Series, feature_cols: list[str]) -> float:
    """Mean per-date IC on a held-out panel.  Same metric LightGBM training uses."""
    preds = model.predict(X[feature_cols])
    dates = X["date"].values if "date" in X.columns else np.zeros(len(X))
    df = pd.DataFrame({"pred": preds, "actual": y.values, "date": dates})
    if pd.Series(dates).nunique() <= 1:
        return float(spearmanr(df["pred"], df["actual"])[0])
    ics = []
    for _, grp in df.groupby("date"):
        if len(grp) < 5:
            continue
        ic, _ = spearmanr(grp["pred"], grp["actual"])
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wf-gate", action="store_true",
        help="Run mini walk-forward Δ Sharpe gate on each in-sample positive "
             "candidate. Adds wf_gate_delta_sharpe column. ~5-10 min per "
             "candidate; expect ~10 candidates → ~30-60 min total. See "
             "mp/ml/wf_gate.py and docs/dialog/ rounds 25-28.",
    )
    p.add_argument(
        "--wf-gate-folds", type=int, default=3,
        help="Number of expanding-window folds for --wf-gate mini WF (default 3).",
    )
    p.add_argument(
        "--wf-gate-threshold", type=float, default=None,
        help="Δ Sharpe threshold to upgrade verdict to 'WF-confirmed'. "
             "Default: None (only mark as in-sample-positive). Threshold should "
             "be calibrated empirically — see docs/dialog/ round 28.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    logger.info("=== Feature importance audit ===")
    if args.wf_gate:
        logger.info("--wf-gate ACTIVE: will validate in-sample positives via "
                    "mini walk-forward (folds={}, threshold={})",
                    args.wf_gate_folds, args.wf_gate_threshold)
    codes = get_recommendation_universe()
    panel = build_dataset(codes, "20220101", horizon=20)
    logger.info("Panel: {:,} rows × {} factors", len(panel), len(FACTOR_COLUMNS))

    # Use ALL FACTOR_COLUMNS that exist in panel
    feature_cols = [c for c in FACTOR_COLUMNS if c in panel.columns]
    panel_clean = panel.dropna(subset=feature_cols + [EXCESS_LABEL])
    logger.info("Clean panel (no NaN): {:,} rows × {} features",
                len(panel_clean), len(feature_cols))

    # Time-based 80/20 split for honest val
    dates_sorted = sorted(panel_clean["date"].unique())
    split_idx = int(len(dates_sorted) * 0.8)
    split_date = dates_sorted[split_idx]
    train = panel_clean[panel_clean["date"] < split_date].copy()
    val = panel_clean[panel_clean["date"] >= split_date].copy()
    logger.info("Train: {:,} rows (< {})  Val: {:,} rows (>= {})",
                len(train), split_date, len(val), split_date)

    # Train a single StockRanker with FACTOR_47
    logger.info("Training StockRanker with all {} features...", len(feature_cols))
    t0 = time.time()
    ranker = StockRanker(feature_cols=feature_cols)
    ranker.train_fast(train)
    logger.info("Trained in {:.0f}s", time.time() - t0)

    # 1. LightGBM gain importance
    importance_df = ranker.feature_importance_report()
    importance_df = importance_df.set_index("feature")
    importance_df["gain_pct"] = importance_df["importance"] / importance_df["importance"].sum() * 100

    # 2. Baseline val IC
    Xv = val[feature_cols + ["date"]]
    yv = val[EXCESS_LABEL]
    baseline_ic = _val_ic(ranker.model, Xv, yv, feature_cols)
    logger.info("Baseline val IC: {:.4f}", baseline_ic)

    # 3. Permutation importance: shuffle each feature, measure IC drop
    logger.info("Computing permutation importance for {} features...", len(feature_cols))
    rng = np.random.default_rng(seed=42)
    perm_results = []
    for i, feat in enumerate(feature_cols, 1):
        Xp = Xv.copy()
        Xp[feat] = rng.permutation(Xp[feat].to_numpy())
        ic_shuffled = _val_ic(ranker.model, Xp, yv, feature_cols)
        delta = baseline_ic - ic_shuffled   # positive = feature was useful
        perm_results.append({"feature": feat, "ic_delta": delta,
                              "ic_shuffled": ic_shuffled})
        if i % 10 == 0:
            logger.info("  perm {}/{}", i, len(feature_cols))
    perm_df = pd.DataFrame(perm_results).set_index("feature")

    # 4. Univariate IR (recompute on same panel for consistency)
    logger.info("Computing univariate IR for cross-check...")
    ir_rows = []
    for feat in feature_cols:
        ir = _ir_from_panel(panel_clean, feat)
        ir_rows.append({"feature": feat, "ir": ir, "abs_ir": abs(ir) if ir == ir else 0})
    ir_df = pd.DataFrame(ir_rows).set_index("feature")

    # Merge all 3 metrics
    audit = pd.concat([ir_df, importance_df[["gain_pct"]], perm_df[["ic_delta"]]],
                       axis=1).fillna(0)
    audit["in_curated"] = audit.index.isin(CURATED_COLUMNS)

    # Sort by permutation importance (gold standard)
    audit = audit.sort_values("ic_delta", ascending=False)

    # --- Optional: walk-forward Δ Sharpe gate ----------------------------
    audit["wf_gate_delta_sharpe"] = np.nan
    if args.wf_gate:
        # Defensive: enforce wf_gate's TRAIN_START boundary at caller site
        # (round 28). wf_gate's internal train_mask doesn't filter by
        # TRAIN_START itself; we enforce here to make the contract explicit.
        gate_panel = panel_clean[
            pd.to_datetime(panel_clean["date"]) >= pd.Timestamp(WF_GATE_TRAIN_START)
        ].copy()
        # In-sample positive candidates: same rule the old verdict used
        candidates = audit[
            (audit["ic_delta"] > 0.005) & (audit["gain_pct"] > 0.5)
        ].index.tolist()
        logger.info("--wf-gate: {} in-sample-positive candidates to validate "
                    "(panel ≥ {}: {:,} rows × {} dates)",
                    len(candidates), WF_GATE_TRAIN_START, len(gate_panel),
                    gate_panel["date"].nunique())
        base = list(FACTOR_COLUMNS)
        for j, feat in enumerate(candidates, 1):
            try:
                t_g0 = time.time()
                result = wf_gate_delta_sharpe(
                    panel=gate_panel,
                    feature_to_test=feat,
                    base_features=base,
                    n_folds=args.wf_gate_folds,
                    ranker_kind="blend",
                    label_col=EXCESS_LABEL,
                )
                audit.loc[feat, "wf_gate_delta_sharpe"] = result["delta_sharpe"]
                logger.info("  [{}/{}] {}: Δ Sharpe = {:+.4f}  "
                            "(with={:+.3f} / without={:+.3f}, {:.0f}s)",
                            j, len(candidates), feat, result["delta_sharpe"],
                            result["sharpe_with"], result["sharpe_without"],
                            time.time() - t_g0)
            except Exception as e:
                logger.warning("  wf_gate failed for {}: {}", feat, e)

    # --- Print report ----------------------------------------------------
    print("\n" + "=" * 120)
    print(f"FEATURE IMPORTANCE AUDIT (val baseline IC = {baseline_ic:.4f})")
    if not args.wf_gate:
        print("⚠  IN-SAMPLE ONLY — verdicts marked 'in-sample positive' need walk-forward")
        print("    validation before CURATED decisions. Rerun with --wf-gate to add the gate.")
        print("    See docs/dialog/ rounds 12-14 (origin) and 25-28 (--wf-gate design).")
    print("=" * 120)
    wf_col = "WF Δ Sharpe" if args.wf_gate else ""
    print(f"{'Rank':>4} {'Feature':<26} {'|IR|':>7} {'LGBM gain%':>11} {'Perm ΔIC':>10} "
          f"{wf_col:>12} {'InCURATED':>10}  Verdict")
    print("-" * 120)
    for i, (feat, row) in enumerate(audit.iterrows(), 1):
        ir, gain, dic = row["abs_ir"], row["gain_pct"], row["ic_delta"]
        wf_delta = row["wf_gate_delta_sharpe"]
        in_cur = "✓" if row["in_curated"] else "✗"
        # Verdict — staged: in-sample positive vs WF-confirmed
        in_sample_positive = dic > 0.005 and gain > 0.5
        if in_sample_positive and args.wf_gate and args.wf_gate_threshold is not None:
            if np.isfinite(wf_delta) and wf_delta > args.wf_gate_threshold:
                verdict = "REAL CONTRIBUTOR (WF-confirmed)"
            else:
                verdict = "in-sample positive (WF rejected)"
        elif in_sample_positive:
            verdict = "in-sample positive (NEEDS WF VALIDATION)"
        elif dic > 0.001:
            verdict = "weak signal"
        elif dic > -0.001:
            verdict = "no signal"
        else:
            verdict = "NOISE (hurts)"
        wf_disp = f"{wf_delta:>+8.4f}" if (args.wf_gate and np.isfinite(wf_delta)) else (
            "      —" if args.wf_gate else "")
        print(f"  {i:>2} {feat:<26} {ir:>7.3f} {gain:>10.2f}% {dic:>+10.5f} "
              f"{wf_disp:>12} {in_cur:>10}  {verdict}")

    # Cross-check: which dropped features have HIGH permutation importance?
    dropped = audit[~audit["in_curated"]].copy()
    misclassified_keep = dropped[(dropped["ic_delta"] > 0.002) & (dropped["gain_pct"] > 0.5)]

    kept = audit[audit["in_curated"]].copy()
    misclassified_drop = kept[(kept["ic_delta"] < 0.001) & (kept["gain_pct"] < 0.5)]

    print("\n" + "=" * 105)
    print("RECOMMENDED ADDITIONS — features dropped from CURATED but have high LGBM contribution:")
    print("=" * 105)
    if misclassified_keep.empty:
        print("  (none — current CURATED is correct)")
    else:
        for feat, row in misclassified_keep.sort_values("ic_delta", ascending=False).iterrows():
            print(f"  + {feat:<26}  |IR|={row['abs_ir']:.3f}  gain={row['gain_pct']:.2f}%  "
                  f"perm ΔIC={row['ic_delta']:+.5f}")

    print("\nCANDIDATES TO REMOVE — kept in CURATED but contributes nothing:")
    print("=" * 105)
    if misclassified_drop.empty:
        print("  (none — all kept features contribute)")
    else:
        for feat, row in misclassified_drop.sort_values("ic_delta").iterrows():
            print(f"  - {feat:<26}  |IR|={row['abs_ir']:.3f}  gain={row['gain_pct']:.2f}%  "
                  f"perm ΔIC={row['ic_delta']:+.5f}")

    # Save to JSON
    out = Path("data/feature_importance_audit.json")
    audit_reset = audit.reset_index()
    out.write_text(audit_reset.to_json(orient="records", indent=2), encoding="utf-8")
    logger.info("Saved → {}", out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
