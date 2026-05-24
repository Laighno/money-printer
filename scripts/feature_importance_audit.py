"""Compare 3 feature-importance metrics to audit CURATED_COLUMNS.

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

⚠ IN-SAMPLE ONLY — verdicts here are NOT binding for CURATED decisions.
P2 2026-05-24 W1/W2 walk-forward A/B showed audit's "REAL CONTRIBUTOR"
classification disagreed with out-of-sample evidence (max_drawdown_20d /
roe_qoq added Sharpe -0.18 / Max DD -3.36pp on the W1=28 baseline, opposite
of in-sample prediction). Use ``mp/ml/wf_gate.py`` for ad-hoc per-feature
OOS experiments, but read its calibration-failure note first: in 2026-05-24
calibration the gate's Δ Sharpe direction did NOT match W1/W2 ground truth
because the counterfactuals differ (wf_gate tests 64-feature LOO, W1/W2
tests small-CURATED add/remove). Feature contribution is conditional on
the baseline feature set; there is no universal ground truth — see
docs/TODO.md P2 ("multi-counterfactual problem") and docs/dialog/ rounds
12-14 (W1/W2 origin) + 25-30 (wf_gate design + calibration failure).

Output: ranked feature table with all three metrics + recommendation.
"""
from __future__ import annotations

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


def main() -> int:
    logger.info("=== Feature importance audit ===")
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

    # Print report
    print("\n" + "=" * 105)
    print(f"FEATURE IMPORTANCE AUDIT (val baseline IC = {baseline_ic:.4f})")
    print("⚠  IN-SAMPLE ONLY — 'REAL CONTRIBUTOR' here is NOT a binding CURATED decision.")
    print("    Use mp/ml/wf_gate.py for ad-hoc per-feature OOS experiments, but read its")
    print("    calibration-failure note first (different counterfactual than W1/W2 ground truth).")
    print("    See docs/TODO.md P2 (multi-counterfactual) + docs/dialog/ rounds 12-14, 25-30.")
    print("=" * 105)
    print(f"{'Rank':>4} {'Feature':<26} {'|IR|':>7} {'LGBM gain%':>11} {'Perm ΔIC':>10} "
          f"{'InCURATED':>10}  Verdict")
    print("-" * 105)
    for i, (feat, row) in enumerate(audit.iterrows(), 1):
        ir, gain, dic = row["abs_ir"], row["gain_pct"], row["ic_delta"]
        in_cur = "✓" if row["in_curated"] else "✗"
        # Verdict (in-sample only — see header warning)
        if dic > 0.005 and gain > 0.5:
            verdict = "REAL CONTRIBUTOR"
        elif dic > 0.001:
            verdict = "weak signal"
        elif dic > -0.001:
            verdict = "no signal"
        else:
            verdict = "NOISE (hurts)"
        print(f"  {i:>2} {feat:<26} {ir:>7.3f} {gain:>10.2f}% {dic:>+10.5f} "
              f"{in_cur:>10}  {verdict}")

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
