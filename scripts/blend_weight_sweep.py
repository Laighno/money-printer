"""Sweep BlendRanker weight_primary post-hoc to validate 0.80 is still optimal.

Leverages per-row component scores (primary_score / extreme_score) now saved
by run_scoring_loop_blend. Runs the walk-forward training ONCE, then rescans
many weights for free. Uses look-ahead-free (PIT) data.

Output: markdown table of metrics per weight.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.prediction_diagnostics import (  # noqa: E402
    run_scoring_loop_blend, CACHE_DIR,
)


WEIGHTS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 0.95, 1.00]
TOP_K = 20


def _metrics_for_weight(diag: pd.DataFrame, w: float) -> dict:
    df = diag.copy()
    # Rank-normalize each component per date
    df["r_primary"] = df.groupby("score_date")["primary_score"].rank(pct=True)
    df["r_extreme"] = df.groupby("score_date")["extreme_score"].rank(pct=True)
    df["blend"] = w * df["r_primary"] + (1 - w) * df["r_extreme"]

    # Per-date top-K mean fwd_ret and rank IC
    per_month = []
    ics = []
    for d, g in df.groupby("score_date"):
        if len(g) < TOP_K * 2:
            continue
        g_sorted = g.sort_values("blend", ascending=False)
        top = g_sorted.head(TOP_K)
        top_ret = top["fwd_ret"].mean()
        # "win" = top_ret > cross-sectional median
        med_ret = g["fwd_ret"].median()
        per_month.append({
            "date": d,
            "top_ret": top_ret,
            "med_ret": med_ret,
            "excess": top_ret - med_ret,
            "win": int(top_ret > med_ret),
        })
        ic = spearmanr(g["blend"], g["fwd_ret"]).correlation
        if pd.notna(ic):
            ics.append(ic)

    pm = pd.DataFrame(per_month)
    ics_arr = np.array(ics)
    return {
        "n_months": len(pm),
        "mean_top_ret": pm["top_ret"].mean(),
        "mean_excess": pm["excess"].mean(),
        "win_rate": pm["win"].mean(),
        "ic_mean": ics_arr.mean() if len(ics_arr) else np.nan,
        "ic_std": ics_arr.std() if len(ics_arr) else np.nan,
        "icir": ics_arr.mean() / ics_arr.std() if len(ics_arr) > 1 and ics_arr.std() > 0 else np.nan,
    }


def main() -> None:
    # Run the walk-forward blend once (cached). Use the default tag=blend_best.
    # We need the cache to have component columns; if stale, force rerun.
    cache_path = CACHE_DIR / "diagnostics_blend_best.parquet"
    need_rerun = True
    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        if "primary_score" in existing.columns and "extreme_score" in existing.columns:
            need_rerun = False
            logger.info("Using cached diagnostics_blend_best.parquet ({} rows, already has components)",
                        len(existing))
            diag = existing
    if need_rerun:
        logger.info("Running walk-forward blend once (will cache component scores)...")
        diag = run_scoring_loop_blend(
            force=True, tag="blend_best",
            weight_primary=0.80, extreme_pctile=0.30,
        )
    if "primary_score" not in diag.columns:
        logger.error("diag_df missing primary_score — rerun with updated run_scoring_loop_blend")
        sys.exit(1)

    # Drop rows with missing fwd_ret (shouldn't happen but be safe)
    diag = diag.dropna(subset=["fwd_ret"]).copy()
    logger.info("{} scored rows across {} months",
                len(diag), diag["score_date"].nunique())

    rows = []
    for w in WEIGHTS:
        m = _metrics_for_weight(diag, w)
        rows.append({"weight_primary": w, **m})
        logger.info("w={:.2f} | top20_mean={:+.2%} | excess={:+.2%} | win={:.1%} | ICIR={:.3f}",
                    w, m["mean_top_ret"], m["mean_excess"], m["win_rate"], m["icir"] or 0)

    out = pd.DataFrame(rows)
    print("\n## BlendRanker weight sweep (PIT-corrected data)\n")
    print("| w_primary | top20 mean | top20 excess vs median | 胜率 | IC mean | ICIR |")
    print("|---:|---:|---:|---:|---:|---:|")
    for _, r in out.iterrows():
        print(f"| {r['weight_primary']:.2f} | {r['mean_top_ret']:+.2%} | {r['mean_excess']:+.2%} "
              f"| {r['win_rate']:.1%} | {r['ic_mean']:+.4f} | {r['icir']:+.3f} |")

    # Best weight by each metric
    best_excess = out.loc[out["mean_excess"].idxmax()]
    best_icir = out.loc[out["icir"].idxmax()]
    print(f"\nBest by top20 excess: w={best_excess['weight_primary']:.2f} "
          f"({best_excess['mean_excess']:+.2%})")
    print(f"Best by ICIR: w={best_icir['weight_primary']:.2f} "
          f"({best_icir['icir']:+.3f})")

    out_path = Path("data/reports/blend_weight_sweep.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("# BlendRanker weight sweep\n\n")
        f.write("Post-hoc sweep using per-row component scores from a single "
                "walk-forward run. PIT-corrected data (industry + constituent "
                "snapshots + survivorship filter).\n\n")
        f.write("| w_primary | top20 mean | top20 excess vs median | 胜率 | IC mean | ICIR |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for _, r in out.iterrows():
            f.write(f"| {r['weight_primary']:.2f} | {r['mean_top_ret']:+.2%} "
                    f"| {r['mean_excess']:+.2%} | {r['win_rate']:.1%} "
                    f"| {r['ic_mean']:+.4f} | {r['icir']:+.3f} |\n")
        f.write(f"\nBest by top20 excess: **w={best_excess['weight_primary']:.2f}** "
                f"({best_excess['mean_excess']:+.2%})\n")
        f.write(f"Best by ICIR: **w={best_icir['weight_primary']:.2f}** "
                f"({best_icir['icir']:+.3f})\n")
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
