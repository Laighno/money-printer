"""Per-regime weight sweep to check whether extreme head has residual value
in specific market regimes (bull/bear/sideways) that averages out to zero.

Reuses the cached per-row component scores from scripts/blend_weight_sweep.py,
so no model retrain needed. PIT-filtered universe assumed (the cache file
is regenerated whenever prediction_diagnostics.py is updated).
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

from scripts.prediction_diagnostics import REGIMES, _classify_regime  # noqa: E402
from mp.ml.ic_utils import icir as _icir  # noqa: E402


CACHE = Path("data/wf_cache/diagnostics_blend_best.parquet")
WEIGHTS = [0.60, 0.70, 0.80, 0.90, 1.00]
TOP_K = 20


def _metrics(df: pd.DataFrame, w: float) -> dict:
    if df.empty or df["score_date"].nunique() < 3:
        return {"n_months": df["score_date"].nunique(), "top20_excess": np.nan,
                "win_rate": np.nan, "ic_mean": np.nan, "icir": np.nan}
    d = df.copy()
    d["r_p"] = d.groupby("score_date")["primary_score"].rank(pct=True)
    d["r_e"] = d.groupby("score_date")["extreme_score"].rank(pct=True)
    d["blend"] = w * d["r_p"] + (1 - w) * d["r_e"]
    rows = []
    ics = []
    for dt, g in d.groupby("score_date"):
        if len(g) < TOP_K * 2:
            continue
        top = g.nlargest(TOP_K, "blend")
        med = g["fwd_ret"].median()
        rows.append({"top_ret": top["fwd_ret"].mean(), "med_ret": med})
        ic = spearmanr(g["blend"], g["fwd_ret"]).correlation
        if pd.notna(ic):
            ics.append(ic)
    pm = pd.DataFrame(rows)
    ics = np.array(ics)
    return {
        "n_months": len(pm),
        "top20_excess": (pm["top_ret"] - pm["med_ret"]).mean(),
        "win_rate": (pm["top_ret"] > pm["med_ret"]).mean(),
        "ic_mean": ics.mean() if len(ics) else np.nan,
        "icir": _icir(ics) if len(ics) > 1 else np.nan,
    }


def main() -> None:
    if not CACHE.exists():
        logger.error("Cache not found: {}. Run blend_weight_sweep.py first.", CACHE)
        sys.exit(1)

    diag = pd.read_parquet(CACHE)
    diag["score_date"] = pd.to_datetime(diag["score_date"])
    if "primary_score" not in diag.columns:
        logger.error("Cache missing primary_score — rerun blend_weight_sweep.py")
        sys.exit(1)

    diag["regime"] = diag["score_date"].apply(_classify_regime)
    regime_counts = diag.groupby("regime")["score_date"].nunique()
    logger.info("Regime month counts:\n{}", regime_counts)

    regimes_to_test = list(REGIMES.keys()) + ["其他 (Other)"]
    print("\n## Per-regime weight sweep (PIT-clean)\n")
    for regime in regimes_to_test:
        sub = diag[diag["regime"] == regime]
        if sub["score_date"].nunique() < 3:
            continue
        print(f"\n### {regime}  ({sub['score_date'].nunique()} months, {len(sub):,} rows)\n")
        print("| w | top20 excess | 胜率 | IC mean | ICIR |")
        print("|---:|---:|---:|---:|---:|")
        for w in WEIGHTS:
            m = _metrics(sub, w)
            print(f"| {w:.2f} | {m['top20_excess']:+.2%} | {m['win_rate']:.1%} "
                  f"| {m['ic_mean']:+.4f} | {m['icir']:+.3f} |")

    # Also compute "extreme-only" baseline (w=0.0) for each regime
    print("\n## Extreme-only baseline per regime (w=0.0)\n")
    print("| regime | n_months | top20 excess | 胜率 | ICIR |")
    print("|---|---:|---:|---:|---:|")
    for regime in regimes_to_test:
        sub = diag[diag["regime"] == regime]
        if sub["score_date"].nunique() < 3:
            continue
        m = _metrics(sub, 0.0)
        print(f"| {regime} | {m['n_months']} | {m['top20_excess']:+.2%} "
              f"| {m['win_rate']:.1%} | {m['icir']:+.3f} |")

    # Save report
    out = Path("data/reports/blend_regime_sweep.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# Per-regime BlendRanker weight sweep\n\n")
        f.write("Regime labels from `prediction_diagnostics.REGIMES`. Metrics "
                "computed post-hoc from cached per-row component scores (PIT-clean).\n\n")
        for regime in regimes_to_test:
            sub = diag[diag["regime"] == regime]
            if sub["score_date"].nunique() < 3:
                continue
            f.write(f"## {regime}  ({sub['score_date'].nunique()} months)\n\n")
            f.write("| w | top20 excess | 胜率 | IC mean | ICIR |\n")
            f.write("|---:|---:|---:|---:|---:|\n")
            for w in WEIGHTS:
                m = _metrics(sub, w)
                f.write(f"| {w:.2f} | {m['top20_excess']:+.2%} | {m['win_rate']:.1%} "
                        f"| {m['ic_mean']:+.4f} | {m['icir']:+.3f} |\n")
            f.write("\n")
        f.write("## Extreme-only baseline (w=0.0)\n\n")
        f.write("| regime | n_months | top20 excess | 胜率 | ICIR |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for regime in regimes_to_test:
            sub = diag[diag["regime"] == regime]
            if sub["score_date"].nunique() < 3:
                continue
            m = _metrics(sub, 0.0)
            f.write(f"| {regime} | {m['n_months']} | {m['top20_excess']:+.2%} "
                    f"| {m['win_rate']:.1%} | {m['icir']:+.3f} |\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
