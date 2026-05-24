"""Cross-sectional IC analysis on the full HS300 + ZZ500 universe.

Generates the data needed to refresh CURATED_COLUMNS — the "kept" feature
list used by the production model — on the new wider universe.

Methodology
-----------
For each trading date in the analysis window:
  1. Take all stocks with valid feature value and valid 20d forward excess
  2. Compute Spearman rank correlation between the feature and excess_ret
  3. That's the IC for that (factor, date)

Aggregate over dates:
  - mean(IC)     ← average predictive power
  - std(IC)      ← stability across regimes
  - ICIR = mean / std         ← risk-adjusted IC (standard definition)
  - t_stat = ICIR * sqrt(N)   ← significance of mean IC (reported but
                                 not used for selection)
  - pos_pct      ← fraction of dates where IC > 0

Threshold for CURATED inclusion: |ICIR| >= 0.15 (matches existing list).

Note: a prior version of this script defined ICIR as mean / std * sqrt(N),
which is actually t_stat. With N ~ 800, sqrt(N) ~ 28x inflated every
factor; all "ICIR" values produced before this fix should be treated as
t-stats, not ICIRs. The current script delegates to mp.ml.ic_utils.

Usage
-----
    python scripts/cross_sectional_ic.py
    python scripts/cross_sectional_ic.py --start 20220101 --output data/ic_curated.json
"""
from __future__ import annotations

import argparse
import json
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
    EXCESS_LABEL,
    FACTOR_COLUMNS,
    build_dataset,
)
from mp.ml.ic_utils import summarize_ic


def compute_cross_sectional_ic(
    panel: pd.DataFrame, feature: str, label: str = EXCESS_LABEL
) -> pd.Series:
    """Return time-series of cross-sectional IC for one feature.

    IC[t] = spearman_rank_corr( panel[t][feature], panel[t][label] )
    NaN values dropped per-date before correlation.
    """
    ic_per_date = {}
    for d, group in panel.groupby("date"):
        x = group[feature].to_numpy(dtype=float)
        y = group[label].to_numpy(dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 30:   # need enough cross-section
            continue
        try:
            ic, _ = spearmanr(x[mask], y[mask])
            if not np.isnan(ic):
                ic_per_date[d] = ic
        except Exception:
            pass
    return pd.Series(ic_per_date).sort_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="20220101",
                    help="dataset start date for IC computation (default: 20220101)")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="|ICIR| threshold for CURATED inclusion (default: 0.15)")
    ap.add_argument("--output", default="data/ic_curated.json",
                    help="output JSON with kept/dropped factor lists")
    args = ap.parse_args()

    logger.info("=== Cross-sectional IC analysis on HS300+ZZ500 universe ===")
    codes = get_recommendation_universe()
    logger.info("Universe: {} stocks (HS300 + ZZ500)", len(codes))

    t0 = time.time()
    panel = build_dataset(codes, args.start, horizon=20)
    if panel.empty:
        logger.error("Empty panel — aborting")
        return 1
    logger.info("Panel built: {:,} rows × {} factors in {:.0f}s",
                len(panel), len(FACTOR_COLUMNS), time.time() - t0)

    logger.info("Computing per-date cross-sectional IC for {} features...",
                len(FACTOR_COLUMNS))

    summary_rows = []
    t1 = time.time()
    for i, feat in enumerate(FACTOR_COLUMNS, 1):
        if feat not in panel.columns:
            logger.warning("Feature {} not in panel, skipping", feat)
            continue
        ic_series = compute_cross_sectional_ic(panel, feat)
        s = summarize_ic(ic_series)
        s["feature"] = feat
        summary_rows.append(s)
        if i % 10 == 0:
            logger.info("  progress {}/{}", i, len(FACTOR_COLUMNS))
    logger.info("IC computation done in {:.0f}s", time.time() - t1)

    df = pd.DataFrame(summary_rows)
    df = df.sort_values("icir", key=lambda s: -s.abs()).reset_index(drop=True)

    # Print report
    print()
    print(f"{'Rank':>4}  {'Feature':<28} {'Mean IC':>9} {'Std IC':>8} {'ICIR':>8} "
          f"{'t_stat':>8} {'Pos%':>7} {'N dates':>8}  Verdict")
    print("-" * 110)
    for i, r in df.iterrows():
        keep = abs(r["icir"]) >= args.threshold
        verdict = ("✓ KEEP " + ("STRONG" if abs(r["icir"]) >= 0.5
                                else "MODERATE" if abs(r["icir"]) >= 0.3
                                else "WEAK")) if keep else "✗ drop"
        print(f"  {i+1:>2}  {r['feature']:<28} {r['mean']:>+9.4f} {r['std']:>8.4f} "
              f"{r['icir']:>+8.2f} {r['t_stat']:>+8.2f} {r['pos_pct']*100:>6.1f}% "
              f"{r['n']:>8d}  {verdict}")

    keep = df[df["icir"].abs() >= args.threshold]["feature"].tolist()
    drop = df[df["icir"].abs() < args.threshold]["feature"].tolist()

    print()
    print(f"=== Summary: |ICIR| >= {args.threshold} ===")
    print(f"KEEP ({len(keep)} features):")
    for f in keep:
        print(f"    {f!r},")
    print(f"\nDROP ({len(drop)} features):")
    for f in drop:
        print(f"    {f!r},")

    # Compare against existing CURATED_COLUMNS
    from mp.ml.dataset import CURATED_COLUMNS
    cur_set, new_set = set(CURATED_COLUMNS), set(keep)
    print(f"\n=== Diff vs existing CURATED_COLUMNS ({len(CURATED_COLUMNS)} features) ===")
    added = new_set - cur_set
    removed = cur_set - new_set
    if added:
        print(f"NEW additions ({len(added)}):")
        for f in sorted(added):
            print(f"    + {f}")
    if removed:
        print(f"DROPPED from existing ({len(removed)}):")
        for f in sorted(removed):
            print(f"    - {f}")
    if not added and not removed:
        print("(no changes)")

    # Save JSON for downstream
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "computed_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "universe_size": len(codes),
        "panel_rows": int(len(panel)),
        "threshold": args.threshold,
        "kept_features": keep,
        "dropped_features": drop,
        "ic_summary": df.to_dict(orient="records"),
        "diff_vs_existing": {
            "added": sorted(added),
            "removed": sorted(removed),
        },
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                              default=lambda x: float(x) if hasattr(x, '__float__') else str(x)),
                   encoding="utf-8")
    logger.info("Saved → {}", out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
