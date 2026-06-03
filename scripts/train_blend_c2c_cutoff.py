"""Train a c2c-labeled BlendRanker with explicit train-cutoff date.
round 205-B (advisor, user-driven A/B with prod n2c at same cutoff).

WHY: `train_blend_cutoff.py` uses the n2c cache
(`factors_label_next_open_to_close.parquet`). To compare label methods at
the same train cutoff we need an equivalent c2c-labeled panel.

The older `factors.parquet` (Apr 28, predates round 162 n2c upgrade on
5/31) has c2c label (`fwd_ret[i] = close[i+20] / close[i] - 1`, see
mp/ml/dataset.py:794). Window 2020-01-01 → 2025-08-31 coverage equals
the n2c cache (1.30M rows each). Universe and feature schema identical.

Usage:
    .venv/bin/python scripts/train_blend_c2c_cutoff.py \\
        --start 20200101 --end 20250831 \\
        --output-prefix data/blend_c2c_cutoff20250831

Output:
    {prefix}_primary.lgb
    {prefix}_extreme.lgb

Rule #4: does NOT touch production data/blend_*.lgb — separate path.
Rule #11 PIT: train data ≤ --end strictly; test window starts > --end.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mp.data.fetcher import get_recommendation_universe  # noqa: E402
from mp.ml.dataset import build_dataset  # noqa: E402
from mp.ml.model import BlendRanker  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="20200101")
    ap.add_argument("--end", required=True,
                    help="Training data CUTOFF (inclusive, YYYYMMDD).")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--output-prefix", required=True,
                    help="e.g. data/blend_c2c_cutoff20250831. "
                         "MUST NOT equal 'data/blend' (Rule #4).")
    ap.add_argument("--cache-path",
                    default="data/wf_cache/factors.parquet",
                    help="Path to c2c factor cache (default: factors.parquet "
                         "= Apr 28 stale build). Use "
                         "data/wf_cache/factors_c2c_FRESH_v2.parquet for "
                         "round 209 fresh rebuild.")
    args = ap.parse_args()

    if args.output_prefix in ("data/blend", "data/blend_"):
        logger.error("Refusing to overwrite production data/blend_*.lgb (Rule #4).")
        return 1

    logger.info("=" * 60)
    logger.info("Training c2c cutoff BlendRanker")
    logger.info("=" * 60)
    logger.info("  start  = {}", args.start)
    logger.info("  end    = {} (cutoff, inclusive)", args.end)
    logger.info("  label  = c2c (close[i+H] / close[i] - 1)")
    logger.info("  output = {}_(primary|extreme).lgb", args.output_prefix)

    # c2c cache (default: Apr 28 stale dump; round 209 use FRESH v2 via --cache-path)
    CACHE = ROOT / args.cache_path
    panel = None
    t0 = time.time()
    if args.horizon == 20 and CACHE.exists():
        import pandas as pd
        logger.info("Loading c2c cache: {}", CACHE)
        cached = pd.read_parquet(CACHE)
        cached["date"] = pd.to_datetime(cached["date"])
        start_ts = pd.Timestamp(args.start)
        end_ts = pd.Timestamp(args.end)
        sub = cached[(cached["date"] >= start_ts) & (cached["date"] <= end_ts)]
        if not sub.empty:
            panel = sub.copy()
            logger.info("Loaded {} rows from c2c cache ({} → {})",
                       len(panel), sub["date"].min().date(), sub["date"].max().date())
        else:
            logger.warning("c2c cache empty for requested window")

    if panel is None:
        codes = get_recommendation_universe()
        logger.info("Universe: {} stocks (HS300 + ZZ500)", len(codes))
        panel = build_dataset(codes, args.start, end=args.end, horizon=args.horizon)
        logger.info("Panel built: {:,} rows in {:.0f}s", len(panel), time.time() - t0)

    if panel.empty:
        logger.error("Empty panel — cannot train")
        return 2

    if panel.index.name is not None or any(n for n in (panel.index.names or [])):
        panel = panel.reset_index(drop=True)

    # factors.parquet (c2c cache) does NOT have excess_ret column.
    # Compute cross-sectional excess (fwd_ret - per-date mean) + winsorize ±50%
    # — identical post-processing to train_blend_cutoff.py for n2c.
    if "excess_ret" not in panel.columns or panel["excess_ret"].isna().all():
        if "fwd_ret" not in panel.columns:
            logger.error("Cache missing fwd_ret column")
            return 3
        logger.info("Computing excess_ret = fwd_ret - cross-section mean")
        sub = panel.dropna(subset=["fwd_ret"]).copy()
        sub["excess_ret"] = sub["fwd_ret"] - sub.groupby("date")["fwd_ret"].transform("mean")
        sub["excess_ret"] = sub["excess_ret"].clip(-0.5, 0.5)
        panel = sub
        logger.info("excess_ret: {:,} non-null (clipped ±50%)",
                   panel["excess_ret"].notna().sum())

    # Same monkey-patch as train_blend_cutoff.py (pandas group_keys quirk)
    _orig_filter = BlendRanker._filter_extremes
    def _patched_filter(self, df):
        out = _orig_filter(self, df)
        return out.reset_index(drop=True)
    BlendRanker._filter_extremes = _patched_filter

    t1 = time.time()
    br = BlendRanker()
    metrics = br.train_fast(panel)
    logger.info("BlendRanker trained in {:.1f}s", time.time() - t1)
    logger.info("  primary IC = {:.3f}", metrics.get("primary", {}).get("ic", 0))
    logger.info("  extreme IC = {:.3f}", metrics.get("extreme", {}).get("ic", 0))

    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)
    br.save(args.output_prefix)
    logger.info("Saved → {}_(primary|extreme).lgb", args.output_prefix)
    logger.info("Total time: {:.1f} min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
