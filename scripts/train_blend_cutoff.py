"""Train a single BlendRanker (primary + extreme) on daily data with
explicit train-cutoff date. round 189 (advisor round 188 spec).

WHY: Production `data/blend_*.lgb` was trained with no `--end` argument,
defaulting to "today". This makes any backtest covering 2025-09 ~ 2026-04
suffer **look-ahead bias** — the model has seen the test window's labels.

This script trains a SEPARATE blend with explicit cutoff so
walk_forward_dual_bucket can run TRUE out-of-sample backtest.

Usage:
    .venv/bin/python scripts/train_blend_cutoff.py \\
        --start 20200101 --end 20250831 \\
        --output-prefix data/blend_cutoff20250831

Output:
    {prefix}_primary.lgb
    {prefix}_extreme.lgb

Rule #4: this does NOT touch production data/blend_*.lgb — separate path.
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
    ap.add_argument("--start", default="20200101",
                    help="Training data start (default: 20200101)")
    ap.add_argument("--end", required=True,
                    help="Training data CUTOFF (inclusive, YYYYMMDD). "
                         "Test window must start the next trading day. "
                         "Required to prevent accidental look-ahead.")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--output-prefix", required=True,
                    help="Output path prefix, e.g. data/blend_cutoff20250831. "
                         "Files written: {prefix}_primary.lgb, {prefix}_extreme.lgb. "
                         "MUST NOT equal 'data/blend' (Rule #4 — no prod overwrite).")
    args = ap.parse_args()

    if args.output_prefix in ("data/blend", "data/blend_"):
        logger.error("Refusing to overwrite production data/blend_*.lgb "
                     "(Rule #4). Use a distinct --output-prefix.")
        return 1

    t0 = time.time()  # whole-run timer; must precede both panel paths (the
    # wf_cache fast path skips the build branch where t0 used to be set, which
    # left line "Total time" crashing on UnboundLocalError — 2026-06-24 fix).
    logger.info("=" * 60)
    logger.info("Training cutoff BlendRanker")
    logger.info("=" * 60)
    logger.info("  start  = {}", args.start)
    logger.info("  end    = {} (cutoff, inclusive)", args.end)
    logger.info("  output = {}_(primary|extreme).lgb", args.output_prefix)

    # round 189 fast path: try wf_cache panel (already pre-built by
    # walk_forward_backtest._load_or_build_factors). If horizon == 20 and
    # the cache covers the requested window, use it — saves 30-60 min of
    # per-stock build_dataset loop. Falls back to full build otherwise.
    CACHE = ROOT / "data" / "wf_cache" / "factors_label_next_open_to_close.parquet"
    panel = None
    if args.horizon == 20 and CACHE.exists():
        import pandas as pd
        logger.info("Trying wf_cache panel: {}", CACHE)
        cached = pd.read_parquet(CACHE)
        cached["date"] = pd.to_datetime(cached["date"])
        start_ts = pd.Timestamp(args.start)
        end_ts = pd.Timestamp(args.end)
        sub = cached[(cached["date"] >= start_ts) & (cached["date"] <= end_ts)]
        if not sub.empty:
            panel = sub.copy()
            logger.info("Loaded {} rows from wf_cache ({} → {}). "
                       "Skipping full build_dataset (saves ~30min).",
                       len(panel), sub["date"].min().date(), sub["date"].max().date())
        else:
            logger.warning("wf_cache empty for requested window, falling back to build_dataset")

    if panel is None:
        codes = get_recommendation_universe()
        logger.info("Universe: {} stocks (HS300 + ZZ500)", len(codes))
        t0 = time.time()
        panel = build_dataset(codes, args.start, end=args.end, horizon=args.horizon)
        logger.info("Panel built: {:,} rows in {:.0f}s", len(panel), time.time() - t0)

    if panel.empty:
        logger.error("Empty panel — cannot train")
        return 2

    # Defensive: drop index in case caller passed pre-grouped df
    if panel.index.name is not None or any(n for n in (panel.index.names or [])):
        panel = panel.reset_index(drop=True)

    # wf_cache panel ships with excess_ret all-NaN (computed lazily after load).
    # Compute cross-sectional excess (fwd_ret − per-date mean) — equivalent to
    # market-neutral fwd_ret. Slightly different from production's ZZ500-bench
    # excess but for cutoff backtest training, this is fine (signal is
    # cross-sectional rank order, not absolute level).
    if panel["excess_ret"].isna().all() and "fwd_ret" in panel.columns:
        logger.info("excess_ret all-NaN in cache; computing cross-sectional demean")
        sub = panel.dropna(subset=["fwd_ret"]).copy()
        sub["excess_ret"] = sub["fwd_ret"] - sub.groupby("date")["fwd_ret"].transform("mean")
        # Winsorize ±50% to match production's add_excess_ret behavior
        sub["excess_ret"] = sub["excess_ret"].clip(-0.5, 0.5)
        panel = sub
        logger.info("excess_ret cross-section: {} non-null (clipped ±50%)",
                   panel["excess_ret"].notna().sum())
    # Monkey-patch BlendRanker._filter_extremes to reset_index after groupby
    # (newer pandas leaves "date" in BOTH index and column after apply with
    # group_keys=False, which trips StockRanker.train_fast on `sort_values("date")`).
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

    # Save to specified prefix
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)
    br.save(args.output_prefix)
    logger.info("Saved → {}_(primary|extreme).lgb", args.output_prefix)

    logger.info("Total time: {:.1f} min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
