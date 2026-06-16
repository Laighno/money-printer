"""Refresh the n2c (next_open_to_close) walk-forward factor cache to the
latest bars, reusing the WALK-FORWARD richer feature path.

Round 268 (advisor, auto-retrain pipeline stage 1; user 拍 B):

WHY: prod blend models were trained 6/2 (label through ~5/5) and never auto-
retrain. The walk-forward cache
  data/wf_cache/factors_label_next_open_to_close.parquet
is stale (last date 5/29) while bars DB is fresh to 6/12. A fresh n2c cache is
the prerequisite for retraining blend_primary/extreme to a newer label window.

CRITICAL — avoid the build_dataset trap (advisor round 264):
  prod n2c models MUST be trained from the WALK-FORWARD `_build_factor_panel`
  feature path (richer, 68 cols), NOT `build_dataset` (simplified 64 cols).
  A trial that used build_dataset produced IC 0.001 (broken) because the
  feature set differed. This script reuses walk_forward_backtest's own
  `_load_or_build_factors` so the feature set is identical by construction.

HOW:
  1. Set LABEL_KIND=next_open_to_close (env, BEFORE importing WF) so the cache
     key resolves to factors_label_next_open_to_close.parquet.
  2. Back up the stale cache → .pre_refresh_<date>.
  3. Move the stale cache aside so `_load_or_build_factors` rebuilds + rewrites
     it fresh (it loads-if-exists, else builds + to_parquet, WF lines 298-427).
  4. Build the universe (PIT constituents) + bars_map exactly as WF does, call
     `_load_or_build_factors` → fresh panel written to the cache path.
  5. SANITY GATES (advisor 267/268 — the most important step; advisor栽过):
     a. col-set equality: fresh cols == stale cols (set equal). ANY diff →
        abort + restore stale. This catches feature drift at the dataset layer,
        before it ever reaches IC.
     b. row magnitude: fresh rows ≥ stale rows (should add new dates, not lose
        half the panel).
     c. date coverage: fresh max date advanced past stale max date.
  6. On any sanity failure: restore the stale cache from backup, exit non-zero.

This script does NOT touch prod models — it only rebuilds the training cache.
Training + verify + swap are separate scripts (train_blend_cutoff.py,
verify_retrain_quality.py, swap_model.py).

Usage:
  python -m scripts.refresh_n2c_cache [--dry-run]

  --dry-run : build into a temp path + run sanity gates, but do NOT overwrite
              the production cache (for testing the pipeline safely).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# LABEL_KIND must be set BEFORE importing walk_forward_backtest — it is read at
# module-import time (walk_forward_backtest.py:134) to branch the cache key.
os.environ.setdefault("LABEL_KIND", "next_open_to_close")

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "wf_cache" / "factors_label_next_open_to_close.parquet"
# The factor panel is built from the bars cache (data/wf_cache/bars.parquet),
# which is ALSO stale. _load_or_fetch_bars loads it if present (WF lines
# 257-266), so a fresh factor rebuild needs a fresh bars cache too — otherwise
# the panel rebuilds from stale bars and the date never advances (round 268
# dry-run #2 caught exactly this). We move the bars cache aside so
# _load_or_fetch_bars refetches from the (DB-first, fresh-to-today) source.
BARS_CACHE = ROOT / "data" / "wf_cache" / "bars.parquet"


def _build_universe_and_bars():
    """Replicate walk_forward_backtest.run_walk_forward()'s universe + bars
    setup (PIT constituents union, with current-members fallback)."""
    import scripts.walk_forward_backtest as wf

    assert wf.LABEL_KIND == "next_open_to_close", (
        f"LABEL_KIND resolved to {wf.LABEL_KIND!r}, expected next_open_to_close "
        "— env must be set before importing walk_forward_backtest"
    )

    # Universe: union of all PIT constituent snapshots (WF lines 871-889).
    try:
        from mp.data.store import DataStore as _DS
        _store = _DS()
        _union, _snap_dates = wf._merged_all_snapshots(_store)
        if len(_snap_dates) >= 2 and _union:
            codes = _union
            logger.info("Universe: {} stocks across {} snapshots ({} → {})",
                        len(codes), len(_snap_dates), _snap_dates[0], _snap_dates[-1])
        else:
            codes = wf._merged_current()
            logger.warning("Only {} snapshot(s); using current constituents "
                           "({} stocks) — SURVIVORSHIP BIAS", len(_snap_dates), len(codes))
    except Exception as e:
        codes = wf._merged_current()
        logger.warning("Snapshot lookup failed ({}); using current constituents "
                       "({} stocks)", e, len(codes))

    bars_map = wf._load_or_fetch_bars(codes)
    return wf, codes, bars_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="build + sanity-check but do NOT overwrite prod cache")
    args = parser.parse_args()

    t0 = time.time()

    if not CACHE.exists():
        logger.error("Stale n2c cache missing: {} — nothing to refresh against "
                     "(no col-set baseline). Aborting.", CACHE)
        return 2

    # Snapshot the stale cache's schema + stats BEFORE touching anything.
    stale = pd.read_parquet(CACHE)
    stale["date"] = pd.to_datetime(stale["date"])
    stale_cols = set(stale.columns)
    stale_rows = len(stale)
    stale_max_date = stale["date"].max()
    logger.info("Stale cache: {} rows, {} cols, date → {}",
                stale_rows, len(stale_cols), stale_max_date.date())

    # Back up the stale cache (timestamped — never overwrite a prior backup).
    stamp = stale_max_date.strftime("%Y%m%d")
    backup = CACHE.with_suffix(f".parquet.pre_refresh_{stamp}")
    if backup.exists():
        # Avoid clobbering an earlier same-day backup; add a counter.
        n = 1
        while backup.with_suffix(f".parquet.pre_refresh_{stamp}_{n}").exists():
            n += 1
        backup = backup.with_suffix(f".parquet.pre_refresh_{stamp}_{n}")
    shutil.copy2(CACHE, backup)
    logger.info("Backed up stale cache → {}", backup.name)

    # Move BOTH the factor cache and the bars cache aside so the rebuild
    # refetches bars fresh (DB-first to today) then rebuilds factors from them.
    aside = CACHE.with_suffix(".parquet.refreshing")
    if aside.exists():
        aside.unlink()
    CACHE.rename(aside)
    logger.info("Moved stale factor cache aside → {}", aside.name)

    bars_aside = None
    if BARS_CACHE.exists():
        bars_aside = BARS_CACHE.with_suffix(".parquet.refreshing")
        if bars_aside.exists():
            bars_aside.unlink()
        BARS_CACHE.rename(bars_aside)
        logger.info("Moved stale bars cache aside → {} (will refetch)",
                    bars_aside.name)

    def _restore_stale():
        """Put the stale factor + bars caches back exactly as they were."""
        if CACHE.exists():
            CACHE.unlink()
        aside.rename(CACHE)
        if bars_aside is not None:
            if BARS_CACHE.exists():
                BARS_CACHE.unlink()
            bars_aside.rename(BARS_CACHE)

    fresh: pd.DataFrame | None = None
    try:
        wf, codes, bars_map = _build_universe_and_bars()
        logger.info("Rebuilding factor panel (richer _build_factor_panel path)...")
        # This builds the panel AND writes it to CACHE (WF lines 426-427).
        fresh = wf._load_or_build_factors(bars_map, codes)
        # Ensure excess_ret present (BlendRanker primary trains on it).
        if "excess_ret" not in fresh.columns and "fwd_ret" in fresh.columns:
            from mp.ml.dataset import add_excess_ret
            logger.info("Adding excess_ret column for BlendRanker compatibility...")
            fresh = add_excess_ret(fresh, horizon=wf.HORIZON)
            fresh.to_parquet(CACHE, index=False)
    except Exception as e:
        logger.error("Rebuild failed: {} — restoring stale caches", e)
        _restore_stale()
        return 3

    # ----- SANITY GATES (advisor 267/268) -----
    fresh["date"] = pd.to_datetime(fresh["date"])
    fresh_cols = set(fresh.columns)
    fresh_rows = len(fresh)
    fresh_max_date = fresh["date"].max()

    logger.info("Fresh cache: {} rows, {} cols, date → {}",
                fresh_rows, len(fresh_cols), fresh_max_date.date())

    failures = []

    # a) col-set equality (THE key gate — catches feature drift)
    if fresh_cols != stale_cols:
        only_new = fresh_cols - stale_cols
        only_old = stale_cols - fresh_cols
        failures.append(
            f"COL-SET MISMATCH: +{sorted(only_new)} / -{sorted(only_old)}"
        )

    # b) row magnitude (should grow, not collapse)
    if fresh_rows < stale_rows:
        failures.append(
            f"ROW SHRANK: fresh {fresh_rows} < stale {stale_rows} "
            f"(expected ≥; lost data?)"
        )

    # c) date coverage advanced
    if fresh_max_date <= stale_max_date:
        failures.append(
            f"DATE NOT ADVANCED: fresh {fresh_max_date.date()} ≤ "
            f"stale {stale_max_date.date()}"
        )

    if failures:
        logger.error("SANITY FAILED — restoring stale caches:")
        for f in failures:
            logger.error("  ✗ {}", f)
        _restore_stale()
        logger.error("Stale caches restored. Backup retained at {}", backup.name)
        return 4

    # All gates passed.
    logger.info("✓ SANITY PASSED:")
    logger.info("  ✓ col-set equal ({} cols)", len(fresh_cols))
    logger.info("  ✓ rows {} ≥ {} (Δ +{})", fresh_rows, stale_rows,
                fresh_rows - stale_rows)
    logger.info("  ✓ date advanced {} → {}", stale_max_date.date(),
                fresh_max_date.date())

    # Report the trainable cutoff (last date with a non-NaN n2c label).
    labeled = fresh.dropna(subset=["fwd_ret"])
    if not labeled.empty:
        cutoff = labeled["date"].max()
        logger.info("  trainable cutoff (last valid n2c label): {}", cutoff.date())

    if args.dry_run:
        logger.info("DRY-RUN: restoring stale caches (no prod overwrite)")
        _restore_stale()
        logger.info("Fresh build validated but NOT deployed (dry-run). "
                    "Backup at {}", backup.name)
    else:
        # Fresh factor + bars caches already written by the rebuild path.
        # Remove the aside copies (the timestamped backup is the rollback path).
        if aside.exists():
            aside.unlink()
        if bars_aside is not None and bars_aside.exists():
            bars_aside.unlink()
        logger.info("Fresh n2c cache deployed → {} (+ fresh bars cache)",
                    CACHE.name)

    logger.info("Total time: {:.1f}s", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
