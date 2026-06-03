"""Refresh c2c factor cache by reusing n2c cache features (fresh through 5/29)
and recomputing c2c fwd_ret from bars.

Round 209 (advisor, after round 208 IC/limit/protect 诊断, user 拍 A 路径):

WHY: data/wf_cache/factors.parquet (c2c cache, Apr 28 build) is 5 weeks stale
relative to data/wf_cache/factors_label_next_open_to_close.parquet (n2c, Jun 1).
Both share 67 feature columns (features depend on bars, not on label). Only
fwd_ret differs (n2c = next_open_to_close, c2c = close[i+H]/close[i]-1).

Output: data/wf_cache/factors_c2c_FRESH_v2.parquet (new cache, does NOT
overwrite the original — preserves rollback path).

After:
  1. Train: scripts/train_blend_c2c_cutoff.py — but pointing to the NEW cache
  2. WF validate: walk_forward_dual_bucket with the v2 model
  3. If pass: swap to prod intraday_blend
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
N2C_CACHE = ROOT / "data" / "wf_cache" / "factors_label_next_open_to_close.parquet"
BARS_CACHE = ROOT / "data" / "wf_cache" / "bars.parquet"
OUT_CACHE = ROOT / "data" / "wf_cache" / "factors_c2c_FRESH_v2.parquet"
HORIZON = 20


def main() -> int:
    if not N2C_CACHE.exists():
        logger.error(f"n2c cache missing: {N2C_CACHE}")
        return 2

    t0 = time.time()
    logger.info(f"Loading n2c cache features: {N2C_CACHE}")
    df = pd.read_parquet(N2C_CACHE)
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    logger.info(f"  {len(df):,} rows, date range {df.date.min().date()} → {df.date.max().date()}")

    logger.info(f"Loading bars cache: {BARS_CACHE}")
    bars = pd.read_parquet(BARS_CACHE)
    bars["date"] = pd.to_datetime(bars["date"])
    bars["code"] = bars["code"].astype(str).str.zfill(6)
    bars = bars[["code", "date", "close"]].sort_values(["code", "date"])
    logger.info(f"  bars max date: {bars.date.max().date()}")

    logger.info("Computing c2c fwd_ret per (code, date)")
    # For each code, shift close by -HORIZON days and compute c2c return
    bars["close_fwd"] = bars.groupby("code")["close"].shift(-HORIZON)
    bars["fwd_ret_c2c"] = (bars["close_fwd"] / bars["close"] - 1.0).where(
        (bars["close"] > 0) & bars["close_fwd"].notna()
    )
    fwd_map = bars[["code", "date", "fwd_ret_c2c"]].dropna()
    logger.info(f"  c2c fwd_ret valid rows: {len(fwd_map):,}")

    logger.info("Merging n2c features + c2c fwd_ret")
    # Drop old fwd_ret + excess_ret (they're n2c)
    df_features = df.drop(columns=["fwd_ret", "excess_ret"], errors="ignore")
    merged = df_features.merge(fwd_map, on=["code", "date"], how="left")
    merged = merged.rename(columns={"fwd_ret_c2c": "fwd_ret"})

    # Compute cross-section excess (same as train_blend_c2c_cutoff.py path)
    logger.info("Computing cross-sectional excess_ret + winsorize ±50%")
    valid = merged.dropna(subset=["fwd_ret"]).copy()
    valid["excess_ret"] = valid["fwd_ret"] - valid.groupby("date")["fwd_ret"].transform("mean")
    valid["excess_ret"] = valid["excess_ret"].clip(-0.5, 0.5)
    logger.info(f"  excess_ret rows: {len(valid):,}, mean={valid.excess_ret.mean():.5f}")

    # Final cache: valid rows only (drops tail HORIZON days per code)
    OUT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    valid.to_parquet(OUT_CACHE, index=False)
    logger.info(f"Saved → {OUT_CACHE}")
    logger.info(f"  final shape: {valid.shape}, max date: {valid.date.max().date()}")
    logger.info(f"  vs old c2c cache (data/wf_cache/factors.parquet) max date: 2026-04-28")
    logger.info(f"  total time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
