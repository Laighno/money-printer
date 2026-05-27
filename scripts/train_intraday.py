"""P11-2: Train intraday BlendRanker on the 68-feature schema.

P11 chain (docs/dialog rounds 73 + 75 + decision_log P11 entry): re-predict
universe at T 14:30 using intraday-aware features. This script is the
**P11-2 training step**: it takes the round-75 schema (INTRADAY_FEATURE_COLS
= FACTOR_COLUMNS + 4 morning extras = 68 cols), constructs a training
panel via EOD-proxy, trains a parallel BlendRanker, and saves to
``data/intraday_blend_*.lgb`` — strictly disjoint from production
``data/blend_*.lgb`` (Rule #4 sacrosanct).

Why EOD-proxy (round 73 supplement)
-----------------------------------
xtdata's 1m history is only available on the ECS Windows QMT install. From
the Mac engineer side we have only EOD daily bars (via the existing
``get_daily_bars``). Per the round-73-supplement spec ("拿不到的价格按
收盘价算, 交易额估个百分比"), each historical row gets a synthetic 14:30
bar derived from its EOD OHLCV with fudge factors:

  - ``overnight_gap``    = (T_open - T-1_close) / T-1_close   — CLEAN, no proxy
                           (open is known at 9:30 sharp; computed from real EOD bars)
  - ``morning_return``   = (T_close - T_open) / T_open × 0.85 — PROXY
                           (assume ~85% of move happens by 14:30)
  - ``morning_vwap_dev`` = (T_close - daily_VWAP) / daily_VWAP — PROXY
                           (daily_VWAP = amount / volume is full-day VWAP, not just morning)
  - ``morning_vol_ratio``= (T_volume × 0.75) / 20d_volume_MA — PROXY
                           (~75% of volume by 14:30)

3 of 4 extras are proxied — this is documented in ``data_quality`` stats
emitted by the training run. P11-3 walk_forward will surface any
distribution shift between proxy training and (future) real intraday
inference. P11-4 will swap proxy for real xtdata on ECS and retrain.

Label
-----
Same 20d ``excess_ret`` label as production (T close → T+20 close vs ZZ500).
We do NOT shift to a 19d / T 14:30 label here — that would require
recomputing benchmarks and complicate A/B against production. The P11-2
model is "production blend + 4 intraday extras"; P11-3 walk-forward will
isolate whether the extras add Sharpe.

Output
------
  data/intraday_blend_primary.lgb   # main excess_ret regressor
  data/intraday_blend_extreme.lgb   # top/bottom 30% regressor

Usage
-----
  # Full universe (~800 codes), 5 years, default seed 42
  .venv/bin/python scripts/train_intraday.py

  # Smoke (3 stocks, 1 year — verify pipeline only)
  .venv/bin/python scripts/train_intraday.py --smoke

  # Custom universe / dates / seed
  LGBM_SEED=43 .venv/bin/python scripts/train_intraday.py --start 20230101
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mp.data.fetcher import get_daily_bars, get_recommendation_universe  # noqa: E402
from mp.ml.dataset import FACTOR_COLUMNS, build_dataset  # noqa: E402
from mp.ml.intraday_features import (  # noqa: E402
    INTRADAY_EXTRA_COLUMNS,
    INTRADAY_FEATURE_COLS,
)
from mp.ml.model import BlendRanker  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# EOD-proxy fudge factors (round-73-supplement spec)
# ─────────────────────────────────────────────────────────────────────
# Assume ~85% of price move by 14:30 (last 90 min of session typically thinner).
MORNING_RETURN_SCALE = 0.85
# Assume ~75% of volume by 14:30 (240 trading min total, 180 min by 14:30 = 75%).
MORNING_VOLUME_SCALE = 0.75


def compute_extras_for_panel(
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the 4 intraday extras for every row of one stock's EOD panel.

    Parameters
    ----------
    bars : DataFrame with columns ``date, open, high, low, close, volume, amount``
        Sorted by date ascending.

    Returns
    -------
    DataFrame with columns ``date`` + INTRADAY_EXTRA_COLUMNS
        Same row count as input; first row's overnight_gap is NaN (no T-1).
        First 20 rows' morning_vol_ratio is NaN (warm-up).
    """
    n = len(bars)
    open_arr = bars["open"].to_numpy(dtype=float)
    close_arr = bars["close"].to_numpy(dtype=float)
    volume_arr = bars["volume"].to_numpy(dtype=float)
    amount_arr = bars["amount"].to_numpy(dtype=float) if "amount" in bars.columns else volume_arr * close_arr

    # overnight_gap — CLEAN, no proxy
    overnight = np.full(n, np.nan)
    for i in range(1, n):
        prev_close = close_arr[i - 1]
        if prev_close > 0 and open_arr[i] > 0:
            overnight[i] = (open_arr[i] - prev_close) / prev_close

    # morning_return — PROXY: 0.85 × full-day return
    morning_ret = np.full(n, np.nan)
    valid = (open_arr > 0) & (close_arr > 0)
    morning_ret[valid] = MORNING_RETURN_SCALE * (close_arr[valid] - open_arr[valid]) / open_arr[valid]

    # morning_vwap_dev — PROXY: close vs daily VWAP
    # daily_VWAP = amount / volume (full-day VWAP, not just morning)
    vwap_dev = np.full(n, np.nan)
    safe = (volume_arr > 0) & (amount_arr > 0) & (close_arr > 0)
    vwap_proxy = np.where(safe, amount_arr / volume_arr, np.nan)
    safe_vwap = safe & (vwap_proxy > 0)
    vwap_dev[safe_vwap] = (close_arr[safe_vwap] - vwap_proxy[safe_vwap]) / vwap_proxy[safe_vwap]

    # morning_vol_ratio — PROXY: 0.75 × T_volume / 20d EOD MA
    vol_ratio = np.full(n, np.nan)
    if n >= 20:
        # Rolling 20-period mean of volume (exclude current day to mirror inference)
        rolling = pd.Series(volume_arr).rolling(window=20, min_periods=20).mean().shift(1).to_numpy()
        valid_vr = (rolling > 0) & (volume_arr > 0)
        vol_ratio[valid_vr] = MORNING_VOLUME_SCALE * volume_arr[valid_vr] / rolling[valid_vr]

    return pd.DataFrame({
        "date": bars["date"].values,
        "overnight_gap": overnight,
        "morning_return": morning_ret,
        "morning_vwap_dev": vwap_dev,
        "morning_vol_ratio": vol_ratio,
    })


def attach_intraday_extras(
    panel: pd.DataFrame,
    codes: List[str],
    start: str,
    end: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Attach 4 intraday extras to an EOD panel by re-fetching OHLCV per code.

    Returns
    -------
    (panel_with_extras, stats)
        stats keys: ``rows_total``, ``rows_with_overnight``, ``rows_with_morning``,
                    ``rows_with_vol_ratio``, ``codes_processed``, ``codes_failed``.
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    extras_frames: List[pd.DataFrame] = []
    n_codes = len(codes)
    stats = {"codes_processed": 0, "codes_failed": 0}

    for i, code in enumerate(codes):
        try:
            bars = get_daily_bars(code, start, end)
        except Exception as e:
            logger.debug("get_daily_bars({}) failed: {}", code, e)
            stats["codes_failed"] += 1
            continue

        if bars is None or bars.empty:
            stats["codes_failed"] += 1
            continue

        bars = bars.sort_values("date").reset_index(drop=True)
        bars["date"] = pd.to_datetime(bars["date"])

        extras = compute_extras_for_panel(bars)
        extras["code"] = code
        extras_frames.append(extras[["code", "date"] + INTRADAY_EXTRA_COLUMNS])
        stats["codes_processed"] += 1

        if (i + 1) % 100 == 0 or (i + 1) == n_codes:
            logger.info("attach_intraday_extras progress: {}/{} ({} ok / {} failed)",
                        i + 1, n_codes, stats["codes_processed"], stats["codes_failed"])

    if not extras_frames:
        logger.error("attach_intraday_extras: no extras computed")
        for col in INTRADAY_EXTRA_COLUMNS:
            panel[col] = np.nan
        return panel, {**stats, "rows_total": len(panel)}

    extras_all = pd.concat(extras_frames, ignore_index=True)
    merged = panel.merge(extras_all, on=["code", "date"], how="left")

    stats["rows_total"] = len(merged)
    for col in INTRADAY_EXTRA_COLUMNS:
        stats[f"rows_with_{col}"] = int(merged[col].notna().sum())

    return merged, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="20200101",
                    help="Training data start (default: 20200101)")
    ap.add_argument("--end", default=None,
                    help="Training data end (default: today)")
    ap.add_argument("--horizon", type=int, default=20,
                    help="Forward-return horizon in trading days (default: 20)")
    ap.add_argument("--output-prefix", default="data/intraday_blend",
                    help="Output path prefix (default: data/intraday_blend → "
                         "data/intraday_blend_primary.lgb + data/intraday_blend_extreme.lgb)")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke run: 3 stocks × 1 year only, verify pipeline; do NOT overwrite production-style artifacts.")
    args = ap.parse_args()

    # Rule #4 guardrail: refuse to overwrite production blend
    if "intraday" not in Path(args.output_prefix).name:
        logger.error("Output prefix '{}' does not contain 'intraday' — refusing to "
                     "overwrite production blend (Rule #4).", args.output_prefix)
        return 1

    t_start = time.time()
    if args.smoke:
        codes = ["000001", "600000", "600036"]  # 3 well-known A-share codes
        start = "20240101"
        logger.info("=== SMOKE RUN: {} codes × 1 year, output → {}_smoke_*.lgb ===",
                    len(codes), args.output_prefix)
        output_prefix = args.output_prefix + "_smoke"
    else:
        codes = get_recommendation_universe()
        start = args.start
        logger.info("=== Full intraday training: {} codes, start={}, horizon={} → {} ===",
                    len(codes), start, args.horizon, args.output_prefix)
        output_prefix = args.output_prefix

    # 1. Build EOD base panel
    t0 = time.time()
    eod_panel = build_dataset(codes, start, args.end, horizon=args.horizon)
    if eod_panel.empty:
        logger.error("Empty EOD panel — aborting")
        return 2
    logger.info("EOD panel: {:,} rows × {} cols in {:.0f}s",
                len(eod_panel), len(eod_panel.columns), time.time() - t0)

    # 2. Attach 4 intraday extras (EOD-proxy)
    t0 = time.time()
    panel, dq_stats = attach_intraday_extras(eod_panel, codes, start, args.end)
    logger.info("Intraday extras attached in {:.0f}s. Data quality:", time.time() - t0)
    for k, v in dq_stats.items():
        if k.startswith("rows_with_"):
            pct = 100.0 * v / max(dq_stats["rows_total"], 1)
            logger.info("  {}: {} ({:.1f}%)", k, v, pct)
        else:
            logger.info("  {}: {}", k, v)

    # 3. Verify schema
    missing = set(INTRADAY_FEATURE_COLS) - set(panel.columns)
    if missing:
        logger.error("Panel missing {} INTRADAY_FEATURE_COLS: {}", len(missing), sorted(missing)[:10])
        return 3

    # 4. Train BlendRanker on the 68-feature schema
    logger.info("Training BlendRanker(feature_cols=INTRADAY_FEATURE_COLS, {} cols)...",
                len(INTRADAY_FEATURE_COLS))
    seed = int(os.environ.get("LGBM_SEED", "42"))
    logger.info("LGBM_SEED={} (override via env)", seed)
    t0 = time.time()
    ranker = BlendRanker(feature_cols=INTRADAY_FEATURE_COLS)
    metrics = ranker.train_fast(panel)
    logger.info("Training complete in {:.1f}s", time.time() - t0)
    logger.info("  primary IC = {:.4f}", metrics.get("primary", {}).get("ic", float("nan")))
    logger.info("  primary MAE = {:.4f}", metrics.get("primary", {}).get("mae", float("nan")))
    logger.info("  extreme IC = {:.4f}", metrics.get("extreme", {}).get("ic", float("nan")))
    logger.info("  extreme MAE = {:.4f}", metrics.get("extreme", {}).get("mae", float("nan")))

    # 5. Save (Rule #4: guard already checked above)
    ranker.save(output_prefix)
    primary_path = Path(f"{output_prefix}_primary.lgb")
    extreme_path = Path(f"{output_prefix}_extreme.lgb")
    primary_size = primary_path.stat().st_size if primary_path.exists() else 0
    extreme_size = extreme_path.stat().st_size if extreme_path.exists() else 0
    logger.info("Saved artifacts:")
    logger.info("  {} ({:,} bytes)", primary_path, primary_size)
    logger.info("  {} ({:,} bytes)", extreme_path, extreme_size)

    logger.info("=" * 60)
    logger.info("Total time: {:.1f} min", (time.time() - t_start) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
