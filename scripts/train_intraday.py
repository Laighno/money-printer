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

# ─────────────────────────────────────────────────────────────────────
# P11-4 round 91: hybrid training — real intraday 9 months + EOD-proxy for older history
# ─────────────────────────────────────────────────────────────────────
INTRADAY_1M_DIR = Path("data/intraday_1m")
INTRADAY_REAL_START = pd.Timestamp("2025-09-01")  # xtquant QMT 1m history floor (round 89)


def load_intraday_1m(dir_path: Path = INTRADAY_1M_DIR) -> pd.DataFrame:
    """Load all available 1m parquet files into a single DataFrame.

    Returns long-form DataFrame with columns ``code, datetime, open, high, low, close, volume``.
    Empty DataFrame if directory missing or empty.

    P11-4 round 91: ~30M rows / ~145MB across 8 monthly partitions
    (2025-09 → 2026-04). Fully fits in memory.
    """
    if not dir_path.exists():
        logger.warning("intraday_1m directory not found: {}", dir_path)
        return pd.DataFrame()
    parts = sorted(dir_path.glob("*.parquet"))
    if not parts:
        logger.warning("intraday_1m directory empty: {}", dir_path)
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.normalize()
    df = df.sort_values(["code", "datetime"]).reset_index(drop=True)
    logger.info("Loaded intraday_1m: {} rows, {} codes, {} dates from {} files",
                len(df), df["code"].nunique(), df["date"].nunique(), len(parts))
    return df


def _real_morning_extras_per_code(
    intraday_code_df: pd.DataFrame,
    eod_volume_ma20: pd.Series,
) -> pd.DataFrame:
    """Compute real morning_return / morning_vwap_dev / morning_vol_ratio per date
    for one stock, from its 1m intraday data.

    Parameters
    ----------
    intraday_code_df : DataFrame for ONE code, columns code, datetime, date, OHLCV
        Must already be filtered to 9:30 ≤ time < 14:30 (matches fetch script PIT).
    eod_volume_ma20 : Series indexed by date — 20-day MA of EOD daily volume (qfq).

    Returns
    -------
    DataFrame indexed by date with columns ``morning_return, morning_vwap_dev, morning_vol_ratio``.
    """
    grouped = intraday_code_df.groupby("date")
    # Per-day aggregates
    agg = grouped.agg(
        open_at_930=("open", "first"),
        close_at_1429=("close", "last"),
        sum_pv=("volume", lambda s: (s * intraday_code_df.loc[s.index, "close"]).sum()),
        sum_v=("volume", "sum"),
    )
    # morning_return = (close@14:29 / open@9:30) - 1
    agg["morning_return"] = agg["close_at_1429"] / agg["open_at_930"] - 1.0
    # morning_vwap = sum(close × volume) / sum(volume)
    morning_vwap = agg["sum_pv"] / agg["sum_v"].replace(0, np.nan)
    agg["morning_vwap_dev"] = (agg["close_at_1429"] - morning_vwap) / morning_vwap
    # morning_vol_ratio = morning_volume / 20d EOD volume MA
    # NOTE qfq alignment caveat (round 88 Note 2): 1m volume is RAW, EOD MA is qfq.
    # For stocks with splits/dividends in the 20d lookback the ratio will be biased.
    # Bounded effect — most stocks have no split events in any given 20d window.
    eod_aligned = eod_volume_ma20.reindex(agg.index)
    agg["morning_vol_ratio"] = agg["sum_v"] / eod_aligned.replace(0, np.nan)
    return agg[["morning_return", "morning_vwap_dev", "morning_vol_ratio"]]


def compute_extras_for_panel_hybrid(
    bars: pd.DataFrame,
    real_extras_by_date: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Hybrid version of compute_extras_for_panel.

    For dates ≥ INTRADAY_REAL_START where ``real_extras_by_date`` provides values,
    use real intraday morning features.  Otherwise fall back to EOD-proxy.
    ``overnight_gap`` is always EOD-derived (no proxy needed).
    """
    n = len(bars)
    open_arr = bars["open"].to_numpy(dtype=float)
    close_arr = bars["close"].to_numpy(dtype=float)
    volume_arr = bars["volume"].to_numpy(dtype=float)
    amount_arr = bars["amount"].to_numpy(dtype=float) if "amount" in bars.columns else volume_arr * close_arr
    dates_arr = pd.to_datetime(bars["date"]).dt.normalize().to_numpy()

    # overnight_gap — CLEAN, no proxy (same as compute_extras_for_panel)
    overnight = np.full(n, np.nan)
    for i in range(1, n):
        prev_close = close_arr[i - 1]
        if prev_close > 0 and open_arr[i] > 0:
            overnight[i] = (open_arr[i] - prev_close) / prev_close

    # Start with EOD-proxy values for all rows
    morning_ret = np.full(n, np.nan)
    valid = (open_arr > 0) & (close_arr > 0)
    morning_ret[valid] = MORNING_RETURN_SCALE * (close_arr[valid] - open_arr[valid]) / open_arr[valid]

    vwap_dev = np.full(n, np.nan)
    safe = (volume_arr > 0) & (amount_arr > 0) & (close_arr > 0)
    vwap_proxy = np.where(safe, amount_arr / volume_arr, np.nan)
    safe_vwap = safe & (vwap_proxy > 0)
    vwap_dev[safe_vwap] = (close_arr[safe_vwap] - vwap_proxy[safe_vwap]) / vwap_proxy[safe_vwap]

    vol_ratio = np.full(n, np.nan)
    if n >= 20:
        rolling = pd.Series(volume_arr).rolling(window=20, min_periods=20).mean().shift(1).to_numpy()
        valid_vr = (rolling > 0) & (volume_arr > 0)
        vol_ratio[valid_vr] = MORNING_VOLUME_SCALE * volume_arr[valid_vr] / rolling[valid_vr]

    # Overlay real values where available
    if real_extras_by_date is not None and not real_extras_by_date.empty:
        date_to_idx: Dict[pd.Timestamp, int] = {pd.Timestamp(d): i for i, d in enumerate(dates_arr)}
        for date, row in real_extras_by_date.iterrows():
            idx = date_to_idx.get(pd.Timestamp(date))
            if idx is None:
                continue
            for col, arr in [("morning_return", morning_ret),
                              ("morning_vwap_dev", vwap_dev),
                              ("morning_vol_ratio", vol_ratio)]:
                v = row.get(col)
                if v is not None and not pd.isna(v):
                    arr[idx] = float(v)

    return pd.DataFrame({
        "date": bars["date"].values,
        "overnight_gap": overnight,
        "morning_return": morning_ret,
        "morning_vwap_dev": vwap_dev,
        "morning_vol_ratio": vol_ratio,
    })


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
    hybrid: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Attach 4 intraday extras to an EOD panel by re-fetching OHLCV per code.

    Parameters
    ----------
    hybrid : bool, default False
        P11-4 round 91: when True, overlay real intraday features for dates
        ≥ INTRADAY_REAL_START where ``data/intraday_1m/*.parquet`` provides
        them.  Other dates still use EOD-proxy.  ``overnight_gap`` always
        from EOD (no proxy needed).

    Returns
    -------
    (panel_with_extras, stats)
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    # Hybrid path: pre-load all 1m parquet and group by code → date → 9 morning extras
    real_extras_by_code: Dict[str, pd.DataFrame] = {}
    if hybrid:
        intra_long = load_intraday_1m()
        if not intra_long.empty:
            for code, code_df in intra_long.groupby("code"):
                # 20d EOD volume MA per code — needed for morning_vol_ratio denominator
                try:
                    eod = get_daily_bars(code, start, end)
                    if eod is None or eod.empty:
                        continue
                    eod = eod.sort_values("date").reset_index(drop=True)
                    eod["date"] = pd.to_datetime(eod["date"]).dt.normalize()
                    eod_vol_ma20 = eod.set_index("date")["volume"].rolling(window=20, min_periods=20).mean().shift(1)
                except Exception:
                    continue
                code_df_typed = code_df.assign(code=code).copy()
                real_extras_by_code[code] = _real_morning_extras_per_code(code_df_typed, eod_vol_ma20)
            logger.info("hybrid mode: real intraday extras prepared for {} codes",
                        len(real_extras_by_code))

    extras_frames: List[pd.DataFrame] = []
    n_codes = len(codes)
    stats = {"codes_processed": 0, "codes_failed": 0, "codes_with_real": 0,
             "rows_real_morning": 0, "rows_proxy_morning": 0}

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

        if hybrid and code in real_extras_by_code:
            extras = compute_extras_for_panel_hybrid(bars, real_extras_by_code[code])
            stats["codes_with_real"] += 1
            # Count how many rows actually had real data
            real_dates = set(real_extras_by_code[code].index)
            for d in bars["date"]:
                if pd.Timestamp(d).normalize() in real_dates:
                    stats["rows_real_morning"] += 1
                else:
                    stats["rows_proxy_morning"] += 1
        else:
            extras = compute_extras_for_panel(bars)
            stats["rows_proxy_morning"] += len(bars)

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
    ap.add_argument("--no-extras", action="store_true",
                    help="P11-2b control: train BlendRanker(feature_cols=FACTOR_COLUMNS) only (64 cols, no 4 intraday extras). "
                         "Same dataset / seed / val_frac / label as the full P11-2 run — clean A/B per Rule #10. "
                         "Output prefix gains '_control' suffix.")
    ap.add_argument("--hybrid", action="store_true",
                    help="P11-4 Phase B: overlay real intraday features for dates ≥ 2025-09-01 "
                         "from data/intraday_1m/*.parquet; fall back to EOD-proxy elsewhere. "
                         "overnight_gap always EOD-derived.")
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
        suffix = "_control" if args.no_extras else ""
        output_prefix = args.output_prefix + suffix
        logger.info("=== Full intraday training: {} codes, start={}, horizon={}, no_extras={} → {} ===",
                    len(codes), start, args.horizon, args.no_extras, output_prefix)

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
    panel, dq_stats = attach_intraday_extras(eod_panel, codes, start, args.end, hybrid=args.hybrid)
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

    # 4. Train BlendRanker. --no-extras (P11-2b control) restricts to
    # FACTOR_COLUMNS only so the only thing varying vs the full P11-2 run
    # is the feature set (Rule #10 single-variable A/B).
    if args.no_extras:
        feature_cols = list(FACTOR_COLUMNS)
        logger.info("P11-2b CONTROL: feature_cols=FACTOR_COLUMNS ({} cols, no 4 intraday extras)",
                    len(feature_cols))
    else:
        feature_cols = INTRADAY_FEATURE_COLS
        logger.info("Training BlendRanker(feature_cols=INTRADAY_FEATURE_COLS, {} cols)...",
                    len(feature_cols))
    seed = int(os.environ.get("LGBM_SEED", "42"))
    logger.info("LGBM_SEED={} (override via env)", seed)
    t0 = time.time()
    ranker = BlendRanker(feature_cols=feature_cols)
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
