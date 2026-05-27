"""P11-4 Step 1: Fetch 9:30-14:30 1-minute OHLCV from xtdata for the
hs300+zz500 universe.

Round 89 update: xtquant 国金 QMT free-tier 1m history is limited to
~9 months back from today. Default date range narrowed to 2025-09-01 →
2026-04-30. Pre-2025-09 download requests return finished but
get_market_data shape=(0,n) empty. The shorter coverage is what user
selected (Option B hybrid training, real-where-available + EOD-proxy
elsewhere).

Output: monthly partitioned parquet files at ``data/intraday_1m/YYYYMM.parquet``
with columns: ``code, datetime, open, high, low, close, volume``.

WHERE TO RUN
============
This script MUST run on the ECS Windows host that has QMT installed
(``xtquant`` package is only available there).  Mac engineer side does
NOT have ``xtdata`` — that's why P11-2 had to use EOD-proxy.

WORKFLOW
========
On ECS Windows (PowerShell):
  cd C:\\money-printer
  py -3 scripts/p11_4_fetch_intraday.py --start 20250901 --end 20260430

The script will:
1. Compute the hs300+zz500 universe via the existing fetcher (~800 stocks)
2. For each (code, year-month) tile:
   a. Call ``xtdata.download_history_data`` to ensure the local cache
      has 1m bars covering that month.
   b. Call ``xtdata.get_local_data`` to retrieve raw bars.
   c. Filter to trading hours 09:30 ≤ time < 14:30 (so the 14:30 bar
      itself is NOT included — that bar's open IS the 14:30 prediction
      anchor; including it would leak the score-time price).
3. Append rows to a per-month buffer, flushed to parquet on month boundary.

Resume safely after crash: existing parquet files are NOT overwritten
unless ``--force`` is passed; the loop skips months where the parquet
already exists.

EXPECTED COSTS
==============
~800 codes × 8 months × ~4800 bars/code-month ≈ 30M rows raw (post round 89).
Parquet zstd compression should produce ~80-150 MB total (8 monthly files).

xtdata.download_history_data is throttled but the short 9-month window
should complete in < 1 hour wall clock (per round 89 ECS spike).

RUNBOOK (post-run on ECS → Mac sync)
====================================
After ECS run completes:
  rsync -av ECS_USER@ECS_HOST:/c/money-printer/data/intraday_1m/ \\
        /Users/laighno/laighno/money-printer/data/intraday_1m/

Then on Mac, sanity-check via:
  ls -la data/intraday_1m/ | head
  .venv/bin/python -c "import pandas as pd; \\
    df = pd.read_parquet('data/intraday_1m/202404.parquet'); \\
    print(df.shape, df.columns.tolist(), df.head())"

OUTPUT SCHEMA (per-month parquet)
=================================
| col      | dtype     | example                  |
|----------|-----------|--------------------------|
| code     | str       | "002385" (6-digit, no suffix) |
| datetime | datetime  | 2024-04-15 09:30:00      |
| open     | float64   | 12.34                    |
| high     | float64   | 12.40                    |
| low      | float64   | 12.30                    |
| close    | float64   | 12.38                    |
| volume   | int64     | 12345 (shares)           |

P11-4 Step 2 (feature recompute) is in a separate script and DOES NOT
run on ECS — that's Mac-side after rsync.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Iterable, List, Optional

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Trading-hour bounds for the 14:30 entry path.  IMPORTANT: we include
# bars FROM 09:30 (the open auction) up to but NOT INCLUDING 14:30.  At
# 14:30 the script that runs in production will be predicting; including
# the 14:30 bar in training data would leak the prediction-time price.
MORNING_OPEN = dt_time(9, 30)
AFTERNOON_CUTOFF = dt_time(14, 30)


def _code_to_xtquant(code: str) -> str:
    """Convert 6-digit A-share code to xtquant code with exchange suffix.

    Rules (no special cases for STAR/CHINEXT — exchange determined by prefix):
      6xxxxx → SH
      otherwise → SZ
    """
    code = str(code).zfill(6)
    return f"{code}.SH" if code.startswith("6") else f"{code}.SZ"


def _xtquant_to_code(xt_code: str) -> str:
    """Strip the .SH/.SZ suffix."""
    return xt_code.split(".")[0]


def _month_range(start_yyyymmdd: str, end_yyyymmdd: str) -> List[str]:
    """Inclusive list of yyyymm tiles spanning ``[start, end]``."""
    import pandas as pd
    start = pd.to_datetime(str(start_yyyymmdd))
    end = pd.to_datetime(str(end_yyyymmdd))
    months: List[str] = []
    cur = start.replace(day=1)
    while cur <= end:
        months.append(cur.strftime("%Y%m"))
        cur = (cur + pd.offsets.MonthBegin(1)).to_pydatetime()
    return months


def _ensure_universe() -> List[str]:
    """Return current hs300+zz500 universe via existing fetcher."""
    from mp.data.fetcher import get_recommendation_universe
    codes = get_recommendation_universe()
    logger.info("Universe: {} codes (hs300+zz500)", len(codes))
    return [str(c).zfill(6) for c in codes]


def _fetch_one_month(
    xt_codes: List[str],
    yyyymm: str,
    force: bool,
    out_dir: Path,
) -> Optional[Path]:
    """Fetch 1m bars for one month-tile, save parquet.  Returns path or None."""
    import pandas as pd
    from xtquant import xtdata  # noqa: WPS433 — windows-only import

    out_path = out_dir / f"{yyyymm}.parquet"
    if out_path.exists() and not force:
        logger.info("[{}] skip (already exists)", yyyymm)
        return out_path

    year = int(yyyymm[:4])
    month = int(yyyymm[4:])
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end_excl = month_start + pd.offsets.MonthBegin(1)
    start_str = month_start.strftime("%Y%m%d000000")
    end_str = month_end_excl.strftime("%Y%m%d000000")

    # Step a — ensure local cache covers this month (throttled).
    # Use download_history_data2 (batch, with progress callback) — per-stock
    # download_history_data was hanging on the 800-code loop in production.
    logger.info("[{}] download_history_data2 for {} codes...", yyyymm, len(xt_codes))
    t0 = time.time()
    progress = {"last_n": 0}
    def _cb(d):
        n = d.get("finished", 0)
        total = d.get("total", 0)
        if n - progress["last_n"] >= 100 or n == total:
            logger.info("[{}]   progress {}/{} ({})", yyyymm, n, total, d.get("message", ""))
            progress["last_n"] = n
    try:
        xtdata.download_history_data2(
            stock_list=xt_codes, period="1m",
            start_time=start_str, end_time=end_str,
            callback=_cb,
        )
    except Exception as e:
        logger.warning("[{}] download_history_data2 failed: {}", yyyymm, e)
    logger.info("[{}] download completed in {:.1f}s", yyyymm, time.time() - t0)

    # Step b — read back from local cache.
    # Round 89 fix: use get_market_data (returns dict[field → DataFrame[time × codes]])
    # rather than get_local_data (returns dict[code → DataFrame[time × fields]]).
    # The wide-by-field shape is what the pivot below expects.
    t0 = time.time()
    field_list = ["open", "high", "low", "close", "volume"]
    raw = xtdata.get_market_data(
        field_list=field_list,
        stock_list=xt_codes,
        period="1m",
        start_time=start_str,
        end_time=end_str,
        count=-1,
        dividend_type="none",
        fill_data=False,
    )
    if not raw or not all(isinstance(v, pd.DataFrame) for v in raw.values()):
        logger.warning("[{}] xtdata.get_market_data returned unexpected shape, skipping", yyyymm)
        return None

    # get_market_data returns dict[field → DataFrame[code × time]]; stack
    # to MultiIndex(code, datetime) then concat columns for fields.
    frames: List[pd.DataFrame] = []
    for field, df in raw.items():
        if df.empty:
            continue
        s = df.stack()
        s.name = field
        s.index.names = ["code_xt", "datetime"]
        frames.append(s.to_frame())
    if not frames:
        logger.warning("[{}] no rows pivoted, skipping", yyyymm)
        return None
    merged = pd.concat(frames, axis=1).reset_index()
    merged["code"] = merged["code_xt"].apply(_xtquant_to_code)
    merged.drop(columns=["code_xt"], inplace=True)

    # Step c — filter to MORNING_OPEN ≤ t < AFTERNOON_CUTOFF (9:30-14:30 exclusive).
    merged["datetime"] = pd.to_datetime(merged["datetime"])
    times = merged["datetime"].dt.time
    mask = (times >= MORNING_OPEN) & (times < AFTERNOON_CUTOFF)
    merged = merged.loc[mask].copy()

    if merged.empty:
        logger.warning("[{}] empty after time filter, skipping", yyyymm)
        return None

    # Cast volume to int64 (xtquant returns float; raw share counts are integer).
    merged["volume"] = merged["volume"].fillna(0).astype("int64")
    for col in ("open", "high", "low", "close"):
        merged[col] = merged[col].astype("float64")
    merged = merged[["code", "datetime", "open", "high", "low", "close", "volume"]]
    merged = merged.sort_values(["code", "datetime"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, compression="zstd", index=False)
    logger.info("[{}] saved {} rows ({} codes × ~{} bars) → {} ({:.1f} MB) in {:.1f}s",
                yyyymm, len(merged), merged["code"].nunique(),
                int(len(merged) / max(merged["code"].nunique(), 1)),
                out_path, out_path.stat().st_size / 1e6,
                time.time() - t0)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Round 89: xtquant 国金 QMT free-tier 1m history limited to ~9 months
    # (back to 2025-08/09). Pre-2025-09 download requests return finished but
    # get_market_data shape=(0,n) empty. Default start matches actual coverage.
    ap.add_argument("--start", default="20250901", help="YYYYMMDD start date (inclusive)")
    ap.add_argument("--end", default="20260430", help="YYYYMMDD end date (inclusive)")
    ap.add_argument("--out-dir", default="data/intraday_1m",
                    help="Output parquet directory (default: data/intraday_1m)")
    ap.add_argument("--force", action="store_true",
                    help="Re-fetch even if monthly parquet already exists")
    ap.add_argument("--limit-codes", type=int, default=0,
                    help="If > 0, only fetch first N codes (for smoke testing on ECS)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        codes = _ensure_universe()
    except Exception as e:
        logger.error("Failed to load universe: {}", e)
        return 1

    if args.limit_codes > 0:
        codes = codes[: args.limit_codes]
        logger.warning("Smoke mode: limited to {} codes", len(codes))

    xt_codes = [_code_to_xtquant(c) for c in codes]

    months = _month_range(args.start, args.end)
    logger.info("=== P11-4 Step 1 fetch: {} codes × {} months ({}~{}) → {} ===",
                len(codes), len(months), args.start, args.end, out_dir)

    t_overall = time.time()
    ok = 0
    failed = 0
    for yyyymm in months:
        try:
            result = _fetch_one_month(xt_codes, yyyymm, args.force, out_dir)
            if result is not None:
                ok += 1
            else:
                failed += 1
        except Exception as e:
            logger.error("[{}] fetch failed: {}", yyyymm, e)
            failed += 1

    logger.info("=" * 60)
    logger.info("Done: {} OK, {} failed, {:.1f} min total",
                ok, failed, (time.time() - t_overall) / 60)
    logger.info("Output dir: {}", out_dir)
    logger.info("Next: rsync to Mac, then P11-4 Step 2 (feature recompute)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
