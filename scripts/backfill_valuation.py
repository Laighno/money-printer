"""Backfill historical PE/PB/market-cap into the valuation table.

The project's valuation table was previously populated only by live-inference
snapshots (one row per stock per day the user ran prediction).  This left
training data with PE/PB all NaN, making those features effectively dead.

Data source
-----------
Akshare 1.18.48 does not ship ``stock_a_indicator_lg`` (Legu).  We fall back to
``stock_zh_valuation_baidu``, which returns one series per indicator with the
following density vs lookback:

  近一年  ~daily      ( ~1y back)
  近三年  ~daily      ( ~3y back)
  近五年  every 2d    ( ~5y back)
  近十年  weekly      (~10y back)   ← used by default, reaches 2016
  全部    biweekly+   (2001-now)    ← use for 2015 coverage

Since valuation factors are slow-moving, weekly samples + ``merge_asof(backward)``
in the dataset alignment step gives look-ahead-free forward-fill with at most
~1 week stale feature values — acceptable for PE/PB/total_mv.

Usage
-----
    # full backfill (ZZ500, 2015-01-01 → today)
    python scripts/backfill_valuation.py

    # smaller range / single index
    python scripts/backfill_valuation.py --universe hs300 --start 2020-01-01

    # quick smoke test on 5 stocks
    python scripts/backfill_valuation.py --codes 600519,000001,600036,000858,601318

Unit note
---------
Baidu's 总市值 is in 亿元.  The live valuation writer (from EM) stores
``total_mv`` in 元.  We multiply by 1e8 on ingest so ``log(total_mv)`` is
directly comparable between historical and live data.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from mp.data.proxy_patch import apply as _apply_proxy_patch  # noqa: E402
from mp.data.store import DataStore  # noqa: E402
from mp.data.fetcher import get_index_constituents, _is_etf_code  # noqa: E402

_apply_proxy_patch()

import akshare as ak  # noqa: E402


# --- Fetch one stock's PE/PB history ---------------------------------------

# Baidu indicator -> output column name
_BAIDU_INDICATORS = {
    "市盈率(TTM)": "pe_ttm",
    "市净率": "pb",
    "总市值": "total_mv",
}


def _fetch_one_indicator(code: str, indicator: str, period: str,
                        retries: int, sleep: float) -> Optional[pd.DataFrame]:
    """One Baidu call with retry.  Returns (date, value) frame or None."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            df = ak.stock_zh_valuation_baidu(symbol=code, indicator=indicator, period=period)
            if df is None or df.empty:
                return None
            if "date" not in df.columns or "value" not in df.columns:
                return None
            return df[["date", "value"]].copy()
        except Exception as e:
            last_exc = e
            time.sleep(sleep * (2 ** attempt))
    logger.debug("{} {}: baidu failed after {} retries ({})", code, indicator, retries, last_exc)
    return None


def _fetch_baidu_valuation(code: str, period: str = "近十年",
                           retries: int = 3, sleep: float = 0.3) -> Optional[pd.DataFrame]:
    """Fetch PE/PB/market-cap history from akshare Baidu source (3 API calls).

    Returns DataFrame with columns: code, date, pe_ttm, pb, total_mv (in 元).
    Returns None if all three calls fail.
    """
    if _is_etf_code(code):
        return None

    frames: dict[str, pd.DataFrame] = {}
    for ind_name, col_name in _BAIDU_INDICATORS.items():
        df = _fetch_one_indicator(code, ind_name, period, retries, sleep)
        if df is None:
            continue
        df = df.rename(columns={"value": col_name})
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        frames[col_name] = df

    if not frames:
        return None

    # Outer-merge on date so we keep every sampled date for any indicator.
    merged: Optional[pd.DataFrame] = None
    for col_name, df in frames.items():
        merged = df if merged is None else merged.merge(df, on="date", how="outer")

    # Ensure all three columns exist
    for col in ("pe_ttm", "pb", "total_mv"):
        if col not in merged.columns:
            merged[col] = pd.NA

    # Unit conversion: Baidu 总市值 is in 亿元; our store uses 元.
    merged["total_mv"] = pd.to_numeric(merged["total_mv"], errors="coerce") * 1e8

    merged["code"] = code
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    merged = merged.sort_values("date").reset_index(drop=True)

    return merged[["code", "date", "pe_ttm", "pb", "total_mv"]]


# --- Orchestration ---------------------------------------------------------

def _load_universe(universe: str) -> list[str]:
    if universe.lower() in ("hs300", "zz500", "zz1000"):
        codes = get_index_constituents(universe.lower())
        logger.info("Loaded {} constituents from {}", len(codes), universe.upper())
        return codes
    raise ValueError(f"Unknown universe: {universe}")


def _clip_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].copy()


def main():
    parser = argparse.ArgumentParser(description="Backfill historical PE/PB/market-cap into valuation table")
    parser.add_argument("--universe", default="zz500", choices=["hs300", "zz500", "zz1000"],
                        help="Index universe (default: zz500)")
    parser.add_argument("--codes", default=None,
                        help="Comma-separated stock codes (overrides --universe)")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD, default: today)")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent workers (default: 6)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch but do not write to DB (for schema verification)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Save to DB every N stocks to free memory (default: 50)")
    parser.add_argument("--period", default="近十年",
                        choices=["近一年", "近三年", "近五年", "近十年", "全部"],
                        help="Baidu lookback period (default: 近十年, weekly back to 2016)")
    args = parser.parse_args()

    start = args.start
    end = args.end or date.today().isoformat()

    # Universe
    if args.codes:
        codes = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]
        logger.info("Using {} explicit codes", len(codes))
    else:
        codes = _load_universe(args.universe)

    logger.info("Backfilling PE/PB/market-cap: {} stocks, {} → {}, period={}, workers={}",
                len(codes), start, end, args.period, args.workers)
    # Rough row estimate: period → samples/year
    _per_year = {"近一年": 250, "近三年": 250, "近五年": 180, "近十年": 75, "全部": 25}
    yrs = (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25
    logger.info("Estimated rows: ~{:.0f}K",
                len(codes) * yrs * _per_year.get(args.period, 75) / 1000)

    store = DataStore()
    buffer: list[pd.DataFrame] = []
    ok_count = 0
    fail_count = 0
    saved_rows = 0
    t0 = time.time()

    def _flush(force: bool = False):
        nonlocal saved_rows
        if not buffer:
            return
        if not force and len(buffer) < args.batch_size:
            return
        batch = pd.concat(buffer, ignore_index=True)
        batch = _clip_date_range(batch, start, end)
        if args.dry_run:
            logger.info("[dry-run] would save {} rows (sample head):\n{}",
                        len(batch), batch.head(3).to_string())
        elif not batch.empty:
            store.save_valuation(batch)
            saved_rows += len(batch)
        buffer.clear()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch_baidu_valuation, code, args.period): code for code in codes}

        for i, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                df = future.result()
            except Exception as e:
                logger.warning("{}: fetch raised: {}", code, e)
                df = None

            if df is not None and not df.empty:
                buffer.append(df)
                ok_count += 1
            else:
                fail_count += 1

            if i % 25 == 0:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(codes) - i) / max(rate, 1e-6)
                logger.info("progress: {}/{}  ok={}  fail={}  rate={:.1f}/s  eta={:.0f}s",
                            i, len(codes), ok_count, fail_count, rate, eta)

            _flush()

    _flush(force=True)

    logger.info("DONE: {} stocks fetched ({} ok, {} failed), {} rows saved in {:.1f}s",
                len(codes), ok_count, fail_count, saved_rows, time.time() - t0)

    # Summary probe
    if not args.dry_run and ok_count > 0:
        vh = store.load_valuation_history(codes=codes, start=start, end=end)
        if not vh.empty:
            logger.info("Verification: valuation table now has {} rows for the range", len(vh))
            logger.info("Coverage sample: {} unique codes, {} unique dates",
                        vh["code"].nunique(), vh["date"].nunique())
            pe_cov = vh["pe_ttm"].notna().mean() * 100
            pb_cov = vh["pb"].notna().mean() * 100
            mv_cov = vh["total_mv"].notna().mean() * 100
            logger.info("Non-null rate: pe_ttm={:.1f}%  pb={:.1f}%  total_mv={:.1f}%",
                        pe_cov, pb_cov, mv_cov)


if __name__ == "__main__":
    main()
