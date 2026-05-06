"""Backfill historical index constituent snapshots via baostock.

Fixes survivorship bias in walk-forward backtests: without point-in-time
constituents, training data built over 2015-2026 uses today's ZZ500
membership, silently including stocks that only joined recently (selected
because of good past returns — classic look-ahead) and excluding stocks
that were later delisted or demoted.

Baostock works here because it uses plain TCP (port 5000) and bypasses
the mihomo/Sparkle fake-ip DNS issue that blocks push2.eastmoney.com.

Coverage: CSI rebalances ZZ500 / HS300 / SZ50 semi-annually (June / Dec).
Baostock returns the prior trading day's snapshot on each rebalance.
Monthly sampling captures every rebalance with zero gap; ~130 snapshots
per index over 2015-01 to 2026-04, <1 min total.

Usage
-----
    python scripts/backfill_constituents.py                # zz500 monthly
    python scripts/backfill_constituents.py --indices zz500,hs300
    python scripts/backfill_constituents.py --freq QE      # quarterly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import baostock as bs  # noqa: E402

from mp.data.store import DataStore  # noqa: E402


_INDEX_FN = {
    "zz500": lambda d: bs.query_zz500_stocks(date=d),
    "hs300": lambda d: bs.query_hs300_stocks(date=d),
    "sz50":  lambda d: bs.query_sz50_stocks(date=d),
}


def _norm_code(bs_code: str) -> str:
    """'sh.600004' -> '600004'."""
    return bs_code.split(".", 1)[1] if "." in bs_code else bs_code


def _fetch_snapshot(index: str, d: str) -> tuple[str, list[str]] | None:
    """Return (authoritative_snapshot_date, codes). Baostock stamps the prior
    trading day — we use that, not the queried date."""
    rs = _INDEX_FN[index](d)
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    snap_date = rows[0][0]  # baostock's updateDate (prev trading day)
    codes = sorted({_norm_code(r[1]) for r in rows})
    return snap_date, codes


def backfill(indices: list[str], start: str, end: str, freq: str) -> None:
    store = DataStore()
    bs.login()
    try:
        dates = pd.date_range(start, end, freq=freq).strftime("%Y-%m-%d").tolist()
        logger.info("Fetching {} snapshots per index × {} indices = {} calls",
                    len(dates), len(indices), len(dates) * len(indices))

        for idx in indices:
            if idx not in _INDEX_FN:
                logger.warning("Unknown index '{}', skipping (choices: {})",
                               idx, list(_INDEX_FN.keys()))
                continue
            stored_dates: set[str] = set()
            for d in dates:
                try:
                    res = _fetch_snapshot(idx, d)
                except Exception as e:
                    logger.warning("{} @ {}: fetch failed: {}", idx, d, e)
                    continue
                if res is None:
                    logger.debug("{} @ {}: no constituents returned", idx, d)
                    continue
                snap_date, codes = res
                if snap_date in stored_dates:
                    # Same snapshot as previous month (no rebalance between)
                    continue
                store.save_constituent_snapshot(idx, codes, snapshot_date=snap_date)
                stored_dates.add(snap_date)
            logger.info("{}: saved {} unique snapshots", idx, len(stored_dates))
    finally:
        bs.logout()

    # Verification
    for idx in indices:
        stored = store.list_constituent_snapshot_dates(idx)
        if stored:
            logger.info("{}: {} snapshots in DB, range {} → {}",
                        idx, len(stored), stored[0], stored[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", default="zz500",
                    help="Comma-separated: zz500,hs300,sz50")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None, help="default: today")
    ap.add_argument("--freq", default="ME",
                    help="pandas date_range freq: ME=monthly, QE=quarterly (default: ME)")
    args = ap.parse_args()

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    indices = [s.strip().lower() for s in args.indices.split(",") if s.strip()]
    backfill(indices, args.start, end, args.freq)


if __name__ == "__main__":
    main()
