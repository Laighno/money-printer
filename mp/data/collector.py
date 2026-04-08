"""Daily external data collector.

Fetches and stores historical external signals that can't be derived from
OHLCV: fund flows, northbound capital, margin balances.

Run daily via cron or manually::

    python -m mp.data.collector          # collect today's snapshot
    python -m mp.data.collector --backfill  # backfill all available history

Data is stored as parquet files in ``data/external/`` with one file per signal
type.  Each file has columns: ``date``, ``code`` (if per-stock), and value columns.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

DATA_DIR = Path("data/external")


def _ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_existing(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _save(name: str, df: pd.DataFrame):
    _ensure_dir()
    path = DATA_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved {} rows to {}", len(df), path)


def _deduplicate(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    return df.drop_duplicates(subset=key_cols, keep="last").sort_values(key_cols).reset_index(drop=True)


# =====================================================================
# 1. Northbound (北向资金) — market-level daily, history from 2014
# =====================================================================

def collect_northbound() -> pd.DataFrame:
    """Fetch full northbound capital history (沪股通 + 深股通)."""
    import akshare as ak

    frames = []
    for sym in ["沪股通", "深股通"]:
        try:
            df = ak.stock_hsgt_hist_em(symbol=sym)
            net_col = next((c for c in df.columns if "净买" in c), df.columns[1])
            date_col = df.columns[0]
            part = pd.DataFrame({
                "date": pd.to_datetime(df[date_col]),
                f"{sym}_net": pd.to_numeric(df[net_col], errors="coerce"),
            })
            frames.append(part)
        except Exception as e:
            logger.warning("Northbound {} failed: {}", sym, e)

    if not frames:
        return pd.DataFrame()

    # Merge on date
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on="date", how="outer")

    result["northbound_net"] = result.get("沪股通_net", 0).fillna(0) + result.get("深股通_net", 0).fillna(0)
    result = result.sort_values("date").reset_index(drop=True)

    # Merge with existing
    existing = _load_existing("northbound")
    if not existing.empty:
        result = pd.concat([existing, result], ignore_index=True)
        result = _deduplicate(result, ["date"])

    _save("northbound", result)
    return result


# =====================================================================
# 2. Fund flow (主力资金) — per-stock daily, ~120 days history
# =====================================================================

def collect_fund_flow(codes: List[str], progress_callback=None) -> pd.DataFrame:
    """Fetch individual stock fund flow for given codes."""
    import akshare as ak

    frames = []
    for i, code in enumerate(codes):
        try:
            market = "sh" if code.startswith(("6", "9")) else "sz"
            df = ak.stock_individual_fund_flow(stock=code, market=market)
            date_col = df.columns[0]
            net_col = next((c for c in df.columns if "主力" in c and "净占比" in c), None)
            net_amt_col = next((c for c in df.columns if "主力" in c and "净额" in c), None)
            if net_col is None:
                continue

            part = pd.DataFrame({
                "date": pd.to_datetime(df[date_col]),
                "code": code,
                "fund_flow_pct": pd.to_numeric(df[net_col], errors="coerce"),
            })
            if net_amt_col:
                part["fund_flow_amt"] = pd.to_numeric(df[net_amt_col], errors="coerce")
            frames.append(part)
        except Exception as e:
            logger.debug("Fund flow for {} failed: {}", code, e)

        if progress_callback:
            progress_callback(i + 1, len(codes))
        if (i + 1) % 50 == 0:
            logger.info("Fund flow progress: {}/{}", i + 1, len(codes))

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    # Merge with existing
    existing = _load_existing("fund_flow")
    if not existing.empty:
        result = pd.concat([existing, result], ignore_index=True)
        result = _deduplicate(result, ["date", "code"])

    _save("fund_flow", result)
    return result


# =====================================================================
# 3. Margin balance (融资余额) — market-level daily, history from 2010
# =====================================================================

def collect_margin() -> pd.DataFrame:
    """Fetch full market-level margin balance history (SSE + SZSE).

    Uses macro_china_market_margin_sh/sz which returns daily data since 2010,
    no date looping needed.
    """
    import akshare as ak

    frames = []
    for func_name, label in [("macro_china_market_margin_sh", "sh"),
                              ("macro_china_market_margin_sz", "sz")]:
        try:
            func = getattr(ak, func_name)
            df = func()
            date_col = next((c for c in df.columns if "日期" in c), df.columns[0])
            bal_col = next((c for c in df.columns if "融资余额" in c), None)
            buy_col = next((c for c in df.columns if "融资买入" in c), None)
            total_col = next((c for c in df.columns if "融资融券余额" in c), None)

            part = pd.DataFrame({"date": pd.to_datetime(df[date_col])})
            if bal_col:
                part[f"margin_balance_{label}"] = pd.to_numeric(df[bal_col], errors="coerce")
            if buy_col:
                part[f"margin_buy_{label}"] = pd.to_numeric(df[buy_col], errors="coerce")
            if total_col:
                part[f"margin_total_{label}"] = pd.to_numeric(df[total_col], errors="coerce")
            frames.append(part)
        except Exception as e:
            logger.warning("Margin {} failed: {}", label, e)

    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on="date", how="outer")

    # Compute combined totals
    for metric in ["margin_balance", "margin_buy", "margin_total"]:
        sh_col = f"{metric}_sh"
        sz_col = f"{metric}_sz"
        if sh_col in result.columns and sz_col in result.columns:
            result[metric] = result[sh_col].fillna(0) + result[sz_col].fillna(0)

    result = result.sort_values("date").reset_index(drop=True)

    # Merge with existing
    existing = _load_existing("margin")
    if not existing.empty:
        result = pd.concat([existing, result], ignore_index=True)
        result = _deduplicate(result, ["date"])

    _save("margin", result)
    return result


# =====================================================================
# Main: collect all
# =====================================================================

def collect_all(codes: Optional[List[str]] = None, backfill: bool = False):
    """Run all collectors.

    Parameters
    ----------
    codes : list[str], optional
        Stock codes for per-stock signals. If None, uses ZZ500.
    backfill : bool
        If True, try to backfill margin data for multiple dates.
    """
    _ensure_dir()

    if codes is None:
        try:
            from mp.data.fetcher import get_index_constituents
            codes = get_index_constituents("zz500")
            logger.info("Using ZZ500 universe: {} stocks", len(codes))
        except Exception:
            codes = []

    # 1. Northbound (always full history)
    logger.info("=== Collecting northbound capital ===")
    nb = collect_northbound()
    logger.info("Northbound: {} rows", len(nb))

    # 2. Fund flow (per-stock, ~120 days)
    if codes:
        logger.info("=== Collecting fund flows for {} stocks ===", len(codes))
        ff = collect_fund_flow(codes)
        logger.info("Fund flow: {} rows", len(ff))

    # 3. Margin (full history, market-level)
    logger.info("=== Collecting margin data ===")
    mg = collect_margin()
    logger.info("Margin: {} rows", len(mg))

    logger.info("=== Collection complete ===")
    # Summary
    for name in ["northbound", "fund_flow", "margin"]:
        existing = _load_existing(name)
        if not existing.empty:
            logger.info("  {}: {} rows, {} ~ {}",
                        name, len(existing),
                        existing["date"].min(), existing["date"].max())


if __name__ == "__main__":
    import sys
    backfill = "--backfill" in sys.argv
    collect_all(backfill=backfill)
