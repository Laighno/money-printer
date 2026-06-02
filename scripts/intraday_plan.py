"""P11-5 Phase A — Intraday 14:30 plan generator.

Per advisor round 95 spec (docs/dialog/to_engineer.md): at 14:30:00 fetch
the morning-session 1m bars for the hs300+zz500 universe, compute
:data:`INTRADAY_FEATURE_COLS` (64+4), score with the hybrid Blend models
``data/intraday_blend_*.lgb`` (P11-4 Phase B output, +0.14 Sharpe lift
over EOD baseline in walk-forward), pick Top-K=10, and write the order
plan to ``data/orders/intraday_latest.json``.

USER DECISIONS (round 95)
-------------------------
- (a) Model = Hybrid intraday_blend (already current state).
- (b) Trigger = strictly 14:30:00. Use sleep-to-snapshot; if invoked
      after 14:30:30 abort (next-day 9:25 EOD path takes over).
- (c) Fallback = next-day 9:30. We do NOT retry — Phase C's 9:25 flag
      check handles that.
- (d) Paper trade = skip. Direct cutover after Phase B/C/D land.

WHERE TO RUN
============
ECS Windows (xtquant required). On Mac the script imports ``xtdata``
inside the fetch path so unit tests and dev imports work; the actual
data-fetch step will raise ImportError off-ECS.

PHASE A SCOPE (this file)
=========================
1. Sleep-to-snapshot 14:30:00 (when invoked under cron at 14:29:xx).
2. Universe = hs300+zz500 minus 创业板/科创板 (same filter as
   daily_report 9:25 path + p11_4_fetch_intraday).
3. xtdata: get_market_data (cache-read, NO download — round 109) for
   today's 1m bars, filter 09:30 ≤ t < 14:30 (Rule #11 — exclude 14:30
   bar to match the PIT contract Phase B trained on).
4. Aggregate 1m → per-code morning bar (open=first, high/low=ext, close
   = last 14:29 bar close, volume/amount = sums).
5. 20-day EOD history per code (for volume MA and prev-close fallback)
   via the same xtdata get_market_data, period=1d.
6. ``build_intraday_panel(codes, asof_dt, intraday_bars, eod_history_map)``
   → 68-column feature panel.
7. ``BlendRanker(feature_cols=INTRADAY_FEATURE_COLS).load("data/intraday_blend")``.
8. Score, sort, apply low-liquidity / 退市 filters, take Top-K=10.
9. Build order list against portfolio.yaml holdings (Phase B will swap
   in QMT live positions; for Phase A we mirror daily_report's
   conviction-target logic exactly so behaviour is identical except for
   the data snapshot).
10. Write ``data/orders/intraday_latest.json`` AND a dated archive
    ``data/orders/intraday_<YYYYMMDD>.json``.

PHASE B+ (NOT this file)
========================
- ECS Task Scheduler trigger (Phase B).
- ECS ``scripts/ecs_intraday_execute.ps1`` reads the JSON and routes to
  QMT (Phase B).
- 9:25 flag check / fallback wiring (Phase C).
- daily_report.sh continues to generate the 9:30 safety-net plan
  (Phase D, no change here).

OUTPUT SCHEMA (data/orders/intraday_latest.json)
------------------------------------------------
Mirrors data/orders/latest.json from daily_report:

    {
      "generated_at": "<ISO8601>",
      "report_date": "YYYY-MM-DD",
      "entry_path": "intraday_14_30",          ← new key, distinguishes from EOD
      "model_version": "intraday_blend_hybrid",  ← new key
      "account_snapshot": { total_assets, cash_available, market_value },
      "holdings_at_plan_time": [...],
      "orders": [...],
      "alerts": [...]
    }

Phase B's executor MUST check ``entry_path == "intraday_14_30"`` before
trusting this file — never mix it up with the 9:30 ``latest.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, date, time as dt_time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# scripts/ is not a package — make daily_report importable for the
# shared order-generation logic.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


MORNING_OPEN = dt_time(9, 30)
AFTERNOON_CUTOFF = dt_time(14, 30)
TRIGGER_TARGET = dt_time(14, 30, 0)
# Round 95: "严格 14:30:00 (sleep-to-snapshot, 不能 14:30:01 之后)".
# We allow 30s of slack for cron drift, but anything later than 14:30:30
# is "next-day 9:30 fallback territory" per decision (c) and the script
# aborts so the 9:30 path picks up cleanly the following morning.
TRIGGER_HARD_DEADLINE = dt_time(14, 30, 30)

TOP_K = 10


# ─────────────────────────────────────────────────────────────────────
# Sleep-to-snapshot trigger
# ─────────────────────────────────────────────────────────────────────

def sleep_to_trigger(now: Optional[datetime] = None) -> Tuple[datetime, bool]:
    """Block until clock time reaches :data:`TRIGGER_TARGET` (14:30:00).

    Returns ``(start_time, aborted)``.  ``aborted=True`` when we were
    invoked after :data:`TRIGGER_HARD_DEADLINE` — caller should exit
    without producing a plan (next-day 9:30 fallback).

    On weekends we abort immediately; Phase B's cron will already skip
    those, but defensive belts-and-braces here costs nothing.
    """
    now = now or datetime.now()
    if now.weekday() >= 5:
        logger.warning("Weekend invocation ({}), aborting", now.strftime("%a"))
        return now, True

    today = now.date()
    target = datetime.combine(today, TRIGGER_TARGET)
    deadline = datetime.combine(today, TRIGGER_HARD_DEADLINE)

    if now > deadline:
        logger.error("Invoked at {} > deadline {} — next-day 9:30 will take over",
                     now.strftime("%H:%M:%S"), TRIGGER_HARD_DEADLINE.strftime("%H:%M:%S"))
        return now, True

    if now < target:
        wait = (target - now).total_seconds()
        logger.info("Sleep-to-snapshot: {:.3f}s until 14:30:00", wait)
        time.sleep(wait)

    actual = datetime.now()
    logger.info("Trigger fired at {}", actual.strftime("%H:%M:%S.%f")[:-3])
    return actual, False


# ─────────────────────────────────────────────────────────────────────
# Universe
# ─────────────────────────────────────────────────────────────────────

def load_universe() -> List[str]:
    """hs300+zz500 minus 创业板/科创板.

    Matches scripts/daily_report.py recommendation filter and
    scripts/p11_4_fetch_intraday.py exactly so the production scoring
    universe equals the training universe (Rule #11).
    """
    from mp.data.fetcher import get_recommendation_universe
    codes = [str(c).zfill(6) for c in get_recommendation_universe()]
    excluded = ("300", "301", "302", "688", "689")
    filtered = [c for c in codes if not c.startswith(excluded)]
    logger.info("Universe: {} codes ({} excluded)",
                len(filtered), len(codes) - len(filtered))
    return filtered


def _code_to_xtquant(code: str) -> str:
    code = str(code).zfill(6)
    return f"{code}.SH" if code.startswith("6") else f"{code}.SZ"


def _xtquant_to_code(xt_code: str) -> str:
    return xt_code.split(".")[0]


def _canonicalize_xtdata_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Self-calibrate xtdata volume/amount to canonical 股/元.

    xtdata's daily volume unit (股 vs 手) is version/config dependent and
    cannot be verified off-ECS (no xtquant on Mac). Rather than guess, we
    derive it from the amount/(volume*close) ratio, which is ≈1 only when
    both are canonical (amount 元, volume 股, close 元/share):
      - ratio ≈ 1      → canonical, no change
      - ratio ≈ 100    → volume is 手 → ×100 to 股
      - ratio ≈ 1e-4   → amount is 万元 → ×1e4 to 元
    This makes the warm correct regardless of xtdata's convention and
    survives future xtquant changes. validate_bars (in save_bars_upsert)
    is the final backstop.
    """
    v = pd.to_numeric(df["volume"], errors="coerce")
    a = pd.to_numeric(df["amount"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    valid = v.notna() & a.notna() & c.notna() & (v > 0) & (c > 0)
    if not valid.any():
        logger.warning("warm: no valid rows to calibrate xtdata units")
        return df
    ratio = float((a[valid] / (v[valid] * c[valid])).median())
    df = df.copy()
    if 30.0 <= ratio <= 300.0:
        df["volume"] = v * 100.0
        logger.warning("warm: xtdata volume looks like 手 (amount/(vol*close) "
                       "median={:.1f}); ×100 → 股", ratio)
    elif ratio < 0.01:
        df["amount"] = a * 1e4
        logger.warning("warm: xtdata amount looks like 万元 (ratio={:.5f}); "
                       "×1e4 → 元", ratio)
    else:
        logger.info("warm: xtdata units canonical (amount/(vol*close) "
                    "median={:.3f})", ratio)
    return df


def warm_daily_bars_via_xtdata(codes: List[str], asof_date: date,
                               max_lookback_cal_days: int = 365,
                               asof_eod: Optional[date] = None) -> dict:
    """Stale-gated read-cache warm: when the local DB lags T-1, read the
    missing daily-bar tail from XtMiniQmt's local cache (get_market_data,
    no download) and upsert it, so the subsequent build_latest_features
    hits a warm DB instead of fetching from Sina serially per stock.

    Why this still exists after round 109 (advisor offered "delete OR keep
    a DB-fresh check, stale-only top-up — just no download_history_data2"):
    the round 107 failure was DB-staleness-induced — the ECS DB sat at 5/26
    on 5/28, so build_latest_features did 615 serial per-code get_daily_bars
    → Sina → ~15min. round 109 swaps intraday_plan's OWN fetches to cache-
    read, but build_latest_features still reads the DB; if the DB is stale
    that 15min path returns. This warm is cheap insurance against exactly
    that: a no-op when the DB is fresh (one GROUP BY), a fast cache-read +
    upsert when it is not. Round 109 change: get_market_data (cache read),
    NOT download_history_data2 (the round 108 mistake that hung for minutes).

    Mechanism: get_daily_bars short-circuits to the DB when
    ``db_max >= _last_expected_trading_day()``. At 14:30 (before 16:00)
    that threshold is T-1, so warming the DB to T-1 makes every per-stock
    get_daily_bars return instantly (no API).

    Scope: only the MISSING tail [earliest db_max + 1, T-1] is read
    (capped at ``max_lookback_cal_days`` for empty/very-stale codes). The
    common case (DB stale by 1 trading day) reads a single day in one
    cache call, and it avoids overwriting existing turnover-bearing Sina
    rows wholesale.

    Correctness (Rule #11): fetch_end = T-1, NEVER today. Today's 14:30
    bar is incomplete; it must not enter the EOD factor window. Today's
    morning session enters separately via intraday_bars in build_intraday_panel.

    Returns a stats dict for the round report. If the cache read comes back
    empty/bad-shape, returns ``warmed=False`` and scoring falls through to
    the per-code path (degraded but not fatal).
    """
    from datetime import timedelta

    from mp.data.store import DataStore
    from mp.data.schema import normalize_bars
    from mp.data.fetcher import _last_expected_trading_day

    codes = [str(c).zfill(6) for c in codes]
    store = DataStore()
    # Anchor the warm target on the caller's explicit asof_eod (T-1) when given
    # (round 111 — deterministic regardless of wall-clock, so an after-close
    # retest warms to the same T-1 a real 14:30 run would). Else fall back to
    # the wall-clock last-expected day.
    expected = asof_eod or _last_expected_trading_day()

    # Per-code max date already in DB (one GROUP BY query).
    db_max_by_code: Dict[str, date] = {}
    if codes:
        placeholders = ",".join(["?"] * len(codes))
        with store.engine.connect() as conn:
            rows = conn.exec_driver_sql(
                f"SELECT code, MAX(date) FROM daily_bars "
                f"WHERE code IN ({placeholders}) GROUP BY code",
                tuple(codes),
            ).fetchall()
        for code, mx in rows:
            if mx:
                db_max_by_code[str(code).zfill(6)] = pd.Timestamp(mx).date()

    stale = [c for c in codes
             if db_max_by_code.get(c) is None or db_max_by_code[c] < expected]
    if not stale:
        logger.info("warm: DB already fresh for all {} codes (>= {}), skip warm",
                    len(codes), expected)
        return {"warmed": False, "reason": "fresh", "expected": str(expected)}

    # Only now do we need xtquant (ECS-only) — the fresh-skip path above
    # stays importable on Mac for unit tests.
    from xtquant import xtdata  # noqa: WPS433

    have_dates = [db_max_by_code[c] for c in codes if c in db_max_by_code]
    gap_start = (min(have_dates) + timedelta(days=1)) if have_dates else \
                (expected - timedelta(days=max_lookback_cal_days))
    floor_start = expected - timedelta(days=max_lookback_cal_days)
    fetch_start = max(gap_start, floor_start)
    fetch_end = expected   # T-1, NOT today

    start_str = fetch_start.strftime("%Y%m%d000000")
    end_str = fetch_end.strftime("%Y%m%d235959")
    xt_codes = [_code_to_xtquant(c) for c in codes]

    logger.info("warm: {}/{} codes stale; xtdata 1d cache-read {}~{} ({} codes)",
                len(stale), len(codes), fetch_start, fetch_end, len(xt_codes))

    field_list = ["open", "high", "low", "close", "volume", "amount"]
    raw = xtdata.get_market_data(
        field_list=field_list, stock_list=xt_codes, period="1d",
        start_time=start_str, end_time=end_str, count=-1,
        dividend_type="front",   # 前复权 (qfq) — match the Sina-qfq DB convention
        fill_data=False,
    )
    if not raw or not all(isinstance(v, pd.DataFrame) for v in raw.values()):
        logger.warning("warm: xtdata 1d returned unexpected shape — skip (scoring "
                       "will fall back to per-stock fetch)")
        return {"warmed": False, "reason": "bad_shape"}

    frames = []
    for field, fdf in raw.items():
        if fdf.empty:
            continue
        s = fdf.stack()
        s.name = field
        s.index.names = ["code_xt", "date"]
        frames.append(s.to_frame())
    if not frames:
        logger.warning("warm: xtdata 1d returned 0 rows — skip")
        return {"warmed": False, "reason": "empty"}

    long = pd.concat(frames, axis=1).reset_index()
    long["code"] = long["code_xt"].apply(_xtquant_to_code)
    long.drop(columns=["code_xt"], inplace=True)
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()
    # Drop any row at/after today (defensive — fetch_end is already T-1).
    long = long[long["date"] <= pd.Timestamp(expected)].copy()
    for col in ("open", "high", "low", "close", "amount"):
        long[col] = pd.to_numeric(long[col], errors="coerce")
    long["volume"] = pd.to_numeric(long["volume"], errors="coerce")
    long = long.dropna(subset=["open", "high", "low", "close"])
    rows_fetched = len(long)
    if rows_fetched == 0:
        logger.warning("warm: 0 usable rows after cleaning — skip")
        return {"warmed": False, "reason": "no_usable_rows"}

    long = _canonicalize_xtdata_bars(long)
    long = normalize_bars(long, source="xtdata")

    written = store.save_bars_upsert(long)
    logger.info("warm: fetched {} rows, upserted {} after validation "
                "(gap {}~{})", rows_fetched, written, fetch_start, fetch_end)
    return {
        "warmed": True,
        "rows_fetched": rows_fetched,
        "rows_written": int(written),
        "fetch_start": str(fetch_start),
        "fetch_end": str(fetch_end),
        "stale_codes": len(stale),
    }


# ─────────────────────────────────────────────────────────────────────
# xtdata fetch — today 1m + 20d EOD
# ─────────────────────────────────────────────────────────────────────

def fetch_today_1m_and_eod_history(
    codes: List[str],
    asof_date: date,
    eod_days_back: int = 30,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Pull today's 1m bars (filtered to 09:30 ≤ t < 14:30) and a 20d
    EOD history per code.

    Returns (today_1m_df, eod_history_map):
      - today_1m_df: long-format DataFrame [code, datetime, open, high,
        low, close, volume, amount].
      - eod_history_map: dict[code → DataFrame[date, open, high, low,
        close, volume, amount]] covering roughly the last 30 calendar
        days (the build_intraday_panel volume-MA only needs 20 bars).

    Raises ImportError when xtquant is unavailable (Mac dev path).
    """
    from xtquant import xtdata  # noqa: WPS433 — ECS-only import

    xt_codes = [_code_to_xtquant(c) for c in codes]
    today_str = asof_date.strftime("%Y%m%d")
    today_start = f"{today_str}000000"
    today_end = f"{today_str}235959"

    # ── 1) today's 1m bars (explicit download + cache-read — round 197) ─────
    # round 197 (advisor 194 C-2 spec): explicit download_history_data2 before
    # get_market_data. round 109's assumption "XtMiniQmt auto-caches today's
    # 1m bars locally" stopped holding 6/1 — XtMiniQmt was running but行情
    # subscription didn't start, so cache was empty and intraday_plan exited
    # with "xtdata 1m returned 0 rows for all fields" (6/1 + 6/2 exit 1).
    # advisor C-1 PS1 fix (ecs_warm_intraday_cache.py + Step 2a in
    # ecs_intraday_execute.ps1) is the operational layer; this C-2 is the
    # belt-and-suspenders in intraday_plan itself so any caller (PS1 step,
    # Mac dry-run, manual ECS test) gets a working cache automatically.
    # We filter t < 14:30 below for PIT (Rule #11) regardless of window.
    logger.info("download_history_data2 (1m, warming cache) for {} codes, asof={} ...",
                len(xt_codes), today_str)
    _t_download = time.time()
    try:
        xtdata.download_history_data2(
            stock_list=xt_codes,
            period="1m",
            start_time=today_str,
            end_time=today_str,
            callback=None,
        )
        logger.info("download_history_data2 done in {:.1f}s", time.time() - _t_download)
    except Exception as e:
        # If download fails, proceed to get_market_data anyway — if cache is
        # really empty, the RuntimeError below catches it. Don't hard-fail
        # here so Mac dev path (which can't actually download) still surfaces
        # the proper "0 rows" diagnostic message.
        logger.warning("download_history_data2 failed ({}); proceeding with cache only", e)

    logger.info("get_market_data (1m cache-read) for {} codes, asof={}",
                len(xt_codes), today_str)

    field_list = ["open", "high", "low", "close", "volume", "amount"]
    raw = xtdata.get_market_data(
        field_list=field_list,
        stock_list=xt_codes,
        period="1m",
        start_time=today_start,
        end_time=today_end,
        count=-1,
        dividend_type="none",
        fill_data=False,
    )
    if not raw or not all(isinstance(v, pd.DataFrame) for v in raw.values()):
        raise RuntimeError(
            f"xtdata.get_market_data (1m) returned unexpected shape "
            f"({type(raw).__name__}); cannot proceed"
        )

    frames = []
    for field, df in raw.items():
        if df.empty:
            continue
        s = df.stack()
        s.name = field
        s.index.names = ["code_xt", "datetime"]
        frames.append(s.to_frame())
    if not frames:
        raise RuntimeError("xtdata 1m returned 0 rows for all fields")

    today_long = pd.concat(frames, axis=1).reset_index()
    today_long["code"] = today_long["code_xt"].apply(_xtquant_to_code)
    today_long.drop(columns=["code_xt"], inplace=True)
    today_long["datetime"] = pd.to_datetime(today_long["datetime"])
    times = today_long["datetime"].dt.time
    mask = (times >= MORNING_OPEN) & (times < AFTERNOON_CUTOFF)
    today_long = today_long.loc[mask].copy()
    for col in ("open", "high", "low", "close", "amount"):
        today_long[col] = today_long[col].astype(float)
    today_long["volume"] = today_long["volume"].fillna(0).astype("int64")
    today_long.sort_values(["code", "datetime"], inplace=True, ignore_index=True)
    logger.info("today 1m rows after filter (09:30<=t<14:30): {}",
                len(today_long))

    # ── 2) 20-day EOD history ────────────────────────────────────────
    eod_start_dt = pd.Timestamp(asof_date) - pd.Timedelta(days=eod_days_back + 5)
    eod_start_str = eod_start_dt.strftime("%Y%m%d000000")
    eod_end_str = (pd.Timestamp(asof_date) - pd.Timedelta(days=1)).strftime("%Y%m%d235959")
    logger.info("get_market_data (1d cache-read) {}~{} for {} codes",
                eod_start_dt.strftime("%Y%m%d"),
                (pd.Timestamp(asof_date) - pd.Timedelta(days=1)).strftime("%Y%m%d"),
                len(xt_codes))

    eod_raw = xtdata.get_market_data(
        field_list=field_list,
        stock_list=xt_codes,
        period="1d",
        start_time=eod_start_str,
        end_time=eod_end_str,
        count=-1,
        dividend_type="none",
        fill_data=False,
    )
    if not eod_raw or not all(isinstance(v, pd.DataFrame) for v in eod_raw.values()):
        raise RuntimeError("xtdata.get_market_data (1d) returned unexpected shape")

    eod_history_map: Dict[str, pd.DataFrame] = {}
    if any(not v.empty for v in eod_raw.values()):
        eod_frames = []
        for field, df in eod_raw.items():
            if df.empty:
                continue
            s = df.stack()
            s.name = field
            s.index.names = ["code_xt", "date"]
            eod_frames.append(s.to_frame())
        if eod_frames:
            eod_long = pd.concat(eod_frames, axis=1).reset_index()
            eod_long["code"] = eod_long["code_xt"].apply(_xtquant_to_code)
            eod_long.drop(columns=["code_xt"], inplace=True)
            eod_long["date"] = pd.to_datetime(eod_long["date"])
            for col in ("open", "high", "low", "close", "amount"):
                eod_long[col] = eod_long[col].astype(float)
            eod_long["volume"] = eod_long["volume"].fillna(0).astype("int64")
            eod_long.sort_values(["code", "date"], inplace=True, ignore_index=True)
            for code, grp in eod_long.groupby("code"):
                eod_history_map[code] = grp.drop(columns="code").reset_index(drop=True)
        logger.info("EOD history: {} codes covered", len(eod_history_map))
    else:
        logger.warning("EOD history fetch returned empty — overnight_gap "
                       "and morning_vol_ratio will be NaN for all codes")

    return today_long, eod_history_map


# ─────────────────────────────────────────────────────────────────────
# Aggregate 1m → morning bar
# ─────────────────────────────────────────────────────────────────────

def aggregate_morning_bars(
    today_1m: pd.DataFrame,
    asof_date: date,
) -> Dict[str, Dict]:
    """Collapse 09:30-14:29 1m bars per code into the dict shape the
    feature pipeline expects.

    Returns {code → {date, open, high, low, close, volume, amount}}.
    Codes with no morning bars (suspended / 停牌) are simply omitted —
    they'll fall out of the feature panel naturally.
    """
    if today_1m.empty:
        logger.error("No 1m bars to aggregate")
        return {}

    asof_ts = pd.Timestamp(asof_date)
    bars: Dict[str, Dict] = {}
    for code, grp in today_1m.groupby("code"):
        if grp.empty:
            continue
        grp = grp.sort_values("datetime")
        bars[str(code).zfill(6)] = {
            "date": asof_ts,
            "open": float(grp["open"].iloc[0]),
            "high": float(grp["high"].max()),
            "low": float(grp["low"].min()),
            "close": float(grp["close"].iloc[-1]),
            "volume": int(grp["volume"].sum()),
            "amount": float(grp["amount"].sum()),
        }
    logger.info("Aggregated morning bars: {} codes", len(bars))
    return bars


# ─────────────────────────────────────────────────────────────────────
# Scoring + Top-K
# ─────────────────────────────────────────────────────────────────────

def score_universe(
    codes: List[str],
    asof_date: date,
    intraday_bars: Dict[str, Dict],
    eod_history_map: Dict[str, pd.DataFrame],
    asof_eod: Optional[date] = None,
) -> pd.DataFrame:
    """Build the panel, load the hybrid intraday Blend model, predict.

    ``asof_eod`` (T-1) anchors the EOD factor window: it is threaded to
    build_intraday_panel → build_latest_features → get_daily_bars as ``end``
    so the 64 base factors stop at T-1 close and never per-code-fetch today's
    not-yet-closed bar (round 111). None → today (non-intraday default).

    Returns DataFrame [code, ml_score, predicted_excess, predicted_return,
    rank_pct, _rank] sorted desc by ml_score with _rank assigned.
    """
    from mp.ml.intraday_features import (
        INTRADAY_FEATURE_COLS,
        build_intraday_panel,
    )
    from mp.ml.model import BlendRanker

    asof_ts = pd.Timestamp(asof_date)
    end_str = asof_eod.strftime("%Y%m%d") if asof_eod else None
    panel = build_intraday_panel(
        codes=codes,
        asof_dt=asof_ts,
        intraday_bars=intraday_bars,
        end=end_str,
        eod_history_map=eod_history_map,
    )
    if panel.empty:
        raise RuntimeError("build_intraday_panel returned empty — abort plan")

    dq = panel.attrs.get("_data_quality", 1.0)
    if dq < 0.5:
        # Same guard as daily_report.recommend_stocks — refuse to issue
        # orders on broken fundamentals data.
        raise RuntimeError(
            f"Data quality {dq:.2%} < 50% — refusing to generate intraday plan "
            f"(matches daily_report 荐股降级 gate)"
        )

    ranker = BlendRanker(feature_cols=INTRADAY_FEATURE_COLS)
    ok = ranker.load("data/intraday_blend")
    if not ok:
        raise RuntimeError(
            "Failed to load data/intraday_blend_{primary,extreme}.lgb — "
            "check P11-4 Phase B artifacts are present"
        )
    logger.info("Loaded BlendRanker(intraday_blend_hybrid, {} features)",
                len(INTRADAY_FEATURE_COLS))

    scores = ranker.predict(panel)
    raw = ranker.predict_raw(panel)
    LONG_TERM_BENCH_20D = 0.005  # same as daily_report

    result = pd.DataFrame({
        "code": panel["code"].astype(str).str.zfill(6).values,
        "ml_score": scores,
        "predicted_excess": (pd.Series(raw) * 100).round(2).values,
        "predicted_return": ((pd.Series(raw) + LONG_TERM_BENCH_20D) * 100).round(2).values,
        "rank_pct": pd.Series(scores).round(4).values,
    })
    result = result.sort_values("ml_score", ascending=False).reset_index(drop=True)
    result["_rank"] = result.index + 1
    return result


ARM_B_PRICE_CAP = 50.0  # RMB; round 161 guardrail (b): high-price tickers
                         # are slippage / cost hostile for ¥20k bucket, drop them.


def apply_top_k_filters(
    full_scored: pd.DataFrame,
    top_k: int = TOP_K,
    price_map: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, list[str]]:
    """Same filter chain as daily_report.recommend_stocks Top-N:
      - drop 688/689/300/301 (already excluded from universe but defend
        in depth);
      - drop low-liquidity (20d avg amount < ¥1亿) on the top-30 head;
      - flag warn-tier (< ¥3亿) with _low_liquidity for the renderer.

    When ``price_map`` (code → most-recent-close, typically the 14:29
    morning-bar close) is supplied, additionally drop any candidate whose
    last close is > ``ARM_B_PRICE_CAP`` (default 50 RMB). This is the
    round 161 guardrail (b) for the 14:30 OOS bucket; the EOD path can
    skip ``price_map`` to keep its prior behaviour.

    Returns (top_k_df, names_for_top_k_codes).
    """
    from scripts.daily_report import (
        _recent_amount_avg,
        LOW_LIQUIDITY_FILTER_AMOUNT,
        LOW_LIQUIDITY_WARN_AMOUNT,
        get_stock_names,
    )

    code_str = full_scored["code"].astype(str)
    is_excluded = code_str.str.startswith(("688", "689", "300", "301"))
    non_star = full_scored[~is_excluded].reset_index(drop=True)
    if int(is_excluded.sum()):
        logger.info("Defensive filter: dropped {} 创业板/科创板 from top-K",
                    int(is_excluded.sum()))

    candidates = non_star.head(30).copy()
    avg_amounts = _recent_amount_avg(candidates["code"].tolist(), days=20)
    candidates["_avg_amount_20d"] = candidates["code"].map(avg_amounts).fillna(0)
    too_illiquid = candidates["_avg_amount_20d"] < LOW_LIQUIDITY_FILTER_AMOUNT
    if int(too_illiquid.sum()):
        logger.info("Low-liquidity filter dropped {}: {}",
                    int(too_illiquid.sum()),
                    list(candidates.loc[too_illiquid, "code"]))
    candidates = candidates[~too_illiquid].reset_index(drop=True)
    candidates["_low_liquidity"] = candidates["_avg_amount_20d"] < LOW_LIQUIDITY_WARN_AMOUNT

    # Arm B price cap (round 161 guardrail (b)): only applied when caller
    # supplies a price_map (typically the 14:30 intraday path).
    if price_map is not None:
        candidates["_last_price"] = candidates["code"].map(
            lambda c: float(price_map.get(str(c).zfill(6), 0.0))
        )
        # If a candidate has no price (missing morning bar) treat as 0 →
        # they would be excluded by the > cap test below being False, but
        # such codes should also be dropped because we can't size them
        # safely.
        no_price = candidates["_last_price"] <= 0
        over_cap = candidates["_last_price"] > ARM_B_PRICE_CAP
        too_pricey = over_cap | no_price
        if int(too_pricey.sum()):
            dropped_rows = candidates.loc[too_pricey, ["code", "_last_price"]]
            logger.info(
                "Arm B price cap (≤¥{:.0f}) dropped {}: {}",
                ARM_B_PRICE_CAP, int(too_pricey.sum()),
                [(row["code"], f"¥{row['_last_price']:.2f}")
                 for _, row in dropped_rows.iterrows()],
            )
        candidates = candidates[~too_pricey].reset_index(drop=True)

    top = candidates.head(top_k).reset_index(drop=True)
    name_map = get_stock_names(top["code"].tolist())
    top["name"] = top["code"].map(name_map)
    return top, name_map


# ─────────────────────────────────────────────────────────────────────
# Order generation (reuse daily_report's conviction-target logic)
# ─────────────────────────────────────────────────────────────────────

def generate_orders(
    holdings_full: list[dict],
    account: dict,
    top_k_df: pd.DataFrame,
    full_scored: pd.DataFrame,
) -> Tuple[list[dict], list[str]]:
    """Delegate to scripts.daily_report.generate_order_list so the
    14:30 path and the 9:30 path emit identical order math for the same
    recommendation + holdings + account inputs.

    The single behavioural difference is the input panel (today's 14:30
    morning bars vs yesterday's EOD), which is exactly the alpha source
    P11-4 walk-forward measured (+0.14 Sharpe over EOD baseline).
    """
    from scripts.daily_report import generate_order_list
    return generate_order_list(holdings_full, account, top_k_df, full_scored)


# ─────────────────────────────────────────────────────────────────────
# Holdings / account snapshot
# ─────────────────────────────────────────────────────────────────────

def load_holdings_and_account() -> Tuple[list[dict], dict | None]:
    """For Phase A, mirror daily_report's portfolio.yaml-based snapshot.

    Phase B's ECS executor will reconcile against QMT live positions
    before placing orders; that's intentionally NOT this script's job
    (keeps Phase A runnable on Mac for dry-run / regression tests).
    """
    from scripts.daily_report import load_holdings_full, load_account
    holdings = load_holdings_full()
    account = load_account()
    return holdings, account


# ─────────────────────────────────────────────────────────────────────
# JSON output
# ─────────────────────────────────────────────────────────────────────

def write_plan_json(
    asof_date: date,
    generated_at: datetime,
    account: dict,
    holdings: list[dict],
    orders: list[dict],
    alerts: list[str],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Write both ``intraday_latest.json`` and ``intraday_<YYYYMMDD>.json``.

    Phase C will read the latter to set ``intraday_success_<YYYYMMDD>.flag``
    after fills land; the former is what Phase B's executor consumes.
    """
    payload = {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "report_date": asof_date.strftime("%Y-%m-%d"),
        "entry_path": "intraday_14_30",
        "model_version": "intraday_blend_hybrid",
        "account_snapshot": {
            "total_assets": account.get("total_assets"),
            "cash_available": account.get("cash_available"),
            "market_value": account.get("market_value"),
        },
        "holdings_at_plan_time": [
            {
                "code": str(h["code"]).zfill(6),
                "name": h.get("name", h["code"]),
                "shares": int(h.get("shares") or 0),
                "avg_cost": float(h.get("avg_cost") or 0.0),
            }
            for h in holdings
        ],
        "orders": orders,
        "alerts": alerts,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dated_path = out_dir / f"intraday_{asof_date.strftime('%Y%m%d')}.json"
    latest_path = out_dir / "intraday_latest.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    dated_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    return dated_path, latest_path


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def run(asof_date: Optional[date] = None, skip_sleep: bool = False) -> int:
    """End-to-end Phase A flow.  Returns process exit code."""
    if asof_date is None:
        asof_date = date.today()

    if not skip_sleep:
        actual, aborted = sleep_to_trigger()
        if aborted:
            return 2

    generated_at = datetime.now()
    logger.info("=" * 60)
    logger.info("P11-5 intraday plan @ {}", generated_at.isoformat(timespec="seconds"))
    logger.info("=" * 60)

    # Round 111: anchor the EOD factor window to a fixed T-1 (last complete
    # trading day before today), deterministic regardless of wall-clock. Threaded
    # as `end` into warm + scoring so get_daily_bars short-circuits on a DB that
    # reaches T-1 and never per-code-fetches today's not-yet-closed bar (the
    # 18min killer seen in the after-close retest). PIT (Rule #11): 14:30 EOD
    # factors = T-1 close; today's morning session enters via intraday_bars.
    from mp.data.trading_calendar import previous_trading_day
    asof_eod = previous_trading_day(pd.Timestamp(asof_date)).date()
    logger.info("asof_date={} → asof_eod (T-1, EOD factor anchor)={}",
                asof_date, asof_eod)

    holdings, account = load_holdings_and_account()
    if account is None:
        logger.error("portfolio.yaml has no `account:` block — abort")
        return 3

    codes = load_universe()

    today_1m, eod_hist = fetch_today_1m_and_eod_history(codes, asof_date)
    intraday_bars = aggregate_morning_bars(today_1m, asof_date)
    if not intraday_bars:
        logger.error("No morning bars for any code — abort (data fetch failed?)")
        return 4

    # Restrict scoring to codes that actually have morning data — codes
    # halted all morning won't have bars and shouldn't be scored on a
    # stale yesterday-close synthetic.
    scoring_codes = sorted(intraday_bars.keys())
    logger.info("Scoring {} codes (universe {} → bars present {})",
                len(scoring_codes), len(codes), len(intraday_bars))

    # Round 109: stale-gated read-cache warm. No-op when the DB is already
    # fresh to T-1 (one GROUP BY); when stale, reads the missing tail from
    # XtMiniQmt's local cache (get_market_data, NOT download) and upserts so
    # build_latest_features hits warm DB (seconds) instead of fetching from
    # Sina serially per stock (~15min for 615 codes — the round 107 failure).
    # Non-fatal: a warm failure just means scoring falls back to the slow path.
    t_warm = time.time()
    try:
        warm_stats = warm_daily_bars_via_xtdata(scoring_codes, asof_date,
                                                asof_eod=asof_eod)
        logger.info("warm done in {:.1f}s: {}", time.time() - t_warm, warm_stats)
    except Exception as e:
        logger.warning("warm_daily_bars_via_xtdata failed ({}) — scoring will "
                       "use slow per-stock fetch path", e)

    full_scored = score_universe(
        codes=scoring_codes,
        asof_date=asof_date,
        intraday_bars=intraday_bars,
        eod_history_map=eod_hist,
        asof_eod=asof_eod,
    )

    # Build code→price map from 14:29 morning bars so apply_top_k_filters
    # can enforce the Arm B ≤¥50 price cap (round 161 guardrail (b)).
    price_map = {
        str(c).zfill(6): float(bar.get("close", 0.0))
        for c, bar in intraday_bars.items()
    }
    top_k_df, _name_map = apply_top_k_filters(
        full_scored, top_k=TOP_K, price_map=price_map,
    )
    if top_k_df.empty:
        logger.error("Top-K filter dropped everything — abort")
        return 5
    logger.info("Top-{}:\n{}", TOP_K,
                top_k_df[["code", "name", "predicted_excess", "_rank"]].to_string(index=False))

    orders, alerts = generate_orders(holdings, account, top_k_df, full_scored)
    logger.info("Generated {} orders, {} alerts", len(orders), len(alerts))

    out_dir = PROJECT_ROOT / "data" / "orders"
    dated, latest = write_plan_json(
        asof_date=asof_date,
        generated_at=generated_at,
        account=account,
        holdings=holdings,
        orders=orders,
        alerts=alerts,
        out_dir=out_dir,
    )
    logger.info("Wrote {}", dated)
    logger.info("Wrote {}", latest)
    logger.info("Done — Phase B executor can read intraday_latest.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--asof", default=None,
                    help="YYYYMMDD trading date (default: today). Useful for replay.")
    ap.add_argument("--skip-sleep", action="store_true",
                    help="Skip the 14:30:00 sleep-to-snapshot (testing only).")
    args = ap.parse_args()

    asof = None
    if args.asof:
        asof = datetime.strptime(args.asof, "%Y%m%d").date()
    return run(asof_date=asof, skip_sleep=args.skip_sleep)


if __name__ == "__main__":
    sys.exit(main())
