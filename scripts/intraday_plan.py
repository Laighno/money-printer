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
3. xtdata: download_history_data2 + get_market_data for today's 1m bars,
   filter 09:30 ≤ t < 14:30 (Rule #11 — exclude 14:30 bar to match the
   PIT contract Phase B trained on).
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

    # ── 1) today's 1m bars ──────────────────────────────────────────
    logger.info("download_history_data2 (1m) for {} codes, asof={}",
                len(xt_codes), today_str)
    import threading

    def _chunk_worker(stock_list, done_event, period, start_str, end_str):
        try:
            xtdata.download_history_data2(
                stock_list=stock_list,
                period=period,
                start_time=start_str,
                end_time=end_str,
                callback=lambda d: None,
            )
        except Exception as e:
            logger.warning("download chunk inner exception: {}", e)
        finally:
            done_event.set()

    chunk_size = 50
    chunk_timeout = 30.0
    timed_out = 0
    for i in range(0, len(xt_codes), chunk_size):
        chunk = xt_codes[i:i + chunk_size]
        done_event = threading.Event()
        worker = threading.Thread(
            target=_chunk_worker,
            args=(chunk, done_event, "1m", today_start, today_end),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=chunk_timeout)
        if not done_event.is_set():
            timed_out += 1
            logger.warning("1m chunk {}-{} timed out", i, i + len(chunk))

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
    logger.info("today 1m rows after filter: {} ({} timed-out chunks)",
                len(today_long), timed_out)

    # ── 2) 20-day EOD history ────────────────────────────────────────
    eod_start_dt = pd.Timestamp(asof_date) - pd.Timedelta(days=eod_days_back + 5)
    eod_start_str = eod_start_dt.strftime("%Y%m%d000000")
    eod_end_str = (pd.Timestamp(asof_date) - pd.Timedelta(days=1)).strftime("%Y%m%d235959")
    logger.info("download_history_data2 (1d) {}~{} for {} codes",
                eod_start_dt.strftime("%Y%m%d"),
                (pd.Timestamp(asof_date) - pd.Timedelta(days=1)).strftime("%Y%m%d"),
                len(xt_codes))
    for i in range(0, len(xt_codes), chunk_size):
        chunk = xt_codes[i:i + chunk_size]
        done_event = threading.Event()
        worker = threading.Thread(
            target=_chunk_worker,
            args=(chunk, done_event, "1d", eod_start_str, eod_end_str),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=chunk_timeout)

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
) -> pd.DataFrame:
    """Build the panel, load the hybrid intraday Blend model, predict.

    Returns DataFrame [code, ml_score, predicted_excess, predicted_return,
    rank_pct, _rank] sorted desc by ml_score with _rank assigned.
    """
    from mp.ml.intraday_features import (
        INTRADAY_FEATURE_COLS,
        build_intraday_panel,
    )
    from mp.ml.model import BlendRanker

    asof_ts = pd.Timestamp(asof_date)
    panel = build_intraday_panel(
        codes=codes,
        asof_dt=asof_ts,
        intraday_bars=intraday_bars,
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


def apply_top_k_filters(full_scored: pd.DataFrame, top_k: int = TOP_K) -> Tuple[pd.DataFrame, list[str]]:
    """Same filter chain as daily_report.recommend_stocks Top-N:
      - drop 688/689/300/301 (already excluded from universe but defend
        in depth);
      - drop low-liquidity (20d avg amount < ¥1亿) on the top-30 head;
      - flag warn-tier (< ¥3亿) with _low_liquidity for the renderer.

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

    full_scored = score_universe(
        codes=scoring_codes,
        asof_date=asof_date,
        intraday_bars=intraday_bars,
        eod_history_map=eod_hist,
    )

    top_k_df, _name_map = apply_top_k_filters(full_scored, top_k=TOP_K)
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
