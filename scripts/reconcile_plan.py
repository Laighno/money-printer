"""P11-5 round 103: 9:25 diff-reconcile against the last 14:30 intraday target.

WHY THIS EXISTS (the bug it fixes)
==================================
The original Phase C used a success-flag gate: ecs_intraday_execute.ps1
wrote ``intraday_success_<date>.flag`` whenever execute_orders *placed*
orders, and ecs_auto_execute.ps1 skipped the 9:30 path when it saw the
flag. But in A-shares a 涨停/跌停 (limit-up/down) name accepts the 14:30
limit order into the queue yet never FILLS (no counterparty) and the
day-order auto-cancels at close. 发单 ≠ 成交. The flag was still written,
so the next-day 9:30 path skipped → the target position for that name
never got filled → silent residual orphan (had to wait for the next
14:30).

THE FIX (user's design, round 103)
===================================
Drop the flag entirely. Make both paths independent and idempotent by
computing "current holdings → target" diffs:
  - Normal day:  14:30 fully filled → holdings == target → 9:30 diff
    empty → no-op.
  - 涨停/跌停 day: 14:30 left a residual → holdings != target → 9:30 diff
    has orders → fills the gap at T open.

KEY INVARIANT (Rule #11): the 9:30 reconcile target is the SAME 14:30
intraday target (same model, same intended portfolio) — just a delayed
fill. It is NOT the EOD blend plan. Reconciling against blend would
churn every normal day and contaminate the Arm-A experiment.

WHAT THIS SCRIPT DOES
=====================
1. Load the last 14:30 target plan (data/orders/intraday_latest.json).
2. Freshness gate by TRADING days (not calendar — Mon-after-Fri is 1
   trading day, not 3). If the target is older than --max-staleness
   trading days, signal the caller to deep-fallback to the EOD blend
   plan (exit 10) — this is the "14:30 infra down ≥2 days" escape.
3. Reconstruct the intended post-14:30 portfolio = holdings_at_plan_time
   with each order applied.
4. Read live QMT positions.
5. residual = target − current (per code, lot-rounded). Buys priced at
   prev-close × 1.01, sells × 0.99 (9:25 is pre-open; prev-close is the
   freshest anchor).
6. Write data/orders/reconcile_latest.json (entry_path="reconcile_930").
   Empty residual → orders:[] (executor sees it and no-ops).

EXIT CODES (consumed by ecs_auto_execute.ps1)
=============================================
  0  = reconcile_latest.json written (may be empty orders) → execute it
  10 = target missing OR stale → caller should execute EOD blend
       latest.json (deep fallback)
  2  = QMT connect / query failure → caller aborts (don't trade blind)
  3  = other fatal error → caller aborts

WHERE IT RUNS
=============
ECS (needs live QMT positions). Pure diff functions
(reconstruct_target_portfolio / compute_residual) are import-safe on Mac
for unit tests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOT = 100


# ──────────────────────────────────────────────────────────────────────
# Pure diff math (unit-testable on Mac, no QMT / no I/O)
# ──────────────────────────────────────────────────────────────────────

def reconstruct_target_portfolio(plan: dict) -> Dict[str, int]:
    """Intended post-14:30 portfolio = holdings_at_plan_time + orders applied.

    Orders use the daily_report schema: cost > 0 ⇒ buy, cost < 0 ⇒ sell.
    Returns {code → shares} with non-positive holdings dropped.
    """
    target: Dict[str, int] = {}
    for h in plan.get("holdings_at_plan_time", []) or []:
        code = str(h["code"]).zfill(6)
        target[code] = target.get(code, 0) + int(h.get("shares") or 0)

    for o in plan.get("orders", []) or []:
        code = str(o["code"]).zfill(6)
        shares = int(o.get("shares") or 0)
        if shares <= 0:
            continue
        cost = float(o.get("cost") or 0.0)
        if cost > 0:          # buy
            target[code] = target.get(code, 0) + shares
        elif cost < 0:        # sell
            target[code] = target.get(code, 0) - shares
        # cost == 0 → ambiguous, skip (shouldn't happen in real plans)

    return {c: s for c, s in target.items() if s > 0}


def compute_residual(
    target: Dict[str, int],
    current: Dict[str, int],
    price_lookup,
    name_lookup=None,
) -> Tuple[List[dict], List[str]]:
    """residual orders = target − current, lot-rounded.

    ``price_lookup(code) -> Optional[float]`` returns the prev-close used
    for the limit (buy ×1.01 / sell ×0.99). A code whose price is
    unavailable is skipped with an alert (never priced at 0).

    Returns (orders, alerts). orders use the execute_orders schema:
    {code, name, action, shares, limit_price, cost, reason}.
    """
    name_lookup = name_lookup or (lambda c: c)
    orders: List[dict] = []
    alerts: List[str] = []

    for code in sorted(set(target) | set(current)):
        tgt = int(target.get(code, 0))
        cur = int(current.get(code, 0))
        delta = tgt - cur
        if delta == 0:
            continue

        px = price_lookup(code)
        if px is None or px <= 0:
            alerts.append(f"{code}: 残差 {delta:+d} 股但无参考价，跳过本单")
            continue

        if delta > 0:                       # need to BUY the gap
            shares = (delta // LOT) * LOT
            if shares < LOT:
                continue
            limit = round(px * 1.01, 2)
            orders.append({
                "code": code,
                "name": name_lookup(code),
                "action": "买入" if cur == 0 else "加仓",
                "shares": int(shares),
                "limit_price": limit,
                "cost": float(shares * limit),
                "reason": "9:25 reconcile 补 14:30 残差（涨停/跌停或未成交）",
            })
        else:                               # need to SELL the excess
            shares = min(cur, (abs(delta) // LOT) * LOT)
            if shares < LOT:
                continue
            limit = round(px * 0.99, 2)
            orders.append({
                "code": code,
                "name": name_lookup(code),
                "action": "清仓" if shares >= cur else "减仓",
                "shares": int(shares),
                "limit_price": limit,
                "cost": float(-shares * limit),
                "reason": "9:25 reconcile 平 14:30 超出目标的残差",
            })

    return orders, alerts


# ──────────────────────────────────────────────────────────────────────
# Freshness (trading-day aware)
# ──────────────────────────────────────────────────────────────────────

def staleness_trading_days(report_date: str, today: Optional[date] = None) -> int:
    """Trading days strictly AFTER ``report_date`` up to and including today.

    0 = plan generated today, 1 = previous trading day (normal case incl
    Mon-after-Fri), 2 = one trading day skipped, ...  Uses
    ``trading_days_between`` (closed interval) minus 1.
    """
    from mp.data.trading_calendar import trading_days_between
    today = today or date.today()
    start = pd.Timestamp(report_date)
    end = pd.Timestamp(today)
    closed = trading_days_between(start, end)   # inclusive of both ends
    return max(closed - 1, 0)


# ──────────────────────────────────────────────────────────────────────
# Live state + prices (ECS)
# ──────────────────────────────────────────────────────────────────────

def fetch_live(account_id: str, userdata: str) -> Tuple[Dict[str, int], dict, Dict[str, str]]:
    """Return (current_shares, account_snapshot, name_map) from live QMT.

    Raises on connect/query failure (caller maps to exit 2).
    """
    from mp.execution.qmt_broker import QMTBroker
    broker = QMTBroker(account_id=account_id, qmt_userdata_path=userdata)
    if not broker.connect():
        raise RuntimeError("QMT connect returned False")
    try:
        positions = broker.get_positions()
        acct = broker.get_account_info()
    finally:
        broker.disconnect()

    current = {p.code: int(p.shares_total) for p in positions if int(p.shares_total) > 0}
    names = {p.code: getattr(p, "name", "") or p.code for p in positions}
    account = {
        "total_assets": float(acct.total_assets),
        "cash_available": float(acct.cash_available),
        "market_value": float(acct.market_value),
    }
    holdings = [
        {"code": p.code, "name": names.get(p.code, p.code),
         "shares": int(p.shares_total), "avg_cost": float(p.avg_cost)}
        for p in positions if int(p.shares_total) > 0
    ]
    account["_holdings"] = holdings
    return current, account, names


def make_price_lookup():
    """prev-close lookup via get_daily_bars (works Mac + ECS). Cached per code."""
    from mp.data.fetcher import get_daily_bars
    cache: Dict[str, Optional[float]] = {}
    end = date.today()
    start = (pd.Timestamp(end) - pd.Timedelta(days=15)).strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")

    def _lookup(code: str) -> Optional[float]:
        if code in cache:
            return cache[code]
        try:
            df = get_daily_bars(code, start, end_s)
            if df is not None and not df.empty:
                cache[code] = float(df.sort_values("date")["close"].iloc[-1])
            else:
                cache[code] = None
        except Exception:
            cache[code] = None
        return cache[code]

    return _lookup


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-plan", default="data/orders/intraday_latest.json")
    ap.add_argument("--out", default="data/orders/reconcile_latest.json")
    ap.add_argument("--qmt-account", required=True)
    ap.add_argument("--qmt-userdata", required=True)
    ap.add_argument("--target-kind", default="intraday", choices=["intraday", "eod"],
                    help="Round 258: target plan kind. 'intraday' (default, "
                         "backward-compat) requires entry_path=='intraday_14_30'. "
                         "'eod' accepts EOD daily_report plans (entry_path None/'eod'/"
                         "'daily_report'). User 拍板 6/9 round 256: 9:25 reconcile "
                         "against EOD top-25 plan, not the OOS 14:30 target.")
    ap.add_argument("--max-staleness-trading-days", type=int, default=2,
                    help="Max trading days after the target's report_date before "
                         "deep-fallback (exit 10). Default 2 covers normal "
                         "(prev trading day), Mon-after-Fri, and 14:30-down-1-day; "
                         ">=3 (14:30 down >=2 days) -> fallback.")
    args = ap.parse_args()

    target_path = Path(args.target_plan)
    if not target_path.exists():
        logger.warning("Target plan missing: {} — signal deep fallback (exit 10)", target_path)
        return 10

    try:
        plan = json.loads(target_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to read target plan: {}", e)
        return 3

    # Round 213 Tier 0: refuse to act on plans not stamped as prod-authoritative.
    # Defends against the 6/4 9:25 incident where an ad-hoc replay overwrote
    # data/orders/intraday_latest.json and reconcile diffed against it, sending
    # 4 unintended buys.
    src = plan.get("source", {}) or {}
    if not src.get("is_prod"):
        logger.error(
            "REJECTED: target plan source not prod-authoritative "
            "(host={host}, user={user}, script={script}, asof={asof}, "
            "allow_prod_write={apw}). Refusing to reconcile — deep fallback "
            "to EOD blend (exit 11).",
            host=src.get("host"), user=src.get("user"),
            script=src.get("script"), asof=src.get("asof"),
            apw=src.get("allow_prod_write"),
        )
        return 11

    # Round 258 (advisor 256/257 ask): kind-aware entry_path check.
    # 'intraday' kind requires entry_path=='intraday_14_30' (existing behavior).
    # 'eod' kind accepts EOD daily_report plans whose entry_path is missing,
    # None, '', 'eod', or 'daily_report'. Adding this so ecs_auto_execute.ps1
    # can switch 9:25 reconcile from OOS 14:30 target -> EOD top-25 latest.json.
    entry_path = plan.get("entry_path")
    if args.target_kind == "intraday":
        if entry_path != "intraday_14_30":
            logger.warning("Target plan entry_path={!r} != 'intraday_14_30' "
                           "(target-kind=intraday) -> deep fallback (exit 10)",
                           entry_path)
            return 10
    else:  # eod
        allowed_eod = {None, "", "eod", "daily_report"}
        if entry_path not in allowed_eod:
            logger.warning("Target plan entry_path={!r} not in allowed EOD set "
                           "(target-kind=eod, allowed={!r}) -> deep fallback (exit 10)",
                           entry_path, sorted(str(x) for x in allowed_eod))
            return 10

    report_date = plan.get("report_date")
    if not report_date:
        logger.warning("Target plan has no report_date — deep fallback (exit 10)")
        return 10

    stale = staleness_trading_days(report_date)
    logger.info("Target plan report_date={} → staleness={} trading day(s) (max {})",
                report_date, stale, args.max_staleness_trading_days)
    if stale > args.max_staleness_trading_days:
        logger.warning("Target stale ({} > {} trading days) — 14:30 infra likely down "
                       "multiple days → deep fallback to EOD blend (exit 10)",
                       stale, args.max_staleness_trading_days)
        return 10

    # Reconstruct intended post-14:30 portfolio
    target = reconstruct_target_portfolio(plan)
    logger.info("Reconstructed 14:30 target: {} codes", len(target))

    # Live QMT state
    try:
        current, account, names = fetch_live(args.qmt_account, args.qmt_userdata)
    except Exception as e:
        logger.error("QMT live fetch failed ({}) — abort (exit 2)", e)
        return 2
    logger.info("Live QMT: {} positions, total={:,.0f}",
                len(current), account.get("total_assets", 0.0))

    # Diff
    def _name(code: str) -> str:
        return names.get(code) or code

    price_lookup = make_price_lookup()
    orders, alerts = compute_residual(target, current, price_lookup, name_lookup=_name)
    logger.info("Residual: {} orders, {} alerts", len(orders), len(alerts))
    for a in alerts:
        logger.warning("  reconcile alert: {}", a)

    # Round 217 Tier 1: stamp source provenance so execute_orders' Tier 0
    # source.is_prod check passes when reconcile is run inside the scheduled
    # ECS task (MP_ALLOW_PROD_WRITE=1 env set by ecs_auto_execute.ps1).
    from mp.common.paths import (
        make_plan_source,
        assert_prod_write_allowed,
        audit_prod_write,
    )
    allow_prod_write = os.environ.get("MP_ALLOW_PROD_WRITE") == "1"
    source = make_plan_source(
        allow_prod_write=allow_prod_write,
        asof=None,
        dry_run=False,
        script="reconcile_plan.py",
    )
    payload = {
        "source": source,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_date": date.today().strftime("%Y-%m-%d"),
        "entry_path": "reconcile_930",
        "model_version": "intraday_blend_hybrid_reconcile",
        "source_intraday_report_date": report_date,
        "source_staleness_trading_days": stale,
        "account_snapshot": {
            "total_assets": account.get("total_assets"),
            "cash_available": account.get("cash_available"),
            "market_value": account.get("market_value"),
        },
        # holdings_at_plan_time = live positions so execute_orders' preflight
        # reconcile passes (we just read them).
        "holdings_at_plan_time": account.get("_holdings", []),
        "orders": orders,
        "alerts": alerts,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert_prod_write_allowed(out_path)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_prod_write(out_path, source)

    if not orders:
        logger.info("Residual empty → 14:30 target already met; wrote no-op plan {}", out_path)
    else:
        logger.info("Wrote reconcile plan ({} residual orders) → {}", len(orders), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
