"""P11-5 Phase B preflight — verify QMT live positions match the 14:30
plan within tolerance before allowing the executor to run.

Per advisor round 97 decision (1):

    expected_codes = {h["code"] for h in plan["holdings_at_plan_time"]}
    live_codes = {p.code for p in live}
    if expected_codes != live_codes: abort
    for h in plan["holdings_at_plan_time"]:
        if abs(live.shares_total - h["shares"]) > h["shares"] * 0.05: abort

Why 5%: portfolio.yaml is daily_report 17:00 + 9:30 fills sync writes;
at 14:30 ECS QMT live should be within a few % of portfolio.yaml.
Anything beyond ~5% signals manual intervention or 9:30 path didn't
land cleanly — safer to abort than to layer an intraday plan onto an
unknown position state.

Why a separate script (not flag on execute_orders):
- Keeps the reconcile step independently logged + replayable.
- ecs_intraday_execute.ps1 can short-circuit before touching the
  executor's broker connection if reconcile fails.
- Doesn't widen execute_orders.py's surface area mid-cutover.

Exit codes (consumed by ecs_intraday_execute.ps1):
  0 = reconcile OK, proceed
  1 = drift detected (codes mismatch OR > tolerance shares drift) — abort
  2 = QMT connection / fetch failure — abort
  3 = plan file missing / not an intraday plan — abort
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="data/orders/intraday_latest.json",
                    help="Path to the intraday plan JSON.")
    ap.add_argument("--qmt-account", required=True)
    ap.add_argument("--qmt-userdata", required=True,
                    help=r"e.g. C:\guojin\userdata_mini")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="Max shares drift fraction; default 0.05 (5%%)")
    args = ap.parse_args()

    # ── plan ────────────────────────────────────────────────────────
    plan_path = Path(args.plan)
    if not plan_path.exists():
        logger.error("ABORT: plan missing: {}", plan_path)
        return 3
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("entry_path") != "intraday_14_30":
        logger.error(
            "ABORT: plan entry_path = {!r}, expected 'intraday_14_30' "
            "(refusing to reconcile against the 9:30 latest.json)",
            plan.get("entry_path"),
        )
        return 3

    holdings = plan.get("holdings_at_plan_time", []) or []
    if not holdings:
        # Empty holdings is legitimate (e.g., cash-only account); no drift to check.
        logger.info("Plan has 0 holdings; nothing to reconcile.")
        return 0

    # ── QMT live ────────────────────────────────────────────────────
    try:
        from mp.execution.qmt_broker import QMTBroker
    except Exception as e:
        logger.error("ABORT: QMTBroker import failed: {}", e)
        return 2

    broker = QMTBroker(account_id=args.qmt_account, qmt_userdata_path=args.qmt_userdata)
    try:
        ok = broker.connect()
    except Exception as e:
        logger.error("ABORT: QMT connect raised: {}", e)
        return 2
    if not ok:
        logger.error("ABORT: QMT connect returned False")
        return 2

    try:
        live = broker.get_positions()
    except Exception as e:
        logger.error("ABORT: get_positions failed: {}", e)
        broker.disconnect()
        return 2

    broker.disconnect()

    # ── codes set check ─────────────────────────────────────────────
    expected_codes = {str(h["code"]).zfill(6) for h in holdings if int(h.get("shares") or 0) > 0}
    live_codes = {p.code for p in live if int(p.shares_total) > 0}

    if expected_codes != live_codes:
        logger.error(
            "DRIFT_CODE_MISMATCH:\n  plan_only:  {}\n  live_only:  {}",
            sorted(expected_codes - live_codes),
            sorted(live_codes - expected_codes),
        )
        return 1

    # ── per-code shares drift ───────────────────────────────────────
    live_by_code = {p.code: p for p in live}
    for h in holdings:
        code = str(h["code"]).zfill(6)
        planned = int(h.get("shares") or 0)
        if planned <= 0:
            continue
        lp = live_by_code.get(code)
        if lp is None:
            logger.error("DRIFT_MISSING: {} planned={} but not in live", code, planned)
            return 1
        drift = abs(int(lp.shares_total) - planned) / planned
        if drift > args.tolerance:
            logger.error(
                "DRIFT_SHARES: {} plan={} live={} drift={:.1%} > tolerance {:.1%}",
                code, planned, lp.shares_total, drift, args.tolerance,
            )
            return 1

    logger.info(
        "RECONCILE_OK: {} codes, all within {:.0%} tolerance",
        len(expected_codes), args.tolerance,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
