"""Bridge order/cancel smoke test (BigQMT migration Phase 3, 2026-09-02).

Controlled REAL-ACCOUNT write test with ~zero fill risk:
  1. connect via FileBridgeBroker (heartbeat liveness gate)
  2. snapshot account
  3. place a 1-lot BUY at a limit far BELOW market (~-6%, clamped above the
     -10% band floor) — parks in the book / exchange holding queue, cannot fill
  4. poll get_orders until the order appears (passorder id resolution path)
  5. cancel it
  6. poll until status == cancelled
Prints PASS/FAIL per step; exits non-zero on any failure. Frozen cash during
the test ≈ one lot of a low-priced stock (~¥250).

Run ON ECS:  set MP_BROKER=bridge; .venv\\Scripts\\python.exe -X utf8 scripts\\bridge_order_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

TEST_CODE = "600808"          # low price (~¥2.6), held, liquid enough
DISCOUNT = 0.94               # limit = market * 0.94 → cannot fill
POLL_SEC = 2.0
POLL_TIMEOUT = 30.0


def main() -> int:
    os.environ.setdefault("MP_BROKER", "bridge")
    from mp.execution.broker_factory import make_broker
    b = make_broker()

    hb = json.loads((ROOT / "data" / "bridge" / "heartbeat.json").read_text())
    age = time.time() - float(hb["ts"])
    logger.info("[0] heartbeat true age = {:.1f}s", age)
    if age > 10:
        logger.error("FAIL: heartbeat stale — bridge strategy not running")
        return 1

    if not b.connect():
        logger.error("FAIL: connect")
        return 1
    logger.info("[1] connect PASS")

    ai = b.get_account_info()
    logger.info("[2] snapshot PASS: total={:,.2f} cash={:,.2f}", ai.total_assets, ai.cash_available)

    pos = b.get_positions()
    mkt = next((p.market_price for p in pos if p.code == TEST_CODE and p.market_price > 0), None)
    if mkt is None:
        logger.error("FAIL: no live market price for {} in positions", TEST_CODE)
        return 1
    limit = max(round(mkt * DISCOUNT, 2), round(mkt * 0.905, 2))
    logger.info("[3] placing 1-lot BUY {} @ {:.2f} (mkt {:.2f}, -{:.0%}) — unfillable park",
                TEST_CODE, limit, mkt, 1 - limit / mkt)
    r = b.place_limit_order(TEST_CODE, "buy", 100, limit, order_remark="bridge-smoke")
    if not r.success:
        logger.error("FAIL: place_limit_order: {}", r.error)
        return 1
    oid = r.order_id
    logger.info("[3] place PASS: order_id={}", oid)

    deadline = time.time() + POLL_TIMEOUT
    seen = None
    while time.time() < deadline:
        for o in b.get_orders():
            if o.order_id == oid:
                seen = o
                break
        if seen:
            break
        time.sleep(POLL_SEC)
    if not seen:
        logger.error("FAIL: order {} not visible in get_orders within {}s", oid, POLL_TIMEOUT)
        return 1
    logger.info("[4] visibility PASS: status={} filled={}/{}",
                seen.status, seen.shares_filled, seen.shares_submitted)
    if seen.shares_filled:
        logger.warning("unexpected fill {} shares — proceed to cancel anyway", seen.shares_filled)

    c = b.cancel_order(oid)
    if not c.success:
        logger.error("FAIL: cancel_order: {}", c.error)
        return 1
    logger.info("[5] cancel request PASS")

    deadline = time.time() + POLL_TIMEOUT
    final = None
    while time.time() < deadline:
        for o in b.get_orders():
            if o.order_id == oid:
                final = o.status
                break
        if final == "cancelled":
            break
        time.sleep(POLL_SEC)
    if final != "cancelled":
        logger.error("FAIL: final status = {} (expected cancelled)", final)
        return 1
    logger.info("[6] cancelled-confirm PASS")
    logger.info("=" * 48)
    logger.info("BRIDGE ORDER SMOKE: ALL PASS")
    logger.info("=" * 48)
    return 0


if __name__ == "__main__":
    sys.exit(main())
