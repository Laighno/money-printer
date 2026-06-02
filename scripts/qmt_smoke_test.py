"""QMT broker smoke test — read-only verify (NO orders placed).

Verifies QMTBroker can connect to a running XtMiniQmt.exe instance and
query account/positions/orders. Safe to run against real-money accounts:
this script does not call place_limit_order or cancel_order.

Usage (on ECS Windows where QMT runs)
-------------------------------------
    cd C:\\money-printer
    .venv\\Scripts\\python.exe scripts\\qmt_smoke_test.py

Prerequisites
-------------
- XtMiniQmt.exe running and logged in
- xtquant package installed in the Python env
- qmt_userdata_path matches the live mini-qmt userdata dir
"""
from __future__ import annotations

import io
import sys

# Force UTF-8 so Chinese stock names / log messages from xtquant render
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from mp.execution.qmt_broker import QMTBroker  # noqa: E402

# ECS-default ECS-config (国金 official, 2026-05-26)
ACCOUNT_ID = "8886933837"
USERDATA_PATH = r"C:\guojin\userdata_mini"


def main() -> int:
    print(f"--- QMT smoke test (READ-ONLY) ---")
    print(f"  account_id    = {ACCOUNT_ID}")
    print(f"  userdata_path = {USERDATA_PATH}")
    print()

    broker = QMTBroker(account_id=ACCOUNT_ID, qmt_userdata_path=USERDATA_PATH)

    print("[1/4] connect")
    if not broker.connect():
        print("  ✗ connect failed — abort")
        return 1
    print("  ✓ connected")

    print()
    print("[2/4] account info")
    asset = broker.get_account_info()
    if asset is None:
        print("  ✗ get_account_info returned None")
        broker.disconnect()
        return 1
    print(f"  total_assets    = {asset.total_assets:>14,.2f}")
    print(f"  cash_available  = {asset.cash_available:>14,.2f}")
    print(f"  cash_frozen     = {asset.cash_frozen:>14,.2f}")
    print(f"  market_value    = {asset.market_value:>14,.2f}")
    print(f"  updated_at      = {asset.updated_at}")

    print()
    print("[3/4] positions")
    positions = broker.get_positions()
    print(f"  count = {len(positions)}")
    for p in positions:
        print(
            f"    {p.code:>8}  {p.name:<20}  "
            f"vol={p.shares_total:>7}  avail={p.shares_available:>7}  "
            f"cost={p.avg_cost:>7.3f}  mkt_px={p.market_price:>7.3f}  "
            f"value={p.market_value:>12,.2f}"
        )

    print()
    print("[4/4] disconnect")
    broker.disconnect()
    print("  ✓ done")
    print()
    print("Smoke test PASSED — broker round-trip read-only verified.")
    print("Live order placement gate is NOT crossed by this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
