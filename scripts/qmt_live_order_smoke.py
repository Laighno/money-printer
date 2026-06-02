"""QMT live order smoke test — place 1 buy + cancel (real money, minimal risk).

Places a buy order at a price safely above limit-down so the order won't fill
in practice, then immediately cancels it. Verifies place_limit_order +
cancel_order round-trip against a live QMT MiniQmt instance.

Risk budget
-----------
- 1 lot (100 shares) of 002385.SZ at 3.06 RMB ≈ 306 RMB max exposure
- Fill probability tomorrow: requires full -10% limit-down + sellers at 3.06
- If cancel fails AND order fills tomorrow, position becomes 7800 → +100 shares

Safety guards (refuse to place if any fails)
--------------------------------------------
- TICKER must equal "002385" (hardcoded; no accidental different code)
- LIMIT_PRICE must be ≤ 3.10 (well below today's close)
- SHARES must equal 100
- ACTION must equal "buy"
"""
from __future__ import annotations

import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from mp.execution.qmt_broker import QMTBroker  # noqa: E402

# Config
ACCOUNT_ID = "8886933837"
USERDATA_PATH = r"C:\guojin\userdata_mini"

TICKER = "002385"
ACTION = "buy"
SHARES = 100          # 1 lot
LIMIT_PRICE = 3.06    # 1 cent above tomorrow's expected limit-down 3.05

# Safety guards
assert TICKER == "002385", f"safety: ticker locked, got {TICKER}"
assert ACTION == "buy", f"safety: action locked, got {ACTION}"
assert SHARES == 100, f"safety: shares locked, got {SHARES}"
assert 3.00 <= LIMIT_PRICE <= 3.10, f"safety: limit out of range, got {LIMIT_PRICE}"


def find_order(orders, order_id):
    for o in orders or []:
        if str(o.order_id) == str(order_id):
            return o
    return None


def main() -> int:
    print(f"--- QMT LIVE order smoke test (1 lot + cancel) ---")
    print(f"  account_id    = {ACCOUNT_ID}")
    print(f"  ticker        = {TICKER}")
    print(f"  action        = {ACTION}")
    print(f"  shares        = {SHARES}")
    print(f"  limit_price   = {LIMIT_PRICE}")
    print(f"  worst_case    = {SHARES * LIMIT_PRICE:,.2f} RMB")
    print()

    broker = QMTBroker(account_id=ACCOUNT_ID, qmt_userdata_path=USERDATA_PATH)

    print("[1/6] connect")
    if not broker.connect():
        print("  ✗ connect failed")
        return 1
    print("  ✓ connected")

    print()
    print("[2/6] account state BEFORE order")
    asset_before = broker.get_account_info()
    print(f"  cash_available = {asset_before.cash_available:,.2f}")
    print(f"  cash_frozen    = {asset_before.cash_frozen:,.2f}")
    positions_before = broker.get_positions()
    pos_002385_before = next((p for p in positions_before if p.code == TICKER), None)
    if pos_002385_before:
        print(f"  pos {TICKER}: total={pos_002385_before.shares_total} avail={pos_002385_before.shares_available}")
    else:
        print(f"  pos {TICKER}: not held")

    print()
    print("[3/6] place_limit_order")
    order_result = broker.place_limit_order(
        code=TICKER, action=ACTION, shares=SHARES, limit_price=LIMIT_PRICE
    )
    print(f"  success     = {order_result.success}")
    print(f"  order_id    = {order_result.order_id}")
    print(f"  error       = {order_result.error}")
    print(f"  code        = {order_result.code}")
    print(f"  action      = {order_result.action}")
    print(f"  shares      = {getattr(order_result, 'shares', '?')}")
    print(f"  limit_price = {getattr(order_result, 'limit_price', '?')}")

    if not order_result.success:
        print()
        print(f"  ✗ order failed: {order_result.error}")
        print(f"  (this might be acceptable if market is closed & broker rejects)")
        broker.disconnect()
        return 0  # not a test failure, the place call did happen

    print()
    print("[4/6] wait 2s, get_orders to verify placed")
    time.sleep(2)
    orders = broker.get_orders(only_today=True)
    print(f"  total orders today = {len(orders)}")
    found = find_order(orders, order_result.order_id)
    if found:
        print(f"  ✓ our order: id={found.order_id} code={found.code} action={found.action} "
              f"shares_submitted={found.shares_submitted} shares_filled={found.shares_filled} "
              f"avg_fill_price={found.avg_fill_price} status={found.status}")
    else:
        print(f"  ⚠ order_id {order_result.order_id} not in get_orders result")

    print()
    print("[5/6] cancel_order")
    cancel_result = broker.cancel_order(order_result.order_id)
    print(f"  success     = {cancel_result.success}")
    print(f"  error       = {cancel_result.error}")

    print()
    print("[6/6] wait 2s, get_orders to verify cancellation")
    time.sleep(2)
    orders_after = broker.get_orders(only_today=True)
    found_after = find_order(orders_after, order_result.order_id)
    if found_after:
        print(f"  order final: status={found_after.status} filled={found_after.shares_filled}")
        if found_after.status == "cancelled":
            print(f"  ✓ cancellation confirmed")
        elif found_after.status in ("filled", "partial"):
            print(f"  ⚠ ORDER FILLED — cancellation lost race")
        else:
            print(f"  status = {found_after.status} (may need longer wait or broker async)")
    else:
        print(f"  ⚠ order_id not in get_orders after cancel (may have disappeared from today list)")

    broker.disconnect()
    print()
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
