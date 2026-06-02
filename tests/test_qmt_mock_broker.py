"""Tests for :class:`mp.execution.qmt_mock_broker.QMTMockBroker`
(P8-β-1a, docs/dialog/ round 56).

Validates the 5 spec'd scenarios:
1. buy sync submit → pending → process → filled (T+1 lock preserved)
2. sell partial fill (200 of 500) then cancel → 300 shares returned
3. cash insufficient → reject (no cash mutation)
4. limit price > limit-up band → reject (with mock pre_close)
5. buy partial fill then cancel → proportional cash refund from frozen

Deterministic: tests with partial fills use ``force_fill_plan`` to
bypass the RNG entirely.
"""
from __future__ import annotations

import pytest

from mp.execution.qmt_broker import Position
from mp.execution.qmt_mock_broker import QMTMockBroker, _QMTMockConfig


def _make_broker(
    cash: float = 100_000.0,
    positions: list[Position] | None = None,
    config: _QMTMockConfig | None = None,
) -> QMTMockBroker:
    b = QMTMockBroker(cash=cash, positions=positions or [], config=config)
    b.connect()
    return b


def test_buy_pending_then_process_then_filled():
    """Single buy: submit → cash frozen → process → filled + position."""
    b = _make_broker(cash=100_000.0)
    r = b.place_limit_order("600000", "buy", 100, 10.0)
    assert r.success
    assert r.order_id.startswith("MOCK-")

    info = b.get_account_info()
    assert info.cash_available == pytest.approx(99_000.0)
    assert info.cash_frozen == pytest.approx(1_000.0)

    orders = b.get_orders()
    assert len(orders) == 1
    assert orders[0].status == "pending"
    assert orders[0].shares_filled == 0

    completed = b.process_pending_orders()
    assert completed == 1

    info = b.get_account_info()
    assert info.cash_available == pytest.approx(99_000.0)
    assert info.cash_frozen == pytest.approx(0.0)
    assert info.market_value == pytest.approx(1_000.0)

    orders = b.get_orders()
    assert orders[0].status == "filled"
    assert orders[0].shares_filled == 100

    positions = b.get_positions()
    assert len(positions) == 1
    assert positions[0].code == "600000"
    assert positions[0].shares_total == 100
    assert positions[0].shares_available == 0  # T+1 locked


def test_sell_partial_fill_then_cancel_returns_shares():
    """500-share sell, force-plan [200, 300], advance once, cancel.

    Asserts: filled 200, cancelled 300 returned to shares_available."""
    pos = Position(
        code="600000", name="浦发银行",
        shares_total=500, shares_available=500,
        avg_cost=10.0, market_price=10.0, market_value=5_000.0,
    )
    config = _QMTMockConfig(force_fill_plan={"600000": [200, 300]})
    b = _make_broker(cash=0.0, positions=[pos], config=config)

    r = b.place_limit_order("600000", "sell", 500, 10.0)
    assert r.success
    order_id = r.order_id

    p = b.get_positions()[0]
    assert p.shares_available == 0  # locked at submit

    completed = b.process_pending_orders()
    assert completed == 0  # still partial after tick 1
    orders = b.get_orders()
    assert orders[0].status == "partial"
    assert orders[0].shares_filled == 200

    info = b.get_account_info()
    assert info.cash_available == pytest.approx(2_000.0)

    rc = b.cancel_order(order_id)
    assert rc.success

    p = b.get_positions()[0]
    assert p.shares_total == 300
    assert p.shares_available == 300


def test_cash_insufficient_reject():
    """Buy 100 @ 10.0 needs 1000 cash, broker has 500 → reject."""
    b = _make_broker(cash=500.0)
    r = b.place_limit_order("600000", "buy", 100, 10.0)
    assert not r.success
    assert "insufficient cash" in (r.error or "").lower()

    info = b.get_account_info()
    assert info.cash_available == pytest.approx(500.0)
    assert info.cash_frozen == pytest.approx(0.0)

    orders = b.get_orders()
    assert len(orders) == 1
    assert orders[0].status == "rejected"
    assert orders[0].error_msg is not None


def test_limit_up_reject():
    """pre_close 10.0, limit_pct 10% → up_lim 11.0; 11.5 limit-price → reject."""
    config = _QMTMockConfig(pre_close={"600000": 10.0}, limit_pct=0.10)
    b = _make_broker(cash=100_000.0, config=config)
    r = b.place_limit_order("600000", "buy", 100, 11.5)
    assert not r.success
    assert "涨停" in (r.error or "")

    orders = b.get_orders()
    assert orders[0].status == "rejected"


def test_buy_partial_cancel_refunds_proportionally():
    """Buy 500 @ 10.0 (5000 frozen), 200 fill (2000 spent),
    cancel 300 (3000 refund)."""
    config = _QMTMockConfig(force_fill_plan={"600000": [200, 300]})
    b = _make_broker(cash=10_000.0, config=config)
    r = b.place_limit_order("600000", "buy", 500, 10.0)
    assert r.success
    order_id = r.order_id

    info = b.get_account_info()
    assert info.cash_available == pytest.approx(5_000.0)
    assert info.cash_frozen == pytest.approx(5_000.0)

    b.process_pending_orders()
    info = b.get_account_info()
    assert info.cash_available == pytest.approx(5_000.0)
    assert info.cash_frozen == pytest.approx(3_000.0)

    rc = b.cancel_order(order_id)
    assert rc.success

    info = b.get_account_info()
    assert info.cash_available == pytest.approx(8_000.0)
    assert info.cash_frozen == pytest.approx(0.0)

    positions = b.get_positions()
    assert len(positions) == 1
    assert positions[0].shares_total == 200
