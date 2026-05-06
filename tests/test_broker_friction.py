"""Broker fee breakdown regression tests (2026-04-30).

Locks in the trade-record schema:
  - buy:  total_friction = slippage_cost + commission + stamp_tax(=0)
  - sell: total_friction = slippage_cost + commission + stamp_tax
  - friction sum is consistent with the cash impact (notional vs value).
"""

from __future__ import annotations

import pytest


def test_buy_records_friction_breakdown():
    from mp.account.broker import SimulatedBroker, FeeSchedule
    fees = FeeSchedule(slippage_bps=10, commission_bps=3, use_sqrt_impact=False)
    broker = SimulatedBroker(initial_capital=1_000_000, fees=fees, silent=True)
    trade = broker.buy(code="600000", price=10.0, target_value=100_000, date="2026-05-04")
    assert trade is not None
    # Required new fields
    for k in ("slippage_cost", "commission", "stamp_tax", "total_friction", "raw_price"):
        assert k in trade, f"buy trade missing {k}"
    # Sanity: sum equals total
    assert abs(trade["slippage_cost"] + trade["commission"] + trade["stamp_tax"]
               - trade["total_friction"]) < 1e-6
    # Buy has zero stamp tax
    assert trade["stamp_tax"] == 0.0
    # Slippage = (exec_price - raw_price) × shares
    expected_slip = (trade["price"] - trade["raw_price"]) * trade["shares"]
    assert abs(trade["slippage_cost"] - expected_slip) < 1e-6
    # Total friction should be positive
    assert trade["total_friction"] > 0


def test_sell_records_friction_breakdown():
    from mp.account.broker import SimulatedBroker, FeeSchedule
    fees = FeeSchedule(slippage_bps=10, commission_bps=3, use_sqrt_impact=False,
                        stamp_tax_bps_new=5, stamp_tax_cut_date="2023-08-28")
    broker = SimulatedBroker(initial_capital=1_000_000, fees=fees, silent=True)
    broker.buy(code="600000", price=10.0, target_value=100_000, date="2026-05-04")
    trade = broker.sell(code="600000", price=11.0, date="2026-05-05")
    assert trade is not None
    for k in ("slippage_cost", "commission", "stamp_tax", "total_friction", "raw_price"):
        assert k in trade, f"sell trade missing {k}"
    assert abs(trade["slippage_cost"] + trade["commission"] + trade["stamp_tax"]
               - trade["total_friction"]) < 1e-6
    # Sell has POSITIVE stamp tax
    assert trade["stamp_tax"] > 0
    # Slippage on sell = (raw - exec) × shares (always >= 0)
    expected_slip = (trade["raw_price"] - trade["price"]) * trade["shares"]
    assert abs(trade["slippage_cost"] - expected_slip) < 1e-6


def test_friction_consistency_with_cash_change():
    """Total friction should equal what's "missing" between notional and value."""
    from mp.account.broker import SimulatedBroker, FeeSchedule
    fees = FeeSchedule(slippage_bps=8, commission_bps=3, use_sqrt_impact=False)
    broker = SimulatedBroker(initial_capital=1_000_000, fees=fees, silent=True)
    cash_before = broker.cash
    trade = broker.buy(code="600000", price=20.0, target_value=200_000, date="2026-05-04")
    cash_after = broker.cash
    # Cash debited = total_cost (notional + commission, slippage already in price)
    notional_at_exec = trade["shares"] * trade["price"]
    cash_change = cash_before - cash_after
    # Cash change matches (notional + commission)
    assert abs(cash_change - (notional_at_exec + trade["commission"])) < 0.01
