"""Unit tests for the unified SimulatedBroker.

Tests cover: FeeSchedule cost calculations, SimulatedBroker buy/sell mechanics,
lot rounding, cash insufficiency, stamp tax date-dependence, avg cost tracking,
peak price tracking, and value conservation across round trips.
"""

import unittest

import pandas as pd

from mp.account.broker import BrokerPosition, FeeSchedule, LOT_SIZE, SimulatedBroker


# ---------------------------------------------------------------------------
# FeeSchedule tests
# ---------------------------------------------------------------------------

class TestFeeSchedule(unittest.TestCase):

    def test_buy_exec_price_adds_slippage(self):
        f = FeeSchedule(slippage_bps=10)
        assert f.buy_exec_price(100.0) == 100.10

    def test_sell_exec_price_subtracts_slippage(self):
        f = FeeSchedule(slippage_bps=10)
        assert f.sell_exec_price(100.0) == 99.90

    def test_buy_fee_commission_only(self):
        f = FeeSchedule(commission_bps=5)
        cost = 100_000.0
        assert abs(f.buy_fee(cost) - 50.0) < 1e-9

    def test_sell_fee_includes_stamp_tax(self):
        f = FeeSchedule(commission_bps=3, stamp_tax_bps_new=5)
        proceeds = 100_000.0
        fee = f.sell_fee(proceeds, "2024-01-01")
        expected = 100_000 * 3 / 10_000 + 100_000 * 5 / 10_000  # 30 + 50 = 80
        assert abs(fee - expected) < 1e-9

    def test_stamp_tax_old_rate_before_cut(self):
        f = FeeSchedule(stamp_tax_bps_old=10, stamp_tax_bps_new=5, stamp_tax_cut_date="2023-08-28")
        assert f._stamp_tax_bps("2023-08-27") == 10

    def test_stamp_tax_new_rate_on_cut_date(self):
        f = FeeSchedule(stamp_tax_bps_old=10, stamp_tax_bps_new=5, stamp_tax_cut_date="2023-08-28")
        assert f._stamp_tax_bps("2023-08-28") == 5

    def test_stamp_tax_new_rate_after_cut(self):
        f = FeeSchedule(stamp_tax_bps_old=10, stamp_tax_bps_new=5)
        assert f._stamp_tax_bps("2025-01-01") == 5

    def test_stamp_tax_default_when_no_date(self):
        f = FeeSchedule(stamp_tax_bps_new=5)
        assert f._stamp_tax_bps("") == 5

    def test_zero_slippage(self):
        f = FeeSchedule(slippage_bps=0)
        assert f.buy_exec_price(100.0) == 100.0
        assert f.sell_exec_price(100.0) == 100.0


# ---------------------------------------------------------------------------
# BrokerPosition tests
# ---------------------------------------------------------------------------

class TestBrokerPosition(unittest.TestCase):

    def test_market_value(self):
        p = BrokerPosition(code="X", shares=1000, current_price=10.0)
        assert p.market_value == 10_000.0

    def test_pnl_pct(self):
        p = BrokerPosition(code="X", shares=100, avg_cost=10.0, current_price=11.0)
        assert abs(p.pnl_pct - 0.10) < 1e-9

    def test_drawdown_from_peak(self):
        p = BrokerPosition(code="X", shares=100, current_price=90.0, peak_price=100.0)
        assert abs(p.drawdown_from_peak - (-0.10)) < 1e-9

    def test_zero_cost_pnl(self):
        p = BrokerPosition(code="X", avg_cost=0.0, current_price=10.0)
        assert p.pnl_pct == 0.0

    def test_zero_peak_drawdown(self):
        p = BrokerPosition(code="X", current_price=10.0, peak_price=0.0)
        assert p.drawdown_from_peak == 0.0


# ---------------------------------------------------------------------------
# SimulatedBroker tests
# ---------------------------------------------------------------------------

class TestSimulatedBroker(unittest.TestCase):

    def test_initial_state(self):
        b = SimulatedBroker(100_000, silent=True)
        assert b.cash == 100_000
        assert b.total_value == 100_000
        assert len(b.positions) == 0
        assert len(b.trade_log) == 0

    def test_buy_basic(self):
        b = SimulatedBroker(100_000, FeeSchedule(slippage_bps=0, commission_bps=0), silent=True)
        trade = b.buy("X", 10.0, shares=500, date="2024-01-01")
        assert trade is not None
        assert trade["shares"] == 500
        assert "X" in b.positions
        assert b.positions["X"].shares == 500
        assert b.positions["X"].entry_date == "2024-01-01"

    def test_buy_lot_rounding(self):
        b = SimulatedBroker(100_000, FeeSchedule(slippage_bps=0, commission_bps=0), silent=True)
        trade = b.buy("X", 10.0, shares=350)
        assert trade is not None
        assert trade["shares"] == 300  # rounded down to 100-lot

    def test_buy_target_value(self):
        fees = FeeSchedule(slippage_bps=0, commission_bps=0)
        b = SimulatedBroker(100_000, fees, silent=True)
        trade = b.buy("X", 10.0, target_value=5_000)
        assert trade is not None
        assert trade["shares"] == 500
        assert b.positions["X"].shares == 500

    def test_buy_cash_insufficient(self):
        fees = FeeSchedule(slippage_bps=0, commission_bps=0)
        b = SimulatedBroker(1_000, fees, silent=True)
        trade = b.buy("X", 10.0, target_value=5_000)
        assert trade is not None
        assert trade["shares"] == 100  # can only afford 100 shares
        assert b.cash >= 0

    def test_buy_zero_cash(self):
        fees = FeeSchedule(slippage_bps=0, commission_bps=0)
        b = SimulatedBroker(50, fees, silent=True)
        trade = b.buy("X", 10.0, target_value=5_000)
        assert trade is None  # can't afford even 1 lot

    def test_buy_adds_to_existing(self):
        fees = FeeSchedule(slippage_bps=0, commission_bps=0)
        b = SimulatedBroker(100_000, fees, silent=True)
        b.buy("X", 10.0, shares=200)
        b.buy("X", 12.0, shares=300)
        assert b.positions["X"].shares == 500
        expected_avg = (10.0 * 200 + 12.0 * 300) / 500
        assert abs(b.positions["X"].avg_cost - expected_avg) < 1e-9

    def test_sell_full(self):
        fees = FeeSchedule(slippage_bps=0, commission_bps=0, stamp_tax_bps_new=0)
        b = SimulatedBroker(100_000, fees, silent=True)
        b.buy("X", 10.0, shares=500)
        cash_after_buy = b.cash
        trade = b.sell("X", 10.0)
        assert trade is not None
        assert trade["shares"] == 500
        assert "X" not in b.positions
        assert abs(b.cash - (cash_after_buy + 500 * 10.0)) < 1e-9

    def test_sell_partial(self):
        fees = FeeSchedule(slippage_bps=0, commission_bps=0, stamp_tax_bps_new=0)
        b = SimulatedBroker(100_000, fees, silent=True)
        b.buy("X", 10.0, shares=500)
        trade = b.sell("X", 10.0, shares=200)
        assert trade is not None
        assert trade["shares"] == 200
        assert b.positions["X"].shares == 300

    def test_sell_nonexistent(self):
        b = SimulatedBroker(100_000, silent=True)
        assert b.sell("Z", 10.0) is None

    def test_sell_with_stamp_tax(self):
        fees = FeeSchedule(slippage_bps=0, commission_bps=3, stamp_tax_bps_new=5)
        b = SimulatedBroker(100_000, fees, silent=True)
        b.buy("X", 10.0, shares=1000, date="2024-01-01")
        trade = b.sell("X", 10.0, date="2024-06-01")
        proceeds = 1000 * 10.0
        expected_fee = proceeds * 3 / 10_000 + proceeds * 5 / 10_000
        assert abs(trade["value"] - (proceeds - expected_fee)) < 1e-6

    def test_buy_sell_round_trip_value_conservation(self):
        """Cash after buy+sell should equal initial minus total fees."""
        fees = FeeSchedule(slippage_bps=5, commission_bps=3, stamp_tax_bps_new=5)
        b = SimulatedBroker(100_000, fees, silent=True)

        price = 10.0
        b.buy("X", price, shares=1000, date="2024-01-01")
        buy_trade = b.trade_log[-1]

        b.sell("X", price, date="2024-06-01")
        sell_trade = b.trade_log[-1]

        # Total value should be less than initial by exactly the fees
        buy_exec = fees.buy_exec_price(price)
        buy_cost = 1000 * buy_exec
        buy_fee = fees.buy_fee(buy_cost)

        sell_exec = fees.sell_exec_price(price)
        sell_proceeds = 1000 * sell_exec
        sell_fee = fees.sell_fee(sell_proceeds, "2024-06-01")

        # Slippage is a real cost — buy higher, sell lower
        slippage_cost = 1000 * (buy_exec - sell_exec)
        total_fees = buy_fee + sell_fee + slippage_cost

        assert abs(b.cash - (100_000 - total_fees)) < 1e-6
        assert b.cash < 100_000  # must have lost money to fees

    def test_update_prices(self):
        fees = FeeSchedule(slippage_bps=0, commission_bps=0)
        b = SimulatedBroker(100_000, fees, silent=True)
        b.buy("X", 10.0, shares=100)
        b.update_prices({"X": 12.0})
        assert b.positions["X"].current_price == 12.0
        assert b.positions["X"].peak_price == 12.0

    def test_update_prices_peak_tracking(self):
        fees = FeeSchedule(slippage_bps=0, commission_bps=0)
        b = SimulatedBroker(100_000, fees, silent=True)
        b.buy("X", 10.0, shares=100)
        b.update_prices({"X": 15.0})
        b.update_prices({"X": 12.0})
        assert b.positions["X"].current_price == 12.0
        assert b.positions["X"].peak_price == 15.0

    def test_total_value_reflects_market(self):
        fees = FeeSchedule(slippage_bps=0, commission_bps=0)
        b = SimulatedBroker(100_000, fees, silent=True)
        b.buy("X", 10.0, shares=1000)
        b.update_prices({"X": 12.0})
        assert abs(b.total_value - (100_000 - 10_000 + 12_000)) < 1e-9

    def test_get_holdings_df_empty(self):
        b = SimulatedBroker(100_000, silent=True)
        df = b.get_holdings_df()
        assert len(df) == 0
        assert "code" in df.columns

    def test_get_holdings_df_with_positions(self):
        fees = FeeSchedule(slippage_bps=0, commission_bps=0)
        b = SimulatedBroker(100_000, fees, silent=True)
        b.buy("X", 10.0, shares=100, date="2024-01-01")
        b.buy("Y", 20.0, shares=200, date="2024-01-02")
        df = b.get_holdings_df()
        assert len(df) == 2
        assert set(df["code"]) == {"X", "Y"}

    def test_buy_zero_price_rejected(self):
        b = SimulatedBroker(100_000, silent=True)
        assert b.buy("X", 0.0, shares=100) is None

    def test_sell_zero_price_rejected(self):
        fees = FeeSchedule(slippage_bps=0, commission_bps=0)
        b = SimulatedBroker(100_000, fees, silent=True)
        b.buy("X", 10.0, shares=100)
        assert b.sell("X", 0.0) is None

    def test_trade_log_accumulated(self):
        fees = FeeSchedule(slippage_bps=0, commission_bps=0, stamp_tax_bps_new=0)
        b = SimulatedBroker(100_000, fees, silent=True)
        b.buy("X", 10.0, shares=100)
        b.buy("Y", 20.0, shares=100)
        b.sell("X", 11.0)
        assert len(b.trade_log) == 3
        assert b.trade_log[0]["action"] == "BUY"
        assert b.trade_log[2]["action"] == "SELL"


# ---------------------------------------------------------------------------
# Integration: walk-forward-style rebalance sequence
# ---------------------------------------------------------------------------

class TestRebalanceSequence(unittest.TestCase):

    def test_rebalance_sell_then_buy(self):
        """Simulate a rebalance: sell old positions, buy new ones. Cash must stay positive."""
        fees = FeeSchedule(slippage_bps=5, commission_bps=3, stamp_tax_bps_new=5)
        b = SimulatedBroker(100_000, fees, silent=True)

        # Initial buy: 3 stocks equal weight
        for code, price in [("A", 10.0), ("B", 20.0), ("C", 15.0)]:
            b.buy(code, price, target_value=33_000, date="2024-01-01")

        assert len(b.positions) == 3
        assert b.cash >= 0

        # Prices change
        b.update_prices({"A": 11.0, "B": 18.0, "C": 16.0})

        # Rebalance: sell B, buy D
        b.sell("B", 18.0, date="2024-02-01")
        b.buy("D", 25.0, target_value=b.total_value / 3, date="2024-02-01")

        assert "B" not in b.positions
        assert "D" in b.positions
        assert b.cash >= 0
        assert b.total_value > 0


if __name__ == "__main__":
    unittest.main()
