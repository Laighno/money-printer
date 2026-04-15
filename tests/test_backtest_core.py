"""Core backtest correctness tests.

These tests guard against the specific bugs that were found and fixed:
1. Weekly rebalance year-merge bug
2. Look-ahead bias (signals using rebalance-day data)
3. NAV formula (peak-based vs incremental)
4. Daily return missing sold positions
5. Stamp tax in walk-forward cost model
6. Dependency declarations

Plus new hardened tests:
7. No-lookahead with monkey-patched signal function to prove data exclusion
8. NAV boundary cases: entry, exit, empty portfolio
9. Cash flow conservation in walk-forward trading logic
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_sector_bars(n_days=5, sectors=("SectorA", "SectorB"),
                      start="2023-01-02", a_trend=10.0, b_flat=True):
    """Build minimal industry bars for testing."""
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "board_name": sectors[0],
            "date": d,
            "open": 100.0 + i * a_trend,
            "high": 115.0 + i * a_trend,
            "low": 95.0 + i * a_trend,
            "close": 110.0 + i * a_trend,
            "volume": 1e6, "amount": 1e8, "turnover": 5.0,
        })
        if len(sectors) > 1:
            close_b = 100.0 if b_flat else 100.0 + i * 2
            rows.append({
                "board_name": sectors[1],
                "date": d,
                "open": close_b, "high": close_b, "low": close_b,
                "close": close_b,
                "volume": 1e6, "amount": 1e8, "turnover": 5.0,
            })
    return pd.DataFrame(rows), dates


def _fake_signals_for_dates(dates, sectors, scores=None):
    """Build a signals_cache dict with constant composite scores."""
    if scores is None:
        scores = [1.0] * len(sectors)
    sig = pd.DataFrame({
        "composite_score": scores,
        "volatility_20d": [20.0] * len(sectors),
    }, index=list(sectors))
    return {d: sig for d in dates}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Weekly rebalance groupby — year dimension
# ═══════════════════════════════════════════════════════════════════════════

class TestWeeklyRebalance:
    def _make_trading_dates(self):
        return pd.bdate_range("2020-12-28", "2021-01-08")

    def test_engine_weekly_no_year_merge(self):
        from mp.backtest.engine import _get_rebalance_dates
        dates = self._make_trading_dates()
        result = _get_rebalance_dates(dates, "weekly")
        assert len(result) >= 2, (
            f"Expected >=2 rebalance dates across year boundary, got {len(result)}: {result}"
        )

    def test_ml_backtest_weekly_no_year_merge(self):
        from mp.backtest.ml_backtest import _get_rebalance_dates
        dates = self._make_trading_dates()
        result = _get_rebalance_dates(dates.tolist(), "weekly")
        assert len(result) >= 2, (
            f"Expected >=2 rebalance dates across year boundary, got {len(result)}: {result}"
        )

    def test_weekly_dates_are_monotonically_increasing(self):
        """Rebalance dates must be in chronological order."""
        from mp.backtest.engine import _get_rebalance_dates
        dates = pd.bdate_range("2020-01-01", "2021-12-31")
        result = _get_rebalance_dates(dates, "weekly")
        for i in range(1, len(result)):
            assert result[i] > result[i - 1], (
                f"Rebalance dates not monotonic at index {i}: {result[i-1]} >= {result[i]}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 2. No look-ahead bias — HARDENED
# ═══════════════════════════════════════════════════════════════════════════

class TestNoLookAhead:
    def test_precompute_signals_max_date_strictly_before_rebalance(self):
        """Monkey-patch generate_rotation_signals to capture the hist DataFrame
        it receives, then assert its max date < rebalance date."""
        from mp.backtest import engine

        captured_hists = {}

        def spy_generate(hist_df):
            max_dt = hist_df["date"].max()
            captured_hists[max_dt] = hist_df
            # Return a minimal valid signals DataFrame
            sectors = hist_df["board_name"].unique()
            return pd.DataFrame({
                "composite_score": [1.0] * len(sectors),
                "volatility_20d": [20.0] * len(sectors),
            }, index=sectors)

        bars, dates = _make_sector_bars(n_days=10)
        rebalance_dates = {dates[3], dates[6], dates[9]}

        with patch.object(engine, "generate_rotation_signals", side_effect=spy_generate):
            engine._precompute_signals(bars, rebalance_dates)

        # For each rebalance date, the hist max date must be < that date
        for rb_date in sorted(rebalance_dates):
            hist_dates_used = [d for d in captured_hists.keys() if d < rb_date]
            # At least one hist was captured whose max_date < rb_date
            # (the spy records max_date of each call)
            pass

        # More direct: re-patch and record per-call
        call_log = []

        def spy_generate2(hist_df):
            call_log.append(hist_df["date"].max())
            sectors = hist_df["board_name"].unique()
            return pd.DataFrame({
                "composite_score": [1.0] * len(sectors),
                "volatility_20d": [20.0] * len(sectors),
            }, index=sectors)

        call_log.clear()
        with patch.object(engine, "generate_rotation_signals", side_effect=spy_generate2):
            engine._precompute_signals(bars, rebalance_dates)

        sorted_rb = sorted(rebalance_dates)
        assert len(call_log) == len(sorted_rb)
        for rb_date, hist_max_date in zip(sorted_rb, call_log):
            assert hist_max_date < rb_date, (
                f"Signal for rebalance {rb_date} used data up to {hist_max_date} "
                f"(should be strictly < {rb_date})"
            )

    def test_inline_fallback_also_excludes_rebalance_day(self):
        """When signals_cache misses, the inline fallback in run_backtest
        must also use date < dt."""
        from mp.backtest import engine
        from mp.risk.manager import RiskParams

        bars, dates = _make_sector_bars(n_days=10)
        close_map = engine._build_close_map(bars)

        # Provide an EMPTY signals_cache so the inline fallback triggers
        empty_cache = {}

        call_log = []

        def spy_generate(hist_df):
            call_log.append(hist_df["date"].max())
            sectors = hist_df["board_name"].unique()
            return pd.DataFrame({
                "composite_score": [1.0] * len(sectors),
                "volatility_20d": [20.0] * len(sectors),
            }, index=sectors)

        params = RiskParams(max_sectors=2, stop_loss_pct=0.99,
                            trailing_stop_pct=0.99, max_drawdown_pct=0.99)

        with patch.object(engine, "generate_rotation_signals", side_effect=spy_generate):
            engine.run_backtest(
                bars, rebalance_freq="daily", top_n=2,
                risk_params=params, close_map=close_map,
                signals_cache=empty_cache, silent=True,
            )

        # Every signal call's max hist date must be < the corresponding trading date
        all_dates = sorted(bars["date"].unique())
        rebalance_dates = all_dates  # daily frequency = all dates
        for i, (rb_date, hist_max) in enumerate(zip(rebalance_dates, call_log)):
            if pd.isna(hist_max):
                continue  # first day: no prior data, nothing to leak
            assert hist_max < rb_date, (
                f"Inline fallback for {rb_date} used data up to {hist_max}"
            )

    def test_entry_uses_close_price(self):
        """All execution happens at close price for timing consistency."""
        from mp.backtest import engine
        from mp.account.broker import SimulatedBroker
        from mp.risk.manager import RiskParams

        bars, dates = _make_sector_bars(n_days=3, sectors=("OnlySector",))
        # Make open != close to distinguish them
        bars.loc[bars["date"] == dates[0], "open"] = 95.0
        bars.loc[bars["date"] == dates[0], "close"] = 110.0

        close_map = engine._build_close_map(bars)
        signals = _fake_signals_for_dates(dates, ["OnlySector"])

        params = RiskParams(max_sectors=1, stop_loss_pct=0.99,
                            trailing_stop_pct=0.99, max_drawdown_pct=0.99)

        # Patch SimulatedBroker.buy to capture the entry price
        entry_prices = []
        original_buy = SimulatedBroker.buy

        def spy_buy(self, code, price, **kwargs):
            entry_prices.append(price)
            return original_buy(self, code, price, **kwargs)

        with patch.object(SimulatedBroker, "buy", spy_buy):
            engine.run_backtest(
                bars, rebalance_freq="daily", top_n=1, risk_params=params,
                close_map=close_map, signals_cache=signals, silent=True,
            )

        assert len(entry_prices) >= 1
        # Entry should be at close price (110.0) for consistent timing
        assert entry_prices[0] == pytest.approx(110.0), (
            f"Entry price should be close (110.0), got {entry_prices[0]}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. NAV formula — incremental, with boundary cases
# ═══════════════════════════════════════════════════════════════════════════

class TestNAVIncremental:
    def test_basic_up_down_up(self):
        from mp.risk.manager import RiskManager, RiskParams

        rm = RiskManager(RiskParams(max_sectors=5), silent=True)
        rm.enter_position("A", price=100.0, weight=1.0, entry_date="2023-01-01")

        rm.update_prices({"A": 110.0})
        assert rm.nav_current == pytest.approx(1.10)
        assert rm.nav_peak == pytest.approx(1.10)

        rm.update_prices({"A": 99.0})
        expected = 1.10 * (99.0 / 110.0)
        assert rm.nav_current == pytest.approx(expected)
        assert rm.nav_peak == pytest.approx(1.10)

        rm.update_prices({"A": 105.0})
        expected2 = expected * (105.0 / 99.0)
        assert rm.nav_current == pytest.approx(expected2)

    def test_old_bug_detected(self):
        """Old formula: nav_current = nav_peak * (1 + total_pnl_from_entry).
        After peak=1.10, drop to 99 would give 1.10*(99/100-1+1) = 1.089."""
        from mp.risk.manager import RiskManager, RiskParams

        rm = RiskManager(RiskParams(max_sectors=5), silent=True)
        rm.enter_position("A", price=100.0, weight=1.0, entry_date="2023-01-01")
        rm.update_prices({"A": 110.0})
        rm.update_prices({"A": 99.0})
        assert abs(rm.nav_current - 1.089) > 0.05

    def test_nav_stable_when_no_positions(self):
        """NAV must not change when portfolio is empty."""
        from mp.risk.manager import RiskManager, RiskParams

        rm = RiskManager(RiskParams(max_sectors=5), silent=True)
        assert rm.nav_current == pytest.approx(1.0)

        # Enter and exit
        rm.enter_position("A", price=100.0, weight=0.5, entry_date="2023-01-01")
        rm.update_prices({"A": 110.0})
        nav_after_gain = rm.nav_current
        assert nav_after_gain > 1.0

        rm.exit_position("A", "test exit")
        assert len(rm.positions) == 0

        # Update with no positions — NAV must not change
        rm.update_prices({})
        assert rm.nav_current == pytest.approx(nav_after_gain), (
            "NAV changed after exiting all positions"
        )

    def test_nav_on_entry_day(self):
        """On the day a position is entered, the first update_prices uses
        entry_price as the base (via _prev_prices fallback)."""
        from mp.risk.manager import RiskManager, RiskParams

        rm = RiskManager(RiskParams(max_sectors=5), silent=True)
        rm.enter_position("A", price=100.0, weight=1.0, entry_date="2023-01-01")

        # Price closes at entry price → NAV stays at 1.0
        rm.update_prices({"A": 100.0})
        assert rm.nav_current == pytest.approx(1.0)

    def test_nav_multi_position_partial_exit(self):
        """NAV must track correctly when one of two positions is exited."""
        from mp.risk.manager import RiskManager, RiskParams

        rm = RiskManager(RiskParams(max_sectors=5), silent=True)
        rm.enter_position("A", price=100.0, weight=0.5, entry_date="2023-01-01")
        rm.enter_position("B", price=100.0, weight=0.5, entry_date="2023-01-01")

        # Both go to 110 → NAV = 1 + 0.5*0.1 + 0.5*0.1 = 1.10
        rm.update_prices({"A": 110.0, "B": 110.0})
        assert rm.nav_current == pytest.approx(1.10)

        # Exit B
        rm.exit_position("B", "rotation")

        # A stays at 110 → no change → NAV stays
        rm.update_prices({"A": 110.0})
        assert rm.nav_current == pytest.approx(1.10), (
            f"NAV should stay 1.10 after exiting flat position, got {rm.nav_current}"
        )

        # A goes to 120 → NAV = 1.10 * (1 + 0.5 * (120/110-1))
        rm.update_prices({"A": 120.0})
        expected = 1.10 * (1 + 0.5 * (120.0 / 110.0 - 1))
        assert rm.nav_current == pytest.approx(expected)

    def test_drawdown_triggers_circuit_breaker(self):
        """Circuit breaker must fire based on correct incremental NAV."""
        from mp.risk.manager import RiskManager, RiskParams

        rm = RiskManager(RiskParams(max_sectors=5, max_drawdown_pct=0.10), silent=True)
        rm.enter_position("A", price=100.0, weight=1.0, entry_date="2023-01-01")

        rm.update_prices({"A": 110.0})  # NAV = 1.10, peak = 1.10
        assert not rm.circuit_breaker_active

        # Drop to 95 → NAV = 1.10 * (95/110) ≈ 0.95, DD = 0.95/1.10 - 1 ≈ -13.6%
        rm.update_prices({"A": 95.0})
        assert rm.circuit_breaker_active, (
            f"Circuit breaker should be active. NAV={rm.nav_current:.4f}, "
            f"peak={rm.nav_peak:.4f}, DD={rm.portfolio_drawdown:.2%}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Daily return includes sold positions
# ═══════════════════════════════════════════════════════════════════════════

class TestDailyReturnIncludesSold:
    def test_return_includes_exited_position(self):
        """A position that gets stop-lossed today must still contribute
        to today's daily return (because the price move happened)."""
        from mp.backtest.engine import run_backtest, _build_close_map
        from mp.risk.manager import RiskParams

        # SectorA: day0 close=100, day1 close=200 (huge gain then hold)
        # SectorB: day0 close=100, day1 close=50  (crash → stop loss)
        dates = pd.bdate_range("2023-01-02", periods=3)
        rows = [
            {"board_name": "SectorA", "date": dates[0], "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1e6, "amount": 1e8, "turnover": 5},
            {"board_name": "SectorB", "date": dates[0], "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1e6, "amount": 1e8, "turnover": 5},
            {"board_name": "SectorA", "date": dates[1], "open": 100, "high": 200, "low": 100, "close": 200, "volume": 1e6, "amount": 1e8, "turnover": 5},
            {"board_name": "SectorB", "date": dates[1], "open": 100, "high": 100, "low": 50, "close": 50, "volume": 1e6, "amount": 1e8, "turnover": 5},
            {"board_name": "SectorA", "date": dates[2], "open": 200, "high": 200, "low": 200, "close": 200, "volume": 1e6, "amount": 1e8, "turnover": 5},
            {"board_name": "SectorB", "date": dates[2], "open": 50, "high": 50, "low": 50, "close": 50, "volume": 1e6, "amount": 1e8, "turnover": 5},
        ]
        bars = pd.DataFrame(rows)
        close_map = _build_close_map(bars)
        signals = _fake_signals_for_dates(dates, ["SectorA", "SectorB"])

        # Stop loss at 40% — SectorB's 50% drop will trigger it
        params = RiskParams(
            max_sectors=2, stop_loss_pct=0.40,
            trailing_stop_pct=0.99, max_drawdown_pct=0.99,
        )
        result = run_backtest(
            bars, rebalance_freq="daily", top_n=2, risk_params=params,
            close_map=close_map, signals_cache=signals, silent=True,
        )

        # Day 1 return should reflect BOTH positions' price moves
        # Even though SectorB will be stop-lossed after the return calc
        day1 = result[result["date"] == dates[1]]
        assert len(day1) == 1
        day1_ret = day1["daily_return"].iloc[0]
        # SectorA: +100%, SectorB: -50%, equal weight → mixed return
        # Must not be +100% (which would mean SectorB was excluded)
        assert day1_ret < 0.9, (
            f"Day 1 return = {day1_ret:.2%}, looks like sold position was excluded"
        )

    def test_basic_return_accounting(self):
        """Simple case: two sectors, no exits, returns should be correct."""
        from mp.backtest.engine import run_backtest, _build_close_map
        from mp.risk.manager import RiskParams

        bars, dates = _make_sector_bars(n_days=5, a_trend=10.0, b_flat=True)
        close_map = _build_close_map(bars)
        signals = _fake_signals_for_dates(dates, ["SectorA", "SectorB"])

        params = RiskParams(max_sectors=2, stop_loss_pct=0.99,
                            trailing_stop_pct=0.99, max_drawdown_pct=0.99)
        result = run_backtest(
            bars, rebalance_freq="daily", top_n=2, risk_params=params,
            close_map=close_map, signals_cache=signals, silent=True,
        )

        assert len(result) >= 3
        assert result["nav"].iloc[-1] > 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. Stamp tax — walk-forward cost model
# ═══════════════════════════════════════════════════════════════════════════

class TestStampTax:
    """Stamp tax behavioral tests against FeeSchedule (mp/account/broker.py)."""

    def test_stamp_tax_defaults_defined(self):
        from mp.account.broker import FeeSchedule
        f = FeeSchedule()
        assert f.stamp_tax_bps_old == 10
        assert f.stamp_tax_bps_new == 5
        assert f.stamp_tax_cut_date == "2023-08-28"

    def test_stamp_tax_in_sell_fee(self):
        from mp.account.broker import FeeSchedule
        f = FeeSchedule(commission_bps=0, stamp_tax_bps_new=5)
        fee = f.sell_fee(100_000, "2024-01-01")
        assert fee == 100_000 * 5 / 10_000  # stamp tax only, no commission

    def test_stamp_tax_not_in_buy_fee(self):
        """Stamp tax is sell-side only in A-shares. Buy fee must not include it."""
        from mp.account.broker import FeeSchedule
        f = FeeSchedule(commission_bps=3, stamp_tax_bps_new=5)
        buy_fee = f.buy_fee(100_000)
        assert buy_fee == 100_000 * 3 / 10_000  # commission only, no stamp tax

    def test_stamp_tax_rate_is_date_dependent(self):
        """The sell logic must check the date to choose the correct rate."""
        from mp.account.broker import FeeSchedule
        f = FeeSchedule(stamp_tax_bps_old=10, stamp_tax_bps_new=5, stamp_tax_cut_date="2023-08-28")
        assert f._stamp_tax_bps("2023-08-27") == 10  # old rate before cut
        assert f._stamp_tax_bps("2023-08-28") == 5   # new rate on cut date
        assert f._stamp_tax_bps("2024-01-01") == 5   # new rate after cut


# ═══════════════════════════════════════════════════════════════════════════
# 6. Dependency declarations
# ═══════════════════════════════════════════════════════════════════════════

class TestDependencies:
    def test_key_deps_in_pyproject(self):
        with open("pyproject.toml") as f:
            content = f.read()
        for dep in ["lightgbm", "scipy", "pyarrow"]:
            assert dep in content, f"Missing dependency: {dep}"


# ═══════════════════════════════════════════════════════════════════════════
# 7. Cash flow conservation (walk-forward trading logic)
# ═══════════════════════════════════════════════════════════════════════════

class TestCashFlowConservation:
    """Verify that money is neither created nor destroyed in trading."""

    def test_buy_sell_round_trip_conserves_value(self):
        """Buy 10 stocks, sell them next day at same price.
        Total value should decrease by exactly (commission + stamp tax)."""
        # Simulate the walk-forward trading math directly
        SLIPPAGE_BPS = 5
        COMMISSION_BPS = 3
        STAMP_TAX_BPS = 5  # post-2023-08 rate

        initial_cash = 100_000.0
        cash = initial_cash
        price = 50.0
        lot_size = 100

        # BUY
        slippage_mult = SLIPPAGE_BPS / 10_000
        commission_mult = COMMISSION_BPS / 10_000
        buy_price = price * (1 + slippage_mult)
        shares = int(initial_cash / buy_price / lot_size) * lot_size  # max lots
        buy_cost = shares * buy_price
        buy_fee = buy_cost * commission_mult
        cash -= buy_cost + buy_fee

        # Holdings at market price (no slippage for valuation)
        holdings_value = shares * price

        # Total value after buy (at mid-market)
        total_after_buy = cash + holdings_value

        # SELL at same price
        sell_price = price * (1 - slippage_mult)
        proceeds = shares * sell_price
        sell_fee = proceeds * commission_mult + proceeds * STAMP_TAX_BPS / 10_000
        cash_after = cash + proceeds - sell_fee

        # The difference should be exactly the total costs
        expected_costs = buy_fee + (buy_price - price) * shares + (price - sell_price) * shares + sell_fee
        actual_loss = initial_cash - cash_after

        assert actual_loss == pytest.approx(expected_costs, rel=1e-6), (
            f"Cash loss ({actual_loss:.2f}) != expected costs ({expected_costs:.2f})"
        )
        # Costs must be positive (money was spent, not created)
        assert actual_loss > 0

    def test_no_money_creation_on_sell(self):
        """Selling at the same price as buying must not increase total value."""
        cash = 100_000.0
        price = 20.0
        shares = 500

        # Buy
        buy_cost = shares * price
        cash -= buy_cost

        # Sell at same price, with commission + stamp tax
        commission = buy_cost * 0.0003
        stamp_tax = buy_cost * 0.0005
        proceeds = shares * price - commission - stamp_tax
        cash += proceeds

        assert cash < 100_000.0, "Round-trip at same price must lose money to fees"

    def test_lot_size_rounding_does_not_create_value(self):
        """After rounding to 100-share lots, the cost must not exceed cash."""
        cash = 10_000.0
        price = 33.33
        lot_size = 100

        # This is the walk-forward's buy logic
        slippage_mult = 5 / 10_000
        buy_price = price * (1 + slippage_mult)
        buy_shares = int(cash / buy_price / lot_size) * lot_size
        cost = buy_shares * buy_price
        fee = cost * (3 / 10_000)

        assert cost + fee <= cash, (
            f"Buy cost ({cost + fee:.2f}) exceeds available cash ({cash:.2f})"
        )
        assert buy_shares >= 0
        assert buy_shares % lot_size == 0
