"""Focused unit tests for Portfolio and RiskGuard.

These test the refactored components in isolation — RiskGuard receives
synthetic position dicts, Portfolio is tested without any risk rules.
"""

import unittest

import numpy as np

from mp.risk.manager import Portfolio, PositionState, RiskGuard, RiskParams


def _pos(name="A", entry=100.0, current=100.0, peak=100.0, weight=0.10) -> PositionState:
    """Quick helper to build a PositionState."""
    return PositionState(
        board_name=name, entry_price=entry, entry_date="2024-01-01",
        current_price=current, peak_price=peak, weight=weight,
    )


# ---------------------------------------------------------------------------
# Portfolio tests — pure accounting, no risk rules
# ---------------------------------------------------------------------------

class TestPortfolio(unittest.TestCase):

    def test_enter_and_exit(self):
        p = Portfolio(silent=True)
        p.enter_position("X", 50.0, 0.20, "2024-01-01")
        assert "X" in p.positions
        assert p.positions["X"].entry_price == 50.0

        exited = p.exit_position("X", "test")
        assert exited is not None
        assert exited.board_name == "X"
        assert "X" not in p.positions

    def test_exit_nonexistent_returns_none(self):
        p = Portfolio(silent=True)
        assert p.exit_position("Z", "ghost") is None

    def test_nav_goes_up(self):
        p = Portfolio(silent=True)
        p.enter_position("A", 100.0, 0.50, "2024-01-01")
        p.update_prices({"A": 110.0})
        assert p.nav_current > 1.0
        expected = 1.0 * (1 + 0.50 * (110 / 100 - 1))  # 1.05
        assert abs(p.nav_current - expected) < 1e-9

    def test_nav_goes_down(self):
        p = Portfolio(silent=True)
        p.enter_position("A", 100.0, 0.50, "2024-01-01")
        p.update_prices({"A": 90.0})
        expected = 1.0 * (1 + 0.50 * (90 / 100 - 1))  # 0.95
        assert abs(p.nav_current - expected) < 1e-9

    def test_nav_stable_no_positions(self):
        p = Portfolio(silent=True)
        p.update_prices({})
        assert p.nav_current == 1.0

    def test_drawdown_property(self):
        p = Portfolio(silent=True)
        p.enter_position("A", 100.0, 1.0, "2024-01-01")
        p.update_prices({"A": 120.0})
        p.update_prices({"A": 96.0})  # 96/120 = 0.8 → dd = -20%
        assert p.portfolio_drawdown < 0
        # nav after up: 1*(1+1*(120/100-1)) = 1.2, peak=1.2
        # nav after down: 1.2*(1+1*(96/120-1)) = 1.2*0.8 = 0.96
        assert abs(p.portfolio_drawdown - (0.96 / 1.2 - 1)) < 1e-9

    def test_get_status_empty(self):
        p = Portfolio(silent=True)
        df = p.get_status()
        assert len(df) == 0
        assert "board_name" in df.columns

    def test_get_status_with_positions(self):
        p = Portfolio(silent=True)
        p.enter_position("A", 100.0, 0.30, "2024-01-01")
        p.enter_position("B", 50.0, 0.20, "2024-01-02")
        df = p.get_status()
        assert len(df) == 2
        assert set(df["board_name"]) == {"A", "B"}

    def test_get_summary(self):
        p = Portfolio(silent=True)
        p.enter_position("A", 100.0, 0.30, "2024-01-01")
        s = p.get_summary()
        assert s["n_positions"] == 1
        assert abs(s["total_weight"] - 0.30) < 1e-9
        assert "circuit_breaker" not in s  # Portfolio knows nothing about risk


# ---------------------------------------------------------------------------
# RiskGuard tests — pure rules, synthetic positions
# ---------------------------------------------------------------------------

class TestRiskGuard(unittest.TestCase):

    def test_check_entry_ok(self):
        g = RiskGuard(RiskParams(max_sectors=3))
        positions = {"A": _pos("A"), "B": _pos("B")}
        ok, msg = g.check_entry("C", positions)
        assert ok is True

    def test_check_entry_max_sectors(self):
        g = RiskGuard(RiskParams(max_sectors=2))
        positions = {"A": _pos("A"), "B": _pos("B")}
        ok, msg = g.check_entry("C", positions)
        assert ok is False
        assert "Max sectors" in msg

    def test_check_entry_already_held(self):
        g = RiskGuard(RiskParams(max_sectors=5))
        positions = {"A": _pos("A")}
        ok, msg = g.check_entry("A", positions)
        assert ok is False
        assert "Already holding" in msg

    def test_check_entry_circuit_breaker(self):
        g = RiskGuard(RiskParams())
        g.circuit_breaker_active = True
        ok, msg = g.check_entry("A", {})
        assert ok is False
        assert "Circuit breaker" in msg

    def test_check_exit_stop_loss(self):
        g = RiskGuard(RiskParams(stop_loss_pct=0.05))
        positions = {"A": _pos("A", entry=100.0, current=94.0, peak=100.0)}
        should_exit, reason = g.check_exit("A", positions)
        assert should_exit is True
        assert "Stop-loss" in reason

    def test_check_exit_trailing_stop(self):
        g = RiskGuard(RiskParams(stop_loss_pct=0.50, trailing_stop_pct=0.10))
        # Price went from 100→150→130: pnl = +30% (no stop-loss), dd from peak = -13.3%
        positions = {"A": _pos("A", entry=100.0, current=130.0, peak=150.0)}
        should_exit, reason = g.check_exit("A", positions)
        assert should_exit is True
        assert "Trailing stop" in reason

    def test_check_exit_hold(self):
        g = RiskGuard(RiskParams(stop_loss_pct=0.10, trailing_stop_pct=0.15))
        positions = {"A": _pos("A", entry=100.0, current=95.0, peak=100.0)}
        should_exit, _ = g.check_exit("A", positions)
        assert should_exit is False

    def test_check_exit_nonexistent(self):
        g = RiskGuard(RiskParams())
        should_exit, _ = g.check_exit("Z", {})
        assert should_exit is False

    def test_position_size_basic(self):
        g = RiskGuard(RiskParams(vol_target=0.15, max_position_pct=0.20))
        w = g.calc_position_size(0, 20.0)
        assert 0.02 <= w <= 0.20

    def test_position_size_high_vol_smaller(self):
        g = RiskGuard(RiskParams(vol_target=0.15, max_position_pct=0.50))
        w_low = g.calc_position_size(0, 10.0)
        w_high = g.calc_position_size(0, 40.0)
        assert w_high < w_low

    def test_position_size_zero_vol(self):
        g = RiskGuard(RiskParams())
        assert g.calc_position_size(0, 0.0) == 0.0
        assert g.calc_position_size(0, float("nan")) == 0.0

    def test_circuit_breaker_latches(self):
        g = RiskGuard(RiskParams(max_drawdown_pct=0.10))
        g.check_circuit_breaker(-0.05, silent=True)
        assert g.circuit_breaker_active is False

        g.check_circuit_breaker(-0.12, silent=True)
        assert g.circuit_breaker_active is True

        # Once active, stays active even if drawdown recovers
        g.check_circuit_breaker(-0.02, silent=True)
        assert g.circuit_breaker_active is True


# ---------------------------------------------------------------------------
# Integration: facade wires Portfolio + RiskGuard correctly
# ---------------------------------------------------------------------------

class TestFacadeWiring(unittest.TestCase):

    def test_update_prices_triggers_circuit_breaker(self):
        """Facade.update_prices must call Portfolio NAV update THEN RiskGuard CB check."""
        from mp.risk.manager import RiskManager
        rm = RiskManager(RiskParams(max_drawdown_pct=0.10), silent=True)
        rm.enter_position("A", 100.0, 1.0, "2024-01-01")
        rm.update_prices({"A": 85.0})  # -15% > 10% threshold
        assert rm.circuit_breaker_active is True

    def test_get_summary_includes_circuit_breaker(self):
        from mp.risk.manager import RiskManager
        rm = RiskManager(RiskParams(), silent=True)
        s = rm.get_summary()
        assert "circuit_breaker" in s
        assert s["circuit_breaker"] is False


if __name__ == "__main__":
    unittest.main()
