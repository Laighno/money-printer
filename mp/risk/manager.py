"""Risk manager - the module that keeps you alive.

Architecture:
- Portfolio: position bookkeeping and NAV accounting (no risk knowledge)
- RiskGuard: pure risk rules — stop-loss, circuit breaker, position sizing
- RiskManager: thin facade that owns both, preserves the original API

Core principle: risk management is about avoiding ruin, not maximizing returns.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class RiskParams:
    max_position_pct: float = 0.20     # max weight per sector
    stop_loss_pct: float = 0.08        # per-position stop-loss
    trailing_stop_pct: float = 0.12    # trailing stop from peak
    max_drawdown_pct: float = 0.15     # portfolio-level circuit breaker
    max_sectors: int = 5               # max concurrent sector holdings
    vol_target: float = 0.15           # target annualized portfolio vol


@dataclass
class PositionState:
    board_name: str
    entry_price: float
    entry_date: str
    current_price: float = 0.0
    peak_price: float = 0.0
    weight: float = 0.0

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price / self.entry_price - 1)

    @property
    def drawdown_from_peak(self) -> float:
        if self.peak_price <= 0:
            return 0.0
        return (self.current_price / self.peak_price - 1)


# ---------------------------------------------------------------------------
# Portfolio — position bookkeeping and NAV accounting
# ---------------------------------------------------------------------------

class Portfolio:
    def __init__(self, silent: bool = False):
        self.positions: dict[str, PositionState] = {}
        self.nav_peak: float = 1.0
        self.nav_current: float = 1.0
        self._prev_prices: dict[str, float] = {}
        self.silent: bool = silent

    @property
    def portfolio_drawdown(self) -> float:
        if self.nav_peak <= 0:
            return 0.0
        return (self.nav_current / self.nav_peak - 1)

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update position prices and recalculate NAV incrementally."""
        for name, price in prices.items():
            if name in self.positions:
                pos = self.positions[name]
                pos.current_price = price
                pos.peak_price = max(pos.peak_price, price)

        daily_ret = 0.0
        for pos in self.positions.values():
            prev_p = self._prev_prices.get(pos.board_name, pos.entry_price)
            if prev_p > 0 and pos.current_price > 0:
                daily_ret += (pos.current_price / prev_p - 1) * pos.weight
        self.nav_current *= (1 + daily_ret)
        self.nav_peak = max(self.nav_peak, self.nav_current)
        self._prev_prices = {pos.board_name: pos.current_price for pos in self.positions.values()}

    def enter_position(self, board_name: str, price: float, weight: float, entry_date: str) -> None:
        self.positions[board_name] = PositionState(
            board_name=board_name,
            entry_price=price,
            entry_date=entry_date,
            current_price=price,
            peak_price=price,
            weight=weight,
        )
        if not self.silent:
            logger.info(f"ENTER {board_name}: price={price:.2f}, weight={weight:.2%}, date={entry_date}")

    def exit_position(self, board_name: str, reason: str) -> PositionState | None:
        pos = self.positions.pop(board_name, None)
        if pos and not self.silent:
            logger.info(f"EXIT {board_name}: pnl={pos.pnl_pct:.2%}, reason={reason}")
        return pos

    def get_status(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame(columns=["board_name", "entry_price", "current_price", "weight", "pnl_pct", "dd_from_peak", "entry_date"])
        rows = []
        for pos in self.positions.values():
            rows.append({
                "board_name": pos.board_name,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "weight": pos.weight,
                "pnl_pct": pos.pnl_pct,
                "dd_from_peak": pos.drawdown_from_peak,
                "entry_date": pos.entry_date,
            })
        return pd.DataFrame(rows)

    def get_summary(self) -> dict:
        return {
            "n_positions": len(self.positions),
            "total_weight": sum(p.weight for p in self.positions.values()),
            "portfolio_dd": f"{self.portfolio_drawdown:.2%}",
            "nav": f"{self.nav_current:.4f}",
        }


# ---------------------------------------------------------------------------
# RiskGuard — pure risk rules, no position state
# ---------------------------------------------------------------------------

class RiskGuard:
    def __init__(self, params: RiskParams):
        self.params = params
        self.circuit_breaker_active: bool = False

    def check_entry(self, board_name: str, positions: dict[str, PositionState]) -> tuple[bool, str]:
        if self.circuit_breaker_active:
            return False, "Circuit breaker active - max drawdown exceeded"
        if len(positions) >= self.params.max_sectors:
            return False, f"Max sectors ({self.params.max_sectors}) reached"
        if board_name in positions:
            return False, f"Already holding {board_name}"
        return True, "OK"

    def check_exit(self, board_name: str, positions: dict[str, PositionState]) -> tuple[bool, str]:
        if board_name not in positions:
            return False, "No position"
        pos = positions[board_name]
        if pos.pnl_pct <= -self.params.stop_loss_pct:
            return True, f"Stop-loss triggered: {pos.pnl_pct:.2%} <= -{self.params.stop_loss_pct:.2%}"
        if pos.drawdown_from_peak <= -self.params.trailing_stop_pct:
            return True, f"Trailing stop triggered: {pos.drawdown_from_peak:.2%} from peak"
        return False, "Hold"

    def calc_position_size(self, n_current_positions: int, volatility_20d: float) -> float:
        """Inverse-volatility position sizing."""
        if volatility_20d <= 0 or np.isnan(volatility_20d):
            return 0.0
        n_positions = max(n_current_positions + 1, 1)
        target_vol_per_pos = self.params.vol_target / np.sqrt(n_positions)
        raw_weight = target_vol_per_pos / (volatility_20d / 100)
        weight = min(raw_weight, self.params.max_position_pct)
        weight = max(weight, 0.02)
        return round(weight, 4)

    def check_circuit_breaker(self, portfolio_drawdown: float, silent: bool = False) -> None:
        if portfolio_drawdown <= -self.params.max_drawdown_pct:
            if not self.circuit_breaker_active:
                if not silent:
                    logger.warning(f"CIRCUIT BREAKER: Portfolio drawdown {portfolio_drawdown:.2%} exceeds limit")
                self.circuit_breaker_active = True


# ---------------------------------------------------------------------------
# RiskManager — facade that delegates to Portfolio + RiskGuard
# ---------------------------------------------------------------------------

class RiskManager:
    def __init__(self, params: RiskParams | None = None, silent: bool = False):
        self.params = params or RiskParams()
        self._portfolio = Portfolio(silent=silent)
        self._guard = RiskGuard(self.params)
        self.silent = silent

    @property
    def positions(self) -> dict[str, PositionState]:
        return self._portfolio.positions

    @property
    def nav_current(self) -> float:
        return self._portfolio.nav_current

    @property
    def nav_peak(self) -> float:
        return self._portfolio.nav_peak

    @property
    def circuit_breaker_active(self) -> bool:
        return self._guard.circuit_breaker_active

    @property
    def portfolio_drawdown(self) -> float:
        return self._portfolio.portfolio_drawdown

    def update_prices(self, prices: dict[str, float]) -> None:
        self._portfolio.update_prices(prices)
        self._guard.check_circuit_breaker(self._portfolio.portfolio_drawdown, self.silent)

    def enter_position(self, board_name: str, price: float, weight: float, entry_date: str) -> None:
        self._portfolio.enter_position(board_name, price, weight, entry_date)

    def exit_position(self, board_name: str, reason: str) -> PositionState | None:
        return self._portfolio.exit_position(board_name, reason)

    def check_entry(self, board_name: str) -> tuple[bool, str]:
        return self._guard.check_entry(board_name, self._portfolio.positions)

    def check_exit(self, board_name: str) -> tuple[bool, str]:
        return self._guard.check_exit(board_name, self._portfolio.positions)

    def calc_position_size(self, board_name: str, volatility_20d: float) -> float:
        return self._guard.calc_position_size(len(self._portfolio.positions), volatility_20d)

    def get_status(self) -> pd.DataFrame:
        return self._portfolio.get_status()

    def get_summary(self) -> dict:
        summary = self._portfolio.get_summary()
        summary["circuit_breaker"] = self._guard.circuit_breaker_active
        return summary
