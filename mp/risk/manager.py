"""Risk manager - the module that keeps you alive.

Core principle: risk management is about avoiding ruin, not maximizing returns.

Implements:
1. Dynamic position sizing (inverse volatility weighting)
2. Per-position stop-loss
3. Portfolio-level max drawdown circuit breaker
4. Entry/exit rules enforcement
"""

from dataclasses import dataclass, field
from datetime import date

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


class RiskManager:
    def __init__(self, params: RiskParams | None = None, silent: bool = False):
        self.params = params or RiskParams()
        self.positions: dict[str, PositionState] = {}
        self.nav_peak: float = 1.0
        self.nav_current: float = 1.0
        self.circuit_breaker_active: bool = False
        self.silent: bool = silent

    @property
    def portfolio_drawdown(self) -> float:
        if self.nav_peak <= 0:
            return 0.0
        return (self.nav_current / self.nav_peak - 1)

    def calc_position_size(self, board_name: str, volatility_20d: float) -> float:
        """Inverse-volatility position sizing.

        Higher volatility -> smaller position. Targets a consistent risk contribution.
        """
        if volatility_20d <= 0 or np.isnan(volatility_20d):
            return 0.0

        # Target vol contribution per position
        n_positions = max(len(self.positions) + 1, 1)
        target_vol_per_pos = self.params.vol_target / np.sqrt(n_positions)

        # Weight = target_vol / actual_vol, capped
        raw_weight = target_vol_per_pos / (volatility_20d / 100)  # vol_20d is in percent
        weight = min(raw_weight, self.params.max_position_pct)
        weight = max(weight, 0.02)  # minimum 2%

        return round(weight, 4)

    def check_entry(self, board_name: str) -> tuple[bool, str]:
        """Check if a new position can be entered."""
        if self.circuit_breaker_active:
            return False, "Circuit breaker active - max drawdown exceeded"

        if len(self.positions) >= self.params.max_sectors:
            return False, f"Max sectors ({self.params.max_sectors}) reached"

        if board_name in self.positions:
            return False, f"Already holding {board_name}"

        return True, "OK"

    def check_exit(self, board_name: str) -> tuple[bool, str]:
        """Check if a position should be exited (stop-loss or trailing stop)."""
        if board_name not in self.positions:
            return False, "No position"

        pos = self.positions[board_name]

        # Hard stop-loss
        if pos.pnl_pct <= -self.params.stop_loss_pct:
            return True, f"Stop-loss triggered: {pos.pnl_pct:.2%} <= -{self.params.stop_loss_pct:.2%}"

        # Trailing stop from peak
        if pos.drawdown_from_peak <= -self.params.trailing_stop_pct:
            return True, f"Trailing stop triggered: {pos.drawdown_from_peak:.2%} from peak"

        return False, "Hold"

    def update_prices(self, prices: dict[str, float]):
        """Update current prices and check circuit breaker."""
        for name, price in prices.items():
            if name in self.positions:
                pos = self.positions[name]
                pos.current_price = price
                pos.peak_price = max(pos.peak_price, price)

        # Update portfolio NAV (simplified)
        total_pnl = sum(pos.pnl_pct * pos.weight for pos in self.positions.values())
        self.nav_current = self.nav_peak * (1 + total_pnl)
        self.nav_peak = max(self.nav_peak, self.nav_current)

        # Circuit breaker check
        if self.portfolio_drawdown <= -self.params.max_drawdown_pct:
            if not self.circuit_breaker_active:
                if not self.silent:
                    logger.warning(f"CIRCUIT BREAKER: Portfolio drawdown {self.portfolio_drawdown:.2%} exceeds limit")
                self.circuit_breaker_active = True

    def enter_position(self, board_name: str, price: float, weight: float, entry_date: str):
        """Record a new position entry."""
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
        """Record a position exit."""
        pos = self.positions.pop(board_name, None)
        if pos and not self.silent:
            logger.info(f"EXIT {board_name}: pnl={pos.pnl_pct:.2%}, reason={reason}")
        return pos

    def get_status(self) -> pd.DataFrame:
        """Get current portfolio status as DataFrame."""
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
        """Get risk summary."""
        return {
            "n_positions": len(self.positions),
            "total_weight": sum(p.weight for p in self.positions.values()),
            "portfolio_dd": f"{self.portfolio_drawdown:.2%}",
            "circuit_breaker": self.circuit_breaker_active,
            "nav": f"{self.nav_current:.4f}",
        }
