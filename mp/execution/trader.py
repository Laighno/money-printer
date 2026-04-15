"""Trade execution - generate orders and simulate/execute trades.

SimulatedTrader delegates to SimulatedBroker for all cash/position/fee logic.
It adds order generation (target portfolio → order list) on top.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd
from loguru import logger

from mp.account.broker import FeeSchedule, SimulatedBroker


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    code: str
    side: OrderSide
    target_weight: float
    price: float = 0.0
    amount: float = 0.0
    timestamp: datetime | None = None


class SimulatedTrader:
    """Simulated trading engine for paper trading.

    Wraps SimulatedBroker for execution, adds order generation logic.
    """

    def __init__(self, initial_capital: float = 1_000_000, slippage_bps: int = 5, commission_bps: int = 3):
        fees = FeeSchedule(slippage_bps=slippage_bps, commission_bps=commission_bps,
                           stamp_tax_bps_old=0, stamp_tax_bps_new=0)
        self._broker = SimulatedBroker(initial_capital, fees, silent=True)
        self.order_history: list[Order] = []

    @property
    def initial_capital(self) -> float:
        return self._broker.initial_capital

    @property
    def cash(self) -> float:
        return self._broker.cash

    @property
    def total_value(self) -> float:
        return self._broker.total_value

    @property
    def positions(self) -> dict:
        return self._broker.positions

    def generate_orders(self, target_portfolio: pd.DataFrame, current_prices: dict[str, float]) -> list[Order]:
        """Generate orders to rebalance from current positions to target.

        Args:
            target_portfolio: DataFrame with 'code' and 'weight' columns
            current_prices: Dict of code -> latest price
        """
        # Update prices first so market_value is current
        self._broker.update_prices(current_prices)

        orders = []
        total = self.total_value
        target_map = dict(zip(target_portfolio["code"], target_portfolio["weight"]))

        # Sell positions not in target or over-weight
        for code, pos in self._broker.positions.items():
            target_w = target_map.get(code, 0.0)
            current_w = pos.market_value / total if total > 0 else 0
            if target_w < current_w:
                orders.append(Order(code=code, side=OrderSide.SELL, target_weight=target_w))

        # Buy new positions or under-weight
        for code, target_w in target_map.items():
            current_w = 0.0
            if code in self._broker.positions:
                current_w = self._broker.positions[code].market_value / total if total > 0 else 0
            if target_w > current_w:
                orders.append(Order(code=code, side=OrderSide.BUY, target_weight=target_w))

        return orders

    def execute_orders(self, orders: list[Order], prices: dict[str, float]):
        """Execute orders at given prices with slippage and commission."""
        total = self.total_value

        # Execute sells first to free up cash
        sells = [o for o in orders if o.side == OrderSide.SELL]
        buys = [o for o in orders if o.side == OrderSide.BUY]

        for order in sells + buys:
            price = prices.get(order.code)
            if price is None or price <= 0:
                continue

            target_value = order.target_weight * total

            if order.side == OrderSide.SELL:
                if order.code not in self._broker.positions:
                    continue
                pos = self._broker.positions[order.code]
                if pos.shares == 0:
                    continue
                current_value = pos.market_value
                sell_value = current_value - target_value
                exec_price = self._broker.fees.sell_exec_price(price)
                sell_shares = int(sell_value / price / 100) * 100
                if sell_shares <= 0:
                    continue
                self._broker.sell(order.code, price, shares=sell_shares)

            elif order.side == OrderSide.BUY:
                current_value = self._broker.positions[order.code].market_value if order.code in self._broker.positions else 0
                buy_value = target_value - current_value
                if buy_value <= 0:
                    continue
                self._broker.buy(order.code, price, target_value=target_value)

            order.price = price
            order.timestamp = datetime.now()
            self.order_history.append(order)

        # Update current prices
        self._broker.update_prices(prices)

        logger.info(f"Executed {len(orders)} orders. Cash: {self.cash:,.0f}, Total: {self.total_value:,.0f}")

    def get_holdings_df(self) -> pd.DataFrame:
        """Return current holdings as DataFrame."""
        return self._broker.get_holdings_df()
