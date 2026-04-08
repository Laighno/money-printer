"""Trade execution - generate orders and simulate/execute trades."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd
from loguru import logger


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


@dataclass
class Position:
    code: str
    shares: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl(self) -> float:
        return self.shares * (self.current_price - self.avg_cost)


class SimulatedTrader:
    """Simulated trading engine for paper trading."""

    def __init__(self, initial_capital: float = 1_000_000, slippage_bps: int = 5, commission_bps: int = 3):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.positions: dict[str, Position] = {}
        self.order_history: list[Order] = []

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def generate_orders(self, target_portfolio: pd.DataFrame, current_prices: dict[str, float]) -> list[Order]:
        """Generate orders to rebalance from current positions to target.

        Args:
            target_portfolio: DataFrame with 'code' and 'weight' columns
            current_prices: Dict of code -> latest price
        """
        orders = []
        total = self.total_value
        target_map = dict(zip(target_portfolio["code"], target_portfolio["weight"]))

        # Sell positions not in target or over-weight
        for code, pos in self.positions.items():
            target_w = target_map.get(code, 0.0)
            current_w = pos.market_value / total if total > 0 else 0
            if target_w < current_w:
                orders.append(Order(code=code, side=OrderSide.SELL, target_weight=target_w))

        # Buy new positions or under-weight
        for code, target_w in target_map.items():
            current_w = 0.0
            if code in self.positions:
                current_w = self.positions[code].market_value / total if total > 0 else 0
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
            slip = price * self.slippage_bps / 10000

            if order.side == OrderSide.SELL:
                pos = self.positions.get(order.code)
                if not pos or pos.shares == 0:
                    continue
                current_value = pos.market_value
                sell_value = current_value - target_value
                sell_shares = int(sell_value / price / 100) * 100  # Round to 100 shares (A-share lot)
                if sell_shares <= 0:
                    continue
                exec_price = price - slip
                proceeds = sell_shares * exec_price
                commission_fee = proceeds * self.commission_bps / 10000
                self.cash += proceeds - commission_fee
                pos.shares -= sell_shares
                if pos.shares <= 0:
                    del self.positions[order.code]

            elif order.side == OrderSide.BUY:
                current_value = self.positions[order.code].market_value if order.code in self.positions else 0
                buy_value = target_value - current_value
                exec_price = price + slip
                buy_shares = int(buy_value / exec_price / 100) * 100
                if buy_shares <= 0:
                    continue
                cost = buy_shares * exec_price
                commission_fee = cost * self.commission_bps / 10000
                total_cost = cost + commission_fee
                if total_cost > self.cash:
                    buy_shares = int(self.cash / exec_price / 100) * 100
                    if buy_shares <= 0:
                        continue
                    cost = buy_shares * exec_price
                    commission_fee = cost * self.commission_bps / 10000
                    total_cost = cost + commission_fee
                self.cash -= total_cost

                if order.code in self.positions:
                    pos = self.positions[order.code]
                    total_shares = pos.shares + buy_shares
                    pos.avg_cost = (pos.avg_cost * pos.shares + exec_price * buy_shares) / total_shares
                    pos.shares = total_shares
                else:
                    self.positions[order.code] = Position(code=order.code, shares=buy_shares, avg_cost=exec_price, current_price=price)

            order.price = price
            order.timestamp = datetime.now()
            self.order_history.append(order)

        # Update current prices
        for code, pos in self.positions.items():
            if code in prices:
                pos.current_price = prices[code]

        logger.info(f"Executed {len(orders)} orders. Cash: {self.cash:,.0f}, Total: {self.total_value:,.0f}")

    def get_holdings_df(self) -> pd.DataFrame:
        """Return current holdings as DataFrame."""
        if not self.positions:
            return pd.DataFrame(columns=["code", "shares", "avg_cost", "current_price", "market_value", "pnl"])
        rows = []
        for pos in self.positions.values():
            rows.append({
                "code": pos.code,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "pnl": pos.pnl,
            })
        return pd.DataFrame(rows)
