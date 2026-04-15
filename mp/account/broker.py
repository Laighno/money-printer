"""Unified simulated broker for A-share backtesting and paper trading.

Consolidates trade execution logic from three previous implementations:
- mp/execution/trader.py (SimulatedTrader)
- mp/backtest/ml_backtest.py (inline dicts)
- scripts/walk_forward_backtest.py (inline dicts + stamp tax)

Handles: slippage, commission, A-share stamp tax (sell-side, date-dependent),
lot rounding (100-share lots), cash management, and trade logging.
"""

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger


@dataclass
class BrokerPosition:
    code: str
    shares: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    peak_price: float = 0.0
    entry_date: str = ""

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.avg_cost <= 0:
            return 0.0
        return self.current_price / self.avg_cost - 1

    @property
    def drawdown_from_peak(self) -> float:
        if self.peak_price <= 0:
            return 0.0
        return self.current_price / self.peak_price - 1


@dataclass
class FeeSchedule:
    slippage_bps: float = 5
    commission_bps: float = 3
    stamp_tax_bps_old: float = 10       # sell-side only, before cut date
    stamp_tax_bps_new: float = 5        # sell-side only, from cut date
    stamp_tax_cut_date: str = "2023-08-28"

    def buy_exec_price(self, price: float) -> float:
        return price * (1 + self.slippage_bps / 10_000)

    def sell_exec_price(self, price: float) -> float:
        return price * (1 - self.slippage_bps / 10_000)

    def buy_fee(self, cost: float) -> float:
        return cost * self.commission_bps / 10_000

    def sell_fee(self, proceeds: float, date: str = "") -> float:
        commission = proceeds * self.commission_bps / 10_000
        stamp_rate = self._stamp_tax_bps(date)
        stamp = proceeds * stamp_rate / 10_000
        return commission + stamp

    def _stamp_tax_bps(self, date: str) -> float:
        if not date:
            return self.stamp_tax_bps_new
        date_str = str(date)[:10]
        if date_str >= self.stamp_tax_cut_date:
            return self.stamp_tax_bps_new
        return self.stamp_tax_bps_old


LOT_SIZE = 100


class SimulatedBroker:
    """Simulated A-share broker with realistic cost modeling."""

    def __init__(self, initial_capital: float, fees: FeeSchedule | None = None, silent: bool = False):
        self.cash: float = initial_capital
        self.initial_capital: float = initial_capital
        self.fees: FeeSchedule = fees or FeeSchedule()
        self.positions: dict[str, BrokerPosition] = {}
        self.trade_log: list[dict] = []
        self.silent: bool = silent

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def buy(self, code: str, price: float, target_value: float | None = None,
            shares: int | None = None, date: str = "", action: str = "BUY") -> dict | None:
        if price <= 0:
            return None

        exec_price = self.fees.buy_exec_price(price)

        if shares is not None:
            buy_shares = int(shares / LOT_SIZE) * LOT_SIZE
        elif target_value is not None:
            if code in self.positions:
                current_value = self.positions[code].shares * price
                delta = target_value - current_value
                if delta <= exec_price * LOT_SIZE * 0.5:
                    return None
            else:
                delta = target_value
            buy_shares = int(delta / exec_price / LOT_SIZE) * LOT_SIZE
        else:
            return None

        if buy_shares <= 0:
            return None

        cost = buy_shares * exec_price
        fee = self.fees.buy_fee(cost)
        total_cost = cost + fee

        if total_cost > self.cash:
            buy_shares = int(self.cash / exec_price / LOT_SIZE) * LOT_SIZE
            if buy_shares <= 0:
                return None
            cost = buy_shares * exec_price
            fee = self.fees.buy_fee(cost)
            total_cost = cost + fee

        self.cash -= total_cost

        if code in self.positions:
            pos = self.positions[code]
            total_shares = pos.shares + buy_shares
            pos.avg_cost = (pos.avg_cost * pos.shares + exec_price * buy_shares) / total_shares
            pos.shares = total_shares
            pos.current_price = price
            pos.peak_price = max(pos.peak_price, price)
        else:
            self.positions[code] = BrokerPosition(
                code=code,
                shares=buy_shares,
                avg_cost=exec_price,
                current_price=price,
                peak_price=price,
                entry_date=str(date),
            )

        trade = {
            "date": date, "code": code, "action": action,
            "shares": buy_shares, "price": exec_price, "value": total_cost,
        }
        self.trade_log.append(trade)
        if not self.silent:
            logger.info("BUY {}: {}shares @ {:.2f}, cost={:.0f}", code, buy_shares, exec_price, total_cost)
        return trade

    def sell(self, code: str, price: float, shares: int | None = None,
             date: str = "", action: str = "SELL") -> dict | None:
        if code not in self.positions:
            return None
        pos = self.positions[code]
        if pos.shares <= 0:
            return None
        if price <= 0:
            return None

        exec_price = self.fees.sell_exec_price(price)

        if shares is None:
            sell_shares = pos.shares
        else:
            sell_shares = int(shares / LOT_SIZE) * LOT_SIZE
            sell_shares = min(sell_shares, pos.shares)

        if sell_shares <= 0:
            return None

        proceeds = sell_shares * exec_price
        fee = self.fees.sell_fee(proceeds, str(date))
        net_proceeds = proceeds - fee

        self.cash += net_proceeds
        pos.shares -= sell_shares

        if pos.shares <= 0:
            del self.positions[code]

        trade = {
            "date": date, "code": code, "action": action,
            "shares": sell_shares, "price": exec_price, "value": net_proceeds,
        }
        self.trade_log.append(trade)
        if not self.silent:
            logger.info("SELL {}: {}shares @ {:.2f}, net={:.0f}", code, sell_shares, exec_price, net_proceeds)
        return trade

    def update_prices(self, prices: dict[str, float]) -> None:
        for code, price in prices.items():
            if code in self.positions:
                pos = self.positions[code]
                pos.current_price = price
                if price > pos.peak_price:
                    pos.peak_price = price

    def get_holdings_df(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame(columns=["code", "shares", "avg_cost", "current_price", "market_value", "pnl_pct", "entry_date"])
        rows = []
        for pos in self.positions.values():
            rows.append({
                "code": pos.code,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "pnl_pct": pos.pnl_pct,
                "entry_date": pos.entry_date,
            })
        return pd.DataFrame(rows)
