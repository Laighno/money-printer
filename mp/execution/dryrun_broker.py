"""In-process broker that prints / logs orders without sending anywhere.

Purpose
-------
**This is a pipeline-validation broker, NOT a backtest shadow.**  It exists
to exercise the order-submission code path (auth → risk check → submit →
status callback → position update) on machines where the real QMT client
is unavailable, and to provide a "preview" before flipping the switch to
live execution.

What this is NOT
----------------
- It is NOT a fill simulator.  Limit orders auto-fill at the limit price
  with zero slippage and zero queue model — a real exchange does neither.
- Its NAV / PnL is therefore NOT comparable to either the backtest engine
  (which uses a sqrt market-impact slippage model) or to live QMT fills.
  Do not use this broker's account state for PnL reconciliation.
- If you need a backtest-faithful fill simulator for forward-test PnL
  attribution, write a separate ``BacktestShadowBroker`` — do not retrofit
  this class.
"""
from __future__ import annotations

import uuid
from dataclasses import field
from datetime import datetime
from typing import Literal

from loguru import logger

from .qmt_broker import AccountInfo, OrderResult, OrderStatus, Position


class DryRunBroker:
    """Pretends to be a broker.  Records orders, never sends them."""

    def __init__(
        self,
        cash: float = 0.0,
        positions: list[Position] | None = None,
        autofill: bool = True,
    ):
        self._cash = float(cash)
        self._positions: dict[str, Position] = {
            p.code: p for p in (positions or [])
        }
        self._orders: list[OrderStatus] = []
        self._connected = False
        # If True, "fill" each limit order immediately at its limit price
        # (good for preview / NAV projection).  If False, orders stay pending.
        self.autofill = autofill
        # Print the "not for PnL reconciliation" warning at most once per
        # process — paper_trade loops connect() per cycle and we don't want
        # to spam logs. See docs/dialog/ round 30 (external-review item 2).
        self._not_for_pnl_warned = False

    # ── lifecycle ────────────────────────────────────────────────

    def connect(self) -> bool:
        self._connected = True
        logger.info("DryRunBroker: connected (cash={:.2f}, {} positions)",
                    self._cash, len(self._positions))
        if not self._not_for_pnl_warned:
            logger.warning(
                "DryRunBroker fills at limit with no slippage model; "
                "do NOT use its NAV/PnL for reconciliation against backtest "
                "or live QMT — use a dedicated BacktestShadowBroker for that."
            )
            self._not_for_pnl_warned = True
        return True

    def disconnect(self):
        self._connected = False
        logger.info("DryRunBroker: disconnected")

    def is_connected(self) -> bool:
        return self._connected

    # ── account queries ─────────────────────────────────────────

    def get_account_info(self) -> AccountInfo:
        self._require_connected()
        mv = sum(p.market_value for p in self._positions.values())
        return AccountInfo(
            cash_available=self._cash,
            cash_frozen=0.0,
            market_value=mv,
            total_assets=self._cash + mv,
            updated_at=datetime.now().isoformat(timespec="seconds"),
        )

    def get_positions(self) -> list[Position]:
        self._require_connected()
        return list(self._positions.values())

    def get_orders(self, only_today: bool = True) -> list[OrderStatus]:
        self._require_connected()
        return list(self._orders)

    # ── trading ─────────────────────────────────────────────────

    def place_limit_order(
        self,
        code: str,
        action: Literal["buy", "sell"],
        shares: int,
        limit_price: float,
        order_remark: str = "",
    ) -> OrderResult:
        self._require_connected()
        if shares < 100 or shares % 100 != 0:
            return OrderResult(success=False, error=f"shares={shares} not a valid lot")
        if limit_price <= 0:
            return OrderResult(success=False, error=f"limit_price={limit_price} invalid")

        order_id = uuid.uuid4().hex[:10]
        logger.info(
            "[DRYRUN] {} {} {} 股 @ ¥{:.2f}  (id={})",
            action.upper(), code, shares, limit_price, order_id,
        )

        # Sanity-check cash for buys (after sells would settle T+0)
        if action == "buy":
            cost = shares * limit_price
            if cost > self._cash:
                logger.warning("[DRYRUN] insufficient cash: need ¥{:.2f}, have ¥{:.2f}",
                               cost, self._cash)
                return OrderResult(
                    success=False, error="insufficient cash (dryrun check)",
                    code=code, action=action, shares=shares, limit_price=limit_price,
                )
        else:  # sell
            cur = self._positions.get(code)
            if not cur or cur.shares_available < shares:
                have = cur.shares_available if cur else 0
                logger.warning("[DRYRUN] insufficient shares: want sell {}, have {}",
                               shares, have)
                return OrderResult(
                    success=False, error=f"insufficient shares (have {have})",
                    code=code, action=action, shares=shares, limit_price=limit_price,
                )

        status = "filled" if self.autofill else "pending"
        self._orders.append(OrderStatus(
            order_id=order_id, code=code, action=action,
            shares_submitted=shares,
            shares_filled=shares if self.autofill else 0,
            avg_fill_price=limit_price if self.autofill else 0.0,
            status=status,
        ))

        # If autofill, mutate account/positions to reflect "fill"
        if self.autofill:
            if action == "buy":
                self._cash -= shares * limit_price
                if code in self._positions:
                    p = self._positions[code]
                    new_total = p.shares_total + shares
                    new_avg_cost = (p.avg_cost * p.shares_total + limit_price * shares) / new_total
                    self._positions[code] = Position(
                        code=p.code, name=p.name,
                        shares_total=new_total,
                        shares_available=p.shares_available,   # new lot locked T+1
                        avg_cost=new_avg_cost,
                        market_price=limit_price,
                        market_value=new_total * limit_price,
                    )
                else:
                    self._positions[code] = Position(
                        code=code, name=code,
                        shares_total=shares,
                        shares_available=0,   # T+1 locked
                        avg_cost=limit_price,
                        market_price=limit_price,
                        market_value=shares * limit_price,
                    )
            else:  # sell
                p = self._positions[code]
                proceeds = shares * limit_price
                self._cash += proceeds
                new_total = p.shares_total - shares
                if new_total == 0:
                    del self._positions[code]
                else:
                    self._positions[code] = Position(
                        code=p.code, name=p.name,
                        shares_total=new_total,
                        shares_available=p.shares_available - shares,
                        avg_cost=p.avg_cost,
                        market_price=limit_price,
                        market_value=new_total * limit_price,
                    )

        return OrderResult(
            success=True, order_id=order_id,
            code=code, action=action,
            shares=shares, limit_price=limit_price,
        )

    def cancel_order(self, order_id: str) -> OrderResult:
        self._require_connected()
        for o in self._orders:
            if o.order_id == order_id:
                if o.status in ("filled", "cancelled"):
                    return OrderResult(success=False, order_id=order_id,
                                        error=f"already {o.status}")
                o.status = "cancelled"
                return OrderResult(success=True, order_id=order_id)
        return OrderResult(success=False, order_id=order_id, error="not found")

    def _require_connected(self):
        if not self._connected:
            raise RuntimeError("DryRunBroker not connected — call connect() first")
