"""QMT (迅投量化交易终端) live broker integration via xtquant SDK.

Background
----------
QMT is the desktop quant terminal supported by ~30 Chinese brokers
including 国金证券 (10 万 asset threshold).  It exposes a Python SDK
(``xtquant``) that the desktop client makes available locally —
ie. you import ``xtquant`` from a Python process running on the
same machine as the QMT client, and the client mediates all order
placement / account queries to the broker.

This module wraps the xtquant API surface we need (account query,
limit order placement, order status, cancellation) behind a small
synchronous interface so the rest of the system doesn't depend on
xtquant being importable (it isn't on macOS — QMT only ships on
Windows).

Requires
--------
- QMT desktop client installed and logged in
- xtquant Python package available (ships with QMT installer)
- Account ID and userdata_path of the running QMT instance

Usage
-----
    broker = QMTBroker(
        account_id="<your account>",
        qmt_userdata_path=r"C:\\国金证券QMT交易端\\userdata_mini",
    )
    broker.connect()
    asset = broker.get_account_info()
    positions = broker.get_positions()
    order_id = broker.place_limit_order(
        code="000539", action="sell", shares=18400, limit_price=6.39
    )
    broker.disconnect()
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

from loguru import logger


# ──────────────────────────────────────────────────────────────────
# Data classes (broker-agnostic — usable for tests without xtquant)
# ──────────────────────────────────────────────────────────────────

@dataclass
class AccountInfo:
    cash_available: float          # 可用资金
    cash_frozen: float             # 冻结资金（在途订单占用）
    market_value: float            # 持仓市值
    total_assets: float            # 总资产 = cash_available + cash_frozen + market_value
    updated_at: str                # ISO timestamp


@dataclass
class Position:
    code: str                      # 6-digit code, no exchange suffix
    name: str
    shares_total: int              # 持仓股数
    shares_available: int          # 可用股数 (T+1 锁定后剩余)
    avg_cost: float
    market_price: float
    market_value: float


@dataclass
class OrderResult:
    """Returned by place_limit_order / cancel_order."""
    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None
    # Echo of submitted params for audit trail
    code: Optional[str] = None
    action: Optional[str] = None
    shares: Optional[int] = None
    limit_price: Optional[float] = None


@dataclass
class OrderStatus:
    """Snapshot of an order's lifecycle."""
    order_id: str
    code: str
    action: Literal["buy", "sell"]
    shares_submitted: int
    shares_filled: int
    avg_fill_price: float
    status: Literal["pending", "partial", "filled", "cancelled", "rejected"]
    error_msg: Optional[str] = None


# ──────────────────────────────────────────────────────────────────
# Code formatting
# ──────────────────────────────────────────────────────────────────

def _xt_code(code: str) -> str:
    """6-digit code → xtquant format with exchange suffix.

    Exchange routing rules (verified against xtquant docs):
      - 60x, 688/689 (上证主板 / 科创板) → .SH
      - 00x, 30x, 002, 003 (深证 / 创业板) → .SZ
      - 4x, 8x (北交所)                → .BJ
    """
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "69")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    # 00x / 30x / 002 / 003 default Shenzhen
    return f"{code}.SZ"


def _plain_code(xt_code: str) -> str:
    """xtquant '000539.SZ' → plain '000539'."""
    return xt_code.split(".", 1)[0]


# ──────────────────────────────────────────────────────────────────
# Callback (xtquant pushes async events here)
# ──────────────────────────────────────────────────────────────────

class _QMTEventCallback:
    """Wraps xtquant.XtQuantTraderCallback.

    We keep this minimal — we don't react to events live, we just
    log them.  The broker queries state synchronously when needed.

    Defined as a plain class so this module is importable without
    xtquant installed; QMTBroker.connect() promotes it to a real
    XtQuantTraderCallback subclass at runtime.
    """

    def __init__(self, broker: "QMTBroker"):
        self._broker = broker

    def on_connected(self):
        logger.info("QMT: connected")

    def on_disconnected(self):
        logger.warning("QMT: disconnected (will need reconnect)")

    def on_stock_order(self, order):
        logger.debug("QMT order update: id={} status={} filled={}/{}",
                     getattr(order, "order_id", "?"),
                     getattr(order, "order_status", "?"),
                     getattr(order, "traded_volume", 0),
                     getattr(order, "order_volume", 0))

    def on_stock_trade(self, trade):
        logger.info("QMT TRADE: code={} side={} qty={} @ {:.3f}",
                    getattr(trade, "stock_code", "?"),
                    getattr(trade, "order_type", "?"),
                    getattr(trade, "traded_volume", 0),
                    getattr(trade, "traded_price", 0.0))

    def on_order_error(self, order_error):
        logger.error("QMT order error: id={} code={} msg={}",
                     getattr(order_error, "order_id", "?"),
                     getattr(order_error, "error_id", "?"),
                     getattr(order_error, "error_msg", "?"))

    def on_cancel_error(self, cancel_error):
        logger.error("QMT cancel error: {}", cancel_error)

    def on_order_stock_async_response(self, response):
        logger.debug("QMT async order response: {}", response)


# ──────────────────────────────────────────────────────────────────
# Main broker class
# ──────────────────────────────────────────────────────────────────

class QMTBroker:
    """Thin synchronous wrapper around xtquant.

    Lazy-imports xtquant inside connect() so this module is importable
    without QMT installed (eg. for unit tests on macOS).
    """

    def __init__(
        self,
        account_id: str,
        qmt_userdata_path: str,
        session_id: Optional[int] = None,
    ):
        self.account_id = account_id
        self.qmt_userdata_path = qmt_userdata_path
        # session_id must be unique per process; default to time+uuid hash
        if session_id is None:
            session_id = int(time.time() * 1000) % 1_000_000_000
        self.session_id = session_id
        self._trader = None       # XtQuantTrader instance
        self._account = None      # StockAccount instance
        self._connected = False
        self._callback = None

    # ── lifecycle ────────────────────────────────────────────────

    def connect(self) -> bool:
        """Open QMT connection.  Raises ImportError on macOS / no QMT."""
        try:
            from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
            from xtquant.xttype import StockAccount
        except ImportError as e:
            raise ImportError(
                "xtquant SDK not available. Install QMT desktop client "
                "(Windows only) and ensure xtquant is on PYTHONPATH. "
                f"Original: {e}"
            ) from e

        self._trader = XtQuantTrader(self.qmt_userdata_path, self.session_id)
        # Promote callback to real XtQuantTraderCallback subclass
        cb_methods = {
            name: getattr(_QMTEventCallback, name)
            for name in dir(_QMTEventCallback)
            if name.startswith("on_")
        }
        ConcreteCallback = type(
            "ConcreteCallback", (XtQuantTraderCallback,), cb_methods,
        )
        self._callback = ConcreteCallback()
        # Bind broker into callback so it can update state
        self._callback._broker = self
        self._trader.register_callback(self._callback)
        self._trader.start()
        rc = self._trader.connect()
        if rc != 0:
            logger.error("QMT connect failed rc={}", rc)
            return False
        self._account = StockAccount(self.account_id, "STOCK")
        sub_rc = self._trader.subscribe(self._account)
        if sub_rc != 0:
            logger.error("QMT subscribe account failed rc={}", sub_rc)
            return False
        self._connected = True
        logger.info("QMT: connected account={} session={}",
                    self.account_id, self.session_id)
        return True

    def disconnect(self):
        if self._trader is not None:
            self._trader.stop()
            self._connected = False
            logger.info("QMT: disconnected")

    def is_connected(self) -> bool:
        return self._connected

    # ── account queries ─────────────────────────────────────────

    def get_account_info(self) -> AccountInfo:
        """Query account asset snapshot."""
        self._require_connected()
        from datetime import datetime
        asset = self._trader.query_stock_asset(self._account)
        if asset is None:
            raise RuntimeError("QMT query_stock_asset returned None")
        return AccountInfo(
            cash_available=float(asset.cash),
            cash_frozen=float(asset.frozen_cash),
            market_value=float(asset.market_value),
            total_assets=float(asset.total_asset),
            updated_at=datetime.now().isoformat(timespec="seconds"),
        )

    def get_positions(self) -> list[Position]:
        """Query all current positions."""
        self._require_connected()
        raw = self._trader.query_stock_positions(self._account) or []
        out: list[Position] = []
        for p in raw:
            out.append(Position(
                code=_plain_code(p.stock_code),
                name=getattr(p, "stock_name", ""),
                shares_total=int(p.volume),
                shares_available=int(p.can_use_volume),
                avg_cost=float(p.open_price),
                market_price=float(getattr(p, "market_price", 0.0)),
                market_value=float(p.market_value),
            ))
        return out

    def get_orders(self, only_today: bool = True) -> list[OrderStatus]:
        """Query today's orders (or all if only_today=False)."""
        self._require_connected()
        raw = self._trader.query_stock_orders(self._account, cancelable_only=False) or []
        # Status mapping per xtconstant.ORDER_STATUS_*
        _status_map = {
            48: "pending", 49: "pending", 50: "partial", 51: "partial",
            52: "filled", 53: "cancelled", 54: "cancelled", 55: "rejected",
            56: "pending",
        }
        out: list[OrderStatus] = []
        for o in raw:
            side = "buy" if int(o.order_type) in (xt_buy_codes := (23, 27)) else "sell"
            out.append(OrderStatus(
                order_id=str(o.order_id),
                code=_plain_code(o.stock_code),
                action=side,
                shares_submitted=int(o.order_volume),
                shares_filled=int(o.traded_volume),
                avg_fill_price=float(o.traded_price) if o.traded_volume else 0.0,
                status=_status_map.get(int(o.order_status), "pending"),
                error_msg=getattr(o, "status_msg", None),
            ))
        return out

    # ── trading ─────────────────────────────────────────────────

    def place_limit_order(
        self,
        code: str,
        action: Literal["buy", "sell"],
        shares: int,
        limit_price: float,
        order_remark: str = "",
    ) -> OrderResult:
        """Submit a limit order.  Returns OrderResult.

        Synchronous: returns the broker-assigned order_id once the order
        is accepted (NOT filled — fills come via callbacks).
        """
        self._require_connected()
        from xtquant import xtconstant

        if shares < 100 or shares % 100 != 0:
            return OrderResult(success=False, error=f"shares={shares} not a valid lot")
        if limit_price <= 0:
            return OrderResult(success=False, error=f"limit_price={limit_price} invalid")

        xt_code = _xt_code(code)
        order_type = xtconstant.STOCK_BUY if action == "buy" else xtconstant.STOCK_SELL

        try:
            order_id = self._trader.order_stock(
                account=self._account,
                stock_code=xt_code,
                order_type=order_type,
                order_volume=int(shares),
                price_type=xtconstant.FIX_PRICE,   # 限价
                price=float(limit_price),
                strategy_name="money-printer",
                order_remark=order_remark or f"mp-{uuid.uuid4().hex[:6]}",
            )
        except Exception as e:
            return OrderResult(success=False, error=str(e),
                                code=code, action=action,
                                shares=shares, limit_price=limit_price)

        if order_id is None or order_id < 0:
            return OrderResult(success=False, error=f"order_stock returned {order_id}",
                                code=code, action=action,
                                shares=shares, limit_price=limit_price)

        return OrderResult(
            success=True, order_id=str(order_id),
            code=code, action=action,
            shares=shares, limit_price=limit_price,
        )

    def cancel_order(self, order_id: str) -> OrderResult:
        self._require_connected()
        try:
            rc = self._trader.cancel_order_stock(self._account, int(order_id))
        except Exception as e:
            return OrderResult(success=False, order_id=order_id, error=str(e))
        if rc < 0:
            return OrderResult(success=False, order_id=order_id,
                                error=f"cancel rc={rc}")
        return OrderResult(success=True, order_id=order_id)

    # ── internal ────────────────────────────────────────────────

    def _require_connected(self):
        if not self._connected:
            raise RuntimeError("QMTBroker not connected — call connect() first")
