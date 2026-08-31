"""FileBridgeBroker — QMTBroker-compatible broker that talks to the BigQMT
built-in bridge strategy (scripts/bigqmt_bridge_strategy.py) via local JSON
files instead of the xtquant miniQMT API.

WHY (2026-08-31): 国金 miniQMT retires 2026-09-27 (stopped new signups
2026-07-06, existing users to be cut off). The sanctioned path is the full QMT
client's built-in Python. All heavy logic stays in external Python 3.12; only
the last hop (place/cancel/query) crosses the bridge:

    external (this class)                 BigQMT built-in strategy
    write  data/bridge/req_<seq>.json  →  timer (2s) scans requests
    poll   data/bridge/resp_<seq>.json ←  passorder / get_trade_detail_data

Same public surface as QMTBroker: connect / disconnect / get_account_info /
get_positions / get_orders / place_limit_order / cancel_order. Consumers
(execute_orders.py, reconcile_plan.py, qmt_snapshot.py) switch via
MP_BROKER=bridge env or the --broker flag, no other changes.

Liveness: the strategy refreshes data/bridge/heartbeat.json every tick;
connect() fails if the heartbeat is older than HEARTBEAT_MAX_AGE (QMT client
not running / strategy not loaded — same failure semantics as miniQMT down).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal, Optional

from loguru import logger

from mp.execution.qmt_broker import (
    AccountInfo, OrderResult, OrderStatus, Position,
)

BRIDGE_DIR_DEFAULT = r"C:\money-printer\data\bridge"
HEARTBEAT_MAX_AGE = 10.0     # sec; 2s tick → 10s means strategy is dead
RESP_TIMEOUT = 25.0          # sec; order resp resolves 1-2 ticks late by design
RESP_POLL_INTERVAL = 0.25


class FileBridgeBroker:
    """Drop-in replacement for QMTBroker over the BigQMT file bridge."""

    def __init__(self, account_id: str = "8886933837",
                 bridge_dir: str = BRIDGE_DIR_DEFAULT,
                 qmt_userdata_path: str = ""):  # accepted+ignored (ctor parity)
        self.account_id = account_id
        self.bridge_dir = Path(bridge_dir)
        self._connected = False

    # ── lifecycle ───────────────────────────────────────────────

    def connect(self) -> bool:
        hb = self.bridge_dir / "heartbeat.json"
        try:
            data = json.loads(hb.read_text(encoding="utf-8"))
            age = time.time() - float(data.get("ts", 0))
        except Exception as e:
            logger.error("bridge connect failed: heartbeat unreadable ({}) — "
                         "is the BigQMT bridge strategy running?", e)
            return False
        if age > HEARTBEAT_MAX_AGE:
            logger.error("bridge connect failed: heartbeat stale {:.0f}s "
                         "(> {:.0f}s) — QMT client/strategy not running",
                         age, HEARTBEAT_MAX_AGE)
            return False
        if str(data.get("acct", "")) != self.account_id:
            logger.error("bridge heartbeat account {} != expected {}",
                         data.get("acct"), self.account_id)
            return False
        self._connected = True
        logger.info("bridge: connected (heartbeat age {:.1f}s, acct={})",
                    age, self.account_id)
        return True

    def disconnect(self) -> None:
        self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("FileBridgeBroker not connected — call connect()")

    # ── request/response plumbing ───────────────────────────────

    def _next_seq(self) -> int:
        # ns timestamp is monotonic enough per-process and unique across runs
        return time.time_ns()

    def _rpc(self, cmd: str, args: Optional[dict] = None,
             timeout: float = RESP_TIMEOUT) -> dict:
        self._require_connected()
        seq = self._next_seq()
        req_p = self.bridge_dir / f"req_{seq}.json"
        resp_p = self.bridge_dir / f"resp_{seq}.json"
        tmp = req_p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"seq": seq, "cmd": cmd, "args": args or {}}),
                       encoding="utf-8")
        os.replace(tmp, req_p)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if resp_p.exists():
                try:
                    resp = json.loads(resp_p.read_text(encoding="utf-8"))
                except Exception:
                    time.sleep(RESP_POLL_INTERVAL)   # partially written; retry
                    continue
                try:
                    resp_p.unlink()
                except OSError:
                    pass
                return resp
            time.sleep(RESP_POLL_INTERVAL)
        raise TimeoutError(
            f"bridge rpc {cmd} seq={seq} no response in {timeout}s "
            "(QMT strategy stalled?)")

    def _snapshot(self) -> dict:
        resp = self._rpc("snapshot")
        if not resp.get("ok"):
            raise RuntimeError(f"bridge snapshot failed: {resp.get('error')}")
        return resp["data"]

    # ── queries ─────────────────────────────────────────────────

    def get_account_info(self) -> AccountInfo:
        a = self._snapshot()["account"]
        return AccountInfo(
            cash_available=float(a.get("cash_available", 0)),
            cash_frozen=float(a.get("cash_frozen", 0)),
            market_value=float(a.get("market_value", 0)),
            total_assets=float(a.get("total_assets", 0)),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def get_positions(self) -> list[Position]:
        out = []
        for p in self._snapshot()["positions"]:
            out.append(Position(
                code=str(p["code"]).zfill(6),
                name=p.get("name", ""),
                shares_total=int(p["shares_total"]),
                shares_available=int(p["shares_available"]),
                avg_cost=float(p["avg_cost"]),
                market_price=float(p.get("market_price", 0)),
                market_value=float(p.get("market_value", 0)),
            ))
        return out

    def get_orders(self, only_today: bool = True) -> list[OrderStatus]:
        # bridge snapshot only carries today's orders — matches only_today=True,
        # which is the sole call pattern in the execution chain.
        out = []
        for o in self._snapshot()["orders"]:
            out.append(OrderStatus(
                order_id=str(o["order_id"]),
                code=str(o["code"]).zfill(6),
                action=o["action"],
                shares_submitted=int(o["shares_submitted"]),
                shares_filled=int(o["shares_filled"]),
                avg_fill_price=float(o.get("avg_fill_price", 0)),
                status=o.get("status", "pending"),
                error_msg=o.get("error_msg"),
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
        if shares < 100 or shares % 100 != 0:
            return OrderResult(success=False, error=f"shares={shares} not a valid lot")
        if limit_price <= 0:
            return OrderResult(success=False, error=f"limit_price={limit_price} invalid")
        try:
            resp = self._rpc("order", {
                "code": str(code).zfill(6), "action": action,
                "shares": int(shares), "limit_price": float(limit_price),
                "remark": order_remark,
            })
        except TimeoutError as e:
            return OrderResult(success=False, error=str(e), code=code,
                               action=action, shares=shares,
                               limit_price=limit_price)
        if not resp.get("ok"):
            return OrderResult(success=False, error=resp.get("error"),
                               code=code, action=action, shares=shares,
                               limit_price=limit_price)
        return OrderResult(success=True,
                           order_id=str(resp["data"]["order_id"]),
                           code=code, action=action, shares=shares,
                           limit_price=limit_price)

    def cancel_order(self, order_id: str) -> OrderResult:
        try:
            resp = self._rpc("cancel", {"order_id": order_id})
        except TimeoutError as e:
            return OrderResult(success=False, error=str(e))
        if not resp.get("ok"):
            return OrderResult(success=False, error=resp.get("error"))
        return OrderResult(success=True, order_id=order_id)
