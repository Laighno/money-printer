"""Broker that talks to QMT through FILENAMES instead of file contents.

Background (2026-10-25)
-----------------------
国金 switched on file-I/O and network restrictions in the QMT built-in Python
on 2026-10-23, which killed the JSON file bridge outright: its loader died on
line 3 with ``PermissionError: Foribdden FileIO``. Probing the sandbox from
inside a strategy mapped the new boundary precisely:

====================  ==========================================
open(path, "r")       BANNED (every path; "rb" and io.open too)
open(path, "w")       allowed
os.listdir            allowed
os.stat / exists      allowed
socket                blocked (timeout)
passorder             available
get_trade_detail_data available
====================  ==========================================

Reading is banned, writing is not. So the strategy cannot receive orders as
file CONTENT, but it can receive them as file NAMES, which ``os.listdir`` still
returns. This module is the external half: it writes one empty file per order
whose name encodes the order, then reads receipts back (external Python has no
such restriction).

Pairs with ``scripts/bigqmt_namebridge_strategy.py``.

This deliberately works around a broker-imposed restriction and should be
treated as a stopgap until miniQMT or an official API is available; it stops
working the moment ``listdir`` is restricted too.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from mp.execution.qmt_broker import (
    AccountInfo,
    OrderResult,
    OrderStatus,
    Position,
)

ORD_DIR_DEFAULT = r"C:\guojin\python\mp_orders"
HEARTBEAT_MAX_AGE = 30.0     # sec; strategy ticks every 2s
RECEIPT_TIMEOUT = 25.0       # sec to wait for a receipt after writing an order
RECEIPT_POLL = 0.25
STATE_MAX_AGE = 120.0        # sec; strategy refreshes state every ~10s


def encode_order(seq: str, side: str, code: str, shares: int,
                 price: float, tag: str = "mp") -> str:
    """Build the filename that carries one order.

    ``<seq>__<B|S>__<code>__<shares>__<price*100>__<tag>.ord``

    Price travels as an integer number of fen: a decimal point in a filename
    survives fine, but integers remove any doubt about locale or rounding on
    the 3.6/GBK side. Measured ceiling for a name ``listdir`` returns intact is
    184 chars; this layout needs 39.
    """
    code = str(code).zfill(6)
    if side.upper() not in ("B", "S"):
        raise ValueError(f"side must be B or S, got {side!r}")
    if shares <= 0 or shares % 100 != 0:
        raise ValueError(f"shares must be a positive multiple of 100, got {shares}")
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    fen = int(round(price * 100))
    safe_tag = "".join(ch for ch in str(tag) if ch.isalnum())[:12] or "mp"
    return f"{seq}__{side.upper()}__{code}__{int(shares)}__{fen:06d}__{safe_tag}.ord"


def decode_receipt(name: str) -> Optional[dict]:
    """``<seq>__<status>__<orderid>__<tag>.rcpt`` -> dict, or None."""
    if not name.lower().endswith(".rcpt"):
        return None
    parts = name[:-5].split("__")
    if len(parts) < 3:
        return None
    return {"seq": parts[0], "status": parts[1],
            "order_id": None if parts[2] in ("none", "") else parts[2],
            "tag": parts[3] if len(parts) > 3 else ""}


class NameBridgeBroker:
    """Submit orders by filename; read receipts and account state as files.

    Implements the broker interface ``execute_orders`` relies on. Account,
    position and order state are published by the strategy as filenames under
    ``acct/``, ``pos/`` and ``ords/`` and refreshed every ~10s, because the
    safety gates (cash/position parity, the idempotency re-check) cannot run
    without them. State older than ``STATE_MAX_AGE`` raises rather than being
    returned stale: trading on a stale book is worse than not trading.
    """

    def __init__(self, ord_dir: str = ORD_DIR_DEFAULT,
                 account_id: str = "8886933837"):
        self.ord_dir = Path(ord_dir)
        self.done_dir = self.ord_dir / "done"
        self.hb_dir = self.ord_dir / "hb"
        self.acct_dir = self.ord_dir / "acct"
        self.pos_dir = self.ord_dir / "pos"
        self.ordq_dir = self.ord_dir / "ords"
        self.account_id = account_id

    # ---------------------------------------------------------------- liveness
    def heartbeat_age(self) -> Optional[float]:
        """Seconds since the strategy's last tick, or None when unknown.

        The strategy writes ``<epoch>.hb`` each tick; the timestamp is in the
        NAME because it cannot write readable content.
        """
        try:
            stamps = []
            for n in os.listdir(self.hb_dir):
                if n.endswith(".hb"):
                    try:
                        stamps.append(int(n[:-3]))
                    except ValueError:
                        continue
            if not stamps:
                return None
            return time.time() - max(stamps)
        except OSError:
            return None

    def connect(self) -> bool:
        age = self.heartbeat_age()
        if age is None:
            logger.error("namebridge: no heartbeat files in {} - is the QMT "
                         "strategy running?", self.hb_dir)
            return False
        if age > HEARTBEAT_MAX_AGE:
            logger.error("namebridge: heartbeat stale {:.0f}s (max {:.0f}) - "
                         "strategy stopped?", age, HEARTBEAT_MAX_AGE)
            return False
        logger.info("namebridge: connected (heartbeat age {:.1f}s)", age)
        return True

    def is_connected(self) -> bool:
        age = self.heartbeat_age()
        return age is not None and age <= HEARTBEAT_MAX_AGE

    # ------------------------------------------------------------------ orders
    def place_limit_order(self, code: str, action: str, shares: int,
                          limit_price: float, seq: Optional[str] = None,
                          tag: str = "mp",
                          timeout: float = RECEIPT_TIMEOUT) -> OrderResult:
        side = "B" if action.lower() in ("buy", "b") else "S"
        if seq is None:
            # Monotonic and unique within a day; the strategy sorts on it.
            seq = f"{int(time.time() * 1000) % 1_000_000_000:09d}"
        name = encode_order(seq, side, code, shares, limit_price, tag)
        self.ord_dir.mkdir(parents=True, exist_ok=True)
        path = self.ord_dir / name
        try:
            path.touch()
        except OSError as e:
            return OrderResult(success=False, error=f"write_failed: {e!r}",
                               code=code, action=action, shares=shares)

        deadline = time.time() + timeout
        while time.time() < deadline:
            rcpt = self._find_receipt(seq)
            if rcpt is not None:
                ok = rcpt["status"] == "sent"
                return OrderResult(
                    success=ok, order_id=rcpt["order_id"],
                    error=None if ok else f"strategy reported {rcpt['status']}",
                    code=code, action=action, shares=shares)
            time.sleep(RECEIPT_POLL)

        # No receipt. The order may still have been submitted - the strategy
        # claims the file by renaming it to .taken BEFORE calling passorder, so
        # a vanished .ord with no receipt means "in flight, outcome unknown".
        claimed = not path.exists()
        status = "unknown_submitted" if claimed else "no_receipt"
        logger.error("namebridge: no receipt for seq={} after {:.0f}s "
                     "(claimed={})", seq, timeout, claimed)
        # execute_orders re-queries get_orders on failure and promotes a match
        # to unknown_submitted itself; the error text says which case this is.
        return OrderResult(success=False, error=f"{status}: receipt timeout",
                           code=code, action=action, shares=shares)

    def _find_receipt(self, seq: str) -> Optional[dict]:
        try:
            for n in os.listdir(self.done_dir):
                r = decode_receipt(n)
                if r and r["seq"] == seq:
                    return r
        except OSError:
            pass
        return None

    # ------------------------------------------------------- published state
    def _newest(self, directory: Path, suffix: str) -> Optional[str]:
        try:
            names = [n for n in os.listdir(directory) if n.endswith(suffix)]
        except OSError:
            return None
        return max(names) if names else None

    def get_account_info(self) -> AccountInfo:
        """``<ts>__<cash_fen>__<mv_fen>__<total_fen>.acct`` -> AccountInfo."""
        name = self._newest(self.acct_dir, ".acct")
        if not name:
            raise RuntimeError(
                f"namebridge: no account snapshot in {self.acct_dir}; "
                "is the QMT strategy running?")
        parts = name[:-5].split("__")
        ts = int(parts[0])
        age = time.time() - ts
        if age > STATE_MAX_AGE:
            raise RuntimeError(
                f"namebridge: account snapshot is {age:.0f}s old "
                f"(max {STATE_MAX_AGE:.0f}s) - strategy stalled?")
        cash, mv, total = (int(parts[1]) / 100.0, int(parts[2]) / 100.0,
                           int(parts[3]) / 100.0)
        return AccountInfo(
            cash_available=cash, cash_frozen=0.0, market_value=mv,
            total_assets=total,
            updated_at=datetime.fromtimestamp(ts).isoformat(timespec="seconds"))

    def get_positions(self) -> list[Position]:
        """``<code>__<total>__<avail>__<cost_fen>__<px_fen>.pos`` -> Positions."""
        out: list[Position] = []
        try:
            names = os.listdir(self.pos_dir)
        except OSError:
            return out
        for n in names:
            if not n.endswith(".pos"):
                continue
            parts = n[:-4].split("__")
            if len(parts) < 5:
                continue
            try:
                total = int(parts[1])
                px = int(parts[4]) / 100.0
                out.append(Position(
                    code=parts[0], name="", shares_total=total,
                    shares_available=int(parts[2]),
                    avg_cost=int(parts[3]) / 100.0,
                    market_price=px, market_value=total * px))
            except ValueError:
                continue
        return out

    def get_orders(self, only_today: bool = True) -> list[OrderStatus]:
        """``<oid>__<code>__<B|S>__<sub>__<filled>__<status>__<px_fen>.oq``.

        The strategy only ever publishes today's orders, so ``only_today`` is
        accepted for interface parity and has no effect.
        """
        out: list[OrderStatus] = []
        try:
            names = os.listdir(self.ordq_dir)
        except OSError:
            return out
        for n in names:
            if not n.endswith(".oq"):
                continue
            parts = n[:-3].split("__")
            if len(parts) < 7:
                continue
            try:
                out.append(OrderStatus(
                    order_id=parts[0], code=parts[1],
                    action="buy" if parts[2] == "B" else "sell",
                    shares_submitted=int(parts[3]),
                    shares_filled=int(parts[4]),
                    avg_fill_price=0.0,
                    status=parts[5],
                    limit_price=int(parts[6]) / 100.0 or None))
            except ValueError:
                continue
        return out

    def disconnect(self) -> None:
        """Nothing to tear down: the channel is a directory."""

    def emergency_liquidate_all(self, *args, **kwargs):
        """Not available through the filename channel.

        Liquidation needs live prices per holding and a tight submit loop; the
        name channel publishes state only every ~10s. Raising is the honest
        answer - a half-working panic button is worse than none.
        """
        raise NotImplementedError(
            "namebridge cannot liquidate; use the QMT client directly")

    # ------------------------------------------------------------- maintenance
    def prune(self, max_age_hours: float = 24.0) -> int:
        """Drop consumed .taken / .rcpt / .hb files older than the cutoff."""
        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        for d in (self.ord_dir, self.done_dir, self.hb_dir):
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in entries:
                if not e.is_file():
                    continue
                if not e.name.endswith((".taken", ".rcpt", ".hb")):
                    continue
                try:
                    if e.stat().st_mtime < cutoff:
                        os.remove(e.path)
                        removed += 1
                except OSError:
                    continue
        if removed:
            logger.info("namebridge: pruned {} stale files", removed)
        return removed
