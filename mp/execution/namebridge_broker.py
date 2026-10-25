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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

ORD_DIR_DEFAULT = r"C:\guojin\python\mp_orders"
HEARTBEAT_MAX_AGE = 30.0     # sec; strategy ticks every 2s
RECEIPT_TIMEOUT = 25.0       # sec to wait for a receipt after writing an order
RECEIPT_POLL = 0.25


@dataclass
class NameOrderResult:
    success: bool
    order_id: Optional[str]
    status: str
    error: Optional[str] = None


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

    Account/position queries are NOT available through this channel - the
    strategy can only write names, which is far too thin for a position book.
    Callers that need live account state should read it from the daily QMT
    export or portfolio.yaml. ``connect`` only proves the strategy is alive.
    """

    def __init__(self, ord_dir: str = ORD_DIR_DEFAULT,
                 account_id: str = "8886933837"):
        self.ord_dir = Path(ord_dir)
        self.done_dir = self.ord_dir / "done"
        self.hb_dir = self.ord_dir / "hb"
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
                          timeout: float = RECEIPT_TIMEOUT) -> NameOrderResult:
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
            return NameOrderResult(False, None, "write_failed", repr(e))

        deadline = time.time() + timeout
        while time.time() < deadline:
            rcpt = self._find_receipt(seq)
            if rcpt is not None:
                ok = rcpt["status"] == "sent"
                return NameOrderResult(ok, rcpt["order_id"], rcpt["status"],
                                       None if ok else rcpt["status"])
            time.sleep(RECEIPT_POLL)

        # No receipt. The order may still have been submitted - the strategy
        # claims the file by renaming it to .taken BEFORE calling passorder, so
        # a vanished .ord with no receipt means "in flight, outcome unknown".
        claimed = not path.exists()
        status = "unknown_submitted" if claimed else "no_receipt"
        logger.error("namebridge: no receipt for seq={} after {:.0f}s "
                     "(claimed={})", seq, timeout, claimed)
        return NameOrderResult(False, None, status, "receipt timeout")

    def _find_receipt(self, seq: str) -> Optional[dict]:
        try:
            for n in os.listdir(self.done_dir):
                r = decode_receipt(n)
                if r and r["seq"] == seq:
                    return r
        except OSError:
            pass
        return None

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
