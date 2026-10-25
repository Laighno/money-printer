"""Broker selection for the live execution chain (miniQMT → BigQMT migration).

make_broker() returns the transport for reaching QMT, selected by env:

    MP_BROKER=miniqmt  (default) → QMTBroker  (xtquant → XtMiniQmt; retired 2026-09-27)
    MP_BROKER=bridge             → FileBridgeBroker (file bridge → BigQMT built-in
                                   strategy scripts/bigqmt_bridge_strategy.py)
                                   DEAD since 2026-10-23: the broker banned file
                                   reads in the built-in Python, so the strategy
                                   cannot even load. Kept for history only.
    MP_BROKER=namebridge         → NameBridgeBroker (filename channel → BigQMT
                                   strategy scripts/bigqmt_namebridge_strategy.py).
                                   Orders and state travel as FILENAMES because
                                   os.listdir survived the ban while open(r) did
                                   not. This is the only live transport.

One env var flips the WHOLE chain (execute_orders, reconcile_plan, qmt_snapshot,
sync_portfolio) — no per-script flags, no drift between scripts. Set it in the
ECS scheduled-task wrappers when cutting over; unset = today's behavior.
"""
from __future__ import annotations

import os

from loguru import logger


def make_broker(account_id: str = "8886933837",
                qmt_userdata_path: str = r"C:\guojin\userdata_mini"):
    kind = os.environ.get("MP_BROKER", "miniqmt").lower()
    if kind == "namebridge":
        from mp.execution.namebridge_broker import NameBridgeBroker
        logger.info("broker_factory: MP_BROKER=namebridge → NameBridgeBroker "
                    "(BigQMT filename channel)")
        return NameBridgeBroker(account_id=account_id)
    if kind == "bridge":
        logger.warning("broker_factory: MP_BROKER=bridge is the pre-2026-10-23 "
                       "file bridge; the broker now blocks file reads in the "
                       "QMT built-in Python, so that strategy cannot run. "
                       "Use MP_BROKER=namebridge.")
        from mp.execution.bridge_broker import FileBridgeBroker
        logger.info("broker_factory: MP_BROKER=bridge → FileBridgeBroker (BigQMT)")
        return FileBridgeBroker(account_id=account_id)
    if kind != "miniqmt":
        logger.warning("broker_factory: unknown MP_BROKER={!r}, using miniqmt", kind)
    from mp.execution.qmt_broker import QMTBroker
    return QMTBroker(account_id=account_id, qmt_userdata_path=qmt_userdata_path)
