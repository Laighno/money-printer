"""Broker selection for the live execution chain (miniQMT → BigQMT migration).

make_broker() returns the transport for reaching QMT, selected by env:

    MP_BROKER=miniqmt  (default) → QMTBroker  (xtquant → XtMiniQmt; retired 2026-09-27)
    MP_BROKER=bridge             → FileBridgeBroker (file bridge → BigQMT built-in
                                   strategy scripts/bigqmt_bridge_strategy.py)

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
    if kind == "bridge":
        from mp.execution.bridge_broker import FileBridgeBroker
        logger.info("broker_factory: MP_BROKER=bridge → FileBridgeBroker (BigQMT)")
        return FileBridgeBroker(account_id=account_id)
    if kind != "miniqmt":
        logger.warning("broker_factory: unknown MP_BROKER={!r}, using miniqmt", kind)
    from mp.execution.qmt_broker import QMTBroker
    return QMTBroker(account_id=account_id, qmt_userdata_path=qmt_userdata_path)
