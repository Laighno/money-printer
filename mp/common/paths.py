"""Output-path isolation for prod vs replay/test (Round 213 Tier 0).

Root cause for 6/4 9:25 incident: `advisor` ran `intraday_plan.py --asof 20260603`
on ECS and wrote to `data/orders/intraday_latest.json`, overwriting the real 6/3
14:30 v0 prod output. The next morning's `reconcile_plan` saw the replay file as
"the target", diffed against live QMT, and `ecs_auto_execute.ps1` sent 4
unintended buys.

This module enforces a default-deny rule: writes to the prod orders path require
an explicit `allow_prod_write=True` (passed only by the scheduled-task PS1
wrappers). Replay / dry-run / ad-hoc invocations land in `data/_scratch/`.

A complementary `source` provenance field is stamped onto every plan JSON via
``make_plan_source``; reconcile/execute reject plans whose source is not
prod-authoritative (exit 11) so even if a replay slips into the prod path the
scheduled executor still refuses to act on it.
"""
from __future__ import annotations

import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROD_ORDERS_DIR = PROJECT_ROOT / "data" / "orders"
SCRATCH_DIR = PROJECT_ROOT / "data" / "_scratch"


def get_orders_output_dir(
    *,
    asof: Optional[str] = None,
    dry_run: bool = False,
    allow_prod_write: bool = False,
) -> Path:
    """Return the directory orders JSON should be written to.

    Defaults to the scratch dir. The prod orders dir is returned only when
    *all* of the following hold:

    - ``allow_prod_write=True`` (callers must opt in explicitly)
    - ``asof`` is None (no replay date)
    - ``dry_run`` is False
    - env ``MP_REPLAY_MODE`` is not set

    Scheduled tasks (ecs_intraday_execute.ps1, ecs_daily_report.ps1) pass
    ``allow_prod_write=True``; everything else (ad-hoc CLI, advisor ssh) lands
    in scratch.
    """
    is_replay = (
        asof is not None
        or dry_run
        or os.environ.get("MP_REPLAY_MODE")
        or not allow_prod_write
    )
    if is_replay:
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        return SCRATCH_DIR
    PROD_ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    return PROD_ORDERS_DIR


def _git_head_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def make_plan_source(
    *,
    allow_prod_write: bool,
    asof: Optional[str] = None,
    dry_run: bool = False,
    script: Optional[str] = None,
) -> dict:
    """Stamp a plan JSON with provenance so reconcile/execute can verify it.

    ``is_prod`` is True only when the same conditions as
    ``get_orders_output_dir`` would route to PROD_ORDERS_DIR. Reconcile and
    execute reject plans with ``is_prod=False``.
    """
    is_prod = (
        allow_prod_write
        and asof is None
        and not dry_run
        and not os.environ.get("MP_REPLAY_MODE")
    )
    return {
        "is_prod": bool(is_prod),
        "host": socket.gethostname(),
        "user": os.environ.get("USERNAME") or os.environ.get("USER") or "unknown",
        "pid": os.getpid(),
        "git_head": _git_head_short(),
        "script": script or "unknown",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "asof": asof,
        "dry_run": bool(dry_run),
        "allow_prod_write": bool(allow_prod_write),
    }
