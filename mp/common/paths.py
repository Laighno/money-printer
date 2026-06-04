"""Output-path isolation + prod-state write protection (Rounds 213, 217).

Round 213 Tier 0:
  Root cause for 6/4 9:25 incident: `advisor` ran `intraday_plan.py
  --asof 20260603` on ECS and wrote to `data/orders/intraday_latest.json`,
  overwriting the real 6/3 14:30 v0 prod output. The next morning's
  `reconcile_plan` saw the replay file as "the target", diffed against live
  QMT, and `ecs_auto_execute.ps1` sent 4 unintended buys.

  Tier 0 enforces a default-deny rule: writes to the prod orders path require
  an explicit `allow_prod_write=True` (passed only by the scheduled-task PS1
  wrappers). Replay / dry-run / ad-hoc invocations land in `data/_scratch/`.
  A complementary `source` provenance field is stamped onto every plan JSON;
  reconcile/execute reject plans whose source is not prod-authoritative
  (exit 11) so even if a replay slips into the prod path the scheduled
  executor still refuses to act on it.

Round 217 Tier 1 (Rule #4.1):
  Defense in depth. ``is_protected_prod_path`` + ``assert_prod_write_allowed``
  catch direct writes that bypass ``get_orders_output_dir`` (e.g. a future
  refactor that hard-codes a path). Gate = env ``MP_ALLOW_PROD_WRITE=1``,
  set by ``--allow-prod-write`` CLI handlers. ``audit_prod_write`` appends
  one line per prod write to ``data/audit/prod_writes.log`` so incidents
  can be reconstructed.
"""
from __future__ import annotations

import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROD_ORDERS_DIR = PROJECT_ROOT / "data" / "orders"
SCRATCH_DIR = PROJECT_ROOT / "data" / "_scratch"
AUDIT_LOG_PATH = PROJECT_ROOT / "data" / "audit" / "prod_writes.log"

# Rule #4.1 (advisor round 214): paths that hold prod state — orders, gates,
# state files, NAV history. Writing to these without an explicit
# MP_ALLOW_PROD_WRITE gate must fail. Glob patterns (e.g. "intraday_*.json")
# match any file in the listed parent directory.
PROTECTED_PROD_PATHS: list[Path] = [
    # orders artifacts (Group A)
    Path("data/orders/latest.json"),
    Path("data/orders/intraday_latest.json"),
    Path("data/orders/reconcile_latest.json"),
    Path("data/orders/orders_*.json"),
    Path("data/orders/intraday_*.json"),
    Path("data/orders/executions/exec_*.json"),
    # gate flags + state files (Group B)
    Path("config/portfolio.yaml"),
    Path("data/.real_money_frozen"),
    Path("data/arm_b_budget_state.json"),
    Path("data/account_nav_history.json"),
]


def is_protected_prod_path(path: str | Path) -> bool:
    """True if ``path`` matches any pattern in PROTECTED_PROD_PATHS.

    Paths outside PROJECT_ROOT (including SCRATCH_DIR) are NEVER protected,
    so replay/test invocations writing under data/_scratch/ pass freely.
    """
    p = Path(path).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        rel = p.relative_to(root)
    except ValueError:
        return False  # outside the repo
    # data/_scratch is explicitly the unprotected sibling of data/orders
    if rel.parts and rel.parts[0] == "data" and len(rel.parts) > 1 and rel.parts[1] == "_scratch":
        return False
    for protected in PROTECTED_PROD_PATHS:
        if "*" in str(protected):
            # Glob: parent dir must match exactly, filename matched as glob
            if rel.parent == protected.parent and rel.match(str(protected.name)):
                return True
        else:
            if rel == protected:
                return True
    return False


def assert_prod_write_allowed(path: str | Path) -> None:
    """Raise RuntimeError if ``path`` is a prod state file and the env gate
    ``MP_ALLOW_PROD_WRITE=1`` is not set.

    CLI handlers in scripts/intraday_plan.py and scripts/daily_report.py set
    this env var when ``--allow-prod-write`` is passed; everything else
    (ad-hoc CLI, replay, ssh sessions) is refused.
    """
    if not is_protected_prod_path(path):
        return
    if os.environ.get("MP_ALLOW_PROD_WRITE") == "1":
        return
    raise RuntimeError(
        f"REFUSED prod state write: {path}. "
        f"MP_ALLOW_PROD_WRITE env not set. Use the scheduled task or pass "
        f"--allow-prod-write to the CLI script that set up this call."
    )


def audit_prod_write(path: str | Path, source: dict) -> None:
    """Append one line to data/audit/prod_writes.log per prod write.

    No-op (file is just opened in append mode and a single line written)
    so this is safe to call from hot paths. Failures are silent — audit
    must never crash a production write.
    """
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = (
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"WROTE {path} "
            f"user={source.get('user')} "
            f"host={source.get('host')} "
            f"script={source.get('script')} "
            f"pid={source.get('pid')} "
            f"git_head={source.get('git_head')} "
            f"is_prod={source.get('is_prod')} "
            f"asof={source.get('asof')}\n"
        )
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


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
