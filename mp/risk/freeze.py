"""Real-money kill switch — round 161 guardrail (d) infrastructure.

A simple file-backed flag. When set, ``is_frozen()`` returns True and any
caller (e.g. ``scripts/execute_orders.py`` in non-dryrun mode) should
refuse to send live orders. Unfreezing requires an explicit call from
the user (per round 159 rule: "止损触发后冻结状态须 user 显式重启").

Flag file: ``data/.real_money_frozen`` (JSON). When file exists with
``frozen: true``, the freeze is active. The state object holds the
trigger reason + timestamp for audit.

Why a file: cron / ScheduledTask processes have no shared in-process
state; a flag file is the simplest cross-process signal that survives
ECS reboots / process restarts.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

DEFAULT_FREEZE_PATH = Path("data/.real_money_frozen")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def freeze_state(path: Optional[Path] = None) -> dict:
    """Return current state dict (or {'frozen': False} if no flag file)."""
    path = path or DEFAULT_FREEZE_PATH
    if not path.exists():
        return {"frozen": False}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Freeze flag {} unreadable ({}); treating as NOT frozen",
                       path, e)
        return {"frozen": False, "error": str(e)}
    if not isinstance(raw, dict):
        return {"frozen": False}
    return raw


def is_frozen(path: Optional[Path] = None) -> bool:
    return bool(freeze_state(path).get("frozen"))


def freeze(reason: str, *, source: str = "monitor",
           path: Optional[Path] = None,
           extra: Optional[dict] = None) -> dict:
    """Engage the freeze flag. Idempotent (re-freezing only updates audit)."""
    path = path or DEFAULT_FREEZE_PATH
    state = freeze_state(path)
    history = list(state.get("history", []))
    history.append({"action": "freeze", "at": _now_iso(),
                     "reason": reason, "source": source, "extra": extra or {}})
    new = {
        "frozen": True,
        "frozen_at": _now_iso(),
        "reason": reason,
        "source": source,
        "extra": extra or {},
        "history": history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(new, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    logger.error("REAL MONEY FROZEN: reason={!r} source={}", reason, source)
    return new


def unfreeze(by: str, *, path: Optional[Path] = None,
             approval_token: Optional[str] = None) -> dict:
    """Lift the freeze. Requires explicit ``by`` identifier — per round 159
    this should map to the user explicitly approving restart.

    ``approval_token`` is logged as audit; callers (e.g. CLI) may use a
    fresh UUID per approval so audit lines correlate.
    """
    path = path or DEFAULT_FREEZE_PATH
    state = freeze_state(path)
    history = list(state.get("history", []))
    history.append({"action": "unfreeze", "at": _now_iso(),
                     "by": by, "approval_token": approval_token})
    new = {
        "frozen": False,
        "unfrozen_at": _now_iso(),
        "unfrozen_by": by,
        "approval_token": approval_token,
        "history": history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(new, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    logger.warning("REAL MONEY UNFROZEN: by={!r} token={}", by, approval_token)
    return new


# ── caller-side guard ────────────────────────────────────────────

def guard_or_raise(mode: str, *, path: Optional[Path] = None) -> None:
    """Raise RuntimeError if frozen and mode is live (not dryrun).

    Use this at the top of any code path that submits real orders.
    Dryrun mode is always allowed (it's the simulation path).
    """
    if mode == "dryrun":
        return
    if is_frozen(path):
        st = freeze_state(path)
        raise RuntimeError(
            f"Real money is FROZEN (reason={st.get('reason')!r}, "
            f"frozen_at={st.get('frozen_at')}). Refusing mode={mode!r}. "
            f"To resume, call mp.risk.freeze.unfreeze() with explicit user "
            f"approval (see round 159 / round 161 (d))."
        )
