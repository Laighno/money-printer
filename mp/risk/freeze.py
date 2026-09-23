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

Monitor heartbeat (fail-closed, 2026-09-23)
-------------------------------------------
The freeze flag alone is only as good as the process that writes it.
Incident class: the writer (``scripts/arm_b_stop_monitor.py``, Mac launchd)
and the reader (``scripts/execute_orders.py`` on ECS Windows) ran on
different machines with no file sync, so the monitor could have fired
and the executor would never have seen the flag. Worse, if the monitor
silently died nobody noticed -- the flag simply never appeared.

Fix: the monitor writes ``data/.arm_b_monitor_heartbeat`` (JSON with
``ts``) every time it completes a check (OK / warn / triggered alike).
``guard_or_raise`` in a live mode now REFUSES to trade unless that
heartbeat exists and is younger than ``FREEZE_MONITOR_MAX_AGE_HOURS``
(default 36h; env override). Missing / stale / unreadable heartbeat all
mean "monitor is dead" and are treated as a hard stop.

DEPLOYMENT REQUIREMENT
    The monitor MUST run on the same machine as the executor (so both
    ``data/.real_money_frozen`` and ``data/.arm_b_monitor_heartbeat``
    live in the executor's PROJECT_ROOT), OR both files must be synced to
    the executor host before every live run. Otherwise every live run is
    refused with exit 4.

    Explicit escape hatch: ``MP_FREEZE_MONITOR_OPTIONAL=1`` skips ONLY the
    heartbeat check (freeze flag is still enforced). Use this consciously
    on hosts where the monitor is knowingly absent; it is logged loudly.

Paths are anchored to ``mp.common.paths.PROJECT_ROOT`` (absolute), not
the process CWD -- a relative ``data/...`` silently pointed at a
different file depending on where the scheduler started the process.
"""
from __future__ import annotations

import json
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

from mp.common.paths import PROJECT_ROOT

DEFAULT_FREEZE_PATH = PROJECT_ROOT / "data" / ".real_money_frozen"
DEFAULT_HEARTBEAT_PATH = PROJECT_ROOT / "data" / ".arm_b_monitor_heartbeat"

# Heartbeat older than this => monitor considered dead => refuse live trading.
# Env ``FREEZE_MONITOR_MAX_AGE_HOURS`` overrides at call time (read lazily so
# tests / ops can change it without re-importing).
FREEZE_MONITOR_MAX_AGE_HOURS_DEFAULT = 36.0
FREEZE_MONITOR_MAX_AGE_ENV = "FREEZE_MONITOR_MAX_AGE_HOURS"
FREEZE_MONITOR_OPTIONAL_ENV = "MP_FREEZE_MONITOR_OPTIONAL"


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


# ── monitor heartbeat ────────────────────────────────────────────

def monitor_max_age_hours() -> float:
    """Effective heartbeat max age (env override, default 36h)."""
    raw = os.environ.get(FREEZE_MONITOR_MAX_AGE_ENV)
    if raw is None or raw.strip() == "":
        return FREEZE_MONITOR_MAX_AGE_HOURS_DEFAULT
    try:
        v = float(raw)
    except ValueError:
        logger.warning("{}={!r} not a number; using default {}h",
                       FREEZE_MONITOR_MAX_AGE_ENV, raw,
                       FREEZE_MONITOR_MAX_AGE_HOURS_DEFAULT)
        return FREEZE_MONITOR_MAX_AGE_HOURS_DEFAULT
    if v <= 0:
        logger.warning("{}={} must be > 0; using default {}h",
                       FREEZE_MONITOR_MAX_AGE_ENV, v,
                       FREEZE_MONITOR_MAX_AGE_HOURS_DEFAULT)
        return FREEZE_MONITOR_MAX_AGE_HOURS_DEFAULT
    return v


def write_monitor_heartbeat(*, exit_code: int, path: Optional[Path] = None,
                            extra: Optional[dict] = None) -> dict:
    """Record that the stop monitor completed a check just now.

    Called by ``scripts/arm_b_stop_monitor.py`` after ``run_check`` returns
    (any exit code 0/1/2 -- OK / warn / triggered). NOT written on internal
    error (exit 3): a monitor that cannot load its data is not monitoring,
    and the guard must fail closed in that case too.
    """
    path = path or DEFAULT_HEARTBEAT_PATH
    now = datetime.now()
    state = {
        "ts": now.isoformat(timespec="seconds"),
        "ts_epoch": now.timestamp(),
        "exit_code": int(exit_code),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
    logger.debug("monitor heartbeat written -> {} (exit_code={})", path, exit_code)
    return state


def monitor_heartbeat_state(path: Optional[Path] = None) -> Optional[dict]:
    """Parsed heartbeat dict, or None if absent / unreadable / malformed."""
    path = path or DEFAULT_HEARTBEAT_PATH
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("monitor heartbeat {} unreadable: {}", path, e)
        return None
    if not isinstance(raw, dict) or "ts" not in raw:
        return None
    return raw


def check_monitor_alive(*, path: Optional[Path] = None,
                        max_age_hours: Optional[float] = None,
                        now: Optional[datetime] = None) -> Tuple[bool, str]:
    """Return (alive, reason). ``alive`` is False for missing / stale /
    unparsable heartbeat -- the caller must treat False as a hard stop."""
    path = path or DEFAULT_HEARTBEAT_PATH
    max_age = monitor_max_age_hours() if max_age_hours is None else float(max_age_hours)
    st = monitor_heartbeat_state(path)
    if st is None:
        if path.exists():
            return False, f"monitor heartbeat {path} exists but is unreadable/malformed"
        return False, f"monitor heartbeat {path} does not exist"
    try:
        ts = datetime.fromisoformat(str(st["ts"]))
    except Exception:
        return False, f"monitor heartbeat {path} has unparsable ts={st.get('ts')!r}"
    now = now or datetime.now()
    age_h = (now - ts).total_seconds() / 3600.0
    if age_h > max_age:
        return False, (f"monitor heartbeat {path} is stale: ts={st['ts']} "
                       f"age={age_h:.1f}h > max {max_age:.1f}h")
    if age_h < -1.0:
        # clock skew beyond 1h in the future is suspicious; still alive but log
        logger.warning("monitor heartbeat ts={} is {:.1f}h in the future (clock skew?)",
                       st["ts"], -age_h)
    return True, f"monitor heartbeat ok: ts={st['ts']} age={age_h:.1f}h (max {max_age:.1f}h)"


# ── caller-side guard ────────────────────────────────────────────

def guard_or_raise(mode: str, *, path: Optional[Path] = None,
                   heartbeat_path: Optional[Path] = None,
                   max_age_hours: Optional[float] = None,
                   now: Optional[datetime] = None) -> None:
    """Raise RuntimeError if trading must be refused in a live mode.

    Use this at the top of any code path that submits real orders.
    Dryrun mode is always allowed (it's the simulation path).

    Order of checks (both fail-closed):
      1. freeze flag set -> refuse (existing round 161 (d) behaviour).
      2. monitor heartbeat missing / stale / unreadable -> refuse, UNLESS
         env ``MP_FREEZE_MONITOR_OPTIONAL=1`` (explicit escape hatch for
         hosts where the monitor is knowingly not co-located; see module
         docstring DEPLOYMENT REQUIREMENT). The escape hatch never skips
         check 1.
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
    if os.environ.get(FREEZE_MONITOR_OPTIONAL_ENV) == "1":
        logger.warning(
            "{}=1: SKIPPING stop-monitor heartbeat check for mode={!r}. "
            "Freeze flag ({}) was checked and is clear. Make sure the monitor "
            "really is running elsewhere and its freeze flag is synced here.",
            FREEZE_MONITOR_OPTIONAL_ENV, mode, path or DEFAULT_FREEZE_PATH,
        )
        return
    alive, why = check_monitor_alive(path=heartbeat_path,
                                     max_age_hours=max_age_hours, now=now)
    if not alive:
        logger.error("STOP MONITOR DEAD -> refusing live trading: {}", why)
        raise RuntimeError(
            f"Stop monitor is not alive ({why}). Refusing mode={mode!r} "
            f"(fail-closed). Either run scripts/arm_b_stop_monitor.py on this "
            f"host (or sync {DEFAULT_HEARTBEAT_PATH.name} + "
            f"{DEFAULT_FREEZE_PATH.name} here), or set "
            f"{FREEZE_MONITOR_OPTIONAL_ENV}=1 to consciously skip this check."
        )
    logger.info("Stop-monitor liveness: {}", why)
