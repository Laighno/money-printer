"""P5-B dead-man-switch for weekly walk-forward (docs/dialog/ round 41-43).

What this monitors
------------------
Production weekly cron writes ``data/reports/backtest_history.json`` every
Friday 18:00 via ``update_production_models`` → ``_save_backtest_snapshot``.
The whole P4-1C threshold-alert pipeline depends on that cron firing and
``send_model_update_report`` running afterwards. If the cron silently
fails (data fetch crash / OOM / dependency break / cron itself disabled),
the entire alert system goes dark: no Feishu message arrives, threshold
breaches go undetected, and production drifts silently for up to a week.

This script is the **dead-man-switch**: scheduled on Saturday 06:00 (12h
after the Friday cron), it checks the mtime of ``backtest_history.json``
and if it's older than 7 days, sends a RED ALERT to Feishu indicating
"weekly walk_forward likely failed — investigate".

Why a separate file with NO walk_forward_backtest imports
---------------------------------------------------------
If ``walk_forward_backtest.py`` broke, anything importing it might fail
to import too — creating a dependency loop where the dead-man-switch
can't fire precisely when it's needed. This module deliberately keeps
imports minimal: stdlib + ``scripts.daily_report.send_to_feishu`` only
(and that send_to_feishu itself only depends on subprocess + lark-cli
binary, no walk_forward chain).

Usage / cron entry
------------------
See ``docs/cron_setup.md`` for the launchd / crontab block.
Manually: ``python scripts/monitor/weekly_heartbeat.py``

Exit codes
----------
0 — heartbeat healthy (file fresh, no alert)
0 — heartbeat alert sent (Feishu accepted; exit 0 so cron doesn't retry)
1 — script-level failure (rare; e.g. cannot resolve repo path)

Note: we intentionally do NOT exit non-zero on alert. Alerts are
expected business state, not errors. Exit non-zero only for failures of
this script itself, which should bubble up via cron mail.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Resolve repo root robustly — works whether script is invoked via
# `python scripts/monitor/weekly_heartbeat.py` from repo root or via
# absolute path from cron.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent.parent
if not (_REPO / "scripts").is_dir():
    print(f"weekly_heartbeat: cannot resolve repo root from {_THIS}", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(_REPO))


SNAPSHOT_PATH = _REPO / "data" / "reports" / "backtest_history.json"
HEARTBEAT_MAX_AGE = timedelta(days=7, hours=12)   # weekly + 12h grace
RED_ALERT_MAX_AGE = timedelta(days=14)            # 2 weeks => RED unambiguous


def _human_age(td: timedelta) -> str:
    secs = int(td.total_seconds())
    days, rem = divmod(secs, 86400)
    hours, _ = divmod(rem, 3600)
    return f"{days}d {hours}h" if days else f"{hours}h"


def check() -> dict:
    """Return heartbeat status dict; no I/O side effects.

    Result schema::

        {"healthy": bool,
         "level": "OK" | "YELLOW" | "RED",
         "snapshot_path": str,
         "snapshot_mtime": Optional[str],
         "age": Optional[str],
         "msg": str}
    """
    if not SNAPSHOT_PATH.exists():
        return {
            "healthy": False,
            "level": "RED",
            "snapshot_path": str(SNAPSHOT_PATH),
            "snapshot_mtime": None,
            "age": None,
            "msg": f"backtest_history.json missing at {SNAPSHOT_PATH}",
        }

    mtime = datetime.fromtimestamp(SNAPSHOT_PATH.stat().st_mtime)
    age = datetime.now() - mtime

    if age > RED_ALERT_MAX_AGE:
        level = "RED"
        healthy = False
        msg = (f"backtest_history.json age {_human_age(age)} > "
               f"{RED_ALERT_MAX_AGE.days}d. Weekly cron has been silent "
               f"for 2+ weeks — production model is going stale; alert "
               f"pipeline is dark. INVESTIGATE.")
    elif age > HEARTBEAT_MAX_AGE:
        level = "YELLOW"
        healthy = False
        msg = (f"backtest_history.json age {_human_age(age)} > "
               f"{HEARTBEAT_MAX_AGE.days}d {HEARTBEAT_MAX_AGE.seconds//3600}h. "
               f"Weekly cron may have skipped — check data/logs/model_update.log "
               f"for last attempt.")
    else:
        level = "OK"
        healthy = True
        msg = f"backtest_history.json age {_human_age(age)} (healthy)"

    try:
        path_disp = str(SNAPSHOT_PATH.relative_to(_REPO))
    except ValueError:
        # SNAPSHOT_PATH not under repo (happens in tests with tmp_path)
        path_disp = str(SNAPSHOT_PATH)
    return {
        "healthy": healthy,
        "level": level,
        "snapshot_path": path_disp,
        "snapshot_mtime": mtime.strftime("%Y-%m-%d %H:%M"),
        "age": _human_age(age),
        "msg": msg,
    }


def format_for_feishu(status: dict) -> str:
    """Render markdown block for Feishu. Empty when healthy."""
    if status["healthy"]:
        return ""
    emoji = "🚨" if status["level"] == "RED" else "⚠"
    return "\n".join([
        f"# {emoji} {status['level']} ALERT: weekly walk-forward heartbeat",
        "",
        f"**File**: `{status['snapshot_path']}`",
        f"**Last mtime**: {status.get('snapshot_mtime') or '(missing)'}",
        f"**Age**: {status.get('age') or 'n/a'}",
        "",
        status["msg"],
        "",
        "Diagnostics:",
        "1. Check `crontab -l` — entry still present? Last cron exit ok?",
        "2. Check `data/logs/model_update.log` tail for Friday's run",
        "3. If cron itself ran, check log for `update_production_models` errors",
        "",
        "Source: `scripts/monitor/weekly_heartbeat.py` (P5-B, docs/dialog/ round 43)",
    ])


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    dry_run = "--dry-run" in argv

    status = check()
    # Always print status (for cron logs)
    print(f"[weekly_heartbeat] {status['level']}: {status['msg']}")

    if status["healthy"]:
        return 0

    block = format_for_feishu(status)
    if dry_run:
        print("=== DRY RUN — would send to Feishu: ===")
        print(block)
        return 0

    try:
        from scripts.daily_report import send_to_feishu
    except Exception as e:
        print(f"[weekly_heartbeat] cannot import send_to_feishu: {e}",
              file=sys.stderr)
        # Still exit 0 — script-level catch already logged, no point in
        # retrying via cron mail
        return 0

    ok = send_to_feishu(block)
    print(f"[weekly_heartbeat] Feishu send returned {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
