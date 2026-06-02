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

# Trading-calendar import is light (only stdlib + pandas + loguru at module
# load — akshare is lazy inside the fetch).  Keeping it at module-level
# lets tests patch ``trading_days_between`` / ``calendar_available`` on
# this module's namespace via monkeypatch.
try:
    import pandas as pd
    from mp.data.trading_calendar import (
        trading_days_between, calendar_available,
    )
    _CALENDAR_IMPORT_ERROR: str | None = None
except Exception as _e:   # pragma: no cover — defensive
    pd = None
    trading_days_between = None
    calendar_available = None
    _CALENDAR_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"


SNAPSHOT_PATH = _REPO / "data" / "reports" / "backtest_history.json"

# Trading-day-aware thresholds (P6-X2, docs/dialog/ round 47). The previous
# wall-clock thresholds (7d12h YELLOW / 14d RED) gave false-positives
# across long weekends, CNY, and National Day — those are *expected* gaps,
# not silent cron failure.
#
# We still keep a wall-clock fallback (RED_ALERT_CAL_AGE) so that
# catastrophic calendar-API failure can't suppress a real "no cron for
# 3+ weeks" alert: even if trading_days_since lies, the calendar age cap
# escalates to RED on its own.
HEARTBEAT_TRADING_DAYS = 5    # YELLOW: >5 trading days since last snapshot
RED_TRADING_DAYS = 10         # RED: >10 trading days
RED_ALERT_CAL_AGE = timedelta(days=21)   # safety: 3 weeks wall-clock → RED


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
            "trading_days_since": None,
            "calendar_source": None,
            "msg": f"backtest_history.json missing at {SNAPSHOT_PATH}",
        }

    mtime = datetime.fromtimestamp(SNAPSHOT_PATH.stat().st_mtime)
    now = datetime.now()
    age = now - mtime

    # Trading-day count between mtime and now (P6-X2). If
    # trading_calendar failed to import at module load (rare), we
    # gracefully degrade to wall-clock — never crash the dead-man-switch.
    if trading_days_between is None or pd is None:
        td_since = None
        cal_src = f"import-failed ({_CALENDAR_IMPORT_ERROR})"
    else:
        try:
            td_since = trading_days_between(pd.Timestamp(mtime), pd.Timestamp(now))
            cal_src = "akshare" if calendar_available() else "weekday-fallback"
        except Exception as e:
            td_since = None
            cal_src = f"runtime-failed ({e!s})"

    # Level decision: trading-day-aware primary, wall-clock secondary
    if age > RED_ALERT_CAL_AGE:
        level = "RED"
        healthy = False
        msg = (f"backtest_history.json calendar age {_human_age(age)} > "
               f"{RED_ALERT_CAL_AGE.days}d. Weekly cron silent for 3+ weeks "
               f"by wall-clock — production model going stale; alert pipeline "
               f"dark. INVESTIGATE.")
    elif td_since is None:
        # trading_calendar import failed; degrade to wall-clock with the
        # ORIGINAL 7d12h / 14d thresholds so we don't silently downgrade.
        if age > timedelta(days=14):
            level = "RED"
            healthy = False
            msg = (f"trading_calendar import failed ({cal_src}); wall-clock "
                   f"age {_human_age(age)} > 14d. RED.")
        elif age > timedelta(days=7, hours=12):
            level = "YELLOW"
            healthy = False
            msg = (f"trading_calendar import failed ({cal_src}); wall-clock "
                   f"age {_human_age(age)} > 7d12h. YELLOW.")
        else:
            level = "OK"
            healthy = True
            msg = (f"trading_calendar import failed ({cal_src}) but wall-clock "
                   f"age {_human_age(age)} healthy")
    elif td_since > RED_TRADING_DAYS:
        level = "RED"
        healthy = False
        msg = (f"backtest_history.json {td_since} trading days old "
               f"(calendar age {_human_age(age)}, src={cal_src}) > "
               f"{RED_TRADING_DAYS} — production model going stale; alert "
               f"pipeline likely dark. INVESTIGATE.")
    elif td_since > HEARTBEAT_TRADING_DAYS:
        level = "YELLOW"
        healthy = False
        msg = (f"backtest_history.json {td_since} trading days old "
               f"(calendar age {_human_age(age)}, src={cal_src}) > "
               f"{HEARTBEAT_TRADING_DAYS} — Weekly cron may have skipped. "
               f"Check data/logs/model_update.log for last attempt.")
    else:
        level = "OK"
        healthy = True
        msg = (f"backtest_history.json {td_since} trading days old "
               f"(calendar age {_human_age(age)}, src={cal_src}) — healthy")

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
        "trading_days_since": td_since,
        "calendar_source": cal_src,
        "msg": msg,
    }


def format_for_feishu(status: dict) -> str:
    """Render markdown block for Feishu. Empty when healthy."""
    if status["healthy"]:
        return ""
    emoji = "🚨" if status["level"] == "RED" else "⚠"
    # Show both data points (trading-day count + calendar age) per
    # P6-X2 round-47 spec — when one path fails to diagnose the cause,
    # the other often does.
    td = status.get("trading_days_since")
    td_line = (f"**Trading days since**: {td} (calendar src: "
               f"{status.get('calendar_source') or 'n/a'})"
               if td is not None
               else f"**Trading days since**: n/a (src: "
                    f"{status.get('calendar_source') or 'unknown'})")
    return "\n".join([
        f"# {emoji} {status['level']} ALERT: weekly walk-forward heartbeat",
        "",
        f"**File**: `{status['snapshot_path']}`",
        f"**Last mtime**: {status.get('snapshot_mtime') or '(missing)'}",
        f"**Calendar age**: {status.get('age') or 'n/a'}",
        td_line,
        "",
        status["msg"],
        "",
        "Diagnostics:",
        "1. Check `crontab -l` — entry still present? Last cron exit ok?",
        "2. Check `data/logs/model_update.log` tail for Friday's run",
        "3. If cron itself ran, check log for `update_production_models` errors",
        "",
        "Source: `scripts/monitor/weekly_heartbeat.py` (P5-B, P6-X2 round 47)",
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
        print("=== DRY RUN — would dispatch alert: ===")
        print(block)
        return 0

    # P8-α-3 (docs/dialog/ round 53): dispatch through multi-channel
    # alert_dispatch (Feishu + JSONL audit + stderr) instead of direct
    # send_to_feishu — kills lark-cli SPOF that would silently swallow
    # alerts in live trading.
    try:
        from mp.monitor.alert_dispatch import dispatch_alert
    except Exception as e:
        print(f"[weekly_heartbeat] cannot import alert_dispatch: {e}",
              file=sys.stderr)
        # Still exit 0 — script-level catch already logged, no point in
        # retrying via cron mail
        return 0

    results = dispatch_alert(
        level=status["level"],
        title=f"{status['level']}: weekly walk-forward heartbeat",
        body=block,
        source="heartbeat",
    )
    print(f"[weekly_heartbeat] dispatch results: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
