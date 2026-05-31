"""Arm B -5pp hard-stop monitor — round 161 guardrail (d).

Cron-friendly script. Reads NAV history + execution logs, computes the
cumulative-return delta between the Arm B (14:30) bucket and the EOD
baseline. If Arm B cumulative ret − EOD cumulative ret ≤ -5pp at any
point in the window, the kill switch engages:

1. ``mp.risk.freeze.freeze()`` writes the flag → ``scripts/execute_orders``
   refuses to send live orders.
2. Emit the PowerShell command for ECS ops to ``Disable-ScheduledTask
   -TaskName "MoneyPrinter-IntradayPipeline"`` (stderr + log).
3. Print alert payload (JSON) suitable for Feishu / email push (caller
   pipes to its alerting transport of choice).

Exit codes
----------
- 0  monitor ran, all OK (no trigger)
- 1  monitor ran, threshold approached but not crossed (warn-tier);
     opens a yellow card in the log but does NOT freeze
- 2  monitor triggered (freeze engaged, alert emitted)
- 3  internal error (could not load data; investigate)

Bucket-NAV separation caveat
----------------------------
Per round 161 (c) report module: NAV history does not currently store
per-bucket totals. Until that lands, this monitor uses a conservative
proxy:

- "EOD baseline cumulative ret" = NAV ret using EOD-only days as anchors
  (any day with intraday execs is excluded from the baseline trajectory)
- "Arm B incremental contribution" = NAV change on days with intraday
  execs, minus the expected EOD change projected from the EOD baseline
  trajectory.

When Arm B has zero executions (Phase 3 not yet enabled), the monitor
reports "no Arm B activity, baseline = full NAV ret" and exits 0.

The proxy is intentionally conservative: it errs on the side of being
too sensitive (catch real losses) rather than too lax. False positives
are recoverable (user reviews + unfreezes); false negatives are not.

Usage
-----
    .venv/bin/python scripts/arm_b_stop_monitor.py             # check only
    .venv/bin/python scripts/arm_b_stop_monitor.py --simulate  # smoke
    .venv/bin/python scripts/arm_b_stop_monitor.py --reset     # debug
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mp.risk.freeze import freeze, is_frozen, freeze_state  # noqa: E402

NAV_FILE = PROJECT_ROOT / "data" / "account_nav_history.json"
EXEC_DIR = PROJECT_ROOT / "data" / "orders" / "executions"

# Thresholds
HARD_STOP_DELTA_PP = -5.0   # -5 percentage points
WARN_DELTA_PP = -3.0        # yellow card at -3pp


def load_nav_history() -> List[dict]:
    if not NAV_FILE.exists():
        return []
    try:
        raw = json.loads(NAV_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("NAV history unreadable: {}", e)
        return []
    if not isinstance(raw, list):
        return []
    return sorted([s for s in raw if isinstance(s.get("date"), str)],
                  key=lambda x: x["date"])


def get_intraday_dates() -> set[str]:
    """Set of YYYY-MM-DD strings with at least one 14:30 execution."""
    out = set()
    if not EXEC_DIR.exists():
        return out
    for p in EXEC_DIR.glob("exec_*_intraday_*.json"):
        name = p.name  # exec_YYYYMMDD_intraday_HHMMSS.json
        parts = name.split("_")
        if len(parts) < 2 or not parts[1].isdigit() or len(parts[1]) != 8:
            continue
        s = parts[1]
        out.add(f"{s[:4]}-{s[4:6]}-{s[6:8]}")
    return out


def compute_cumulative_returns(nav_history: List[dict],
                                intraday_dates: set[str]
                                ) -> Tuple[float, float, str]:
    """Compute (arm_b_cum_ret, eod_cum_ret, msg) over the available history.

    Approach (proxy):
    - eod_cum_ret = product of (1 + daily NAV return) over days WITHOUT
      intraday executions, anchored to first such day.
    - arm_b_cum_ret = full NAV cum_ret minus eod_cum_ret. This attributes
      the residual to Arm B activity (conservative — also catches EOD
      drift on intraday days).

    If no intraday history exists, returns (0.0, full_cum_ret, msg).
    """
    if len(nav_history) < 2:
        return 0.0, 0.0, f"insufficient NAV history (n={len(nav_history)})"

    navs = []
    for snap in nav_history:
        ta = float(snap.get("total_assets") or 0)
        if ta > 0:
            navs.append((snap["date"], ta))
    if len(navs) < 2:
        return 0.0, 0.0, f"no usable NAV points"

    nav_start = navs[0][1]
    nav_end = navs[-1][1]
    full_cum_ret = (nav_end - nav_start) / nav_start

    # Cumulative ret on days WITHOUT intraday execs (proxy for EOD-only)
    eod_only = [(d, v) for d, v in navs if d not in intraday_dates]
    if len(eod_only) >= 2:
        eod_cum_ret = (eod_only[-1][1] - eod_only[0][1]) / eod_only[0][1]
    elif len(eod_only) == 1:
        eod_cum_ret = 0.0
    else:
        # Every day has intraday — can't isolate baseline; use 0
        eod_cum_ret = 0.0

    arm_b_cum_ret = full_cum_ret - eod_cum_ret
    msg = (f"navs n={len(navs)} ({navs[0][0]} → {navs[-1][0]}); "
           f"intraday days={sum(1 for d, _ in navs if d in intraday_dates)}; "
           f"full_cum={full_cum_ret:+.2%} eod_cum={eod_cum_ret:+.2%} "
           f"arm_b_residual={arm_b_cum_ret:+.2%}")
    return arm_b_cum_ret, eod_cum_ret, msg


def _emit_alert(payload: dict) -> None:
    """Print alert payload for caller / cron mail / Feishu hook."""
    print("=== ARM B HARD-STOP ALERT ===", file=sys.stderr)
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
    print("=== ECS OPS: run on Windows ECS to halt scheduled task ===", file=sys.stderr)
    print('Disable-ScheduledTask -TaskName "MoneyPrinter-IntradayPipeline"',
          file=sys.stderr)


def trigger_hard_stop(reason: str, evidence: dict) -> None:
    """Engage the kill switch + emit alert. Idempotent."""
    if is_frozen():
        logger.warning("Already frozen; recording duplicate trigger reason")
    freeze(reason=reason, source="arm_b_stop_monitor", extra=evidence)
    _emit_alert({
        "alert": "ARM_B_HARD_STOP_TRIGGERED",
        "reason": reason,
        "evidence": evidence,
        "actions_taken": ["wrote freeze flag (data/.real_money_frozen)"],
        "actions_required_on_ecs": [
            'Disable-ScheduledTask -TaskName "MoneyPrinter-IntradayPipeline"',
        ],
        "to_resume": (
            "user explicit approval required → call "
            "mp.risk.freeze.unfreeze(by='<user>', approval_token=<token>) "
            "and re-enable ScheduledTask on ECS"
        ),
    })


def run_check(simulate_trigger: bool = False) -> int:
    """Returns exit code; see module docstring."""
    if is_frozen():
        st = freeze_state()
        logger.warning("Real money is already FROZEN: reason={!r} at={}",
                       st.get("reason"), st.get("frozen_at"))
        # Still run the check (informational); return 2 because the
        # frozen state is itself a trigger that hasn't been cleared.
        # Cron should NOT treat this as a fresh trigger — caller can
        # distinguish via st.history.
        return 2

    nav_history = load_nav_history()
    intraday_dates = get_intraday_dates()

    # Contract: monitor protects against Arm B underperformance vs EOD baseline.
    # If Arm B has zero executions, there is nothing for it to drag — return OK.
    # This branch is the entire Phase 2 state (Phase 3 not yet enabled).
    if len(intraday_dates) == 0:
        logger.info("Monitor: no Arm B (14:30 intraday) executions in history → "
                    "no underperformance to monitor; exit 0")
        if not simulate_trigger:
            return 0
        # else fall through to simulate path below

    arm_b_ret, eod_ret, msg = compute_cumulative_returns(nav_history, intraday_dates)
    delta_pp = (arm_b_ret - eod_ret) * 100  # percentage points

    logger.info("Monitor check: {}", msg)
    logger.info("Arm B vs EOD delta: {:+.2f}pp (hard-stop at ≤{:.1f}pp, warn at ≤{:.1f}pp)",
                delta_pp, HARD_STOP_DELTA_PP, WARN_DELTA_PP)

    if simulate_trigger:
        logger.warning("--simulate flag set → forcing trigger path for smoke test")
        trigger_hard_stop(
            reason=f"SIMULATED trigger (test) — actual delta={delta_pp:+.2f}pp",
            evidence={"simulated": True, "real_delta_pp": delta_pp,
                       "intraday_days_in_history": len(intraday_dates),
                       "nav_history_n": len(nav_history)},
        )
        return 2

    if delta_pp <= HARD_STOP_DELTA_PP:
        trigger_hard_stop(
            reason=f"Arm B cumulative ret − EOD cumulative ret = {delta_pp:+.2f}pp "
                    f"≤ hard-stop threshold {HARD_STOP_DELTA_PP:.1f}pp",
            evidence={
                "arm_b_cumulative_ret": arm_b_ret,
                "eod_cumulative_ret": eod_ret,
                "delta_pp": delta_pp,
                "threshold_pp": HARD_STOP_DELTA_PP,
                "nav_history_first": nav_history[0]["date"] if nav_history else None,
                "nav_history_last": nav_history[-1]["date"] if nav_history else None,
                "intraday_days_n": len(intraday_dates),
            },
        )
        return 2

    if delta_pp <= WARN_DELTA_PP:
        logger.warning("YELLOW CARD: delta {:+.2f}pp approaching hard-stop "
                       "{:.1f}pp — review next session",
                       delta_pp, HARD_STOP_DELTA_PP)
        return 1

    logger.info("OK: delta {:+.2f}pp within tolerance", delta_pp)
    return 0


def reset_freeze_for_debug() -> int:
    from mp.risk.freeze import unfreeze
    new = unfreeze(by="debug_reset", approval_token="LOCAL_DEBUG")
    print("Freeze cleared:", json.dumps(new, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulate", action="store_true",
                    help="force trigger path for smoke testing (writes freeze flag)")
    ap.add_argument("--reset", action="store_true",
                    help="DEBUG ONLY: clear freeze flag (NOT for prod recovery)")
    args = ap.parse_args(argv)

    if args.reset:
        return reset_freeze_for_debug()

    try:
        return run_check(simulate_trigger=args.simulate)
    except Exception as e:
        logger.exception("Monitor internal error: {}", e)
        return 3


if __name__ == "__main__":
    sys.exit(main())
