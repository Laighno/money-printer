"""Auto-retrain orchestrator: refresh cache → train to current cutoff → verify
gate → (gated) swap. Auto-retrain pipeline stage 4 (advisor; user 拍 ①B/②B).

ROOT CAUSE this fixes: the weekly retrain (Mac Friday cron, docs/cron_setup.md)
silently died ~4/24 when the laptop slept; prod blend froze at 6/2. This
orchestrator is meant to run on ECS (always-on) via Task Scheduler so the model
never silently rots again.

Pipeline (each step aborts the run on failure → non-zero exit → dead-man alert):
  1. scripts/refresh_n2c_cache.py        — rebuild wf n2c factor cache to latest
                                            bars; emits the trainable cutoff
                                            (last date with a complete 20-day
                                            n2c label).
  2. scripts/train_blend_cutoff.py       — train a fresh BlendRanker to that
                                            cutoff (NOT 'today': the label needs
                                            ~20 forward days). Output:
                                            data/blend_auto_<cutoff>_(primary|extreme).lgb
  3. scripts/verify_retrain_quality.py   — relative gate (non-degeneracy +
                                            not-worse + regime-tracking). exit 0
                                            = PASS (eligible for swap).
  4. governance (②B):
       gate PASS + first swap not yet approved → write data/retrain_pending_swap.json
           and STOP (human runs swap_model.py once to approve).
       gate PASS + --auto-swap + first swap already approved → swap_model.py auto.
       gate FAIL → keep prod, record reasons, exit non-zero (RED).

Always writes data/auto_retrain_last.json (consumed by the dead-man-switch).

Usage (ECS scheduled, see scripts/ecs_auto_retrain.ps1):
    .venv/bin/python scripts/auto_retrain.py --auto-swap --allow-prod-write

    # train+verify only, never swap (CI / dry inspection):
    .venv/bin/python scripts/auto_retrain.py

Rule #4.1: only swap_model.py writes prod; this passes the gate through to it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

PY = sys.executable
FIRST_SWAP_SENTINEL = ROOT / "data" / ".first_swap_done"
SUMMARY = ROOT / "data" / "auto_retrain_last.json"
PENDING_SWAP = ROOT / "data" / "retrain_pending_swap.json"
_CUTOFF_RE = re.compile(r"trainable cutoff \(last valid n2c label\):\s*(\d{4}-\d{2}-\d{2})")


def _run(argv: list[str], capture: bool, env: dict) -> subprocess.CompletedProcess:
    logger.info("RUN: {}", " ".join(argv))
    return subprocess.run(
        argv, cwd=str(ROOT), env=env, text=True,
        capture_output=capture,
    )


def _write_summary(d: dict) -> None:
    d["ts"] = datetime.now().isoformat(timespec="seconds")
    SUMMARY.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
    logger.info("summary → {}", SUMMARY)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="20200101", help="train window start")
    ap.add_argument("--horizon", type=int, default=20, help="n2c label horizon")
    ap.add_argument("--prod-prefix", default="data/blend")
    ap.add_argument("--skip-refresh", action="store_true",
                    help="skip cache refresh; requires --end")
    ap.add_argument("--end", default=None,
                    help="override cutoff (YYYYMMDD); required with --skip-refresh")
    ap.add_argument("--auto-swap", action="store_true",
                    help="swap automatically on gate PASS (only after first swap "
                         "approved; ②B). Without it, PASS only stages a pending marker.")
    ap.add_argument("--allow-prod-write", action="store_true",
                    help="propagate MP_ALLOW_PROD_WRITE=1 to swap_model.py")
    args = ap.parse_args()

    env = dict(os.environ)
    if args.allow_prod_write:
        env["MP_ALLOW_PROD_WRITE"] = "1"

    summary: dict = {"refresh_ok": None, "cutoff": None, "train_ok": None,
                     "verify_pass": None, "swapped": False, "new_prefix": None,
                     "reasons": [], "verdict_json": None}

    # ── Step 1: refresh cache + derive cutoff ──
    if args.skip_refresh:
        if not args.end:
            logger.error("--skip-refresh requires --end YYYYMMDD")
            return 2
        cutoff_compact = args.end
        summary["refresh_ok"] = "skipped"
    else:
        r = _run([PY, "-X", "utf8", "scripts/refresh_n2c_cache.py"], capture=True, env=env)
        sys.stdout.write(r.stdout or "")
        sys.stderr.write(r.stderr or "")
        if r.returncode != 0:
            summary["refresh_ok"] = False
            summary["reasons"].append(f"refresh_n2c_cache exit {r.returncode}")
            _write_summary(summary)
            logger.error("refresh failed — abort")
            return 3
        summary["refresh_ok"] = True
        m = _CUTOFF_RE.search((r.stdout or "") + (r.stderr or ""))
        if args.end:
            cutoff_compact = args.end
        elif m:
            cutoff_compact = m.group(1).replace("-", "")
        else:
            summary["reasons"].append("could not parse trainable cutoff from refresh")
            _write_summary(summary)
            logger.error("cutoff not found in refresh output — abort (pass --end to override)")
            return 4
    summary["cutoff"] = cutoff_compact
    new_prefix = f"data/blend_auto_{cutoff_compact}"
    summary["new_prefix"] = new_prefix
    logger.info("cutoff = {} → new model prefix = {}", cutoff_compact, new_prefix)

    # ── Step 2: train to cutoff ──
    r = _run([PY, "-X", "utf8", "scripts/train_blend_cutoff.py",
              "--start", args.start, "--end", cutoff_compact,
              "--horizon", str(args.horizon),
              "--output-prefix", new_prefix], capture=False, env=env)
    if r.returncode != 0:
        summary["train_ok"] = False
        summary["reasons"].append(f"train_blend_cutoff exit {r.returncode}")
        _write_summary(summary)
        logger.error("train failed — abort")
        return 5
    summary["train_ok"] = True

    # ── Step 3: verify gate ──
    verdict_json = str(ROOT / "data" / f"retrain_verify_{cutoff_compact}.json")
    summary["verdict_json"] = verdict_json
    r = _run([PY, "-X", "utf8", "scripts/verify_retrain_quality.py",
              "--new-prefix", new_prefix, "--prod-prefix", args.prod_prefix,
              "--json-out", verdict_json], capture=False, env=env)
    verify_pass = (r.returncode == 0)
    summary["verify_pass"] = verify_pass
    try:
        v = json.loads(Path(verdict_json).read_text(encoding="utf-8"))
        summary["reasons"] = v.get("reasons", [])
    except Exception:
        pass

    if not verify_pass:
        _write_summary(summary)
        logger.error("VERIFY GATE FAILED — keeping prod {}; reasons: {}",
                     args.prod_prefix, summary["reasons"])
        return 6

    # ── Step 4: governance (②B) ──
    first_swap_approved = FIRST_SWAP_SENTINEL.exists()
    if args.auto_swap and first_swap_approved:
        r = _run([PY, "-X", "utf8", "scripts/swap_model.py",
                  "--new-prefix", new_prefix, "--prod-prefix", args.prod_prefix,
                  "--allow-prod-write",
                  "--reason", f"auto_retrain cutoff {cutoff_compact}"],
                 capture=False, env=env)
        if r.returncode != 0:
            summary["reasons"].append(f"swap_model exit {r.returncode}")
            _write_summary(summary)
            logger.error("swap failed — prod unchanged (swap_model rolls back on md5 fail)")
            return 7
        summary["swapped"] = True
        _write_summary(summary)
        logger.info("AUTO-SWAP done: {} → {}", new_prefix, args.prod_prefix)
        return 0

    # gate PASS but not auto-swapping → stage a pending marker for human approval.
    PENDING_SWAP.write_text(json.dumps({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "new_prefix": new_prefix, "prod_prefix": args.prod_prefix,
        "cutoff": cutoff_compact, "verdict_json": verdict_json,
        "first_swap": not first_swap_approved,
        "approve_cmd": (f".venv/bin/python scripts/swap_model.py --new-prefix "
                        f"{new_prefix} --allow-prod-write --reason "
                        f"'first manual swap cutoff {cutoff_compact}'"),
    }, indent=2), encoding="utf-8")
    _write_summary(summary)
    why = ("first swap needs your manual approval (②B)" if not first_swap_approved
           else "--auto-swap not set")
    logger.info("=" * 56)
    logger.info("GATE PASS — eligible for swap, but NOT swapping: {}", why)
    logger.info("  staged → {}", PENDING_SWAP)
    logger.info("  approve: see approve_cmd in that file")
    logger.info("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
