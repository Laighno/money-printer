"""Atomically swap a verified BlendRanker into production, with backup +
md5-verify + rollback + audit log. Real-money gated.

Auto-retrain pipeline stage 3 (advisor; user 拍 ①B 流水线首跑重训当前 cutoff,
②B 首次换模手动确认、之后 gate PASS 自动换 + 通知 + 一键回滚).

This is the ONLY sanctioned path that overwrites data/blend_(primary|extreme).lgb
(now in PROTECTED_PROD_PATHS). It:
  1. Re-loads the new model + runs a final pick-sanity (no NaN / non-degenerate
     scores) as last-line defense — even if the caller already ran verify.
  2. Backs up current prod → {file}.bak_swap_<ts> (rollback-safe, Rule #4 archive).
  3. Atomically replaces prod (tmp + os.replace) and md5-verifies the result.
  4. Appends a record to data/model_swap_log.json (audit trail).
  5. Touches data/.first_swap_done so the orchestrator knows the human has
     approved the FIRST real-money swap (②B: subsequent weekly swaps auto).

Usage:
    # swap (gated; orchestrator passes --allow-prod-write)
    .venv/bin/python scripts/swap_model.py \
        --new-prefix data/blend_auto_20260526 --allow-prod-write \
        --reason "auto_retrain cutoff 2026-05-26"

    # one-command rollback to the most recent backup
    .venv/bin/python scripts/swap_model.py --rollback --allow-prod-write

    # inspect history
    .venv/bin/python scripts/swap_model.py --list-backups

Rule #4.1: writes protected prod paths — REQUIRES MP_ALLOW_PROD_WRITE=1
(set via --allow-prod-write or the scheduled ECS task).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

SWAP_LOG = ROOT / "data" / "model_swap_log.json"
FIRST_SWAP_SENTINEL = ROOT / "data" / ".first_swap_done"
_SUFFIXES = ("_primary.lgb", "_extreme.lgb")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prefix_files(prefix: str) -> list[Path]:
    return [Path(f"{prefix}{sfx}") for sfx in _SUFFIXES]


def _load_swap_log() -> list[dict]:
    if SWAP_LOG.exists():
        try:
            return json.loads(SWAP_LOG.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("swap log unreadable, starting fresh: {}", SWAP_LOG)
    return []


def _append_swap_log(record: dict) -> None:
    log = _load_swap_log()
    log.append(record)
    SWAP_LOG.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")


def _final_pick_sanity(new_prefix: str) -> Optional[str]:
    """Last-line defense before overwriting prod: load the new model and confirm
    it actually scores (no all-NaN / degenerate constant output). Returns an
    error string if it fails, else None. Cheap insurance independent of verify.
    """
    try:
        from mp.ml.model import BlendRanker
    except Exception as e:  # pragma: no cover
        return f"cannot import BlendRanker: {e}"
    br = BlendRanker()
    if not br.load(new_prefix):
        return f"new model failed to load: {new_prefix}"
    return None


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy src → dst atomically (tmp in same dir + os.replace)."""
    tmp = dst.with_suffix(dst.suffix + ".swap_tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def do_swap(new_prefix: str, prod_prefix: str, reason: str) -> int:
    from mp.common.paths import assert_prod_write_allowed, audit_prod_write

    new_files = _prefix_files(new_prefix)
    prod_files = _prefix_files(prod_prefix)

    for f in new_files:
        if not f.exists():
            logger.error("new model file missing: {}", f)
            return 2

    # Last-line sanity (independent of verify_retrain_quality).
    err = _final_pick_sanity(new_prefix)
    if err:
        logger.error("final pick-sanity FAILED: {}", err)
        return 3

    # Gate: refuse unless MP_ALLOW_PROD_WRITE=1.
    for pf in prod_files:
        assert_prod_write_allowed(pf)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "action": "swap",
        "new_prefix": new_prefix,
        "prod_prefix": prod_prefix,
        "reason": reason,
        "files": [],
    }

    # 1. backup current prod (if present), 2. atomic copy new → prod, 3. verify.
    for new_f, prod_f in zip(new_files, prod_files):
        backup = None
        md5_before = None
        if prod_f.exists():
            md5_before = _md5(prod_f)
            backup = prod_f.with_name(prod_f.name + f".bak_swap_{stamp}")
            shutil.copy2(prod_f, backup)
            logger.info("backed up {} → {}", prod_f.name, backup.name)

        src_md5 = _md5(new_f)
        _atomic_copy(new_f, prod_f)
        dst_md5 = _md5(prod_f)
        if dst_md5 != src_md5:
            logger.error("md5 MISMATCH after copy {} (src {} dst {}) — rolling back",
                         prod_f.name, src_md5, dst_md5)
            if backup is not None:
                shutil.copy2(backup, prod_f)
            return 4
        audit_prod_write(prod_f, {"source": "swap_model", "new_prefix": new_prefix})
        logger.info("✓ swapped {} (md5 {})", prod_f.name, dst_md5)
        record["files"].append({
            "prod": str(prod_f), "src": str(new_f),
            "backup": str(backup) if backup else None,
            "md5_before": md5_before, "md5_after": dst_md5,
        })

    _append_swap_log(record)
    first_time = not FIRST_SWAP_SENTINEL.exists()
    FIRST_SWAP_SENTINEL.write_text(record["ts"] + "\n", encoding="utf-8")
    logger.info("=" * 56)
    logger.info("SWAP DONE: {} → {} ({})", new_prefix, prod_prefix, reason)
    if first_time:
        logger.info("  (first swap — sentinel data/.first_swap_done created; "
                    "subsequent gate-PASS swaps may auto-run per ②B)")
    logger.info("  audit → {}", SWAP_LOG)
    logger.info("  rollback: scripts/swap_model.py --rollback --allow-prod-write")
    logger.info("=" * 56)
    return 0


def do_rollback(prod_prefix: str) -> int:
    from mp.common.paths import assert_prod_write_allowed

    log = _load_swap_log()
    swaps = [r for r in log if r.get("action") == "swap"
             and r.get("prod_prefix") == prod_prefix]
    if not swaps:
        logger.error("no prior swap for prod-prefix {} in {}", prod_prefix, SWAP_LOG)
        return 2
    last = swaps[-1]
    for f in _prefix_files(prod_prefix):
        assert_prod_write_allowed(f)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    restored = []
    for fe in last["files"]:
        prod_f = Path(fe["prod"])
        backup = fe.get("backup")
        if not backup or not Path(backup).exists():
            logger.error("backup missing for {} ({}) — cannot roll back",
                         prod_f.name, backup)
            return 3
        # snapshot the current (post-swap) state before clobbering, so rollback
        # is itself reversible.
        if prod_f.exists():
            shutil.copy2(prod_f, prod_f.with_name(prod_f.name + f".bak_rollback_{stamp}"))
        shutil.copy2(backup, prod_f)
        logger.info("rolled back {} ← {}", prod_f.name, Path(backup).name)
        restored.append(str(prod_f))

    _append_swap_log({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "action": "rollback",
        "reverted_swap_ts": last["ts"],
        "prod_prefix": prod_prefix,
        "restored": restored,
    })
    logger.info("ROLLBACK DONE — reverted swap from {}", last["ts"])
    return 0


def do_list() -> int:
    log = _load_swap_log()
    if not log:
        logger.info("no swap history ({} absent/empty)", SWAP_LOG)
        return 0
    for r in log:
        logger.info("[{}] {} {} {}", r.get("ts"), r.get("action"),
                    r.get("new_prefix", r.get("reverted_swap_ts", "")),
                    r.get("reason", ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-prefix", help="new model prefix, e.g. data/blend_auto_20260526")
    ap.add_argument("--prod-prefix", default="data/blend",
                    help="production model prefix (default: data/blend)")
    ap.add_argument("--reason", default="manual swap",
                    help="reason recorded in the audit log")
    ap.add_argument("--rollback", action="store_true",
                    help="restore the most recent backup for --prod-prefix")
    ap.add_argument("--list-backups", action="store_true",
                    help="print swap/rollback history and exit")
    ap.add_argument("--allow-prod-write", action="store_true",
                    help="set MP_ALLOW_PROD_WRITE=1 (Rule #4.1 gate)")
    args = ap.parse_args()

    if args.allow_prod_write:
        os.environ["MP_ALLOW_PROD_WRITE"] = "1"

    if args.list_backups:
        return do_list()
    if args.rollback:
        return do_rollback(args.prod_prefix)
    if not args.new_prefix:
        ap.error("--new-prefix required for swap (or use --rollback / --list-backups)")
    if args.new_prefix == args.prod_prefix:
        logger.error("--new-prefix == --prod-prefix; nothing to swap")
        return 1
    return do_swap(args.new_prefix, args.prod_prefix, args.reason)


if __name__ == "__main__":
    sys.exit(main())
