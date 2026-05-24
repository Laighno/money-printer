"""P6-X1 cron-drift detector (docs/dialog/ round 47).

What this monitors
------------------
The production crontab is the source of truth for *when* `walk_forward_backtest`
and `weekly_heartbeat` fire.  But the crontab itself lives in user
home directory state — anyone can `crontab -e` and quietly mutate it,
and the modification leaves no commit trail.  ``docs/cron_setup.md``
records the **intended** crontab; if it diverges from what's actually
loaded, the production schedule is silently drifting.

This script hashes both:

  1. live ``crontab -l`` output
  2. the fenced ``cron`` block under ``## Current crontab`` in
     ``docs/cron_setup.md``

normalizes them (strip comments / blank lines / trailing whitespace) and
SHA256s the result.  Mismatch → RED Feishu alert.

Why a separate file with no walk_forward imports
------------------------------------------------
Same reason as ``weekly_heartbeat.py``: if `walk_forward_backtest`
breaks, anything importing it might not load, and the alert can't fire
when most needed.  Kept dependency-light: stdlib + ``scripts.daily_report.send_to_feishu``
(which only needs lark-cli) on alert.

Usage / cron entry
------------------
Add to crontab (see ``docs/cron_setup.md``)::

  0 7 * * * /Users/laighno/laighno/money-printer/.venv/bin/python \
            scripts/monitor/cron_drift_detect.py \
            >> data/logs/cron_drift.log 2>&1

07:00 daily, after weekly_heartbeat's 06:00 slot (independent failure
domain — both write their own log + send their own alert).

Manual: ``python scripts/monitor/cron_drift_detect.py [--dry-run]``

Exit codes
----------
0 — drift OK (or alert sent / dry-run; cron should not retry)
1 — script-level failure (couldn't resolve paths etc.)

We deliberately exit 0 on drift detected — the alert is the business-
level signal, not the script's exit status.  cron's retry logic
wouldn't help here anyway (drift won't self-resolve).
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

# Resolve repo root robustly
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent.parent
if not (_REPO / "scripts").is_dir():
    print(f"cron_drift_detect: cannot resolve repo root from {_THIS}",
          file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(_REPO))


DOCS_PATH = _REPO / "docs" / "cron_setup.md"
CURRENT_ANCHOR = "## Current crontab"
SUBPROCESS_TIMEOUT_SEC = 10


# ───────────────────────────────────────────────────────────────────────
# Normalize / hash
# ───────────────────────────────────────────────────────────────────────

def _normalize_cron(cron_text: str) -> str:
    """Strip blank lines, full-line comments, and trailing whitespace.

    The hash is computed on the normalized form so cosmetic edits to
    comments / spacing don't trigger false-positive drift alerts.
    Comments inside a line (e.g. ``* * * * * cmd  # foo``) are NOT
    stripped — that's part of the actual cron entry.
    """
    lines = []
    for line in cron_text.splitlines():
        line = line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _cron_hash(cron_text: str) -> str:
    return hashlib.sha256(_normalize_cron(cron_text).encode("utf-8")).hexdigest()


# ───────────────────────────────────────────────────────────────────────
# Parser: pull the expected crontab block out of docs/cron_setup.md
# ───────────────────────────────────────────────────────────────────────

def extract_expected_cron(docs_text: str) -> str:
    """Find the first ```cron … ``` fenced block AFTER the anchor
    ``## Current crontab`` in ``docs_text``.  Returns the block's text.

    Raises ``ValueError`` if anchor or fenced block missing — fail-fast
    is intentional per round-47 spec: a silent fallback (e.g. hashing
    "") would make drift detect always report OK and silently lose its
    purpose.
    """
    anchor_pos = docs_text.find(CURRENT_ANCHOR)
    if anchor_pos == -1:
        raise ValueError(
            f"docs/cron_setup.md missing anchor heading '{CURRENT_ANCHOR}'"
        )
    tail = docs_text[anchor_pos:]
    # The fence format is ```cron <newline> ... <newline>```. DOTALL so
    # the body can span multiple lines.
    m = re.search(r"```cron\s*\n(.*?)```", tail, re.DOTALL)
    if m is None:
        raise ValueError(
            f"docs/cron_setup.md has '{CURRENT_ANCHOR}' anchor but no "
            f"```cron``` fenced block follows it"
        )
    return m.group(1)


# ───────────────────────────────────────────────────────────────────────
# Live crontab fetch (defensive: timeout, missing binary, nonzero rc)
# ───────────────────────────────────────────────────────────────────────

def read_live_crontab() -> tuple[str | None, str | None]:
    """Run ``crontab -l`` and capture stdout. Returns ``(text, err)``
    where exactly one is ``None``.

    Defensive bracketing:
      - ``timeout=SUBPROCESS_TIMEOUT_SEC`` so a Full Disk Access prompt
        can't hang us forever (round-46 confirmed READ doesn't trigger
        FDA in practice, but it's still cheap insurance).
      - ``FileNotFoundError`` for the rare ``crontab`` binary missing case.
      - non-zero return code (most commonly "no crontab for $USER" = rc 1)
        reported as drift-equivalent (live is *empty* — definitely drift
        from any nonempty docs).
    """
    try:
        result = subprocess.run(
            ["crontab", "-l"],
            timeout=SUBPROCESS_TIMEOUT_SEC,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return None, (
            f"crontab -l timed out after {SUBPROCESS_TIMEOUT_SEC}s "
            f"(Full Disk Access prompt?)"
        )
    except FileNotFoundError:
        return None, "crontab command not found on PATH"
    except Exception as e:   # pragma: no cover — defensive catch-all
        return None, f"crontab -l raised: {type(e).__name__}: {e}"

    if result.returncode != 0:
        return None, (
            f"crontab -l returned rc={result.returncode}; stderr="
            f"{(result.stderr or '').strip()[:200]}"
        )
    return result.stdout, None


# ───────────────────────────────────────────────────────────────────────
# Top-level check
# ───────────────────────────────────────────────────────────────────────

def check() -> dict:
    """Run the comparison; return a status dict (no I/O side effects
    besides reading ``docs/cron_setup.md`` and shelling out to crontab).

    Result schema::

        {"healthy": bool,
         "level": "OK" | "RED",
         "msg": str,
         "live_hash": Optional[str],
         "expected_hash": Optional[str]}
    """
    # ── 1. Read expected from docs ────────────────────────────────
    if not DOCS_PATH.exists():
        return {
            "healthy": False,
            "level": "RED",
            "msg": f"docs/cron_setup.md not found at {DOCS_PATH}",
            "live_hash": None,
            "expected_hash": None,
        }
    try:
        docs_text = DOCS_PATH.read_text(encoding="utf-8")
        expected = extract_expected_cron(docs_text)
        expected_h = _cron_hash(expected)
    except ValueError as e:
        return {
            "healthy": False,
            "level": "RED",
            "msg": f"Cannot extract expected crontab from docs: {e}",
            "live_hash": None,
            "expected_hash": None,
        }

    # ── 2. Read live crontab ──────────────────────────────────────
    live, err = read_live_crontab()
    if live is None:
        return {
            "healthy": False,
            "level": "RED",
            "msg": f"Cannot read live crontab: {err}",
            "live_hash": None,
            "expected_hash": expected_h,
        }

    live_h = _cron_hash(live)
    if live_h == expected_h:
        return {
            "healthy": True,
            "level": "OK",
            "msg": (f"live crontab matches docs/cron_setup.md "
                    f"(sha256={live_h[:12]}...)"),
            "live_hash": live_h,
            "expected_hash": expected_h,
        }

    # Drift
    return {
        "healthy": False,
        "level": "RED",
        "msg": (
            "crontab drift detected!\n"
            f"  live      sha256={live_h}\n"
            f"  expected  sha256={expected_h}\n"
            "Either crontab was modified without updating docs, or docs "
            "was updated without applying. Reconcile by editing one or "
            "the other and re-running this check."
        ),
        "live_hash": live_h,
        "expected_hash": expected_h,
    }


# ───────────────────────────────────────────────────────────────────────
# Render alert
# ───────────────────────────────────────────────────────────────────────

def format_for_feishu(status: dict) -> str:
    if status["healthy"]:
        return ""
    return "\n".join([
        "# 🚨 RED ALERT: production crontab drift",
        "",
        status["msg"],
        "",
        "Diagnostics:",
        "1. `crontab -l` to see what's currently scheduled",
        "2. `cat docs/cron_setup.md` to see expected '## Current crontab' block",
        "3. Reconcile drift:",
        "   - if **live is intended**: update docs/cron_setup.md then commit",
        "   - if **docs is intended**: write the docs block to /tmp/cron then",
        "     `crontab /tmp/cron` in Terminal.app (FDA blocks Claude shell)",
        "",
        "Source: `scripts/monitor/cron_drift_detect.py` (P6-X1, round 47)",
    ])


# ───────────────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    dry_run = "--dry-run" in argv

    status = check()
    # Always print status for cron log
    print(f"[cron_drift_detect] {status['level']}: {status['msg']}")

    if status["healthy"]:
        return 0

    block = format_for_feishu(status)
    if dry_run:
        print("=== DRY RUN — would send to Feishu ===")
        print(block)
        return 0

    try:
        from scripts.daily_report import send_to_feishu
    except Exception as e:
        print(f"[cron_drift_detect] cannot import send_to_feishu: {e}",
              file=sys.stderr)
        return 0

    ok = send_to_feishu(block)
    print(f"[cron_drift_detect] Feishu send returned {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
