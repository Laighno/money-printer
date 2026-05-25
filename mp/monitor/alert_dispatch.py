"""Centralized alert dispatch with multi-channel fallback.

P8-α-3 (docs/dialog/ round 53): until this module existed all four
production monitors (`threshold_alert`, `weekly_heartbeat`,
`cron_drift_detect`, `paper_trade_drift_detect`) sent alerts via a
single channel — ``scripts.daily_report.send_to_feishu`` → ``lark-cli``
→ Feishu webhook.  If ``lark-cli`` was missing, the webhook secret was
rotated, the Feishu app was muted, or the host had no network, **all
four monitors went silent**.  That single point of failure was
acceptable in dryrun (lost alerts cost nothing) but is unacceptable
for the upcoming live-trading phase (a silently-dropped Sharpe-breach
alert during real trading = real losses unnoticed).

This module ships alerts through three independent channels in a
**belt-and-suspenders** way:

  1. **Primary** — ``scripts.daily_report.send_to_feishu`` (existing
     lark-cli → Feishu).  Best for low-friction visibility.
  2. **Fallback 1** — append a JSONL record to ``data/logs/alerts.jsonl``
     (durable, grep-able audit trail).
  3. **Fallback 2** — write to stderr (visible if running interactively
     or captured by cron mail).

All three **always fire** — not "fallback only if primary fails".
Reason: primary may silently succeed (subprocess returns 200 OK) while
downstream delivery has actually dropped (webhook misconfigured, app
push muted on the operator's device, group chat archived).  The other
two channels are unaffected by any of those failure modes.

Public API
----------

``dispatch_alert(level, title, body, source) -> dict``

  Always returns a per-channel result dict.  **Never raises** — alert
  dispatch must not propagate exceptions to the caller (an alert failure
  must not break the monitor itself).

  ``body`` should be the complete markdown alert (including any emoji
  / heading lines the monitor wants in the Feishu render).  This module
  does **not** wrap or prepend anything — the body the caller passes is
  what Feishu shows.  ``title`` is used only for the JSONL audit row
  and the stderr line; it is a short human label like ``"YELLOW: weekly
  walk-forward heartbeat"`` for grep-ability.

Usage
-----

    from mp.monitor.alert_dispatch import dispatch_alert

    block = format_for_feishu(status)  # caller builds markdown
    results = dispatch_alert(
        level="RED",
        title="weekly walk-forward heartbeat",
        body=block,
        source="heartbeat",
    )
    # results == {"feishu": "ok", "jsonl": "ok", "stderr": "ok"}
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


# Resolve repo root robustly so the JSONL log lands in the right place
# regardless of which directory the caller cwd'd into.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent.parent
ALERTS_LOG = _REPO / "data" / "logs" / "alerts.jsonl"


def dispatch_alert(
    level: str,
    title: str,
    body: str,
    source: str,
) -> dict:
    """Dispatch an alert through all three channels.

    Parameters
    ----------
    level : str
        Severity tag, conventionally one of ``OK`` / ``YELLOW`` / ``RED``.
        Stored verbatim in the JSONL record and the stderr line.
    title : str
        Short human-readable label (used for the JSONL record and the
        stderr first line).  Should be unique enough to grep for in
        ``data/logs/alerts.jsonl``.  Example:
        ``"RED: weekly walk-forward heartbeat (file age 21d)"``.
    body : str
        Full markdown alert payload — what the Feishu message renders
        as-is.  Callers should pre-format with whatever emojis / sub-
        headings they want.  Do **not** wrap title here; the caller is
        responsible for the rendered look.
    source : str
        Identifier for the upstream monitor (``"threshold_alert"`` /
        ``"heartbeat"`` / ``"cron_drift"`` / ``"paper_drift"``).
        Recorded in the JSONL row for filtering.

    Returns
    -------
    dict
        Per-channel result: ``{"feishu": "ok"|"err: ...", "jsonl": ...,
        "stderr": ...}``.  Useful for tests; in production callers
        typically ignore the return value.

    Notes
    -----
    Never raises.  An exception during any channel's send is caught,
    converted to an ``"err: <type>: <msg>"`` string, and recorded in the
    return dict.  The other channels still attempt their send.  This
    contract is load-bearing: monitors call this from inside their own
    try/except (since alert dispatch failure must not abort the monitor
    itself), and the inner try/except here prevents the monitor's outer
    handler from ever firing on a channel-level error.
    """
    ts = datetime.now().isoformat(timespec="seconds")
    record = {
        "ts": ts,
        "level": level,
        "title": title,
        "body": body,
        "source": source,
    }
    results: dict = {}

    # ── Channel 1: Feishu (primary) ────────────────────────────
    # Body is passed verbatim — caller is expected to ship a complete
    # markdown alert already.  No prepending of title here.
    try:
        from scripts.daily_report import send_to_feishu
        send_to_feishu(body)
        results["feishu"] = "ok"
    except Exception as e:
        results["feishu"] = f"err: {type(e).__name__}: {e}"

    # ── Channel 2: durable JSONL audit (fallback 1) ────────────
    # One line per alert; preserves full body so a Feishu-silent week
    # is still recoverable from disk.  Auto-creates the parent dir
    # because cron startups sometimes hit `data/logs/` not existing.
    try:
        ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ALERTS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        results["jsonl"] = "ok"
    except Exception as e:
        results["jsonl"] = f"err: {type(e).__name__}: {e}"

    # ── Channel 3: stderr (fallback 2) ─────────────────────────
    # Captured by cron mail if cron is configured to mail stderr;
    # visible in shell session if running interactively.  Cheap; will
    # only fail in pathological environments (closed stderr stream).
    try:
        sys.stderr.write(
            f"\n[ALERT {level}] {ts} | {source}\n{title}\n{body}\n"
        )
        sys.stderr.flush()
        results["stderr"] = "ok"
    except Exception as e:
        results["stderr"] = f"err: {type(e).__name__}: {e}"

    return results
