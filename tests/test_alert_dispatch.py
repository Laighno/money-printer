"""P8-α-3 tests for mp/monitor/alert_dispatch.py (docs/dialog/ round 53).

Covers the multi-channel belt-and-suspenders contract:

  1. happy path — all 3 channels succeed
  2. Feishu fails → other 2 still succeed, dispatch_alert does NOT raise
  3. JSONL fails (permission denied) → Feishu + stderr still succeed,
     dispatch_alert does NOT raise
  4. all 3 fail → dispatch_alert returns dict, does NOT raise
  5. ALERTS_LOG parent dir missing → auto-created

The non-raise contract is load-bearing — monitor callers don't wrap
this in their own try/except for channel-level errors.
"""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

# Add repo root to sys.path so the import works in the test runner.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mp.monitor import alert_dispatch as ad


@pytest.fixture
def tmp_alerts_log(tmp_path, monkeypatch):
    """Redirect ALERTS_LOG to a tmp file we can inspect / sabotage."""
    p = tmp_path / "logs" / "alerts.jsonl"
    monkeypatch.setattr(ad, "ALERTS_LOG", p)
    return p


# ───────────────────────────────────────────────────────────────────────
# 1. Happy path — all 3 channels succeed
# ───────────────────────────────────────────────────────────────────────

def test_all_three_channels_succeed(tmp_alerts_log, monkeypatch, capsys):
    sent_feishu = []
    monkeypatch.setattr(
        "scripts.daily_report.send_to_feishu",
        lambda body, **kw: sent_feishu.append(body) or True,
    )

    results = ad.dispatch_alert(
        level="YELLOW",
        title="test alert",
        body="# Hello\nBody text",
        source="unittest",
    )

    assert results == {"feishu": "ok", "jsonl": "ok", "stderr": "ok"}
    # Feishu got body verbatim (no prepend)
    assert sent_feishu == ["# Hello\nBody text"]
    # JSONL contains one line with the full record
    line = tmp_alerts_log.read_text().splitlines()[0]
    rec = json.loads(line)
    assert rec["level"] == "YELLOW"
    assert rec["title"] == "test alert"
    assert rec["body"] == "# Hello\nBody text"
    assert rec["source"] == "unittest"
    assert "ts" in rec
    # stderr line printed
    captured = capsys.readouterr()
    assert "[ALERT YELLOW]" in captured.err
    assert "unittest" in captured.err
    assert "test alert" in captured.err


# ───────────────────────────────────────────────────────────────────────
# 2. Feishu fails — other two still succeed
# ───────────────────────────────────────────────────────────────────────

def test_feishu_fails_other_two_succeed(tmp_alerts_log, monkeypatch):
    def boom(body, **kw):
        raise RuntimeError("lark-cli not found (test)")
    monkeypatch.setattr("scripts.daily_report.send_to_feishu", boom)

    # Must not raise
    results = ad.dispatch_alert(
        level="RED", title="t", body="b", source="unittest",
    )
    assert results["feishu"].startswith("err: RuntimeError"), results
    assert results["jsonl"] == "ok"
    assert results["stderr"] == "ok"

    # JSONL still got the record
    assert tmp_alerts_log.exists()
    assert "RED" in tmp_alerts_log.read_text()


# ───────────────────────────────────────────────────────────────────────
# 3. JSONL fails — Feishu + stderr still succeed
# ───────────────────────────────────────────────────────────────────────

def test_jsonl_fails_other_two_succeed(tmp_path, monkeypatch):
    sent_feishu = []
    monkeypatch.setattr(
        "scripts.daily_report.send_to_feishu",
        lambda body, **kw: sent_feishu.append(body) or True,
    )
    # Make ALERTS_LOG a path inside a directory we can't write to.
    # Easiest: point ALERTS_LOG.parent at an existing FILE (which kills
    # both mkdir and open in append mode).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    # Now alerts.jsonl would be at blocker/alerts.jsonl — but blocker is
    # a regular file, so mkdir(parents=True, exist_ok=True) raises
    # FileExistsError when it sees the path exists as a file.
    monkeypatch.setattr(ad, "ALERTS_LOG", blocker / "alerts.jsonl")

    results = ad.dispatch_alert(
        level="RED", title="t", body="b", source="unittest",
    )
    assert results["feishu"] == "ok"
    assert results["jsonl"].startswith("err:"), results
    assert results["stderr"] == "ok"
    assert sent_feishu == ["b"]


# ───────────────────────────────────────────────────────────────────────
# 4. All 3 fail — dispatch_alert does NOT raise, returns dict
# ───────────────────────────────────────────────────────────────────────

def test_all_three_fail_no_raise(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("primary fail")
    monkeypatch.setattr("scripts.daily_report.send_to_feishu", boom)

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setattr(ad, "ALERTS_LOG", blocker / "alerts.jsonl")

    # Sabotage stderr by replacing sys.stderr with an object whose write raises
    class _BadStderr:
        def write(self, *a, **kw): raise RuntimeError("stderr broken")
        def flush(self): pass
    monkeypatch.setattr(sys, "stderr", _BadStderr())

    # Must not raise
    results = ad.dispatch_alert(
        level="RED", title="t", body="b", source="unittest",
    )
    # Restore stderr now so subsequent test output works
    monkeypatch.undo()
    assert all(v.startswith("err:") for v in results.values()), results


# ───────────────────────────────────────────────────────────────────────
# 5. ALERTS_LOG parent dir auto-created
# ───────────────────────────────────────────────────────────────────────

def test_jsonl_parent_dir_autocreated(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.daily_report.send_to_feishu",
        lambda body, **kw: True,
    )
    nested = tmp_path / "a" / "b" / "c" / "alerts.jsonl"
    assert not nested.parent.exists()
    monkeypatch.setattr(ad, "ALERTS_LOG", nested)

    results = ad.dispatch_alert(
        level="YELLOW", title="t", body="b", source="unittest",
    )
    assert results["jsonl"] == "ok"
    assert nested.parent.is_dir()
    assert nested.exists()


# ───────────────────────────────────────────────────────────────────────
# 6. (extra) body passed to Feishu verbatim — no title prepend
# ───────────────────────────────────────────────────────────────────────

def test_body_passed_verbatim_no_prepend(tmp_alerts_log, monkeypatch):
    sent = []
    monkeypatch.setattr(
        "scripts.daily_report.send_to_feishu",
        lambda body, **kw: sent.append(body),
    )
    body = "# 🚨 RED ALERT: weekly heartbeat\n\ndetail line"
    ad.dispatch_alert(
        level="RED", title="X: heartbeat", body=body, source="heartbeat",
    )
    # No "# X: heartbeat" header prepended — body is final markdown
    assert sent == [body]


# ───────────────────────────────────────────────────────────────────────
# 7. (extra) multiple alerts append (don't overwrite)
# ───────────────────────────────────────────────────────────────────────

def test_multiple_alerts_append(tmp_alerts_log, monkeypatch):
    monkeypatch.setattr(
        "scripts.daily_report.send_to_feishu",
        lambda body, **kw: True,
    )
    ad.dispatch_alert(level="YELLOW", title="a1", body="b1", source="s1")
    ad.dispatch_alert(level="RED", title="a2", body="b2", source="s2")
    lines = tmp_alerts_log.read_text().strip().splitlines()
    assert len(lines) == 2
    r1 = json.loads(lines[0])
    r2 = json.loads(lines[1])
    assert r1["title"] == "a1"
    assert r2["title"] == "a2"
    assert r1["source"] == "s1"
    assert r2["source"] == "s2"
