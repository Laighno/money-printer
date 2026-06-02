"""P6-X1 tests for scripts/monitor/cron_drift_detect.py (docs/dialog/ round 47).

Covers:
- _normalize_cron strips comments / blanks / trailing whitespace
- _cron_hash is stable across cosmetic edits, sensitive to real ones
- extract_expected_cron parser: happy / missing-anchor / missing-fence
- read_live_crontab: happy / timeout / nonzero rc / FileNotFound
- check() composes correctly: drift / match / docs-missing / live-error
- main() --dry-run does not contact Feishu
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.monitor import cron_drift_detect as cdd


# ───────────────────────────────────────────────────────────────────────
# _normalize_cron
# ───────────────────────────────────────────────────────────────────────

def test_normalize_strips_blank_lines():
    n = cdd._normalize_cron("\n\n0 18 * * 5 a\n\n0 6 * * 6 b\n\n")
    assert n == "0 18 * * 5 a\n0 6 * * 6 b"


def test_normalize_strips_full_line_comments():
    n = cdd._normalize_cron("# header\n0 18 * * 5 a\n# another\n0 6 * * 6 b\n")
    assert n == "0 18 * * 5 a\n0 6 * * 6 b"


def test_normalize_strips_trailing_whitespace():
    n = cdd._normalize_cron("0 18 * * 5 a   \n0 6 * * 6 b\t\n")
    assert n == "0 18 * * 5 a\n0 6 * * 6 b"


def test_normalize_preserves_inline_pseudo_comments():
    """A # in the middle of a cron line is part of the command, not a
    full-line comment.  Don't strip."""
    line = "0 18 * * 5 echo hi # part of cmd"
    n = cdd._normalize_cron(line)
    assert n == line


# ───────────────────────────────────────────────────────────────────────
# _cron_hash
# ───────────────────────────────────────────────────────────────────────

def test_hash_comment_only_diff_is_equal():
    """Cosmetic edits to comments should NOT cause drift alert."""
    a = "# old explanation\n0 18 * * 5 cmd\n"
    b = "# new explanation\n# more notes\n0 18 * * 5 cmd\n"
    assert cdd._cron_hash(a) == cdd._cron_hash(b)


def test_hash_real_change_differs():
    a = "0 18 * * 5 /old/path\n"
    b = "0 18 * * 5 /new/path\n"
    assert cdd._cron_hash(a) != cdd._cron_hash(b)


def test_hash_whitespace_only_diff_is_equal():
    a = "0 18 * * 5 cmd\n\n\n"
    b = "0 18 * * 5 cmd"
    assert cdd._cron_hash(a) == cdd._cron_hash(b)


# ───────────────────────────────────────────────────────────────────────
# extract_expected_cron parser
# ───────────────────────────────────────────────────────────────────────

def test_extract_happy_path():
    docs = (
        "# Cron setup\n\n"
        "## Current crontab\n\n"
        "```cron\n"
        "0 18 * * 5 /usr/bin/python wf.py\n"
        "0 6 * * 6 /usr/bin/python hb.py\n"
        "```\n\n"
        "## Other section\n"
        "```cron\n"
        "ignored 9 * * * * cmd\n"
        "```\n"
    )
    block = cdd.extract_expected_cron(docs)
    assert "0 18 * * 5" in block
    assert "0 6 * * 6" in block
    assert "ignored" not in block, (
        "extract must pick the FIRST cron block after the anchor, not later"
    )


def test_extract_missing_anchor_raises():
    docs = "# Cron setup\n\n## Other section\n\n```cron\n0 18 * * 5 cmd\n```\n"
    with pytest.raises(ValueError, match="missing anchor"):
        cdd.extract_expected_cron(docs)


def test_extract_missing_fence_raises():
    """Anchor present but no ```cron``` block anywhere after it → raise."""
    docs = (
        "```cron\n"
        "this fence is BEFORE the anchor and must be ignored\n"
        "```\n"
        "## Current crontab\n\n"
        "Just text, no fenced block after the anchor.\n"
        "## Other section with only python fence below\n"
        "```python\n"
        "print('not a cron fence')\n"
        "```\n"
    )
    with pytest.raises(ValueError, match="no ```cron``` fenced block follows"):
        cdd.extract_expected_cron(docs)


def test_extract_real_repo_cron_setup():
    """Smoke test against the actual checked-in docs file — drift
    detect can only work if extract succeeds on the real doc."""
    repo = Path(__file__).resolve().parent.parent
    docs_text = (repo / "docs" / "cron_setup.md").read_text()
    block = cdd.extract_expected_cron(docs_text)
    # The current production crontab references walk_forward_backtest.py
    # and weekly_heartbeat.py.  If this assertion ever fails, somebody
    # edited the docs in a way that broke the parser.
    assert "walk_forward_backtest.py" in block
    assert "weekly_heartbeat.py" in block


# ───────────────────────────────────────────────────────────────────────
# read_live_crontab
# ───────────────────────────────────────────────────────────────────────

def _make_completed_proc(rc: int, stdout: str = "", stderr: str = ""):
    """Build a subprocess.CompletedProcess fake."""
    return subprocess.CompletedProcess(
        args=["crontab", "-l"], returncode=rc, stdout=stdout, stderr=stderr,
    )


def test_read_live_happy(monkeypatch):
    def fake_run(args, **kw):
        return _make_completed_proc(rc=0, stdout="0 18 * * 5 cmd\n")
    monkeypatch.setattr(cdd.subprocess, "run", fake_run)
    text, err = cdd.read_live_crontab()
    assert err is None
    assert text == "0 18 * * 5 cmd\n"


def test_read_live_timeout(monkeypatch):
    def boom(args, **kw):
        raise subprocess.TimeoutExpired(cmd=args, timeout=10)
    monkeypatch.setattr(cdd.subprocess, "run", boom)
    text, err = cdd.read_live_crontab()
    assert text is None
    assert "timed out" in err
    assert "Full Disk Access" in err


def test_read_live_no_binary(monkeypatch):
    def boom(args, **kw):
        raise FileNotFoundError("no crontab")
    monkeypatch.setattr(cdd.subprocess, "run", boom)
    text, err = cdd.read_live_crontab()
    assert text is None
    assert "not found" in err


def test_read_live_nonzero_rc(monkeypatch):
    def fake_run(args, **kw):
        return _make_completed_proc(rc=1, stdout="", stderr="no crontab for user")
    monkeypatch.setattr(cdd.subprocess, "run", fake_run)
    text, err = cdd.read_live_crontab()
    assert text is None
    assert "rc=1" in err
    assert "no crontab for user" in err


# ───────────────────────────────────────────────────────────────────────
# check() integration
# ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_docs(tmp_path, monkeypatch):
    """Redirect DOCS_PATH to a controllable temp file."""
    p = tmp_path / "cron_setup.md"
    monkeypatch.setattr(cdd, "DOCS_PATH", p)
    return p


def _docs_block(cron_body: str) -> str:
    return (
        "# Cron setup\n\n"
        "## Current crontab\n\n"
        "```cron\n"
        f"{cron_body}\n"
        "```\n"
    )


def test_check_match(tmp_docs, monkeypatch):
    body = "0 18 * * 5 cmd_a\n0 6 * * 6 cmd_b"
    tmp_docs.write_text(_docs_block(body))
    monkeypatch.setattr(
        cdd, "read_live_crontab",
        lambda: (body + "\n", None),
    )
    status = cdd.check()
    assert status["healthy"] is True
    assert status["level"] == "OK"
    assert status["live_hash"] == status["expected_hash"]


def test_check_drift(tmp_docs, monkeypatch):
    tmp_docs.write_text(_docs_block("0 18 * * 5 docs_cmd"))
    monkeypatch.setattr(
        cdd, "read_live_crontab",
        lambda: ("0 18 * * 5 LIVE_cmd\n", None),
    )
    status = cdd.check()
    assert status["healthy"] is False
    assert status["level"] == "RED"
    assert status["live_hash"] != status["expected_hash"]
    assert "drift detected" in status["msg"]


def test_check_docs_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cdd, "DOCS_PATH", tmp_path / "missing.md")
    status = cdd.check()
    assert status["level"] == "RED"
    assert "not found" in status["msg"]


def test_check_docs_no_anchor(tmp_docs, monkeypatch):
    """Anchor missing in docs → RED with parser error message."""
    tmp_docs.write_text("# Just a doc, no anchor.\n")
    monkeypatch.setattr(cdd, "read_live_crontab", lambda: ("foo\n", None))
    status = cdd.check()
    assert status["level"] == "RED"
    assert "anchor" in status["msg"].lower() or "extract" in status["msg"].lower()


def test_check_live_unreadable(tmp_docs, monkeypatch):
    tmp_docs.write_text(_docs_block("0 18 * * 5 cmd"))
    monkeypatch.setattr(
        cdd, "read_live_crontab",
        lambda: (None, "timed out (Full Disk Access prompt?)"),
    )
    status = cdd.check()
    assert status["level"] == "RED"
    assert "Cannot read live crontab" in status["msg"]
    assert "Full Disk Access" in status["msg"]


def test_check_comment_only_diff_is_match(tmp_docs, monkeypatch):
    """Confirm the normalize → hash chain ignores comment-only edits in
    check() too (not just in unit tests for _normalize)."""
    tmp_docs.write_text(_docs_block("# notes A\n0 18 * * 5 same_cmd"))
    monkeypatch.setattr(
        cdd, "read_live_crontab",
        lambda: ("# notes B\n# notes C\n0 18 * * 5 same_cmd\n", None),
    )
    status = cdd.check()
    assert status["healthy"] is True, (
        f"comment-only diff triggered false-positive drift: {status['msg']}"
    )


# ───────────────────────────────────────────────────────────────────────
# main() Feishu wiring
# ───────────────────────────────────────────────────────────────────────

def test_main_healthy_no_feishu(tmp_docs, monkeypatch, capsys):
    body = "0 18 * * 5 cmd"
    tmp_docs.write_text(_docs_block(body))
    monkeypatch.setattr(cdd, "read_live_crontab", lambda: (body + "\n", None))

    def boom(*a, **kw):
        raise AssertionError("send_to_feishu must NOT be called on healthy")
    monkeypatch.setattr("scripts.daily_report.send_to_feishu", boom)

    rc = cdd.main([])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_main_dry_run_no_feishu(tmp_docs, monkeypatch, capsys):
    tmp_docs.write_text(_docs_block("0 18 * * 5 docs_cmd"))
    monkeypatch.setattr(cdd, "read_live_crontab",
                        lambda: ("0 18 * * 5 LIVE_cmd\n", None))

    def boom(*a, **kw):
        raise AssertionError("send_to_feishu must NOT be called in --dry-run")
    monkeypatch.setattr("scripts.daily_report.send_to_feishu", boom)

    rc = cdd.main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "RED ALERT" in out


def test_main_drift_calls_feishu(tmp_docs, monkeypatch, capsys):
    tmp_docs.write_text(_docs_block("0 18 * * 5 docs_cmd"))
    monkeypatch.setattr(cdd, "read_live_crontab",
                        lambda: ("0 18 * * 5 LIVE_cmd\n", None))
    calls = []
    monkeypatch.setattr(
        "scripts.daily_report.send_to_feishu",
        lambda md, **kw: calls.append(md) or True,
    )
    rc = cdd.main([])
    assert rc == 0
    assert len(calls) == 1
    assert "RED ALERT" in calls[0]


def test_format_includes_drift_diagnostics():
    drift_status = {
        "healthy": False,
        "level": "RED",
        "msg": "crontab drift detected!\n  live      sha256=abc\n  expected  sha256=def",
        "live_hash": "abc",
        "expected_hash": "def",
    }
    msg = cdd.format_for_feishu(drift_status)
    assert "🚨" in msg
    assert "crontab -l" in msg
    assert "docs/cron_setup.md" in msg
    assert "Terminal.app" in msg   # FDA hint
