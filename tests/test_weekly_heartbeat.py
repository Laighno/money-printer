"""Tests for scripts/monitor/weekly_heartbeat.py (P5-B dead-man-switch)."""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.monitor import weekly_heartbeat as hb


@pytest.fixture
def tmp_snapshot(tmp_path, monkeypatch):
    """Redirect SNAPSHOT_PATH to a tmp file we control mtime on."""
    p = tmp_path / "backtest_history.json"
    monkeypatch.setattr(hb, "SNAPSHOT_PATH", p)
    return p


def _set_mtime(p: Path, days_ago: float):
    p.write_text("{}")
    target = time.time() - days_ago * 86400
    os.utime(p, (target, target))


def test_healthy_fresh_file(tmp_snapshot):
    _set_mtime(tmp_snapshot, days_ago=1)
    status = hb.check()
    assert status["healthy"] is True
    assert status["level"] == "OK"


def test_yellow_band(tmp_snapshot):
    """7d + 12h < age < 14d → YELLOW."""
    _set_mtime(tmp_snapshot, days_ago=8)
    status = hb.check()
    assert status["healthy"] is False
    assert status["level"] == "YELLOW"


def test_red_band_two_weeks(tmp_snapshot):
    _set_mtime(tmp_snapshot, days_ago=15)
    status = hb.check()
    assert status["healthy"] is False
    assert status["level"] == "RED"


def test_missing_file_is_red(tmp_path, monkeypatch):
    monkeypatch.setattr(hb, "SNAPSHOT_PATH", tmp_path / "does_not_exist.json")
    status = hb.check()
    assert status["healthy"] is False
    assert status["level"] == "RED"
    assert "missing" in status["msg"].lower()


def test_format_for_feishu_healthy_empty(tmp_snapshot):
    _set_mtime(tmp_snapshot, days_ago=1)
    status = hb.check()
    assert hb.format_for_feishu(status) == ""


def test_format_for_feishu_red_contains_emojis_and_diagnostics(tmp_snapshot):
    _set_mtime(tmp_snapshot, days_ago=15)
    status = hb.check()
    msg = hb.format_for_feishu(status)
    assert "🚨" in msg
    assert "RED ALERT" in msg
    assert "crontab -l" in msg
    assert "model_update.log" in msg


def test_format_for_feishu_yellow_uses_warning_emoji(tmp_snapshot):
    _set_mtime(tmp_snapshot, days_ago=8)
    status = hb.check()
    msg = hb.format_for_feishu(status)
    assert "⚠" in msg
    assert "YELLOW ALERT" in msg


def test_main_dry_run_does_not_call_feishu(tmp_snapshot, capsys, monkeypatch):
    """--dry-run on RED status should print message but not import send_to_feishu."""
    _set_mtime(tmp_snapshot, days_ago=20)

    def boom_if_called(*a, **kw):
        raise AssertionError("send_to_feishu should NOT be called in --dry-run")

    monkeypatch.setattr("scripts.daily_report.send_to_feishu", boom_if_called)
    rc = hb.main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "RED ALERT" in out


def test_main_healthy_returns_0_no_send(tmp_snapshot, capsys, monkeypatch):
    _set_mtime(tmp_snapshot, days_ago=2)

    def boom_if_called(*a, **kw):
        raise AssertionError("send_to_feishu should NOT be called on healthy")

    monkeypatch.setattr("scripts.daily_report.send_to_feishu", boom_if_called)
    rc = hb.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


def test_main_alert_calls_send_to_feishu(tmp_snapshot, capsys, monkeypatch):
    _set_mtime(tmp_snapshot, days_ago=20)
    calls = []

    def fake_send(markdown, **kw):
        calls.append(markdown)
        return True

    monkeypatch.setattr("scripts.daily_report.send_to_feishu", fake_send)
    rc = hb.main([])
    assert rc == 0
    assert len(calls) == 1
    assert "RED ALERT" in calls[0]
