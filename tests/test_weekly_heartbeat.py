"""Tests for scripts/monitor/weekly_heartbeat.py (P5-B dead-man-switch).

P6-X2 update (docs/dialog/ round 47): thresholds are trading-day-aware,
so most tests now monkeypatch ``hb.trading_days_between`` / ``hb.calendar_available``
to control the trading-day count exactly. A few wall-clock tests are
kept to exercise the "import failed → fall back to wall-clock" path.
"""
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


@pytest.fixture
def stub_trading_calendar(monkeypatch):
    """Replace trading_days_between / calendar_available with deterministic
    stubs.  Returns a setter ``(td: int, cal_ok: bool) -> None``.
    """
    state = {"td": 0, "cal_ok": True}

    def _td(start, end):
        return state["td"]

    def _ok():
        return state["cal_ok"]

    monkeypatch.setattr(hb, "trading_days_between", _td)
    monkeypatch.setattr(hb, "calendar_available", _ok)

    def setter(td: int, cal_ok: bool = True):
        state["td"] = td
        state["cal_ok"] = cal_ok

    return setter


def _set_mtime(p: Path, days_ago: float):
    p.write_text("{}")
    target = time.time() - days_ago * 86400
    os.utime(p, (target, target))


# ───────────────────────────────────────────────────────────────────────
# Healthy path
# ───────────────────────────────────────────────────────────────────────

def test_healthy_fresh_file(tmp_snapshot, stub_trading_calendar):
    """1 day old, 1 trading day → OK."""
    _set_mtime(tmp_snapshot, days_ago=1)
    stub_trading_calendar(td=1)
    status = hb.check()
    assert status["healthy"] is True
    assert status["level"] == "OK"
    assert status["trading_days_since"] == 1


# ───────────────────────────────────────────────────────────────────────
# YELLOW / RED bands (trading-day-aware)
# ───────────────────────────────────────────────────────────────────────

def test_yellow_band_six_trading_days(tmp_snapshot, stub_trading_calendar):
    """6 trading days > 5 threshold → YELLOW."""
    _set_mtime(tmp_snapshot, days_ago=8)   # calendar age for display
    stub_trading_calendar(td=6)
    status = hb.check()
    assert status["healthy"] is False
    assert status["level"] == "YELLOW"
    assert status["trading_days_since"] == 6


def test_boundary_exactly_five_trading_days_is_ok(tmp_snapshot, stub_trading_calendar):
    """Threshold is `>5` not `>=5` — exactly 5 trading days is still OK
    (one full trading week, cron may have just barely fired on time)."""
    _set_mtime(tmp_snapshot, days_ago=7)
    stub_trading_calendar(td=5)
    status = hb.check()
    assert status["healthy"] is True
    assert status["level"] == "OK"


def test_red_band_eleven_trading_days(tmp_snapshot, stub_trading_calendar):
    """11 trading days > 10 → RED."""
    _set_mtime(tmp_snapshot, days_ago=15)
    stub_trading_calendar(td=11)
    status = hb.check()
    assert status["healthy"] is False
    assert status["level"] == "RED"


def test_boundary_exactly_ten_trading_days_is_yellow(tmp_snapshot, stub_trading_calendar):
    """Threshold is `>10` not `>=10` — exactly 10 trading days is still
    YELLOW, not yet RED."""
    _set_mtime(tmp_snapshot, days_ago=14)
    stub_trading_calendar(td=10)
    status = hb.check()
    assert status["level"] == "YELLOW"


# ───────────────────────────────────────────────────────────────────────
# Wall-clock RED safety net
# ───────────────────────────────────────────────────────────────────────

def test_red_safety_net_three_weeks_wall_clock(tmp_snapshot, stub_trading_calendar):
    """Even if trading_days_between reports a low number (e.g. calendar
    lied), 21d+ wall-clock age still escalates to RED."""
    _set_mtime(tmp_snapshot, days_ago=22)
    # Pretend trading_days returns 3 — way below RED threshold
    stub_trading_calendar(td=3)
    status = hb.check()
    assert status["level"] == "RED"


# ───────────────────────────────────────────────────────────────────────
# Calendar fallback (import / runtime failure)
# ───────────────────────────────────────────────────────────────────────

def test_calendar_import_failed_uses_wallclock_thresholds_yellow(tmp_snapshot, monkeypatch):
    """If trading_calendar can't be imported, fall back to ORIGINAL
    wall-clock thresholds (7d12h YELLOW)."""
    monkeypatch.setattr(hb, "trading_days_between", None)
    monkeypatch.setattr(hb, "calendar_available", None)
    monkeypatch.setattr(hb, "pd", None)
    _set_mtime(tmp_snapshot, days_ago=8)
    status = hb.check()
    assert status["level"] == "YELLOW"
    assert status["trading_days_since"] is None
    assert "import-failed" in status["calendar_source"]


def test_calendar_import_failed_uses_wallclock_thresholds_red(tmp_snapshot, monkeypatch):
    """trading_calendar unavailable + age > 14d wall-clock → RED."""
    monkeypatch.setattr(hb, "trading_days_between", None)
    monkeypatch.setattr(hb, "calendar_available", None)
    monkeypatch.setattr(hb, "pd", None)
    _set_mtime(tmp_snapshot, days_ago=15)
    status = hb.check()
    assert status["level"] == "RED"


def test_calendar_runtime_failure_does_not_crash(tmp_snapshot, monkeypatch):
    """If trading_days_between raises mid-call, the heartbeat still runs
    (falls back to wall-clock thresholds)."""
    def boom(start, end):
        raise RuntimeError("synthetic akshare error")
    monkeypatch.setattr(hb, "trading_days_between", boom)
    _set_mtime(tmp_snapshot, days_ago=8)
    status = hb.check()
    assert status["level"] == "YELLOW"
    assert "runtime-failed" in status["calendar_source"]


# ───────────────────────────────────────────────────────────────────────
# Missing file
# ───────────────────────────────────────────────────────────────────────

def test_missing_file_is_red(tmp_path, monkeypatch):
    monkeypatch.setattr(hb, "SNAPSHOT_PATH", tmp_path / "does_not_exist.json")
    status = hb.check()
    assert status["healthy"] is False
    assert status["level"] == "RED"
    assert "missing" in status["msg"].lower()


# ───────────────────────────────────────────────────────────────────────
# format_for_feishu rendering
# ───────────────────────────────────────────────────────────────────────

def test_format_for_feishu_healthy_empty(tmp_snapshot, stub_trading_calendar):
    _set_mtime(tmp_snapshot, days_ago=1)
    stub_trading_calendar(td=1)
    status = hb.check()
    assert hb.format_for_feishu(status) == ""


def test_format_for_feishu_red_contains_emojis_and_diagnostics(tmp_snapshot, stub_trading_calendar):
    _set_mtime(tmp_snapshot, days_ago=15)
    stub_trading_calendar(td=11)
    status = hb.check()
    msg = hb.format_for_feishu(status)
    assert "🚨" in msg
    assert "RED ALERT" in msg
    assert "crontab -l" in msg
    assert "model_update.log" in msg
    # P6-X2 added trading-days line in alert (round 47 spec)
    assert "Trading days since" in msg


def test_format_for_feishu_yellow_uses_warning_emoji(tmp_snapshot, stub_trading_calendar):
    _set_mtime(tmp_snapshot, days_ago=8)
    stub_trading_calendar(td=6)
    status = hb.check()
    msg = hb.format_for_feishu(status)
    assert "⚠" in msg
    assert "YELLOW ALERT" in msg


def test_format_for_feishu_shows_calendar_source(tmp_snapshot, stub_trading_calendar):
    """Alert message must surface the calendar source so operators can
    tell when we fell back to weekday counting."""
    _set_mtime(tmp_snapshot, days_ago=15)
    stub_trading_calendar(td=11, cal_ok=False)
    status = hb.check()
    msg = hb.format_for_feishu(status)
    assert "weekday-fallback" in msg


# ───────────────────────────────────────────────────────────────────────
# main() — dry run / healthy / alert wiring
# ───────────────────────────────────────────────────────────────────────

def test_main_dry_run_does_not_call_feishu(tmp_snapshot, capsys, monkeypatch, stub_trading_calendar):
    """--dry-run on RED status should print message but not import send_to_feishu."""
    _set_mtime(tmp_snapshot, days_ago=20)
    stub_trading_calendar(td=14)

    def boom_if_called(*a, **kw):
        raise AssertionError("send_to_feishu should NOT be called in --dry-run")

    monkeypatch.setattr("scripts.daily_report.send_to_feishu", boom_if_called)
    rc = hb.main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "RED ALERT" in out


def test_main_healthy_returns_0_no_send(tmp_snapshot, capsys, monkeypatch, stub_trading_calendar):
    _set_mtime(tmp_snapshot, days_ago=2)
    stub_trading_calendar(td=2)

    def boom_if_called(*a, **kw):
        raise AssertionError("send_to_feishu should NOT be called on healthy")

    monkeypatch.setattr("scripts.daily_report.send_to_feishu", boom_if_called)
    rc = hb.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


def test_main_alert_calls_send_to_feishu(tmp_snapshot, capsys, monkeypatch, stub_trading_calendar):
    _set_mtime(tmp_snapshot, days_ago=20)
    stub_trading_calendar(td=14)
    calls = []

    def fake_send(markdown, **kw):
        calls.append(markdown)
        return True

    monkeypatch.setattr("scripts.daily_report.send_to_feishu", fake_send)
    rc = hb.main([])
    assert rc == 0
    assert len(calls) == 1
    assert "RED ALERT" in calls[0]
