"""mp.risk.freeze fail-closed guard (2026-09-23).

Covers:
- DEFAULT_FREEZE_PATH / DEFAULT_HEARTBEAT_PATH are absolute, under PROJECT_ROOT
- guard_or_raise: dryrun always passes; live refuses on frozen flag, missing /
  stale / malformed heartbeat; MP_FREEZE_MONITOR_OPTIONAL=1 skips ONLY the
  heartbeat check; FREEZE_MONITOR_MAX_AGE_HOURS env override
- arm_b_stop_monitor.main writes the heartbeat on completed checks and NOT
  on internal error
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mp.common.paths import PROJECT_ROOT  # noqa: E402
from mp.risk import freeze as fz  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(fz.FREEZE_MONITOR_OPTIONAL_ENV, raising=False)
    monkeypatch.delenv(fz.FREEZE_MONITOR_MAX_AGE_ENV, raising=False)


@pytest.fixture
def paths(tmp_path):
    return {
        "flag": tmp_path / ".real_money_frozen",
        "hb": tmp_path / ".arm_b_monitor_heartbeat",
    }


def _write_hb(path: Path, age: timedelta, exit_code: int = 0) -> None:
    ts = datetime.now() - age
    path.write_text(json.dumps({"ts": ts.isoformat(timespec="seconds"),
                                "exit_code": exit_code}), encoding="utf-8")


def _guard(mode, paths, **kw):
    return fz.guard_or_raise(mode, path=paths["flag"],
                             heartbeat_path=paths["hb"], **kw)


# ── path anchoring ────────────────────────────────────────────────

def test_default_paths_are_absolute_under_project_root():
    assert fz.DEFAULT_FREEZE_PATH.is_absolute()
    assert fz.DEFAULT_HEARTBEAT_PATH.is_absolute()
    assert fz.DEFAULT_FREEZE_PATH == PROJECT_ROOT / "data" / ".real_money_frozen"
    assert fz.DEFAULT_HEARTBEAT_PATH == PROJECT_ROOT / "data" / ".arm_b_monitor_heartbeat"


# ── heartbeat primitives ──────────────────────────────────────────

def test_write_heartbeat_then_alive(paths):
    st = fz.write_monitor_heartbeat(exit_code=1, path=paths["hb"])
    assert paths["hb"].exists()
    raw = json.loads(paths["hb"].read_text(encoding="utf-8"))
    assert raw["exit_code"] == 1 and raw["ts"] == st["ts"]
    datetime.fromisoformat(raw["ts"])  # parseable
    alive, why = fz.check_monitor_alive(path=paths["hb"])
    assert alive, why
    assert not paths["hb"].with_name(paths["hb"].name + ".tmp").exists()


def test_check_alive_missing_file(paths):
    alive, why = fz.check_monitor_alive(path=paths["hb"])
    assert not alive and "does not exist" in why


def test_check_alive_stale_default_36h(paths):
    _write_hb(paths["hb"], timedelta(hours=37))
    alive, why = fz.check_monitor_alive(path=paths["hb"])
    assert not alive and "stale" in why
    _write_hb(paths["hb"], timedelta(hours=35))
    alive, _ = fz.check_monitor_alive(path=paths["hb"])
    assert alive


def test_check_alive_malformed(paths):
    paths["hb"].write_text("{not json", encoding="utf-8")
    alive, why = fz.check_monitor_alive(path=paths["hb"])
    assert not alive and "unreadable" in why
    paths["hb"].write_text(json.dumps({"no_ts": 1}), encoding="utf-8")
    alive, _ = fz.check_monitor_alive(path=paths["hb"])
    assert not alive
    paths["hb"].write_text(json.dumps({"ts": "garbage"}), encoding="utf-8")
    alive, why = fz.check_monitor_alive(path=paths["hb"])
    assert not alive and "unparsable" in why


def test_max_age_env_override(paths, monkeypatch):
    monkeypatch.setenv(fz.FREEZE_MONITOR_MAX_AGE_ENV, "1")
    assert fz.monitor_max_age_hours() == 1.0
    _write_hb(paths["hb"], timedelta(hours=2))
    alive, _ = fz.check_monitor_alive(path=paths["hb"])
    assert not alive
    _write_hb(paths["hb"], timedelta(minutes=30))
    alive, _ = fz.check_monitor_alive(path=paths["hb"])
    assert alive


def test_max_age_env_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(fz.FREEZE_MONITOR_MAX_AGE_ENV, "abc")
    assert fz.monitor_max_age_hours() == fz.FREEZE_MONITOR_MAX_AGE_HOURS_DEFAULT
    monkeypatch.setenv(fz.FREEZE_MONITOR_MAX_AGE_ENV, "-5")
    assert fz.monitor_max_age_hours() == fz.FREEZE_MONITOR_MAX_AGE_HOURS_DEFAULT


# ── guard_or_raise ────────────────────────────────────────────────

def test_dryrun_always_allowed_even_without_heartbeat(paths):
    _guard("dryrun", paths)  # no raise
    fz.freeze("x", path=paths["flag"])
    _guard("dryrun", paths)  # still no raise


@pytest.mark.parametrize("mode", ["auto", "interactive"])
def test_live_refused_when_heartbeat_missing(paths, mode):
    with pytest.raises(RuntimeError, match="Stop monitor is not alive"):
        _guard(mode, paths)


def test_live_allowed_with_fresh_heartbeat(paths):
    _write_hb(paths["hb"], timedelta(minutes=10))
    _guard("auto", paths)


def test_live_refused_when_heartbeat_stale(paths):
    _write_hb(paths["hb"], timedelta(hours=40))
    with pytest.raises(RuntimeError, match="stale"):
        _guard("auto", paths)


def test_live_refused_when_heartbeat_malformed(paths):
    paths["hb"].write_text("nope", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Stop monitor is not alive"):
        _guard("auto", paths)


def test_frozen_flag_wins_even_with_fresh_heartbeat(paths):
    _write_hb(paths["hb"], timedelta(minutes=1))
    fz.freeze("arm b -5pp", path=paths["flag"])
    with pytest.raises(RuntimeError, match="FROZEN"):
        _guard("auto", paths)


def test_optional_env_skips_heartbeat_but_not_freeze(paths, monkeypatch):
    monkeypatch.setenv(fz.FREEZE_MONITOR_OPTIONAL_ENV, "1")
    _guard("auto", paths)  # no heartbeat, escape hatch -> ok
    fz.freeze("arm b -5pp", path=paths["flag"])
    with pytest.raises(RuntimeError, match="FROZEN"):
        _guard("auto", paths)


def test_optional_env_must_be_exactly_1(paths, monkeypatch):
    monkeypatch.setenv(fz.FREEZE_MONITOR_OPTIONAL_ENV, "true")
    with pytest.raises(RuntimeError, match="Stop monitor is not alive"):
        _guard("auto", paths)


def test_guard_explicit_max_age_and_now(paths):
    _write_hb(paths["hb"], timedelta(hours=0))
    later = datetime.now() + timedelta(hours=3)
    with pytest.raises(RuntimeError, match="stale"):
        _guard("auto", paths, max_age_hours=2, now=later)
    _guard("auto", paths, max_age_hours=4, now=later)


def test_unfreeze_then_guard_passes(paths):
    _write_hb(paths["hb"], timedelta(minutes=1))
    fz.freeze("x", path=paths["flag"])
    fz.unfreeze(by="user", path=paths["flag"], approval_token="t")
    _guard("auto", paths)


# ── monitor writes heartbeat ──────────────────────────────────────

def _load_monitor():
    from scripts import arm_b_stop_monitor as m
    return m


@pytest.mark.parametrize("rc", [0, 1, 2])
def test_monitor_main_writes_heartbeat_on_completed_check(paths, monkeypatch, rc):
    m = _load_monitor()
    monkeypatch.setattr(m, "run_check", lambda simulate_trigger=False: rc)
    monkeypatch.setattr(
        m, "write_monitor_heartbeat",
        lambda **kw: fz.write_monitor_heartbeat(path=paths["hb"], **kw),
    )
    assert m.main([]) == rc
    raw = json.loads(paths["hb"].read_text(encoding="utf-8"))
    assert raw["exit_code"] == rc
    alive, _ = fz.check_monitor_alive(path=paths["hb"])
    assert alive


def test_monitor_main_no_heartbeat_on_internal_error(paths, monkeypatch):
    m = _load_monitor()

    def _boom(simulate_trigger=False):
        raise RuntimeError("nav unreadable")

    monkeypatch.setattr(m, "run_check", _boom)
    monkeypatch.setattr(
        m, "write_monitor_heartbeat",
        lambda **kw: fz.write_monitor_heartbeat(path=paths["hb"], **kw),
    )
    assert m.main([]) == 3
    assert not paths["hb"].exists()


def test_monitor_heartbeat_write_failure_does_not_mask_verdict(paths, monkeypatch):
    m = _load_monitor()
    monkeypatch.setattr(m, "run_check", lambda simulate_trigger=False: 2)

    def _fail(**kw):
        raise OSError("disk full")

    monkeypatch.setattr(m, "write_monitor_heartbeat", _fail)
    assert m.main([]) == 2
