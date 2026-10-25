"""Pre-market liveness alarm, retargeted at the name bridge (2026-10-25).

The old check read ``data/bridge/heartbeat.json``. That file still exists on
ECS but is frozen at its last pre-ban write (2026-10-23 20:52), so a check
pointed at it would fire RED every morning forever while the real transport is
healthy. These tests pin the new source of truth: the newest ``<epoch>.hb``
filename under the strategy's hb directory.
"""

from __future__ import annotations

import time

import pytest

import scripts.monitor.bridge_heartbeat_check as chk


@pytest.fixture()
def hb_dir(tmp_path, monkeypatch):
    d = tmp_path / "hb"
    d.mkdir()
    monkeypatch.setattr(chk, "HB_DIR", d)
    return d


def test_age_comes_from_the_newest_filename(hb_dir):
    now = int(time.time())
    for offset in (600, 120, 4):
        (hb_dir / f"{now - offset}.hb").touch()
    age = chk._heartbeat_age()
    assert age is not None and 0 <= age <= 15


def test_missing_directory_or_files_reads_as_unknown(hb_dir, monkeypatch, tmp_path):
    assert chk._heartbeat_age() is None, "empty dir -> unknown, not 0"
    monkeypatch.setattr(chk, "HB_DIR", tmp_path / "does-not-exist")
    assert chk._heartbeat_age() is None


def test_non_heartbeat_files_are_ignored(hb_dir):
    (hb_dir / "notes.txt").touch()
    (hb_dir / "garbage.hb").touch()          # non-numeric stem
    assert chk._heartbeat_age() is None
    (hb_dir / f"{int(time.time())}.hb").touch()
    assert chk._heartbeat_age() is not None


def test_fresh_heartbeat_is_quiet(hb_dir, monkeypatch):
    sent = []
    monkeypatch.setattr(chk, "_send", lambda md: sent.append(md))
    monkeypatch.setattr(chk, "_prune", lambda: None)
    monkeypatch.setattr(chk, "_is_weekend", lambda: False)
    (hb_dir / f"{int(time.time())}.hb").touch()

    assert chk.main() == 0
    assert sent == [], "a healthy bridge must not page anyone"


def test_stale_heartbeat_alerts_and_names_the_strategy(hb_dir, monkeypatch):
    sent = []
    monkeypatch.setattr(chk, "_send", lambda md: sent.append(md))
    monkeypatch.setattr(chk, "_prune", lambda: None)
    monkeypatch.setattr(chk, "_is_weekend", lambda: False)
    (hb_dir / f"{int(time.time()) - 7200}.hb").touch()

    assert chk.main() == 0
    assert len(sent) == 1
    assert "NAMEBRIDGE" in sent[0], "the alert must name the strategy to restart"
    assert "MONEY" not in sent[0], "MONEY was the dead file-bridge strategy"


def test_absent_heartbeat_alerts(hb_dir, monkeypatch):
    sent = []
    monkeypatch.setattr(chk, "_send", lambda md: sent.append(md))
    monkeypatch.setattr(chk, "_prune", lambda: None)
    monkeypatch.setattr(chk, "_is_weekend", lambda: False)

    assert chk.main() == 0
    assert len(sent) == 1 and "NAMEBRIDGE" in sent[0]


def test_weekend_is_quiet_even_when_dead(hb_dir, monkeypatch):
    sent = []
    monkeypatch.setattr(chk, "_send", lambda md: sent.append(md))
    monkeypatch.setattr(chk, "_is_weekend", lambda: True)
    assert chk.main() == 0
    assert sent == []


def test_check_points_at_the_name_bridge_not_the_dead_json_one():
    """Guard against a revert to the frozen heartbeat.json."""
    hb = str(chk.HB_DIR).replace("\\", "/")
    assert hb.endswith("mp_orders/hb"), f"HB_DIR is {hb}"
    assert "data/bridge" not in hb
