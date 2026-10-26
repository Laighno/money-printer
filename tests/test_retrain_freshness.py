"""Retrain dead-man switch: threshold must match the real cadence.

2026-10-24 fired a RED "超期 13 天未运行" while the pipeline was healthy. The
job is scheduled Friday 18:30, but the Mac is asleep then, and the agreed
behaviour (2026-06-24: "睡过等醒来补跑") is for launchd to re-run on next wake.
Measured across 8 consecutive runs, every single one landed Sat/Sun/Mon, 1-3
days late - so two consecutive weekly runs can legitimately sit ~10 days apart.
An 8-day threshold paged for that normal drift; 12 days still catches a truly
missed week.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

import scripts.monitor.retrain_freshness_check as chk


@pytest.fixture()
def summary(tmp_path, monkeypatch):
    path = tmp_path / "auto_retrain_last.json"
    monkeypatch.setattr(chk, "SUMMARY", path)

    def write(days_ago: float, **over):
        d = {"refresh_ok": True, "cutoff": "20260917", "train_ok": True,
             "verify_pass": True, "swapped": True,
             "new_prefix": "data/blend_auto_20260917", "reasons": [],
             "ts": (datetime.now() - timedelta(days=days_ago)).isoformat(
                 timespec="seconds")}
        d.update(over)
        path.write_text(json.dumps(d), encoding="utf-8")
        return d

    return write


@pytest.fixture()
def sent(monkeypatch):
    box: list[str] = []
    monkeypatch.setattr(chk, "_send", lambda md: box.append(md))
    return box


def test_normal_weekend_catchup_does_not_page(summary, sent):
    """Fri-scheduled run that landed the following Sunday, 9 days on."""
    summary(9)
    assert chk.main() == 0
    assert sent == [], f"paged on a normal catch-up: {sent}"


def test_the_2026_10_24_false_alarm_would_not_fire_now(summary, sent):
    """That page reported 13 days; the run was late but the pipeline was fine."""
    summary(10)
    assert chk.main() == 0
    assert sent == []


def test_a_genuinely_missed_week_still_pages(summary, sent):
    summary(15)
    assert chk.main() == 0
    assert len(sent) == 1
    assert "超期" in sent[0]


def test_threshold_leaves_room_for_the_measured_lag():
    """Weekly + up to 3 days catch-up = 10; the bound must sit above it."""
    assert chk.RED_AGE >= timedelta(days=11), (
        "threshold is back inside the normal catch-up window")


def test_missing_summary_pages(tmp_path, monkeypatch, sent):
    monkeypatch.setattr(chk, "SUMMARY", tmp_path / "nope.json")
    assert chk.main() == 0
    assert len(sent) == 1


def test_corrupt_summary_pages_and_signals_failure(tmp_path, monkeypatch, sent):
    """Unparseable state is an error condition, so it also exits non-zero."""
    p = tmp_path / "auto_retrain_last.json"
    p.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(chk, "SUMMARY", p)
    assert chk.main() == 1
    assert len(sent) == 1


def test_fresh_run_that_failed_its_gate_is_not_silenced(summary, sent):
    """A recent run is not enough: a frozen model must still surface."""
    summary(1, verify_pass=False)
    assert chk.main() == 0
    assert len(sent) == 1, "gate failure within the window must still page"
