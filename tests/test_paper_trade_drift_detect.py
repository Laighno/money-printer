"""P6-X3 tests for scripts/monitor/paper_trade_drift_detect.py
(docs/dialog/ round 47).

Covers:
- rolling_sharpe min-N floor (N < window+1 → None)
- rolling_sharpe happy path (mocked rising trend → positive Sharpe)
- rolling_sharpe zero-std → None
- check() cold-start guard (NAV history < 15 → OK with skip msg)
- check() drift detection: |Δ| > 0.5 → YELLOW, |Δ| > 1.0 + paper<0 → RED
- check() with walk_forward stale (>21d) → skip
- main() --dry-run does not call Feishu
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.monitor import paper_trade_drift_detect as dd


# ───────────────────────────────────────────────────────────────────────
# rolling_sharpe
# ───────────────────────────────────────────────────────────────────────

def test_rolling_sharpe_min_n_floor():
    """Needs window+1 NAV points; below that → None."""
    navs = [100.0] * dd.ROLLING_WINDOW   # exactly window points, need window+1
    assert dd.rolling_sharpe(navs) is None
    navs2 = [100.0] * (dd.ROLLING_WINDOW + 1)
    # All equal NAV → 0 std → still None
    assert dd.rolling_sharpe(navs2) is None


def _noisy_walk(start: float, drift: float, noise: float, n: int, seed: int = 0):
    """Deterministic random walk with given drift + noise std.

    Returns n NAV points. Drift in % per step, noise in % per step.
    Pure geometric series have zero std and rolling_sharpe returns None,
    so realistic NAV tests need a noise component.
    """
    import random
    rng = random.Random(seed)
    navs = [start]
    for _ in range(n - 1):
        step = drift + rng.gauss(0.0, noise)
        navs.append(navs[-1] * (1.0 + step))
    return navs


def test_rolling_sharpe_rising_trend_positive():
    """Up-drift with small noise → positive Sharpe."""
    navs = _noisy_walk(100.0, drift=0.005, noise=0.002, n=dd.ROLLING_WINDOW + 1, seed=1)
    s = dd.rolling_sharpe(navs)
    assert s is not None
    assert s > 1.0, f"up-drift should give positive Sharpe, got {s}"


def test_rolling_sharpe_negative_for_losing_trend():
    """Down-drift with small noise → negative Sharpe."""
    navs = _noisy_walk(100.0, drift=-0.005, noise=0.002, n=dd.ROLLING_WINDOW + 1, seed=2)
    s = dd.rolling_sharpe(navs)
    assert s is not None
    assert s < -1.0


def test_rolling_sharpe_zero_prev_returns_none():
    """Defensive: if some prior NAV is 0 or negative, return None
    rather than dividing by zero."""
    navs = [100.0] * (dd.ROLLING_WINDOW - 1) + [0.0, 0.0]
    assert dd.rolling_sharpe(navs) is None


# ───────────────────────────────────────────────────────────────────────
# Data loader helpers
# ───────────────────────────────────────────────────────────────────────

def test_parse_sharpe_field_string():
    assert dd._parse_sharpe_field({"bt_metrics": {"sharpe_ratio": "1.21"}}) == 1.21


def test_parse_sharpe_field_float():
    assert dd._parse_sharpe_field({"bt_metrics": {"sharpe_ratio": 1.21}}) == 1.21


def test_parse_sharpe_field_missing():
    assert dd._parse_sharpe_field({}) is None
    assert dd._parse_sharpe_field({"bt_metrics": {}}) is None
    assert dd._parse_sharpe_field({"bt_metrics": {"sharpe_ratio": None}}) is None


def test_parse_sharpe_field_unparseable():
    assert dd._parse_sharpe_field({"bt_metrics": {"sharpe_ratio": "not a number"}}) is None


def test_parse_history_date():
    assert dd._parse_history_date({"date": "2026-05-22"}) == datetime(2026, 5, 22)
    assert dd._parse_history_date({"date": ""}) is None
    assert dd._parse_history_date({}) is None
    assert dd._parse_history_date({"date": "2026/05/22"}) is None


# ───────────────────────────────────────────────────────────────────────
# check() integration
# ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_paths(tmp_path, monkeypatch):
    """Redirect STATE_PATH / HISTORY_PATH to a tmp dir we control."""
    state = tmp_path / "state.json"
    history = tmp_path / "backtest_history.json"
    monkeypatch.setattr(dd, "STATE_PATH", state)
    monkeypatch.setattr(dd, "HISTORY_PATH", history)
    return state, history


def _write_state(p: Path, nav_seq: list[float]):
    """Write a state.json with the given NAV sequence."""
    nav_history = [
        {"date": f"2026-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}",
         "cash": 0.0, "positions_value": 0.0, "nav": float(v)}
        for i, v in enumerate(nav_seq)
    ]
    p.write_text(json.dumps({"nav_history": nav_history}, indent=2))


def _write_history(p: Path, sharpe: float, date: str = "2026-05-22"):
    """Write a backtest_history.json with one entry."""
    entry = {
        "date": date,
        "bt_metrics": {"sharpe_ratio": f"{sharpe}"},
        "bt_benchmark": {},
        "model_ic": {},
        "model_hit_rate": {},
    }
    p.write_text(json.dumps([entry], indent=2))


# ── Cold start ─────────────────────────────────────────────────

def test_check_cold_start_below_min(fake_paths):
    """N < 15 → OK with skip msg, no alert."""
    state_p, history_p = fake_paths
    _write_state(state_p, [100.0] * 10)
    _write_history(history_p, sharpe=1.0)
    status = dd.check()
    assert status["healthy"] is True
    assert status["level"] == "OK"
    assert "insufficient NAV history" in status["msg"]
    assert status["nav_n"] == 10


def test_check_state_file_missing(fake_paths):
    """No state.json at all → cold-start, no alert."""
    state_p, history_p = fake_paths
    _write_history(history_p, sharpe=1.0)
    status = dd.check()
    assert status["healthy"] is True
    assert "insufficient NAV history" in status["msg"]
    assert status["nav_n"] == 0


# ── Healthy in-range ──────────────────────────────────────────

def test_check_in_tolerance_is_ok(fake_paths):
    """N=21, paper Sharpe ≈ wf Sharpe → OK no alert."""
    state_p, history_p = fake_paths
    # Steady NAV with 1% daily drift — paper sharpe ~mid
    navs = _noisy_walk(100.0, drift=0.001, noise=0.005, n=21, seed=11)
    _write_state(state_p, navs)
    paper = dd.rolling_sharpe(navs)
    assert paper is not None
    _write_history(history_p, sharpe=paper - 0.2, date="2026-05-22")
    status = dd.check(now=datetime(2026, 5, 23))
    assert status["healthy"] is True
    assert "Within ±" in status["msg"]


# ── YELLOW: |Δ| > 0.5 ─────────────────────────────────────────

def test_check_yellow_divergence(fake_paths):
    state_p, history_p = fake_paths
    navs = _noisy_walk(100.0, drift=0.003, noise=0.001, n=21, seed=12)   # strong up → high Sharpe
    _write_state(state_p, navs)
    paper = dd.rolling_sharpe(navs)
    assert paper is not None
    # walk_forward Sharpe far below paper → Δ > 0.5
    _write_history(history_p, sharpe=paper - 5.0, date="2026-05-22")
    status = dd.check(now=datetime(2026, 5, 23))
    assert status["healthy"] is False
    assert status["level"] == "YELLOW"
    assert abs(status["delta"]) > dd.YELLOW_THRESHOLD


# ── RED: |Δ| > 1.0 AND paper < 0 ──────────────────────────────

def test_check_red_negative_paper_and_large_delta(fake_paths):
    """Paper negative AND |Δ| > 1.0 → RED."""
    state_p, history_p = fake_paths
    # Steady downtrend → strongly negative Sharpe
    navs = _noisy_walk(100.0, drift=-0.003, noise=0.001, n=21, seed=13)
    _write_state(state_p, navs)
    paper = dd.rolling_sharpe(navs)
    assert paper is not None and paper < 0
    # wf says +3.0 → Δ very large
    _write_history(history_p, sharpe=3.0, date="2026-05-22")
    status = dd.check(now=datetime(2026, 5, 23))
    assert status["healthy"] is False
    assert status["level"] == "RED"
    assert abs(status["delta"]) > dd.RED_THRESHOLD
    assert status["paper_sharpe"] < 0


def test_check_red_threshold_but_positive_paper_only_yellow(fake_paths):
    """|Δ| > 1.0 BUT paper Sharpe positive → YELLOW, not RED
    (paper outperforming is unusual but not 'execution broken')."""
    state_p, history_p = fake_paths
    navs = _noisy_walk(100.0, drift=0.003, noise=0.001, n=21, seed=12)
    _write_state(state_p, navs)
    paper = dd.rolling_sharpe(navs)
    assert paper is not None and paper > 0
    # wf 5 units lower — |Δ|>1, but paper is positive
    _write_history(history_p, sharpe=paper - 5.0, date="2026-05-22")
    status = dd.check(now=datetime(2026, 5, 23))
    assert status["level"] == "YELLOW", (
        "RED requires paper<0; positive paper with large Δ stays YELLOW"
    )


# ── Walk-forward staleness ────────────────────────────────────

def test_check_walk_forward_stale_skips(fake_paths):
    state_p, history_p = fake_paths
    navs = _noisy_walk(100.0, drift=-0.003, noise=0.001, n=21, seed=13)   # paper would be RED
    _write_state(state_p, navs)
    # wf data >21 days old → heartbeat should have fired, don't double-alert
    _write_history(history_p, sharpe=3.0, date="2026-04-01")
    status = dd.check(now=datetime(2026, 5, 23))
    assert status["healthy"] is True
    assert status["level"] == "OK"
    assert "old" in status["msg"]
    assert "heartbeat" in status["msg"]


# ── walk_forward missing entirely ─────────────────────────────

def test_check_history_missing_skips(fake_paths):
    state_p, history_p = fake_paths
    navs = _noisy_walk(100.0, drift=-0.003, noise=0.001, n=21, seed=13)
    _write_state(state_p, navs)
    # don't write history at all
    status = dd.check(now=datetime(2026, 5, 23))
    assert status["healthy"] is True
    assert "missing or empty" in status["msg"]


def test_check_history_no_parseable_sharpe(fake_paths):
    state_p, history_p = fake_paths
    navs = _noisy_walk(100.0, drift=-0.003, noise=0.001, n=21, seed=13)
    _write_state(state_p, navs)
    # Garbage Sharpe
    history_p.write_text(json.dumps([{
        "date": "2026-05-22",
        "bt_metrics": {"sharpe_ratio": "garbage"},
    }]))
    status = dd.check(now=datetime(2026, 5, 23))
    assert status["healthy"] is True
    assert "no parseable" in status["msg"]


# ───────────────────────────────────────────────────────────────────────
# format_for_feishu
# ───────────────────────────────────────────────────────────────────────

def test_format_healthy_empty():
    assert dd.format_for_feishu({"healthy": True, "level": "OK", "msg": "ok"}) == ""


def test_format_red_alert_contains_diagnostics():
    status = {
        "healthy": False, "level": "RED",
        "msg": "test red",
        "paper_sharpe": -0.5, "wf_sharpe": 2.0, "delta": -2.5,
        "nav_n": 21, "wf_date": "2026-05-22",
    }
    msg = dd.format_for_feishu(status)
    assert "🚨" in msg
    assert "RED ALERT" in msg
    assert "paper_trade rolling Sharpe" in msg
    assert "walk_forward Sharpe" in msg
    assert "Δ" in msg
    assert "Diagnostics:" in msg
    # Number formatting
    assert "-0.500" in msg
    assert "+2.000" in msg or "2.000" in msg   # accept either sign-format


def test_format_yellow_uses_warning_emoji():
    status = {
        "healthy": False, "level": "YELLOW",
        "msg": "warn",
        "paper_sharpe": 1.0, "wf_sharpe": 2.0, "delta": -1.0,
        "nav_n": 16, "wf_date": "2026-05-22",
    }
    msg = dd.format_for_feishu(status)
    assert "⚠" in msg
    assert "YELLOW ALERT" in msg


# ───────────────────────────────────────────────────────────────────────
# main() Feishu wiring
# ───────────────────────────────────────────────────────────────────────

def test_main_dry_run_red_does_not_call_feishu(fake_paths, capsys, monkeypatch):
    state_p, history_p = fake_paths
    navs = _noisy_walk(100.0, drift=-0.003, noise=0.001, n=21, seed=13)
    _write_state(state_p, navs)
    _write_history(history_p, sharpe=3.0, date=datetime.now().strftime("%Y-%m-%d"))

    def boom(*a, **kw):
        raise AssertionError("send_to_feishu must NOT be called in --dry-run")
    monkeypatch.setattr("scripts.daily_report.send_to_feishu", boom)

    rc = dd.main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "RED" in out


def test_main_healthy_no_feishu(fake_paths, capsys, monkeypatch):
    state_p, history_p = fake_paths
    _write_state(state_p, [100.0] * 5)   # cold start
    _write_history(history_p, sharpe=1.0)

    def boom(*a, **kw):
        raise AssertionError("send_to_feishu must NOT be called on healthy")
    monkeypatch.setattr("scripts.daily_report.send_to_feishu", boom)

    rc = dd.main([])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_main_alert_calls_feishu(fake_paths, capsys, monkeypatch):
    state_p, history_p = fake_paths
    navs = _noisy_walk(100.0, drift=-0.003, noise=0.001, n=21, seed=13)
    _write_state(state_p, navs)
    _write_history(history_p, sharpe=3.0, date=datetime.now().strftime("%Y-%m-%d"))
    calls = []
    monkeypatch.setattr(
        "scripts.daily_report.send_to_feishu",
        lambda md, **kw: calls.append(md) or True,
    )
    rc = dd.main([])
    assert rc == 0
    assert len(calls) == 1
    assert "RED ALERT" in calls[0]
