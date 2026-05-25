"""P4-1C tests for mp.monitor.threshold_alert.

P8-α-1 update (docs/dialog/ round 54): operator re-anchored
thresholds to deterministic baseline 1.20.  Test scenarios updated
accordingly. Lock-in test ``test_thresholds_anchored_to_120``
catches future drift away from operator's chosen scale relation.

Inject mock breach scenarios + assert alert payload. No live Feishu
calls — format_for_feishu produces the markdown block that
walk_forward_backtest.py::send_model_update_report appends.
"""
from __future__ import annotations

from mp.monitor.threshold_alert import (
    YELLOW,
    RED,
    check_thresholds,
    format_for_feishu,
)


def test_healthy_returns_no_alerts():
    """Production-like deterministic numbers (Sharpe 1.20, annual 38.74%,
    DD -25%) under P8-α-1 thresholds → 0 alerts.

    Note: actual deterministic backtest Max DD is -32.74% which would trip
    the new YELLOW -30% — this test uses -25% to stay healthy across all
    three indicators.  See P8-α-1 docstring "Heads-up" for that calibration
    edge case.
    """
    bt = {"sharpe_ratio": 1.20, "annual_return": 0.3874, "max_drawdown": -0.25}
    alerts = check_thresholds(bt)
    assert alerts == [], f"expected no alerts on healthy metrics, got {alerts}"


def test_sharpe_yellow_breach():
    """Sharpe 0.80 < YELLOW 0.90 but ≥ RED 0.50 → YELLOW only."""
    bt = {"sharpe_ratio": 0.80, "annual_return": 0.50, "max_drawdown": -0.25}
    alerts = check_thresholds(bt)
    assert len(alerts) == 1, f"expected 1 alert, got {alerts}"
    a = alerts[0]
    assert a["level"] == "YELLOW"
    assert a["indicator"] == "Sharpe"
    assert a["actual"] == 0.80
    assert a["threshold"] == YELLOW["sharpe_ratio"]


def test_sharpe_red_breach():
    """Sharpe 0.40 < RED 0.50 → RED."""
    bt = {"sharpe_ratio": 0.40, "annual_return": 0.50, "max_drawdown": -0.25}
    alerts = check_thresholds(bt)
    assert len(alerts) == 1
    assert alerts[0]["level"] == "RED"
    assert alerts[0]["indicator"] == "Sharpe"


def test_multiple_simultaneous_breaches():
    """Sharpe RED + annual RED + DD YELLOW under new thresholds.

    sharpe 0.4 < RED 0.5 → RED
    annual 10% < RED 15% → RED
    max_drawdown -0.35 (i.e., -35%): -35 < YELLOW -30 → YELLOW,
                                       -35 NOT < RED -40 → not RED
    """
    bt = {"sharpe_ratio": 0.4, "annual_return": 0.10, "max_drawdown": -0.35}
    alerts = check_thresholds(bt)
    levels = sorted([(a["indicator"], a["level"]) for a in alerts])
    assert levels == [
        ("Sharpe", "RED"),
        ("annual_return", "RED"),
        ("max_drawdown", "YELLOW"),
    ]


def test_percent_input_accepted():
    """bt_metrics may pass annual / max_drawdown as percent (60.0) rather than fraction (0.6)."""
    bt_frac = {"sharpe_ratio": 0.4, "annual_return": 0.10, "max_drawdown": -0.35}
    bt_pct = {"sharpe_ratio": 0.4, "annual_return": 10.0, "max_drawdown": -35.0}
    assert check_thresholds(bt_frac) == check_thresholds(bt_pct)


def test_missing_metric_is_silent():
    """Missing metric should not crash + no alert for that indicator.

    sharpe 0.4 < RED 0.5 → 1 Sharpe RED alert; other indicators absent.
    """
    bt = {"sharpe_ratio": 0.4}
    alerts = check_thresholds(bt)
    assert len(alerts) == 1
    assert alerts[0]["indicator"] == "Sharpe"


def test_format_for_feishu_empty_returns_empty():
    assert format_for_feishu([]) == ""


def test_format_for_feishu_contains_breach_msg():
    """RED + YELLOW present → both section headers render."""
    bt = {"sharpe_ratio": 0.4, "annual_return": 0.10, "max_drawdown": -0.35}
    alerts = check_thresholds(bt)
    text = format_for_feishu(alerts)
    # Red section header (Sharpe + annual are RED)
    assert "🚨 RED ALERT" in text
    # Yellow section header (max_drawdown is YELLOW)
    assert "⚠ YELLOW ALERT" in text
    # Source-of-truth footnote
    assert "mp/monitor/threshold_alert.py" in text
    # Each alert message present
    for a in alerts:
        assert a["msg"] in text


def test_red_overrides_yellow_for_same_indicator():
    """If actual passes RED threshold, only RED alert fires (not both).

    sharpe 0.4 < RED 0.5 → RED only (and not also YELLOW for the same indicator).
    """
    bt = {"sharpe_ratio": 0.4}
    alerts = check_thresholds(bt)
    sharpe_alerts = [a for a in alerts if a["indicator"] == "Sharpe"]
    assert len(sharpe_alerts) == 1
    assert sharpe_alerts[0]["level"] == "RED"


# ─────────────────────────────────────────────────────────────────
# P8-α-1 anchor lock-in (docs/dialog/ round 54)
# ─────────────────────────────────────────────────────────────────

def test_thresholds_anchored_to_120():
    """Lock-in: thresholds preserve operator-chosen scale ratio to the
    deterministic baseline 1.20 (P8-α-1).

    If anyone later edits YELLOW / RED Sharpe away from the 75% / 42%
    relation without re-running the operator anchoring exercise, this
    test fails and flags the drift to the reviewer.
    """
    DETERMINISTIC_BASELINE = 1.20

    yellow_ratio = YELLOW["sharpe_ratio"] / DETERMINISTIC_BASELINE
    red_ratio = RED["sharpe_ratio"] / DETERMINISTIC_BASELINE

    # YELLOW ≈ 75% of baseline (0.90 / 1.20)
    assert abs(yellow_ratio - 0.75) < 0.05, (
        f"YELLOW Sharpe / 1.20 = {yellow_ratio:.3f}, expected ~0.75 "
        f"(P8-α-1 operator anchor — see docs/dialog/ round 54)"
    )
    # RED ≈ 42% of baseline (0.50 / 1.20)
    assert abs(red_ratio - 0.42) < 0.05, (
        f"RED Sharpe / 1.20 = {red_ratio:.3f}, expected ~0.42 "
        f"(P8-α-1 operator anchor — see docs/dialog/ round 54)"
    )

    # MaxDD lock — operator tightened from -42/-50 to -30/-40 to model
    # live slippage / overnight gap risk > backtest sim.
    assert YELLOW["max_drawdown_pct"] == -30.0, (
        f"YELLOW Max DD = {YELLOW['max_drawdown_pct']}, expected -30.0 "
        f"(P8-α-1 operator anchor)"
    )
    assert RED["max_drawdown_pct"] == -40.0, (
        f"RED Max DD = {RED['max_drawdown_pct']}, expected -40.0 "
        f"(P8-α-1 operator anchor)"
    )
