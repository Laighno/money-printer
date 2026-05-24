"""P4-1C tests for mp.monitor.threshold_alert.

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
    """Production-like numbers (Sharpe 1.90, annual 60.42%, DD -36.30%) → 0 alerts."""
    bt = {"sharpe_ratio": 1.90, "annual_return": 0.6042, "max_drawdown": -0.363}
    alerts = check_thresholds(bt)
    assert alerts == [], f"expected no alerts on healthy metrics, got {alerts}"


def test_sharpe_yellow_breach():
    bt = {"sharpe_ratio": 1.20, "annual_return": 0.50, "max_drawdown": -0.30}
    alerts = check_thresholds(bt)
    assert len(alerts) == 1
    a = alerts[0]
    assert a["level"] == "YELLOW"
    assert a["indicator"] == "Sharpe"
    assert a["actual"] == 1.20
    assert a["threshold"] == YELLOW["sharpe_ratio"]


def test_sharpe_red_breach():
    bt = {"sharpe_ratio": 0.80, "annual_return": 0.50, "max_drawdown": -0.30}
    alerts = check_thresholds(bt)
    assert len(alerts) == 1
    assert alerts[0]["level"] == "RED"
    assert alerts[0]["indicator"] == "Sharpe"


def test_multiple_simultaneous_breaches():
    """Sharpe RED + annual RED + DD YELLOW."""
    bt = {"sharpe_ratio": 0.5, "annual_return": 0.10, "max_drawdown": -0.45}
    alerts = check_thresholds(bt)
    levels = sorted([(a["indicator"], a["level"]) for a in alerts])
    assert levels == [
        ("Sharpe", "RED"),
        ("annual_return", "RED"),
        ("max_drawdown", "YELLOW"),
    ]


def test_percent_input_accepted():
    """bt_metrics may pass annual / max_drawdown as percent (60.0) rather than fraction (0.6)."""
    bt_frac = {"sharpe_ratio": 0.5, "annual_return": 0.10, "max_drawdown": -0.45}
    bt_pct = {"sharpe_ratio": 0.5, "annual_return": 10.0, "max_drawdown": -45.0}
    assert check_thresholds(bt_frac) == check_thresholds(bt_pct)


def test_missing_metric_is_silent():
    """Missing metric should not crash + no alert for that indicator."""
    bt = {"sharpe_ratio": 0.5}  # missing annual + dd
    alerts = check_thresholds(bt)
    assert len(alerts) == 1
    assert alerts[0]["indicator"] == "Sharpe"


def test_format_for_feishu_empty_returns_empty():
    assert format_for_feishu([]) == ""


def test_format_for_feishu_contains_breach_msg():
    bt = {"sharpe_ratio": 0.8, "annual_return": 0.10, "max_drawdown": -0.45}
    alerts = check_thresholds(bt)
    text = format_for_feishu(alerts)
    # Red section header
    assert "🚨 RED ALERT" in text
    # Yellow section header (because max_drawdown is yellow at -45%)
    assert "⚠ YELLOW ALERT" in text
    # Source-of-truth footnote
    assert "mp/monitor/threshold_alert.py" in text
    # Each alert message present
    for a in alerts:
        assert a["msg"] in text


def test_red_overrides_yellow_for_same_indicator():
    """If actual passes RED threshold, only RED alert fires (not both)."""
    bt = {"sharpe_ratio": 0.5}  # below RED (0.9), so definitely below YELLOW (1.4)
    alerts = check_thresholds(bt)
    sharpe_alerts = [a for a in alerts if a["indicator"] == "Sharpe"]
    assert len(sharpe_alerts) == 1
    assert sharpe_alerts[0]["level"] == "RED"
