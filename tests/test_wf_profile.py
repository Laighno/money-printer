"""WF_PROFILE (prod | research) resolution + pure production-rule helpers.

Only imports mp.backtest.wf_profile (no side effects); never imports
scripts/walk_forward_backtest (which chdir()s and reads caches on import).
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from mp.backtest import wf_profile


KNOBS = ("WF_PROFILE", *wf_profile.knob_names())


def _clean_env(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)


def _reload(monkeypatch, **env):
    _clean_env(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(wf_profile)


# ── profile defaults ────────────────────────────────────────────────────

def test_default_profile_is_prod_with_production_knobs(monkeypatch):
    m = _reload(monkeypatch)
    assert m.PROFILE == "prod"
    p = m.CURRENT
    assert p["TOP_K"] == 22                 # daily_report n_recommend=22 (→ ~25 holds)
    assert p["LIVE_UNIVERSE"] is True
    assert p["EXCL_CHINEXT"] is True
    assert p["EXCL_HS300"] is False         # HS300 included since 2026-07-17
    assert p["EXCL_ILLIQ"] is True
    assert p["LIVE_ILLIQ_AMOUNT"] == 1e8    # LOW_LIQUIDITY_FILTER_AMOUNT
    assert p["GROSS_EXPOSURE"] == 0.85      # portfolio.yaml target_position_pct
    assert p["LIMIT_LOCK"] is True
    assert p["POSITION_SIZING"] == "conviction"
    assert p["HARD_CAP"] == 0.40            # hard_single_weight default
    assert p["HOLD_RANK_BAND"] == 30
    assert p["CLEAR_RANK_BAND"] == 100
    assert p["REBALANCE_TOLERANCE"] == 0.02
    assert p["ORDER_PASS"] == "prod"
    assert p["COST_AWARE_REBALANCE"] is False
    assert p["RETRAIN_FREQ"] == "weekly"
    assert p["explicit"] == set()


def test_research_profile_restores_legacy_defaults(monkeypatch):
    m = _reload(monkeypatch, WF_PROFILE="research")
    p = m.CURRENT
    assert m.PROFILE == "research"
    assert p["TOP_K"] == 10
    assert p["LIVE_UNIVERSE"] is False
    assert p["EXCL_CHINEXT"] is True
    assert p["EXCL_HS300"] is True
    assert p["EXCL_ILLIQ"] is True
    assert p["GROSS_EXPOSURE"] == 1.0
    assert p["LIMIT_LOCK"] is True
    assert p["POSITION_SIZING"] == "conviction"
    assert p["HARD_CAP"] == 0.0
    assert p["HOLD_RANK_BAND"] == 0
    assert p["REBALANCE_TOLERANCE"] == 0.0
    assert p["ORDER_PASS"] == "legacy"
    assert p["COST_AWARE_REBALANCE"] is True
    assert p["RETRAIN_FREQ"] == "monthly"


def test_explicit_env_overrides_profile_default(monkeypatch):
    m = _reload(monkeypatch, TOP_K="25", EXCL_HS300="1", GROSS_EXPOSURE="1.0",
                RETRAIN_FREQ="monthly", HOLD_RANK_BAND="0")
    p = m.CURRENT
    assert p["profile"] == "prod"
    assert p["TOP_K"] == 25
    assert p["EXCL_HS300"] is True
    assert p["GROSS_EXPOSURE"] == 1.0
    assert p["RETRAIN_FREQ"] == "monthly"
    assert p["HOLD_RANK_BAND"] == 0
    assert p["explicit"] == {"TOP_K", "EXCL_HS300", "GROSS_EXPOSURE", "RETRAIN_FREQ", "HOLD_RANK_BAND"}
    # untouched knobs keep the prod default
    assert p["HARD_CAP"] == 0.40
    assert "env overrides" in m.profile_line(p)


def test_explicit_env_overrides_research_too(monkeypatch):
    m = _reload(monkeypatch, WF_PROFILE="research", TOP_K="22", LIVE_UNIVERSE="1")
    assert m.CURRENT["TOP_K"] == 22
    assert m.CURRENT["LIVE_UNIVERSE"] is True
    assert m.CURRENT["HOLD_RANK_BAND"] == 0   # not overridden → research default


def test_bad_profile_and_bad_freq_raise(monkeypatch):
    with pytest.raises(ValueError):
        wf_profile.resolve({"WF_PROFILE": "live"})
    with pytest.raises(ValueError):
        wf_profile.resolve({"RETRAIN_FREQ": "daily"})
    with pytest.raises(ValueError):
        wf_profile.resolve({"ORDER_PASS": "fast"})


def test_profile_line_mentions_key_knobs():
    line = wf_profile.profile_line(wf_profile.resolve({}))
    assert line.startswith("prod (22 recs")
    for frag in ("cap 40%", "exposure 85%", "hold band 30/clear 100", "weekly retrain",
                 "verify gate", "离线不可复现"):
        assert frag in line
    assert wf_profile.profile_line(wf_profile.resolve({"WF_PROFILE": "research"})).startswith("research (")


def test_unaligned_items_listed():
    assert len(wf_profile.UNALIGNED_ITEMS) >= 3
    assert any("verify gate" in s for s in wf_profile.UNALIGNED_ITEMS)
    assert any("close×1.01" in s for s in wf_profile.UNALIGNED_ITEMS)


# ── hold band (daily_report Pass 2) ─────────────────────────────────────

def test_hold_band_held_rank_28_is_kept():
    assert wf_profile.hold_band_action(28, 30, 100) == "hold"
    assert wf_profile.holding_decision(True, 28, 22, 30, 100) == "hold"


def test_hold_band_held_rank_31_is_sold_half():
    assert wf_profile.hold_band_action(31, 30, 100) == "half"
    assert wf_profile.holding_decision(True, 31, 22, 30, 100) == "half"


def test_hold_band_held_rank_101_is_cleared():
    assert wf_profile.hold_band_action(101, 30, 100) == "sell"
    assert wf_profile.hold_band_action(None, 30, 100) == "sell"


def test_hold_band_non_held_rank_28_is_not_bought():
    assert wf_profile.holding_decision(False, 28, 22, 30, 100) == "none"
    # inside top-K it is a sized target whether held or not
    assert wf_profile.holding_decision(False, 22, 22, 30, 100) == "target"
    assert wf_profile.holding_decision(True, 5, 22, 30, 100) == "target"


def test_hold_band_disabled_sells_on_dropout():
    assert wf_profile.hold_band_action(28, 0, 0) == "sell"
    assert wf_profile.hold_band_action(11, 0, 100) == "sell"


def test_half_lot_shares_mirrors_prod():
    assert wf_profile.half_lot_shares(1100) == 500
    assert wf_profile.half_lot_shares(200) == 100
    assert wf_profile.half_lot_shares(100) == 0     # < 1 lot → skip


# ── retrain schedule ────────────────────────────────────────────────────

def test_weekly_retrain_fires_once_per_trading_week():
    # 2025-06-02 (Mon) .. 2025-06-27 (Fri): 4 ISO weeks of business days
    days = pd.date_range("2025-06-02", "2025-06-27", freq="B")
    rd = wf_profile.retrain_dates(days, "weekly")
    assert rd == [pd.Timestamp(d) for d in ("2025-06-02", "2025-06-09", "2025-06-16", "2025-06-23")]


def test_weekly_retrain_uses_first_trading_day_when_monday_is_holiday():
    days = [d for d in pd.date_range("2025-06-02", "2025-06-13", freq="B")
            if d != pd.Timestamp("2025-06-09")]   # Monday holiday
    rd = wf_profile.retrain_dates(days, "weekly")
    assert rd == [pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-10")]


def test_monthly_retrain_fires_once_per_month():
    days = pd.date_range("2025-06-02", "2025-08-29", freq="B")
    rd = wf_profile.retrain_dates(days, "monthly")
    assert rd == [pd.Timestamp("2025-06-02"), pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01")]


def test_retrain_dates_dedup_and_empty():
    assert wf_profile.retrain_dates([], "weekly") == []
    dup = ["2025-06-02", "2025-06-02", "2025-06-03"]
    assert wf_profile.retrain_dates(dup, "weekly") == [pd.Timestamp("2025-06-02")]


# ── budget / total-position reconciliation ──────────────────────────────

def test_scale_buys_no_scaling_when_within_caps():
    out = wf_profile.scale_buys_to_caps({"a": 100.0, "b": 300.0}, 900.0, 1000.0, 1000.0)
    assert out == {"a": 100.0, "b": 300.0}


def test_scale_buys_cash_constraint_is_proportional_with_5pct_buffer():
    out = wf_profile.scale_buys_to_caps({"a": 100.0, "b": 300.0}, 400.0, 10_000.0, 200.0)
    assert out["a"] == pytest.approx(190.0 * 0.25)
    assert out["b"] == pytest.approx(190.0 * 0.75)


def test_scale_buys_total_position_cap():
    # positions 800 + buys 400 = 1200 > cap 1000 → buys shrink by excess 200
    out = wf_profile.scale_buys_to_caps({"a": 100.0, "b": 300.0}, 1200.0, 1000.0, 10_000.0)
    assert sum(out.values()) == pytest.approx(200.0)
    assert out["a"] == pytest.approx(50.0)


def test_scale_buys_drops_all_when_cap_already_breached():
    out = wf_profile.scale_buys_to_caps({"a": 100.0}, 1300.0, 1000.0, 10_000.0)
    assert out == {}
    assert wf_profile.scale_buys_to_caps({}, 0.0, 1.0, 1.0) == {}
