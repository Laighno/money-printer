"""Paper-trade pipeline regression tests.

Locks in the口径 documented in BASELINE.md §一 for the autonomous
模拟交易 mode launched 2026-04-29.  Network-dependent paths
(get_daily_bars, akshare benchmark fetch, Feishu) are mocked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────
# State IO round-trip
# ──────────────────────────────────────────────────────────────────────

def test_state_round_trip(tmp_path):
    """Saving and reloading state must preserve cash, positions, and trade_log."""
    from scripts import paper_trade as pt

    state = pt._new_state("2026-04-29")
    state["cash"] = 280_000.0
    state["positions"] = [
        {"code": "600166", "shares": 1500, "avg_cost": 18.20,
         "current_price": 19.10, "peak_price": 19.50, "entry_date": "2026-04-29"},
    ]
    state["trade_log"] = [{"code": "600166", "action": "BUY_OPEN", "shares": 1500}]

    with patch.object(pt, "STATE_PATH", tmp_path / "state.json"):
        pt.save_state(state)
        reloaded = pt.load_state("2026-04-30")

    assert reloaded["cash"] == 280_000.0
    assert len(reloaded["positions"]) == 1
    assert reloaded["positions"][0]["code"] == "600166"
    assert reloaded["positions"][0]["shares"] == 1500
    assert len(reloaded["trade_log"]) == 1


def test_initial_state_has_30万_cash():
    """First run with no prior state file should start at 300,000 cash."""
    from scripts import paper_trade as pt
    s = pt._new_state("2026-04-29")
    assert s["cash"] == 300_000
    assert s["initial_capital"] == 300_000
    assert s["positions"] == []
    assert s["pending_trades"] == []


# ──────────────────────────────────────────────────────────────────────
# Broker reconstruction
# ──────────────────────────────────────────────────────────────────────

def test_broker_reconstruction_preserves_positions():
    """state_to_broker must rebuild a broker that mirrors the persisted state."""
    from scripts import paper_trade as pt

    state = pt._new_state("2026-04-29")
    state["cash"] = 240_000.0
    state["positions"] = [
        {"code": "600166", "shares": 1500, "avg_cost": 18.20,
         "current_price": 19.10, "peak_price": 19.50, "entry_date": "2026-04-29"},
        {"code": "688615", "shares": 800, "avg_cost": 35.0,
         "current_price": 36.5, "peak_price": 37.0, "entry_date": "2026-04-29"},
    ]
    broker = pt.state_to_broker(state)
    assert broker.cash == 240_000.0
    assert len(broker.positions) == 2
    assert broker.positions["600166"].shares == 1500
    assert broker.positions["688615"].avg_cost == 35.0


def test_broker_to_state_round_trip():
    """broker_to_state, after BUY then SELL, should reflect cash and emptied positions."""
    from scripts import paper_trade as pt
    state = pt._new_state("2026-04-29")
    broker = pt.state_to_broker(state)
    broker.buy(code="600166", price=18.0, target_value=30_000, date="2026-04-29")
    pt.broker_to_state(state, broker)
    assert any(p["code"] == "600166" for p in state["positions"])
    assert state["cash"] < 300_000

    broker.sell(code="600166", price=19.0, date="2026-04-30")
    pt.broker_to_state(state, broker)
    assert all(p["code"] != "600166" for p in state["positions"])


# ──────────────────────────────────────────────────────────────────────
# Decision: cost-aware swap respects BASELINE thresholds
# ──────────────────────────────────────────────────────────────────────

def test_plan_first_day_buys_full_top_k_when_no_positions():
    """Day 1: empty broker → 10 buys, conviction-weighted (BASELINE 2026-04-29).

    Targets sum to total NAV (~300k); higher predicted excess → bigger position.
    """
    from scripts import paper_trade as pt

    state = pt._new_state("2026-04-29")
    broker = pt.state_to_broker(state)

    # Construct 20 stocks with descending scores (top 10 will be picked).
    # raw_excess decreasing → top1 gets biggest weight.
    rows = []
    for i in range(20):
        rows.append({"code": f"60{i:04d}", "ml_score": 0.9 - i * 0.02,
                     "raw_excess": 0.05 - i * 0.005, "rank_pct": 1 - i * 0.05})
    scored = pd.DataFrame(rows)

    pending, selected = pt.plan_tomorrow_trades(broker, scored,
                                                pd.Timestamp("2026-04-29"), dq=1.0)
    assert len(selected) == 10
    assert all(t["action"] == "buy" for t in pending)
    assert len(pending) == 10

    # Targets sum to ~total NAV (300k for fresh broker)
    targets = [t["target_value"] for t in pending]
    assert abs(sum(targets) - 300_000) < 1, \
        f"Sum of targets must equal NAV 300k, got {sum(targets)}"

    # Conviction: higher raw_excess (lower-index stocks in our setup) → bigger target
    # Build {code: target} for ordering check
    code_to_target = {t["code"]: t["target_value"] for t in pending}
    code_to_excess = dict(zip(scored["code"], scored["raw_excess"]))
    # Sort by excess desc and verify targets follow
    by_excess = sorted(code_to_target.keys(), key=lambda c: -code_to_excess[c])
    targets_by_excess = [code_to_target[c] for c in by_excess]
    # First (highest excess) should be largest, last should be smallest
    assert targets_by_excess[0] >= targets_by_excess[-1], (
        f"Top-conviction stock should have largest target. Order: {targets_by_excess}"
    )

    # Each individual target must include 'weight' field (added in conviction path)
    assert all("weight" in t for t in pending), \
        "Conviction-sized buys must include 'weight' field for transparency"


def test_plan_freezes_on_low_data_quality():
    """data_quality < 0.5 must produce zero trades (freeze)."""
    from scripts import paper_trade as pt

    state = pt._new_state("2026-04-29")
    broker = pt.state_to_broker(state)
    scored = pd.DataFrame({
        "code": [f"60{i:04d}" for i in range(20)],
        "ml_score": [0.9 - i * 0.02 for i in range(20)],
        "raw_excess": [0.05 - i * 0.005 for i in range(20)],
        "rank_pct": [1 - i * 0.05 for i in range(20)],
    })
    pending, selected = pt.plan_tomorrow_trades(
        broker, scored, pd.Timestamp("2026-04-29"), dq=0.3,
    )
    assert pending == []
    assert selected == []


def test_plan_no_trades_when_holdings_match_top_k():
    """If currently holding the same Top-K stocks, no swaps."""
    from scripts import paper_trade as pt

    state = pt._new_state("2026-04-29")
    state["cash"] = 0
    state["positions"] = [
        {"code": f"60{i:04d}", "shares": 1500, "avg_cost": 18.0,
         "current_price": 18.0, "peak_price": 18.0, "entry_date": "2026-04-29"}
        for i in range(10)
    ]
    broker = pt.state_to_broker(state)

    rows = []
    for i in range(20):
        rows.append({"code": f"60{i:04d}", "ml_score": 0.9 - i * 0.02,
                     "raw_excess": 0.05 - i * 0.005, "rank_pct": 1 - i * 0.05})
    scored = pd.DataFrame(rows)

    pending, selected = pt.plan_tomorrow_trades(broker, scored,
                                                pd.Timestamp("2026-04-29"), dq=1.0)
    assert len(selected) == 10
    assert pending == [], "Should be no pending trades when holdings already = Top-K"


def test_plan_swap_when_holding_drops_out():
    """Holding a stock that's no longer in Top-K AND a clear winner exists →
    sell the dropout, buy the new entrant."""
    from scripts import paper_trade as pt

    state = pt._new_state("2026-04-29")
    state["cash"] = 0
    # Hold stocks 0..9, but top scores are now 0..8 + a new winner code "999999"
    state["positions"] = [
        {"code": f"60{i:04d}", "shares": 1500, "avg_cost": 18.0,
         "current_price": 18.0, "peak_price": 18.0, "entry_date": "2026-04-29"}
        for i in range(10)
    ]
    broker = pt.state_to_broker(state)

    rows = []
    rows.append({"code": "999999", "ml_score": 0.95,
                 "raw_excess": 0.10, "rank_pct": 1.0})  # new winner
    for i in range(20):
        # Stock 9 (which we hold) drops to score 0.50 — far behind top 10
        score = 0.9 - i * 0.05 if i < 9 else (0.50 - (i - 9) * 0.02)
        rows.append({"code": f"60{i:04d}", "ml_score": score,
                     "raw_excess": 0.05 - i * 0.005, "rank_pct": 1 - i * 0.05})
    scored = pd.DataFrame(rows)

    pending, selected = pt.plan_tomorrow_trades(broker, scored,
                                                pd.Timestamp("2026-04-29"), dq=1.0)
    sells = [t for t in pending if t["action"] == "sell"]
    buys = [t for t in pending if t["action"] == "buy"]
    sell_codes = {t["code"] for t in sells}
    buy_codes = {t["code"] for t in buys}
    assert "999999" in buy_codes, "New winner must be in buy list"
    assert "600009" in sell_codes, "Dropout (600009 → score 0.50) must be in sell list"


# ──────────────────────────────────────────────────────────────────────
# NAV history maintenance
# ──────────────────────────────────────────────────────────────────────

def test_append_nav_overwrites_same_day():
    """Running paper_trade twice on the same day shouldn't double-count NAV history."""
    from scripts import paper_trade as pt

    state = pt._new_state("2026-04-29")
    broker = pt.state_to_broker(state)
    today = pd.Timestamp("2026-04-29")

    pt.append_nav(state, today, broker)
    pt.append_nav(state, today, broker)
    assert len(state["nav_history"]) == 1

    # Different day — should append
    pt.append_nav(state, pd.Timestamp("2026-04-30"), broker)
    assert len(state["nav_history"]) == 2


# ──────────────────────────────────────────────────────────────────────
# 不交易 ETF (BASELINE §一 要求)
# ──────────────────────────────────────────────────────────────────────

def test_universe_excludes_etfs():
    """ZZ500 constituents from get_index_constituents should not include ETFs.
    Light-weight contract test: paper_trade must not introduce its own ETF
    symbols.  We only check that paper_trade doesn't hard-code ETF codes."""
    src = (Path(__file__).parent.parent / "scripts" / "paper_trade.py").read_text()
    # Common ETF code prefixes/numbers that would indicate a leak
    forbidden_etf_codes = ["512660", "510300", "510500", "159915"]
    for code in forbidden_etf_codes:
        assert code not in src, (
            f"ETF code {code} appears in paper_trade.py — paper trading "
            "must restrict to ZZ500 individual stocks per BASELINE §一."
        )


# ──────────────────────────────────────────────────────────────────────
# Configuration sanity (BASELINE alignment)
# ──────────────────────────────────────────────────────────────────────

def test_config_matches_baseline():
    """Pin the paper-trade configuration to BASELINE.md §一 values."""
    from scripts import paper_trade as pt
    assert pt.INITIAL_CAPITAL == 300_000
    assert pt.TOP_K == 10
    assert pt.SLIPPAGE_BPS == 5
    assert pt.COMMISSION_BPS == 3
    assert pt.UNIVERSE_NAME == "zz500"
    assert pt.DATA_QUALITY_GATE == 0.5
