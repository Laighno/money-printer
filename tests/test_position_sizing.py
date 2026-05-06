"""Position-sizing regression tests.

Locks in the 2026-04-29 decision to use `conviction` sizing as production
default (BASELINE §一).  Tests guard against:

  1. The sizing helper correctly handles all 4 schemes (+oracle).
  2. paper_trade.py uses conviction (not equal-weight) when planning buys.
  3. The walk_forward default is `conviction`, not `equal`.
"""

from __future__ import annotations

import inspect
import re

import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────
# 1. _compute_position_weights helper
# ──────────────────────────────────────────────────────────────────────

def test_equal_weights():
    """1/N for each code."""
    from scripts.walk_forward_backtest import _compute_position_weights
    panel_by_date = {pd.Timestamp("2026-04-29"): pd.DataFrame()}
    w = _compute_position_weights(
        ["A", "B", "C", "D", "E"], pd.Timestamp("2026-04-29"),
        panel_by_date, sizing="equal",
    )
    assert len(w) == 5
    assert all(abs(v - 0.2) < 1e-9 for v in w.values())
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_conviction_weights_higher_for_higher_excess():
    """Stocks with higher predicted excess get bigger weights."""
    from scripts.walk_forward_backtest import _compute_position_weights
    panel_by_date = {pd.Timestamp("2026-04-29"): pd.DataFrame()}
    raw_scores = {
        "TOP1": 0.10,    # +10% predicted excess
        "MID":  0.03,    # +3%
        "LOW":  0.005,   # +0.5%
        "NEG":  -0.02,   # negative
    }
    w = _compute_position_weights(
        ["TOP1", "MID", "LOW", "NEG"], pd.Timestamp("2026-04-29"),
        panel_by_date, sizing="conviction", raw_scores=raw_scores,
    )
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # Higher excess = bigger weight
    assert w["TOP1"] > w["MID"] > w["LOW"] > w["NEG"]
    # Top1 should dominate
    assert w["TOP1"] > 0.4


def test_conviction_oracle_uses_realized_returns():
    """conviction_oracle takes the same path as conviction (helper-level).
    Distinction lives in the caller (walk_forward) which feeds realized fwd_ret."""
    from scripts.walk_forward_backtest import _compute_position_weights
    raw_scores = {"A": 0.08, "B": 0.02}
    w = _compute_position_weights(
        ["A", "B"], pd.Timestamp("2026-04-29"),
        {pd.Timestamp("2026-04-29"): pd.DataFrame()},
        sizing="conviction_oracle", raw_scores=raw_scores,
    )
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["A"] > w["B"]


def test_inverse_vol_weights_lower_for_higher_vol():
    """Lower-vol stocks get bigger weights."""
    from scripts.walk_forward_backtest import _compute_position_weights
    panel_data = pd.DataFrame({
        "code": ["A", "B", "C"],
        "volatility_20d": [0.20, 0.40, 0.10],   # C lowest vol → highest weight
    })
    panel_by_date = {pd.Timestamp("2026-04-29"): panel_data}
    w = _compute_position_weights(
        ["A", "B", "C"], pd.Timestamp("2026-04-29"),
        panel_by_date, sizing="inverse_vol",
    )
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # C (vol 0.10) > A (vol 0.20) > B (vol 0.40)
    assert w["C"] > w["A"] > w["B"]


def test_inverse_vol_falls_back_to_equal_on_missing_vol():
    """Missing volatility → fall back to equal-weight."""
    from scripts.walk_forward_backtest import _compute_position_weights
    panel_data = pd.DataFrame({
        "code": ["A", "B"],
        "volatility_20d": [0.20, None],   # B missing
    })
    panel_by_date = {pd.Timestamp("2026-04-29"): panel_data}
    w = _compute_position_weights(
        ["A", "B"], pd.Timestamp("2026-04-29"),
        panel_by_date, sizing="inverse_vol",
    )
    assert all(abs(v - 0.5) < 1e-9 for v in w.values())


def test_vol_target_caps_at_unit_leverage():
    """vol_target never exceeds 1.0 total exposure (no margin)."""
    from scripts.walk_forward_backtest import _compute_position_weights
    # Very low vol → leverage would want to >1, must cap at equal/N
    panel_data = pd.DataFrame({
        "code": ["A", "B", "C", "D", "E"],
        "volatility_20d": [0.10] * 5,
    })
    panel_by_date = {pd.Timestamp("2026-04-29"): panel_data}
    w = _compute_position_weights(
        ["A", "B", "C", "D", "E"], pd.Timestamp("2026-04-29"),
        panel_by_date, sizing="vol_target",
    )
    # Sum should be ≤ 1.0; at very low vol it caps at 1.0/N each
    s = sum(w.values())
    assert s <= 1.0 + 1e-9
    # All weights equal (vols identical → equal-weight under leverage cap)
    assert all(abs(v - w["A"]) < 1e-9 for v in w.values())


def test_vol_target_reduces_exposure_when_high_vol():
    """High portfolio vol → lower total exposure (some cash retained)."""
    from scripts.walk_forward_backtest import _compute_position_weights
    panel_data = pd.DataFrame({
        "code": ["A", "B", "C"],
        "volatility_20d": [0.80, 0.80, 0.80],   # very high → would target<full
    })
    panel_by_date = {pd.Timestamp("2026-04-29"): panel_data}
    w = _compute_position_weights(
        ["A", "B", "C"], pd.Timestamp("2026-04-29"),
        panel_by_date, sizing="vol_target",
    )
    s = sum(w.values())
    assert s < 1.0   # at vol 80%, target 25% → leverage ~0.31
    assert s > 0


# ──────────────────────────────────────────────────────────────────────
# 2. paper_trade integration
# ──────────────────────────────────────────────────────────────────────

def test_paper_trade_uses_conviction_sizing():
    """paper_trade.plan_tomorrow_trades must apply conviction weighting,
    not equal-weight, per BASELINE 2026-04-29."""
    from scripts import paper_trade
    src = inspect.getsource(paper_trade.plan_tomorrow_trades)
    # Must reference raw_excess for conviction
    assert "raw_excess" in src, (
        "paper_trade.plan_tomorrow_trades must use raw_excess (model's predicted "
        "excess) for conviction weighting per BASELINE 2026-04-29."
    )
    # Must NOT just divide by TOP_K (the old equal-weight path is gone)
    assert "broker.total_value / TOP_K" not in src, (
        "paper_trade must not use equal-weight (target=total/TOP_K).  "
        "Conviction sizing is the new BASELINE default."
    )


# ──────────────────────────────────────────────────────────────────────
# 3. walk_forward default
# ──────────────────────────────────────────────────────────────────────

def test_walk_forward_position_sizing_default_is_conviction():
    """walk_forward_backtest.py default POSITION_SIZING must be 'conviction'."""
    from pathlib import Path
    src = Path("scripts/walk_forward_backtest.py").read_text()
    # Look for the os.environ.get default
    m = re.search(r'POSITION_SIZING\s*=\s*os\.environ\.get\(\s*"POSITION_SIZING"\s*,\s*"([^"]+)"', src)
    assert m is not None, "Could not find POSITION_SIZING default in walk_forward"
    default = m.group(1)
    assert default == "conviction", (
        f"walk_forward POSITION_SIZING default should be 'conviction' per "
        f"BASELINE 2026-04-29, got '{default}'."
    )
