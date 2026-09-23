"""涨跌停 / 停牌 fill-blocking rules (2026-09-23 limit-lock).

Covers mp.account.broker.fill_blocked / limit_pct / round_cent / board_of and
the walk-forward helpers _build_bar_lookup / _trade_cost_stats.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mp.account.broker import (
    board_of,
    fill_blocked,
    limit_pct,
    round_cent,
)


def _bar(o, h, l, vol=1_000_000.0):
    return {"open": o, "high": h, "low": l, "volume": vol}


# ── 主板 10% ────────────────────────────────────────────────────────────

def test_one_word_limit_up_blocks_buy():
    # prev 10.00 → limit 11.00, 一字板 open=high=low=11.00
    assert fill_blocked("buy", _bar(11.00, 11.00, 11.00), 10.00) == "limit_up"


def test_open_at_limit_with_range_is_fillable_by_default():
    # 开盘涨停但盘中打开 (high > low) → 按可成交
    assert fill_blocked("buy", _bar(11.00, 11.00, 10.60), 10.00) is None


def test_open_at_limit_with_range_blocked_when_strict():
    assert fill_blocked("buy", _bar(11.00, 11.00, 10.60), 10.00, strict=True) == "limit_up"


def test_below_limit_is_fillable_even_when_one_word():
    # 一字但没到涨停 (10.50 vs limit 11.00) 不算封板
    assert fill_blocked("buy", _bar(10.50, 10.50, 10.50), 10.00) is None
    assert fill_blocked("buy", _bar(10.50, 10.50, 10.50), 10.00, strict=True) is None


def test_one_word_limit_down_blocks_sell():
    assert fill_blocked("sell", _bar(9.00, 9.00, 9.00), 10.00) == "limit_down"


def test_limit_down_does_not_block_buy_and_limit_up_does_not_block_sell():
    assert fill_blocked("buy", _bar(9.00, 9.00, 9.00), 10.00) is None
    assert fill_blocked("sell", _bar(11.00, 11.00, 11.00), 10.00) is None


# ── 停牌 ────────────────────────────────────────────────────────────────

def test_missing_bar_is_suspended_for_both_sides():
    assert fill_blocked("buy", None, 10.00) == "suspended"
    assert fill_blocked("sell", None, 10.00) == "suspended"


def test_zero_volume_is_suspended():
    assert fill_blocked("buy", _bar(10.5, 10.6, 10.4, vol=0), 10.00) == "suspended"
    assert fill_blocked("sell", _bar(10.5, 10.6, 10.4, vol=0.0), 10.00) == "suspended"


def test_missing_prev_close_only_applies_suspension_rule():
    assert fill_blocked("buy", _bar(11.00, 11.00, 11.00), None) is None
    assert fill_blocked("buy", None, None) == "suspended"


# ── 创业板 / 科创板 20% ───────────────────────────────────────────────────

def test_chinext_20pct_after_reform():
    dt = "2021-06-01"
    # +10% 一字 is NOT a limit on 创业板 after 2020-08-24
    assert fill_blocked("buy", _bar(11.00, 11.00, 11.00), 10.00,
                        board="chinext", dt=dt) is None
    assert fill_blocked("buy", _bar(12.00, 12.00, 12.00), 10.00,
                        board="chinext", dt=dt) == "limit_up"
    assert fill_blocked("sell", _bar(8.00, 8.00, 8.00), 10.00,
                        board="chinext", dt=dt) == "limit_down"


def test_chinext_10pct_before_reform():
    assert fill_blocked("buy", _bar(11.00, 11.00, 11.00), 10.00,
                        board="chinext", dt="2020-03-02") == "limit_up"


def test_star_20pct():
    assert fill_blocked("buy", _bar(12.00, 12.00, 12.00), 10.00, board="star",
                        dt="2024-01-05") == "limit_up"
    assert fill_blocked("buy", _bar(11.00, 11.00, 11.00), 10.00, board="star",
                        dt="2024-01-05") is None


def test_bse_ignored():
    assert limit_pct("bse") is None
    assert fill_blocked("buy", _bar(13.00, 13.00, 13.00), 10.00, board="bse") is None


def test_board_of_prefixes():
    assert board_of("300001") == "chinext"
    assert board_of("301001") == "chinext"
    assert board_of("688001") == "star"
    assert board_of("689009") == "star"
    assert board_of("600000") == "main"
    assert board_of("002001") == "main"
    assert board_of("000001") == "main"
    assert board_of("430047") == "bse"
    assert board_of("830799") == "bse"


def test_limit_pct_values():
    assert limit_pct("main") == pytest.approx(0.10)
    assert limit_pct("main", is_st=True) == pytest.approx(0.05)
    assert limit_pct("chinext", "2020-08-24") == pytest.approx(0.20)
    assert limit_pct("chinext", "2020-08-21") == pytest.approx(0.10)
    assert limit_pct("star", "2022-01-01") == pytest.approx(0.20)


# ── 浮点边界 ─────────────────────────────────────────────────────────────

def test_round_cent_float_artifacts():
    assert 9.99 * 1.1 != 10.99          # the artefact we guard against
    assert round_cent(9.99 * 1.1) == 10.99
    assert round_cent(3.54 * 1.1) == 3.89   # 3.894 rounds DOWN → +9.89% is the real limit
    assert round_cent(10.945) == 10.95      # 四舍五入, not banker's rounding
    assert round_cent(9.99 * 0.9) == 8.99


def test_float_boundary_prev_close_9_99():
    # 9.99 × 1.1 = 10.989000000000001 → limit 10.99; 一字 10.99 must block
    assert fill_blocked("buy", _bar(10.99, 10.99, 10.99), 9.99) == "limit_up"
    # 10.98 (one tick below limit, within 1-cent tolerance) 一字 → still blocked
    # (bars are qfq-rerounded so a 1-tick miss is treated as at-limit)
    assert fill_blocked("buy", _bar(10.98, 10.98, 10.98), 9.99) == "limit_up"
    # two ticks below → fillable
    assert fill_blocked("buy", _bar(10.97, 10.97, 10.97), 9.99) is None
    # tolerance can be switched off
    assert fill_blocked("buy", _bar(10.98, 10.98, 10.98), 9.99, tol_cents=0) is None


def test_limit_that_rounds_down_is_recognised():
    # prev 3.54 → 3.894 → limit 3.89 (+9.89%): a 9.95% threshold would miss it
    assert fill_blocked("buy", _bar(3.89, 3.89, 3.89), 3.54) == "limit_up"


def test_price_override_for_14_30_fill():
    # ENTRY_TIME=14_30: fill at the 14:29 close, strict; open is irrelevant
    bar = _bar(10.20, 11.00, 10.10)
    assert fill_blocked("buy", bar, 10.00, price=11.00, strict=True) == "limit_up"
    assert fill_blocked("buy", bar, 10.00, price=10.80, strict=True) is None
    assert fill_blocked("sell", bar, 10.00, price=9.00, strict=True) == "limit_down"


def test_invalid_action_raises():
    with pytest.raises(ValueError):
        fill_blocked("hold", _bar(10, 10, 10), 10.0)


# ── walk-forward helpers ─────────────────────────────────────────────────

def test_build_bar_lookup_prev_close_and_missing_day():
    from scripts.walk_forward_backtest import _build_bar_lookup
    d = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])  # 01-04 missing
    df = pd.DataFrame({"date": d, "open": [10.0, 11.0, 12.1], "high": [10.5, 11.0, 12.1],
                       "low": [9.8, 11.0, 12.1], "close": [10.0, 11.0, 12.1],
                       "volume": [1e6, 2e6, 3e6], "amount": [1e7, 2e7, 3e7]})
    lk = _build_bar_lookup({"600000": df})
    o, h, l, v, pc = lk[("600000", d[1])]
    assert (o, h, l, v) == (11.0, 11.0, 11.0, 2e6)
    assert pc == 10.0
    assert pd.isna(lk[("600000", d[0])][4])  # first row has no prev close
    assert ("600000", pd.Timestamp("2024-01-04")) not in lk
    # end-to-end: 一字涨停 on 01-03 blocks a buy, suspended on 01-04
    o, h, l, v, pc = lk[("600000", d[1])]
    assert fill_blocked("buy", {"open": o, "high": h, "low": l, "volume": v}, pc) == "limit_up"
    assert fill_blocked("buy", lk.get(("600000", pd.Timestamp("2024-01-04"))), 11.0) == "suspended"


def test_trade_cost_stats():
    from scripts.walk_forward_backtest import _trade_cost_stats
    nav = pd.DataFrame({"nav": [1.0] * 252})
    trade_log = [
        {"shares": 1000, "price": 10.0, "total_friction": 5.0},   # buy 10,000
        {"shares": 1000, "price": 12.0, "total_friction": 15.0},  # sell 12,000
    ]
    st = _trade_cost_stats(trade_log, nav, initial_capital=100_000, final_value=110_000)
    # (10,000 + 12,000) / 2 / 100,000 × 252/252 = 0.11
    assert st["turnover_annual"] == pytest.approx(0.11)
    assert st["friction_total"] == pytest.approx(20.0)
    assert st["friction_pct_of_final_nav"] == pytest.approx(20.0 / 110_000)
    empty = _trade_cost_stats([], pd.DataFrame({"nav": []}), 100_000, 100_000)
    assert empty["friction_total"] == 0.0
