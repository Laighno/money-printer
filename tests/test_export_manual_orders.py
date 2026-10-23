"""Tests for the manual order-sheet fallback (2026-10-23 QMT Python limits)."""

from __future__ import annotations

import csv
import json

import pytest

from scripts.export_manual_orders import (
    CSV_DIALECTS,
    _csv_row,
    _exchange_hint,
    _is_buy,
    build_sheet,
    write_outputs,
)


def _plan(orders):
    return {
        "generated_at": "2026-10-23T17:30:00",
        "account_snapshot": {
            "total_assets": 161513.0,
            "cash_available": 26265.0,
            "market_value": 135267.0,
        },
        "orders": orders,
        "alerts": [],
    }


BUY = {"code": "600363", "name": "联创光电", "action": "加仓",
       "shares": 100, "limit_price": 8.52, "cost": 852.0}
SELL = {"code": "000582", "name": "北部湾港", "action": "减仓",
        "shares": 300, "limit_price": 8.77, "cost": -2631.0}


def test_direction_comes_from_cost_not_action_text():
    """``action`` is display text (加仓/减仓/换仓/清仓); ``cost`` carries the sign."""
    assert _is_buy(BUY) is True
    assert _is_buy(SELL) is False
    # 换仓 appears on both sides of a swap, so the text alone is ambiguous.
    assert _is_buy({"action": "换仓", "cost": 500.0}) is True
    assert _is_buy({"action": "换仓", "cost": -500.0}) is False


def test_exchange_hint_covers_the_main_boards():
    assert _exchange_hint("600363") == "上交所"
    assert _exchange_hint("688001") == "上交所"
    assert _exchange_hint("000582") == "深交所"
    assert _exchange_hint("300750") == "深交所"
    assert _exchange_hint("830799") == "北交所"


def test_sheet_lists_every_order_and_puts_sells_first():
    md = build_sheet(_plan([BUY, SELL]), __import__("pathlib").Path("latest.json"))
    assert "600363" in md and "000582" in md
    assert md.index("① 卖出") < md.index("② 买入"), "sells must be actioned first"
    assert "8.52" in md and "8.77" in md


def test_empty_plan_says_so_without_crashing(tmp_path):
    md_path, csv_path = write_outputs(
        _plan([]), tmp_path / "latest.json", tmp_path, "guojin")
    assert csv_path is None, "no orders -> no import csv"
    assert "无需操作" in md_path.read_text(encoding="utf-8")


def test_csv_is_gbk_and_sells_come_first(tmp_path):
    """QMT on Windows reads GBK; utf-8 would show mojibake in the dialog."""
    _, csv_path = write_outputs(
        _plan([BUY, SELL]), tmp_path / "latest.json", tmp_path, "guojin")
    raw = csv_path.read_bytes()
    decoded = raw.decode("gbk")          # must not raise
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")              # proves it is really GBK, not ASCII-only

    rows = list(csv.reader(decoded.splitlines()))
    assert rows[0] == CSV_DIALECTS["guojin"]
    assert rows[1][0] == "000582" and rows[1][1] == "卖出"
    assert rows[2][0] == "600363" and rows[2][1] == "买入"


def test_csv_row_shapes_match_their_dialect():
    assert _csv_row(BUY, "guojin") == ["600363", "买入", "限价", "8.52", "100"]
    assert _csv_row(SELL, "generic") == ["000582", "sell", "8.77", "300"]
    for name, header in CSV_DIALECTS.items():
        assert len(_csv_row(BUY, name)) == len(header), f"{name} row/header mismatch"


def test_codes_are_zero_padded(tmp_path):
    """A plan may carry ints; 582 must not reach the broker as '582'."""
    thin = dict(SELL, code=582)
    _, csv_path = write_outputs(
        _plan([thin]), tmp_path / "latest.json", tmp_path, "guojin")
    rows = list(csv.reader(csv_path.read_text(encoding="gbk").splitlines()))
    assert rows[1][0] == "000582"
