"""Tests for the order execution pipeline — safety gates and dry-run flow.

The actual QMT integration is not testable in CI (xtquant only runs on
Windows with QMT installed), but everything around it — pre-flight
checks, ordering, dryrun semantics — is plain Python and fully tested.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mp.execution.dryrun_broker import DryRunBroker
from mp.execution.qmt_broker import Position, _plain_code, _xt_code


# ──────────────────────────────────────────────────────────────────
# Exchange routing
# ──────────────────────────────────────────────────────────────────

class TestExchangeRouting:
    def test_shanghai_main_board(self):
        assert _xt_code("600519") == "600519.SH"   # 茅台
        assert _xt_code("603707") == "603707.SH"   # 健友股份

    def test_shanghai_star(self):
        assert _xt_code("688520") == "688520.SH"   # 神州细胞 (科创板)
        assert _xt_code("689009") == "689009.SH"   # CDR (科创板 CDR)

    def test_shenzhen_main(self):
        assert _xt_code("000539") == "000539.SZ"   # 粤电力A
        assert _xt_code("002439") == "002439.SZ"   # 启明星辰

    def test_chinext(self):
        assert _xt_code("300033") == "300033.SZ"   # 同花顺 (创业板)
        assert _xt_code("300782") == "300782.SZ"   # 卓胜微

    def test_beijing(self):
        assert _xt_code("835174") == "835174.BJ"   # 北证 8 字头
        assert _xt_code("430090") == "430090.BJ"   # 北证 4 字头

    def test_roundtrip(self):
        for code in ["000539", "600673", "688520", "300033"]:
            assert _plain_code(_xt_code(code)) == code


# ──────────────────────────────────────────────────────────────────
# Pre-flight: account reconcile
# ──────────────────────────────────────────────────────────────────

class TestAccountReconcile:
    def test_exact_match_passes(self):
        from mp.execution.qmt_broker import AccountInfo
        from scripts.execute_orders import preflight_account_reconcile
        live = AccountInfo(cash_available=10000, cash_frozen=0, market_value=90000,
                           total_assets=100000, updated_at="2026-05-20")
        ok, msg = preflight_account_reconcile({"total_assets": 100000}, live)
        assert ok

    def test_within_tolerance_passes(self):
        from mp.execution.qmt_broker import AccountInfo
        from scripts.execute_orders import preflight_account_reconcile
        live = AccountInfo(cash_available=10000, cash_frozen=0, market_value=89000,
                           total_assets=99000, updated_at="2026-05-20")
        ok, msg = preflight_account_reconcile({"total_assets": 100000}, live,
                                                tolerance_pct=0.02)
        assert ok, msg

    def test_drifted_account_fails(self):
        from mp.execution.qmt_broker import AccountInfo
        from scripts.execute_orders import preflight_account_reconcile
        live = AccountInfo(cash_available=10000, cash_frozen=0, market_value=80000,
                           total_assets=90000, updated_at="2026-05-20")
        ok, msg = preflight_account_reconcile({"total_assets": 100000}, live,
                                                tolerance_pct=0.02)
        assert not ok
        assert "drifted" in msg

    def test_missing_plan_total_passes(self):
        """No plan_total to compare → don't block, just warn."""
        from mp.execution.qmt_broker import AccountInfo
        from scripts.execute_orders import preflight_account_reconcile
        live = AccountInfo(cash_available=0, cash_frozen=0, market_value=0,
                           total_assets=0, updated_at="2026-05-20")
        ok, msg = preflight_account_reconcile({}, live)
        assert ok


# ──────────────────────────────────────────────────────────────────
# Pre-flight: positions reconcile
# ──────────────────────────────────────────────────────────────────

class TestPositionsReconcile:
    def test_all_match(self):
        from scripts.execute_orders import preflight_positions_reconcile
        plan = [{"name": "粤电力A", "code": "000539", "shares": 26300}]
        live = [Position(code="000539", name="粤电力A", shares_total=26300,
                          shares_available=14200, avg_cost=7.429,
                          market_price=6.46, market_value=170000)]
        ok, warnings = preflight_positions_reconcile(plan, live)
        assert ok
        assert warnings == []

    def test_missing_holding(self):
        from scripts.execute_orders import preflight_positions_reconcile
        plan = [{"name": "粤电力A", "code": "000539", "shares": 26300}]
        live = []
        ok, warnings = preflight_positions_reconcile(plan, live)
        assert not ok
        assert "broker has 0" in warnings[0]

    def test_share_count_diff(self):
        from scripts.execute_orders import preflight_positions_reconcile
        plan = [{"name": "X", "code": "000001", "shares": 1000}]
        live = [Position(code="000001", name="X", shares_total=800,
                          shares_available=800, avg_cost=10, market_price=10,
                          market_value=8000)]
        ok, warnings = preflight_positions_reconcile(plan, live)
        assert not ok
        assert "1000" in warnings[0] and "800" in warnings[0]


# ──────────────────────────────────────────────────────────────────
# Pre-flight: price drift
# ──────────────────────────────────────────────────────────────────

class TestPriceDrift:
    def test_buy_within_tolerance(self, monkeypatch):
        from scripts import execute_orders
        monkeypatch.setattr(execute_orders, "_fetch_current_price", lambda c: 10.10)
        ok, msg, cur = execute_orders.preflight_price_drift("000001", 10.00, "buy", 0.02)
        assert ok
        assert cur == 10.10

    def test_buy_above_ceiling_skipped(self, monkeypatch):
        from scripts import execute_orders
        # limit 10, max_drift 2% → ceiling 10.20, current 10.50 → too high
        monkeypatch.setattr(execute_orders, "_fetch_current_price", lambda c: 10.50)
        ok, msg, cur = execute_orders.preflight_price_drift("000001", 10.00, "buy", 0.02)
        assert not ok
        assert "too expensive" in msg

    def test_sell_within_tolerance(self, monkeypatch):
        from scripts import execute_orders
        # limit 10, max_drift 2% → floor 9.80, current 9.90 → OK
        monkeypatch.setattr(execute_orders, "_fetch_current_price", lambda c: 9.90)
        ok, msg, cur = execute_orders.preflight_price_drift("000001", 10.00, "sell", 0.02)
        assert ok

    def test_sell_below_floor_skipped(self, monkeypatch):
        from scripts import execute_orders
        # limit 10, max_drift 2% → floor 9.80, current 9.50 → too low
        monkeypatch.setattr(execute_orders, "_fetch_current_price", lambda c: 9.50)
        ok, msg, cur = execute_orders.preflight_price_drift("000001", 10.00, "sell", 0.02)
        assert not ok
        assert "dump too low" in msg

    def test_unavailable_price_proceeds(self, monkeypatch):
        """If sina is down, don't block — let the broker decide."""
        from scripts import execute_orders
        monkeypatch.setattr(execute_orders, "_fetch_current_price", lambda c: None)
        ok, msg, cur = execute_orders.preflight_price_drift("000001", 10.00, "buy", 0.02)
        assert ok
        assert cur is None


# ──────────────────────────────────────────────────────────────────
# DryRunBroker behaviour
# ──────────────────────────────────────────────────────────────────

class TestDryRunBroker:
    def test_buy_consumes_cash(self):
        b = DryRunBroker(cash=10000, autofill=True)
        b.connect()
        r = b.place_limit_order("000001", "buy", 100, 50.0)
        assert r.success
        info = b.get_account_info()
        assert info.cash_available == pytest.approx(5000)
        pos = b.get_positions()
        assert pos[0].code == "000001" and pos[0].shares_total == 100

    def test_buy_rejects_when_insufficient_cash(self):
        b = DryRunBroker(cash=1000, autofill=True)
        b.connect()
        r = b.place_limit_order("000001", "buy", 100, 50.0)
        assert not r.success
        assert "insufficient cash" in r.error

    def test_sell_releases_cash_immediately(self):
        """T+0 cash: sells fund subsequent buys same session."""
        existing = Position(code="000539", name="粤电力A", shares_total=10000,
                             shares_available=10000, avg_cost=7.0,
                             market_price=6.5, market_value=65000)
        b = DryRunBroker(cash=0, positions=[existing], autofill=True)
        b.connect()
        # Sell some 粤电力A
        r1 = b.place_limit_order("000539", "sell", 5000, 6.50)
        assert r1.success
        # Cash should now be ¥32,500
        info = b.get_account_info()
        assert info.cash_available == pytest.approx(32500)
        # Now buy something else with that cash
        r2 = b.place_limit_order("000001", "buy", 1000, 30.0)
        assert r2.success
        assert b.get_account_info().cash_available == pytest.approx(2500)

    def test_sell_rejects_more_than_available(self):
        existing = Position(code="000001", name="X", shares_total=500,
                             shares_available=200,    # 300 T+1 locked
                             avg_cost=10, market_price=10, market_value=5000)
        b = DryRunBroker(cash=0, positions=[existing], autofill=True)
        b.connect()
        # Try to sell 400 (more than 200 available)
        r = b.place_limit_order("000001", "sell", 400, 10.0)
        assert not r.success
        assert "insufficient shares" in r.error

    def test_invalid_lot_size_rejected(self):
        b = DryRunBroker(cash=10000, autofill=True)
        b.connect()
        # A-shares trade in lots of 100
        for shares in [50, 99, 150, 1, 99999 + 1]:
            r = b.place_limit_order("000001", "buy", shares, 10.0)
            assert not r.success or shares % 100 == 0, f"shares={shares} should fail"


# ──────────────────────────────────────────────────────────────────
# End-to-end: full run() with DryRunBroker on realistic plan
# ──────────────────────────────────────────────────────────────────

class TestRunOrchestration:
    def _make_plan(self):
        """Replicate today's actual rebalance plan."""
        return {
            "generated_at": "2026-05-20T17:30:00",
            "report_date": "2026-05-20",
            "account_snapshot": {
                "total_assets": 271428.80,
                "cash_available": 502.80,
                "market_value": 270926.00,
            },
            "holdings_at_plan_time": [
                {"name": "粤电力A", "code": "000539",
                 "shares": 26300, "avg_cost": 7.429},
                {"name": "启明星辰", "code": "002439",
                 "shares": 4000, "avg_cost": 16.156},
                {"name": "大北农", "code": "002385",
                 "shares": 7200, "avg_cost": 3.860},
            ],
            "orders": [
                # sells first (priority 0)
                {"code": "000539", "name": "粤电力A", "action": "减仓",
                 "shares": 18400, "limit_price": 6.39, "cost": -117576,
                 "reason": "目标仓位 19% (现 63%)"},
                # buys (priority 1)
                {"code": "600363", "name": "联创光电", "action": "买入",
                 "shares": 600, "limit_price": 42.86, "cost": 25716,
                 "reason": "top #2"},
                {"code": "603529", "name": "爱玛科技", "action": "买入",
                 "shares": 1200, "limit_price": 22.15, "cost": 26580,
                 "reason": "top #3"},
            ],
            "alerts": [],
        }

    def test_dryrun_executes_all_orders(self, monkeypatch):
        """End-to-end: all preflights pass, all orders fill."""
        from scripts import execute_orders

        # Mock prices within tolerance
        monkeypatch.setattr(execute_orders, "_fetch_current_price",
                            lambda c: {
                                "000539": 6.40,
                                "600363": 42.50,
                                "603529": 22.00,
                            }.get(c))

        plan = self._make_plan()
        plan_positions = [
            Position(code=h["code"], name=h["name"],
                     shares_total=h["shares"], shares_available=h["shares"],
                     avg_cost=h["avg_cost"], market_price=0, market_value=0)
            for h in plan["holdings_at_plan_time"]
        ]
        broker = DryRunBroker(
            cash=plan["account_snapshot"]["cash_available"],
            positions=plan_positions, autofill=True,
        )

        # Bug: DryRunBroker's market_value is 0 (no real prices),
        # so its total_assets won't match plan's total_assets.
        # Use loose reconcile tolerance for this test.
        from mp.execution import qmt_broker as qb
        monkeypatch.setattr(execute_orders, "preflight_account_reconcile",
                            lambda plan_acct, live, tolerance_pct=0.02: (True, "ok"))

        results = execute_orders.run(broker, plan, mode="dryrun",
                                       fill_wait_seconds=0)

        # All 3 should be sent
        sent = [r for r in results if r["status"] == "sent"]
        assert len(sent) == 3
        # Sells should execute before buys (priority sort)
        assert sent[0]["action"] == "sell"
        assert sent[1]["action"] == "buy"
        assert sent[2]["action"] == "buy"

    def test_skips_when_price_drifted(self, monkeypatch):
        """Buy with current price > limit + 2% is skipped."""
        from scripts import execute_orders

        # 600363 current 50 vs limit 42.86 → drift 17% → skip
        monkeypatch.setattr(execute_orders, "_fetch_current_price",
                            lambda c: {
                                "000539": 6.40, "600363": 50.00, "603529": 22.00,
                            }.get(c))
        monkeypatch.setattr(execute_orders, "preflight_account_reconcile",
                            lambda *a, **kw: (True, "ok"))

        plan = self._make_plan()
        # Build positions directly (NOT via place_limit_order) so shares_available
        # = shares_total (not T+1 locked from a fake "today" buy).
        plan_positions = [
            Position(code=h["code"], name=h["name"],
                     shares_total=h["shares"], shares_available=h["shares"],
                     avg_cost=h["avg_cost"],
                     market_price=h["avg_cost"],
                     market_value=h["shares"] * h["avg_cost"])
            for h in plan["holdings_at_plan_time"]
        ]
        broker = DryRunBroker(cash=20000, positions=plan_positions, autofill=True)
        broker.connect()

        results = execute_orders.run(broker, plan, mode="dryrun",
                                       fill_wait_seconds=0)
        codes_sent = [r["code"] for r in results if r["status"] == "sent"]
        codes_skipped = [r["code"] for r in results if r["status"] == "skipped"]
        assert "600363" in codes_skipped
        assert "603529" in codes_sent
        # 000539 sell: limit 6.39, current 6.40 → within tolerance, sell OK
        assert "000539" in codes_sent

    def test_max_orders_cap(self, monkeypatch):
        """If plan exceeds max_orders, run aborts entirely."""
        from scripts import execute_orders
        plan = self._make_plan()
        broker = DryRunBroker(cash=300000, autofill=True)
        results = execute_orders.run(broker, plan, mode="dryrun", max_orders=2)
        assert results == []   # aborted

    def test_concentration_cap_blocks_runaway_buy(self, monkeypatch):
        """If a buy would push single-stock concentration over the cap, skip it."""
        from scripts import execute_orders

        monkeypatch.setattr(execute_orders, "_fetch_current_price",
                            lambda c: {"600363": 42.86}.get(c))
        monkeypatch.setattr(execute_orders, "preflight_account_reconcile",
                            lambda *a, **kw: (True, "ok"))

        # Tiny portfolio, but plan tries to buy a HUGE position
        broker = DryRunBroker(cash=100000, autofill=True)
        plan = {
            "generated_at": "x",
            "account_snapshot": {"total_assets": 100000, "cash_available": 100000,
                                  "market_value": 0},
            "holdings_at_plan_time": [],
            "orders": [
                # 1000 shares × ¥42.86 = ¥42,860, that's 43% of 100k assets
                # If max_single_pct = 0.30, this should be blocked
                {"code": "600363", "name": "联创光电", "action": "买入",
                 "shares": 1000, "limit_price": 42.86, "cost": 42860,
                 "reason": "test"},
            ],
            "alerts": [],
        }

        results = execute_orders.run(broker, plan, mode="dryrun",
                                       max_single_pct=0.30,
                                       fill_wait_seconds=0)
        assert results[0]["status"] == "skipped"
        assert "cap" in results[0]["note"]

    def test_buy_limit_capped_at_price_cage_when_stock_opens_low(self, monkeypatch):
        """Round 260 Bug 1: plan limit prev_close*1.01 > cage when stock opens
        low → re-price down to cage. cage_max = max(live*1.02, live+0.10).
        """
        from scripts import execute_orders

        # Plan limit ¥42.86, live ¥40.00 → cage = max(40.80, 40.10) = 40.80
        # 42.86 > 40.80 → cap to 40.80
        monkeypatch.setattr(execute_orders, "_fetch_current_price",
                            lambda c: {"600363": 40.00}.get(c))
        monkeypatch.setattr(execute_orders, "preflight_account_reconcile",
                            lambda *a, **kw: (True, "ok"))

        broker = DryRunBroker(cash=100000, autofill=True)
        plan = {
            "generated_at": "x",
            "account_snapshot": {"total_assets": 100000, "cash_available": 100000,
                                  "market_value": 0},
            "holdings_at_plan_time": [],
            "orders": [
                {"code": "600363", "name": "test", "action": "buy",
                 "shares": 100, "limit_price": 42.86, "cost": 4286,
                 "reason": "test"},
            ],
            "alerts": [],
        }

        results = execute_orders.run(broker, plan, mode="dryrun",
                                       max_single_pct=0.95,
                                       fill_wait_seconds=0,
                                       cash_settle_wait_seconds=0)
        assert results[0]["status"] == "sent"
        assert abs(results[0]["limit_price"] - 40.80) < 0.01, \
            f"expected cage ¥40.80, got ¥{results[0]['limit_price']}"

    def test_buy_limit_preserved_when_within_cage(self, monkeypatch):
        """Round 260 Bug 1: plan limit BELOW cage_max → preserved (no upward repricing).
        """
        from scripts import execute_orders

        # Plan ¥42.86, live ¥42.50 → cage = max(43.35, 42.60) = 43.35
        # Plan 42.86 < cage 43.35 → no reprice
        monkeypatch.setattr(execute_orders, "_fetch_current_price",
                            lambda c: {"600363": 42.50}.get(c))
        monkeypatch.setattr(execute_orders, "preflight_account_reconcile",
                            lambda *a, **kw: (True, "ok"))

        broker = DryRunBroker(cash=100000, autofill=True)
        plan = {
            "generated_at": "x",
            "account_snapshot": {"total_assets": 100000, "cash_available": 100000,
                                  "market_value": 0},
            "holdings_at_plan_time": [],
            "orders": [
                {"code": "600363", "name": "test", "action": "buy",
                 "shares": 100, "limit_price": 42.86, "cost": 4286,
                 "reason": "test"},
            ],
            "alerts": [],
        }

        results = execute_orders.run(broker, plan, mode="dryrun",
                                       max_single_pct=0.95,
                                       fill_wait_seconds=0,
                                       cash_settle_wait_seconds=0)
        assert results[0]["status"] == "sent"
        assert results[0]["limit_price"] == 42.86

    def test_buy_skipped_when_cash_insufficient(self, monkeypatch):
        """Round 260 Bug 2: broker.cash_available < buy notional → skip cleanly.

        Use big existing positions to satisfy concentration check while
        keeping cash too small. Mimics real scenario where sells haven't
        settled yet so account has positions+cash but cash is low.
        """
        from scripts import execute_orders

        monkeypatch.setattr(execute_orders, "_fetch_current_price",
                            lambda c: {"600363": 42.50}.get(c))
        monkeypatch.setattr(execute_orders, "preflight_account_reconcile",
                            lambda *a, **kw: (True, "ok"))

        # 500k assets total, but only ¥1,000 cash (rest in positions)
        # Buy needs 100 × ¥42.86 = ¥4,286 → cash short
        # 4286/500000 = 0.86% (well under concentration cap)
        seed_pos = Position(code="999999", name="seed",
                             shares_total=10000, shares_available=10000,
                             avg_cost=49.90, market_price=49.90,
                             market_value=499000)
        broker = DryRunBroker(cash=1000, positions=[seed_pos], autofill=True)
        plan = {
            "generated_at": "x",
            "account_snapshot": {"total_assets": 500000, "cash_available": 1000,
                                  "market_value": 499000},
            "holdings_at_plan_time": [],
            "orders": [
                {"code": "600363", "name": "test", "action": "buy",
                 "shares": 100, "limit_price": 42.86, "cost": 4286,
                 "reason": "test"},
            ],
            "alerts": [],
        }

        results = execute_orders.run(broker, plan, mode="dryrun",
                                       max_single_pct=0.95,
                                       fill_wait_seconds=0,
                                       cash_settle_wait_seconds=0)
        assert results[0]["status"] == "skipped"
        assert "insufficient cash" in results[0]["note"]

    def test_sell_limit_raised_to_cage_when_stock_opens_high(self, monkeypatch):
        """Round 265 Bug 1b (advisor 263): symmetric SELL cage fix.

        Plan limit = prev_close × 0.99. When stock high-opens, cage floor
        = min(live × 0.98, live - 0.10) may be ABOVE plan → broker rejects.
        Clip plan limit UP to cage_min.

        Real example (6/12 002402): plan ¥23.62 < cage ¥23.72 (live ¥24.20).
        """
        from scripts import execute_orders

        # Plan SELL ¥23.62 (prev_close 23.86 × 0.99), live ¥24.20
        # cage_min = min(24.20*0.98, 24.20-0.10) = min(23.72, 24.10) = 23.72
        # 23.62 < 23.72 → raise to 23.72
        monkeypatch.setattr(execute_orders, "_fetch_current_price",
                            lambda c: {"002402": 24.20}.get(c))
        monkeypatch.setattr(execute_orders, "preflight_account_reconcile",
                            lambda *a, **kw: (True, "ok"))

        # Need an existing position to sell
        seed_pos = Position(code="002402", name="test",
                             shares_total=1000, shares_available=1000,
                             avg_cost=23.86, market_price=24.20,
                             market_value=24200)
        broker = DryRunBroker(cash=10000, positions=[seed_pos], autofill=True)
        plan = {
            "generated_at": "x",
            "account_snapshot": {"total_assets": 34200, "cash_available": 10000,
                                  "market_value": 24200},
            "holdings_at_plan_time": [],
            "orders": [
                {"code": "002402", "name": "test", "action": "sell",
                 "shares": 100, "limit_price": 23.62, "cost": -2362,
                 "reason": "test"},
            ],
            "alerts": [],
        }

        results = execute_orders.run(broker, plan, mode="dryrun",
                                       fill_wait_seconds=0,
                                       cash_settle_wait_seconds=0)
        assert results[0]["status"] == "sent"
        assert abs(results[0]["limit_price"] - 23.72) < 0.01, \
            f"expected cage ¥23.72, got ¥{results[0]['limit_price']}"

    def test_sell_limit_preserved_when_above_cage(self, monkeypatch):
        """Round 265 Bug 1b: SELL plan limit ABOVE cage_min → preserved
        (no downward clipping; cage only raises, never lowers).
        """
        from scripts import execute_orders

        # Plan SELL ¥23.62, live ¥23.50 → cage_min = min(23.03, 23.40) = 23.03
        # plan 23.62 > cage 23.03 → no reprice
        monkeypatch.setattr(execute_orders, "_fetch_current_price",
                            lambda c: {"002402": 23.50}.get(c))
        monkeypatch.setattr(execute_orders, "preflight_account_reconcile",
                            lambda *a, **kw: (True, "ok"))

        seed_pos = Position(code="002402", name="test",
                             shares_total=1000, shares_available=1000,
                             avg_cost=23.86, market_price=23.50,
                             market_value=23500)
        broker = DryRunBroker(cash=10000, positions=[seed_pos], autofill=True)
        plan = {
            "generated_at": "x",
            "account_snapshot": {"total_assets": 33500, "cash_available": 10000,
                                  "market_value": 23500},
            "holdings_at_plan_time": [],
            "orders": [
                {"code": "002402", "name": "test", "action": "sell",
                 "shares": 100, "limit_price": 23.62, "cost": -2362,
                 "reason": "test"},
            ],
            "alerts": [],
        }

        results = execute_orders.run(broker, plan, mode="dryrun",
                                       fill_wait_seconds=0,
                                       cash_settle_wait_seconds=0)
        assert results[0]["status"] == "sent"
        assert results[0]["limit_price"] == 23.62

    def test_sells_run_before_buys_with_phase_separation(self, monkeypatch):
        """Round 260 Bug 2: sells/buys split into two phases."""
        from scripts import execute_orders

        monkeypatch.setattr(execute_orders, "_fetch_current_price",
                            lambda c: {"000539": 6.40, "603529": 22.00,
                                       "600363": 42.86}.get(c))
        monkeypatch.setattr(execute_orders, "preflight_account_reconcile",
                            lambda *a, **kw: (True, "ok"))

        plan = self._make_plan()
        plan_positions = [
            Position(code=h["code"], name=h["name"],
                     shares_total=h["shares"], shares_available=h["shares"],
                     avg_cost=h["avg_cost"],
                     market_price=h["avg_cost"],
                     market_value=h["shares"] * h["avg_cost"])
            for h in plan["holdings_at_plan_time"]
        ]
        broker = DryRunBroker(cash=200000, positions=plan_positions, autofill=True)
        broker.connect()

        results = execute_orders.run(broker, plan, mode="dryrun",
                                       fill_wait_seconds=0,
                                       cash_settle_wait_seconds=0)

        sent = [r for r in results if r["status"] == "sent"]
        actions = [r["action"] for r in sent]
        last_sell_idx = max((i for i, a in enumerate(actions) if a == "sell"),
                             default=-1)
        first_buy_idx = min((i for i, a in enumerate(actions) if a == "buy"),
                             default=len(actions))
        assert last_sell_idx < first_buy_idx, \
            f"sells must precede buys; got {actions}"
