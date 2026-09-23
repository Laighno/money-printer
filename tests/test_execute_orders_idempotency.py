"""execute_orders.run(): same-day idempotency + timeout re-check (2026-09-23).

Uses a minimal fake broker so we can script get_orders / place_limit_order
behaviour independent of DryRunBroker's autofill semantics.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mp.execution.dryrun_broker import DryRunBroker  # noqa: E402
from mp.execution.qmt_broker import (  # noqa: E402
    AccountInfo,
    OrderResult,
    OrderStatus,
    Position,
)
from scripts import execute_orders as eo  # noqa: E402

# ── fake broker ───────────────────────────────────────────────────

@dataclass
class FakeBroker:
    """Scriptable broker. ``orders`` is what get_orders returns; ``reject``
    is a set of codes whose place_limit_order returns success=False;
    ``ghost_on_reject`` makes a rejected submission ALSO show up in the
    next get_orders (simulating a timeout / lost ack)."""
    cash: float = 1_000_000.0
    positions: list = field(default_factory=list)
    orders: list = field(default_factory=list)
    reject: set = field(default_factory=set)
    ghost_on_reject: bool = False
    raise_on_get_orders: bool = False
    placed: list = field(default_factory=list)
    get_orders_calls: int = 0
    _seq: int = 100

    def is_connected(self):
        return True

    def connect(self):
        return True

    def disconnect(self):
        pass

    def get_account_info(self):
        mv = sum(p.market_value for p in self.positions)
        return AccountInfo(cash_available=self.cash, cash_frozen=0.0,
                           market_value=mv, total_assets=self.cash + mv,
                           updated_at="now")

    def get_positions(self):
        return list(self.positions)

    def get_orders(self, only_today=True):
        self.get_orders_calls += 1
        if self.raise_on_get_orders:
            raise TimeoutError("bridge rpc snapshot no response")
        return list(self.orders)

    def place_limit_order(self, code, action, shares, limit_price, order_remark=""):
        self.placed.append((code, action, shares, round(limit_price, 2)))
        self._seq += 1
        oid = str(self._seq)
        if code in self.reject:
            if self.ghost_on_reject:
                self.orders.append(OrderStatus(
                    order_id=oid, code=code, action=action,
                    shares_submitted=shares, shares_filled=0,
                    avg_fill_price=0.0, status="pending",
                    limit_price=float(limit_price)))
            return OrderResult(success=False, error="timeout waiting for ack",
                               code=code, action=action, shares=shares,
                               limit_price=limit_price)
        self.orders.append(OrderStatus(
            order_id=oid, code=code, action=action,
            shares_submitted=shares, shares_filled=0,
            avg_fill_price=0.0, status="pending",
            limit_price=float(limit_price)))
        return OrderResult(success=True, order_id=oid, code=code,
                           action=action, shares=shares,
                           limit_price=limit_price)


def _order(code, action, shares, limit):
    cost = shares * limit * (1 if action == "buy" else -1)
    return {"code": code, "name": code, "action": action, "shares": shares,
            "limit_price": limit, "cost": cost, "reason": "t"}


def _plan(orders):
    return {"generated_at": "x", "account_snapshot": {},
            "holdings_at_plan_time": [], "orders": orders, "alerts": []}


def _existing(code, action, shares, limit, status="pending", oid="1",
              limit_price="same"):
    return OrderStatus(order_id=oid, code=code, action=action,
                       shares_submitted=shares, shares_filled=0,
                       avg_fill_price=0.0, status=status,
                       limit_price=(limit if limit_price == "same" else limit_price))


@pytest.fixture(autouse=True)
def _no_price(monkeypatch):
    # No live price -> no drift skip, no cage re-price; plan limit is final.
    monkeypatch.setattr(eo, "_fetch_current_price", lambda c: None)


def _run(broker, plan, mode="auto"):
    return eo.run(broker, plan, mode=mode, fill_wait_seconds=0,
                  cash_settle_wait_seconds=0)


# ── helpers ───────────────────────────────────────────────────────

class TestHelpers:
    def test_order_key(self):
        assert eo._order_key("1", "buy", 100, 10.004) == ("000001", "buy", 100, 10.0)

    def test_find_matching_within_tolerance(self):
        o = _existing("000001", "buy", 100, 10.00)
        assert eo._find_matching_order([o], "000001", "buy", 100, [10.01]) is o
        assert eo._find_matching_order([o], "000001", "buy", 100, [9.99]) is o
        # keys are round(., 2): 10.02 is 2 fen away > 0.015 tolerance
        assert eo._find_matching_order([o], "000001", "buy", 100, [10.02]) is None

    def test_find_matching_ignores_cancelled_and_rejected(self):
        for st in ("cancelled", "rejected"):
            o = _existing("000001", "buy", 100, 10.0, status=st)
            assert eo._find_matching_order([o], "000001", "buy", 100, [10.0]) is None
        for st in ("pending", "partial", "filled"):
            o = _existing("000001", "buy", 100, 10.0, status=st)
            assert eo._find_matching_order([o], "000001", "buy", 100, [10.0]) is o

    def test_find_matching_requires_same_action_and_shares(self):
        o = _existing("000001", "buy", 100, 10.0)
        assert eo._find_matching_order([o], "000001", "sell", 100, [10.0]) is None
        assert eo._find_matching_order([o], "000001", "buy", 200, [10.0]) is None

    def test_find_matching_none_price_is_wildcard(self):
        o = _existing("000001", "buy", 100, 10.0, limit_price=None)
        assert eo._find_matching_order([o], "000001", "buy", 100, [99.0]) is o

    def test_find_matching_exclude_ids(self):
        o = _existing("000001", "buy", 100, 10.0, oid="7")
        assert eo._find_matching_order([o], "000001", "buy", 100, [10.0],
                                       exclude_ids={"7"}) is None


# ── (a) idempotency ───────────────────────────────────────────────

class TestIdempotency:
    def test_duplicate_skipped_not_resent(self):
        b = FakeBroker(orders=[_existing("000001", "buy", 100, 10.00)])
        res = _run(b, _plan([_order("000001", "buy", 100, 10.00),
                             _order("000002", "buy", 100, 5.00)]))
        by = {r["code"]: r for r in res}
        assert by["000001"]["status"] == "skipped_duplicate"
        assert by["000001"]["order_id"] == "1"
        assert by["000002"]["status"] == "sent"
        assert b.placed == [("000002", "buy", 100, 5.0)]

    def test_duplicate_price_tolerance(self):
        b = FakeBroker(orders=[_existing("000001", "buy", 100, 10.01)])
        res = _run(b, _plan([_order("000001", "buy", 100, 10.00)]))
        assert res[0]["status"] == "skipped_duplicate"
        b = FakeBroker(orders=[_existing("000001", "buy", 100, 10.03)])
        res = _run(b, _plan([_order("000001", "buy", 100, 10.00)]))
        assert res[0]["status"] == "sent"

    def test_cancelled_same_key_is_resent(self):
        b = FakeBroker(orders=[_existing("000001", "buy", 100, 10.0, status="cancelled")])
        res = _run(b, _plan([_order("000001", "buy", 100, 10.0)]))
        assert res[0]["status"] == "sent"

    def test_sell_duplicate_skipped(self):
        pos = Position(code="000001", name="x", shares_total=1000,
                       shares_available=1000, avg_cost=9, market_price=10,
                       market_value=10000)
        b = FakeBroker(positions=[pos],
                       orders=[_existing("000001", "sell", 500, 10.0, status="filled")])
        res = _run(b, _plan([_order("000001", "sell", 500, 10.0)]))
        assert res[0]["status"] == "skipped_duplicate"
        assert b.placed == []

    def test_snapshot_taken_once_before_first_order(self):
        b = FakeBroker()
        _run(b, _plan([_order("000001", "buy", 100, 10.0),
                       _order("000002", "buy", 100, 10.0)]))
        assert b.get_orders_calls == 1
        assert len(b.placed) == 2

    def test_duplicate_matches_repriced_limit(self, monkeypatch):
        """Previous run re-priced buy to the cage; this run sees the same
        live price -> final limit equals the existing order -> duplicate."""
        monkeypatch.setattr(eo, "_fetch_current_price", lambda c: 10.00)
        # plan 10.30 > cage max(10.2, 10.1)=10.20 -> final 10.20
        b = FakeBroker(orders=[_existing("000001", "buy", 100, 10.20)])
        res = _run(b, _plan([_order("000001", "buy", 100, 10.30)]))
        assert res[0]["status"] == "skipped_duplicate"

    def test_get_orders_failure_aborts_before_any_order(self):
        b = FakeBroker(raise_on_get_orders=True)
        res = _run(b, _plan([_order("000001", "buy", 100, 10.0)]))
        assert len(res) == 1 and res[0]["status"] == "failed"
        assert res[0]["name"] == "PREFLIGHT"
        assert b.placed == []

    def test_broker_without_get_orders_degrades_to_noop(self):
        b = FakeBroker()
        b2 = type("NoOrders", (), {})()
        for name in ("is_connected", "connect", "disconnect", "get_account_info",
                     "get_positions", "place_limit_order"):
            setattr(b2, name, getattr(b, name))
        res = _run(b2, _plan([_order("000001", "buy", 100, 10.0)]))
        assert res[0]["status"] == "sent"

    def test_dryrun_does_not_dedup(self):
        """DryRunBroker autofills; a pre-seeded same-key order must NOT block
        the preview -- dryrun is exempt from both live guards."""
        b = DryRunBroker(cash=100000, autofill=True)
        b.connect()
        b.place_limit_order("000001", "buy", 100, 10.0)   # seed same key
        res = _run(b, _plan([_order("000001", "buy", 100, 10.0)]), mode="dryrun")
        assert res[0]["status"] == "sent"


# ── (b) timeout re-check ──────────────────────────────────────────

class TestTimeoutRecheck:
    def test_plain_failure_stays_failed(self):
        b = FakeBroker(reject={"000001"})
        res = _run(b, _plan([_order("000001", "buy", 100, 10.0),
                             _order("000002", "buy", 100, 10.0)]))
        by = {r["code"]: r for r in res}
        assert by["000001"]["status"] == "failed"
        assert by["000002"]["status"] == "sent"       # not blocked
        assert b.get_orders_calls == 2                # snapshot + recheck

    def test_ghost_order_marks_unknown_and_blocks_later_buys(self):
        b = FakeBroker(reject={"000001"}, ghost_on_reject=True)
        res = _run(b, _plan([_order("000001", "buy", 100, 10.0),
                             _order("000002", "buy", 100, 10.0),
                             _order("000003", "buy", 100, 10.0)]))
        assert [r["status"] for r in res] == [
            "unknown_submitted", "blocked_after_unknown", "blocked_after_unknown"]
        assert res[0]["order_id"] == "101"
        # only the first order ever reached the broker
        assert [p[0] for p in b.placed] == ["000001"]

    def test_ghost_during_sells_blocks_all_buys_but_not_sells(self):
        pos = [Position(code=c, name=c, shares_total=1000, shares_available=1000,
                        avg_cost=9, market_price=10, market_value=10000)
               for c in ("000001", "000002")]
        b = FakeBroker(positions=pos, reject={"000001"}, ghost_on_reject=True)
        res = _run(b, _plan([_order("000001", "sell", 100, 10.0),
                             _order("000002", "sell", 100, 10.0),
                             _order("000003", "buy", 100, 10.0)]))
        by = {r["code"]: r for r in res}
        assert by["000001"]["status"] == "unknown_submitted"
        assert by["000002"]["status"] == "sent"
        assert by["000003"]["status"] == "blocked_after_unknown"

    def test_preexisting_same_key_not_mistaken_for_ghost(self):
        """A same-key order that was already in the snapshot is a duplicate
        (skipped) -- it never reaches place_limit_order, so it cannot be
        misread as a ghost. And a cancelled pre-existing order must not
        satisfy the re-check either."""
        b = FakeBroker(reject={"000001"},
                       orders=[_existing("000001", "buy", 100, 10.0,
                                         status="cancelled", oid="old")])
        res = _run(b, _plan([_order("000001", "buy", 100, 10.0)]))
        assert res[0]["status"] == "failed"

    def test_recheck_failure_treated_as_unknown(self):
        class Flaky(FakeBroker):
            def get_orders(self, only_today=True):
                self.get_orders_calls += 1
                if self.get_orders_calls >= 2:
                    raise TimeoutError("bridge stalled")
                return list(self.orders)

        b = Flaky(reject={"000001"})
        res = _run(b, _plan([_order("000001", "buy", 100, 10.0),
                             _order("000002", "buy", 100, 10.0)]))
        assert res[0]["status"] == "unknown_submitted"
        assert res[0]["order_id"] is None
        assert res[1]["status"] == "blocked_after_unknown"

    def test_dryrun_failure_no_recheck(self):
        """DryRunBroker rejects odd lots -> failed; in dryrun a failure never
        escalates to unknown_submitted and never blocks the next order."""
        b = DryRunBroker(cash=100000, autofill=True)
        b.connect()
        res = _run(b, _plan([_order("000001", "buy", 150, 10.0),
                             _order("000002", "buy", 150, 10.0)]), mode="dryrun")
        assert [r["status"] for r in res] == ["failed", "failed"]
        assert all("valid lot" in r["note"] for r in res)


# ── summary + exit code ───────────────────────────────────────────

class TestSummaryAndExit:
    def test_summary_handles_new_statuses(self):
        res = [
            {"status": "skipped_duplicate", "code": "000001", "name": "a",
             "action": "buy", "shares": 100, "limit_price": 10.0, "note": "n"},
            {"status": "unknown_submitted", "code": "000002", "name": "b",
             "action": "buy", "shares": 100, "limit_price": 10.0,
             "order_id": "77", "note": "n"},
            {"status": "blocked_after_unknown", "code": "000003", "name": "c",
             "action": "buy", "shares": 100, "limit_price": 10.0, "note": "n"},
        ]
        md = eo._format_summary(res, mode="auto")
        assert "UNKNOWN_SUBMITTED" in md
        assert "order_id=77" in md
        assert "重复跳过 1" in md
        assert "阻断 1" in md

    def test_main_exits_12_on_unknown_submitted(self, tmp_path, monkeypatch):
        plan_p = tmp_path / "plan.json"
        import json
        plan = _plan([_order("000001", "buy", 100, 10.0)])
        plan["source"] = {"is_prod": True}
        plan_p.write_text(json.dumps(plan), encoding="utf-8")

        fake = FakeBroker(reject={"000001"}, ghost_on_reject=True)
        import mp.execution.broker_factory as bf
        monkeypatch.setattr(bf, "make_broker", lambda **kw: fake)
        monkeypatch.setattr(eo, "ROOT", tmp_path)          # exec log -> tmp
        import mp.common.paths as paths
        monkeypatch.setattr(paths, "AUDIT_LOG_PATH", tmp_path / "audit.log")
        monkeypatch.setattr("mp.risk.freeze.guard_or_raise", lambda mode, **kw: None)
        monkeypatch.setattr(sys, "argv", [
            "execute_orders.py", "--mode", "auto", "--plan", str(plan_p),
            "--qmt-account", "1", "--qmt-userdata", "x", "--cash-settle-wait", "0",
        ])
        monkeypatch.setattr(eo.time, "sleep", lambda s: None)
        assert eo.main() == eo.EXIT_UNKNOWN_SUBMITTED == 12
