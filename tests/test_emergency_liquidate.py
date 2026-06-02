"""Tests for ``emergency_liquidate_all`` across the DryRun + QMTMock
brokers (P8-β-1b, docs/dialog/ round 57).

Five spec'd scenarios, parametrized across both brokers where the
scenario is broker-agnostic.  QMTBroker itself is *not* tested here:

- xtquant is unavailable on macOS dev hosts, so :meth:`QMTBroker.connect`
  cannot be exercised without mocking the entire xtquant surface;
- the live-QMT path's real validation is deferred to β-3 (user manual
  Windows VNC + 1 case Approach B run, per advisor round 56);
- QMTBroker's ``emergency_liquidate_all`` is a one-line delegation to
  the same shared :func:`_emergency_liquidate_impl` that DryRun /
  QMTMock use, so behaviour fidelity is already covered.
"""
from __future__ import annotations

import pytest

from mp.execution.dryrun_broker import DryRunBroker
from mp.execution.qmt_broker import EmergencyResult, Position
from mp.execution.qmt_mock_broker import QMTMockBroker, _QMTMockConfig


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(params=["dryrun", "mock"])
def broker_kind(request) -> str:
    return request.param


@pytest.fixture
def make_broker(broker_kind):
    """Factory: callable that returns a connected broker of ``broker_kind``.

    ``config`` is only honored for QMTMockBroker; passing it to DryRun
    triggers a skip (the scenario is broker-specific)."""
    def _make(
        cash: float = 0.0,
        positions: list[Position] | None = None,
        account_id: str = "ACCT123",
        config: _QMTMockConfig | None = None,
    ):
        if broker_kind == "dryrun":
            if config is not None:
                pytest.skip(
                    "scenario uses _QMTMockConfig (mock-only); "
                    "covered by the mock parametrization"
                )
            b = DryRunBroker(
                cash=cash, positions=positions or [],
                autofill=True, account_id=account_id,
            )
        else:
            b = QMTMockBroker(
                cash=cash, positions=positions or [],
                config=config, account_id=account_id,
            )
        b.connect()
        return b

    return _make


def _make_positions(*specs):
    """Build Position list from ``(code, name, shares, market_price)`` tuples."""
    return [
        Position(
            code=c, name=name,
            shares_total=shares, shares_available=shares,
            avg_cost=market_price * 0.95,
            market_price=market_price,
            market_value=shares * market_price,
        )
        for (c, name, shares, market_price) in specs
    ]


# ──────────────────────────────────────────────
# Test 1: bad confirm_string → ValueError, zero state mutation
# ──────────────────────────────────────────────

def test_bad_confirm_raises_valueerror_no_mutation(broker_kind, make_broker):
    """confirm_string mismatch → ValueError; broker state unchanged."""
    positions = _make_positions(("600000", "浦发银行", 1000, 10.0))
    b = make_broker(cash=0.0, positions=positions)

    info_before = b.get_account_info()
    pos_before = list(b.get_positions())
    orders_before = list(b.get_orders())

    with pytest.raises(ValueError, match="confirm_string mismatch"):
        b.emergency_liquidate_all(confirm_string="WRONG")

    info_after = b.get_account_info()
    pos_after = list(b.get_positions())
    orders_after = list(b.get_orders())

    assert info_after.cash_available == info_before.cash_available
    assert info_after.market_value == info_before.market_value
    assert pos_after == pos_before
    assert orders_after == orders_before


# ──────────────────────────────────────────────
# Test 2: 3 positions, all submit successfully
# ──────────────────────────────────────────────

def test_3_positions_all_submitted(broker_kind, make_broker):
    """3 sell-able positions → 3 attempted = 3 succeeded, 0 failed."""
    positions = _make_positions(
        ("600000", "浦发银行", 1000, 10.0),
        ("000001", "平安银行", 500, 20.0),
        ("300750", "宁德时代", 100, 200.0),
    )
    b = make_broker(cash=0.0, positions=positions)
    result = b.emergency_liquidate_all(
        confirm_string="EMERGENCY_LIQUIDATE_ACCT123",
    )

    assert isinstance(result, EmergencyResult)
    assert sorted(result.attempted_codes) == ["000001", "300750", "600000"]
    assert sorted(result.succeeded_codes) == ["000001", "300750", "600000"]
    assert result.failed_codes == []
    assert result.duration_seconds >= 0

    if broker_kind == "dryrun":
        # autofill → realized cash present (limit_offset_pct=-0.5% default)
        # 1000×10×0.995 + 500×20×0.995 + 100×200×0.995 = 9950 + 9950 + 19900
        assert result.total_realized_cash == pytest.approx(39_800.0, abs=10.0)
    else:
        # mock async — sells queued, cash not yet realized
        assert result.total_realized_cash == pytest.approx(0.0, abs=0.01)


# ──────────────────────────────────────────────
# Test 3: partial fail — 1 of 3 codes rejected (limit-up)
# ──────────────────────────────────────────────

def test_partial_fail_one_rejects(broker_kind, make_broker):
    """One code rejected by broker → 2 succeeded + 1 failed with reason."""
    positions = _make_positions(
        ("600000", "浦发银行", 1000, 10.0),
        ("000001", "平安银行", 500, 20.0),
        ("300750", "宁德时代", 200, 200.0),
    )
    config = _QMTMockConfig(force_reject={"300750": "涨停跌停拒单"})
    b = make_broker(cash=0.0, positions=positions, config=config)

    result = b.emergency_liquidate_all(
        confirm_string="EMERGENCY_LIQUIDATE_ACCT123",
    )

    assert sorted(result.attempted_codes) == ["000001", "300750", "600000"]
    assert sorted(result.succeeded_codes) == ["000001", "600000"]
    assert len(result.failed_codes) == 1
    code, err = result.failed_codes[0]
    assert code == "300750"
    assert "涨停" in err or "跌停" in err


# ──────────────────────────────────────────────
# Test 4: empty portfolio → no-op
# ──────────────────────────────────────────────

def test_empty_portfolio(broker_kind, make_broker):
    """No positions + no pending orders → empty EmergencyResult."""
    b = make_broker(cash=10_000.0, positions=[])

    result = b.emergency_liquidate_all(
        confirm_string="EMERGENCY_LIQUIDATE_ACCT123",
    )

    assert result.attempted_codes == []
    assert result.succeeded_codes == []
    assert result.failed_codes == []
    assert result.cancelled_order_ids == []
    assert result.total_realized_cash == pytest.approx(0.0)


# ──────────────────────────────────────────────
# Test 5: pending order cancelled before liquidate runs
# ──────────────────────────────────────────────

def test_pending_orders_cancelled_before_liquidate(broker_kind, make_broker):
    """In-flight pending buy is cancelled first, then sell-able positions
    are submitted for liquidation."""
    if broker_kind == "dryrun":
        pytest.skip(
            "DryRunBroker autofills orders so there is no 'pending' state "
            "to test cancel-first ordering; behaviour is covered by "
            "QMTMockBroker variant where async fill is meaningful"
        )

    positions = _make_positions(("600000", "浦发银行", 1000, 10.0))
    b = make_broker(cash=10_000.0, positions=positions)

    buy = b.place_limit_order("000001", "buy", 100, 50.0)
    assert buy.success
    pending_order_id = buy.order_id
    pending_pre = [o for o in b.get_orders() if o.status == "pending"]
    assert any(o.order_id == pending_order_id for o in pending_pre)

    result = b.emergency_liquidate_all(
        confirm_string="EMERGENCY_LIQUIDATE_ACCT123",
    )

    assert pending_order_id in result.cancelled_order_ids
    assert "600000" in result.succeeded_codes

    pending_post_ids = {o.order_id for o in b.get_orders() if o.status == "pending"}
    assert pending_order_id not in pending_post_ids
    cancelled = [
        o for o in b.get_orders()
        if o.status == "cancelled" and o.order_id == pending_order_id
    ]
    assert len(cancelled) == 1
