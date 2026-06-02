"""β-1c N=10 case fidelity bot-test (P8-β-1c, docs/dialog/ round 58).

Compares :class:`DryRunBroker` ↔ :class:`QMTMockBroker` under the
Permanent Rule #8 three-constraint fidelity gate (see
``docs/TODO.md::教训（永久规则）::#8`` after this commit):

  (i)   ``nav_diff_pct ≤ 0.1%`` (fidelity_score ≥ 0.999)
  (ii)  ``order_count_diff == 0``  (non-rejected orders only)
  (iii) ``position_shares_diff ≤ 100``  (1 lot tolerance)

N=10 cases (fixed seed=42, broker version qmt_mock 2026-05-25):

| # | name                              | src         |
|---|-----------------------------------|-------------|
| 1 | production_20260521               | production  |
| 2 | production_20260522               | production  |
| 3 | production_20260525               | production  |
| 4 | single_buy_empty_portfolio        | synthetic   |
| 5 | single_sell_full_position         | synthetic   |
| 6 | hold_no_orders                    | synthetic   |
| 7 | small_rebalance_sell_then_buy     | synthetic   |
| 8 | large_rebalance_3sells_3buys      | synthetic   |
| 9 | sell_to_full_cash_close_all       | synthetic   |
|10 | exact_cash_boundary_buy           | synthetic   |

Approach A only (CI-friendly).  Approach B (real Windows QMT-paper,
1 case manual) is β-3 user-action per advisor round 56/58.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mp.execution.dryrun_broker import DryRunBroker
from mp.execution.qmt_broker import Position
from mp.execution.qmt_mock_broker import QMTMockBroker

ROOT = Path(__file__).resolve().parent.parent


# ─── Helpers ──────────────────────────────────────────────────────

def _make_position(code: str, shares: int, price: float) -> Position:
    return Position(
        code=code, name=code,
        shares_total=shares, shares_available=shares,
        avg_cost=price, market_price=price,
        market_value=shares * price,
    )


def _load_production_plan(date: str):
    """Load ``data/orders/orders_<date>.json`` and return
    ``(cash, positions, orders)`` ready to feed a broker."""
    p = ROOT / f"data/orders/orders_{date}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    cash = float(data["account_snapshot"]["cash_available"])
    positions = [
        Position(
            code=h["code"], name=h.get("name", h["code"]),
            shares_total=int(h["shares"]),
            shares_available=int(h["shares"]),
            avg_cost=float(h["avg_cost"]),
            market_price=float(h["avg_cost"]),
            market_value=int(h["shares"]) * float(h["avg_cost"]),
        )
        for h in data["holdings_at_plan_time"]
    ]
    orders = [
        (
            o["code"],
            "buy" if o["cost"] > 0 else "sell",
            int(o["shares"]),
            float(o["limit_price"]),
        )
        for o in data["orders"]
    ]
    return cash, positions, orders


# ─── Case definitions ─────────────────────────────────────────────
# Each tuple: (case_id, name, src, cash, positions, orders).
# Orders are pre-sorted "sells-first, buys-second" so cash flows allow
# subsequent buys to fund from sell proceeds — mirrors the production
# ``execute_orders.run()`` pipeline.

_PRODUCTION_DATES = ["20260521", "20260522", "20260525"]


def _build_cases() -> list[tuple]:
    cases: list[tuple] = []
    for i, date in enumerate(_PRODUCTION_DATES, start=1):
        cash, positions, orders = _load_production_plan(date)
        orders_sorted = sorted(orders, key=lambda o: 0 if o[1] == "sell" else 1)
        cases.append((i, f"production_{date}", "production", cash, positions, orders_sorted))

    cases.extend([
        (4, "single_buy_empty_portfolio", "synthetic",
         50_000.0, [], [("600000", "buy", 1000, 10.0)]),
        (5, "single_sell_full_position", "synthetic",
         0.0, [_make_position("600000", 1000, 10.0)],
         [("600000", "sell", 1000, 10.0)]),
        (6, "hold_no_orders", "synthetic",
         5_000.0, [_make_position("600000", 500, 10.0)],
         []),
        (7, "small_rebalance_sell_then_buy", "synthetic",
         0.0, [_make_position("600000", 500, 10.0)],
         [("600000", "sell", 500, 10.0), ("000001", "buy", 200, 20.0)]),
        (8, "large_rebalance_3sells_3buys", "synthetic", 0.0,
         [_make_position("600000", 1000, 10.0),
          _make_position("000001", 500, 20.0),
          _make_position("300750", 100, 200.0)],
         [("600000", "sell", 1000, 10.0),
          ("000001", "sell", 500, 20.0),
          ("300750", "sell", 100, 200.0),
          ("600100", "buy", 1000, 15.0),
          ("000002", "buy", 500, 25.0),
          ("300800", "buy", 100, 100.0)]),
        (9, "sell_to_full_cash_close_all", "synthetic", 0.0,
         [_make_position("600000", 1000, 10.0),
          _make_position("000001", 500, 20.0)],
         [("600000", "sell", 1000, 10.0),
          ("000001", "sell", 500, 20.0)]),
        (10, "exact_cash_boundary_buy", "synthetic",
         10_000.0, [], [("600000", "buy", 1000, 10.0)]),
    ])
    return cases


CASES = _build_cases()


# ─── Execution + scoring ──────────────────────────────────────────

def _run_plan(broker, orders) -> tuple[float, int, dict[str, int]]:
    """Submit each order; settle pending fills after each (mirrors the
    real QMT request → fill_wait → next-request flow).

    Returns ``(nav, n_orders_non_rejected, position_shares)`` where:
    - ``nav`` = ``cash_available + cash_frozen + market_value``
    - ``n_orders_non_rejected`` excludes status='rejected' so accept-side
      apples-to-apples between DryRun (silently rejects) and Mock
      (records rejected as audit trail)
    - ``position_shares`` = ``{code: shares_total}`` for every non-zero
      end-state position
    """
    broker.connect()
    for code, action, shares, limit in orders:
        broker.place_limit_order(code, action, shares, limit)
        if hasattr(broker, "process_pending_orders"):
            broker.process_pending_orders()
    if hasattr(broker, "process_pending_orders"):
        for _ in range(30):  # safety bound for any partial-fill plans
            broker.process_pending_orders()
            still = any(
                o.status in ("pending", "partial")
                for o in broker.get_orders()
            )
            if not still:
                break
    info = broker.get_account_info()
    nav = info.cash_available + info.cash_frozen + info.market_value
    n_orders = len([o for o in broker.get_orders() if o.status != "rejected"])
    pos_shares = {p.code: p.shares_total for p in broker.get_positions()}
    return nav, n_orders, pos_shares


def fidelity_score(
    dr_state: tuple[float, int, dict[str, int]],
    qmt_state: tuple[float, int, dict[str, int]],
) -> dict:
    """Compute the 3 Rule #8 fidelity metrics from two broker end-states."""
    nav_dr, n_dr, pos_dr = dr_state
    nav_qmt, n_qmt, pos_qmt = qmt_state
    avg_nav = (nav_dr + nav_qmt) / 2 if (nav_dr + nav_qmt) > 0 else 1.0
    nav_diff_pct = abs(nav_dr - nav_qmt) / avg_nav
    order_count_diff = abs(n_dr - n_qmt)
    all_codes = set(pos_dr) | set(pos_qmt)
    pos_diffs = [
        abs(pos_dr.get(c, 0) - pos_qmt.get(c, 0))
        for c in all_codes
    ]
    position_shares_diff = max(pos_diffs) if pos_diffs else 0
    return {
        "nav_diff_pct": nav_diff_pct,
        "order_count_diff": order_count_diff,
        "position_shares_diff": position_shares_diff,
    }


# ─── Parametrized fidelity test ───────────────────────────────────

@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[f"{c[0]:02d}_{c[1]}" for c in CASES],
)
def test_fidelity_case(case):
    """Rule #8 three-constraint per case (see ``docs/TODO.md::教训::#8``)."""
    case_id, name, src, cash, positions, orders = case

    dr = DryRunBroker(
        cash=cash, positions=copy.deepcopy(positions), autofill=True,
    )
    qmt = QMTMockBroker(
        cash=cash, positions=copy.deepcopy(positions),
    )

    dr_state = _run_plan(dr, orders)
    qmt_state = _run_plan(qmt, orders)

    score = fidelity_score(dr_state, qmt_state)
    rule_8_pass = (
        score["nav_diff_pct"] <= 0.001
        and score["order_count_diff"] == 0
        and score["position_shares_diff"] <= 100
    )

    # Per-case report row — visible via ``pytest -s`` or report capture
    print(
        f"\n  case {case_id:2d} {name:35s} {src:11s} "
        f"nav_diff={score['nav_diff_pct']:.5f}  "
        f"order_diff={score['order_count_diff']}  "
        f"pos_diff={score['position_shares_diff']}  "
        f"{'PASS' if rule_8_pass else 'FAIL'}"
    )

    # Rule #8 three-constraint assertions (each individually)
    assert score["nav_diff_pct"] <= 0.001, (
        f"[case {case_id} {name}] (i) nav_diff_pct={score['nav_diff_pct']:.6f} > 0.001 "
        f"(dr_nav={dr_state[0]:.2f}, qmt_nav={qmt_state[0]:.2f})"
    )
    assert score["order_count_diff"] == 0, (
        f"[case {case_id} {name}] (ii) order_count_diff={score['order_count_diff']} "
        f"(dr={dr_state[1]}, qmt={qmt_state[1]})"
    )
    assert score["position_shares_diff"] <= 100, (
        f"[case {case_id} {name}] (iii) position_shares_diff={score['position_shares_diff']} > 100 "
        f"(dr={dr_state[2]}, qmt={qmt_state[2]})"
    )


# ─── Fidelity-score function smoke test ───────────────────────────

def test_fidelity_score_zero_diff_for_identical_states():
    """Sanity: identical broker states → all 3 metrics = 0."""
    state = (100_000.0, 5, {"A": 1000, "B": 500})
    score = fidelity_score(state, state)
    assert score["nav_diff_pct"] == pytest.approx(0.0)
    assert score["order_count_diff"] == 0
    assert score["position_shares_diff"] == 0


def test_fidelity_score_detects_nav_drift():
    s1 = (100_000.0, 5, {"A": 1000})
    s2 = (101_000.0, 5, {"A": 1000})
    score = fidelity_score(s1, s2)
    # 1000 diff / 100500 avg ≈ 0.995%
    assert score["nav_diff_pct"] == pytest.approx(0.00995, abs=0.0001)
    assert score["order_count_diff"] == 0
    assert score["position_shares_diff"] == 0


def test_fidelity_score_detects_position_drift():
    s1 = (100.0, 1, {"A": 1000})
    s2 = (100.0, 1, {"A": 900})
    score = fidelity_score(s1, s2)
    assert score["position_shares_diff"] == 100
