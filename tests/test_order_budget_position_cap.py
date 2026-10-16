"""Regression: the buy budget must respect BOTH cash and the position cap.

2026-10-16 incident. The live account drifted from ~85% invested down to 29%
over roughly three weeks: sells executed every morning, buys never did. Every
daily plan carried alerts reading "受总仓位上限约束，跳过本笔加仓（当前总仓
90.9% > 上限 85%）" even though real holdings were only ~36% of total assets.

Root cause was two consecutive steps in ``generate_order_list`` disagreeing:

  1. Budget reconciliation sized buys against CASH only (``cash × 0.95``).
     On a cash-heavy book (¥107k cash vs ¥60k holdings) that permitted ~¥102k
     of buys.
  2. The total-position step then projected holdings + those buys against
     ``target_position_pct × total_assets`` and, finding the projection over
     the cap, dropped every buy outright rather than trimming them.

So the more cash the account held, the more certainly its buys were cancelled,
which freed yet more cash the next day. The "90.9%" in the alert was the
projection INCLUDING the oversized buys, never the actual book.

The fix derives one budget from both constraints, so buys are scaled to fit
under the cap instead of being discarded.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _recs(codes_excess: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"code": c, "name": c, "predicted_excess": ex}
         for c, ex in codes_excess.items()]
    )


def _scored(codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"code": c, "name": c, "_rank": i + 1} for i, c in enumerate(codes)]
    )


@pytest.fixture()
def cash_heavy_account():
    """Mirrors the live book on 2026-10-15: 36% invested, 64% cash."""
    return {
        "total_assets": 168403.38,
        "cash_available": 107653.38,
        "market_value": 60750.0,
        "target_position_pct": 0.85,
        "hard_single_weight": 0.40,
        "rebalance_tolerance": 0.02,
    }


def test_buys_are_emitted_when_underinvested(cash_heavy_account, monkeypatch):
    """The whole point: an underinvested book must still buy.

    Before the fix this returned zero buys and a pile of "总仓位上限" alerts.
    """
    from scripts import daily_report

    closes = {"000001": 10.0, "000002": 20.0, "000003": 25.0}
    monkeypatch.setattr(daily_report, "_latest_closes", lambda codes: closes)

    holdings = [{"code": "000003", "shares": 400, "avg_cost": 24.0, "name": "000003"}]
    recs = _recs({"000001": 5.0, "000002": 4.0})
    orders, alerts = daily_report.generate_order_list(
        holdings, cash_heavy_account, recs, _scored(["000001", "000002", "000003"]),
    )

    buys = [o for o in orders if o["cost"] > 0]
    assert buys, (
        "underinvested account emitted no buy orders; alerts=" + repr(alerts)
    )
    assert not any("总仓位上限" in a for a in alerts), (
        "position-cap alert fired on a 36%-invested book: " + repr(alerts)
    )


def test_buys_respect_the_position_cap(cash_heavy_account, monkeypatch):
    """Buys may use the cash, but only up to the target position cap."""
    from scripts import daily_report

    closes = {"000001": 10.0, "000002": 20.0, "000003": 25.0}
    monkeypatch.setattr(daily_report, "_latest_closes", lambda codes: closes)

    holdings = [{"code": "000003", "shares": 400, "avg_cost": 24.0, "name": "000003"}]
    recs = _recs({"000001": 5.0, "000002": 4.0})
    orders, _ = daily_report.generate_order_list(
        holdings, cash_heavy_account, recs, _scored(["000001", "000002", "000003"]),
    )

    acct = cash_heavy_account
    held_after = 400 * closes["000003"]
    for o in orders:
        if o["code"] == "000003" and o["cost"] < 0:
            held_after -= o["shares"] * o["limit_price"]
    bought = sum(o["cost"] for o in orders if o["cost"] > 0)
    projected = held_after + bought
    cap = acct["total_assets"] * acct["target_position_pct"]

    assert projected <= cap * 1.01, (
        f"projected position {projected:,.0f} exceeds cap {cap:,.0f}"
    )


def test_buys_never_exceed_available_cash(cash_heavy_account, monkeypatch):
    """The cash constraint must still hold after the fix."""
    from scripts import daily_report

    closes = {"000001": 10.0, "000002": 20.0, "000003": 25.0}
    monkeypatch.setattr(daily_report, "_latest_closes", lambda codes: closes)

    poor = dict(cash_heavy_account)
    poor["cash_available"] = 3000.0          # far less than the cap allows

    holdings = [{"code": "000003", "shares": 400, "avg_cost": 24.0, "name": "000003"}]
    recs = _recs({"000001": 5.0, "000002": 4.0})
    orders, _ = daily_report.generate_order_list(
        holdings, poor, recs, _scored(["000001", "000002", "000003"]),
    )

    proceeds = sum(-o["cost"] for o in orders if o["cost"] < 0)
    spent = sum(o["cost"] for o in orders if o["cost"] > 0)
    assert spent <= (poor["cash_available"] + proceeds) * 0.95 + 1e-6, (
        f"spent {spent:,.0f} exceeds cash budget"
    )


def test_cap_alert_still_fires_when_genuinely_overinvested(monkeypatch):
    """Guard the other direction: a real over-allocation must not buy more."""
    from scripts import daily_report

    closes = {"000001": 10.0, "000003": 25.0}
    monkeypatch.setattr(daily_report, "_latest_closes", lambda codes: closes)

    over = {
        "total_assets": 100000.0,
        "cash_available": 2000.0,
        "market_value": 95000.0,
        "target_position_pct": 0.85,
        "hard_single_weight": 0.40,
        "rebalance_tolerance": 0.02,
    }
    holdings = [{"code": "000003", "shares": 3800, "avg_cost": 25.0, "name": "000003"}]
    recs = _recs({"000001": 5.0})
    orders, _ = daily_report.generate_order_list(
        holdings, over, recs, _scored(["000001", "000003"]),
    )

    # Holdings alone (95% of assets) already breach the 85% cap, so the
    # correct behaviour is to add NO new exposure; sells may still trim it.
    bought = sum(o["cost"] for o in orders if o["cost"] > 0)
    assert bought == 0, f"added {bought:,.0f} of buys to an already-overinvested book"
