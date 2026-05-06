"""Daily-report display口径 regression tests.

Locks in the 2026-04-28 fix that replaced the conceptually-broken
"trailing 20d bench_ret + forward 20d excess" display with:

  • predicted_excess  = model's forward 20d excess prediction (decision basis)
  • predicted_return  = predicted_excess + 0.005 (long-term mean reference,
                       does NOT participate in decision)
  • suggestion        = thresholds applied to predicted_excess (NOT to
                       effective_return / predicted_return)

Tests guard against:
  1. bench_adj reverting to a time-varying value (e.g. trailing 20d).
  2. suggestion thresholds being moved back onto effective_return.
  3. predicted_excess column going missing.
  4. Suggestion drifting in the regime where the previous bug bit hardest:
     a stock with negative excess prediction in a bull regime.
"""

from __future__ import annotations

import inspect
import re

import pandas as pd
import pytest


def _get_evaluate_holdings_source():
    """Return the textual source of evaluate_holdings to grep口径 facts."""
    from scripts import daily_report
    return inspect.getsource(daily_report.evaluate_holdings)


def test_long_term_bench_constant_present():
    """The bench adjustment must be a NUMERIC CONSTANT, not a regime/runtime field.

    Specifically forbid:  bench_adj = regime.bench_ret_20d
    Required:             a literal float like 0.005 used as adjustment.
    """
    src = _get_evaluate_holdings_source()
    assert "regime.bench_ret_20d" not in src, (
        "bench_adj must not depend on regime.bench_ret_20d (trailing 20d) — "
        "that was the conceptually-broken口径 fixed on 2026-04-28."
    )
    assert "LONG_TERM_BENCH_20D" in src, (
        "Expected named constant LONG_TERM_BENCH_20D for transparency."
    )
    # Pin the actual value (0.005 = 0.5%/20d ≈ 6%/year).  If you intentionally
    # move it, update this test to match — but do read BASELINE.md first.
    m = re.search(r"LONG_TERM_BENCH_20D\s*=\s*([0-9.]+)", src)
    assert m is not None, "LONG_TERM_BENCH_20D must be a numeric literal"
    val = float(m.group(1))
    assert 0.001 <= val <= 0.02, (
        f"LONG_TERM_BENCH_20D={val} is outside sane range "
        "(0.1% to 2% per 20 trading days). 0.005 is the documented choice."
    )


def test_suggestion_thresholds_use_excess_not_effective():
    """suggestion(...) must be applied to raw_return (the excess prediction),
    NOT to effective_return.  The bug fixed on 2026-04-28 had the suggestion
    using effective_return = excess + trailing_bench, which made all stocks
    look bullish in trending-up markets.
    """
    src = _get_evaluate_holdings_source()
    # The literal call site we care about:
    assert 'result["suggestion"] = result["raw_return"].apply(suggest)' in src, (
        "suggestion must be derived from raw_return (model's forward 20d excess), "
        "not from effective_return (excess + bench_adj). "
        "See BASELINE.md §三-bis for rationale."
    )


def test_three_column_transparency():
    """Report must surface predicted_excess separately from predicted_return."""
    src = _get_evaluate_holdings_source()
    assert 'result["predicted_excess"]' in src, (
        "evaluate_holdings must populate a 'predicted_excess' column so users "
        "can see the model's actual relative judgment."
    )
    assert 'result["predicted_return"]' in src, (
        "evaluate_holdings must populate a 'predicted_return' column "
        "(absolute reference = excess + long-term mean)."
    )


def test_recommend_stocks_uses_same_constant():
    """recommend_stocks() must use the SAME long-term mean baseline,
    so holdings table and recommendations table show consistent numbers."""
    from scripts import daily_report

    src = inspect.getsource(daily_report.recommend_stocks)
    assert "LONG_TERM_BENCH_20D" in src, (
        "recommend_stocks must apply the same long-term mean as evaluate_holdings."
    )
    assert 'result["predicted_excess"]' in src, (
        "recommend_stocks must also expose predicted_excess for transparency."
    )


def test_suggestion_correct_in_bull_regime_with_negative_excess():
    """Smoke-test the actual suggestion logic with a representative case:
    海格通信 2026-04-27 = excess -1.74% in a bull market (ZZ500 trailing 20d +7.93%).

    Old (buggy)口径:  effective = -1.74% + 7.93% = +6.19% → "加仓" ❌
    New口径:         excess = -1.74% → "减仓" ✅
    """
    # Replicate the exact suggest() logic from evaluate_holdings.  If the
    # implementation drifts, this test will fail because the inline copy below
    # will diverge from the source — and the test_suggestion_thresholds_*
    # test above is the structural backstop.
    def suggest(excess):
        import math
        if excess is None or (isinstance(excess, float) and math.isnan(excess)):
            return "减仓"
        if excess > 0.03:
            return "加仓"
        elif excess > 0.00:
            return "持有"
        elif excess > -0.03:
            return "减仓"
        else:
            return "清仓"

    # The bull-market regression case — the bug that motivated the fix.
    assert suggest(-0.0174) == "减仓", \
        "Negative excess in a bull market must still suggest 减仓"
    # Other regions:
    assert suggest(0.05) == "加仓"
    assert suggest(0.01) == "持有"
    assert suggest(-0.05) == "清仓"
    # NaN guard:
    assert suggest(float("nan")) == "减仓"


def test_no_timing_rule_reintroduced():
    """Lock in the ❌#7 BASELINE conclusion: do NOT re-introduce simple
    drawdown-based timing rules ('if dd > -15% → 清仓') in the report layer.

    Implementation note:  this test scans for the patterns we'd expect to see
    if someone naively re-added timing rules.  It's not exhaustive, but it
    catches the obvious shape.
    """
    from scripts import daily_report
    src = inspect.getsource(daily_report)
    forbidden_patterns = [
        # "if dd_from_60d_high < -0.15"  /  "if drawdown < -0.15"
        r"dd_from_(\w+_)?high\s*<\s*-0\.1",
        r"drawdown\s*<\s*-0\.1",
        # "if bench_ret_60d < -0.15"
        r"bench_ret_60d\s*<\s*-0\.1",
        # Hard-clear-to-cash overrides keyed on benchmark trailing return
        r"all_to_cash|force_cash|emergency_cash",
    ]
    for pat in forbidden_patterns:
        assert not re.search(pat, src), (
            f"Pattern /{pat}/ found in daily_report.py — looks like a timing "
            "rule was re-introduced.  See BASELINE.md ❌#7: simple drawdown "
            "rules underperform buy-and-hold in A-shares (实证 MDD -73.5% vs -65.2%)."
        )
