"""Regression tests for the 2026-05-07 fix: intraday volume/amount must be
extrapolated to estimated full-day before injection.

Without this scaling, midday inference was systematically penalizing
low-turnover stocks (utilities, banks, defensives) because volume-based
technical features (amount_ratio, volume_trend, amihud_illiq, mfi_14, …)
were comparing half-day values against a 20-day rolling mean of full-day
values, producing artificially low scores.
"""

from __future__ import annotations

from datetime import datetime

import pytest


# ── _trading_minutes_elapsed ────────────────────────────────────────────────

def _e(h, m):
    from scripts.daily_report import _trading_minutes_elapsed
    return _trading_minutes_elapsed(datetime(2026, 5, 7, h, m))


def test_pre_market_zero():
    assert _e(8, 0) == 0
    assert _e(9, 29) == 0


def test_morning_session_progresses():
    assert _e(9, 30) == 0
    assert _e(10, 30) == 60
    assert _e(11, 30) == 120


def test_lunch_break_holds_at_120():
    """During 11:30-13:00 the morning session is complete; afternoon hasn't started."""
    assert _e(11, 30) == 120
    assert _e(12, 0) == 120
    assert _e(12, 30) == 120
    assert _e(12, 59) == 120


def test_afternoon_session_progresses():
    assert _e(13, 0) == 120
    assert _e(13, 30) == 150
    assert _e(14, 0) == 180
    assert _e(15, 0) == 240


def test_post_close_caps_at_240():
    assert _e(15, 30) == 240
    assert _e(18, 0) == 240
    assert _e(23, 59) == 240


# ── _full_day_session_scale ─────────────────────────────────────────────────

def _s(h, m):
    from scripts.daily_report import _full_day_session_scale
    return _full_day_session_scale(datetime(2026, 5, 7, h, m))


def test_post_close_no_scaling():
    """After 15:00, volume is full-day already — no scaling."""
    assert _s(15, 30) == pytest.approx(1.0)
    assert _s(18, 0) == pytest.approx(1.0)


def test_midday_doubles():
    """At 12:00 (midday lunch), only morning session traded → scale ≈ 2."""
    assert _s(12, 0) == pytest.approx(2.0, rel=0.01)


def test_morning_close_doubles():
    """11:30 = 50% of session → scale 2.0."""
    assert _s(11, 30) == pytest.approx(2.0, rel=0.01)


def test_two_pm_scales_by_four_thirds():
    """At 14:00, 75% of session has elapsed → scale ≈ 1.33."""
    assert _s(14, 0) == pytest.approx(240.0 / 180.0, rel=0.01)


def test_close_to_close_scale_approaches_one():
    """At 14:55, ~98% of session → scale ≈ 1.02."""
    assert _s(14, 55) == pytest.approx(240.0 / 235.0, rel=0.01)


def test_pre_market_returns_one():
    """Before market open volume is 0; scaling is irrelevant — return 1.0."""
    assert _s(8, 0) == pytest.approx(1.0)
    assert _s(9, 29) == pytest.approx(1.0)


def test_lower_bound_clamp():
    """A 9:35 fetch (5min in) shouldn't produce a 48× blowup — clamp at 5×."""
    s = _s(9, 35)
    assert s <= 5.0 + 1e-6, f"Expected clamp ≤ 5, got {s}"


# ── End-to-end: scale is applied to volume and amount ──────────────────────

def test_realtime_prices_scales_volume_and_amount(monkeypatch):
    """When _fetch_realtime_prices runs at midday, returned volume/amount
    must be ≈2× the raw Sina half-day values."""
    from scripts import daily_report as dr

    # Pin "now" to 12:00 so scale = 2.0
    fixed_now = datetime(2026, 5, 7, 12, 0)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(dr, "datetime", _FixedDatetime)

    # Mock Sina HTTP response.
    # ⚠️ 2026-05-11 unit correction: hq.sinajs.cn returns fields[8] in
    # SHARES, not 手.  Verified by cross-checking 002385 EOD volume on
    # 2026-05-11: DB volume = 70,455,338 shares ≡ sina fields[8].
    # The prior *100 conversion in _fetch_realtime_prices inflated volume
    # 100× — caught when midday model predictions for 大北农/康泰生物
    # diverged from EOD predictions by ~2pp even though intraday data
    # matched DB EOD almost exactly.
    #
    # Mock: raw half-day volume = 100,000 SHARES, amount = 5,000,000 元.
    # amount/volume = 50元/股 (unrealistic but irrelevant — we test scaling).
    sample_response = (
        'var hq_str_sh603707="健友股份,30.0,29.5,30.5,31.0,29.8,30.4,30.5,'
        '100000,5000000,0,0,0,0,0,0,2026-05-07,11:30:00,00,";'
    )

    class _MockResp:
        text = sample_response

    def _mock_get(*a, **kw):
        return _MockResp()

    import httpx
    monkeypatch.setattr(httpx, "get", _mock_get)

    result = dr._fetch_realtime_prices(["603707"])

    assert "603707" in result, f"got {result!r}"
    r = result["603707"]
    # Raw: 100,000 shares × scale 2.0 = 200,000 shares (NO 手→股 conversion).
    # Amount: 5,000,000 × scale 2.0 = 10,000,000.
    assert r["volume"] == pytest.approx(200_000, rel=0.05), \
        f"volume not scaled correctly: {r['volume']}"
    assert r["amount"] == pytest.approx(10_000_000, rel=0.05), \
        f"amount not scaled correctly: {r['amount']}"
    # Sanity: amount/volume ≈ 50 (the mock's implied unit price)
    # If volume is wrongly ×100, ratio collapses to 0.5 — caught here.
    ratio = r["amount"] / r["volume"]
    assert 10 < ratio < 200, \
        f"amount/volume ratio {ratio:.2f} suggests volume unit bug"
    # Price/high/low must NOT be scaled
    assert r["price"] == pytest.approx(30.5)
    assert r["high"] == pytest.approx(31.0)
    assert r["low"] == pytest.approx(29.8)
