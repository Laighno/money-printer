"""P6-X2 — tests for mp/data/trading_calendar.py (docs/dialog/ round 47).

Covers:
- weekday short-circuit (no network needed)
- akshare-mocked happy path: Mon-Fri inclusive count == 5
- CNY-spanning interval: 2026-02-15 → 2026-02-25, mocked calendar with
  holiday gap, count should match the gap-aware result
- akshare exception path does NOT raise (caller relies on this for
  fail-soft behavior — heartbeat falls back to wall-clock if so)
"""
from __future__ import annotations

import pandas as pd
import pytest

from mp.data import trading_calendar as tc


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with a fresh cache so akshare mocks compose
    cleanly (the module caches the calendar after first successful
    fetch)."""
    tc.reset_cache()
    yield
    tc.reset_cache()


# ───────────────────────────────────────────────────────────────────────
# is_trading_day weekday short-circuit (no network involved)
# ───────────────────────────────────────────────────────────────────────

def test_is_trading_day_saturday_short_circuits(monkeypatch):
    """Weekday >= 5 returns False without ever fetching the calendar."""
    # Trigger an explosion if akshare is touched at all
    def boom(*a, **kw):
        raise AssertionError("akshare should NOT be called for Sat short-circuit")
    monkeypatch.setattr(tc, "_fetch_trading_dates", boom)
    monkeypatch.setattr(tc, "_zz500_eod_probe", boom)

    sat = pd.Timestamp("2026-05-23")   # Saturday
    assert sat.weekday() == 5, "fixture sanity"
    assert tc.is_trading_day(sat) is False


def test_is_trading_day_sunday_short_circuits(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("akshare should NOT be called for Sun short-circuit")
    monkeypatch.setattr(tc, "_fetch_trading_dates", boom)
    monkeypatch.setattr(tc, "_zz500_eod_probe", boom)

    sun = pd.Timestamp("2026-05-24")
    assert sun.weekday() == 6
    assert tc.is_trading_day(sun) is False


# ───────────────────────────────────────────────────────────────────────
# is_trading_day calendar-path (akshare mocked)
# ───────────────────────────────────────────────────────────────────────

def test_is_trading_day_weekday_present_in_calendar(monkeypatch):
    """Mon present in the trading calendar → True."""
    cal_dates = {
        pd.Timestamp("2026-05-18"),   # Mon
        pd.Timestamp("2026-05-19"),   # Tue
        pd.Timestamp("2026-05-20"),   # Wed
        pd.Timestamp("2026-05-21"),   # Thu
        pd.Timestamp("2026-05-22"),   # Fri
    }
    monkeypatch.setattr(tc, "_fetch_trading_dates", lambda: cal_dates)
    assert tc.is_trading_day(pd.Timestamp("2026-05-18")) is True


def test_is_trading_day_weekday_absent_from_calendar(monkeypatch):
    """Weekday not in calendar (e.g. holiday Monday) → False."""
    cal_dates = {pd.Timestamp("2026-05-19")}   # only Tue
    monkeypatch.setattr(tc, "_fetch_trading_dates", lambda: cal_dates)
    assert tc.is_trading_day(pd.Timestamp("2026-05-18")) is False


# ───────────────────────────────────────────────────────────────────────
# trading_days_between
# ───────────────────────────────────────────────────────────────────────

def test_trading_days_between_mon_to_fri(monkeypatch):
    """Closed interval Mon → Fri (no holidays) → 5."""
    cal_dates = {
        pd.Timestamp("2026-05-18"),
        pd.Timestamp("2026-05-19"),
        pd.Timestamp("2026-05-20"),
        pd.Timestamp("2026-05-21"),
        pd.Timestamp("2026-05-22"),
        # Add some other dates outside the range — should not be counted
        pd.Timestamp("2026-05-15"),
        pd.Timestamp("2026-05-25"),
    }
    monkeypatch.setattr(tc, "_fetch_trading_dates", lambda: cal_dates)
    n = tc.trading_days_between(pd.Timestamp("2026-05-18"),
                                pd.Timestamp("2026-05-22"))
    assert n == 5, f"expected 5 trading days Mon-Fri, got {n}"


def test_trading_days_between_spans_cny(monkeypatch):
    """2026-02-15 → 2026-02-25 spans (mocked) CNY week. Should count
    only the trading days, not the holidays."""
    # Mock: 2026-02-15 (Sun) ─ 2026-02-25 (Wed)
    #       Weekdays: Mon 16, Tue 17, Wed 18, Thu 19, Fri 20,
    #                 Mon 23, Tue 24, Wed 25
    #       Assume CNY closes the market 16-20 (mocked example).
    #       Trading days in interval: 23, 24, 25 → 3
    cal_dates = {
        pd.Timestamp("2026-02-23"),
        pd.Timestamp("2026-02-24"),
        pd.Timestamp("2026-02-25"),
        # Earlier / later context so the cache looks "real"
        pd.Timestamp("2026-02-13"),
        pd.Timestamp("2026-02-26"),
    }
    monkeypatch.setattr(tc, "_fetch_trading_dates", lambda: cal_dates)
    n = tc.trading_days_between(pd.Timestamp("2026-02-15"),
                                pd.Timestamp("2026-02-25"))
    assert n == 3
    # Sanity: calendar-day span was 11 days
    cal_span_days = (pd.Timestamp("2026-02-25") - pd.Timestamp("2026-02-15")).days + 1
    assert cal_span_days == 11
    assert n < cal_span_days, "trading-day count must be < calendar-day count when CNY in interval"


def test_trading_days_between_end_before_start(monkeypatch):
    """end < start → 0, never raise."""
    monkeypatch.setattr(tc, "_fetch_trading_dates", lambda: {pd.Timestamp("2026-05-20")})
    n = tc.trading_days_between(pd.Timestamp("2026-05-25"),
                                pd.Timestamp("2026-05-20"))
    assert n == 0


def test_trading_days_between_same_day_trading(monkeypatch):
    """Same day, that day is trading → 1 (closed interval)."""
    monkeypatch.setattr(
        tc, "_fetch_trading_dates",
        lambda: {pd.Timestamp("2026-05-22")},
    )
    n = tc.trading_days_between(pd.Timestamp("2026-05-22"),
                                pd.Timestamp("2026-05-22"))
    assert n == 1


# ───────────────────────────────────────────────────────────────────────
# Fallback when akshare unavailable
# ───────────────────────────────────────────────────────────────────────

def test_trading_days_between_falls_back_to_weekday_count(monkeypatch):
    """If the calendar API is down, trading_days_between falls back to
    weekday count (Mon-Fri) — over-counts holidays but never raises."""
    monkeypatch.setattr(tc, "_fetch_trading_dates", lambda: None)
    # Mon 2026-05-18 → Wed 2026-05-27: 8 weekdays
    n = tc.trading_days_between(pd.Timestamp("2026-05-18"),
                                pd.Timestamp("2026-05-27"))
    # Mon-Fri week 1 (5) + Mon-Wed week 2 (3) = 8
    assert n == 8


def test_fetch_dates_swallows_akshare_exception(monkeypatch):
    """If akshare.tool_trade_date_hist_sina raises, _fetch_trading_dates
    returns None (does not propagate)."""
    import sys, types
    fake = types.ModuleType("akshare")
    def boom():
        raise RuntimeError("network down (test)")
    fake.tool_trade_date_hist_sina = boom
    monkeypatch.setitem(sys.modules, "akshare", fake)

    # Should NOT raise
    dates = tc._fetch_trading_dates()
    assert dates is None
    assert tc.calendar_available() is False


def test_is_trading_day_when_both_apis_fail(monkeypatch):
    """Calendar fails AND ZZ500 probe fails → returns False (matches
    original paper_trade fallback behavior)."""
    monkeypatch.setattr(tc, "_fetch_trading_dates", lambda: None)
    monkeypatch.setattr(tc, "_zz500_eod_probe", lambda today: False)
    # Use a weekday so we don't get the short-circuit
    mon = pd.Timestamp("2026-05-18")
    assert tc.is_trading_day(mon) is False


# ───────────────────────────────────────────────────────────────────────
# Caching behavior
# ───────────────────────────────────────────────────────────────────────

def test_fetch_dates_is_cached(monkeypatch):
    """_fetch_trading_dates should only hit akshare once per process."""
    import sys, types
    calls = []
    fake = types.ModuleType("akshare")
    def fetch():
        calls.append(1)
        return pd.DataFrame({"trade_date": ["2026-05-22"]})
    fake.tool_trade_date_hist_sina = fetch
    monkeypatch.setitem(sys.modules, "akshare", fake)

    tc._fetch_trading_dates()
    tc._fetch_trading_dates()
    tc._fetch_trading_dates()
    assert len(calls) == 1
