"""A-share trading-day utilities, centralized.

Lifted from ``scripts/paper_trade.py:646`` in P6-X2 (docs/dialog/ round 47)
because more than one caller needs trading-day awareness — most recently
``scripts/monitor/weekly_heartbeat.py`` which needs to count trading days
between two timestamps to decide YELLOW/RED alert levels (vs wall-clock,
which fires false-positives across long weekends / CNY / National Day).

Implementation history (preserved deliberately so future readers
understand why this is more than a weekday check)
-------------------------------------------------------------------
Old approach (probing ZZ500 EOD bar): too strict — it required EOD data
to already be published, which doesn't happen until ~30-60 min after
market close.  A 16:00 cron firing 1 min after close would see no data
yet and incorrectly mark the day as non-trading (this exact bug skipped
paper_trade on 2026-04-30).

Current approach: use the canonical akshare ``tool_trade_date_hist_sina``
calendar.  Module-level cache so repeated calls within one process don't
re-fetch (akshare hits Sina each time, ~1-3 s per call).  Falls back to
ZZ500 EOD probe if the calendar API is unavailable, and ultimately to a
weekday short-circuit only — the fallback chain is conservative on
purpose: when we don't *know* whether today is a trading day, we'd
rather say "yes, run" (and let downstream skip silently if there's no
data) than fail-silent on an actual trading day.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger


# Module-level cache for the akshare calendar fetch (idempotent within a
# single Python process).  Cleared by tests via ``reset_cache()``.
_TRADING_DATES: Optional[set[pd.Timestamp]] = None
_CALENDAR_AVAILABLE: Optional[bool] = None   # tri-state: None=unfetched


def reset_cache() -> None:
    """Clear the module-level calendar cache (test hook)."""
    global _TRADING_DATES, _CALENDAR_AVAILABLE
    _TRADING_DATES = None
    _CALENDAR_AVAILABLE = None


def _fetch_trading_dates() -> Optional[set[pd.Timestamp]]:
    """Fetch the A-share trading calendar from akshare (cached after first
    success).  Returns ``None`` if fetch fails — caller must handle.
    """
    global _TRADING_DATES, _CALENDAR_AVAILABLE
    if _TRADING_DATES is not None:
        return _TRADING_DATES
    if _CALENDAR_AVAILABLE is False:
        # Don't retry within the same process — fetch already known to fail
        return None
    try:
        import akshare as ak
        cal = ak.tool_trade_date_hist_sina()
        cal["trade_date"] = pd.to_datetime(cal["trade_date"])
        _TRADING_DATES = set(cal["trade_date"])
        _CALENDAR_AVAILABLE = True
        return _TRADING_DATES
    except Exception as e:
        logger.warning("Trade calendar fetch failed ({}); calendar-aware "
                       "logic will fall back", e)
        _CALENDAR_AVAILABLE = False
        return None


def _zz500_eod_probe(today: pd.Timestamp) -> bool:
    """Last-resort fallback: did sh000905 publish an EOD bar for ``today``?

    Used only when the trading calendar API is also unavailable.  Note
    this is too strict for real-time use (won't say "yes" until ~30-60 min
    after close), which is why it's the fallback rather than the primary
    method — see the 2026-04-30 history note at module top.

    Returns ``False`` on any error: that matches the original
    paper_trade.is_trading_day fallback behavior (failed probe → not a
    trading day → skip).  ``trading_days_between`` has its own separate
    "calendar unavailable → degraded count" path that doesn't go through
    this probe.
    """
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol="sh000905")
        df["date"] = pd.to_datetime(df["date"])
        # Also require a previous day's bar — matches the original
        # get_zz500_close (it returned None if idx==0) so a brand-new
        # universe doesn't accidentally trigger.
        dates = sorted(set(pd.to_datetime(df["date"])))
        today_norm = today.normalize()
        if today_norm not in dates:
            return False
        idx = dates.index(today_norm)
        return idx > 0
    except Exception as e:
        logger.warning("ZZ500 EOD probe also failed ({}); treating {} as "
                       "non-trading (matches original paper_trade behavior)",
                       e, today.date())
        return False


def is_trading_day(today: pd.Timestamp) -> bool:
    """Check whether ``today`` is an A-share trading day.

    Primary signal: akshare ``tool_trade_date_hist_sina`` (the official
    SSE calendar).  Falls back to ZZ500 EOD bar presence if calendar API
    is down.  Both fallbacks ultimately conservative — see module-level
    docstring for why.
    """
    if today.weekday() >= 5:   # Sat=5 Sun=6
        return False
    dates = _fetch_trading_dates()
    if dates is not None:
        return today.normalize() in dates
    return _zz500_eod_probe(today)


def trading_days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Count A-share trading days in the closed interval [start, end].

    Examples
    --------
    >>> # Mon 2026-05-18 → Fri 2026-05-22 (no holidays)
    >>> trading_days_between(pd.Timestamp("2026-05-18"),
    ...                      pd.Timestamp("2026-05-22"))
    5

    Fallback behavior
    -----------------
    If the akshare calendar is unreachable, falls back to weekday count
    (Mon–Fri ∈ interval). This is conservative for typical use — short
    weekends are correctly counted, but holidays (CNY, National Day) will
    be over-counted, which makes downstream "days since last cron run"
    estimates higher than reality → more likely to trigger alerts, not
    fewer.  ``calendar_available()`` exposes which path was used.
    """
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    if e < s:
        return 0
    dates = _fetch_trading_dates()
    if dates is not None:
        return sum(1 for d in dates if s <= d <= e)
    # Fallback: weekday count
    n_days = (e - s).days + 1
    return sum(1 for i in range(n_days)
               if (s + pd.Timedelta(days=i)).weekday() < 5)


def calendar_available() -> bool:
    """True iff the akshare calendar has been successfully fetched (or is
    cached). Use to detect when trading_days_between fell back to weekday
    count — caller may want to log the degraded state in alerts.
    """
    if _CALENDAR_AVAILABLE is None:
        _fetch_trading_dates()
    return _CALENDAR_AVAILABLE is True
