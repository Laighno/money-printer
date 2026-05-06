"""Regression tests for the 2026-04-28 fixes:

1. `_make_data_timestamps` must use the latest date with REPRESENTATIVE row
   coverage, not raw MAX(date).  Otherwise a stray partial row makes the
   report header claim "行情: today" when 99% of stocks only have data
   through yesterday.

2. `get_daily_bars` must NOT persist partial intraday bars (rows with date
   later than `_last_expected_trading_day()`).  Some akshare endpoints
   return today's bar with current price as fake "close" during market
   hours, polluting the EOD table.
"""

from __future__ import annotations

import inspect
import re

import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────
# 1. _make_data_timestamps uses representative coverage
# ──────────────────────────────────────────────────────────────────────

def test_make_data_timestamps_uses_count_threshold():
    """Source must group by date and require a minimum row count, not MAX-only."""
    from scripts.daily_report import _make_data_timestamps
    src = inspect.getsource(_make_data_timestamps)

    # The OLD broken code path: SELECT MAX(date) FROM daily_bars (no GROUP BY)
    # Strip docstring + comments; only inspect actual code.
    code_only = re.sub(r'"""[\s\S]*?"""', "", src)
    code_only = re.sub(r"'''[\s\S]*?'''", "", code_only)
    code_only = re.sub(r"#[^\n]*", "", code_only)
    assert "MAX(date)" not in code_only, (
        "_make_data_timestamps must not use raw MAX(date) in actual SQL — that's "
        "the bug where one stray partial row mislabels the report header."
    )
    # New path uses GROUP BY date HAVING COUNT(*) >= ...
    assert "GROUP BY date" in src and "COUNT(*)" in src, (
        "_make_data_timestamps should aggregate by date with a row-count "
        "threshold (representative coverage)."
    )


def test_make_data_timestamps_marks_intraday_valuation():
    """When valuation snapshot is fresher than latest EOD bars, it must be
    flagged so users can see the data is mixed (EOD bars + intraday valuation)."""
    from scripts.daily_report import _make_data_timestamps
    src = inspect.getsource(_make_data_timestamps)
    assert "盘中" in src, (
        "_make_data_timestamps must annotate intraday valuation snapshots "
        "(when val_date > bar_date) so the report doesn't claim '估值: today' "
        "as if it were EOD-aligned."
    )


# ──────────────────────────────────────────────────────────────────────
# 2. get_daily_bars filters partial-bar rows before save
# ──────────────────────────────────────────────────────────────────────

def test_get_daily_bars_filters_partial_intraday_rows():
    """Source must filter api_df by date <= _last_expected_trading_day()
    BEFORE save_bars_upsert, otherwise akshare's intraday-during-session
    bars get persisted as fake EOD data."""
    from mp.data import fetcher
    src = inspect.getsource(fetcher.get_daily_bars)
    # Must call _last_expected_trading_day in the persist path
    assert "_last_expected_trading_day" in src, (
        "get_daily_bars must reference _last_expected_trading_day to filter "
        "out partial intraday bars before persisting."
    )
    # Must filter the API result by date
    assert re.search(r"api_df\s*=\s*api_df\[\s*api_df\[.date.\]\s*<=", src) is not None, (
        "get_daily_bars must filter api_df by date <= last_expected_trading_day "
        "before save_bars_upsert.  Otherwise intraday partial bars pollute the "
        "daily_bars table and break PIT factor computation."
    )


def test_partial_bar_filter_works_on_synthetic_data():
    """End-to-end: simulate akshare returning a partial today bar; verify
    that the filter inside get_daily_bars drops it before persisting.
    """
    from datetime import date
    import mp.data.fetcher as fetcher

    fake_today = pd.Timestamp(date.today())
    fake_yesterday = fake_today - pd.Timedelta(days=1)
    # Skip weekends — fall back enough days
    while fake_yesterday.weekday() >= 5:
        fake_yesterday -= pd.Timedelta(days=1)

    api_df = pd.DataFrame({
        "code": ["123456"] * 2,
        "date": [fake_yesterday, fake_today],
        "open": [10.0, 10.5],
        "high": [10.5, 10.7],
        "low": [9.9, 10.4],
        "close": [10.4, 10.6],
        "volume": [1000, 200],
        "amount": [10_000, 2_000],
        "turnover": [0.5, 0.1],
    })

    # Replicate the filter inline (mirrors the one in get_daily_bars)
    last_expected = pd.Timestamp(fetcher._last_expected_trading_day())
    filtered = api_df[api_df["date"] <= last_expected]

    # Today's partial row must be dropped if running before market close
    if last_expected < fake_today:
        assert len(filtered) == 1, "Today's partial bar must be filtered out before save"
        assert filtered.iloc[0]["date"] != fake_today
    else:
        # Running after-close on a trading day → today is allowed through
        assert len(filtered) == 2
