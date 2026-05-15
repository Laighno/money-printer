"""Regression tests for the 2026-05-07 fix: turnover unit normalization.

Bug: Eastmoney returns 换手率 as 百分数 (e.g. 7.14% → 7.14), Sina as 小数
(0.0714).  The fetcher had no normalization, so EM-fallback rows got stored
100× larger than Sina rows.  This polluted turnover_5d / turnover_pctile
features and caused predictions to swing wildly between report runs (e.g.
粤电力A predicted +11.85% when its 5-6 EOD bar came from EM, then +2.94%
once 5-7 EOD bar came from Sina).

Fixes:
1. _get_daily_bars_em / _get_daily_bars_etf divide turnover by 100 at fetch
2. save_bars_upsert auto-normalizes any turnover > 1.0 as a backstop
3. One-time backfill: UPDATE daily_bars SET turnover=turnover/100 WHERE >1
"""

from __future__ import annotations

import os
import pandas as pd
import pytest


# ── Source-level normalization (fetcher) ────────────────────────────────────

def test_em_fetcher_normalizes_turnover(monkeypatch):
    """_get_daily_bars_em must divide EM's percentage turnover by 100."""
    from mp.data import fetcher

    fake_em_response = pd.DataFrame({
        "日期": ["2026-05-06", "2026-05-07"],
        "开盘": [6.74, 7.04],
        "最高": [7.16, 7.38],
        "最低": [6.66, 6.99],
        "收盘": [7.03, 7.27],
        "成交量": [1825716, 1994048],   # 手 (lots)
        "成交额": [1256392221.0, 1439034572.0],
        "换手率": [7.14, 7.80],          # 百分数 from EM
    })

    monkeypatch.setattr(fetcher.ak, "stock_zh_a_hist", lambda **kw: fake_em_response)

    df = fetcher._get_daily_bars_em("000539", "20260506", "20260507")

    # turnover should be normalized to 小数 (decimal)
    assert df["turnover"].iloc[0] == pytest.approx(0.0714, rel=1e-4)
    assert df["turnover"].iloc[1] == pytest.approx(0.0780, rel=1e-4)
    # All values should be < 1.0
    assert (df["turnover"] < 1.0).all()
    # Volume should still be normalized to 股 (×100)
    assert df["volume"].iloc[0] == pytest.approx(1825716 * 100)


def test_etf_fetcher_normalizes_turnover(monkeypatch):
    """_get_daily_bars_etf must divide EM's percentage turnover by 100."""
    from mp.data import fetcher

    fake_etf_response = pd.DataFrame({
        "日期": ["2026-05-07"],
        "开盘": [1.50], "最高": [1.55], "最低": [1.48], "收盘": [1.52],
        "成交量": [10000], "成交额": [15200000.0],
        "换手率": [3.40],  # 百分数
    })

    monkeypatch.setattr(fetcher.ak, "fund_etf_hist_em", lambda **kw: fake_etf_response)

    df = fetcher._get_daily_bars_etf("512660", "20260507", "20260507")
    assert df["turnover"].iloc[0] == pytest.approx(0.034, rel=1e-4)


# ── Storage-level normalization (save_bars_upsert backstop) ─────────────────

def test_save_bars_upsert_normalizes_high_turnover(tmp_path, monkeypatch):
    """If a row sneaks through with turnover > 1.0, save_bars_upsert must
    auto-normalize as a defensive backstop."""
    monkeypatch.setenv("MP_DB_PATH", str(tmp_path / "test.db"))

    from mp.data.store import DataStore
    store = DataStore()
    # Bootstrap schema
    from sqlalchemy import text
    with store.engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                code TEXT, date TEXT, open REAL, high REAL, low REAL,
                close REAL, volume REAL, amount REAL, turnover REAL,
                PRIMARY KEY (code, date)
            )
        """))

    # Row with bug-shape: turnover stored as percent (7.14)
    df = pd.DataFrame([{
        "code": "000539", "date": "2026-05-06",
        "open": 6.74, "high": 7.16, "low": 6.66, "close": 7.03,
        "volume": 182571600.0, "amount": 1256392221.0,
        "turnover": 7.14,   # 百分数, BUG
    }])

    store.save_bars_upsert(df)

    # Read back; should have been auto-normalized to 0.0714
    with store.engine.connect() as conn:
        row = conn.execute(text(
            "SELECT turnover FROM daily_bars WHERE code='000539' AND date='2026-05-06'"
        )).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(0.0714, rel=1e-4), \
        f"Expected normalized 0.0714, got {row[0]}"


def test_save_bars_upsert_passes_through_decimal_turnover(tmp_path, monkeypatch):
    """Already-correct decimal turnover must NOT be touched."""
    monkeypatch.setenv("MP_DB_PATH", str(tmp_path / "test.db"))

    from mp.data.store import DataStore
    store = DataStore()
    from sqlalchemy import text
    with store.engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                code TEXT, date TEXT, open REAL, high REAL, low REAL,
                close REAL, volume REAL, amount REAL, turnover REAL,
                PRIMARY KEY (code, date)
            )
        """))

    df = pd.DataFrame([{
        "code": "000539", "date": "2026-05-07",
        "open": 7.04, "high": 7.38, "low": 6.99, "close": 7.27,
        "volume": 199404841.0, "amount": 1439034572.0,
        "turnover": 0.0780,  # already 小数
    }])
    store.save_bars_upsert(df)

    with store.engine.connect() as conn:
        row = conn.execute(text(
            "SELECT turnover FROM daily_bars WHERE code='000539' AND date='2026-05-07'"
        )).fetchone()
    assert row[0] == pytest.approx(0.0780, rel=1e-4)


# ── Sentinel: production DB must never contain turnover > 1.0 ───────────────

def test_no_turnover_pollution_in_production_db():
    """Guards against silent regression — fail if any row has turnover > 1.0
    in the production market.db, which would re-trigger the prediction
    swing bug."""
    from sqlalchemy import text
    from mp.data.store import DataStore, DEFAULT_DB_URL
    store = DataStore(db_url=DEFAULT_DB_URL)
    with store.engine.connect() as conn:
        n_bad = conn.execute(text(
            "SELECT COUNT(*) FROM daily_bars WHERE turnover > 1.0"
        )).scalar()
    assert n_bad == 0, (
        f"Found {n_bad} rows with turnover > 1.0 in production market.db. "
        "Some fetcher path returned 百分数 instead of 小数 — investigate "
        "_get_daily_bars_em / _get_daily_bars_etf and run backfill: "
        "UPDATE daily_bars SET turnover = turnover/100 WHERE turnover > 1.0"
    )
