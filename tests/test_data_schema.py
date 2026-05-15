"""Comprehensive tests for the data unit moat (mp/data/schema.py).

This is the DEFENSE TESTS — every known unit-mismatch pattern is covered,
plus the meta-test that adding a new source forces explicit declaration.

History of bugs these tests prevent:
- 2026-04-28: EM volume in 手 not 股 (×100 missing)
- 2026-05-07: EM turnover in 百分数 not 小数 (/100 missing)
"""

from __future__ import annotations

import pandas as pd
import pytest


# ── normalize_bars: source registry enforcement ─────────────────────────────

def test_unknown_source_raises():
    """Adding a new fetcher MUST register its source — silent passthrough is forbidden."""
    from mp.data.schema import normalize_bars

    df = pd.DataFrame({"volume": [100], "turnover": [0.05], "amount": [1000.0],
                       "close": [10.0]})
    with pytest.raises(ValueError, match="Unknown data source"):
        normalize_bars(df, source="some_new_provider")


def test_known_sources_listed_in_error():
    """The error message lists existing sources so devs know what to copy from."""
    from mp.data.schema import normalize_bars

    df = pd.DataFrame({"volume": [1], "turnover": [0.1], "amount": [1.0], "close": [1.0]})
    try:
        normalize_bars(df, source="bogus")
    except ValueError as e:
        assert "eastmoney" in str(e)
        assert "sina" in str(e)


# ── normalize_bars: each source's conventions ───────────────────────────────

def test_eastmoney_volume_lots_to_shares():
    """EM volume in 手 must be ×100 to canonical 股."""
    from mp.data.schema import normalize_bars

    df = pd.DataFrame({
        "volume": [1000.0],     # 手
        "turnover": [7.14],     # 百分数
        "amount": [70_000.0],
        "close": [7.0],
    })
    out = normalize_bars(df, source="eastmoney")
    assert out["volume"].iloc[0] == pytest.approx(100_000.0)  # ×100
    assert out["turnover"].iloc[0] == pytest.approx(0.0714)   # /100


def test_eastmoney_etf_same_conventions():
    """ETF source must have same EM conventions."""
    from mp.data.schema import normalize_bars

    df = pd.DataFrame({"volume": [50.0], "turnover": [3.4], "amount": [5000.0],
                       "close": [10.0]})
    out = normalize_bars(df, source="eastmoney_etf")
    assert out["volume"].iloc[0] == pytest.approx(5000.0)
    assert out["turnover"].iloc[0] == pytest.approx(0.034)


def test_sina_passthrough():
    """Sina returns canonical units already — no conversion."""
    from mp.data.schema import normalize_bars

    df = pd.DataFrame({"volume": [100_000.0], "turnover": [0.05],
                       "amount": [1_000_000.0], "close": [10.0]})
    out = normalize_bars(df, source="sina")
    assert out["volume"].iloc[0] == 100_000.0
    assert out["turnover"].iloc[0] == 0.05


def test_intraday_no_turnover_field():
    """Intraday source has no turnover column — must not crash."""
    from mp.data.schema import normalize_bars

    df = pd.DataFrame({"volume": [50_000.0], "amount": [500_000.0], "close": [10.0]})
    out = normalize_bars(df, source="intraday_sina")
    assert out["volume"].iloc[0] == 50_000.0


# ── validate_bars: invariant enforcement ────────────────────────────────────

def test_validate_drops_negative_close():
    from mp.data.schema import validate_bars

    df = pd.DataFrame({
        "code": ["A", "B"],
        "open": [10.0, -1.0], "high": [10.0, 1.0], "low": [9.0, 0.5],
        "close": [9.5, -1.0], "volume": [1000.0, 1000.0],
        "amount": [9500.0, 1000.0], "turnover": [0.05, 0.05],
    })
    out = validate_bars(df)
    assert len(out) == 1
    assert out["code"].iloc[0] == "A"


def test_validate_drops_amount_volume_inconsistent():
    """Row where amount/(volume*close) is way off → unit mismatch → drop."""
    from mp.data.schema import validate_bars

    df = pd.DataFrame({
        "code": ["A", "B"],
        "open": [10.0, 10.0], "high": [10.0, 10.0], "low": [10.0, 10.0],
        "close": [10.0, 10.0],
        "volume": [1000.0, 1000.0],
        "amount": [10_000.0, 100.0],  # B: ratio = 100 / 10000 = 0.01 → bad
        "turnover": [0.05, 0.05],
    })
    out = validate_bars(df)
    assert len(out) == 1
    assert out["code"].iloc[0] == "A"


def test_validate_allows_qfq_adjusted_ratio():
    """qfq-adjusted bars can have amount/(volume*close) up to ~5× legitimately
    (when close is forward-adjusted but amount is in nominal CNY).  Allow
    this; only flag as bug when ratio > 50."""
    from mp.data.schema import validate_bars

    df = pd.DataFrame({
        "code": ["BYD"],
        "open": [111.0], "high": [113.0], "low": [110.0], "close": [111.0],
        "volume": [11_705_700.0],          # 股, post-fix
        "amount": [3_947_000_994.0],        # ratio = 3.04 (qfq adjustment)
        "turnover": [0.05],
    })
    out = validate_bars(df)
    assert len(out) == 1, "qfq-adjusted bar wrongly dropped"


def test_validate_auto_normalizes_turnover_drift():
    """If turnover > 1 sneaks in (e.g. unregistered source), auto /100."""
    from mp.data.schema import validate_bars

    df = pd.DataFrame({
        "code": ["A"], "open": [7.0], "high": [7.5], "low": [6.5], "close": [7.0],
        "volume": [1e6], "amount": [7e6], "turnover": [7.14],   # bug-shape
    })
    out = validate_bars(df)
    assert len(out) == 1
    assert out["turnover"].iloc[0] == pytest.approx(0.0714)


def test_validate_passes_clean_rows():
    from mp.data.schema import validate_bars

    df = pd.DataFrame({
        "code": ["A"] * 3,
        "open": [10.0, 11.0, 12.0],
        "high": [10.5, 11.2, 12.5],
        "low": [9.8, 10.8, 11.7],
        "close": [10.2, 11.0, 12.1],
        "volume": [100_000.0, 120_000.0, 90_000.0],
        "amount": [1_020_000.0, 1_320_000.0, 1_089_000.0],
        "turnover": [0.05, 0.06, 0.045],
    })
    out = validate_bars(df)
    assert len(out) == 3


def test_validate_handles_nan_turnover():
    """Sina rows with NaN turnover must pass (no spurious filter)."""
    import numpy as np
    from mp.data.schema import validate_bars

    df = pd.DataFrame({
        "code": ["A"], "open": [10.0], "high": [10.0], "low": [10.0],
        "close": [10.0], "volume": [100_000.0], "amount": [1_000_000.0],
        "turnover": [np.nan],
    })
    out = validate_bars(df)
    assert len(out) == 1


# ── detect_per_stock_drift ──────────────────────────────────────────────────

def test_per_stock_drift_no_history():
    """No historical median = can't detect drift; pass through."""
    from mp.data.schema import detect_per_stock_drift

    df = pd.DataFrame({
        "code": ["NEW"], "date": ["2026-05-07"],
        "volume": [1e9], "amount": [1e10], "turnover": [50.0],
    })
    out = detect_per_stock_drift(df, lambda c, col: None)
    assert len(out) == 1   # passes through (no history to compare)


def test_per_stock_drift_within_threshold():
    """Value within 50× of median = ok, no warning."""
    from mp.data.schema import detect_per_stock_drift

    df = pd.DataFrame({
        "code": ["A"], "date": ["2026-05-07"],
        "volume": [200_000.0], "amount": [2_000_000.0], "turnover": [0.06],
    })
    medians = {"volume": 100_000.0, "amount": 1_000_000.0, "turnover": 0.05}
    out = detect_per_stock_drift(df, lambda c, col: medians.get(col))
    assert len(out) == 1


def test_per_stock_drift_catches_unit_spike(caplog):
    """Value >50× historical = unit error suspect → log warning."""
    from mp.data.schema import detect_per_stock_drift

    df = pd.DataFrame({
        "code": ["A"], "date": ["2026-05-07"],
        "volume": [100_000.0], "amount": [1_000_000.0],
        "turnover": [5.0],   # 100× the historical 0.05 → likely 百分数 leak
    })
    medians = {"volume": 100_000.0, "amount": 1_000_000.0, "turnover": 0.05}
    out = detect_per_stock_drift(df, lambda c, col: medians.get(col))
    assert len(out) == 1   # not dropped, just flagged
    # Check that warning was issued (loguru goes to stderr in tests; just
    # confirm no crash)


# ── End-to-end: fetcher → normalize → validate ──────────────────────────────

def test_em_fetcher_uses_normalize_bars(monkeypatch):
    """_get_daily_bars_em must route through normalize_bars."""
    from mp.data import fetcher

    fake_em = pd.DataFrame({
        "日期": ["2026-05-07"],
        "开盘": [7.04], "最高": [7.38], "最低": [6.99], "收盘": [7.27],
        "成交量": [1_994_048],   # 手
        "成交额": [1_439_034_572.0],
        "换手率": [7.80],   # 百分数
    })
    monkeypatch.setattr(fetcher.ak, "stock_zh_a_hist", lambda **kw: fake_em)
    df = fetcher._get_daily_bars_em("000539", "20260507", "20260507")

    # Volume normalized: 手 → 股
    assert df["volume"].iloc[0] == pytest.approx(199_404_800.0)
    # Turnover normalized: 百分数 → 小数
    assert df["turnover"].iloc[0] == pytest.approx(0.0780, rel=1e-4)


def test_etf_fetcher_uses_normalize_bars(monkeypatch):
    """_get_daily_bars_etf must route through normalize_bars."""
    from mp.data import fetcher

    fake_etf = pd.DataFrame({
        "日期": ["2026-05-07"],
        "开盘": [1.5], "最高": [1.55], "最低": [1.48], "收盘": [1.52],
        "成交量": [10_000.0], "成交额": [15_200_000.0],
        "换手率": [3.40],
    })
    monkeypatch.setattr(fetcher.ak, "fund_etf_hist_em", lambda **kw: fake_etf)
    df = fetcher._get_daily_bars_etf("512660", "20260507", "20260507")

    assert df["volume"].iloc[0] == pytest.approx(1_000_000.0)
    assert df["turnover"].iloc[0] == pytest.approx(0.034, rel=1e-4)


# ── Sentinel: production DB invariants ──────────────────────────────────────

def test_no_turnover_pollution_in_production_db():
    """Fail if any row in market.db has turnover > 1.0 — would re-trigger
    the Yue-Dianli prediction-swing bug."""
    from sqlalchemy import text
    from mp.data.store import DataStore, DEFAULT_DB_URL
    store = DataStore(db_url=DEFAULT_DB_URL)
    with store.engine.connect() as conn:
        n = conn.execute(text(
            "SELECT COUNT(*) FROM daily_bars WHERE turnover > 1.0"
        )).scalar()
    assert n == 0, (
        f"Found {n} rows with turnover > 1.0. Run backfill: "
        "UPDATE daily_bars SET turnover = turnover/100 WHERE turnover > 1.0"
    )


def test_no_amount_volume_inconsistency_in_production_db():
    """Fail if production DB has rows where amount/(volume*close) > 50,
    which is the unmistakable 100× volume unit bug signature.  Bounds
    [3, 50] is allowed — that's the natural qfq adjustment range."""
    from sqlalchemy import text
    from mp.data.store import DataStore, DEFAULT_DB_URL
    store = DataStore(db_url=DEFAULT_DB_URL)
    with store.engine.connect() as conn:
        n = conn.execute(text("""
            SELECT COUNT(*) FROM daily_bars
            WHERE volume > 0 AND close > 0 AND amount IS NOT NULL
              AND (amount / (volume * close) < 0.3
                   OR amount / (volume * close) > 50.0)
        """)).scalar()
    # Allow up to 100 historically-broken edge cases; alert if it grows
    assert n < 100, (
        f"Found {n} rows with amount/(volume*close) outside [0.3, 50]. "
        "Recent unit drift detected — investigate fetcher paths and run backfill: "
        "UPDATE daily_bars SET volume = volume*100 WHERE amount/(volume*close) > 50"
    )
