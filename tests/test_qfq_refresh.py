"""Regression tests for the 2026-05-07 fix: weekly full-history qfq refresh.

Bug being prevented: incremental fetching only pulls today's qfq-adjusted
bar.  When a stock has a corporate action, historical DB rows are at the
OLD adjustment factor while new rows arrive at the NEW factor.  This
poisons ret_1d with fake ±30% drops that the model trains on.

The qfq_refresh script runs weekly to re-pull the full universe history,
overwriting all rows so the entire series is consistent with today's qfq
adjustment factor.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest


# Allow `import scripts.qfq_refresh as ...` from tests
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_force_fetch_bypasses_db_freshness(monkeypatch):
    """_force_fetch_full_history must NOT consult DB for freshness; it must
    always call the API for the full requested range."""
    import scripts.qfq_refresh as qfq

    api_calls: list[tuple[str, str, str]] = []

    def _fake_sina(code: str, start: str, end: str) -> pd.DataFrame:
        api_calls.append((code, start, end))
        return pd.DataFrame({
            "code": [code], "date": [pd.Timestamp("2026-05-07")],
            "open": [10.0], "high": [10.0], "low": [10.0], "close": [10.0],
            "volume": [100_000.0], "amount": [1_000_000.0], "turnover": [0.05],
        })

    monkeypatch.setattr(qfq, "_get_daily_bars_sina", _fake_sina)
    monkeypatch.setattr(qfq, "_with_retry", lambda fn: fn())

    df = qfq._force_fetch_full_history("000539", "20230101", "20260507")
    assert df is not None
    assert len(api_calls) == 1
    code, start, end = api_calls[0]
    assert code == "000539"
    assert start == "20230101"   # not just the missing-from-DB tail
    assert end == "20260507"


def test_force_fetch_falls_back_to_em(monkeypatch):
    """When Sina fails, must fall back to EM (mirrors get_daily_bars
    behavior so the same routing applies under qfq refresh)."""
    import scripts.qfq_refresh as qfq

    def _fail_sina(*a, **kw):
        raise RuntimeError("sina down")

    em_called = []

    def _fake_em(code: str, start: str, end: str) -> pd.DataFrame:
        em_called.append(code)
        return pd.DataFrame({
            "code": [code], "date": [pd.Timestamp("2026-05-07")],
            "open": [10.0], "high": [10.0], "low": [10.0], "close": [10.0],
            "volume": [100_000.0], "amount": [1_000_000.0], "turnover": [0.05],
        })

    monkeypatch.setattr(qfq, "_get_daily_bars_sina", _fail_sina)
    monkeypatch.setattr(qfq, "_get_daily_bars_em", _fake_em)
    monkeypatch.setattr(qfq, "_with_retry", lambda fn: fn())

    df = qfq._force_fetch_full_history("000539", "20230101", "20260507")
    assert df is not None
    assert em_called == ["000539"]


def test_force_fetch_uses_etf_for_etf_codes(monkeypatch):
    """ETF codes must use _get_daily_bars_etf (stock APIs return wrong data)."""
    import scripts.qfq_refresh as qfq

    etf_called = []
    sina_called = []

    def _fake_etf(code, start, end):
        etf_called.append(code)
        return pd.DataFrame({
            "code": [code], "date": [pd.Timestamp("2026-05-07")],
            "open": [1.5], "high": [1.5], "low": [1.5], "close": [1.5],
            "volume": [10_000.0], "amount": [15_000.0], "turnover": [0.034],
        })

    def _fake_sina(*a, **kw):
        sina_called.append(a)
        return pd.DataFrame()

    monkeypatch.setattr(qfq, "_get_daily_bars_etf", _fake_etf)
    monkeypatch.setattr(qfq, "_get_daily_bars_sina", _fake_sina)
    monkeypatch.setattr(qfq, "_with_retry", lambda fn: fn())

    df = qfq._force_fetch_full_history("512660", "20230101", "20260507")
    assert df is not None
    assert etf_called == ["512660"]
    assert sina_called == [], "ETF must not call stock API"


def test_count_diffs_detects_qfq_adjustment(tmp_path, monkeypatch):
    """When new fetched close differs > 0.5% from DB close, count_diffs must
    report it — the canary for qfq adjustment."""
    monkeypatch.setenv("MP_DB_PATH", str(tmp_path / "test.db"))
    import scripts.qfq_refresh as qfq
    importlib.reload(qfq)   # pick up new MP_DB_PATH

    from sqlalchemy import text
    from mp.data.store import DataStore
    store = DataStore()
    with store.engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                code TEXT, date TEXT, open REAL, high REAL, low REAL,
                close REAL, volume REAL, amount REAL, turnover REAL,
                PRIMARY KEY (code, date)
            )
        """))
        # OLD close (pre-adjustment scale)
        conn.execute(text(
            "INSERT INTO daily_bars VALUES ('300033', '2026-04-08', "
            "302.18, 319.49, 301.55, 319.39, 1.0, 1.0, 0.01)"
        ))

    # NEW close (post-adjustment scale)
    new_df = pd.DataFrame({
        "code": ["300033"], "date": [pd.Timestamp("2026-04-08")],
        "open": [212.27], "high": [224.43], "low": [211.83],
        "close": [224.36],   # ≈ 30% lower than old 319.39
        "volume": [1.0], "amount": [1.0], "turnover": [0.01],
    })
    n_changed, max_chg = qfq._count_diffs(store, "300033", new_df)
    assert n_changed == 1
    assert max_chg > 0.20, f"Expected ~30% adjustment, got {max_chg:.1%}"


def test_count_diffs_ignores_tiny_changes(tmp_path, monkeypatch):
    """Floating-point noise (< 0.5% change) must NOT count as a real diff."""
    monkeypatch.setenv("MP_DB_PATH", str(tmp_path / "test.db"))
    import scripts.qfq_refresh as qfq
    importlib.reload(qfq)

    from sqlalchemy import text
    from mp.data.store import DataStore
    store = DataStore()
    with store.engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                code TEXT, date TEXT, open REAL, high REAL, low REAL,
                close REAL, volume REAL, amount REAL, turnover REAL,
                PRIMARY KEY (code, date)
            )
        """))
        conn.execute(text(
            "INSERT INTO daily_bars VALUES ('000539', '2026-05-07', "
            "7.04, 7.38, 6.99, 7.27, 1.0, 1.0, 0.078)"
        ))

    # 0.1% rounding noise — should NOT be flagged
    new_df = pd.DataFrame({
        "code": ["000539"], "date": [pd.Timestamp("2026-05-07")],
        "open": [7.04], "high": [7.38], "low": [6.99], "close": [7.272],
        "volume": [1.0], "amount": [1.0], "turnover": [0.078],
    })
    n_changed, _ = qfq._count_diffs(store, "000539", new_df)
    assert n_changed == 0


def test_universe_includes_holdings(tmp_path, monkeypatch):
    """Universe must include portfolio holdings even if they're not in the
    index universe."""
    import yaml
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "portfolio.yaml").write_text(yaml.safe_dump({
        "holdings": [
            # Use a fake code not in any real index so we can pin len==1
            {"name": "非指数股票", "code": "999999", "type": "stock"},
            {"name": "板块占位", "code": None, "type": "board"},
        ]
    }), encoding="utf-8")

    import scripts.qfq_refresh as qfq
    from mp.data import fetcher as fetcher_mod
    monkeypatch.chdir(tmp_path)
    # qfq_refresh now goes through get_recommendation_universe → which
    # calls fetcher.get_index_constituents internally.  Patch at the
    # fetcher module so both paths return empty.
    monkeypatch.setattr(fetcher_mod, "get_index_constituents", lambda *a, **kw: [])

    universe = qfq._build_universe()
    assert "999999" in universe
    assert len(universe) == 1   # board entry must be skipped (no code)


# ---------------------------------------------------------------------------
# Default start date: per-code earliest DB date (2026-09 fix).
# The old fixed default 20230101 never re-anchored the ~54% of rows dated
# before 2023, so those rows stayed at whatever qfq factor they were first
# fetched under.
# ---------------------------------------------------------------------------

def _make_isolated_store(tmp_path, monkeypatch):
    """Fresh SQLite under tmp_path; MP_DB_PATH + chdir so nothing touches
    data/market.db (Rule #4.1 prod-state protection)."""
    monkeypatch.setenv("MP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.chdir(tmp_path)
    import scripts.qfq_refresh as qfq
    importlib.reload(qfq)
    from sqlalchemy import text
    from mp.data.store import DataStore
    store = DataStore()
    with store.engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                code TEXT, date TEXT, open REAL, high REAL, low REAL,
                close REAL, volume REAL, amount REAL, turnover REAL,
                PRIMARY KEY (code, date)
            )
        """))
        conn.execute(text(
            "INSERT INTO daily_bars VALUES ('300033', '2015-03-02', "
            "1, 1, 1, 1, 1.0, 1.0, 0.01)"
        ))
        conn.execute(text(
            "INSERT INTO daily_bars VALUES ('300033', '2026-04-08', "
            "1, 1, 1, 1, 1.0, 1.0, 0.01)"
        ))
    return qfq, store


def test_resolve_start_uses_earliest_db_date(tmp_path, monkeypatch):
    qfq, store = _make_isolated_store(tmp_path, monkeypatch)
    assert qfq._resolve_start(store, "300033") == "20150302"
    # No rows for this code -> global floor, not 20230101
    assert qfq._resolve_start(store, "999999") == qfq.GLOBAL_START_FLOOR == "20140101"
    # Explicit --since wins
    assert qfq._resolve_start(store, "300033", since="20230101") == "20230101"


def test_refresh_qfq_default_start_is_per_code_not_2023(tmp_path, monkeypatch):
    """refresh_qfq(start=None) must fetch each code from its earliest DB
    date (floor 20140101), never from the old fixed 20230101."""
    qfq, _ = _make_isolated_store(tmp_path, monkeypatch)

    calls: list[tuple[str, str, str]] = []

    def _fake_fetch(code, start, end):
        calls.append((code, start, end))
        return None   # counted as failed -> no upsert, no diff

    monkeypatch.setattr(qfq, "_force_fetch_full_history", _fake_fetch)

    summary = qfq.refresh_qfq(["300033", "999999"], start=None, end="20260507")
    assert summary["total"] == 2
    assert calls == [
        ("300033", "20150302", "20260507"),
        ("999999", "20140101", "20260507"),
    ]
    assert all(s != "20230101" for _, s, _ in calls)

    # Legacy fixed start still honoured when given explicitly
    calls.clear()
    qfq.refresh_qfq(["300033", "999999"], start="20230101", end="20260507")
    assert [s for _, s, _ in calls] == ["20230101", "20230101"]


def test_cli_default_since_is_none_and_start_alias(tmp_path, monkeypatch):
    """CLI: no flag -> start=None (per-code); --since / --start both set it."""
    qfq, _ = _make_isolated_store(tmp_path, monkeypatch)

    seen: list = []

    def _fake_refresh(codes, start, end):
        seen.append(start)
        return {"total": 0, "processed": 0, "failed": 0, "with_changes": 0,
                "rows_changed": 0, "biggest_delta_code": "", "biggest_delta_pct": 0.0,
                "elapsed_seconds": 0}

    monkeypatch.setattr(qfq, "refresh_qfq", _fake_refresh)

    for argv, expected in [
        (["qfq_refresh.py", "--codes", "300033"], None),
        (["qfq_refresh.py", "--codes", "300033", "--since", "20230101"], "20230101"),
        (["qfq_refresh.py", "--codes", "300033", "--start", "20240101"], "20240101"),
    ]:
        monkeypatch.setattr(sys, "argv", argv)
        assert qfq.main() == 0
        assert seen[-1] == expected, argv
