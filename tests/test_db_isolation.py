"""Regression tests for the 2026-04-29 fix: DataStore must respect
``MP_DB_PATH`` environment variable so tests can isolate without writing
to the real production DB.

Earlier `test_save_bars_upsert_rejects_unit_mismatch` set MP_DB_PATH but
DataStore ignored it, polluting the real `data/market.db` with two test rows.
"""

from __future__ import annotations

import inspect
import os
import re

import pytest


def test_datastore_signature_supports_optional_url():
    """DataStore.__init__ must accept None / no argument, deferring to env var."""
    from mp.data.store import DataStore
    sig = inspect.signature(DataStore.__init__)
    params = sig.parameters
    assert "db_url" in params, "DataStore must accept db_url param"
    # Default must be None or env-fallback (not a hard-coded production path)
    default = params["db_url"].default
    assert default is None or default == inspect.Parameter.empty, (
        f"DataStore default db_url should be None (env-resolved), got: {default!r}"
    )


def test_datastore_respects_mp_db_path(tmp_path, monkeypatch):
    """When MP_DB_PATH is set, DataStore must use that path, not the production default."""
    from mp.data.store import DataStore

    db_path = tmp_path / "isolated.db"
    monkeypatch.setenv("MP_DB_PATH", str(db_path))
    store = DataStore()
    assert str(db_path) in str(store.engine.url), (
        f"DataStore ignored MP_DB_PATH; engine={store.engine.url}"
    )
    assert "data/market.db" not in str(store.engine.url), (
        "DataStore should NOT fall back to production market.db when MP_DB_PATH set"
    )


def test_datastore_accepts_full_sqlite_url(tmp_path, monkeypatch):
    """MP_DB_PATH may be either a path or a full sqlite:/// URL."""
    from mp.data.store import DataStore

    db_path = tmp_path / "isolated2.db"
    monkeypatch.setenv("MP_DB_PATH", f"sqlite:///{db_path}")
    store = DataStore()
    assert str(db_path) in str(store.engine.url)


def test_datastore_explicit_arg_overrides_env(tmp_path, monkeypatch):
    """Explicit db_url argument must override MP_DB_PATH (test predictability)."""
    from mp.data.store import DataStore

    env_path = tmp_path / "env.db"
    explicit_path = tmp_path / "explicit.db"
    monkeypatch.setenv("MP_DB_PATH", str(env_path))
    store = DataStore(db_url=f"sqlite:///{explicit_path}")
    assert str(explicit_path) in str(store.engine.url)
    assert str(env_path) not in str(store.engine.url)


def test_datastore_falls_back_to_default_when_no_env(monkeypatch):
    """When MP_DB_PATH is not set, falls back to production default."""
    from mp.data.store import DataStore, DEFAULT_DB_URL

    monkeypatch.delenv("MP_DB_PATH", raising=False)
    # We don't actually want to create the engine (would touch real DB),
    # so we just check that _resolve_db_url returns the default.
    from mp.data.store import _resolve_db_url
    assert _resolve_db_url() == DEFAULT_DB_URL


def test_no_test_pollution_in_market_db():
    """Ensure no test-only fake codes (111111-444444) leak into market.db.

    This guards against future tests forgetting to set MP_DB_PATH and
    silently polluting prod data again.
    """
    from sqlalchemy import text
    # Use the production default explicitly
    from mp.data.store import DataStore, DEFAULT_DB_URL
    store = DataStore(db_url=DEFAULT_DB_URL)
    with store.engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT COUNT(*) FROM daily_bars WHERE code IN "
            "('111111', '222222', '333333', '444444')"
        )).scalar()
    assert rows == 0, (
        f"Found {rows} polluted test rows in production market.db. "
        "Some test wrote test fixtures to the real DB without isolation."
    )
