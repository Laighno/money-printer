"""TTL-based disk cache for API response DataFrames.

Stores results as parquet + a companion .ts (Unix timestamp) file in data/cache/.
Cache keys are SHA-1 hashes of the (function_name, **kwargs) mapping.

Usage::

    from mp.data.cache import disk_cache

    df = disk_cache("get_industry_list", ttl=4 * 3600, fetch=_get_industry_list_ths)

    # Or as a look-up / store pair:
    from mp.data.cache import cache_get, cache_put
    cached = cache_get("valuation_snapshot", ttl=6 * 3600)
    if cached is None:
        cached = _fetch_from_api()
        cache_put("valuation_snapshot", cached)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from loguru import logger

_CACHE_DIR = Path("data/cache")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _key(func_name: str, **kwargs) -> str:
    raw = json.dumps({"fn": func_name, **kwargs}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _parquet_path(key: str) -> Path:
    return _cache_dir() / f"{key}.parquet"


def _meta_path(key: str) -> Path:
    return _cache_dir() / f"{key}.ts"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cache_get(func_name: str, ttl: int, **kwargs) -> pd.DataFrame | None:
    """Return cached DataFrame if it exists and is fresher than *ttl* seconds.

    Parameters
    ----------
    func_name:
        Logical name used as part of the cache key (e.g. "get_industry_list").
    ttl:
        Maximum age in seconds.  Pass ``0`` to force a miss.
    kwargs:
        Additional key components (e.g. ``board_name="半导体"``).

    Returns
    -------
    DataFrame or None
        None means cache miss (missing, expired, or corrupt).
    """
    if ttl <= 0:
        return None

    key = _key(func_name, **kwargs)
    p = _parquet_path(key)
    m = _meta_path(key)

    if not p.exists() or not m.exists():
        return None

    try:
        written_at = float(m.read_text().strip())
        age = datetime.now().timestamp() - written_at
        if age > ttl:
            logger.debug("cache miss (expired {:.0f}s > {}s): {}", age, ttl, func_name)
            return None
        df = pd.read_parquet(p)
        logger.debug("cache hit ({:.0f}s old, {} rows): {}", age, len(df), func_name)
        return df
    except Exception as exc:
        logger.debug("cache read error for {}: {}", func_name, exc)
        return None


def cache_put(func_name: str, df: pd.DataFrame, **kwargs) -> None:
    """Persist *df* to disk cache under *func_name* + *kwargs* key."""
    if df is None or df.empty:
        return
    key = _key(func_name, **kwargs)
    try:
        df.to_parquet(_parquet_path(key), index=False)
        _meta_path(key).write_text(str(datetime.now().timestamp()))
        logger.debug("cached {} rows for {}", len(df), func_name)
    except Exception as exc:
        logger.debug("cache write error for {}: {}", func_name, exc)


def disk_cache(
    func_name: str,
    ttl: int,
    fetch: Callable[[], pd.DataFrame],
    **kwargs,
) -> pd.DataFrame:
    """Get from cache or call *fetch()* and cache the result.

    Parameters
    ----------
    func_name:
        Logical cache key name.
    ttl:
        Max age in seconds.
    fetch:
        Zero-argument callable that returns a fresh DataFrame.
    kwargs:
        Extra key components forwarded to cache_get / cache_put.
    """
    cached = cache_get(func_name, ttl, **kwargs)
    if cached is not None:
        return cached

    df = fetch()
    cache_put(func_name, df, **kwargs)
    return df


def invalidate(func_name: str, **kwargs) -> None:
    """Delete cached entry so next call triggers a fresh fetch."""
    key = _key(func_name, **kwargs)
    for path in (_parquet_path(key), _meta_path(key)):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
