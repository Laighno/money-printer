"""Market data fetcher using akshare with multi-source fallback.

Data source priority:
  - Index constituents: CSIndex primary, eastmoney fallback
  - Industry boards:    THS (同花顺) primary, eastmoney fallback
  - Stock bars:         DB cache first → Sina primary → eastmoney fallback
  - Valuation:          Eastmoney primary, Sina fallback; backed by TTL disk cache
  - Financial:          Eastmoney primary; results persisted to DB, served from DB on failure

Caching / persistence strategy
-------------------------------
* ``get_industry_list()`` and ``get_valuation_snapshot()`` are cached to
  ``data/cache/`` for a configurable TTL (4 h and 6 h respectively).  A stale
  cache is always used as last-resort if every live source fails.

* ``get_index_constituents()`` is cached for 24 h (index membership changes
  quarterly).

* ``get_daily_bars()`` uses a **DB-first incremental** pattern:
    1. Load what's already in the local SQLite DB.
    2. If the DB is fresh enough (last trading day covered), return it directly.
    3. Otherwise fetch only the *missing* date range from the API, save to DB,
       and return the merged result.
  This avoids re-downloading years of history on every run.

* ``get_financial_data()`` saves successful fetches to DB and falls back to the
  DB when the live source fails.

Retry
-----
All single-source helpers wrap their akshare call in ``_with_retry()`` which
retries up to 3 times with exponential back-off (1 s, 2 s, 4 s).
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Callable, TypeVar

import akshare as ak
import pandas as pd
from loguru import logger

from . import proxy_patch
from .cache import cache_get, cache_put

proxy_patch.apply()

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _with_retry(fn: Callable[[], T], retries: int = 3, base_delay: float = 1.0) -> T:
    """Call *fn()* up to *retries* extra times on exception with exponential back-off."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                delay = base_delay * (2 ** attempt)
                logger.debug("Retry {}/{} after {:.1f}s ({})", attempt + 1, retries, delay, exc)
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Index / Universe
# ---------------------------------------------------------------------------

_INDEX_TTL = 24 * 3600  # index membership changes quarterly

def get_index_constituents(index: str = "hs300") -> list[str]:
    """Get constituent stock codes for a given index.

    Primary:  CSIndex (中证指数)
    Fallback: Eastmoney spot list filtered by market cap (approximate)
    Cache:    24-hour TTL in data/cache/; stale copy used on total failure
    """
    code_map = {"hs300": "000300", "zz500": "000905", "zz1000": "000852"}
    if index not in code_map:
        raise ValueError(f"Unsupported index: {index}. Use hs300/zz500/zz1000.")

    cached = cache_get("index_constituents", ttl=_INDEX_TTL, index=index)
    if cached is not None and not cached.empty:
        codes = cached["code"].tolist()
        logger.info(f"Index {index} constituents from cache: {len(codes)} stocks")
        return codes

    logger.info(f"Fetching {index} constituents...")

    # Primary: CSIndex
    try:
        df = _with_retry(lambda: ak.index_stock_cons_weight_csindex(symbol=code_map[index]))
        codes = df["成分券代码"].astype(str).str.zfill(6).tolist()
        logger.info(f"Got {len(codes)} stocks for {index} from CSIndex")
        cache_put("index_constituents", pd.DataFrame({"code": codes}), index=index)
        return codes
    except Exception as e:
        logger.warning(f"CSIndex constituents for {index} failed: {e}, trying eastmoney...")

    # Fallback: Eastmoney spot — top-N by market cap as proxy
    try:
        spot = _with_retry(lambda: ak.stock_zh_a_spot_em())
        spot["code"] = spot["代码"].astype(str).str.zfill(6)
        spot["total_mv"] = pd.to_numeric(spot["总市值"], errors="coerce")
        n_map = {"hs300": 300, "zz500": 500, "zz1000": 1000}
        top = spot.nlargest(n_map[index], "total_mv")
        codes = top["code"].tolist()
        logger.warning(f"Using EM market-cap proxy for {index}: {len(codes)} stocks")
        cache_put("index_constituents", pd.DataFrame({"code": codes}), index=index)
        return codes
    except Exception as e:
        logger.warning(f"EM fallback for {index} also failed: {e}")

    # Last resort: stale cache (7 days)
    stale = cache_get("index_constituents", ttl=7 * 24 * 3600, index=index)
    if stale is not None and not stale.empty:
        codes = stale["code"].tolist()
        logger.warning(f"Using stale cache for {index} constituents: {len(codes)} stocks")
        return codes

    raise RuntimeError(f"All sources failed for index constituents: {index}")


# ---------------------------------------------------------------------------
# Industry / Sector
# ---------------------------------------------------------------------------

_INDUSTRY_LIST_TTL = 4 * 3600  # industry list rarely changes intraday

def get_industry_list() -> pd.DataFrame:
    """Get all industry board names from THS (primary) or eastmoney (fallback).

    Returns:
        DataFrame with: date, board_name, board_code, close, change_pct, ...

    Results are cached for 4 hours.  If all live sources fail, the most recent
    cached copy is returned (stale-on-error).
    """
    cached = cache_get("industry_list", ttl=_INDUSTRY_LIST_TTL)
    if cached is not None:
        return cached

    try:
        df = _with_retry(_get_industry_list_ths)
        cache_put("industry_list", df)
        return df
    except Exception as e:
        logger.warning(f"THS industry list failed: {e}, trying eastmoney...")

    try:
        df = _with_retry(_get_industry_list_em)
        cache_put("industry_list", df)
        return df
    except Exception as e:
        logger.warning(f"EM industry list also failed: {e}")

    # Stale-on-error
    stale = cache_get("industry_list", ttl=7 * 24 * 3600)
    if stale is not None:
        logger.warning("Returning stale industry list from cache (all live sources failed)")
        return stale

    return pd.DataFrame()


def _get_industry_list_ths() -> pd.DataFrame:
    """THS industry board list with summary data."""
    logger.info("Fetching industry boards from THS (同花顺)...")
    boards = ak.stock_board_industry_name_ths()

    try:
        summary = ak.stock_board_industry_summary_ths()
        summary_map = {}
        for _, row in summary.iterrows():
            summary_map[row["板块"]] = {
                "change_pct": row.get("涨跌幅", 0),
                "up_count": row.get("上涨家数", 0),
                "down_count": row.get("下跌家数", 0),
            }
    except Exception as e:
        logger.debug(f"THS summary failed (non-critical): {e}")
        summary_map = {}

    rows = []
    for _, b in boards.iterrows():
        name = b["name"]
        extra = summary_map.get(name, {})
        rows.append({
            "date": date.today().isoformat(),
            "board_name": name,
            "board_code": b["code"],
            "close": 0.0,
            "change_pct": extra.get("change_pct", 0),
            "total_mv": 0.0,
            "turnover": 0.0,
            "up_count": extra.get("up_count", 0),
            "down_count": extra.get("down_count", 0),
        })

    result = pd.DataFrame(rows)
    logger.info(f"Got {len(result)} industry boards from THS")
    return result


def _get_industry_list_em() -> pd.DataFrame:
    """Eastmoney industry board list (original implementation)."""
    logger.info("Fetching industry board list from eastmoney...")
    df = ak.stock_board_industry_name_em()
    result = df.rename(columns={
        "板块名称": "board_name",
        "板块代码": "board_code",
        "最新价": "close",
        "涨跌幅": "change_pct",
        "总市值": "total_mv",
        "换手率": "turnover",
        "上涨家数": "up_count",
        "下跌家数": "down_count",
    })
    result["date"] = date.today().isoformat()
    logger.info(f"Got {len(result)} industry boards from eastmoney")
    return result[["date", "board_name", "board_code", "close", "change_pct",
                    "total_mv", "turnover", "up_count", "down_count"]]


def get_industry_hist(board_name: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Get daily K-line for an industry board. THS primary, eastmoney fallback.

    Args:
        board_name: e.g. '半导体', '白酒'
        start: 'YYYYMMDD'
        end: 'YYYYMMDD', defaults to today
    """
    if end is None:
        end = date.today().strftime("%Y%m%d")

    try:
        return _with_retry(lambda: _get_industry_hist_ths(board_name, start, end))
    except Exception as e:
        logger.debug(f"THS hist for '{board_name}' failed: {e}, trying eastmoney...")

    return _with_retry(lambda: _get_industry_hist_em(board_name, start, end))


def _get_industry_hist_ths(board_name: str, start: str, end: str) -> pd.DataFrame:
    """THS industry board K-line."""
    df = ak.stock_board_industry_index_ths(symbol=board_name, start_date=start, end_date=end)
    result = df.rename(columns={
        "日期": "date",
        "开盘价": "open",
        "收盘价": "close",
        "最高价": "high",
        "最低价": "low",
        "成交量": "volume",
        "成交额": "amount",
    })
    result["board_name"] = board_name
    result["date"] = pd.to_datetime(result["date"])
    result["change_pct"] = result["close"].pct_change() * 100
    result["turnover"] = 0.0
    return result[["board_name", "date", "open", "high", "low", "close",
                    "change_pct", "volume", "amount", "turnover"]]


def _get_industry_hist_em(board_name: str, start: str, end: str) -> pd.DataFrame:
    """Eastmoney industry board K-line (original)."""
    df = ak.stock_board_industry_hist_em(symbol=board_name, period="日k",
                                          start_date=start, end_date=end, adjust="")
    result = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "涨跌幅": "change_pct",
        "成交量": "volume", "成交额": "amount", "换手率": "turnover",
    })
    result["board_name"] = board_name
    result["date"] = pd.to_datetime(result["date"])
    return result[["board_name", "date", "open", "high", "low", "close",
                    "change_pct", "volume", "amount", "turnover"]]


def get_industry_hist_batch(board_names: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Fetch history for multiple industry boards."""
    frames = []
    for i, name in enumerate(board_names):
        try:
            df = get_industry_hist(name, start, end)
            frames.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch board '{name}': {e}")
        if (i + 1) % 20 == 0:
            logger.info(f"Fetched industry hist: {i + 1}/{len(board_names)}")
    if not frames:
        return pd.DataFrame()
    logger.info(f"Fetched industry hist: {len(frames)}/{len(board_names)} boards")
    return pd.concat(frames, ignore_index=True)


def get_industry_constituents(board_name: str) -> pd.DataFrame:
    """Get constituent stocks of an industry board. SW (申万) primary, eastmoney fallback.

    Returns DataFrame with: code, name, close, pe_ttm, pb, board_name, ...
    """
    try:
        return _get_industry_constituents_sw(board_name)
    except Exception as e:
        logger.debug(f"SW constituents for '{board_name}' failed: {e}, trying eastmoney...")

    return _get_industry_constituents_em(board_name)


# THS board name -> SW industry code mapping (built lazily)
_THS_SW_MAP: dict[str, str] | None = None

_THS_SW_MANUAL = {
    "公路铁路运输": "801179",
    "军工装备": "801744",
    "塑料制品": "801036",
    "建筑材料": "801710",
    "建筑装饰": "801720",
    "房地产": "801181",
    "文化传媒": "801760",
    "旅游及酒店": "801993",
    "机场航运": "801991",
    "橡胶制品": "801037",
    "汽车整车": "801091",
    "汽车服务及其他": "801092",
    "油气开采及服务": "801960",
    "港口航运": "801992",
    "煤炭开采加工": "801951",
    "石油加工贸易": "801073",
    "种植业与林业": "801016",
    "美容护理": "801980",
    "其他社会服务": "801881",
    "钢铁": "801040",
    "银行": "801780",
    "零售": "801203",
    "食品加工制造": "801124",
    "饮料制造": "801127",
}


def _build_ths_sw_map() -> dict[str, str]:
    global _THS_SW_MAP
    if _THS_SW_MAP is not None:
        return _THS_SW_MAP

    try:
        sw2 = ak.sw_index_second_info()
    except Exception:
        _THS_SW_MAP = dict(_THS_SW_MANUAL)
        return _THS_SW_MAP

    sw_map = {}
    for _, row in sw2.iterrows():
        sw_name = row["行业名称"]
        sw_code = row["行业代码"].replace(".SI", "")
        clean = sw_name.replace("Ⅱ", "").replace("Ⅲ", "")
        sw_map[sw_name] = sw_code
        if clean != sw_name:
            sw_map[clean] = sw_code

    sw_map.update(_THS_SW_MANUAL)
    _THS_SW_MAP = sw_map
    return _THS_SW_MAP


def _get_industry_constituents_sw(board_name: str) -> pd.DataFrame:
    """SW (申万) industry constituents with PE/PB."""
    mapping = _build_ths_sw_map()
    sw_code = mapping.get(board_name)
    if not sw_code:
        raise ValueError(f"No SW mapping for THS board '{board_name}'")

    try:
        sw3_cons = ak.sw_index_third_cons(symbol=f"{sw_code}.SI")
        result = sw3_cons.rename(columns={
            "股票代码": "code", "股票简称": "name", "价格": "close",
            "市盈率ttm": "pe_ttm", "市净率": "pb", "市值": "total_mv",
        })
        result["code"] = result["code"].astype(str).str.replace(r"\.\w+$", "", regex=True).str.zfill(6)
        result["board_name"] = board_name
        result["change_pct"] = 0.0
        result["amount"] = 0.0
        result["turnover"] = 0.0
        cols = ["code", "name", "close", "change_pct", "amount",
                "turnover", "pe_ttm", "pb", "board_name"]
        for c in cols:
            if c not in result.columns:
                result[c] = None
        logger.info(f"Got {len(result)} constituents for '{board_name}' from SW (申万)")
        return result[cols]
    except Exception as e:
        logger.debug(f"sw_index_third_cons for {sw_code} failed: {e}")

    try:
        df = ak.index_component_sw(symbol=sw_code)
    except Exception as e:
        raise ValueError(f"SW index_component_sw failed for {sw_code}: {e}")

    code_col = next((c for c in df.columns if "代码" in c), None)
    name_col = next((c for c in df.columns if "名称" in c or "简称" in c), None)
    if not code_col or not name_col:
        raise ValueError(f"Unexpected columns from SW: {df.columns.tolist()}")

    codes = df[code_col].astype(str).str.replace(r"\.\w+$", "", regex=True).str.zfill(6).tolist()
    names = df[name_col].tolist()

    rows = []
    for code, name in zip(codes, names):
        row = {"code": code, "name": name, "close": 0, "change_pct": 0,
               "amount": 0, "turnover": 0, "pe_ttm": None, "pb": None,
               "board_name": board_name}
        if len(codes) <= 50:
            try:
                prefix = "SH" if code.startswith(("6", "9")) else "SZ"
                xq = ak.stock_individual_spot_xq(symbol=f"{prefix}{code}")
                val = dict(zip(xq["item"], xq["value"]))
                row["close"] = float(val.get("现价", 0) or 0)
                row["pe_ttm"] = float(val.get("市盈率(TTM)", 0) or 0)
                row["pb"] = float(val.get("市净率", 0) or 0)
            except Exception:
                pass
        rows.append(row)

    result = pd.DataFrame(rows)
    logger.info(f"Got {len(result)} constituents for '{board_name}' from SW+XQ")
    return result


def _get_industry_constituents_em(board_name: str) -> pd.DataFrame:
    """Eastmoney industry constituents."""
    df = ak.stock_board_industry_cons_em(symbol=board_name)
    result = df.rename(columns={
        "代码": "code", "名称": "name", "最新价": "close",
        "涨跌幅": "change_pct", "成交额": "amount",
        "换手率": "turnover", "市盈率-动态": "pe_ttm", "市净率": "pb",
    })
    result["code"] = result["code"].astype(str).str.zfill(6)
    result["board_name"] = board_name
    return result


# ---------------------------------------------------------------------------
# Stock bars — DB-first incremental pattern
# ---------------------------------------------------------------------------

def get_daily_bars(code: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV bars with a DB-first incremental strategy.

    Flow:
    1. Load existing data from local SQLite for the requested range.
    2. If DB already covers up to the last expected trading day, return it directly.
    3. Otherwise fetch only the *missing* tail from API (Sina → EM fallback).
    4. Persist new API rows back to DB via upsert.
    5. Return merged result.

    This avoids re-downloading full history on every run.
    """
    from mp.data.store import DataStore

    if end is None:
        end = date.today().strftime("%Y%m%d")

    store = DataStore()

    # Step 1: load from DB
    db_df = store.load_bars(codes=[code], start=start, end=end)

    # Step 2: check freshness — only fetch what's missing
    fetch_start: str
    if not db_df.empty:
        db_max = db_df["date"].max().date()
        expected = _last_expected_trading_day()
        if db_max >= expected:
            logger.debug(f"{code}: DB is fresh (last={db_max}), skipping API")
            return db_df
        fetch_start = (db_max + timedelta(days=1)).strftime("%Y%m%d")
        logger.debug(f"{code}: DB ends {db_max}, fetching {fetch_start}~{end}")
    else:
        fetch_start = start

    # Step 3: incremental API fetch with retry
    api_df: pd.DataFrame | None = None
    try:
        api_df = _with_retry(lambda: _get_daily_bars_sina(code, fetch_start, end))
    except Exception as e:
        logger.debug(f"Sina bars for {code} ({fetch_start}~{end}): {e}, trying EM...")

    if api_df is None or api_df.empty:
        try:
            api_df = _with_retry(lambda: _get_daily_bars_em(code, fetch_start, end))
        except Exception as e:
            logger.debug(f"EM bars for {code}: {e}")

    # Step 4: persist new rows to DB
    if api_df is not None and not api_df.empty:
        try:
            store.save_bars_upsert(api_df)
        except Exception as e:
            logger.warning(f"Failed to save bars for {code} to DB: {e}")

    # Step 5: merge and return
    if not db_df.empty and api_df is not None and not api_df.empty:
        combined = pd.concat([db_df, api_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["code", "date"]).sort_values("date").reset_index(drop=True)
        return combined
    if api_df is not None and not api_df.empty:
        return api_df
    if not db_df.empty:
        return db_df

    logger.warning(f"No bar data found for {code} ({start}~{end})")
    return pd.DataFrame()


def _last_expected_trading_day() -> date:
    """Last trading day we expect to have closing data for.

    After 16:00 → today; before → yesterday.  Skip weekends (not holidays).
    """
    now = datetime.now()
    candidate = now.date() if now.hour >= 16 else now.date() - timedelta(days=1)
    while candidate.weekday() >= 5:  # Sat=5, Sun=6
        candidate -= timedelta(days=1)
    return candidate


def _code_to_sina(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _get_daily_bars_sina(code: str, start: str, end: str) -> pd.DataFrame:
    """Sina daily bars."""
    sina_code = _code_to_sina(code)
    df = ak.stock_zh_a_daily(symbol=sina_code, start_date=start, end_date=end, adjust="qfq")
    df = df.rename(columns={
        "date": "date", "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume", "amount": "amount",
    })
    df["code"] = code
    df["date"] = pd.to_datetime(df["date"])
    if "turnover" not in df.columns:
        df["turnover"] = float("nan")
    return df[["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover"]]


def _get_daily_bars_em(code: str, start: str, end: str) -> pd.DataFrame:
    """Eastmoney daily bars."""
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
        "收盘": "close", "成交量": "volume", "成交额": "amount",
        "换手率": "turnover",
    })
    df["code"] = code
    df["date"] = pd.to_datetime(df["date"])
    if "turnover" not in df.columns:
        df["turnover"] = float("nan")
    return df[["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover"]]


def get_daily_bars_batch(codes: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Fetch daily bars for multiple stocks."""
    frames = []
    for i, code in enumerate(codes):
        try:
            frames.append(get_daily_bars(code, start, end))
        except Exception as e:
            logger.warning(f"Failed to fetch {code}: {e}")
        if (i + 1) % 50 == 0:
            logger.info(f"Fetched {i + 1}/{len(codes)} stocks")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Valuation (real-time snapshot)
# ---------------------------------------------------------------------------

_VALUATION_TTL = 6 * 3600  # intraday data, 6-hour cache

def get_valuation_snapshot() -> pd.DataFrame:
    """Fetch real-time PE/PB for all A-shares.

    Priority: Eastmoney → Sina → TTL disk cache → stale disk cache.
    Cache TTL: 6 hours.
    """
    cached = cache_get("valuation_snapshot", ttl=_VALUATION_TTL)
    if cached is not None:
        return cached

    try:
        df = _with_retry(_get_valuation_snapshot_em)
        cache_put("valuation_snapshot", df)
        return df
    except Exception as e:
        logger.warning(f"EM valuation snapshot failed: {e}, trying Sina...")

    try:
        df = _with_retry(_get_valuation_snapshot_sina)
        cache_put("valuation_snapshot", df)
        return df
    except Exception as e:
        logger.warning(f"Sina valuation snapshot also failed: {e}")

    # Stale-on-error (up to 2 days old)
    stale = cache_get("valuation_snapshot", ttl=2 * 24 * 3600)
    if stale is not None:
        logger.warning("Returning stale valuation snapshot (all live sources failed)")
        return stale

    return pd.DataFrame()


def _get_valuation_snapshot_em() -> pd.DataFrame:
    """Eastmoney valuation snapshot (original)."""
    logger.info("Fetching real-time valuation snapshot from eastmoney...")
    df = ak.stock_zh_a_spot_em()
    df = df.rename(columns={
        "代码": "code", "名称": "name", "市盈率-动态": "pe_ttm",
        "市净率": "pb", "总市值": "total_mv", "最新价": "close",
    })
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = date.today().isoformat()
    result = df[["code", "date", "name", "close", "pe_ttm", "pb", "total_mv"]].copy()
    result = result[result["total_mv"] > 0]
    logger.info(f"Got valuation for {len(result)} stocks from eastmoney")
    return result


def _get_valuation_snapshot_sina() -> pd.DataFrame:
    """Sina spot snapshot (slower, no PE/PB)."""
    logger.info("Fetching real-time snapshot from Sina (no PE/PB)...")
    df = ak.stock_zh_a_spot()
    df = df.rename(columns={
        "代码": "code", "名称": "name", "最新价": "close",
        "成交额": "amount",
    })
    df["code"] = df["code"].astype(str).str.replace(r"^[a-z]+", "", regex=True).str.zfill(6)
    df["date"] = date.today().isoformat()
    df["pe_ttm"] = None
    df["pb"] = None
    df["total_mv"] = None
    result = df[["code", "date", "name", "close", "pe_ttm", "pb", "total_mv"]].copy()
    result = result[pd.to_numeric(result["close"], errors="coerce") > 0]
    logger.info(f"Got snapshot for {len(result)} stocks from Sina (no PE/PB)")
    return result


# ---------------------------------------------------------------------------
# Financial statements — DB-backed with live fallthrough
# ---------------------------------------------------------------------------

def _code_to_em(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    return f"{code}.SZ"


def get_financial_data(code: str) -> pd.DataFrame:
    """Fetch financial indicators from eastmoney with DB persistence.

    On success, saves to DB so future calls can survive API outages.
    On failure, falls back to the most recent DB copy.
    """
    em_code = _code_to_em(code)
    df: pd.DataFrame | None = None

    try:
        raw = _with_retry(lambda: ak.stock_financial_analysis_indicator_em(symbol=em_code))
        if raw is not None and not raw.empty:
            df = pd.DataFrame({
                "code": code,
                "report_date": pd.to_datetime(raw["REPORT_DATE"]),
                "publish_date": pd.to_datetime(raw["NOTICE_DATE"]),
                "roe": pd.to_numeric(raw["ROEJQ"], errors="coerce"),
                "eps": pd.to_numeric(raw["EPSJB"], errors="coerce"),
                "bps": pd.to_numeric(raw["BPS"], errors="coerce"),
                "debt_ratio": pd.to_numeric(raw["ZCFZL"], errors="coerce"),
                "gross_margin": pd.to_numeric(raw.get("XSMLL"), errors="coerce"),
                "net_margin": pd.to_numeric(raw["XSJLL"], errors="coerce"),
                "revenue_growth": pd.to_numeric(raw["YYZSRGDHBZC"], errors="coerce"),
                "profit_growth": pd.to_numeric(raw["NETPROFITRPHBZC"], errors="coerce"),
            })
            # Persist to DB so it's available offline
            try:
                from mp.data.store import DataStore
                DataStore().save_financial(df)
            except Exception as save_err:
                logger.debug(f"Failed to save financial data for {code}: {save_err}")
    except Exception as e:
        logger.warning(f"Live financial fetch failed for {code}: {e}, trying DB fallback...")

    if df is not None and not df.empty:
        return df

    # DB fallback — return all historical rows for this stock
    try:
        from mp.data.store import DataStore
        db_df = DataStore().load_financial(codes=[code])
        if not db_df.empty:
            logger.debug(f"Financial data for {code} served from DB ({len(db_df)} rows)")
            return db_df
    except Exception as db_err:
        logger.debug(f"DB fallback for {code} failed: {db_err}")

    return pd.DataFrame()


def get_financial_data_batch(codes: list[str]) -> pd.DataFrame:
    """Fetch financial data for multiple stocks."""
    frames = []
    for i, code in enumerate(codes):
        df = get_financial_data(code)
        if not df.empty:
            frames.append(df)
        if (i + 1) % 50 == 0:
            logger.info(f"Fetched financial data: {i + 1}/{len(codes)}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_industry_mapping(universe: list[str] | None = None) -> dict[str, str]:
    """Return a {stock_code: industry_name} mapping for the given universe.

    Fetches all THS/EM industry board constituent lists concurrently and builds
    the mapping.  Results are cached to disk for 7 days (industry membership
    changes rarely).

    Parameters
    ----------
    universe : list[str] or None
        If provided, only codes in this list are included in the result.
        If None, all codes found across all boards are returned.

    Returns
    -------
    dict mapping 6-digit code → board_name string.  Empty dict on total failure.
    """
    cache_key = "industry_mapping_v1"
    cached = cache_get(cache_key, ttl=60 * 60 * 24 * 7)  # 7 days
    if cached is not None and not cached.empty:
        mapping = dict(zip(cached["code"], cached["board_name"]))
        if universe:
            mapping = {c: v for c, v in mapping.items() if c in set(universe)}
        logger.debug(f"Industry mapping served from cache ({len(mapping)} stocks)")
        return mapping

    logger.info("Fetching industry mapping (all boards)...")
    try:
        boards_df = get_industry_list()
        if boards_df.empty:
            logger.warning("get_industry_list returned empty, industry mapping unavailable")
            return {}
        board_names = boards_df["board_name"].dropna().unique().tolist()
    except Exception as e:
        logger.warning(f"Failed to get industry list for mapping: {e}")
        return {}

    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    rows: list[dict] = []

    def _fetch_one(bname: str) -> list[dict]:
        try:
            cons = get_industry_constituents(bname)
            if cons.empty or "code" not in cons.columns:
                return []
            return [{"code": c, "board_name": bname} for c in cons["code"].dropna().unique()]
        except Exception:
            return []

    workers = min(12, len(board_names))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_fetch_one, b): b for b in board_names}
        for fut in _as_completed(futs):
            rows.extend(fut.result())

    if not rows:
        logger.warning("Industry mapping: no constituent data retrieved")
        return {}

    result_df = pd.DataFrame(rows).drop_duplicates(subset=["code"], keep="first")
    cache_put(cache_key, result_df)
    logger.info(f"Industry mapping built: {len(result_df)} stocks across {len(board_names)} boards")

    mapping = dict(zip(result_df["code"], result_df["board_name"]))
    if universe:
        mapping = {c: v for c, v in mapping.items() if c in set(universe)}
    return mapping
