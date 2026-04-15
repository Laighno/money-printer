"""Market data fetcher using akshare with multi-source fallback.

Data source priority:
  - Industry boards: THS (同花顺) primary, eastmoney fallback
  - Stock bars: Sina (新浪) primary, eastmoney fallback
  - Valuation: Xueqiu (雪球) per-stock, Sina spot fallback
  - Financial: eastmoney (unique source)
"""

from datetime import date

import akshare as ak
import pandas as pd
from loguru import logger

from . import proxy_patch
proxy_patch.apply()


# === Index / Universe ===

def get_index_constituents(index: str = "hs300") -> list[str]:
    """Get constituent stock codes for a given index."""
    code_map = {"hs300": "000300", "zz500": "000905", "zz1000": "000852"}
    if index not in code_map:
        raise ValueError(f"Unsupported index: {index}. Use hs300/zz500/zz1000.")

    logger.info(f"Fetching {index} constituents...")
    df = ak.index_stock_cons_weight_csindex(symbol=code_map[index])
    codes = df["成分券代码"].astype(str).str.zfill(6).tolist()
    logger.info(f"Got {len(codes)} stocks for {index}")
    return codes


# === Industry / Sector ===

def get_industry_list() -> pd.DataFrame:
    """Get all industry board names from THS (primary) or eastmoney (fallback).

    Returns:
        DataFrame with: date, board_name, board_code, close, change_pct, ...
    """
    # Try THS first
    try:
        return _get_industry_list_ths()
    except Exception as e:
        logger.warning(f"THS industry list failed: {e}, trying eastmoney...")

    # Fallback to eastmoney
    return _get_industry_list_em()


def _get_industry_list_ths() -> pd.DataFrame:
    """THS industry board list with summary data."""
    logger.info("Fetching industry boards from THS (同花顺)...")
    # Get board name/code mapping (reliable)
    boards = ak.stock_board_industry_name_ths()

    # Try to get summary for extra data (change_pct etc.), but don't fail if unavailable
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

    # Try THS first
    try:
        return _get_industry_hist_ths(board_name, start, end)
    except Exception as e:
        logger.debug(f"THS hist for '{board_name}' failed: {e}, trying eastmoney...")

    # Fallback to eastmoney
    return _get_industry_hist_em(board_name, start, end)


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
    # THS doesn't provide change_pct and turnover directly, compute change_pct
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
    # Try SW (申万) first
    try:
        return _get_industry_constituents_sw(board_name)
    except Exception as e:
        logger.debug(f"SW constituents for '{board_name}' failed: {e}, trying eastmoney...")

    # Fallback to eastmoney
    return _get_industry_constituents_em(board_name)


# THS board name -> SW industry code mapping (built lazily)
_THS_SW_MAP: dict[str, str] | None = None

# Manual mappings for THS boards that don't auto-match SW second-level
_THS_SW_MANUAL = {
    "公路铁路运输": "801179",   # SW二级: 铁路公路
    "军工装备": "801744",       # SW二级: 航空装备Ⅱ
    "塑料制品": "801036",       # SW二级: 塑料
    "建筑材料": "801710",       # SW一级: 建筑材料
    "建筑装饰": "801720",       # SW一级: 建筑装饰
    "房地产": "801181",         # SW二级: 房地产开发
    "文化传媒": "801760",       # SW一级: 传媒
    "旅游及酒店": "801993",     # SW二级: 旅游及景区
    "机场航运": "801991",       # SW二级: 航空机场
    "橡胶制品": "801037",       # SW二级: 橡胶
    "汽车整车": "801091",       # SW二级: 乘用车
    "汽车服务及其他": "801092",  # SW二级: 汽车服务
    "油气开采及服务": "801960",  # SW一级: 石油石化
    "港口航运": "801992",       # SW二级: 航运港口
    "煤炭开采加工": "801951",   # SW二级: 煤炭开采
    "石油加工贸易": "801073",   # SW二级: 石油化工
    "种植业与林业": "801016",   # SW二级: 种植业
    "美容护理": "801980",       # SW一级: 美容护理
    "其他社会服务": "801881",   # SW二级: 摩托车及其他(近似)
    "钢铁": "801040",           # SW一级: 钢铁
    "银行": "801780",           # SW一级: 银行
    "零售": "801203",           # SW二级: 一般零售
    "食品加工制造": "801124",   # SW二级: 食品加工
    "饮料制造": "801127",       # SW二级: 饮料乳品
}


def _build_ths_sw_map() -> dict[str, str]:
    """Build THS board name -> SW industry code mapping."""
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

    # Add manual overrides
    sw_map.update(_THS_SW_MANUAL)
    _THS_SW_MAP = sw_map
    return _THS_SW_MAP


def _get_industry_constituents_sw(board_name: str) -> pd.DataFrame:
    """SW (申万) industry constituents with PE/PB."""
    mapping = _build_ths_sw_map()
    sw_code = mapping.get(board_name)
    if not sw_code:
        raise ValueError(f"No SW mapping for THS board '{board_name}'")

    # Try sw_index_third_cons first (has PE/PB/市值 built in, best source)
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

    # Fallback: index_component_sw (basic list) + XQ per-stock valuation
    try:
        df = ak.index_component_sw(symbol=sw_code)
    except Exception as e:
        raise ValueError(f"SW index_component_sw failed for {sw_code}: {e}")

    # Column names vary between versions
    code_col = next((c for c in df.columns if "代码" in c), None)
    name_col = next((c for c in df.columns if "名称" in c or "简称" in c), None)
    if not code_col or not name_col:
        raise ValueError(f"Unexpected columns from SW: {df.columns.tolist()}")

    codes = df[code_col].astype(str).str.replace(r"\.\w+$", "", regex=True).str.zfill(6).tolist()
    names = df[name_col].tolist()

    # Batch XQ valuation (only for small boards, skip for >50 stocks)
    rows = []
    for i, (code, name) in enumerate(zip(codes, names)):
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


# === Stock bars ===

def get_daily_bars(code: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV bars. Sina primary, eastmoney fallback, local DB supplement.

    If the API returns data ending before what's in the local DB, the local DB
    rows are appended so that manually-ingested bars (e.g. from Sina realtime)
    are picked up by downstream feature computation.
    """
    if end is None:
        end = date.today().strftime("%Y%m%d")

    # Try Sina first
    df = None
    try:
        df = _get_daily_bars_sina(code, start, end)
    except Exception as e:
        logger.debug(f"Sina bars for {code} failed: {e}, trying eastmoney...")

    # Fallback to eastmoney
    if df is None or df.empty:
        try:
            df = _get_daily_bars_em(code, start, end)
        except Exception as e:
            logger.debug(f"EM bars for {code} failed: {e}, trying local DB...")

    # Supplement from local DB if API data is stale
    try:
        from mp.data.store import DataStore
        store = DataStore()
        db_df = store.load_bars(codes=[code], start=start, end=end)
        if not db_df.empty:
            db_df["date"] = pd.to_datetime(db_df["date"])
            if df is None or df.empty:
                return db_df
            api_max = df["date"].max()
            db_extra = db_df[db_df["date"] > api_max]
            if not db_extra.empty:
                # Align columns
                cols = df.columns.tolist()
                for c in cols:
                    if c not in db_extra.columns:
                        db_extra[c] = float("nan")
                df = pd.concat([df, db_extra[cols]], ignore_index=True)
                logger.debug(f"{code}: supplemented {len(db_extra)} bars from local DB")
    except Exception as e:
        logger.debug(f"DB supplement for {code} failed: {e}")

    return df if df is not None else pd.DataFrame()


def _code_to_sina(code: str) -> str:
    """Convert 6-digit code to Sina format (sz/sh prefix)."""
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
    # Sina doesn't provide turnover, fill with NaN
    if "turnover" not in df.columns:
        df["turnover"] = float("nan")
    return df[["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover"]]


def _get_daily_bars_em(code: str, start: str, end: str) -> pd.DataFrame:
    """Eastmoney daily bars (original)."""
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


# === Valuation (real-time snapshot) ===

def get_valuation_snapshot() -> pd.DataFrame:
    """Fetch real-time PE/PB for all A-shares. Eastmoney primary, Sina fallback."""
    # Try eastmoney first (fastest, has PE/PB)
    try:
        return _get_valuation_snapshot_em()
    except Exception as e:
        logger.warning(f"EM valuation snapshot failed: {e}, trying Sina...")

    # Fallback to Sina (no PE/PB but has basic price data)
    return _get_valuation_snapshot_sina()


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


# === Financial statements ===

def _code_to_em(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    return f"{code}.SZ"


def get_financial_data(code: str) -> pd.DataFrame:
    """Fetch financial indicators from eastmoney. Includes report publish date for time-alignment."""
    em_code = _code_to_em(code)
    try:
        df = ak.stock_financial_analysis_indicator_em(symbol=em_code)
    except Exception as e:
        logger.warning(f"Failed to fetch financial data for {code}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "code": code,
        "report_date": pd.to_datetime(df["REPORT_DATE"]),
        "publish_date": pd.to_datetime(df["NOTICE_DATE"]),
        "roe": pd.to_numeric(df["ROEJQ"], errors="coerce"),
        "eps": pd.to_numeric(df["EPSJB"], errors="coerce"),
        "bps": pd.to_numeric(df["BPS"], errors="coerce"),
        "debt_ratio": pd.to_numeric(df["ZCFZL"], errors="coerce"),
        "gross_margin": pd.to_numeric(df.get("XSMLL"), errors="coerce"),
        "net_margin": pd.to_numeric(df["XSJLL"], errors="coerce"),
        "revenue_growth": pd.to_numeric(df["YYZSRGDHBZC"], errors="coerce"),
        "profit_growth": pd.to_numeric(df["NETPROFITRPHBZC"], errors="coerce"),
    })


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
