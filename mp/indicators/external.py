"""External data signals fetched from akshare APIs.

Each function is independently callable and returns Signal | None.
Failures are logged and return None — never block other signals.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional

import akshare as ak
import pandas as pd
from loguru import logger

from . import Signal


def _is_a_share(code: Optional[str]) -> bool:
    return code is not None and len(code) == 6 and code.isdigit()


# === 主力资金净流入 ===

def fetch_fund_flow_signal(code: str) -> Signal | None:
    try:
        market = "sh" if code.startswith(("6", "9")) else "sz"
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        recent = df.tail(5)
        col = next((c for c in recent.columns if "主力" in c and "净占比" in c), None)
        if col is None:
            return None
        avg_pct = pd.to_numeric(recent[col], errors="coerce").mean()

        if avg_pct > 5:
            sig, detail = "bullish", f"5日主力净流入占比{avg_pct:+.1f}%, 大资金持续流入"
        elif avg_pct > 0:
            sig, detail = "bullish", f"5日主力净流入占比{avg_pct:+.1f}%, 温和流入"
        elif avg_pct > -5:
            sig, detail = "neutral", f"5日主力净流入占比{avg_pct:+.1f}%"
        else:
            sig, detail = "bearish", f"5日主力净流入占比{avg_pct:+.1f}%, 主力持续流出"

        return Signal(dimension="资金面", name="主力资金", value=round(avg_pct, 2),
                      signal=sig, detail=detail)
    except Exception as e:
        logger.debug(f"Fund flow failed for {code}: {e}")
        return None


# === 融资余额变动 ===

def fetch_margin_signal(code: str) -> Signal | None:
    try:
        today = date.today()
        # Try last few days in case today has no data
        for offset in range(5):
            d = today - timedelta(days=offset)
            try:
                df = ak.stock_margin_detail_sse(date=d.strftime("%Y%m%d"))
                if not df.empty:
                    break
            except Exception:
                continue
        else:
            return None

        code_col = next((c for c in df.columns if "代码" in c), None)
        if code_col is None:
            return None
        df[code_col] = df[code_col].astype(str).str.zfill(6)
        row = df[df[code_col] == code]
        if row.empty:
            return None

        buy_col = next((c for c in df.columns if "融资买入" in c), None)
        repay_col = next((c for c in df.columns if "融资偿还" in c), None)
        bal_col = next((c for c in df.columns if "融资余额" in c), None)
        if not all([buy_col, repay_col, bal_col]):
            return None

        margin_buy = float(row[buy_col].iloc[0])
        margin_repay = float(row[repay_col].iloc[0])
        balance = float(row[bal_col].iloc[0])
        net = margin_buy - margin_repay

        if net > 0:
            sig, detail = "bullish", f"融资净买入{net / 1e4:.0f}万, 杠杆资金加仓"
        elif abs(net) > balance * 0.01:
            sig, detail = "bearish", f"融资净偿还{abs(net) / 1e4:.0f}万, 杠杆资金减仓"
        else:
            sig, detail = "neutral", f"融资余额{balance / 1e8:.1f}亿, 变化不大"

        return Signal(dimension="资金面", name="融资余额", value=round(net / 1e4, 1),
                      signal=sig, detail=detail)
    except Exception as e:
        logger.debug(f"Margin data failed for {code}: {e}")
        return None


# === 北向资金趋势 (市场级) ===

def fetch_northbound_signal() -> Signal | None:
    try:
        net_col = None
        total = 0.0
        for sym in ["沪股通", "深股通"]:
            df = ak.stock_hsgt_hist_em(symbol=sym)
            if net_col is None:
                net_col = next((c for c in df.columns if "净买" in c), df.columns[1])
            vals = pd.to_numeric(df[net_col].tail(5), errors="coerce")
            total += vals.sum()

        if total > 50:
            sig, detail = "bullish", f"北向5日净买入{total:.0f}亿, 外资大幅流入"
        elif total > 0:
            sig, detail = "bullish", f"北向5日净买入{total:.0f}亿"
        elif total > -50:
            sig, detail = "neutral", f"北向5日净流出{abs(total):.0f}亿, 外资观望"
        else:
            sig, detail = "bearish", f"北向5日净流出{abs(total):.0f}亿, 外资撤离"

        return Signal(dimension="市场情绪", name="北向资金", value=round(total, 1),
                      signal=sig, detail=detail)
    except Exception as e:
        logger.debug(f"Northbound data failed: {e}")
        return None


# === 龙虎榜 ===

def fetch_lhb_signal(code: str) -> Signal | None:
    try:
        end = date.today()
        start = end - timedelta(days=10)
        df = ak.stock_lhb_detail_em(
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        code_col = next((c for c in df.columns if "代码" in c), None)
        if code_col is None:
            return None
        df[code_col] = df[code_col].astype(str).str.zfill(6)
        matches = df[df[code_col] == code]

        if matches.empty:
            return Signal(dimension="市场情绪", name="龙虎榜", value=None,
                          signal="neutral", detail="近5日未上龙虎榜")

        net_col = next((c for c in matches.columns if "净买" in c), None)
        reason_col = next((c for c in matches.columns if "原因" in c), None)
        net = pd.to_numeric(matches[net_col], errors="coerce").sum() if net_col else 0
        reason = ", ".join(matches[reason_col].unique()[:2]) if reason_col else ""

        if net > 0:
            sig, detail = "bullish", f"上龙虎榜({reason}), 机构净买入{net / 1e4:.0f}万"
        else:
            sig, detail = "bearish", f"上龙虎榜({reason}), 机构净卖出{abs(net) / 1e4:.0f}万"

        return Signal(dimension="市场情绪", name="龙虎榜", value=round(net / 1e4, 1),
                      signal=sig, detail=detail)
    except Exception as e:
        logger.debug(f"LHB failed for {code}: {e}")
        return None


# === 估值分位 ===

def fetch_valuation_percentile_signal(code: str) -> Signal | None:
    try:
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        df = ak.stock_zh_valuation_baidu(symbol=f"{prefix}{code}",
                                          indicator="总市值", period="近一年")
        val_col = next((c for c in df.columns if c != "date"), df.columns[-1])
        values = pd.to_numeric(df[val_col], errors="coerce").dropna().values
        if len(values) < 20:
            return None

        current = values[-1]
        pct = float((values < current).sum() / len(values) * 100)

        if pct < 10:
            sig, detail = "bullish", f"市值分位{pct:.0f}%, 近一年极低区间"
        elif pct < 30:
            sig, detail = "bullish", f"市值分位{pct:.0f}%, 偏低"
        elif pct > 90:
            sig, detail = "bearish", f"市值分位{pct:.0f}%, 近一年极高区间"
        elif pct > 70:
            sig, detail = "bearish", f"市值分位{pct:.0f}%, 偏高"
        else:
            sig, detail = "neutral", f"市值分位{pct:.0f}%, 合理区间"

        return Signal(dimension="估值", name="估值分位", value=round(pct, 1),
                      signal=sig, detail=detail)
    except Exception as e:
        logger.debug(f"Valuation failed for {code}: {e}")
        return None


# === 股东增减持 ===

def fetch_insider_signal(code: str) -> Signal | None:
    try:
        df = ak.stock_inner_trade_xq()
        code_col = next((c for c in df.columns if "代码" in c), None)
        if code_col is None:
            return None
        df[code_col] = df[code_col].astype(str).str.zfill(6)
        matches = df[df[code_col] == code]

        if matches.empty:
            return Signal(dimension="市场情绪", name="股东增减持", value=None,
                          signal="neutral", detail="近期无股东增减持记录")

        # Detect buy/sell from shares changed column
        shares_col = next((c for c in matches.columns if "变动股数" in c or "变动数" in c), None)
        if shares_col:
            shares = pd.to_numeric(matches[shares_col], errors="coerce")
            buys = int((shares > 0).sum())
            sells = int((shares < 0).sum())
        else:
            buys, sells = len(matches), 0

        if buys > sells:
            sig, detail = "bullish", f"近期股东增持{buys}笔, 减持{sells}笔"
        elif sells > buys:
            sig, detail = "bearish", f"近期股东减持{sells}笔, 增持{buys}笔"
        else:
            sig, detail = "neutral", f"增持{buys}笔, 减持{sells}笔, 持平"

        return Signal(dimension="市场情绪", name="股东增减持", value=buys - sells,
                      signal=sig, detail=detail)
    except Exception as e:
        logger.debug(f"Insider trading failed for {code}: {e}")
        return None


# === Orchestrator ===

def fetch_all_external_signals(code: Optional[str]) -> List[Signal]:
    """Fetch all external signals for an A-share stock. Returns [] for non-A-share."""
    if not _is_a_share(code):
        return []

    signals: list[Signal] = []

    for fn in [fetch_fund_flow_signal, fetch_margin_signal, fetch_lhb_signal,
               fetch_valuation_percentile_signal, fetch_insider_signal]:
        result = fn(code)
        if result is not None:
            signals.append(result)

    nb = fetch_northbound_signal()
    if nb is not None:
        signals.append(nb)

    return signals
