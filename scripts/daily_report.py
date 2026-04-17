"""Daily portfolio evaluation & stock recommendation report.

Evaluates current holdings via ML model, scans ZZ500 for top picks,
generates position change suggestions, and sends report to Feishu.

Run daily via launchd or manually::

    python scripts/daily_report.py              # generate + send
    python scripts/daily_report.py --dry-run    # generate only, don't send
    python scripts/daily_report.py --chat-id oc_xxx  # send to group chat
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from mp.ml.model import StockRanker, BlendRanker, FEATURE_COLS
from mp.ml.dataset import build_latest_features
from mp.regime import RegimeDetector, MarketRegime, ROUND_TRIP_COST
from mp.screener.signal_screener import screen_stocks

import yaml


# =====================================================================
# Config
# =====================================================================

PORTFOLIO_PATH = PROJECT_ROOT / "config" / "portfolio.yaml"
MODEL_PATH = PROJECT_ROOT / "data" / "model.lgb"

# Default: send DM to self
DEFAULT_USER_ID = "ou_da792f0119461fb14c41b21b40834b09"


def load_holdings() -> List[dict]:
    """Load A-share holdings from portfolio.yaml."""
    with open(PORTFOLIO_PATH) as f:
        cfg = yaml.safe_load(f)
    return [h for h in cfg.get("holdings", []) if h.get("type") == "stock" and h.get("code")]


def get_stock_names(codes: List[str]) -> Dict[str, str]:
    """Fetch stock names via Sina quote API (single HTTP call, proxy-safe)."""
    import httpx

    name_map = {}
    # Build Sina symbols: sh for 6xx, sz for everything else
    symbols = []
    for code in codes:
        prefix = "sh" if code.startswith("6") else "sz"
        symbols.append(f"{prefix}{code}")

    try:
        url = f"https://hq.sinajs.cn/list={','.join(symbols)}"
        resp = httpx.get(url, headers={"Referer": "https://finance.sina.com.cn"}, timeout=10)
        for line in resp.text.strip().split("\n"):
            if "=" not in line:
                continue
            var_part, data_part = line.split("=", 1)
            sina_code = var_part.split("_")[-1]  # e.g. sz000678
            code = sina_code[2:]
            fields = data_part.strip('";\r').split(",")
            if fields and fields[0]:
                name_map[code] = fields[0]
    except Exception as e:
        logger.warning("Sina name fetch failed: {}", e)

    # Fallback: use code itself for any missing
    for code in codes:
        if code not in name_map:
            name_map[code] = code
    return name_map


# =====================================================================
# Universe scoring (for meaningful BlendRanker percentiles)
# =====================================================================

def score_universe(ranker, holding_codes: list[str], intraday_bars: dict | None = None) -> pd.DataFrame:
    """Score holdings within ZZ500 universe for meaningful percentiles.

    BlendRanker's .rank(pct=True) only makes sense with a large pool.
    This scores holdings + ZZ500 together, then returns holdings rows.

    Returns DataFrame with columns: code, ml_score, rank_pct, raw_return.
    """
    from mp.data.fetcher import get_index_constituents

    is_rank = getattr(ranker, "score_type", "predicted_return") == "rank_percentile"

    if not is_rank:
        features = build_latest_features(holding_codes, include_fundamentals=True,
                                         intraday_bars=intraday_bars)
        if features.empty:
            return pd.DataFrame()
        scores = ranker.predict(features)
        df = pd.DataFrame({
            "code": features["code"].values,
            "ml_score": scores,
            "rank_pct": pd.Series(scores).rank(pct=True).values,
            "raw_return": scores,
        })
        if "_data_warnings" in features.columns:
            df["_data_warnings"] = features["_data_warnings"].values
        return df

    # BlendRanker: score holdings + ZZ500 together
    try:
        zz500_codes = get_index_constituents("zz500")
    except Exception as e:
        logger.warning("ZZ500 fetch failed in score_universe, scoring holdings only: {}", e)
        zz500_codes = []

    all_codes = list(set(holding_codes + zz500_codes))
    features = build_latest_features(all_codes, include_fundamentals=True,
                                     intraday_bars=intraday_bars)
    if features.empty:
        return pd.DataFrame()

    # If universe data quality is bad, rank_pct is unreliable
    dq = features.attrs.get("_data_quality", 1.0)
    if dq < 0.5:
        logger.warning("⚠ ZZ500数据质量差({:.0f}%完整)，排名不可信，仅返回持仓原始预测", dq * 100)
        # Fall back: only score holdings with their own features (baidu-backed)
        h_features = build_latest_features(holding_codes, include_fundamentals=True,
                                           intraday_bars=intraday_bars)
        if h_features.empty:
            return pd.DataFrame()
        raw = ranker.predict_raw(h_features)
        fallback = pd.DataFrame({
            "code": h_features["code"].values,
            "ml_score": raw,
            "rank_pct": np.nan,  # rank unavailable
            "raw_return": raw,
        })
        if "_data_warnings" in h_features.columns:
            fallback["_data_warnings"] = h_features["_data_warnings"].values
        return fallback

    blend_scores = ranker.predict(features)
    raw_scores = ranker.predict_raw(features)

    full_df = pd.DataFrame({
        "code": features["code"].values,
        "ml_score": blend_scores,
        "rank_pct": blend_scores,
        "raw_return": raw_scores,
    })
    if "_data_warnings" in features.columns:
        full_df["_data_warnings"] = features["_data_warnings"].values

    return full_df[full_df["code"].isin(holding_codes)].reset_index(drop=True)


# =====================================================================
# Holdings Evaluation
# =====================================================================

def evaluate_holdings(ranker, regime: MarketRegime | None = None, intraday_bars: dict | None = None) -> pd.DataFrame:
    """Evaluate current holdings with ML model.

    Returns DataFrame with: code, name, ml_score, predicted_return, suggestion,
    rank_pct.  Thresholds and display adapt to ranker.score_type:
      - "rank_percentile" (BlendRanker): percentile thresholds, rank in ZZ500 pool
      - "predicted_return" (StockRanker): absolute return thresholds
    """
    holdings = load_holdings()
    if not holdings:
        logger.warning("No A-share holdings found")
        return pd.DataFrame()

    codes = [h["code"] for h in holdings]
    name_map = {h["code"]: h["name"] for h in holdings}

    is_rank = getattr(ranker, "score_type", "predicted_return") == "rank_percentile"

    if is_rank:
        logger.info("Scoring {} holdings within ZZ500 universe...", len(codes))
        result = score_universe(ranker, codes, intraday_bars=intraday_bars)
        if result.empty:
            logger.error("Failed to score holdings in universe")
            return pd.DataFrame()
        # score_universe builds features internally
        features = build_latest_features(codes, include_fundamentals=True,
                                         intraday_bars=intraday_bars)
    else:
        logger.info("Building features for {} holdings...", len(codes))
        features = build_latest_features(codes, include_fundamentals=True,
                                         intraday_bars=intraday_bars)
        if features.empty:
            logger.error("Failed to build features for holdings")
            return pd.DataFrame()
        scores = ranker.predict(features)
        result = pd.DataFrame({
            "code": features["code"].values,
            "ml_score": scores,
            "rank_pct": pd.Series(scores).rank(pct=True).values,
            "raw_return": scores,
        })
        if "_data_warnings" in features.columns:
            result["_data_warnings"] = features["_data_warnings"].values

    # Merge _data_warnings from features if not already present (is_rank branch)
    if "_data_warnings" not in result.columns and not features.empty and "_data_warnings" in features.columns:
        warn_map = dict(zip(features["code"].astype(str).str.zfill(6), features["_data_warnings"]))
        result["_data_warnings"] = result["code"].map(warn_map).fillna("")

    result["name"] = result["code"].map(name_map)
    result = result.sort_values("ml_score", ascending=False).reset_index(drop=True)

    if is_rank:
        # ── Absolute-return based recommendation ─────────────────────────────
        # BlendRanker predicts excess_ret (market-relative); to answer "should
        # I hold this stock?" we need estimated absolute return:
        #
        #   effective_ret = predicted_excess_ret + bench_ret_20d
        #
        # bench_ret_20d is ZZ500's actual 20-trading-day return fetched in
        # real time by RegimeDetector — not a fixed constant.  This means:
        #   bear market where ZZ500 dropped 8%: bench_adj = -8%
        #   bull market where ZZ500 rose 5%:    bench_adj = +5%
        #
        # Falls back to 0 if regime is unavailable.
        bench_adj = regime.bench_ret_20d if regime else 0.0

        result["effective_return"] = result["raw_return"] + bench_adj

        def suggest(eff):
            import math
            if eff is None or (isinstance(eff, float) and math.isnan(eff)):
                return "减仓"
            if eff > 0.04:
                return "加仓"
            elif eff > 0.00:
                return "持有"
            elif eff > -0.03:
                return "减仓"
            else:
                return "清仓"

        result["suggestion"] = result["effective_return"].apply(suggest)
        result["predicted_return"] = (result["effective_return"] * 100).round(2)
    else:
        # Original return-based thresholds with regime shift (±2pp)
        shift = regime.score * 0.02 if regime else 0.0

        def suggest(score):
            if score > 0.03 - shift:
                return "加仓"
            elif score > 0.00 - shift:
                return "持有"
            elif score > -0.03 - shift:
                return "减仓"
            else:
                return "清仓"

        result["suggestion"] = result["ml_score"].apply(suggest)
        result["predicted_return"] = (result["ml_score"] * 100).round(2)

    result["rank_pct"] = result["rank_pct"].round(4)

    # Get top factors for each holding
    importance = ranker.feature_importance_report()
    top_factors = importance.head(5)["feature"].tolist() if not importance.empty else []
    result["top_factors"] = ""
    for idx, row in result.iterrows():
        code = row["code"]
        feat_row = features[features["code"] == code]
        if not feat_row.empty and top_factors:
            parts = []
            for f in top_factors[:3]:
                if f in feat_row.columns:
                    val = feat_row[f].values[0]
                    if pd.notna(val):
                        parts.append(f"{f}={val:.3f}")
            result.at[idx, "top_factors"] = ", ".join(parts)

    return result


# =====================================================================
# 60-Day Technical Analysis
# =====================================================================

def evaluate_holdings_60d(codes: list[str]) -> list[dict]:
    """Compute 60-day technical indicators for each holding.

    Fetches ~150 bars per stock and calculates price position, MA comparisons,
    MA60 trend, momentum decomposition, volatility change, and volume trend.
    """
    from mp.data.fetcher import get_daily_bars

    # Go back ~8 months to ensure 150+ trading days
    start = (date.today() - pd.Timedelta(days=240)).strftime("%Y%m%d")
    results = []

    for code in codes:
        try:
            df = get_daily_bars(code, start=start)
        except Exception as e:
            logger.warning("Failed to fetch bars for {}: {}", code, e)
            results.append({"code": code, "error": str(e)})
            continue

        if df is None or df.empty or "date" not in df.columns:
            logger.warning("{}: no valid bar data returned", code)
            results.append({"code": code, "error": "no data"})
            continue

        df = df.sort_values("date").reset_index(drop=True)

        if len(df) < 60:
            logger.warning("{}: only {} bars, need at least 60", code, len(df))
            results.append({"code": code, "error": f"insufficient bars ({len(df)})"})
            continue

        close = df["close"].values.astype(float)
        volume = df["volume"].values.astype(float)
        current_price = close[-1]

        # --- Price position in 60-day range ---
        high_60 = np.max(close[-60:])
        low_60 = np.min(close[-60:])
        price_range = high_60 - low_60
        price_position = ((current_price - low_60) / price_range * 100) if price_range > 0 else 50.0

        # --- MA comparison ---
        ma20 = np.mean(close[-20:]) if len(close) >= 20 else np.nan
        ma60 = np.mean(close[-60:]) if len(close) >= 60 else np.nan
        ma120 = np.mean(close[-120:]) if len(close) >= 120 else np.nan

        def pct_vs(price, ma):
            if np.isnan(ma) or ma == 0:
                return np.nan
            return (price - ma) / ma * 100

        pct_vs_ma20 = pct_vs(current_price, ma20)
        pct_vs_ma60 = pct_vs(current_price, ma60)
        pct_vs_ma120 = pct_vs(current_price, ma120)

        above_ma20 = bool(current_price > ma20) if not np.isnan(ma20) else None
        above_ma60 = bool(current_price > ma60) if not np.isnan(ma60) else None
        above_ma120 = bool(current_price > ma120) if not np.isnan(ma120) else None

        # --- MA60 trend (compare current MA60 vs MA60 from 20 days ago) ---
        if len(close) >= 80:
            ma60_now = np.mean(close[-60:])
            ma60_prev = np.mean(close[-80:-20])
            ma60_trend_pct = (ma60_now - ma60_prev) / ma60_prev * 100 if ma60_prev != 0 else 0.0
            ma60_rising = bool(ma60_now > ma60_prev)
        else:
            ma60_trend_pct = np.nan
            ma60_rising = None

        # --- Momentum decomposition ---
        price_60d_ago = close[-61] if len(close) >= 61 else close[0]
        price_20d_ago = close[-21] if len(close) >= 21 else close[0]
        momentum_60d = (current_price - price_60d_ago) / price_60d_ago * 100 if price_60d_ago != 0 else 0.0
        momentum_first40 = (price_20d_ago - price_60d_ago) / price_60d_ago * 100 if price_60d_ago != 0 else 0.0
        momentum_last20 = (current_price - price_20d_ago) / price_20d_ago * 100 if price_20d_ago != 0 else 0.0

        # --- Volatility change ---
        if len(close) >= 21:
            returns_20d = np.diff(close[-21:]) / close[-21:-1]
            vol_20d = float(np.std(returns_20d, ddof=1) * np.sqrt(252) * 100)
        else:
            vol_20d = np.nan

        if len(close) >= 61:
            returns_60d = np.diff(close[-61:]) / close[-61:-1]
            vol_60d = float(np.std(returns_60d, ddof=1) * np.sqrt(252) * 100)
        else:
            vol_60d = np.nan

        # --- Volume trend ---
        if len(volume) >= 60:
            avg_vol_20 = np.mean(volume[-20:])
            avg_vol_60 = np.mean(volume[-60:])
            vol_ratio = avg_vol_20 / avg_vol_60 if avg_vol_60 > 0 else np.nan
        else:
            vol_ratio = np.nan

        results.append({
            "code": code,
            "price_position": round(price_position, 1),
            "pct_vs_ma20": round(pct_vs_ma20, 1) if not np.isnan(pct_vs_ma20) else None,
            "pct_vs_ma60": round(pct_vs_ma60, 1) if not np.isnan(pct_vs_ma60) else None,
            "pct_vs_ma120": round(pct_vs_ma120, 1) if not np.isnan(pct_vs_ma120) else None,
            "above_ma20": above_ma20,
            "above_ma60": above_ma60,
            "above_ma120": above_ma120,
            "ma60_rising": ma60_rising,
            "ma60_trend_pct": round(ma60_trend_pct, 1) if not np.isnan(ma60_trend_pct) else None,
            "momentum_60d": round(momentum_60d, 1),
            "momentum_first40": round(momentum_first40, 1),
            "momentum_last20": round(momentum_last20, 1),
            "vol_20d": round(vol_20d, 1) if not np.isnan(vol_20d) else None,
            "vol_60d": round(vol_60d, 1) if not np.isnan(vol_60d) else None,
            "vol_ratio": round(vol_ratio, 2) if not np.isnan(vol_ratio) else None,
        })

    return results


def format_60d_section(holdings_60d: list[dict], name_map: dict, scores_60d: dict | None = None) -> str:
    """Format 60-day technical analysis as markdown section."""
    lines = ["## 60日趋势分析", ""]

    for item in holdings_60d:
        code = item["code"]
        name = name_map.get(code, code)

        if "error" in item:
            lines.append(f"**{name}** ({code})")
            lines.append(f"  数据不足: {item['error']}")
            lines.append("")
            continue

        # --- ML 60d prediction (if available) ---
        pred_60d_line = None
        if scores_60d and code in scores_60d:
            ret_60d = scores_60d[code] * 100
            pred_60d_line = f"  预测60日收益: **{ret_60d:+.2f}%**"

        # --- Momentum description ---
        m60 = item["momentum_60d"]
        m40 = item["momentum_first40"]
        m20 = item["momentum_last20"]

        # Characterize the momentum shape
        if m40 < -5 and m20 > 5:
            shape = "超跌反弹"
        elif m40 > 5 and m20 < -5:
            shape = "冲高回落"
        elif m40 > 0 and m20 > 0:
            shape = "持续上涨"
        elif m40 < 0 and m20 < 0:
            shape = "持续下跌"
        else:
            shape = "震荡"

        momentum_line = (
            f"  60日动量: {m60:+.1f}% "
            f"(前40日{m40:+.1f}% → 后20日{m20:+.1f}%, {shape})"
        )

        # --- Price position line ---
        pos = item["price_position"]
        parts_pos = [f"60日区间{pos:.0f}%"]
        if item["pct_vs_ma60"] is not None:
            label = "上方" if item["above_ma60"] else "下方"
            parts_pos.append(f"MA60{label}{abs(item['pct_vs_ma60']):.1f}%")
        if item["pct_vs_ma120"] is not None:
            label = "上方" if item["above_ma120"] else "下方"
            parts_pos.append(f"MA120{label}{abs(item['pct_vs_ma120']):.1f}%")
        position_line = f"  价格位置: {' | '.join(parts_pos)}"

        # --- MA60 trend line ---
        if item["ma60_trend_pct"] is not None:
            trend_dir = "上行" if item["ma60_rising"] else "下行"
            ma60_line = f"  MA60趋势: {trend_dir}({item['ma60_trend_pct']:+.1f}%)"
        else:
            ma60_line = "  MA60趋势: 数据不足"

        # --- Vol & volume line ---
        vol_parts = []
        if item["vol_20d"] is not None:
            vol_parts.append(f"20日{item['vol_20d']:.0f}%")
        if item["vol_60d"] is not None:
            vol_parts.append(f"60日{item['vol_60d']:.0f}%")
        vol_str = " / ".join(vol_parts) if vol_parts else "N/A"
        vol_ratio_str = f"{item['vol_ratio']:.2f}x" if item["vol_ratio"] is not None else "N/A"
        vol_line = f"  波动率: {vol_str}  量能: {vol_ratio_str}"

        # --- 60-day outlook (signal counting) ---
        bulls = []
        bears = []

        if m60 > 5:
            bulls.append("正动量")
        elif m60 < -5:
            bears.append("负动量")

        if item["above_ma60"] is not None:
            if item["above_ma60"]:
                bulls.append("站上MA60")
            else:
                bears.append("低于MA60")

        if item["above_ma120"] is not None:
            if item["above_ma120"]:
                bulls.append("站上MA120")
            else:
                bears.append("低于MA120")

        if item["ma60_rising"] is not None:
            if item["ma60_rising"]:
                bulls.append("MA60上行")
            else:
                bears.append("MA60下行")

        if pos < 20:
            bulls.append("超卖区间")
        elif pos > 80:
            bears.append("超买区间")

        if m40 < -5 and m20 > 5:
            bears.append("超跌反弹形态")

        n_bull = len(bulls)
        n_bear = len(bears)

        if n_bull > n_bear:
            outlook = "偏多"
        elif n_bear > n_bull:
            outlook = "偏空"
        else:
            outlook = "均衡"

        signal_summary = f"{n_bull}多{n_bear}空"
        signal_detail = " vs ".join(
            filter(None, ["/".join(bulls) if bulls else None, "/".join(bears) if bears else None])
        )
        outlook_line = f"  60日展望: {outlook}({signal_summary}) — {signal_detail}"

        lines.append(f"**{name}** ({code})")
        if pred_60d_line:
            lines.append(pred_60d_line)
        lines.append(momentum_line)
        lines.append(position_line)
        lines.append(ma60_line)
        lines.append(vol_line)
        lines.append(outlook_line)
        lines.append("")

    return "\n".join(lines)


# =====================================================================
# Rebalance Advice
# =====================================================================

def generate_rebalance_advice(
    holdings_eval: pd.DataFrame,
    recommendations: pd.DataFrame,
    regime: MarketRegime | None = None,
) -> List[dict]:
    """Compare holdings vs ZZ500 top picks to generate swap suggestions.

    Transaction costs (≈0.15% round-trip) are factored into swap decisions.
    In bear regime, no new positions are recommended — only reduce/sell.

    Returns list of dicts with keys:
      action: "换仓" | "清仓" | "保持"
      sell_code, sell_name, sell_score
      buy_code, buy_name, buy_score  (None for 清仓/保持)
      reason: str
    """
    advice: List[dict] = []
    if holdings_eval.empty:
        return advice

    # If recommendations are degraded, skip swap advice
    if "_degraded_reason" in recommendations.columns:
        logger.info("荐股已降级，跳过换仓建议")
        return advice

    # Detect if we're in rank mode (BlendRanker) vs return mode (StockRanker)
    is_rank = "rank_pct" in holdings_eval.columns and holdings_eval["ml_score"].max() <= 1.05

    # Swap edge threshold: rank-based (5pp) or return-based (0.65%)
    if is_rank:
        swap_edge = 0.05  # 5 percentile points in rank space
    else:
        swap_edge = ROUND_TRIP_COST + 0.005  # ≈ 0.65% in return space
    if regime and regime.regime == "bull":
        swap_edge *= 0.7  # lower bar in bull market
    elif regime and regime.regime == "bear":
        swap_edge *= 1.5  # higher bar in bear (though candidates empty)

    # Held codes
    held_codes = set(holdings_eval["code"].tolist())

    # Candidate pool: top recs NOT already held, sorted by score desc
    # Bear market: no new positions
    candidates = []
    if regime and regime.regime == "bear":
        pass  # empty — bear market, don't recommend buying
    elif not recommendations.empty:
        for _, r in recommendations.iterrows():
            if r["code"] not in held_codes:
                candidates.append(r)

    # Walk through holdings from worst to best
    sorted_holdings = holdings_eval.sort_values("ml_score", ascending=True).reset_index(drop=True)
    candidate_idx = 0

    for _, h in sorted_holdings.iterrows():
        h_score = h["ml_score"]
        h_ret = h["predicted_return"]
        suggestion = h["suggestion"]

        # Case 1: model says 清仓 — always suggest selling
        if suggestion == "清仓":
            entry = {
                "action": "清仓",
                "sell_code": h["code"], "sell_name": h["name"], "sell_score": h_ret,
                "buy_code": None, "buy_name": None, "buy_score": None,
                "reason": f"预测收益 {h_ret:+.2f}%，模型看空",
            }
            # If there's a candidate that beats cost, upgrade to 换仓
            if candidate_idx < len(candidates):
                c = candidates[candidate_idx]
                if c["ml_score"] > h_score + swap_edge:
                    entry.update({
                        "action": "换仓",
                        "buy_code": c["code"], "buy_name": c["name"], "buy_score": c["predicted_return"],
                        "reason": f"预测收益 {h_ret:+.2f}% → {c['predicted_return']:+.2f}%（扣除交易成本后仍有优势）",
                    })
                    candidate_idx += 1
            advice.append(entry)

        # Case 2: model says 减仓 — swap if candidate beats by > swap_edge
        elif suggestion == "减仓":
            if candidate_idx < len(candidates):
                c = candidates[candidate_idx]
                if c["ml_score"] > h_score + swap_edge:
                    if is_rank:
                        c_rank = c.get("rank_pct", c["ml_score"])
                        h_rank = h.get("rank_pct", h_score)
                        reason = f"排名提升 前{(1-h_rank)*100:.0f}% → 前{(1-c_rank)*100:.0f}%，预测收益 {h_ret:+.2f}% → {c['predicted_return']:+.2f}%"
                    else:
                        reason = f"预测收益 {h_ret:+.2f}% → {c['predicted_return']:+.2f}%，扣费后净提升 {c['predicted_return'] - h_ret - ROUND_TRIP_COST * 100:+.2f}pp"
                    advice.append({
                        "action": "换仓",
                        "sell_code": h["code"], "sell_name": h["name"], "sell_score": h_ret,
                        "buy_code": c["code"], "buy_name": c["name"], "buy_score": c["predicted_return"],
                        "reason": reason,
                    })
                    candidate_idx += 1
                else:
                    advice.append({
                        "action": "保持",
                        "sell_code": h["code"], "sell_name": h["name"], "sell_score": h_ret,
                        "buy_code": None, "buy_name": None, "buy_score": None,
                        "reason": f"虽偏弱({h_ret:+.2f}%)，但{'排名差距不足' if is_rank else '扣除交易成本后无更优替代'}",
                    })
            else:
                advice.append({
                    "action": "保持",
                    "sell_code": h["code"], "sell_name": h["name"], "sell_score": h_ret,
                    "buy_code": None, "buy_name": None, "buy_score": None,
                    "reason": f"偏弱({h_ret:+.2f}%)，候选池已耗尽",
                })

        # Case 3: 持有 or 加仓 — keep, but swap if candidate beats by > 2x swap_edge
        else:
            if candidate_idx < len(candidates):
                c = candidates[candidate_idx]
                if c["ml_score"] > h_score + swap_edge * 2:
                    if is_rank:
                        c_rank = c.get("rank_pct", c["ml_score"])
                        h_rank = h.get("rank_pct", h_score)
                        reason = f"发现更优标的: 排名 前{(1-h_rank)*100:.0f}% → 前{(1-c_rank)*100:.0f}%，预测收益 {h_ret:+.2f}% → {c['predicted_return']:+.2f}%"
                    else:
                        reason = f"发现更优标的: {h_ret:+.2f}% → {c['predicted_return']:+.2f}%，扣费后净提升 {c['predicted_return'] - h_ret - ROUND_TRIP_COST * 100:+.2f}pp"
                    advice.append({
                        "action": "换仓",
                        "sell_code": h["code"], "sell_name": h["name"], "sell_score": h_ret,
                        "buy_code": c["code"], "buy_name": c["name"], "buy_score": c["predicted_return"],
                        "reason": reason,
                    })
                    candidate_idx += 1
                else:
                    advice.append({
                        "action": "保持",
                        "sell_code": h["code"], "sell_name": h["name"], "sell_score": h_ret,
                        "buy_code": None, "buy_name": None, "buy_score": None,
                        "reason": f"预测收益 {h_ret:+.2f}%，{suggestion}",
                    })
            else:
                advice.append({
                    "action": "保持",
                    "sell_code": h["code"], "sell_name": h["name"], "sell_score": h_ret,
                    "buy_code": None, "buy_name": None, "buy_score": None,
                    "reason": f"预测收益 {h_ret:+.2f}%，{suggestion}",
                })

    return advice


# =====================================================================
# Stock Recommendations
# =====================================================================

def recommend_stocks(ranker, n_recommend: int = 5, intraday_bars: dict | None = None) -> Tuple[pd.DataFrame, list[dict], dict]:
    """Scan ZZ500 for top ML-scored stocks.

    Returns (top_df, rec_60d, rec_name_map):
      - top_df: top N stocks by ML predicted return
      - rec_60d: 60-day analysis for each recommended stock
      - rec_name_map: code -> name mapping for recommended stocks
    """
    from mp.data.fetcher import get_index_constituents

    logger.info("Fetching ZZ500 constituents...")
    try:
        codes = get_index_constituents("zz500")
    except Exception as e:
        logger.error("Failed to get ZZ500 constituents: {}", e)
        return pd.DataFrame(), [], {}

    logger.info("Building features for {} ZZ500 stocks...", len(codes))
    features = build_latest_features(codes, include_fundamentals=True,
                                     intraday_bars=intraday_bars)
    if features.empty:
        logger.error("Failed to build features for ZZ500")
        return pd.DataFrame(), [], {}

    # --- Data quality gate: degrade if fundamentals largely missing ---
    dq = features.attrs.get("_data_quality", 1.0)
    if dq < 0.5:
        logger.error("⚠ 数据源异常: 仅{:.0f}%股票有完整基本面数据，荐股模块降级", dq * 100)
        degraded = pd.DataFrame({"_degraded_reason": ["数据源异常：PE/PB等基本面数据大面积缺失，荐股结果不可信"]})
        return degraded, [], {}

    scores = ranker.predict(features)
    result = pd.DataFrame({
        "code": features["code"].values,
        "ml_score": scores,
    })
    if "_data_warnings" in features.columns:
        result["_data_warnings"] = features["_data_warnings"].values

    is_rank = getattr(ranker, "score_type", "predicted_return") == "rank_percentile"
    if is_rank:
        raw = ranker.predict_raw(features)
        result["predicted_return"] = (pd.Series(raw) * 100).round(2)
        result["rank_pct"] = result["ml_score"].round(4)
    else:
        result["predicted_return"] = (result["ml_score"] * 100).round(2)
        result["rank_pct"] = pd.Series(scores).rank(pct=True).round(4)

    # Sort and take top N, then fetch names only for those
    result = result.sort_values("ml_score", ascending=False).head(n_recommend).reset_index(drop=True)
    rec_name_map = get_stock_names(result["code"].tolist())
    result["name"] = result["code"].map(rec_name_map)

    # Top factors for each recommended stock
    importance = ranker.feature_importance_report()
    top_factors = importance.head(5)["feature"].tolist() if not importance.empty else []
    result["top_factors"] = ""
    for idx, row in result.iterrows():
        code = row["code"]
        feat_row = features[features["code"] == code]
        if not feat_row.empty and top_factors:
            parts = []
            for f in top_factors[:3]:
                if f in feat_row.columns:
                    val = feat_row[f].values[0]
                    if pd.notna(val):
                        parts.append(f"{f}={val:.3f}")
            result.at[idx, "top_factors"] = ", ".join(parts)

    # 60-day analysis for recommended stocks
    rec_codes = result["code"].tolist()
    logger.info("--- 60-day analysis for {} recommendations ---", len(rec_codes))
    rec_60d = evaluate_holdings_60d(rec_codes) if rec_codes else []

    return result, rec_60d, rec_name_map


# =====================================================================
# Report Formatting
# =====================================================================

def _collect_data_timestamps(codes: list[str]) -> dict:
    """Query DB for data freshness info to include in report footer."""
    from datetime import datetime as _dt
    result = {"generated_at": _dt.now().strftime("%H:%M"), "bar_date": "N/A", "valuation_date": "N/A"}
    try:
        from mp.data.store import DataStore
        from sqlalchemy import text
        store = DataStore()
        with store.engine.connect() as conn:
            if codes:
                placeholders = ",".join(f":c{i}" for i in range(len(codes)))
                params = {f"c{i}": c for i, c in enumerate(codes)}
                row = conn.execute(
                    text(f"SELECT MAX(date) FROM daily_bars WHERE code IN ({placeholders})"),
                    params
                ).scalar()
                if row:
                    result["bar_date"] = str(row)[:10]
            val_row = conn.execute(text("SELECT MAX(date) FROM valuation")).scalar()
            if val_row:
                result["valuation_date"] = str(val_row)[:10]
    except Exception as e:
        from loguru import logger
        logger.debug("Failed to collect data timestamps: {}", e)
    return result


def format_report(
    holdings_eval: pd.DataFrame,
    recommendations: pd.DataFrame,
    holdings_60d: list[dict] | None = None,
    name_map: dict | None = None,
    rec_60d: list[dict] | None = None,
    rec_name_map: dict | None = None,
    holdings_60d_scores: dict | None = None,
    rec_60d_scores: dict | None = None,
    holdings_screen: pd.DataFrame | None = None,
    rec_screen: pd.DataFrame | None = None,
    rebalance_advice: list[dict] | None = None,
    regime: MarketRegime | None = None,
    midday: bool = False,
    data_timestamps: dict | None = None,
) -> str:
    """Format evaluation results as markdown string (saved to file)."""
    return _format_markdown(
        holdings_eval, recommendations, holdings_60d, name_map,
        rec_60d, rec_name_map, holdings_60d_scores, rec_60d_scores,
        holdings_screen, rec_screen, rebalance_advice, regime,
        midday=midday,
        data_timestamps=data_timestamps,
    )


def _format_markdown(
    holdings_eval, recommendations, holdings_60d, name_map,
    rec_60d, rec_name_map, holdings_60d_scores, rec_60d_scores,
    holdings_screen, rec_screen, rebalance_advice=None, regime=None,
    midday: bool = False,
    data_timestamps: dict | None = None,
) -> str:
    """Plain markdown format (for file saving)."""
    today = date.today().strftime("%Y-%m-%d")
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    wd = weekday_cn[date.today().weekday()]

    title = "午间快报" if midday else "每日持仓评估报告"
    lines = [f"# {title}", f"**{today} {wd}**", ""]

    # --- Market regime ---
    if regime:
        regime_icon = {"bull": "🟢", "sideways": "🟡", "bear": "🔴"}.get(regime.regime, "⚪")
        lines.append("## 市场环境")
        lines.append(f"{regime_icon} **{regime.label_cn}** (得分: {regime.score:+.2f})")
        lines.append("")
        lines.append("| 信号 | 方向 | 明细 |")
        lines.append("|------|------|------|")
        for key, name_cn in [("index_trend", "指数趋势"), ("northbound", "北向资金"), ("margin", "两融余额")]:
            sig = regime.signals.get(key, {})
            direction = sig.get("direction", "未知")
            detail = sig.get("detail", "-")
            d_icon = {"偏多": "🟢", "偏空": "🔴", "中性": "🟡"}.get(direction, "⚪")
            lines.append(f"| {name_cn} | {d_icon} {direction} | {detail} |")
        lines.append("")
        if regime.regime == "bear":
            lines.append("> **熊市环境：个股建议阈值已上调，优先减仓/清仓，不建议新开仓。**")
        elif regime.regime == "bull":
            lines.append("> **牛市环境：个股建议阈值已下调，可积极配置。**")
        else:
            lines.append("> **震荡环境：建议阈值基本不变，关注个股分化。**")
        lines.append("")

    # --- Data quality global warning ---
    _has_data_warnings = False
    for _df in [holdings_eval, recommendations]:
        if not _df.empty and "_data_warnings" in _df.columns:
            if (_df["_data_warnings"].fillna("") != "").any():
                _has_data_warnings = True
                break
    if _has_data_warnings:
        lines.append("> ⚠️ **数据源异常**: 部分股票缺少基本面数据(PE/PB等)，相关预测可能不准确，请谨慎参考。")
        lines.append("")

    lines.append("## 持仓评估")
    is_rank = "rank_pct" in holdings_eval.columns if not holdings_eval.empty else False
    if holdings_eval.empty:
        lines.append("*无 A 股持仓*")
    else:
        lines.append("")
        for _, row in holdings_eval.iterrows():
            icon = {"加仓": "🟢", "持有": "🟡", "减仓": "🟠", "清仓": "🔴"}.get(row["suggestion"], "⚪")
            if is_rank and pd.notna(row.get("rank_pct")):
                score_text = f"综合排名: **前{(1-row['rank_pct'])*100:.1f}%**  预测收益: **{row['predicted_return']:+.2f}%**"
            else:
                score_text = f"预测20日收益: **{row['predicted_return']:+.2f}%**"
            # Realtime price info (midday report)
            rt_text = ""
            if pd.notna(row.get("realtime_price")):
                sign = "+" if row["realtime_change_pct"] >= 0 else ""
                rt_text = f"  现价: ¥{row['realtime_price']:.2f} ({sign}{row['realtime_change_pct']:.2f}%)"
            lines.append(
                f"**{row['name']}** ({row['code']}){rt_text}  "
                f"{score_text}  "
                f"建议: {icon} **{row['suggestion']}**"
            )
            if row.get("top_factors"):
                lines.append(f"  关键因子: {row['top_factors']}")
            if holdings_screen is not None and not holdings_screen.empty:
                match = holdings_screen[holdings_screen["code"] == row["code"]]
                if not match.empty:
                    sr = match.iloc[0]
                    lines.append(f"  信号评分: **{sr['signal_score']:.3f}** {sr['rating']}  (多{sr['bull']}空{sr['bear']}中{sr['neutral']})")
            warn = row.get("_data_warnings", "")
            if warn:
                lines.append(f"  🔸 **数据缺失({warn})** 预测仅供参考")
            lines.append("")

    # --- Rebalance Advice ---
    if rebalance_advice:
        lines.append("## 换仓建议")
        lines.append("")
        swaps = [a for a in rebalance_advice if a["action"] == "换仓"]
        sells = [a for a in rebalance_advice if a["action"] == "清仓"]
        keeps = [a for a in rebalance_advice if a["action"] == "保持"]

        if swaps:
            for a in swaps:
                lines.append(
                    f"- **换仓**: 卖出 {a['sell_name']}({a['sell_code']}) "
                    f"{a['sell_score']:+.2f}% → 买入 {a['buy_name']}({a['buy_code']}) "
                    f"{a['buy_score']:+.2f}%  _{a['reason']}_"
                )
        if sells:
            for a in sells:
                lines.append(
                    f"- **清仓**: {a['sell_name']}({a['sell_code']}) "
                    f"{a['sell_score']:+.2f}%  _{a['reason']}_"
                )
        if keeps:
            for a in keeps:
                lines.append(
                    f"- **保持**: {a['sell_name']}({a['sell_code']}) "
                    f"{a['sell_score']:+.2f}%  _{a['reason']}_"
                )

        if not swaps and not sells:
            lines.append("当前持仓无需调整")
        lines.append("")

    if holdings_60d and name_map:
        lines.append(format_60d_section(holdings_60d, name_map, scores_60d=holdings_60d_scores))
        lines.append("")

    lines.append("## 推荐关注 (ZZ500)")
    _rec_degraded = "_degraded_reason" in recommendations.columns if not recommendations.empty else False
    rec_is_rank = "rank_pct" in recommendations.columns if not recommendations.empty and not _rec_degraded else False
    if _rec_degraded:
        reason = recommendations["_degraded_reason"].values[0]
        lines.append(f"> 🔸 **荐股模块已降级**: {reason}")
    elif recommendations.empty:
        lines.append("*未能生成推荐*")
    else:
        lines.append("")
        for i, row in recommendations.iterrows():
            if rec_is_rank and pd.notna(row.get("rank_pct")):
                score_text = f"综合排名: **前{(1-row['rank_pct'])*100:.1f}%**  预测收益: **{row['predicted_return']:+.2f}%**"
            else:
                score_text = f"预测20日收益: **{row['predicted_return']:+.2f}%**"
            lines.append(
                f"{i+1}. **{row['name']}** ({row['code']})  "
                f"{score_text}"
            )
            if row.get("top_factors"):
                lines.append(f"   关键因子: {row['top_factors']}")
            if rec_screen is not None and not rec_screen.empty:
                match = rec_screen[rec_screen["code"] == row["code"]]
                if not match.empty:
                    sr = match.iloc[0]
                    lines.append(f"   信号评分: **{sr['signal_score']:.3f}** {sr['rating']}  (多{sr['bull']}空{sr['bear']}中{sr['neutral']})")
            warn = row.get("_data_warnings", "")
            if warn:
                lines.append(f"   🔸 **数据缺失({warn})** 预测仅供参考")
        lines.append("")

    if rec_60d and rec_name_map:
        lines.append("### 推荐股60日趋势")
        lines.append("")
        lines.append(format_60d_section(rec_60d, rec_name_map, scores_60d=rec_60d_scores))
        lines.append("")

    lines.append("---")
    ts = data_timestamps or {}
    bar_d = ts.get("bar_date", "N/A")
    val_d = ts.get("valuation_date", "N/A")
    gen_t = ts.get("generated_at", datetime.now().strftime("%H:%M"))
    lines.append(f"*行情数据: {bar_d} | 估值数据: {val_d} | 生成于: {gen_t}*")
    return "\n".join(lines)


def format_feishu_card(
    holdings_eval: pd.DataFrame,
    recommendations: pd.DataFrame,
    holdings_60d: list[dict] | None = None,
    name_map: dict | None = None,
    rec_60d: list[dict] | None = None,
    rec_name_map: dict | None = None,
    holdings_60d_scores: dict | None = None,
    rec_60d_scores: dict | None = None,
    holdings_screen: pd.DataFrame | None = None,
    rec_screen: pd.DataFrame | None = None,
    rebalance_advice: list[dict] | None = None,
    regime: MarketRegime | None = None,
    midday: bool = False,
    data_timestamps: dict | None = None,
) -> dict:
    """Format evaluation results as a Feishu interactive card JSON."""
    import json

    today = date.today().strftime("%Y-%m-%d")
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    wd = weekday_cn[date.today().weekday()]

    elements: list[dict] = []

    # --- Market regime ---
    if regime:
        regime_icon = {"bull": "🟢", "sideways": "🟡", "bear": "🔴"}.get(regime.regime, "⚪")
        regime_text = f"{regime_icon} **市场环境: {regime.label_cn}** (得分: {regime.score:+.2f})\n{regime.summary_cn}"
        elements.append({"tag": "markdown", "content": regime_text})
        elements.append({"tag": "hr"})

    # --- Data quality warning ---
    _feishu_has_warnings = False
    for _df in [holdings_eval, recommendations]:
        if not _df.empty and "_data_warnings" in _df.columns:
            if (_df["_data_warnings"].fillna("") != "").any():
                _feishu_has_warnings = True
                break
    if _feishu_has_warnings:
        elements.append({"tag": "markdown", "content": "⚠️ **数据源异常**: 部分股票缺少基本面数据(PE/PB等)，相关预测可能不准确"})
        elements.append({"tag": "hr"})

    # --- Holdings table ---
    is_rank = "rank_pct" in holdings_eval.columns if not holdings_eval.empty else False
    if not holdings_eval.empty:
        # Build table rows
        rows = []
        for _, row in holdings_eval.iterrows():
            icon = {"加仓": "🟢", "持有": "🟡", "减仓": "🟠", "清仓": "🔴"}.get(row["suggestion"], "⚪")
            if is_rank and pd.notna(row.get("rank_pct")):
                ret_str = f"前{(1-row['rank_pct'])*100:.1f}% ({row['predicted_return']:+.2f}%)"
            else:
                ret_str = f"{row['predicted_return']:+.2f}%"

            # Signal info
            sig_str = "-"
            rating_str = ""
            if holdings_screen is not None and not holdings_screen.empty:
                match = holdings_screen[holdings_screen["code"] == row["code"]]
                if not match.empty:
                    sr = match.iloc[0]
                    sig_str = f"{sr['signal_score']:.3f}"
                    rating_str = f" {sr['rating']}"

            # 60d prediction
            pred_60d = "-"
            if holdings_60d_scores and row["code"] in holdings_60d_scores:
                pred_60d = f"{holdings_60d_scores[row['code']] * 100:+.2f}%"

            # Realtime price (midday)
            rt_str = ""
            if pd.notna(row.get("realtime_price")):
                sign = "+" if row["realtime_change_pct"] >= 0 else ""
                color = "green" if row["realtime_change_pct"] >= 0 else "red"
                rt_str = f" | <font color='{color}'>{sign}{row['realtime_change_pct']:.2f}%</font>"

            warn_mark = " [缺]" if row.get("_data_warnings", "") else ""
            if rt_str:
                rows.append(f"| {row['name']}{rt_str} | {ret_str}{warn_mark} | {pred_60d} | {sig_str}{rating_str} | {icon}{row['suggestion']} |")
            else:
                rows.append(f"| {row['name']} | {ret_str}{warn_mark} | {pred_60d} | {sig_str}{rating_str} | {icon}{row['suggestion']} |")

        header_score = "排名(预测)" if is_rank else "预测20d"
        if any(pd.notna(r.get("realtime_price")) for _, r in holdings_eval.iterrows()):
            table_md = (
                f"| 股票 | 涨跌 | {header_score} | 预测60d | 信号评分 | 建议 |\n"
                "|---|---|---|---|---|---|\n"
                + "\n".join(rows)
            )
        else:
            table_md = (
                f"| 股票 | {header_score} | 预测60d | 信号评分 | 建议 |\n"
                "|---|---|---|---|---|\n"
                + "\n".join(rows)
            )
        elements.append({"tag": "markdown", "content": "**📋 持仓评估**"})
        elements.append({"tag": "markdown", "content": table_md})
    else:
        elements.append({"tag": "markdown", "content": "*无 A 股持仓*"})

    # --- Rebalance Advice ---
    if rebalance_advice:
        swaps = [a for a in rebalance_advice if a["action"] == "换仓"]
        sells = [a for a in rebalance_advice if a["action"] == "清仓"]
        keeps = [a for a in rebalance_advice if a["action"] == "保持"]

        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "**🔄 换仓建议**"})
        advice_lines = []
        for a in swaps:
            advice_lines.append(
                f"换仓: {a['sell_name']} {a['sell_score']:+.2f}% → "
                f"**{a['buy_name']}** {a['buy_score']:+.2f}%"
            )
        for a in sells:
            advice_lines.append(f"清仓: {a['sell_name']} {a['sell_score']:+.2f}%")
        for a in keeps:
            advice_lines.append(f"保持: {a['sell_name']} {a['sell_score']:+.2f}%")
        if not advice_lines:
            advice_lines.append("当前持仓无需调整")
        elements.append({"tag": "markdown", "content": "\n".join(advice_lines)})

    # --- Holdings 60d details (compact) ---
    if holdings_60d and name_map:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "**📈 持仓60日趋势**"})

        for item in holdings_60d:
            code = item["code"]
            nm = name_map.get(code, code)
            if "error" in item:
                elements.append({"tag": "markdown", "content": f"**{nm}**: 数据不足"})
                continue

            m60 = item["momentum_60d"]
            m40 = item["momentum_first40"]
            m20 = item["momentum_last20"]
            if m40 < -5 and m20 > 5:
                shape = "超跌反弹"
            elif m40 > 5 and m20 < -5:
                shape = "冲高回落"
            elif m40 > 0 and m20 > 0:
                shape = "持续上涨"
            elif m40 < 0 and m20 < 0:
                shape = "持续下跌"
            else:
                shape = "震荡"

            pos = item["price_position"]
            ma60_dir = "↑" if item.get("ma60_rising") else "↓"
            ma60_pct = f"{item['ma60_trend_pct']:+.1f}%" if item.get("ma60_trend_pct") is not None else ""

            detail = (
                f"**{nm}**  动量{m60:+.1f}%({shape}) | "
                f"区间{pos:.0f}% | MA60{ma60_dir}{ma60_pct}"
            )
            elements.append({"tag": "markdown", "content": detail})

    # --- Recommendations table ---
    _feishu_rec_degraded = "_degraded_reason" in recommendations.columns if not recommendations.empty else False
    rec_is_rank = "rank_pct" in recommendations.columns if not recommendations.empty and not _feishu_rec_degraded else False
    if _feishu_rec_degraded:
        reason = recommendations["_degraded_reason"].values[0]
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": f"🔸 **荐股模块已降级**: {reason}"})
    elif not recommendations.empty:
        elements.append({"tag": "hr"})
        rec_rows = []
        for i, row in recommendations.iterrows():
            if rec_is_rank and pd.notna(row.get("rank_pct")):
                ret_str = f"前{(1-row['rank_pct'])*100:.1f}% ({row['predicted_return']:+.2f}%)"
            else:
                ret_str = f"{row['predicted_return']:+.2f}%"

            sig_str = "-"
            rating_str = ""
            if rec_screen is not None and not rec_screen.empty:
                match = rec_screen[rec_screen["code"] == row["code"]]
                if not match.empty:
                    sr = match.iloc[0]
                    sig_str = f"{sr['signal_score']:.3f}"
                    rating_str = f" {sr['rating']}"

            pred_60d = "-"
            if rec_60d_scores and row["code"] in rec_60d_scores:
                pred_60d = f"{rec_60d_scores[row['code']] * 100:+.2f}%"

            warn_mark = " [缺]" if row.get("_data_warnings", "") else ""
            rec_rows.append(f"| {row['name']} | {ret_str}{warn_mark} | {pred_60d} | {sig_str}{rating_str} |")

        rec_header_score = "排名(预测)" if rec_is_rank else "预测20d"
        rec_table_md = (
            f"| 股票 | {rec_header_score} | 预测60d | 信号评分 |\n"
            "|---|---|---|---|\n"
            + "\n".join(rec_rows)
        )
        elements.append({"tag": "markdown", "content": "**🎯 推荐关注 (ZZ500)**"})
        elements.append({"tag": "markdown", "content": rec_table_md})

    # --- Recommendation 60d details ---
    if rec_60d and rec_name_map:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "**📊 推荐股60日趋势**"})

        for item in rec_60d:
            code = item["code"]
            nm = rec_name_map.get(code, code)
            if "error" in item:
                elements.append({"tag": "markdown", "content": f"**{nm}**: 数据不足"})
                continue

            m60 = item["momentum_60d"]
            m40 = item["momentum_first40"]
            m20 = item["momentum_last20"]
            if m40 < -5 and m20 > 5:
                shape = "超跌反弹"
            elif m40 > 5 and m20 < -5:
                shape = "冲高回落"
            elif m40 > 0 and m20 > 0:
                shape = "持续上涨"
            elif m40 < 0 and m20 < 0:
                shape = "持续下跌"
            else:
                shape = "震荡"

            pos = item["price_position"]
            ma60_dir = "↑" if item.get("ma60_rising") else "↓"
            ma60_pct = f"{item['ma60_trend_pct']:+.1f}%" if item.get("ma60_trend_pct") is not None else ""

            detail = (
                f"**{nm}**  动量{m60:+.1f}%({shape}) | "
                f"区间{pos:.0f}% | MA60{ma60_dir}{ma60_pct}"
            )
            elements.append({"tag": "markdown", "content": detail})

    # --- Footer ---
    elements.append({"tag": "hr"})
    ts = data_timestamps or {}
    bar_d = ts.get("bar_date", "N/A")
    val_d = ts.get("valuation_date", "N/A")
    gen_t = ts.get("generated_at", datetime.now().strftime("%H:%M"))
    prefix = "ML评分基于当日上午行情" if midday else "Money Printer ML"
    elements.append({
        "tag": "note",
        "elements": [{"tag": "plain_text", "content": f"{prefix} · 行情: {bar_d} · 估值: {val_d} · {gen_t}"}],
    })

    card = {
        "header": {
            "title": {"tag": "plain_text", "content": f"{'📊 午间快报' if midday else '💰 每日持仓报告'} {today} {wd}"},
            "template": "wathet" if midday else "blue",
        },
        "elements": elements,
    }
    return card


# =====================================================================
# Feishu Send
# =====================================================================

def send_to_feishu(
    markdown: str,
    chat_id: Optional[str] = None,
    user_id: Optional[str] = None,
    card: Optional[dict] = None,
) -> bool:
    """Send report via lark-cli. Uses card (interactive) if provided, else markdown."""
    import json

    cmd = ["lark-cli", "im", "+messages-send", "--as", "bot"]

    if chat_id:
        cmd.extend(["--chat-id", chat_id])
    elif user_id:
        cmd.extend(["--user-id", user_id])
    else:
        cmd.extend(["--user-id", DEFAULT_USER_ID])

    if card:
        cmd.extend(["--msg-type", "interactive", "--content", json.dumps(card, ensure_ascii=False)])
    else:
        cmd.extend(["--markdown", markdown])

    logger.info("Sending report to Feishu...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("Report sent successfully")
            return True
        else:
            logger.error("Failed to send: {}", result.stderr or result.stdout)
            return False
    except Exception as e:
        logger.error("Feishu send error: {}", e)
        return False


# =====================================================================
# Midday Report (午间快报)
# =====================================================================

def _fetch_realtime_prices(codes: List[str]) -> Dict[str, dict]:
    """Fetch realtime prices via Sina quote API. Returns {code: {price, prev_close, change_pct, name}}."""
    import httpx

    result = {}
    symbols = []
    for code in codes:
        prefix = "sh" if code.startswith("6") else "sz"
        symbols.append(f"{prefix}{code}")

    try:
        url = f"https://hq.sinajs.cn/list={','.join(symbols)}"
        resp = httpx.get(url, headers={"Referer": "https://finance.sina.com.cn"}, timeout=10)
        for line in resp.text.strip().split("\n"):
            if "=" not in line:
                continue
            var_part, data_part = line.split("=", 1)
            sina_code = var_part.split("_")[-1]
            code = sina_code[2:]
            fields = data_part.strip('";\r').split(",")
            if len(fields) >= 10 and fields[0]:
                name = fields[0]
                # Sina fields: 0=name, 1=open, 2=prev_close, 3=current,
                #   4=high, 5=low, 6=bid, 7=ask, 8=volume(手), 9=amount(元)
                try:
                    prev_close = float(fields[2])
                    current = float(fields[3])
                    if prev_close > 0 and current > 0:
                        change_pct = (current / prev_close - 1) * 100
                        result[code] = {
                            "name": name,
                            "price": current,
                            "prev_close": prev_close,
                            "change_pct": change_pct,
                            "open": float(fields[1]),
                            "high": float(fields[4]),
                            "low": float(fields[5]),
                            "volume": float(fields[8]) * 100,  # 手→股
                            "amount": float(fields[9]),
                        }
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        logger.warning("Sina realtime fetch failed: {}", e)

    return result


def _fetch_index_realtime(index_code: str = "sh000905") -> Optional[dict]:
    """Fetch realtime index quote (default: ZZ500=sh000905)."""
    import httpx

    try:
        url = f"https://hq.sinajs.cn/list={index_code}"
        resp = httpx.get(url, headers={"Referer": "https://finance.sina.com.cn"}, timeout=10)
        line = resp.text.strip()
        if "=" not in line:
            return None
        _, data_part = line.split("=", 1)
        fields = data_part.strip('";\r').split(",")
        if len(fields) >= 4:
            name = fields[0]
            prev_close = float(fields[2])
            current = float(fields[3])
            if prev_close > 0 and current > 0:
                return {
                    "name": name,
                    "price": current,
                    "prev_close": prev_close,
                    "change_pct": (current / prev_close - 1) * 100,
                }
    except Exception as e:
        logger.warning("Sina index fetch failed: {}", e)
    return None


def format_midday_markdown(
    holdings_data: List[dict],
    index_data: Optional[dict] = None,
    regime: MarketRegime | None = None,
    data_timestamps: dict | None = None,
) -> str:
    """Format midday report as markdown."""
    today = date.today().strftime("%Y-%m-%d")
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    wd = weekday_cn[date.today().weekday()]

    lines = [f"# 午间快报", f"**{today} {wd}**", ""]

    # Regime status
    if regime:
        emoji = {"bull": "🟢", "sideways": "🟡", "bear": "🔴"}.get(regime.regime, "⚪")
        lines.append(f"市场环境: {emoji} **{regime.label_cn}** (得分: {regime.score:+.2f}) — {regime.summary_cn}")
        lines.append("")

    # Index
    if index_data:
        sign = "+" if index_data["change_pct"] >= 0 else ""
        lines.append(f"大盘: **{index_data['name']}** {index_data['price']:.2f} ({sign}{index_data['change_pct']:.2f}%)")
        lines.append("")

    # Holdings table
    lines.append("## 持仓上午表现")
    lines.append("")

    alerts = []
    for h in holdings_data:
        sign = "+" if h["change_pct"] >= 0 else ""
        icon = {"加仓": "🟢", "持有": "🟡", "减仓": "🟠", "清仓": "🔴"}.get(h.get("suggestion", ""), "⚪")
        suggestion = h.get("suggestion", "-")
        ret_str = f"{h.get('predicted_return', 0):+.2f}%" if h.get("predicted_return") is not None else "-"
        warn_str = f" [缺]" if h.get("_data_warnings") else ""

        lines.append(
            f"**{h['name']}** ({h['code']})  "
            f"现价: ¥{h['price']:.2f}  "
            f"涨跌: **{sign}{h['change_pct']:.2f}%**  "
            f"ML预测: {ret_str}{warn_str}  "
            f"建议: {icon} **{suggestion}**"
        )

        if h["change_pct"] <= -3:
            alerts.append(f"⚠️ **{h['name']}** 上午跌幅 {h['change_pct']:.2f}%，关注止损")
        elif h["change_pct"] >= 5:
            alerts.append(f"📈 **{h['name']}** 上午涨幅 +{h['change_pct']:.2f}%")

    lines.append("")

    # Alerts
    if alerts:
        lines.append("## 异动提醒")
        for a in alerts:
            lines.append(f"- {a}")
    else:
        lines.append("无异动")
    lines.append("")

    lines.append("---")
    ts = data_timestamps or {}
    bar_d = ts.get("bar_date", "N/A")
    val_d = ts.get("valuation_date", "N/A")
    gen_t = ts.get("generated_at", datetime.now().strftime("%H:%M"))
    lines.append(f"*行情数据: {bar_d} | 估值数据: {val_d} | 生成于: {gen_t}*")
    return "\n".join(lines)


def format_midday_feishu_card(
    holdings_data: List[dict],
    index_data: Optional[dict] = None,
    regime: MarketRegime | None = None,
    data_timestamps: dict | None = None,
) -> dict:
    """Format midday report as Feishu interactive card."""
    today = date.today().strftime("%Y-%m-%d")
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    wd = weekday_cn[date.today().weekday()]

    elements: list[dict] = []

    # Regime status
    if regime:
        emoji = {"bull": "🟢", "sideways": "🟡", "bear": "🔴"}.get(regime.regime, "⚪")
        elements.append({
            "tag": "markdown",
            "content": f"{emoji} 市场环境: **{regime.label_cn}** (得分: {regime.score:+.2f}) — {regime.summary_cn}",
        })

    # Index header
    if index_data:
        sign = "+" if index_data["change_pct"] >= 0 else ""
        idx_color = "green" if index_data["change_pct"] >= 0 else "red"
        elements.append({
            "tag": "markdown",
            "content": f"大盘: **{index_data['name']}** {index_data['price']:.2f} (<font color='{idx_color}'>{sign}{index_data['change_pct']:.2f}%</font>)"
        })

    # Holdings
    elements.append({"tag": "hr"})
    elements.append({"tag": "markdown", "content": "**持仓上午表现**"})

    alerts = []
    for h in holdings_data:
        sign = "+" if h["change_pct"] >= 0 else ""
        color = "green" if h["change_pct"] >= 0 else "red"
        icon = {"加仓": "🟢", "持有": "🟡", "减仓": "🟠", "清仓": "🔴"}.get(h.get("suggestion", ""), "⚪")
        suggestion = h.get("suggestion", "-")
        ret_str = f"{h.get('predicted_return', 0):+.2f}%" if h.get("predicted_return") is not None else "-"

        warn_mark = " [缺]" if h.get("_data_warnings") else ""
        elements.append({
            "tag": "markdown",
            "content": (
                f"**{h['name']}**({h['code']}) "
                f"¥{h['price']:.2f} "
                f"<font color='{color}'>{sign}{h['change_pct']:.2f}%</font> "
                f"| ML: {ret_str}{warn_mark} {icon}{suggestion}"
            ),
        })

        if h["change_pct"] <= -3:
            alerts.append(f"⚠️ **{h['name']}** 上午跌 {h['change_pct']:.2f}%")
        elif h["change_pct"] >= 5:
            alerts.append(f"📈 **{h['name']}** 上午涨 +{h['change_pct']:.2f}%")

    # Alerts
    if alerts:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "**异动提醒**\n" + "\n".join(alerts)})

    # Footer
    elements.append({"tag": "hr"})
    ts = data_timestamps or {}
    bar_d = ts.get("bar_date", "N/A")
    val_d = ts.get("valuation_date", "N/A")
    gen_t = ts.get("generated_at", datetime.now().strftime("%H:%M"))
    elements.append({
        "tag": "note",
        "elements": [{"tag": "plain_text", "content": f"ML评分基于昨日收盘 · 行情: {bar_d} · 估值: {val_d} · {gen_t}"}],
    })

    return {
        "header": {
            "title": {"tag": "plain_text", "content": f"📊 午间快报 {today} {wd}"},
            "template": "wathet",
        },
        "elements": elements,
    }


def run_midday(dry_run: bool = False, chat_id: Optional[str] = None, user_id: Optional[str] = None):
    """Run midday report — same pipeline as daily, with intraday bars injected."""
    from mp.data.fetcher import get_index_constituents

    if date.today().weekday() >= 5:
        logger.info("Weekend — skipping midday report")
        return

    logger.info("=== Midday Report: {} ===", date.today())

    # 1. Load models (same as daily)
    ranker = BlendRanker()
    if ranker.load():
        logger.info("Using BlendRanker (excess_ret + extreme30)")
    else:
        logger.info("Blend models not found, falling back to StockRanker")
        ranker = StockRanker()
        if not ranker.load():
            logger.error("No ML model found. Run training first.")
            return

    ranker_60d = StockRanker()
    has_60d_model = ranker_60d.load(path="data/model_60d.lgb")
    if not has_60d_model:
        logger.info("No 60d model — 60d predictions will be skipped")

    # 2. Detect market regime
    try:
        regime = RegimeDetector().detect()
        logger.info("Midday regime: {} (score={:.2f})", regime.label_cn, regime.score)
    except Exception as e:
        logger.warning("Regime detection failed (non-critical): {}", e)
        regime = None

    # 3. Fetch realtime OHLCV for all relevant stocks (holdings + ZZ500)
    holdings = load_holdings()
    if not holdings:
        logger.warning("No holdings — skipping midday report")
        return
    h_codes = [h["code"] for h in holdings]
    name_map = {h["code"]: h["name"] for h in holdings}

    try:
        zz500_codes = get_index_constituents("zz500")
    except Exception as e:
        logger.warning("ZZ500 constituents fetch failed: {}", e)
        zz500_codes = []

    all_codes = list(set(h_codes + zz500_codes))
    logger.info("Fetching realtime OHLCV for {} stocks...", len(all_codes))
    prices = _fetch_realtime_prices(all_codes)
    if not prices:
        logger.error("Failed to fetch realtime prices")
        return
    logger.info("Got realtime data for {} stocks", len(prices))

    # 4. Build intraday_bars dict
    intraday_bars: Dict[str, dict] = {}
    for code, p in prices.items():
        if p.get("open") and p["open"] > 0:
            intraday_bars[code] = {
                "date": date.today(),
                "open": p["open"],
                "high": p["high"],
                "low": p["low"],
                "close": p["price"],
                "volume": p["volume"],
                "amount": p["amount"],
            }

    # 5. Evaluate holdings (with intraday bars → fresh ML)
    logger.info("--- Evaluating holdings (with intraday data) ---")
    holdings_eval = evaluate_holdings(ranker, regime=regime, intraday_bars=intraday_bars)

    # 5b. Append realtime price/change to holdings_eval for display
    if not holdings_eval.empty:
        for idx, row in holdings_eval.iterrows():
            code = row["code"]
            if code in prices:
                holdings_eval.at[idx, "realtime_price"] = prices[code]["price"]
                holdings_eval.at[idx, "realtime_change_pct"] = prices[code]["change_pct"]

    # 6. 60-day technical analysis
    logger.info("--- 60-day technical analysis ---")
    holdings_60d = evaluate_holdings_60d(h_codes) if h_codes else []

    # 6b. 60d ML predictions
    holdings_60d_scores = {}
    if has_60d_model and h_codes:
        h_features = build_latest_features(h_codes, include_fundamentals=True,
                                           intraday_bars=intraday_bars)
        if not h_features.empty:
            s60 = ranker_60d.predict(h_features)
            for c, s in zip(h_features["code"].values, s60):
                holdings_60d_scores[c] = s
            logger.info("60d predictions for {} holdings", len(holdings_60d_scores))

    # 7. Signal screening for holdings
    holdings_screen = pd.DataFrame()
    if h_codes:
        logger.info("Running signal screening on holdings...")
        holdings_screen = screen_stocks(h_codes, names=name_map)
        logger.info("Holdings screened: {} stocks", len(holdings_screen))

    # 8. Recommend stocks from ZZ500 (with intraday bars)
    logger.info("--- Scanning ZZ500 for recommendations ---")
    recommendations, rec_60d, rec_name_map = recommend_stocks(
        ranker, n_recommend=5, intraday_bars=intraday_bars)

    # 8b. 60d ML predictions for recommendations
    rec_degraded = "_degraded_reason" in recommendations.columns
    rec_60d_scores = {}
    if has_60d_model and not recommendations.empty and not rec_degraded:
        rec_features = build_latest_features(recommendations["code"].tolist(),
                                             include_fundamentals=True,
                                             intraday_bars=intraday_bars)
        if not rec_features.empty:
            s60 = ranker_60d.predict(rec_features)
            for c, s in zip(rec_features["code"].values, s60):
                rec_60d_scores[c] = s
            logger.info("60d predictions for {} recommendations", len(rec_60d_scores))

    # 8c. Signal screening for recommendations
    rec_screen = pd.DataFrame()
    if not recommendations.empty and not rec_degraded:
        rec_codes = recommendations["code"].tolist()
        rec_nm = rec_name_map or {}
        logger.info("Running signal screening on recommendations...")
        rec_screen = screen_stocks(rec_codes, names=rec_nm)
        logger.info("Recommendations screened: {} stocks", len(rec_screen))

    # 9. Rebalance advice
    rebalance_advice = generate_rebalance_advice(holdings_eval, recommendations, regime=regime)
    n_swaps = sum(1 for a in rebalance_advice if a["action"] in ("换仓", "清仓"))
    logger.info("Rebalance advice: {} actions ({} swaps/sells)", len(rebalance_advice), n_swaps)

    # 10. Format report (same as daily, with midday=True)
    data_timestamps = _collect_data_timestamps(h_codes or [])
    report = format_report(
        holdings_eval, recommendations,
        holdings_60d=holdings_60d, name_map=name_map,
        rec_60d=rec_60d, rec_name_map=rec_name_map,
        holdings_60d_scores=holdings_60d_scores or None,
        rec_60d_scores=rec_60d_scores or None,
        holdings_screen=holdings_screen if not holdings_screen.empty else None,
        rec_screen=rec_screen if not rec_screen.empty else None,
        rebalance_advice=rebalance_advice,
        regime=regime,
        midday=True,
        data_timestamps=data_timestamps,
    )
    logger.info("Midday report generated ({} chars)", len(report))

    card = format_feishu_card(
        holdings_eval, recommendations,
        holdings_60d=holdings_60d, name_map=name_map,
        rec_60d=rec_60d, rec_name_map=rec_name_map,
        holdings_60d_scores=holdings_60d_scores or None,
        rec_60d_scores=rec_60d_scores or None,
        holdings_screen=holdings_screen if not holdings_screen.empty else None,
        rec_screen=rec_screen if not rec_screen.empty else None,
        rebalance_advice=rebalance_advice,
        regime=regime,
        midday=True,
        data_timestamps=data_timestamps,
    )

    # 11. Save & send
    if dry_run:
        logger.info("Dry run — not sending to Feishu")
        print(report)
    else:
        send_to_feishu(report, chat_id=chat_id, user_id=user_id, card=card)


# =====================================================================
# Main
# =====================================================================

def run(dry_run: bool = False, chat_id: Optional[str] = None, user_id: Optional[str] = None):
    """Run full daily report pipeline."""
    logger.info("=== Daily Report: {} ===", date.today())

    # Check if it's a trading day (skip weekends)
    if date.today().weekday() >= 5:
        logger.info("Weekend - skipping report")
        return

    # Load models — prefer BlendRanker, fall back to StockRanker
    ranker = BlendRanker()
    if ranker.load():
        logger.info("Using BlendRanker (excess_ret + extreme30)")
    else:
        logger.info("Blend models not found, falling back to StockRanker")
        ranker = StockRanker()
        if not ranker.load():
            logger.error("No ML model found at {}. Run training first.", MODEL_PATH)
            return

    ranker_60d = StockRanker()
    has_60d_model = ranker_60d.load(path="data/model_60d.lgb")
    if not has_60d_model:
        logger.info("No 60d model — 60d predictions will be skipped")

    # 0. Detect market regime
    try:
        regime = RegimeDetector().detect()
        logger.info("Market regime: {} (score={:.2f})", regime.label_cn, regime.score)
    except Exception as e:
        logger.warning("Regime detection failed (non-critical): {}", e)
        regime = None

    # 1. Evaluate holdings
    logger.info("--- Evaluating holdings ---")
    holdings_eval = evaluate_holdings(ranker, regime=regime)

    # 1b. 60-day technical analysis for holdings
    holdings = load_holdings()
    codes = [h["code"] for h in holdings]
    name_map = {h["code"]: h["name"] for h in holdings}
    logger.info("--- 60-day technical analysis ---")
    holdings_60d = evaluate_holdings_60d(codes) if codes else []

    # 1c. 60d ML predictions for holdings
    holdings_60d_scores = {}
    if has_60d_model and codes:
        h_features = build_latest_features(codes, include_fundamentals=True)
        if not h_features.empty:
            s60 = ranker_60d.predict(h_features)
            for c, s in zip(h_features["code"].values, s60):
                holdings_60d_scores[c] = s
            logger.info("60d predictions for {} holdings", len(holdings_60d_scores))

    # Signal screening for holdings
    holdings_screen = pd.DataFrame()
    if codes:
        logger.info("Running signal screening on holdings...")
        name_map_screen = name_map or {}
        holdings_screen = screen_stocks(codes, names=name_map_screen)
        logger.info(f"Holdings screened: {len(holdings_screen)} stocks")

    # 2. Recommend stocks from ZZ500
    logger.info("--- Scanning ZZ500 for recommendations ---")
    recommendations, rec_60d, rec_name_map = recommend_stocks(ranker, n_recommend=5)

    # 2b. 60d ML predictions for recommendations
    rec_degraded = "_degraded_reason" in recommendations.columns
    rec_60d_scores = {}
    if has_60d_model and not recommendations.empty and not rec_degraded:
        rec_features = build_latest_features(recommendations["code"].tolist(), include_fundamentals=True)
        if not rec_features.empty:
            s60 = ranker_60d.predict(rec_features)
            for c, s in zip(rec_features["code"].values, s60):
                rec_60d_scores[c] = s
            logger.info("60d predictions for {} recommendations", len(rec_60d_scores))

    # Signal screening for recommendations
    rec_screen = pd.DataFrame()
    if not recommendations.empty and not rec_degraded:
        rec_codes = recommendations["code"].tolist()
        rec_nm = rec_name_map or {}
        logger.info("Running signal screening on recommendations...")
        rec_screen = screen_stocks(rec_codes, names=rec_nm)
        logger.info(f"Recommendations screened: {len(rec_screen)} stocks")

    # 2c. Generate rebalance advice
    rebalance_advice = generate_rebalance_advice(holdings_eval, recommendations, regime=regime)
    n_swaps = sum(1 for a in rebalance_advice if a["action"] in ("换仓", "清仓"))
    logger.info("Rebalance advice: {} actions ({} swaps/sells)", len(rebalance_advice), n_swaps)

    # 3. Format report
    data_timestamps = _collect_data_timestamps(codes or [])
    report = format_report(
        holdings_eval, recommendations,
        holdings_60d=holdings_60d, name_map=name_map,
        rec_60d=rec_60d, rec_name_map=rec_name_map,
        holdings_60d_scores=holdings_60d_scores or None,
        rec_60d_scores=rec_60d_scores or None,
        holdings_screen=holdings_screen if not holdings_screen.empty else None,
        rec_screen=rec_screen if not rec_screen.empty else None,
        rebalance_advice=rebalance_advice,
        regime=regime,
        data_timestamps=data_timestamps,
    )
    logger.info("Report generated ({} chars)", len(report))

    # Save locally
    report_dir = PROJECT_ROOT / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"daily_{date.today().strftime('%Y%m%d')}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to {}", report_path)

    # 3b. Format Feishu card
    card = format_feishu_card(
        holdings_eval, recommendations,
        holdings_60d=holdings_60d, name_map=name_map,
        rec_60d=rec_60d, rec_name_map=rec_name_map,
        holdings_60d_scores=holdings_60d_scores or None,
        rec_60d_scores=rec_60d_scores or None,
        holdings_screen=holdings_screen if not holdings_screen.empty else None,
        rec_screen=rec_screen if not rec_screen.empty else None,
        rebalance_advice=rebalance_advice,
        regime=regime,
        data_timestamps=data_timestamps,
    )

    # 4. Send to Feishu
    if dry_run:
        logger.info("Dry run - not sending to Feishu")
        print(report)
    else:
        send_to_feishu(report, chat_id=chat_id, user_id=user_id, card=card)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Daily portfolio report")
    parser.add_argument("--dry-run", action="store_true", help="Generate report without sending")
    parser.add_argument("--midday", action="store_true", help="Run midday report (noon)")
    parser.add_argument("--chat-id", type=str, help="Feishu group chat ID (oc_xxx)")
    parser.add_argument("--user-id", type=str, help="Feishu user ID for DM (ou_xxx)")
    args = parser.parse_args()

    if args.midday:
        run_midday(dry_run=args.dry_run, chat_id=args.chat_id, user_id=args.user_id)
    else:
        run(dry_run=args.dry_run, chat_id=args.chat_id, user_id=args.user_id)
