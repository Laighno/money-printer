"""Daily portfolio evaluation & stock recommendation report.

Evaluates current holdings via ML model, scans ZZ500 for top picks,
generates position change suggestions, and sends report to Feishu.

Run daily via launchd or manually::

    python scripts/daily_report.py              # generate + send
    python scripts/daily_report.py --dry-run    # generate only, don't send
    python scripts/daily_report.py --chat-id oc_xxx  # send to group chat
"""

from __future__ import annotations

import json
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


def load_account() -> dict | None:
    """Load structured account snapshot (total_assets / cash_available / target_position_pct).

    Returns None if portfolio.yaml has no `account:` block — orderlist generation
    will then be skipped gracefully.
    """
    if not PORTFOLIO_PATH.exists():
        return None
    try:
        with open(PORTFOLIO_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        acct = cfg.get("account")
        if not acct:
            return None
        # Sanity: require total_assets & cash_available
        if "total_assets" not in acct or "cash_available" not in acct:
            return None
        return acct
    except Exception as e:
        logger.warning("load_account failed: {}", e)
        return None


def load_holdings_full() -> List[dict]:
    """Like load_holdings but includes shares/avg_cost/entry_date."""
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

def score_universe(ranker, holding_codes: list[str], intraday_bars: dict | None = None,
                   precomputed_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score holdings within ZZ500 universe for meaningful percentiles.

    BlendRanker's .rank(pct=True) only makes sense with a large pool.
    This scores holdings + ZZ500 together, then returns holdings rows.

    Returns DataFrame with columns: code, ml_score, rank_pct, raw_return.

    ``precomputed_features`` lets the caller (run() / run_midday()) build
    the panel ONCE for universe ∪ holdings and share it between
    score_universe and recommend_stocks.  Without this shared panel,
    a flaky API mid-run can have evaluate_holdings see one DB snapshot
    (no today close) and recommend_stocks see another (today close
    arrived between the two calls), producing inconsistent predictions
    for the same stock in the same report (2026-05-15 粤电力A bug:
    持仓 +1.98% vs 推荐 +4.75%).
    """
    from mp.data.fetcher import get_index_constituents

    is_rank = getattr(ranker, "score_type", "predicted_return") == "rank_percentile"

    if not is_rank:
        if precomputed_features is not None:
            features = precomputed_features[precomputed_features["code"].isin(holding_codes)].copy()
        else:
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

    # BlendRanker: score holdings + recommendation universe together
    if precomputed_features is not None:
        features = precomputed_features
    else:
        try:
            from mp.data.fetcher import get_recommendation_universe
            zz500_codes = get_recommendation_universe()   # HS300 + ZZ500 ≈ 800
        except Exception as e:
            logger.warning("Universe fetch failed in score_universe, scoring holdings only: {}", e)
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
        if precomputed_features is not None:
            h_features = precomputed_features[precomputed_features["code"].isin(holding_codes)].copy()
        else:
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

def evaluate_holdings(ranker, regime: MarketRegime | None = None, intraday_bars: dict | None = None,
                      precomputed_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """Evaluate current holdings with ML model.

    Returns DataFrame with: code, name, ml_score, predicted_return, suggestion,
    rank_pct.  Thresholds and display adapt to ranker.score_type:
      - "rank_percentile" (BlendRanker): percentile thresholds, rank in ZZ500 pool
      - "predicted_return" (StockRanker): absolute return thresholds

    ``precomputed_features`` — see :func:`score_universe`.
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
        result = score_universe(ranker, codes, intraday_bars=intraday_bars,
                                precomputed_features=precomputed_features)
        if result.empty:
            logger.error("Failed to score holdings in universe")
            return pd.DataFrame()
        # score_universe builds features internally (unless precomputed provided)
        if precomputed_features is not None:
            features = precomputed_features[precomputed_features["code"].isin(codes)].copy()
        else:
            features = build_latest_features(codes, include_fundamentals=True,
                                             intraday_bars=intraday_bars)
    else:
        logger.info("Building features for {} holdings...", len(codes))
        if precomputed_features is not None:
            features = precomputed_features[precomputed_features["code"].isin(codes)].copy()
        else:
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
        # ── Display + suggestion口径（2026-04-28 修正）─────────────────────────
        # 旧实现把模型预测超额（forward 20d）与中证500 trailing 20d 实际涨跌相加，
        # 概念混淆（forward + backward 时间窗），且在趋势市严重失真——大涨时
        # 几乎所有票显示正预测、大跌时全部显示负预测，全部被建议加仓/清仓。
        #
        # 修正后：
        #   • bench_adj = 长期均值常数（中证500 年化 ~6% / 12 ≈ +0.5%/月）
        #     —— 不假装能预测短期市场方向，给一个无偏的参考基线
        #   • suggestion 阈值改回基于 raw_return（即模型预测的 excess_ret），
        #     与回测的换仓决策口径一致（横截面排名 + 超额）
        #   • effective_return 仍然显示，作为绝对收益参考，但不进入决策
        #
        # 历史数据（2015-2026）证实 A 股的简单"跌 X% 空仓"规则**不可靠**
        # （详见 BASELINE.md），故不加任何择时空仓信号。市场风险敞口由
        # broker 层的止损 / 追踪止损 / 组合熔断兜底。
        LONG_TERM_BENCH_20D = 0.005   # 中证500 长期年化 ~6% / 12 个月 ≈ 0.5%

        result["bench_adj_long_term"] = LONG_TERM_BENCH_20D
        result["effective_return"] = result["raw_return"] + LONG_TERM_BENCH_20D

        def suggest(excess):
            """基于模型预测超额（raw_return）做换仓建议。
            阈值与回测换仓口径一致：超额排序是首要决策依据。"""
            import math
            if excess is None or (isinstance(excess, float) and math.isnan(excess)):
                return "减仓"
            if excess > 0.03:        # 模型看好（超额 > 3%）
                return "加仓"
            elif excess > 0.00:      # 模型看平偏多
                return "持有"
            elif excess > -0.03:     # 模型小幅看空
                return "减仓"
            else:                    # 模型明显看空
                return "清仓"

        # suggestion 用 raw_return（excess），不用 effective_return（绝对预期）
        result["suggestion"] = result["raw_return"].apply(suggest)
        result["predicted_return"] = (result["effective_return"] * 100).round(2)
        result["predicted_excess"] = (result["raw_return"] * 100).round(2)
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
                "reason": f"绝对参考 {h_ret:+.2f}%，模型看空",
            }
            # If there's a candidate that beats cost, upgrade to 换仓
            if candidate_idx < len(candidates):
                c = candidates[candidate_idx]
                if c["ml_score"] > h_score + swap_edge:
                    entry.update({
                        "action": "换仓",
                        "buy_code": c["code"], "buy_name": c["name"], "buy_score": c["predicted_return"],
                        "reason": f"绝对参考 {h_ret:+.2f}% → {c['predicted_return']:+.2f}%（扣除交易成本后仍有优势）",
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
                        reason = f"排名提升 前{(1-h_rank)*100:.0f}% → 前{(1-c_rank)*100:.0f}%，绝对参考 {h_ret:+.2f}% → {c['predicted_return']:+.2f}%"
                    else:
                        reason = f"绝对参考 {h_ret:+.2f}% → {c['predicted_return']:+.2f}%，扣费后净提升 {c['predicted_return'] - h_ret - ROUND_TRIP_COST * 100:+.2f}pp"
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
                        reason = f"发现更优标的: 排名 前{(1-h_rank)*100:.0f}% → 前{(1-c_rank)*100:.0f}%，绝对参考 {h_ret:+.2f}% → {c['predicted_return']:+.2f}%"
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
                        "reason": f"绝对参考 {h_ret:+.2f}%，{suggestion}",
                    })
            else:
                advice.append({
                    "action": "保持",
                    "sell_code": h["code"], "sell_name": h["name"], "sell_score": h_ret,
                    "buy_code": None, "buy_name": None, "buy_score": None,
                    "reason": f"绝对参考 {h_ret:+.2f}%，{suggestion}",
                })

    return advice


# =====================================================================
# Stock Recommendations
# =====================================================================

# =====================================================================
# Midday → Afternoon Watchlist Tracking
# =====================================================================
# Midday report saves its top-N recommendations as a "watchlist" so the
# afternoon report can show how those picks performed once the full day's
# information is in.  Mismatch between midday and afternoon is structural
# (midday only sees morning session), so this gives the user transparency
# into which midday picks held up vs faded.

WATCHLIST_DIR = Path("data/reports/watchlist")


def _save_midday_watchlist(recs: pd.DataFrame, today: date) -> None:
    """Persist midday recommendations so afternoon report can track them."""
    if recs is None or recs.empty:
        return
    if "_degraded_reason" in recs.columns:
        return  # don't save degraded mode placeholder
    rows = []
    for _, r in recs.iterrows():
        def _f(v):
            try:
                return float(v) if pd.notna(v) else None
            except (TypeError, ValueError):
                return None
        rows.append({
            "code": str(r["code"]),
            "name": r.get("name", "") or "",
            "midday_excess": _f(r.get("predicted_excess")),
            "midday_return": _f(r.get("predicted_return")),
            "midday_rank_pct": _f(r.get("rank_pct")),
        })
    p = WATCHLIST_DIR / f"{today.strftime('%Y%m%d')}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Midday watchlist saved: {} stocks → {}", len(rows), p)


def _load_midday_watchlist(today: date) -> list[dict]:
    """Load today's midday watchlist (empty list if midday didn't run)."""
    p = WATCHLIST_DIR / f"{today.strftime('%Y%m%d')}.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load midday watchlist: {}", e)
        return []


def _evaluate_watchlist_afternoon(
    full_scored: pd.DataFrame | None,
    watchlist: list[dict],
) -> list[dict]:
    """Look up each watchlist code's afternoon rank/score from the full ZZ500 ranking.

    `full_scored` should be the full sorted DataFrame from recommend_stocks
    (one row per ZZ500 stock), with columns: code, predicted_excess,
    predicted_return, rank_pct, _rank.
    """
    if not watchlist:
        return []
    if full_scored is None or full_scored.empty:
        # Afternoon ranking unavailable — surface watchlist with no delta.
        return [{**w, "afternoon_rank": None, "afternoon_excess": None,
                 "excess_delta": None, "universe_size": None}
                for w in watchlist]
    n_universe = len(full_scored)
    out = []
    for w in watchlist:
        code = w["code"]
        match = full_scored[full_scored["code"] == code]
        if match.empty:
            out.append({**w, "afternoon_rank": None, "afternoon_excess": None,
                        "excess_delta": None, "universe_size": n_universe})
            continue
        row = match.iloc[0]
        a_excess = float(row["predicted_excess"]) if pd.notna(row.get("predicted_excess")) else None
        a_rank = int(row["_rank"]) if pd.notna(row.get("_rank")) else None
        a_rank_pct = float(row["rank_pct"]) if pd.notna(row.get("rank_pct")) else None
        m_excess = w.get("midday_excess")
        delta = (a_excess - m_excess) if (a_excess is not None and m_excess is not None) else None
        out.append({
            **w,
            "afternoon_rank": a_rank,
            "afternoon_rank_pct": a_rank_pct,
            "afternoon_excess": a_excess,
            "excess_delta": delta,
            "universe_size": n_universe,
        })
    return out


def recommend_stocks(ranker, n_recommend: int = 5, intraday_bars: dict | None = None,
                     precomputed_features: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, list[dict], dict, pd.DataFrame]:
    """Scan ZZ500 for top ML-scored stocks.

    Returns (top_df, rec_60d, rec_name_map, full_scored):
      - top_df: top N stocks by ML predicted return
      - rec_60d: 60-day analysis for each recommended stock
      - rec_name_map: code -> name mapping for recommended stocks
      - full_scored: ALL ZZ500 stocks sorted by ml_score, with _rank column;
        used by afternoon report to evaluate the midday watchlist without
        re-running build_latest_features.  Empty DataFrame if scoring failed.

    ``precomputed_features`` — see :func:`score_universe`.
    """
    from mp.data.fetcher import get_recommendation_universe

    if precomputed_features is not None:
        features = precomputed_features
        codes = features["code"].tolist()
        logger.info("Using precomputed features panel: {} stocks", len(codes))
    else:
        logger.info("Fetching recommendation universe (HS300 + ZZ500)...")
        try:
            codes = get_recommendation_universe()
        except Exception as e:
            logger.error("Failed to get recommendation universe: {}", e)
            return pd.DataFrame(), [], {}, pd.DataFrame()

        logger.info("Building features for {} universe stocks...", len(codes))
        features = build_latest_features(codes, include_fundamentals=True,
                                         intraday_bars=intraday_bars)
    if features.empty:
        logger.error("Failed to build features for ZZ500")
        return pd.DataFrame(), [], {}, pd.DataFrame()

    # --- Data quality gate: degrade if fundamentals largely missing ---
    dq = features.attrs.get("_data_quality", 1.0)
    if dq < 0.5:
        logger.error("⚠ 数据源异常: 仅{:.0f}%股票有完整基本面数据，荐股模块降级", dq * 100)
        degraded = pd.DataFrame({"_degraded_reason": ["数据源异常：PE/PB等基本面数据大面积缺失，荐股结果不可信"]})
        return degraded, [], {}, pd.DataFrame()

    scores = ranker.predict(features)
    result = pd.DataFrame({
        "code": features["code"].values,
        "ml_score": scores,
    })
    if "_data_warnings" in features.columns:
        result["_data_warnings"] = features["_data_warnings"].values

    is_rank = getattr(ranker, "score_type", "predicted_return") == "rank_percentile"
    if is_rank:
        # BlendRanker: predict_raw 返回 primary 模型的 excess_ret 预测（小数）
        # 显示口径与 evaluate_holdings 保持一致：
        #   • predicted_excess = 模型超额预测百分比（决策依据）
        #   • predicted_return = predicted_excess + 长期均值 0.5%（绝对参考）
        # 不再用 trailing 20d 作为基准加项（详见 BASELINE.md）
        LONG_TERM_BENCH_20D = 0.005
        raw = ranker.predict_raw(features)
        raw_pct = pd.Series(raw)
        result["predicted_excess"] = (raw_pct * 100).round(2)
        result["predicted_return"] = ((raw_pct + LONG_TERM_BENCH_20D) * 100).round(2)
        result["rank_pct"] = result["ml_score"].round(4)
    else:
        result["predicted_return"] = (result["ml_score"] * 100).round(2)
        result["rank_pct"] = pd.Series(scores).rank(pct=True).round(4)

    # Sort full ranking, attach rank index, then take top N for display
    full_scored = result.sort_values("ml_score", ascending=False).reset_index(drop=True)
    full_scored["_rank"] = full_scored.index + 1
    result = full_scored.head(n_recommend).reset_index(drop=True)
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

    return result, rec_60d, rec_name_map, full_scored


# =====================================================================
# Report Formatting
# =====================================================================

# =====================================================================
# Next-day open order list (afternoon report only)
# =====================================================================
# Convert model recommendations + current portfolio into actionable orders
# you can fill straight into your broker at next open.
#
# Sizing: top-N conviction-weighted (weight ∝ max(predicted_excess, 0) +
# 0.5pp epsilon).  Target_value_per_stock = total_assets × target_pos_pct
# × weight.  Existing holdings in top-N are adjusted by delta; holdings
# outside top-30 trigger reduce/clear suggestions.
#
# Limit pricing: buy = close × 1.01, sell = close × 0.99.  Open auction
# noise typically < 1%; this gets you filled without paying market.

def _latest_closes(codes: List[str]) -> Dict[str, float]:
    """Latest close per code from DB (no API call)."""
    out: Dict[str, float] = {}
    if not codes:
        return out
    try:
        from mp.data.store import DataStore, DEFAULT_DB_URL
        from sqlalchemy import text
        store = DataStore(db_url=DEFAULT_DB_URL)
        with store.engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT code, close FROM daily_bars "
                "WHERE code IN :codes AND date = ("
                "  SELECT MAX(date) FROM daily_bars WHERE code = daily_bars.code"
                ")"
            ).bindparams(__import__("sqlalchemy").bindparam("codes", expanding=True)),
            {"codes": list(codes)}).fetchall()
            for code, close in rows:
                if close is not None:
                    out[str(code).zfill(6)] = float(close)
    except Exception as e:
        logger.warning("_latest_closes failed: {}", e)
    # Per-code fallback for any missing
    missing = [c for c in codes if c not in out]
    if missing:
        try:
            from mp.data.store import DataStore, DEFAULT_DB_URL
            from sqlalchemy import text
            store = DataStore(db_url=DEFAULT_DB_URL)
            with store.engine.connect() as conn:
                for code in missing:
                    row = conn.execute(text(
                        "SELECT close FROM daily_bars WHERE code = :c "
                        "ORDER BY date DESC LIMIT 1"
                    ), {"c": code}).fetchone()
                    if row and row[0] is not None:
                        out[code] = float(row[0])
        except Exception as e:
            logger.warning("_latest_closes per-code fallback failed: {}", e)
    return out


def generate_order_list(
    holdings_full: list[dict],
    account: dict | None,
    recommendations: pd.DataFrame,
    full_scored: pd.DataFrame | None,
) -> list[dict]:
    """Generate next-day open order list.

    Returns list of dicts: {code, name, action, shares, limit_price, cost, reason}.
    Empty list if account snapshot or recommendations missing.

    Logic:
      Top-N (default 5) → conviction-weighted target value, delta vs current
      In-portfolio + rank > 100 → 清仓
      In-portfolio + 30 < rank ≤ 100 → 减仓 50%
      In-portfolio + rank ≤ 30 (but not top N) → silent hold
    """
    if account is None or recommendations is None or recommendations.empty:
        return []
    if "_degraded_reason" in recommendations.columns:
        return []

    cash_available = float(account.get("cash_available", 0))
    # Buy budget: 95% of cash, leaving 5% for fees/slippage
    buy_budget = cash_available * 0.95

    # Filter out HS300 stocks: model is trained on ZZ500 only, predictions
    # on HS300 names are out-of-distribution extrapolations.  Surface them
    # in the report (with ⚠️ marker) but do NOT auto-include in orders.
    hs300 = _hs300_set()
    all_rec_codes = [str(c).zfill(6) for c in recommendations["code"].tolist()]
    rec_codes = [c for c in all_rec_codes if c not in hs300]
    if len(rec_codes) < len(all_rec_codes):
        skipped = [c for c in all_rec_codes if c in hs300]
        logger.info("Order list: skipped {} HS300 recs (extrapolation): {}",
                    len(skipped), skipped)
    held_shares = {str(h["code"]).zfill(6): int(h.get("shares", 0) or 0) for h in holdings_full}
    held_names = {str(h["code"]).zfill(6): h.get("name", h["code"]) for h in holdings_full}

    closes = _latest_closes(list(set(rec_codes) | set(held_shares.keys())))

    # Conviction weights — predicted_excess is in pp (already × 100)
    excess_map = {}
    rec_name_map = {}
    for _, r in recommendations.iterrows():
        c = str(r["code"]).zfill(6)
        ex = r.get("predicted_excess")
        excess_map[c] = float(ex) if pd.notna(ex) else 0.0
        rec_name_map[c] = r.get("name", c)

    # Rank lookup (full ZZ500 ranking)
    rank_map: Dict[str, int] = {}
    universe_size = 500
    if full_scored is not None and not full_scored.empty and "_rank" in full_scored.columns:
        for _, r in full_scored.iterrows():
            rank_map[str(r["code"]).zfill(6)] = int(r["_rank"])
        universe_size = len(full_scored)

    orders: list[dict] = []
    rec_set = set(rec_codes)

    # Pass 1: NEW BUYS only — top-N codes not currently held, conviction-weighted
    # over buy_budget (= cash × 95%).  Already-held top-N codes are signal
    # confirmation: model still likes them, no fresh action needed.
    new_buy_codes = [c for c in rec_codes if held_shares.get(c, 0) == 0]
    if new_buy_codes:
        epsilon = 0.5  # pp floor — weakest still gets a slice
        adj = {c: max(excess_map.get(c, 0.0), 0.0) + epsilon for c in new_buy_codes}
        s = sum(adj.values())
        weights = ({c: adj[c] / s for c in new_buy_codes}
                   if s > 0 else {c: 1.0 / len(new_buy_codes) for c in new_buy_codes})

        for c in new_buy_codes:
            close = closes.get(c)
            if not close or close <= 0:
                continue
            limit = round(close * 1.01, 2)
            slot_value = buy_budget * weights[c]
            shares = int(slot_value / limit / 100) * 100
            if shares < 100:
                continue
            money = shares * limit
            rank = rank_map.get(c)
            ex_pp = excess_map.get(c, 0.0)
            rank_str = f"#{rank}/{universe_size}" if rank else "—"
            orders.append({
                "code": c,
                "name": rec_name_map.get(c, c),
                "action": "买入",
                "shares": shares,
                "limit_price": limit,
                "cost": money,
                "reason": f"模型 {rank_str}, 超额 {ex_pp:+.2f}%",
            })

    # Pass 1b: held codes that ARE in top-N → silent confirmation, no order
    # (No rebalance toward "ideal weights" — avoids churn on noisy signals.)

    # Pass 2: holdings NOT in top-N — possible reduce/clear
    for code, current in held_shares.items():
        if code in rec_set or current <= 0:
            continue
        close = closes.get(code)
        if not close or close <= 0:
            continue
        rank = rank_map.get(code)
        if rank is None:
            continue  # not in ZZ500 — silent
        if rank > 100:
            shares = current
            action = "清仓"
            reason = f"模型排名 #{rank}/{universe_size}（已不在 Top 100）"
        elif rank > 30:
            shares = (current // 2 // 100) * 100
            if shares < 100:
                continue
            action = "减仓"
            reason = f"模型排名 #{rank}/{universe_size}（跌出 Top 30，减半仓）"
        else:
            continue  # in top 30 — silent hold

        limit = round(close * 0.99, 2)
        proceeds = shares * limit
        orders.append({
            "code": code,
            "name": held_names.get(code, code),
            "action": action,
            "shares": shares,
            "limit_price": limit,
            "cost": -proceeds,
            "reason": reason,
        })

    return orders


def format_order_list_section(orders: list[dict], account: dict | None) -> str:
    """Render the order list section as markdown.  Empty string if no orders."""
    if not orders:
        return ""
    lines = ["", "## 📋 明日开盘订单清单", ""]
    if account and account.get("cash_available") is not None:
        cash = float(account['cash_available'])
        lines.append(
            f"> 可用资金 ¥{cash:,.0f}（买单预算 = 可用 × 95% = ¥{cash * 0.95:,.0f}）"
        )
        lines.append("")

    lines.append("| 股票 | 方向 | 股数 | 限价 | 资金 | 原因 |")
    lines.append("|---|---|---:|---:|---:|---|")
    for o in orders:
        cost = o["cost"]
        cost_str = (f"-¥{abs(cost):,.0f}" if cost < 0 else f"+¥{cost:,.0f}")
        lines.append(
            f"| {o['name']} ({o['code']}) | {o['action']} | "
            f"{o['shares']:,} | ¥{o['limit_price']:.2f} | {cost_str} | {o['reason']} |"
        )

    net = sum(o["cost"] for o in orders)
    if account and account.get("cash_available") is not None:
        cash_after = float(account["cash_available"]) - net
        sign = "占用" if net > 0 else "释放"
        lines.append("")
        lines.append(
            f"> 净资金{sign}：¥{abs(net):,.0f}"
            f" | 执行后可用：¥{cash_after:,.0f}"
        )

    lines.append("")
    lines.append(
        "**报价规则**：买单 = 收盘 × 1.01，卖单 = 收盘 × 0.99。"
        "买不到/卖不掉就放弃，等次日。仓位按 conviction 加权（top-N 权重 ∝ 模型超额预测）。"
    )
    lines.append("")
    return "\n".join(lines)


def _make_data_timestamps(
    bar_fetched_at: "datetime | None" = None,
    intraday: bool = False,
) -> dict:
    """Build data_timestamps by querying actual data dates from DB.

    bar_fetched_at is only used for intraday (midday) reports where bars are
    live real-time quotes — in that case we show the fetch time because the
    data IS from right now.  For daily reports, bar data comes from DB close
    prices and we show the DB date so stale data is visible.

    Both bar_date and valuation_date use the most recent date with SUBSTANTIAL
    coverage (>= MIN_REPRESENTATIVE_ROWS), not raw MAX(date).  This avoids the
    failure mode where a stray partial-bar row for "today" (e.g. an ETF intraday
    quote accidentally upserted into daily_bars) makes the report claim 行情:
    today when 99% of stocks only have data through yesterday.
    """
    from mp.data.store import DataStore
    from sqlalchemy import text as _text

    # Threshold: a date must have at least this many rows to be reported as
    # "the data date".  Calibrated for ZZ500 (~500 stocks); set well below 500
    # so that minor data outages don't trigger fallbacks.
    MIN_REPRESENTATIVE_ROWS = 100

    fmt_dt = "%Y-%m-%d %H:%M"
    result = {
        "bar_date": "N/A",
        "valuation_date": "N/A",
        "generated_at": datetime.now().strftime(fmt_dt),
    }

    def _representative_date(conn, table: str) -> str:
        """Return the most recent date with >= MIN_REPRESENTATIVE_ROWS in *table*."""
        row = conn.execute(_text(
            f"SELECT date FROM {table} GROUP BY date "
            f"HAVING COUNT(*) >= :min_rows ORDER BY date DESC LIMIT 1"
        ), {"min_rows": MIN_REPRESENTATIVE_ROWS}).scalar()
        return str(row)[:10] if row else "N/A"

    try:
        store = DataStore()
        with store.engine.connect() as conn:
            bar_date = _representative_date(conn, "daily_bars")
            val_date = _representative_date(conn, "valuation")

            # If valuation snapshot is fresher than the latest EOD bars, it
            # means a real-time intraday snapshot was just refreshed.  Mark it
            # explicitly so the report doesn't misleadingly claim "估值: today"
            # when the underlying PE/PB are derived from morning prices, not EOD.
            if val_date != "N/A" and bar_date != "N/A" and val_date > bar_date:
                val_date = f"{val_date}(盘中)"

            result["valuation_date"] = val_date

            if intraday and bar_fetched_at is not None:
                # Midday: bars are live intraday quotes fetched right now
                result["bar_date"] = bar_fetched_at.strftime(fmt_dt)
            else:
                # Daily: latest DB date with substantial coverage
                result["bar_date"] = bar_date
    except Exception as e:
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
    midday_watchlist_eval: list[dict] | None = None,
    order_list: list[dict] | None = None,
    account: dict | None = None,
    session_label: str = "midday",
) -> str:
    """Format evaluation results as markdown string (saved to file)."""
    return _format_markdown(
        holdings_eval, recommendations, holdings_60d, name_map,
        rec_60d, rec_name_map, holdings_60d_scores, rec_60d_scores,
        holdings_screen, rec_screen, rebalance_advice, regime,
        midday=midday,
        data_timestamps=data_timestamps,
        midday_watchlist_eval=midday_watchlist_eval,
        order_list=order_list,
        account=account,
        session_label=session_label,
    )


def _format_markdown(
    holdings_eval, recommendations, holdings_60d, name_map,
    rec_60d, rec_name_map, holdings_60d_scores, rec_60d_scores,
    holdings_screen, rec_screen, rebalance_advice=None, regime=None,
    midday: bool = False,
    data_timestamps: dict | None = None,
    midday_watchlist_eval: list[dict] | None = None,
    order_list: list[dict] | None = None,
    account: dict | None = None,
    session_label: str = "midday",
) -> str:
    """Plain markdown format (for file saving)."""
    today = date.today().strftime("%Y-%m-%d")
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    wd = weekday_cn[date.today().weekday()]

    if midday:
        title = {"midday": "午间快报", "2pm": "盘中快报 14:00"}.get(session_label, "盘中快报")
    else:
        title = "每日持仓评估报告"
    _hs300 = _hs300_set()   # for ⚠️ extrapolation marking
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
    lines.append("> 口径说明：决策看**模型超额预测 + 综合排名**；"
                 "**绝对参考 = 模型超额 + 长期均值 +0.5%**（中证500 长期年化 6%/12 个月），"
                 "不再使用 trailing 20d 涨跌做加项。")
    is_rank = "rank_pct" in holdings_eval.columns if not holdings_eval.empty else False
    if holdings_eval.empty:
        lines.append("*无 A 股持仓*")
    else:
        lines.append("")
        for _, row in holdings_eval.iterrows():
            icon = {"加仓": "🟢", "持有": "🟡", "减仓": "🟠", "清仓": "🔴"}.get(row["suggestion"], "⚪")
            if is_rank and pd.notna(row.get("rank_pct")):
                excess = row.get("predicted_excess", float("nan"))
                if pd.notna(excess):
                    score_text = (f"排名: **前{(1-row['rank_pct'])*100:.1f}%**  "
                                  f"模型超额: **{excess:+.2f}%**  "
                                  f"绝对参考: {row['predicted_return']:+.2f}%")
                else:
                    score_text = f"排名: **前{(1-row['rank_pct'])*100:.1f}%**  绝对参考: {row['predicted_return']:+.2f}%"
            else:
                score_text = f"预测20日收益: **{row['predicted_return']:+.2f}%**"
            # Realtime price info (midday report)
            rt_text = ""
            if pd.notna(row.get("realtime_price")):
                sign = "+" if row["realtime_change_pct"] >= 0 else ""
                rt_text = f"  现价: ¥{row['realtime_price']:.2f} ({sign}{row['realtime_change_pct']:.2f}%)"
            extra_mark = _EXTRAPOLATION_MARK if row["code"] in _hs300 else ""
            lines.append(
                f"**{row['name']}**{extra_mark} ({row['code']}){rt_text}  "
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

    lines.append("## 推荐关注 (HS300 + ZZ500)")
    _has_hs300_in_recs = (
        not recommendations.empty
        and any(c in _hs300 for c in recommendations["code"].tolist())
    )
    if _has_hs300_in_recs:
        lines.append(_EXTRAPOLATION_FOOTNOTE)
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
                excess = row.get("predicted_excess", float("nan"))
                if pd.notna(excess):
                    score_text = (f"排名: **前{(1-row['rank_pct'])*100:.1f}%**  "
                                  f"模型超额: **{excess:+.2f}%**  "
                                  f"绝对参考: {row['predicted_return']:+.2f}%")
                else:
                    score_text = f"排名: **前{(1-row['rank_pct'])*100:.1f}%**  绝对参考: {row['predicted_return']:+.2f}%"
            else:
                score_text = f"预测20日收益: **{row['predicted_return']:+.2f}%**"
            extra_mark = _EXTRAPOLATION_MARK if row["code"] in _hs300 else ""
            lines.append(
                f"{i+1}. **{row['name']}**{extra_mark} ({row['code']})  "
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

    # --- Midday watchlist tracking ---
    # Shown in 14:00 intraday and 16:00 EOD reports.  Compares the model's
    # noon picks against the latest ranking to surface signal drift through
    # the trading day.  Skipped in the noon report itself (it's the SAVER).
    _show_wl = (midday and session_label != "midday") or (not midday)
    if _show_wl and midday_watchlist_eval:
        _later_label = {"2pm": "14:00", "midday": "现时"}.get(session_label, "下午")
        lines.append("## 午间观察列表跟踪")
        lines.append(f"> 中午推荐的股票，对比 {_later_label} 数据后的最新排名/超额预测变动。")
        lines.append("")
        lines.append(f"| 股票 | 中午超额 | {_later_label} 超额 | 变动 | {_later_label} 排名 |")
        lines.append("|---|---:|---:|---:|---|")
        for w in midday_watchlist_eval:
            name = w.get("name") or w["code"]
            code = w["code"]
            m_ex = w.get("midday_excess")
            a_ex = w.get("afternoon_excess")
            delta = w.get("excess_delta")
            a_rank = w.get("afternoon_rank")
            n_uni = w.get("universe_size")

            m_str = f"{m_ex:+.2f}%" if m_ex is not None else "—"
            a_str = f"{a_ex:+.2f}%" if a_ex is not None else "—"
            if delta is None:
                d_str = "—"
            else:
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                d_str = f"{arrow}{delta:+.2f}pp"
            if a_rank is None:
                r_str = "未在 ZZ500"
            elif n_uni:
                r_str = f"#{a_rank} / {n_uni}"
            else:
                r_str = f"#{a_rank}"
            lines.append(f"| {name} ({code}) | {m_str} | {a_str} | {d_str} | {r_str} |")
        lines.append("")

    # --- Next-day open order list (afternoon report only) ---
    if not midday and order_list:
        lines.append(format_order_list_section(order_list, account))

    lines.append("---")
    ts = data_timestamps or {}
    bar_d = ts.get("bar_date", "N/A")
    val_d = ts.get("valuation_date", "N/A")
    gen_t = ts.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    lines.append(f"*行情数据: {bar_d} | 估值数据: {val_d} | 生成于: {gen_t}*")
    return "\n".join(lines)


_INTRADAY_TITLE = {"midday": "📊 午间快报", "2pm": "⏰ 盘中快报 14:00"}


# Extrapolation marker: HS300 stocks are in the inference universe (since
# 2026-05-14 widening) but the production BlendRanker was trained on ZZ500
# only.  Predictions on HS300 names are out-of-distribution extrapolations
# until Layer 2 (HS300 historical constituent backfill) + Layer 3 (retrain)
# complete.  Mark them in reports and exclude from automated order lists.
_EXTRAPOLATION_MARK = " ⚠️*"
_EXTRAPOLATION_FOOTNOTE = (
    "> ⚠️* 标记：该股属于 HS300 大盘股，模型当前训练池只覆盖 ZZ500，"
    "对它的预测是**外推**——仅供参考，不进入换仓建议/订单清单。"
    "HS300 历史成分股 backfill + 模型重训完成后会移除此标记。"
)


def _hs300_set() -> set[str]:
    """Cached set of current HS300 codes (used to flag out-of-training-pool stocks)."""
    try:
        from mp.data.fetcher import get_index_constituents
        return set(get_index_constituents("hs300"))
    except Exception as e:
        logger.debug("HS300 fetch failed (extrapolation marker off): {}", e)
        return set()


def _card_title(midday: bool, session_label: str, today: str, wd: str) -> str:
    if not midday:
        return f"💰 每日持仓报告 {today} {wd}"
    prefix = _INTRADAY_TITLE.get(session_label, "📊 盘中快报")
    return f"{prefix} {today} {wd}"


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
    midday_watchlist_eval: list[dict] | None = None,
    order_list: list[dict] | None = None,
    account: dict | None = None,
    session_label: str = "midday",
) -> dict:
    """Format evaluation results as a Feishu interactive card JSON."""
    import json

    today = date.today().strftime("%Y-%m-%d")
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    wd = weekday_cn[date.today().weekday()]
    _hs300 = _hs300_set()   # for ⚠️ extrapolation marking

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
                excess = row.get("predicted_excess", float("nan"))
                if pd.notna(excess):
                    ret_str = f"前{(1-row['rank_pct'])*100:.1f}% (超额{excess:+.2f}%)"
                else:
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
            extrap = _EXTRAPOLATION_MARK if row["code"] in _hs300 else ""
            if rt_str:
                rows.append(f"| {row['name']}{extrap}{rt_str} | {ret_str}{warn_mark} | {pred_60d} | {sig_str}{rating_str} | {icon}{row['suggestion']} |")
            else:
                rows.append(f"| {row['name']}{extrap} | {ret_str}{warn_mark} | {pred_60d} | {sig_str}{rating_str} | {icon}{row['suggestion']} |")

        header_score = "排名(模型超额)" if is_rank else "预测20d"
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
                # Show pure model excess (consistent with holdings table at line 1134),
                # not predicted_return (which is excess + 0.5% baseline = '绝对参考').
                excess = row.get("predicted_excess", float("nan"))
                if pd.notna(excess):
                    ret_str = f"前{(1-row['rank_pct'])*100:.1f}% (超额{excess:+.2f}%)"
                else:
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
            extrap = _EXTRAPOLATION_MARK if row["code"] in _hs300 else ""
            rec_rows.append(f"| {row['name']}{extrap} | {ret_str}{warn_mark} | {pred_60d} | {sig_str}{rating_str} |")

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

    # --- Midday watchlist tracking ---
    _show_wl_card = (midday and session_label != "midday") or (not midday)
    if _show_wl_card and midday_watchlist_eval:
        _later_label_c = {"2pm": "14:00", "midday": "现时"}.get(session_label, "下午")
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": f"**🕛 午间观察列表跟踪 → {_later_label_c}**"})
        wl_rows = []
        for w in midday_watchlist_eval:
            nm = w.get("name") or w["code"]
            m_ex = w.get("midday_excess")
            a_ex = w.get("afternoon_excess")
            delta = w.get("excess_delta")
            a_rank = w.get("afternoon_rank")
            n_uni = w.get("universe_size")

            m_str = f"{m_ex:+.2f}%" if m_ex is not None else "—"
            a_str = f"{a_ex:+.2f}%" if a_ex is not None else "—"
            if delta is None:
                d_str = "—"
            else:
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                d_str = f"{arrow}{delta:+.2f}pp"
            if a_rank is None:
                r_str = "—"
            elif n_uni:
                r_str = f"#{a_rank}/{n_uni}"
            else:
                r_str = f"#{a_rank}"
            wl_rows.append(f"| {nm} | {m_str} | {a_str} | {d_str} | {r_str} |")
        wl_table = (
            f"| 股票 | 中午超额 | {_later_label_c} 超额 | 变动 | {_later_label_c} 排名 |\n"
            "|---|---|---|---|---|\n"
            + "\n".join(wl_rows)
        )
        elements.append({"tag": "markdown", "content": wl_table})

    # --- Next-day open order list (afternoon report only) ---
    if not midday and order_list:
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": "**📋 明日开盘订单清单**"})
        if account and account.get("cash_available") is not None:
            cash = float(account['cash_available'])
            elements.append({"tag": "markdown", "content":
                f"可用 ¥{cash:,.0f} · 买单预算 ¥{cash * 0.95:,.0f}"})
        ol_rows = []
        for o in order_list:
            cost = o["cost"]
            cost_str = (f"-¥{abs(cost):,.0f}" if cost < 0 else f"+¥{cost:,.0f}")
            ol_rows.append(
                f"| {o['name']} | {o['action']} | {o['shares']:,} | "
                f"¥{o['limit_price']:.2f} | {cost_str} | {o['reason']} |"
            )
        ol_table = (
            "| 股票 | 方向 | 股数 | 限价 | 资金 | 原因 |\n"
            "|---|---|---|---|---|---|\n"
            + "\n".join(ol_rows)
        )
        elements.append({"tag": "markdown", "content": ol_table})
        net = sum(o["cost"] for o in order_list)
        if account and account.get("cash_available") is not None:
            cash_after = float(account["cash_available"]) - net
            sign = "占用" if net > 0 else "释放"
            elements.append({"tag": "markdown", "content":
                f"净资金{sign} ¥{abs(net):,.0f} · 执行后 ¥{cash_after:,.0f}"})
        elements.append({"tag": "markdown", "content":
            "*买单 = 收盘×1.01，卖单 = 收盘×0.99；按 conviction 加权*"})

    # --- Footer ---
    elements.append({"tag": "hr"})
    ts = data_timestamps or {}
    bar_d = ts.get("bar_date", "N/A")
    val_d = ts.get("valuation_date", "N/A")
    gen_t = ts.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    prefix = "ML评分基于当日上午行情" if midday else "Money Printer ML"
    elements.append({
        "tag": "note",
        "elements": [{"tag": "plain_text", "content": f"{prefix} · 行情: {bar_d} · 估值: {val_d} · {gen_t}"}],
    })

    card = {
        "header": {
            "title": {"tag": "plain_text", "content": _card_title(midday, session_label, today, wd)},
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

def _trading_minutes_elapsed(now: datetime | None = None) -> int:
    """A-share trading minutes elapsed by `now` (0–240).

    Sessions: 9:30–11:30 (120m) + 13:00–15:00 (120m) = 240m.
    Lunch break (11:30–13:00) returns 120 (morning complete).
    Pre-open returns 0; post-close returns 240.
    """
    if now is None:
        now = datetime.now()
    t = now.hour * 60 + now.minute
    morning = max(0, min(t - (9 * 60 + 30), 120))
    afternoon = max(0, min(t - 13 * 60, 120))
    return morning + afternoon


def _full_day_session_scale(now: datetime | None = None) -> float:
    """Multiplier to convert intraday partial volume/amount to *estimated full-day*.

    Why: model technical features (amount_ratio, volume_trend, amihud_illiq,
    mfi_14, vwap_dev, …) compare today's volume/amount against a 5/20/60-day
    rolling mean of *full-day* values.  Injecting a half-day value at midday
    systematically biases these features low for low-turnover stocks (utilities,
    banks, defensives), causing midday rank to be artificially depressed.
    Scaling by ``1/session_fraction`` extrapolates the partial bar to a
    full-day estimate, restoring distributional consistency with training data.

    Returns 1.0 outside trading hours (post-close = no scaling needed).
    """
    elapsed = _trading_minutes_elapsed(now)
    if elapsed <= 0:
        return 1.0  # before market open: volume is 0 anyway, no scaling
    fraction = elapsed / 240.0
    # Clamp lower bound so a 9:35 fetch (only 5 minutes elapsed) doesn't
    # produce a 48× extrapolation that's noisier than just using the raw value.
    fraction = max(fraction, 0.20)
    return 1.0 / fraction


def _fetch_realtime_prices(codes: List[str]) -> Dict[str, dict]:
    """Fetch realtime prices via Sina quote API. Returns {code: {price, prev_close, change_pct, name}}.

    Volume/amount are scaled to *estimated full-day* via :func:`_full_day_session_scale`
    so model features that compare today's flow against historical full-day
    averages (amount_ratio, volume_trend, amihud_illiq, mfi_14, …) operate on
    the same distribution they were trained on.  Midday: ×2.0; 14:00: ×1.33;
    post-close: ×1.0.
    """
    import httpx

    scale = _full_day_session_scale()
    logger.info("Realtime volume/amount scale = {:.3f} (session fraction {:.0%})",
                scale, 1.0 / scale)

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
                #   4=high, 5=low, 6=bid, 7=ask, 8=volume(股), 9=amount(元)
                #
                # ⚠️ Unit verified 2026-05-11: hq.sinajs.cn returns volume
                # in SHARES, not 手 — checked 002385 EOD where DB volume =
                # 70,455,338 (shares) exactly matches sina fields[8].
                # Previous *100 conversion inflated volume by 100x, crushing
                # amihud_illiq to ~0 and making model wildly overweight
                # small-caps midday.  Source convention matches schema.py's
                # SOURCE_CONVENTIONS["sina"] = {"volume": "shares"}.
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
                            # Scale to estimated full-day so model features
                            # operate on the same distribution as training.
                            # No 手→股 conversion — sina already returns shares.
                            "volume": float(fields[8]) * scale,
                            "amount": float(fields[9]) * scale,
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
    gen_t = ts.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
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
    _hs300 = _hs300_set()   # for ⚠️ extrapolation marking

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
    gen_t = ts.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
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


def run_midday(dry_run: bool = False, chat_id: Optional[str] = None, user_id: Optional[str] = None,
               session_label: str = "midday"):
    """Run intraday report — same pipeline as daily, with intraday bars injected.

    session_label drives the title, file name, and watchlist behaviour:
      - "midday"  (~12:00, scale ≈ 2.0): saves recommendations as watchlist,
        no disk persistence (feishu only).
      - "2pm"     (~14:00, scale ≈ 1.33): saves report as
        ``daily_YYYYMMDD_14h.md``, does NOT overwrite the noon watchlist
        (the afternoon EOD report tracks the noon picks specifically).

    Volume scaling is handled automatically by :func:`_full_day_session_scale`
    based on wall-clock time — at 14:00 it returns ~1.333 (since session
    fraction = 180/240 = 75%).  No code change needed for that.
    """
    from mp.data.fetcher import get_index_constituents

    if date.today().weekday() >= 5:
        logger.info("Weekend — skipping {} report", session_label)
        return

    is_midday_proper = session_label == "midday"
    label_zh = {"midday": "午间快报", "2pm": "盘中快报 14:00"}.get(session_label, session_label)
    logger.info("=== {} Report: {} ===", label_zh, date.today())

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
        from mp.data.fetcher import get_recommendation_universe
        zz500_codes = get_recommendation_universe()   # HS300 + ZZ500
    except Exception as e:
        logger.warning("Universe fetch failed: {}", e)
        zz500_codes = []

    all_codes = list(set(h_codes + zz500_codes))
    logger.info("Fetching realtime OHLCV for {} stocks...", len(all_codes))
    prices = _fetch_realtime_prices(all_codes)
    if not prices:
        logger.error("Failed to fetch realtime prices")
        return
    bar_fetched_at = datetime.now()   # ← 行情数据拉取完成时刻
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

    # 5a. Build shared feature panel ONCE for universe ∪ holdings.
    # Same fix as run() — see comment there.  Without this, holdings_eval
    # and recommend_stocks can see two different DB snapshots when the
    # API recovers mid-run and DB advances between calls.
    panel_codes = sorted(set(zz500_codes) | set(h_codes))
    logger.info("--- Building shared feature panel ({} stocks) ---", len(panel_codes))
    shared_features = build_latest_features(
        panel_codes, include_fundamentals=True, intraday_bars=intraday_bars)
    if shared_features.empty:
        logger.error("Shared feature panel is empty — aborting midday")
        return

    # 5. Evaluate holdings using the shared panel (with intraday bars baked in)
    logger.info("--- Evaluating holdings (with intraday data) ---")
    holdings_eval = evaluate_holdings(ranker, regime=regime,
                                       intraday_bars=intraday_bars,
                                       precomputed_features=shared_features)
    valuation_fetched_at = datetime.now()   # ← 估值数据（PE/PB）拉取完成时刻

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

    # 6b. 60d ML predictions (reuse shared panel)
    holdings_60d_scores = {}
    if has_60d_model and h_codes:
        h_features = shared_features[shared_features["code"].isin(h_codes)].copy()
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

    # 8. Recommend stocks using the same shared panel
    logger.info("--- Scanning universe for recommendations ---")
    recommendations, rec_60d, rec_name_map, full_scored = recommend_stocks(
        ranker, n_recommend=5, intraday_bars=intraday_bars,
        precomputed_features=shared_features)

    # 8a. Save midday picks as watchlist for afternoon report to track.
    # Only the 12:00 run owns the watchlist — the 14:00 intraday should
    # NOT overwrite it (the 16:00 EOD report specifically tracks how the
    # noon picks held up vs faded as full-day data came in).
    midday_watchlist_eval: list[dict] | None = None
    if is_midday_proper:
        _save_midday_watchlist(recommendations, date.today())
    else:
        # 2pm (or later intraday) — read midday watchlist and evaluate it
        # against the just-computed full_scored ranking so users see how
        # noon's picks have evolved through the early afternoon.
        midday_watchlist = _load_midday_watchlist(date.today())
        midday_watchlist_eval = _evaluate_watchlist_afternoon(full_scored, midday_watchlist)
        if midday_watchlist_eval:
            logger.info("Midday watchlist tracked at {}: {} stocks",
                        session_label, len(midday_watchlist_eval))

    # 8b. 60d ML predictions for recommendations (reuse shared panel)
    rec_degraded = "_degraded_reason" in recommendations.columns
    rec_60d_scores = {}
    if has_60d_model and not recommendations.empty and not rec_degraded:
        rec_codes_list = recommendations["code"].tolist()
        rec_features = shared_features[shared_features["code"].isin(rec_codes_list)].copy()
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

    # 10. Format report (intraday — midday=True flag, custom title via session_label)
    data_timestamps = _make_data_timestamps(bar_fetched_at=bar_fetched_at, intraday=True)
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
        session_label=session_label,
        midday_watchlist_eval=midday_watchlist_eval,
    )
    logger.info("{} report generated ({} chars)", label_zh, len(report))

    # Save 2pm reports to disk so they don't get lost (midday is feishu-only).
    if not is_midday_proper:
        report_dir = PROJECT_ROOT / "data" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        suffix = {"2pm": "_14h"}.get(session_label, f"_{session_label}")
        report_path = report_dir / f"daily_{date.today().strftime('%Y%m%d')}{suffix}.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info("Report saved to {}", report_path)

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
        session_label=session_label,
        midday=True,
        data_timestamps=data_timestamps,
        midday_watchlist_eval=midday_watchlist_eval,
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

    # 0a. Build feature panel ONCE for universe ∪ holdings.
    # ALL downstream scoring (evaluate_holdings, recommend_stocks) shares
    # this single snapshot.  Without this, a flaky API mid-run can have
    # evaluate_holdings see DB-as-of-X while recommend_stocks sees
    # DB-as-of-X+1 → same stock gets two different predictions in the
    # same report (粤电力A 2026-05-15: 持仓 +1.98% vs 推荐 +4.75%).
    from mp.data.fetcher import get_recommendation_universe
    holdings = load_holdings()
    holding_codes = [h["code"] for h in holdings]
    try:
        universe = get_recommendation_universe()
    except Exception as e:
        logger.warning("Universe fetch failed, using holdings only: {}", e)
        universe = []
    panel_codes = sorted(set(universe) | set(holding_codes))
    logger.info("--- Building shared feature panel ({} stocks) ---", len(panel_codes))
    shared_features = build_latest_features(panel_codes, include_fundamentals=True)
    if shared_features.empty:
        logger.error("Shared feature panel is empty — aborting")
        return

    # 1. Evaluate holdings using the shared panel
    logger.info("--- Evaluating holdings ---")
    holdings_eval = evaluate_holdings(ranker, regime=regime,
                                       precomputed_features=shared_features)

    # 1b. 60-day technical analysis for holdings
    codes = holding_codes
    name_map = {h["code"]: h["name"] for h in holdings}
    logger.info("--- 60-day technical analysis ---")
    holdings_60d = evaluate_holdings_60d(codes) if codes else []

    # 1c. 60d ML predictions for holdings (reuse shared panel)
    holdings_60d_scores = {}
    if has_60d_model and codes:
        h_features = shared_features[shared_features["code"].isin(codes)].copy()
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

    # 2. Recommend stocks using the same shared panel
    logger.info("--- Scanning universe for recommendations ---")
    recommendations, rec_60d, rec_name_map, full_scored = recommend_stocks(
        ranker, n_recommend=5, precomputed_features=shared_features)

    # 2a. Evaluate today's midday watchlist (if midday ran) — show how those picks held up
    midday_watchlist = _load_midday_watchlist(date.today())
    midday_watchlist_eval = _evaluate_watchlist_afternoon(full_scored, midday_watchlist)
    if midday_watchlist_eval:
        logger.info("Midday watchlist tracked: {} stocks", len(midday_watchlist_eval))

    # 2b. 60d ML predictions for recommendations (reuse shared panel)
    rec_degraded = "_degraded_reason" in recommendations.columns
    rec_60d_scores = {}
    if has_60d_model and not recommendations.empty and not rec_degraded:
        rec_codes_list = recommendations["code"].tolist()
        rec_features = shared_features[shared_features["code"].isin(rec_codes_list)].copy()
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

    # 2d. Generate next-day open order list (conviction-weighted, top-5)
    account = load_account()
    order_list: list[dict] = []
    if account is None:
        logger.info("portfolio.yaml has no `account:` block — order list skipped")
    elif rec_degraded:
        logger.info("recommendations degraded — order list skipped")
    else:
        try:
            order_list = generate_order_list(
                holdings_full=holdings,
                account=account,
                recommendations=recommendations,
                full_scored=full_scored,
            )
            logger.info("Order list: {} orders", len(order_list))
        except Exception as e:
            logger.warning("Order list generation failed (non-fatal): {}", e)
            order_list = []

    # 3. Format report
    data_timestamps = _make_data_timestamps()  # daily: both dates from DB
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
        midday_watchlist_eval=midday_watchlist_eval or None,
        order_list=order_list or None,
        account=account,
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
        midday_watchlist_eval=midday_watchlist_eval or None,
        order_list=order_list or None,
        account=account,
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
    parser.add_argument("--intraday-2pm", action="store_true",
                        help="Run 14:00 intraday report (same flow as midday, scale ≈ 1.33×)")
    parser.add_argument("--chat-id", type=str, help="Feishu group chat ID (oc_xxx)")
    parser.add_argument("--user-id", type=str, help="Feishu user ID for DM (ou_xxx)")
    args = parser.parse_args()

    if args.intraday_2pm:
        run_midday(dry_run=args.dry_run, chat_id=args.chat_id, user_id=args.user_id,
                   session_label="2pm")
    elif args.midday:
        run_midday(dry_run=args.dry_run, chat_id=args.chat_id, user_id=args.user_id,
                   session_label="midday")
    else:
        run(dry_run=args.dry_run, chat_id=args.chat_id, user_id=args.user_id)
