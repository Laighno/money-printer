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

from mp.ml.model import StockRanker, FEATURE_COLS
from mp.ml.dataset import build_latest_features
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
    """Fetch stock names for a small list of codes."""
    import akshare as ak
    name_map = {}
    for code in codes:
        try:
            df = ak.stock_individual_info_em(symbol=code)
            name_row = df[df["item"] == "股票简称"]
            if not name_row.empty:
                name_map[code] = name_row["value"].values[0]
                continue
        except Exception:
            pass
        name_map[code] = code
    return name_map


# =====================================================================
# Holdings Evaluation
# =====================================================================

def evaluate_holdings(ranker: StockRanker) -> pd.DataFrame:
    """Evaluate current holdings with ML model.

    Returns DataFrame with: code, name, ml_score, predicted_return, suggestion
    """
    holdings = load_holdings()
    if not holdings:
        logger.warning("No A-share holdings found")
        return pd.DataFrame()

    codes = [h["code"] for h in holdings]
    name_map = {h["code"]: h["name"] for h in holdings}

    logger.info("Building features for {} holdings...", len(codes))
    features = build_latest_features(codes, include_fundamentals=True)
    if features.empty:
        logger.error("Failed to build features for holdings")
        return pd.DataFrame()

    scores = ranker.predict(features)
    result = pd.DataFrame({
        "code": features["code"].values,
        "ml_score": scores,
    })
    result["name"] = result["code"].map(name_map)
    result = result.sort_values("ml_score", ascending=False).reset_index(drop=True)

    # Position suggestion based on predicted return
    # ml_score is predicted 20d forward return (e.g., 0.05 = +5%)
    def suggest(score):
        if score > 0.03:
            return "加仓"
        elif score > 0.00:
            return "持有"
        elif score > -0.03:
            return "减仓"
        else:
            return "清仓"

    result["predicted_return"] = (result["ml_score"] * 100).round(2)
    result["suggestion"] = result["ml_score"].apply(suggest)

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
# Stock Recommendations
# =====================================================================

def recommend_stocks(ranker: StockRanker, n_recommend: int = 5) -> pd.DataFrame:
    """Scan ZZ500 for top ML-scored stocks.

    Returns top N stocks by ML predicted return.
    """
    from mp.data.fetcher import get_index_constituents

    logger.info("Fetching ZZ500 constituents...")
    try:
        codes = get_index_constituents("zz500")
    except Exception as e:
        logger.error("Failed to get ZZ500 constituents: {}", e)
        return pd.DataFrame()

    logger.info("Building features for {} ZZ500 stocks...", len(codes))
    features = build_latest_features(codes, include_fundamentals=True)
    if features.empty:
        logger.error("Failed to build features for ZZ500")
        return pd.DataFrame()

    scores = ranker.predict(features)
    result = pd.DataFrame({
        "code": features["code"].values,
        "ml_score": scores,
    })

    result["predicted_return"] = (result["ml_score"] * 100).round(2)

    # Sort and take top N, then fetch names only for those
    result = result.sort_values("ml_score", ascending=False).head(n_recommend).reset_index(drop=True)
    name_map = get_stock_names(result["code"].tolist())
    result["name"] = result["code"].map(name_map)

    return result


# =====================================================================
# Report Formatting
# =====================================================================

def format_report(holdings_eval: pd.DataFrame, recommendations: pd.DataFrame) -> str:
    """Format evaluation results as markdown for Feishu."""
    today = date.today().strftime("%Y-%m-%d")
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    wd = weekday_cn[date.today().weekday()]

    lines = [
        f"# 每日持仓评估报告",
        f"**{today} {wd}**",
        "",
    ]

    # Holdings evaluation
    lines.append("## 持仓评估")
    if holdings_eval.empty:
        lines.append("*无 A 股持仓*")
    else:
        lines.append("")
        for _, row in holdings_eval.iterrows():
            icon = {"加仓": "🟢", "持有": "🟡", "减仓": "🟠", "清仓": "🔴"}.get(row["suggestion"], "⚪")
            lines.append(
                f"**{row['name']}** ({row['code']})  "
                f"预测20日收益: **{row['predicted_return']:+.2f}%**  "
                f"建议: {icon} **{row['suggestion']}**"
            )
            if row.get("top_factors"):
                lines.append(f"  关键因子: {row['top_factors']}")
            lines.append("")

    # Recommendations
    lines.append("## 推荐关注 (ZZ500)")
    if recommendations.empty:
        lines.append("*未能生成推荐*")
    else:
        lines.append("")
        for i, row in recommendations.iterrows():
            lines.append(
                f"{i+1}. **{row['name']}** ({row['code']})  "
                f"预测20日收益: **{row['predicted_return']:+.2f}%**"
            )
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*由 Money Printer ML 模型自动生成 | {datetime.now().strftime('%H:%M')}*")

    return "\n".join(lines)


# =====================================================================
# Feishu Send
# =====================================================================

def send_to_feishu(markdown: str, chat_id: Optional[str] = None, user_id: Optional[str] = None) -> bool:
    """Send markdown report via lark-cli."""
    cmd = ["lark-cli", "im", "+messages-send", "--as", "bot"]

    if chat_id:
        cmd.extend(["--chat-id", chat_id])
    elif user_id:
        cmd.extend(["--user-id", user_id])
    else:
        cmd.extend(["--user-id", DEFAULT_USER_ID])

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
# Main
# =====================================================================

def run(dry_run: bool = False, chat_id: Optional[str] = None, user_id: Optional[str] = None):
    """Run full daily report pipeline."""
    logger.info("=== Daily Report: {} ===", date.today())

    # Check if it's a trading day (skip weekends)
    if date.today().weekday() >= 5:
        logger.info("Weekend - skipping report")
        return

    # Load model
    ranker = StockRanker()
    if not ranker.load():
        logger.error("No ML model found at {}. Run training first.", MODEL_PATH)
        return

    # 1. Evaluate holdings
    logger.info("--- Evaluating holdings ---")
    holdings_eval = evaluate_holdings(ranker)

    # 2. Recommend stocks from ZZ500
    logger.info("--- Scanning ZZ500 for recommendations ---")
    recommendations = recommend_stocks(ranker, n_recommend=5)

    # 3. Format report
    report = format_report(holdings_eval, recommendations)
    logger.info("Report generated ({} chars)", len(report))

    # Save locally
    report_dir = PROJECT_ROOT / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"daily_{date.today().strftime('%Y%m%d')}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to {}", report_path)

    # 4. Send to Feishu
    if dry_run:
        logger.info("Dry run - not sending to Feishu")
        print(report)
    else:
        send_to_feishu(report, chat_id=chat_id, user_id=user_id)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Daily portfolio report")
    parser.add_argument("--dry-run", action="store_true", help="Generate report without sending")
    parser.add_argument("--chat-id", type=str, help="Feishu group chat ID (oc_xxx)")
    parser.add_argument("--user-id", type=str, help="Feishu user ID for DM (ou_xxx)")
    args = parser.parse_args()

    run(dry_run=args.dry_run, chat_id=args.chat_id, user_id=args.user_id)
