"""Market regime detection using three macro signals.

Signals:
  1. ZZ500 index trend (MA20/MA60) — 50% weight
  2. Northbound capital 5-day net flow — 30% weight
  3. Margin balance 5-day change rate — 20% weight

Composite score ∈ [-1, +1] → bull / sideways / bear classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import akshare as ak
import pandas as pd
from loguru import logger


# A-share transaction cost constants
STAMP_DUTY = 0.0005      # 印花税 0.05% (sell only)
COMMISSION = 0.00025     # 佣金 0.025% (each way)
TRANSFER_FEE = 0.00001   # 过户费 0.001% (each way)
ROUND_TRIP_COST = STAMP_DUTY + COMMISSION * 2 + TRANSFER_FEE * 2  # ≈ 0.15%


@dataclass
class MarketRegime:
    regime: Literal["bull", "sideways", "bear"]
    score: float           # [-1, +1]
    signals: dict          # per-signal details
    label_cn: str          # "牛市" / "震荡" / "熊市"
    summary_cn: str        # human-readable summary


_REGIME_LABELS = {"bull": "牛市", "sideways": "震荡", "bear": "熊市"}


class RegimeDetector:
    """Three-signal market regime detector."""

    def detect(self) -> MarketRegime:
        """Fetch live data and classify current market regime."""
        trend_val, trend_detail = self._index_trend_signal()
        nb_val, nb_detail = self._northbound_signal()
        margin_val, margin_detail = self._margin_signal()

        composite = 0.5 * trend_val + 0.3 * nb_val + 0.2 * margin_val

        if composite > 0.3:
            regime = "bull"
        elif composite < -0.3:
            regime = "bear"
        else:
            regime = "sideways"

        signals = {
            "index_trend": {"value": trend_val, **trend_detail},
            "northbound": {"value": nb_val, **nb_detail},
            "margin": {"value": margin_val, **margin_detail},
        }

        # Build Chinese summary
        parts = []
        for name_cn, detail in [
            ("指数趋势", trend_detail),
            ("北向资金", nb_detail),
            ("两融余额", margin_detail),
        ]:
            direction = detail.get("direction", "中性")
            parts.append(f"{name_cn}{direction}")
        summary_cn = "，".join(parts)

        return MarketRegime(
            regime=regime,
            score=round(composite, 3),
            signals=signals,
            label_cn=_REGIME_LABELS[regime],
            summary_cn=summary_cn,
        )

    # ------------------------------------------------------------------
    # Sub-signals
    # ------------------------------------------------------------------

    def _index_trend_signal(self) -> tuple[float, dict]:
        """ZZ500 MA trend: close vs MA20, MA20 vs MA60."""
        try:
            idx = ak.stock_zh_index_daily(symbol="sh000905")
            idx["date"] = pd.to_datetime(idx["date"])
            idx = idx.sort_values("date").tail(80)
            close = idx["close"].astype(float)

            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1]
            last = close.iloc[-1]

            if last > ma20 and ma20 > ma60:
                signal, direction = 1.0, "偏多"
                detail_str = f"收盘{last:.0f} > MA20({ma20:.0f}) > MA60({ma60:.0f})"
            elif last < ma20 and ma20 < ma60:
                signal, direction = -1.0, "偏空"
                detail_str = f"收盘{last:.0f} < MA20({ma20:.0f}) < MA60({ma60:.0f})"
            else:
                signal, direction = 0.0, "中性"
                detail_str = f"收盘{last:.0f}, MA20={ma20:.0f}, MA60={ma60:.0f}"

            return signal, {"direction": direction, "detail": detail_str}

        except Exception as e:
            logger.warning("Index trend signal failed: {}", e)
            return 0.0, {"direction": "未知", "detail": f"获取失败: {e}", "error": str(e)}

    def _northbound_signal(self) -> tuple[float, dict]:
        """Northbound capital 5-day cumulative net buy (亿元)."""
        try:
            total = 0.0
            net_col = None
            for sym in ["沪股通", "深股通"]:
                df = ak.stock_hsgt_hist_em(symbol=sym)
                if net_col is None:
                    net_col = next((c for c in df.columns if "净买" in c), df.columns[1])
                vals = pd.to_numeric(df[net_col].tail(5), errors="coerce")
                total += vals.sum()

            if total > 50:
                signal, direction = 1.0, "偏多"
            elif total < -50:
                signal, direction = -1.0, "偏空"
            else:
                signal, direction = 0.0, "中性"

            detail_str = f"5日净{'买入' if total >= 0 else '流出'}{abs(total):.0f}亿"
            return signal, {"direction": direction, "detail": detail_str, "net_5d": round(total, 1)}

        except Exception as e:
            logger.warning("Northbound signal failed: {}", e)
            return 0.0, {"direction": "未知", "detail": f"获取失败: {e}", "error": str(e)}

    def _margin_signal(self) -> tuple[float, dict]:
        """Market-level margin balance 5-day change rate."""
        try:
            df = ak.stock_margin_sse(start_date="20200101")
            bal_col = next((c for c in df.columns if "余额" in c and "融资" in c), None)
            if bal_col is None:
                # Fallback: try last numeric column
                bal_col = df.select_dtypes("number").columns[-1]

            recent = pd.to_numeric(df[bal_col].tail(10), errors="coerce").dropna()
            if len(recent) < 6:
                return 0.0, {"direction": "未知", "detail": "两融数据不足"}

            latest = recent.iloc[-1]
            prev = recent.iloc[-6]  # ~5 trading days ago
            change_rate = (latest - prev) / prev if prev > 0 else 0.0

            if change_rate > 0.01:
                signal, direction = 1.0, "偏多"
            elif change_rate < -0.01:
                signal, direction = -1.0, "偏空"
            else:
                signal, direction = 0.0, "中性"

            detail_str = f"5日余额变化{change_rate:+.2%}，当前{latest / 1e8:.0f}亿"
            return signal, {"direction": direction, "detail": detail_str, "change_rate": round(change_rate, 4)}

        except Exception as e:
            logger.warning("Margin signal failed: {}", e)
            return 0.0, {"direction": "未知", "detail": f"获取失败: {e}", "error": str(e)}
