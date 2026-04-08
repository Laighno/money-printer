"""Indicator and signal computation for individual stocks."""

from __future__ import annotations

from typing import Literal, Optional, TypedDict


class Signal(TypedDict):
    dimension: str                                    # e.g. "技术指标", "资金面", "市场情绪", "估值"
    name: str                                         # e.g. "RSI(14)", "主力资金净流入"
    value: Optional[float]                            # raw numeric value
    signal: Literal["bullish", "bearish", "neutral"]  # direction
    detail: str                                       # human-readable explanation
