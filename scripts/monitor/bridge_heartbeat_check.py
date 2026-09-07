"""Pre-market bridge liveness alarm (2026-09-07 incident).

9/5 Sat night the BigQMT client auto-restarted; the MONEY bridge strategy did
not keep running. Nothing alerted, so Monday 9:25 execution aborted on the
(correctly) stale-heartbeat gate and the trading day was silently lost.

This check runs every trading day at 09:10 (15 min before execution): if the
bridge heartbeat is stale, send Feishu RED so the user can RDP + start the
MONEY strategy before 09:25. Quiet when healthy. Minimal imports (same
discipline as retrain/plan freshness checks).

Exit: 0 = checked (quiet or alert sent), 1 = unexpected error.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
HB = _REPO / "data" / "bridge" / "heartbeat.json"
STALE_SEC = 60.0


def _send(markdown: str) -> None:
    try:
        sys.path.insert(0, str(_REPO))
        from scripts.daily_report import send_to_feishu
        send_to_feishu(markdown)
    except Exception as e:
        sys.stderr.write(f"[bridge_hb] Feishu send failed: {e}\n")


def _is_weekend() -> bool:
    return datetime.now().weekday() >= 5


def main() -> int:
    if _is_weekend():
        sys.stdout.write("[bridge_hb] weekend — quiet\n")
        return 0
    age = None
    try:
        hb = json.loads(HB.read_text(encoding="utf-8"))
        age = time.time() - float(hb["ts"])
    except Exception:
        pass
    if age is None:
        _send("🔴 **RED — 交易桥心跳文件缺失/不可读**\n\n"
              "9:25 执行将 abort(不交易)。请立刻 RDP 到 ECS:确认大 QMT 客户端已登录,"
              "「模型交易」里启动 MONEY 策略(运行模式=实盘,勿勾本地python)。")
        return 0
    if age > STALE_SEC:
        hrs = age / 3600
        _send(f"🔴 **RED — 交易桥已停 {hrs:.1f} 小时**\n\n"
              f"BigQMT 的 MONEY 桥策略没在运行(心跳 {age:.0f}s 未刷)。"
              f"**9:25 执行将 abort,今天将不交易**,除非在此之前恢复:\n"
              f"RDP → 大QMT客户端(若未登录先登录)→ 模型交易 → MONEY → ▶ 启动。\n"
              f"(常见原因:客户端夜间自动重启后策略未能继续运行)")
        return 0
    sys.stdout.write(f"[bridge_hb] OK age={age:.1f}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
