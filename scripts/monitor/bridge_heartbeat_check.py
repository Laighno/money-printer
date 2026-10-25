"""Pre-market bridge liveness alarm (2026-09-07 incident).

2026-10-25: retargeted at the NAME bridge. 国金 banned file reads in the QMT
built-in Python on 2026-10-23, which killed the JSON file bridge outright, so
the old ``data/bridge/heartbeat.json`` is frozen at its last pre-ban write and
would now alert every single morning. The replacement strategy cannot write
readable content either, so it publishes liveness as FILENAMES:
``mp_orders/hb/<epoch>.hb``. This check reads the newest such name.

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

import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
# The strategy ticks every 2s and writes one <epoch>.hb per tick.
HB_DIR = Path(r"C:\guojin\python\mp_orders\hb")
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


def _heartbeat_age() -> float | None:
    """Seconds since the newest ``<epoch>.hb``, or None when none readable.

    The timestamp is in the NAME because the sandboxed strategy can create
    files but cannot write content this side can read back.
    """
    try:
        stamps = []
        for p in HB_DIR.iterdir():
            if p.suffix == ".hb":
                try:
                    stamps.append(int(p.stem))
                except ValueError:
                    continue
        if not stamps:
            return None
        return time.time() - max(stamps)
    except OSError:
        return None


def main() -> int:
    if _is_weekend():
        sys.stdout.write("[bridge_hb] weekend — quiet\n")
        return 0
    age = _heartbeat_age()
    if age is None:
        _send("🔴 **RED — 交易桥心跳缺失**\n\n"
              f"未找到任何心跳文件({HB_DIR})。9:25 执行将 abort(不交易)。\n"
              "请 RDP 到 ECS:确认大 QMT 客户端已登录,「模型交易」里加载并启动 "
              "NAMEBRIDGE 策略(运行模式=实盘,勿勾本地python)。\n\n"
              "注意:2026-09-25 起登录框带图片验证码,看守脚本不再自动登录,"
              "客户端重启后必须人工登录;策略也不会自动恢复,需手动启动。")
        return 0
    if age > STALE_SEC:
        hrs = age / 3600
        _send(f"🔴 **RED — 交易桥已停 {hrs:.1f} 小时**\n\n"
              f"BigQMT 的 NAMEBRIDGE 策略没在运行(心跳 {age:.0f}s 未刷)。"
              f"**9:25 执行将 abort,今天将不交易**,除非在此之前恢复:\n"
              f"RDP → 大QMT客户端(若未登录先登录)→ 模型交易 → NAMEBRIDGE → ▶ 启动。\n"
              f"(常见原因:客户端夜间自动重启后策略未能继续运行)\n\n"
              f"**注意:2026-09-25 起登录框带图片验证码,看守脚本已不再自动登录**,"
              f"客户端重启后必须人工登录。9/25-9/28 因此静默丢了 3 个交易日。")
        return 0
    sys.stdout.write(f"[bridge_hb] OK age={age:.1f}s\n")
    _prune()
    return 0


def _prune() -> None:
    """Housekeeping: drop done_req_*/resp_* older than 24h (audit M3). Lazy
    import + broad except so housekeeping can never break the alert path."""
    try:
        from mp.execution.namebridge_broker import NameBridgeBroker
        n = NameBridgeBroker(ord_dir=str(HB_DIR.parent)).prune(max_age_hours=24.0)
        sys.stdout.write(f"[bridge_hb] pruned {n} stale bridge files\n")
    except Exception as e:  # noqa: BLE001
        sys.stdout.write(f"[bridge_hb] prune skipped: {e}\n")


if __name__ == "__main__":
    sys.exit(main())
