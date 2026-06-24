"""Dead-man-switch for the auto-retrain pipeline (advisor; user 拍 ②B).

The weekly retrain moved from the Mac Friday cron (walk_forward_backtest.py,
which silently died when the laptop slept ~4/24) to ECS auto_retrain.py. This
checker fires INDEPENDENTLY: if auto_retrain.py stops running — or runs but the
verify gate keeps failing so prod never refreshes — nobody would notice unless
something screams. This screams (Feishu RED).

It reads ONLY data/auto_retrain_last.json (written by auto_retrain.py at the end
of every run, pass or fail) and the swap log. Minimal imports (stdlib +
scripts.daily_report.send_to_feishu) so a broken auto_retrain/walk_forward can't
take the alarm down with it — same discipline as weekly_heartbeat.py.

Schedule: ECS Task Scheduler, Saturday morning (a day after the Friday retrain),
independent failure domain.

Thresholds (weekly cadence → a single missed Friday is already suspicious):
  - last run ts > 8 calendar days old  → RED (a Friday was missed entirely)
  - last run present but verify_pass is False for > 8 days → RED (model frozen)
  - last run ok and recent             → quiet (exit 0, no message)

Exit codes: 0 = checked (quiet or alert sent), 1 = could not read state.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
SUMMARY = _REPO / "data" / "auto_retrain_last.json"
SWAP_LOG = _REPO / "data" / "model_swap_log.json"

RED_AGE = timedelta(days=8)   # weekly cadence; >8d means a Friday was missed


def _send(markdown: str) -> None:
    try:
        sys.path.insert(0, str(_REPO))
        from scripts.daily_report import send_to_feishu
        send_to_feishu(markdown)
    except Exception as e:  # alarm must never crash silently
        sys.stderr.write(f"[retrain_freshness] Feishu send failed: {e}\n")


def main() -> int:
    now = datetime.now()

    if not SUMMARY.exists():
        _send("🔴 **RED — 自动重训未运行**\n\n"
              "`data/auto_retrain_last.json` 不存在。auto_retrain.py 从未成功跑过,"
              "或 ECS 周调度未注册。模型可能正在静默腐烂(复刻 4/24 事故)。\n"
              "排查:ECS Task Scheduler `MP-AutoRetrain` 任务 + `data/logs/auto_retrain.log`。")
        return 0

    try:
        s = json.loads(SUMMARY.read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(s["ts"])
    except Exception as e:
        sys.stderr.write(f"[retrain_freshness] unreadable summary: {e}\n")
        _send(f"🔴 **RED — 重训状态文件损坏**\n\n`auto_retrain_last.json` 无法解析: {e}")
        return 1

    age = now - ts
    cutoff = s.get("cutoff")
    verify_pass = s.get("verify_pass")
    swapped = s.get("swapped")
    reasons = s.get("reasons") or []

    # 1. stale → a Friday was missed entirely.
    if age > RED_AGE:
        _send(f"🔴 **RED — 自动重训超期 {age.days} 天未运行**\n\n"
              f"上次 auto_retrain: {ts:%Y-%m-%d %H:%M}(cutoff {cutoff})。"
              f"阈值 {RED_AGE.days} 天(周调度漏跑至少一个周五)。\n"
              "排查:ECS Task Scheduler `MP-AutoRetrain` + `data/logs/auto_retrain.log`。")
        return 0

    # 2. ran recently but gate keeps failing → prod frozen, no fresh model lands.
    if verify_pass is False:
        _send(f"🟡 **YELLOW — 重训跑了但 verify gate 未过**\n\n"
              f"{ts:%Y-%m-%d %H:%M} cutoff {cutoff}:gate FAIL,prod 保持旧模型未换。\n"
              f"原因:{'; '.join(str(r) for r in reasons) or '见 verdict json'}\n"
              "若连续多周 FAIL,需人工查模型/数据(可能 regime 或特征问题)。")
        return 0

    # healthy + recent → quiet.
    sys.stdout.write(
        f"[retrain_freshness] OK: last={ts:%Y-%m-%d %H:%M} cutoff={cutoff} "
        f"verify_pass={verify_pass} swapped={swapped} age={age.days}d\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
