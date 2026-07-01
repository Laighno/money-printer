"""Dead-man-switch for the DAILY EOD plan (advisor 2026-07-01; user 拍 "要").

Sibling of retrain_freshness_check.py, for the daily plan instead of the weekly
model. The 17:00 ECS daily_report.py can die at the scoring step (ProcessPool
OOM on 8GB — 6/22/6/25/6/30 all failed silently), leaving data/orders/latest.json
frozen at an older date. The 9:25 reconcile then diffs against a STALE target and
fires only a tiny residual — the user notices "今天怎么才 2 单" instead of an alert.

This screams (Feishu RED) when latest.json isn't today's on a trading day. Runs
as an INDEPENDENT ECS task (~17:50, after daily_report should have finished) so
it fires even when the daily_report wrapper itself was OOM-killed mid-run.

Minimal imports (stdlib + trading_calendar + scripts.daily_report.send_to_feishu)
so a broken daily_report can't take the alarm down with it.

Exit: 0 = checked (quiet or alert sent), 1 = could not read plan.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
LATEST = _REPO / "data" / "orders" / "latest.json"

try:
    from mp.data.trading_calendar import trading_days_between  # type: ignore
except Exception:  # pragma: no cover
    trading_days_between = None


def _send(markdown: str) -> None:
    try:
        sys.path.insert(0, str(_REPO))
        from scripts.daily_report import send_to_feishu
        send_to_feishu(markdown)
    except Exception as e:
        sys.stderr.write(f"[plan_freshness] Feishu send failed: {e}\n")


def _is_trading_day(d) -> bool:
    """Is date d a trading day? Weekend → False; else use the calendar if
    available (catches CNY / National Day holidays), else assume weekday=trading."""
    if d.weekday() >= 5:
        return False
    if trading_days_between is None:
        return True
    try:
        # trading_days_between(d, d+1) counts trading days in [d, d+1); 1 if d trades.
        from datetime import timedelta
        return trading_days_between(d, d + timedelta(days=1)) >= 1
    except Exception:
        return True


def main() -> int:
    today = datetime.now().date()

    if not _is_trading_day(today):
        sys.stdout.write(f"[plan_freshness] {today} not a trading day — quiet\n")
        return 0

    if not LATEST.exists():
        _send("🔴 **RED — 今日 EOD 计划缺失**\n\n"
              "`data/orders/latest.json` 不存在。daily_report.py 未产出计划(可能 OOM)。\n"
              "9:25 将无计划可 reconcile。排查:`data/logs/ecs_daily_report.log` Step 4。")
        return 0

    try:
        src = (json.loads(LATEST.read_text(encoding="utf-8")).get("source") or {})
        gen = src.get("generated_at") or ""
        gen_date = datetime.fromisoformat(gen.replace("Z", "")).date()
    except Exception as e:
        sys.stderr.write(f"[plan_freshness] unreadable latest.json: {e}\n")
        _send(f"🔴 **RED — EOD 计划文件损坏**\n\n`latest.json` 无法解析: {e}")
        return 1

    if gen_date < today:
        stale_days = (today - gen_date).days
        _send(f"🔴 **RED — 今日 EOD 计划未刷新(旧 {stale_days} 天)**\n\n"
              f"`latest.json` generated_at={gen_date},今天={today}。"
              f"daily_report.py 今天没成功产出新计划(大概率 scoring OOM)。\n"
              f"9:25 会拿旧计划 reconcile → 残差极小(少量委托)。\n"
              f"排查:`data/logs/ecs_daily_report.log` Step 4;必要时跑 mac_fallback_plan.sh。")
        return 0

    sys.stdout.write(f"[plan_freshness] OK: latest.json generated {gen_date} == today\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
