"""Execute the latest daily-report order plan via a broker (QMT or dry-run).

Workflow
--------
1. Load the latest order plan JSON written by ``daily_report.py``
   (``data/orders/latest.json``).
2. Connect to a broker (QMT live, or DryRunBroker preview).
3. Pre-flight safety gates:
   - confirm broker state roughly matches portfolio.yaml (cash & holdings)
   - per-order: fetch current price; abort that order if it has drifted
     > price_drift_pct away from the planned limit (default 2%)
4. Execute sells first (T+0 frees cash for buys), then buys.
5. Pause briefly between orders to let the broker register fills.
6. Reconcile: fetch final account state, write summary JSON + push Feishu.

Modes
-----
- ``--mode dryrun``      no broker side effects, just prints what would happen
- ``--mode interactive`` real broker, but ask y/N before EACH order
- ``--mode auto``        real broker, send everything (assumes you reviewed
                         the plan in the daily report already)

CLI
---
    # Preview (no QMT needed — works on macOS)
    python scripts/execute_orders.py --mode dryrun

    # Real execution, confirm each order
    python scripts/execute_orders.py --mode interactive \\
        --qmt-account 12345678 \\
        --qmt-userdata 'C:\\国金证券QMT交易端\\userdata_mini'

Safety caps
-----------
- ``--max-orders``       hard limit on total orders this run (default 20)
- ``--max-single-pct``   reject buy that would push a single holding above
                         this fraction of total assets (default 0.50)
- ``--price-drift-pct``  max allowed gap between planned limit and current
                         market price (default 0.02 = 2%)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mp.execution.dryrun_broker import DryRunBroker
from mp.execution.qmt_broker import (
    AccountInfo, Position, QMTBroker,
)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _fetch_current_price(code: str) -> Optional[float]:
    """Get sina realtime price for a single code (cheap, no scaling).

    Returns None if the price is unavailable OR clearly stale/missing:
    sina returns field 3 = 0.00 before the market session has produced
    its first trade of the day (pre-market, between 09:00 and ~09:30).
    Treating 0.00 as a valid price falsely triggers SELL skip ("dump too
    low") in preflight_price_drift; callers expect None to mean unknown.
    """
    try:
        import httpx
        prefix = "sh" if code.startswith(("6", "68", "69")) else "sz"
        url = f"https://hq.sinajs.cn/list={prefix}{code}"
        r = httpx.get(url, headers={"Referer": "https://finance.sina.com.cn"},
                      timeout=5)
        line = r.text.strip()
        if "=" not in line:
            return None
        data = line.split("=", 1)[1].strip('";\r').split(",")
        if len(data) >= 4:
            price = float(data[3])   # field 3 = current price
            if price <= 0.0:
                return None          # no trade yet / market closed / bad data
            return price
    except Exception as e:
        logger.warning("realtime price fetch failed for {}: {}", code, e)
    return None


def _confirm(prompt: str) -> bool:
    """interactive y/N prompt; default N."""
    try:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in ("y", "yes")


def _format_summary(results: list[dict], mode: str = "dryrun") -> str:
    """round 174: prefix title with DRYRUN tag so consumers (Feishu / log
    readers) don't mistake a preview run for a real fill summary. The 6/1
    09:25 execute-preview launchd run fills DryRunBroker (autofill=True) →
    every row shows ``sent ✅`` even though no real order hit QMT.
    """
    sent = sum(1 for r in results if r["status"] == "sent")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    title = "# 实盘执行汇报" if mode != "dryrun" else "# 🧪 DRYRUN — 真单未发 (预览模式)"
    lines = [
        title,
        f"已发 {sent} / 跳过 {skipped} / 失败 {failed}",
    ]
    if mode == "dryrun":
        lines.append("")
        lines.append("> ⚠️ 这是预览模式 (execute-preview launchd 9:25 触发)。"
                     "下面表格里的 ✅sent 是 DryRunBroker 模拟撮合, **没有真单进 QMT**。"
                     "真单走 execute-live (当前 `.disabled`) 或人工 RDP / ssh 跑 `--mode auto`。")
    lines += [
        "",
        "| 股票 | 方向 | 股数 | 限价 | 状态 | 备注 |",
        "|---|---|---:|---:|---|---|",
    ]
    for r in results:
        status_emoji = {"sent": "✅", "skipped": "⚪", "failed": "❌"}[r["status"]]
        lines.append(
            f"| {r['name']} ({r['code']}) | {r['action']} | {r['shares']:,} | "
            f"¥{r['limit_price']:.2f} | {status_emoji}{r['status']} | "
            f"{r.get('note', '')} |"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# Pre-flight checks
# ──────────────────────────────────────────────────────────────────

def preflight_account_reconcile(
    plan_account: dict, live: AccountInfo, tolerance_pct: float = 0.02
) -> tuple[bool, str]:
    """Verify broker's reported total_assets is within tolerance_pct of plan.

    Big mismatches mean something happened overnight (deposits, withdrawals,
    corporate actions) that the plan didn't see — abort to be safe.
    """
    plan_total = plan_account.get("total_assets")
    if plan_total is None:
        return True, "no plan total_assets to compare"
    delta = abs(live.total_assets - plan_total) / max(plan_total, 1)
    if delta > tolerance_pct:
        return False, (
            f"account snapshot drifted {delta*100:.1f}% "
            f"(plan ¥{plan_total:,.0f} vs live ¥{live.total_assets:,.0f}). "
            f"Re-run daily_report to refresh the plan before executing."
        )
    return True, f"reconciled within {delta*100:.2f}%"


def preflight_positions_reconcile(
    plan_holdings: list[dict], live_positions: list[Position]
) -> tuple[bool, list[str]]:
    """Verify each plan-time holding still exists at >= the planned shares.

    For SELLS to be executable, broker must have at least the planned shares
    in 可用 (T+1 unlocked).
    """
    live_by_code = {p.code: p for p in live_positions}
    warnings: list[str] = []
    for h in plan_holdings:
        code = h["code"]
        planned = h.get("shares", 0)
        live = live_by_code.get(code)
        if live is None and planned > 0:
            warnings.append(f"{h['name']} ({code}): plan had {planned} 股, broker has 0")
            continue
        if live and live.shares_total != planned:
            warnings.append(
                f"{h['name']} ({code}): plan had {planned} 股, broker has "
                f"{live.shares_total} 股 (avail {live.shares_available})"
            )
    return (len(warnings) == 0), warnings


def preflight_price_drift(
    code: str, planned_limit: float, action: str, max_drift_pct: float
) -> tuple[bool, str, Optional[float]]:
    """Check whether current market price has drifted too far from planned limit.

    For BUYS: if current > limit × (1 + max_drift) → skip (would have to chase)
    For SELLS: if current < limit × (1 - max_drift) → skip (would dump too low)

    Returns (ok, msg, current_price).
    """
    cur = _fetch_current_price(code)
    if cur is None:
        return True, "price unavailable, proceeding", None
    if action == "buy":
        ceiling = planned_limit * (1 + max_drift_pct)
        if cur > ceiling:
            return False, (
                f"current ¥{cur:.2f} > limit ¥{planned_limit:.2f} "
                f"× (1 + {max_drift_pct*100:.0f}%) — too expensive to chase"
            ), cur
    else:   # sell
        floor = planned_limit * (1 - max_drift_pct)
        if cur < floor:
            return False, (
                f"current ¥{cur:.2f} < limit ¥{planned_limit:.2f} "
                f"× (1 - {max_drift_pct*100:.0f}%) — would dump too low"
            ), cur
    return True, f"current ¥{cur:.2f} within tolerance", cur


# ──────────────────────────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────────────────────────

def run(
    broker,
    plan: dict,
    *,
    mode: str = "dryrun",
    max_orders: int = 20,
    max_single_pct: float = 0.50,
    price_drift_pct: float = 0.02,
    fill_wait_seconds: float = 2.0,
    reconcile_tolerance: Optional[float] = None,
    arm_b_tracker=None,
) -> list[dict]:
    """Execute the plan.  Returns per-order execution records.

    When ``arm_b_tracker`` is non-None (typically auto-engaged for the
    14:30 OOS bucket — see ``main()``), each buy is gated by the tracker's
    daily 20,000 RMB cap (round 159 / round 161 guardrail (a)).
    """
    orders = plan.get("orders", [])
    if not orders:
        logger.info("Plan has no orders — nothing to execute")
        return []
    if len(orders) > max_orders:
        logger.error("Plan has {} orders > max_orders={}, aborting", len(orders), max_orders)
        return []

    # Pre-flight: connect + reconcile
    if not broker.is_connected():
        logger.info("Connecting broker...")
        broker.connect()

    live_acct = broker.get_account_info()
    logger.info("Broker reports: cash=¥{:.0f}, mv=¥{:.0f}, total=¥{:.0f}",
                live_acct.cash_available, live_acct.market_value, live_acct.total_assets)

    plan_acct = plan.get("account_snapshot", {}) or {}
    # Dryrun broker can't perfectly replicate live MV (no realtime prices),
    # so use a looser tolerance there.  Live modes stay strict (2%).
    tol = reconcile_tolerance
    if tol is None:
        tol = 0.10 if mode == "dryrun" else 0.02
    ok, msg = preflight_account_reconcile(plan_acct, live_acct, tolerance_pct=tol)
    logger.info("Account reconcile: {}", msg)
    if not ok:
        logger.error("Pre-flight failed — aborting")
        return [{"status": "failed", "note": msg, "code": "-", "name": "PREFLIGHT",
                  "action": "-", "shares": 0, "limit_price": 0}]

    plan_holdings = plan.get("holdings_at_plan_time", []) or []
    live_positions = broker.get_positions()
    pos_ok, warnings = preflight_positions_reconcile(plan_holdings, live_positions)
    for w in warnings:
        logger.warning("Positions drift: {}", w)
    if not pos_ok and mode == "auto":
        logger.error("Positions drifted but mode=auto; aborting to be safe")
        return [{"status": "failed", "note": "positions drift", "code": "-",
                  "name": "PREFLIGHT", "action": "-", "shares": 0, "limit_price": 0}]

    # Reorder: sells first, then buys (T+0 cash flow)
    sorted_orders = sorted(orders, key=lambda o: 0 if o["cost"] < 0 else 1)

    results: list[dict] = []
    for i, o in enumerate(sorted_orders, 1):
        code, action, shares, limit = (
            o["code"],
            "buy" if o["cost"] > 0 else "sell",
            int(o["shares"]),
            float(o["limit_price"]),
        )
        name = o.get("name", code)
        logger.info("--- [{}/{}] {} {} {} 股 @ ¥{:.2f} ---",
                    i, len(sorted_orders), action.upper(), code, shares, limit)

        # Price drift gate
        ok, msg, cur_price = preflight_price_drift(code, limit, action, price_drift_pct)
        if not ok:
            logger.warning("SKIP {}: {}", code, msg)
            results.append({"status": "skipped", "code": code, "name": name,
                             "action": action, "shares": shares,
                             "limit_price": limit, "note": msg})
            continue

        # Concentration check for buys: would this push code past single-cap?
        if action == "buy":
            cur_pos = next((p for p in broker.get_positions() if p.code == code), None)
            cur_val = cur_pos.market_value if cur_pos else 0
            cur_acct = broker.get_account_info()
            new_val = cur_val + shares * limit
            new_pct = new_val / max(cur_acct.total_assets, 1)
            if new_pct > max_single_pct:
                msg = (f"buy would push {code} to {new_pct*100:.1f}% > "
                       f"{max_single_pct*100:.0f}% cap; skip")
                logger.warning(msg)
                results.append({"status": "skipped", "code": code, "name": name,
                                 "action": action, "shares": shares,
                                 "limit_price": limit, "note": msg})
                continue

            # Arm B daily budget cap (round 161 guardrail (a)): if a tracker
            # is engaged (intraday 14:30 path), refuse buys that would push
            # cumulative Arm B buy notional past the daily cap (default
            # 20,000 RMB). Sells & non-armB pipelines bypass this gate.
            if arm_b_tracker is not None:
                ok, msg = arm_b_tracker.check_buy(code, shares, limit)
                if not ok:
                    logger.warning("Arm B cap SKIP {}: {}", code, msg)
                    results.append({"status": "skipped", "code": code, "name": name,
                                     "action": action, "shares": shares,
                                     "limit_price": limit,
                                     "note": f"arm_b_cap: {msg}"})
                    continue

        # Interactive confirm
        if mode == "interactive":
            if not _confirm(f"  → 确认 {action.upper()} {code} {shares} @ ¥{limit:.2f}?"):
                results.append({"status": "skipped", "code": code, "name": name,
                                 "action": action, "shares": shares,
                                 "limit_price": limit, "note": "user declined"})
                continue

        # Execute
        result = broker.place_limit_order(code, action, shares, limit)
        if result.success:
            logger.info("  ✅ order_id={}", result.order_id)
            # Commit Arm B budget after successful broker accept.
            # If commit raises (race / extreme edge), surface as failed
            # rather than silently corrupting cap accounting.
            if arm_b_tracker is not None and action == "buy":
                try:
                    arm_b_tracker.commit_buy(code, shares, limit,
                                              order_id=str(result.order_id))
                except ValueError as e:
                    logger.error("Arm B commit failed post-submit (state may "
                                 "be inconsistent — investigate): {}", e)
            results.append({"status": "sent", "code": code, "name": name,
                             "action": action, "shares": shares,
                             "limit_price": limit, "order_id": result.order_id,
                             "note": f"current ¥{cur_price:.2f}" if cur_price else ""})
        else:
            logger.error("  ❌ {}", result.error)
            results.append({"status": "failed", "code": code, "name": name,
                             "action": action, "shares": shares,
                             "limit_price": limit, "note": result.error or "?"})

        # Brief pause to let broker register state
        if mode != "dryrun" and fill_wait_seconds > 0:
            time.sleep(fill_wait_seconds)

    return results


def _print_emergency_result(result) -> None:
    """Pretty-print EmergencyResult to stdout / logger (CLI side)."""
    logger.info("EMERGENCY LIQUIDATE complete:")
    logger.info("  attempted:        {} codes", len(result.attempted_codes))
    logger.info("  succeeded submit: {} codes", len(result.succeeded_codes))
    logger.info("  failed submit:    {} codes", len(result.failed_codes))
    logger.info("  cancelled orders: {} ({})",
                len(result.cancelled_order_ids),
                ",".join(result.cancelled_order_ids) or "-")
    logger.info("  realized cash:    ¥{:,.2f}", result.total_realized_cash)
    logger.info("  duration:         {:.2f}s", result.duration_seconds)
    for code, err in result.failed_codes:
        logger.error("  FAIL {}: {}", code, err)


def _run_emergency(args) -> int:
    """Handle the --emergency CLI path.  Returns the process exit code."""
    if not args.confirm:
        logger.error("--emergency requires --confirm=EMERGENCY_LIQUIDATE_<account_id>")
        return 1

    if args.mode == "dryrun":
        from mp.execution.qmt_broker import Position
        plan_positions: list[Position] = []
        approx_cash = 0.0
        plan_path = Path(args.plan)
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan_acct = plan.get("account_snapshot", {}) or {}
            plan_positions = [
                Position(code=h["code"], name=h["name"],
                         shares_total=int(h.get("shares") or 0),
                         shares_available=int(h.get("shares") or 0),
                         avg_cost=float(h.get("avg_cost") or 0),
                         market_price=float(h.get("avg_cost") or 0),
                         market_value=int(h.get("shares") or 0) * float(h.get("avg_cost") or 0))
                for h in (plan.get("holdings_at_plan_time", []) or [])
            ]
            approx_mv = sum(p.market_value for p in plan_positions)
            plan_total = float(plan_acct.get("total_assets") or 0)
            approx_cash = plan_total - approx_mv
            if approx_cash < 0:
                approx_cash = float(plan_acct.get("cash_available") or 0)
        else:
            logger.warning("No plan at {} — running --emergency on empty DryRunBroker (no positions to liquidate)",
                           plan_path)
        account_id = args.qmt_account or "dryrun"
        broker = DryRunBroker(
            cash=approx_cash, positions=plan_positions, autofill=True,
            account_id=account_id,
        )
    else:
        if not args.qmt_account or not args.qmt_userdata:
            logger.error("--qmt-account and --qmt-userdata are required for live --emergency")
            return 1
        broker = QMTBroker(
            account_id=args.qmt_account,
            qmt_userdata_path=args.qmt_userdata,
        )

    broker.connect()
    try:
        result = broker.emergency_liquidate_all(confirm_string=args.confirm)
    except ValueError as e:
        logger.error("emergency_liquidate_all rejected: {}", e)
        if broker.is_connected():
            broker.disconnect()
        return 1
    _print_emergency_result(result)

    log_dir = ROOT / "data" / "orders" / "executions"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = log_dir / f"emergency_{stamp}.json"
    out.write_text(json.dumps({
        "executed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "emergency",
        "broker_mode": args.mode,
        "attempted_codes": result.attempted_codes,
        "succeeded_codes": result.succeeded_codes,
        "failed_codes": result.failed_codes,
        "cancelled_order_ids": result.cancelled_order_ids,
        "total_realized_cash": result.total_realized_cash,
        "duration_seconds": result.duration_seconds,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Emergency log → {}", out)

    if broker.is_connected():
        broker.disconnect()
    return 0 if not result.failed_codes else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="data/orders/latest.json",
                        help="path to order plan JSON (default: data/orders/latest.json)")
    parser.add_argument("--mode", choices=["dryrun", "interactive", "auto"],
                        default="dryrun")
    parser.add_argument("--qmt-account", help="QMT 资金账号")
    parser.add_argument("--qmt-userdata",
                        help=r"QMT userdata path, e.g. 'C:\\国金证券QMT交易端\\userdata_mini'")
    parser.add_argument("--max-orders", type=int, default=20)
    parser.add_argument("--max-single-pct", type=float, default=0.50)
    parser.add_argument("--price-drift-pct", type=float, default=0.02)
    parser.add_argument("--feishu", action="store_true",
                        help="push summary to Feishu when done")
    parser.add_argument(
        "--emergency", action="store_true",
        help="EMERGENCY: cancel pending orders + liquidate all sell-able "
             "positions. Requires --confirm=EMERGENCY_LIQUIDATE_<account_id>. "
             "Bypasses the normal plan execution path entirely.",
    )
    parser.add_argument(
        "--confirm",
        help="confirmation string for --emergency (must equal "
             "f'EMERGENCY_LIQUIDATE_{account_id}'). Mandatory when "
             "--emergency is passed; ignored otherwise.",
    )
    args = parser.parse_args()

    # Real-money kill switch (round 161 guardrail (d)): refuse live modes
    # if the freeze flag is set. Dryrun + --emergency liquidation bypass
    # this gate — emergency liquidation is the recovery path, not new
    # exposure.
    if not args.emergency:
        from mp.risk.freeze import guard_or_raise
        try:
            guard_or_raise(args.mode)
        except RuntimeError as e:
            logger.error("{}", e)
            return 4

    if args.emergency:
        return _run_emergency(args)

    plan_path = Path(args.plan)
    if not plan_path.exists():
        logger.error("Order plan not found: {}", plan_path)
        return 1
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    logger.info("Loaded plan from {} (generated {}, {} orders, {} alerts)",
                plan_path, plan.get("generated_at"),
                len(plan.get("orders", [])), len(plan.get("alerts", [])))

    # round 174: loud banner so log readers cannot mistake dryrun for live.
    if args.mode == "dryrun":
        logger.warning(
            "═══════════════════════════════════════════════════════════════"
        )
        logger.warning(
            "  DRYRUN MODE — 真单不会发到 QMT (DryRunBroker autofill 模拟)"
        )
        logger.warning(
            "  真单走 execute-live (当前 .disabled) 或人工 ssh `--mode auto`"
        )
        logger.warning(
            "═══════════════════════════════════════════════════════════════"
        )

    # Broker selection
    if args.mode == "dryrun":
        plan_acct = plan.get("account_snapshot", {}) or {}
        # Seed dryrun broker with plan-time account so reconcile passes.
        # We don't know live market prices; approximate position market_value
        # using shares × avg_cost (close enough for reconcile tolerance).
        from mp.execution.qmt_broker import Position
        plan_positions = [
            Position(code=h["code"], name=h["name"],
                     shares_total=int(h.get("shares") or 0),
                     shares_available=int(h.get("shares") or 0),
                     avg_cost=float(h.get("avg_cost") or 0),
                     market_price=float(h.get("avg_cost") or 0),
                     market_value=int(h.get("shares") or 0) * float(h.get("avg_cost") or 0))
            for h in (plan.get("holdings_at_plan_time", []) or [])
        ]
        # Approximate cash so total_assets matches the plan snapshot
        approx_mv = sum(p.market_value for p in plan_positions)
        plan_total = float(plan_acct.get("total_assets") or 0)
        approx_cash = plan_total - approx_mv
        broker = DryRunBroker(
            cash=approx_cash if approx_cash > 0 else float(plan_acct.get("cash_available") or 0),
            positions=plan_positions,
            autofill=True,
        )
    else:
        if not args.qmt_account or not args.qmt_userdata:
            logger.error("--qmt-account and --qmt-userdata are required for live modes")
            return 1
        broker = QMTBroker(
            account_id=args.qmt_account,
            qmt_userdata_path=args.qmt_userdata,
        )

    # Engage Arm B daily-budget tracker iff this plan came from the 14:30
    # OOS bucket. EOD / morning 9:30 plans run uncapped (their cap is
    # account-level concentration above). Round 161 guardrail (a).
    arm_b_tracker = None
    if plan.get("entry_path") == "intraday_14_30":
        from mp.risk.arm_b_budget import ArmBBudgetTracker
        arm_b_tracker = ArmBBudgetTracker.load()
        logger.info("Arm B budget tracker engaged: committed={:.0f}/{:.0f} RMB",
                    arm_b_tracker.committed(), arm_b_tracker.budget_max())

    results = run(
        broker, plan,
        mode=args.mode,
        max_orders=args.max_orders,
        max_single_pct=args.max_single_pct,
        price_drift_pct=args.price_drift_pct,
        arm_b_tracker=arm_b_tracker,
    )

    summary_md = _format_summary(results, mode=args.mode)
    print("\n" + summary_md)

    # Persist execution log. Per P11-5 round 97 decision (4): co-locate
    # 14:30 + 9:30 fills under data/orders/executions/ but tag the
    # intraday filename so account_report grep and dir cleanup stay
    # uniform. Detection comes from the plan's own entry_path marker
    # (set by scripts/intraday_plan.py) so the executor doesn't need a
    # new CLI flag.
    log_dir = ROOT / "data" / "orders" / "executions"
    log_dir.mkdir(parents=True, exist_ok=True)
    date_part = time.strftime("%Y%m%d")
    time_part = time.strftime("%H%M%S")
    if plan.get("entry_path") == "intraday_14_30":
        out = log_dir / f"exec_{date_part}_intraday_{time_part}.json"
    else:
        out = log_dir / f"exec_{date_part}_{time_part}.json"
    out.write_text(json.dumps({
        "executed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "entry_path": plan.get("entry_path"),
        "plan_generated_at": plan.get("generated_at"),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Execution log → {}", out)

    if args.feishu:
        try:
            from scripts.daily_report import send_to_feishu
            send_to_feishu(summary_md)
        except Exception as e:
            logger.warning("Feishu push failed (non-fatal): {}", e)

    if broker.is_connected():
        broker.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
