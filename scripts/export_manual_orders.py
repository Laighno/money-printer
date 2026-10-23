"""Export the daily plan as a human-executable order sheet (manual fallback).

WHY (2026-10-23): 国金 announced that from 2026-10-23 the QMT built-in Python
blocks file I/O and network access. The BigQMT file bridge IS file I/O end to
end — heartbeat, req_*/resp_* RPC, pending-order state — so once the limit
lands the bridge stops and ``execute_orders.py`` can no longer reach the
broker. This script is the fallback that needs no bridge, no xtquant and no
built-in Python: it turns ``data/orders/latest.json`` into

  1. a Markdown sheet the user reads in Feishu / on screen, and
  2. a CSV laid out for QMT's "导入委托" dialog,

so the user can place the same orders by hand in ~2 minutes.

The CSV column layout is a BEST GUESS at QMT's import format (code, direction,
price type, price, quantity) and is NOT verified against the live client — the
install carries no import template to copy. Treat the Markdown as the source of
truth, check the CSV against the import dialog once, and adjust ``CSV_HEADER``
/ ``_csv_row`` if the client disagrees. ``--csv-dialect`` switches between the
layouts seen in the wild.

Usage (ECS or Mac, no broker needed)::

    python scripts/export_manual_orders.py
    python scripts/export_manual_orders.py --plan data/orders/orders_20261023.json
    python scripts/export_manual_orders.py --feishu        # also push the sheet

Exit codes: 0 = sheet written (even when the plan is empty), 1 = no plan found.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_PLAN = _REPO / "data" / "orders" / "latest.json"
OUT_DIR = _REPO / "data" / "orders" / "manual"

# QMT import dialects. "guojin" is the layout most 国金 QMT builds accept;
# "generic" is the bare code/direction/price/qty form. Verify once against the
# real 导入委托 dialog and keep the one that loads cleanly.
CSV_DIALECTS: dict[str, list[str]] = {
    "guojin": ["证券代码", "买卖方向", "委托类型", "委托价格", "委托数量"],
    "generic": ["code", "direction", "price", "volume"],
}


def _is_buy(order: dict[str, Any]) -> bool:
    """Plan orders carry a Chinese ``action`` plus a signed ``cost``.

    ``cost`` is the reliable discriminator: > 0 spends cash (buy), < 0 raises
    cash (sell). ``action`` is display text (买入/加仓/减仓/清仓/换仓/保持).
    """
    return float(order.get("cost", 0) or 0) > 0


def _csv_row(order: dict[str, Any], dialect: str) -> list[str]:
    code = str(order["code"]).zfill(6)
    direction_cn = "买入" if _is_buy(order) else "卖出"
    price = f"{float(order['limit_price']):.2f}"
    qty = str(int(order["shares"]))
    if dialect == "guojin":
        return [code, direction_cn, "限价", price, qty]
    return [code, "buy" if _is_buy(order) else "sell", price, qty]


def _exchange_hint(code: str) -> str:
    """Some import dialogs want the exchange; show it so the user can fill in."""
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "51", "11")):
        return "上交所"
    if code.startswith(("00", "30", "12", "15")):
        return "深交所"
    if code.startswith(("8", "4", "92")):
        return "北交所"
    return "?"


def build_sheet(plan: dict[str, Any], plan_path: Path) -> str:
    orders = plan.get("orders", []) or []
    buys = [o for o in orders if _is_buy(o)]
    sells = [o for o in orders if not _is_buy(o)]
    snap = plan.get("account_snapshot") or {}

    spend = sum(float(o["shares"]) * float(o["limit_price"]) for o in buys)
    proceeds = sum(float(o["shares"]) * float(o["limit_price"]) for o in sells)

    lines: list[str] = []
    lines.append("# 手工下单清单（交易桥不可用时使用）")
    lines.append("")
    lines.append(f"- 计划文件：`{plan_path.name}`，生成于 {plan.get('generated_at', '?')}")
    lines.append(f"- 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if snap:
        ta = float(snap.get("total_assets") or 0)
        mv = float(snap.get("market_value") or 0)
        cash = float(snap.get("cash_available") or 0)
        pct = (mv / ta * 100) if ta else 0
        lines.append(f"- 计划时账户：总资产 {ta:,.0f}，持仓 {mv:,.0f}（{pct:.1f}%），可用 {cash:,.0f}")
    lines.append(f"- 本次：买入 {len(buys)} 笔约 {spend:,.0f} 元，卖出 {len(sells)} 笔约 {proceeds:,.0f} 元")
    lines.append("")
    lines.append("**先卖后买**：A 股卖出当日资金可用，先执行卖单可为买单腾出资金。")
    lines.append("")

    if not orders:
        lines.append("今日计划没有订单，无需操作。")
        alerts = plan.get("alerts") or []
        if alerts:
            lines.append("")
            lines.append("## 提示")
            for a in alerts:
                lines.append(f"- {a}")
        return "\n".join(lines) + "\n"

    for title, group in (("① 卖出", sells), ("② 买入", buys)):
        if not group:
            continue
        lines.append(f"## {title}（{len(group)} 笔）")
        lines.append("")
        lines.append("| 代码 | 名称 | 交易所 | 股数 | 限价 | 金额 |")
        lines.append("|---|---|---|---:|---:|---:|")
        for o in group:
            code = str(o["code"]).zfill(6)
            amount = float(o["shares"]) * float(o["limit_price"])
            lines.append(
                f"| {code} | {o.get('name', '')} | {_exchange_hint(code)} | "
                f"{int(o['shares'])} | {float(o['limit_price']):.2f} | {amount:,.0f} |"
            )
        lines.append("")

    alerts = plan.get("alerts") or []
    if alerts:
        lines.append("## 计划提示")
        for a in alerts:
            lines.append(f"- {a}")
        lines.append("")

    lines.append("## 操作步骤")
    lines.append("")
    lines.append("1. 打开 QMT 客户端，确认已登录交易。")
    lines.append("2. 优先手工输入上表（笔数少时最稳）；或用「导入委托」载入同目录的 csv。")
    lines.append("3. 核对方向、股数、限价后提交。")
    lines.append("4. 收盘后在「委托查询」确认成交，未成交的次日会重新进入计划。")
    lines.append("")
    lines.append("> csv 的列格式未在客户端实测过，**以上表为准**；首次导入若报错，")
    lines.append("> 把导入对话框要求的列告诉我，我调整 `--csv-dialect`。")
    return "\n".join(lines) + "\n"


def write_outputs(plan: dict[str, Any], plan_path: Path, out_dir: Path,
                  dialect: str) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    md_path = out_dir / f"manual_orders_{stamp}.md"
    md_path.write_text(build_sheet(plan, plan_path), encoding="utf-8")

    orders = plan.get("orders", []) or []
    if not orders:
        return md_path, None

    csv_path = out_dir / f"manual_orders_{stamp}.csv"
    # QMT reads GBK on Windows; utf-8-sig would show mojibake in the dialog.
    with csv_path.open("w", encoding="gbk", newline="", errors="replace") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_DIALECTS[dialect])
        for o in sorted(orders, key=lambda x: _is_buy(x)):   # sells first
            w.writerow(_csv_row(o, dialect))
    return md_path, csv_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN,
                    help=f"plan JSON (default {DEFAULT_PLAN})")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help=f"output directory (default {OUT_DIR})")
    ap.add_argument("--csv-dialect", choices=sorted(CSV_DIALECTS), default="guojin",
                    help="column layout for the import csv (default guojin)")
    ap.add_argument("--feishu", action="store_true",
                    help="also push the sheet to Feishu")
    args = ap.parse_args()

    if not args.plan.exists():
        sys.stderr.write(f"[manual_orders] plan not found: {args.plan}\n")
        return 1

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    md_path, csv_path = write_outputs(plan, args.plan, args.out_dir, args.csv_dialect)

    sys.stdout.write(f"[manual_orders] sheet  -> {md_path}\n")
    if csv_path:
        sys.stdout.write(f"[manual_orders] import -> {csv_path}\n")
    else:
        sys.stdout.write("[manual_orders] plan has no orders; csv skipped\n")

    if args.feishu:
        try:
            from scripts.daily_report import send_to_feishu
            send_to_feishu(md_path.read_text(encoding="utf-8"))
            sys.stdout.write("[manual_orders] pushed to Feishu\n")
        except Exception as e:      # noqa: BLE001 - notification must not fail the export
            sys.stderr.write(f"[manual_orders] Feishu push failed: {e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
