"""Re-sync config/portfolio.yaml from QMT 8886933837 (国金证券) via SSH to ECS.

Solves the recurring stale-yaml problem: the file used to be hand-edited from
broker screenshots, and after live trading runs the holdings drifted from
reality (5/28 burn: yaml said 002984 was held, user had already cleared it →
plan tried to "清仓" a zero-share position).

This script:
  1. SSH to ECS Windows, run a tiny xtquant snippet that prints JSON to stdout
     (no file write on ECS; one round-trip).
  2. Fetch stock 名称 for each held code via Sina API (proxy-safe, like
     daily_report.get_stock_names).
  3. Atomically overwrite config/portfolio.yaml, preserving the doc-header
     block (everything above `account:`). Body is fully regenerated.

Wire it into scripts/daily_report.sh BEFORE daily_report.py runs.

Usage:
    .venv/bin/python scripts/sync_portfolio_from_qmt.py
        [--ecs-user Administrator] [--ecs-host 14.103.49.51]
        [--portfolio config/portfolio.yaml]

Exits non-zero on failure; daily_report.sh should gate the rest of the pipeline.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def fetch_qmt_snapshot(ecs_user: str, ecs_host: str) -> Dict[str, Any]:
    """SSH to ECS, run scripts/qmt_snapshot.py via .venv python, return parsed JSON.

    Assumes ECS has the repo pulled at C:\\money-printer with scripts/qmt_snapshot.py
    present (committed alongside this file). No stdin-piped snippet — the script
    lives in the repo so SSH command is simple.
    """
    cmd = [
        "ssh", "-o", "ConnectTimeout=15", f"{ecs_user}@{ecs_host}",
        "powershell -Command "
        + shlex.quote(
            r'cd C:\money-printer; '
            r'$env:PYTHONIOENCODING="utf-8"; '
            r'.venv\Scripts\python.exe -X utf8 scripts\qmt_snapshot.py'
        ),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", timeout=60,
    )
    if result.returncode != 0:
        sys.stderr.write(
            f"[sync_portfolio] ECS SSH failed (exit {result.returncode})\n"
            f"  STDOUT: {result.stdout[-500:]}\n  STDERR: {result.stderr[-500:]}\n"
        )
        raise RuntimeError("ECS snippet failed")

    # Find the marker line — ECS may emit xtquant connect banner etc. before it.
    marker = "__QMT_JSON__ "
    for line in result.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError(
        "no __QMT_JSON__ marker in ECS stdout:\n" + result.stdout[-500:]
    )


def fetch_names(codes: List[str]) -> Dict[str, str]:
    """Reuse daily_report.get_stock_names via Sina API."""
    if not codes:
        return {}
    from scripts.daily_report import get_stock_names
    return get_stock_names(codes)


def render_portfolio_yaml(snapshot: Dict[str, Any], names: Dict[str, str],
                          header: str) -> str:
    """Render the new yaml string. Header is preserved verbatim above
    `account:`."""
    acc = snapshot["account"]
    pos = snapshot["positions"]
    today = datetime.now().strftime("%Y-%m-%d")

    pos_pct = acc["market_value"] / acc["total_assets"] if acc["total_assets"] > 0 else 0.0

    lines: List[str] = []
    lines.append(header.rstrip())
    lines.append("")
    lines.append("# 账户快照（结构化，供 4 点日报订单清单生成用）")
    lines.append("account:")
    lines.append(f"  total_assets: {acc['total_assets']:.2f}      # 总资产 (QMT auto-sync {today})")
    lines.append(f"  market_value: {acc['market_value']:.2f}       # 持仓市值")
    lines.append(f"  cash_available: {acc['cash_available']:.2f}     # 可用资金")
    lines.append(f"  position_pct: {pos_pct:.3f}          # 仓位")
    lines.append(f"  target_position_pct: 0.70    # 订单生成目标仓位（可调）")
    lines.append(f"  updated_at: '{today}'")
    lines.append("")
    lines.append("holdings:")
    for p in pos:
        code = p["code"]
        name = names.get(code, code)
        lines.append(f"- name: {name}")
        lines.append(f"  code: '{code}'")
        lines.append(f"  type: stock")
        lines.append(f"  board: ''")
        lines.append(f"  shares: {p['shares']}")
        lines.append(f"  avg_cost: {p['avg_cost']:.3f}")
        lines.append(f"  entry_date: '{today}'   # auto-sync 写入时间; 实际建仓日未知")
    return "\n".join(lines) + "\n"


def extract_header(text: str) -> str:
    """Everything above the first `account:` line (or whole file if absent)."""
    out = []
    for line in text.splitlines():
        if line.strip().startswith("account:"):
            break
        out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecs-user", default="Administrator")
    ap.add_argument("--ecs-host", default="14.103.49.51")
    ap.add_argument("--portfolio", default="config/portfolio.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="print rendered yaml to stdout, do not overwrite")
    args = ap.parse_args()

    portfolio_path = Path(args.portfolio)
    if not portfolio_path.exists():
        sys.stderr.write(f"[sync_portfolio] {portfolio_path} not found\n")
        sys.exit(2)

    print(f"[sync_portfolio] fetching QMT snapshot via SSH {args.ecs_user}@{args.ecs_host} ...")
    snapshot = fetch_qmt_snapshot(args.ecs_user, args.ecs_host)
    n_pos = len(snapshot["positions"])
    print(f"[sync_portfolio]   account total={snapshot['account']['total_assets']:.2f} "
          f"cash={snapshot['account']['cash_available']:.2f} positions={n_pos}")

    codes = [p["code"] for p in snapshot["positions"]]
    names = fetch_names(codes) if codes else {}
    print(f"[sync_portfolio]   resolved {len(names)}/{len(codes)} names via Sina")

    header = extract_header(portfolio_path.read_text(encoding="utf-8"))
    new_text = render_portfolio_yaml(snapshot, names, header)

    if args.dry_run:
        print("--- rendered yaml (dry-run, not written) ---")
        print(new_text)
        return

    # Atomic write: tmp then rename
    tmp = portfolio_path.with_suffix(".yaml.tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(portfolio_path)
    print(f"[sync_portfolio] {portfolio_path} updated ({n_pos} positions)")


if __name__ == "__main__":
    main()
