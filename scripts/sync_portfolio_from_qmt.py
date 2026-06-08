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


def fetch_qmt_snapshot_local() -> Dict[str, Any]:
    """ECS-local mode (round 197 fix): build snapshot in-process, no SSH.

    Use when running ON ECS (e.g. from scheduled task) — SSH self-loop would
    hang. Mirrors qmt_snapshot.py logic, returns dict (not stdout-marker form).
    """
    from mp.execution.qmt_broker import QMTBroker
    broker = QMTBroker(account_id="8886933837", qmt_userdata_path=r"C:\guojin\userdata_mini")
    broker.connect()
    try:
        info = broker.get_account_info()
        positions = broker.get_positions()
    finally:
        broker.disconnect()
    return {
        "account": {
            "total_assets": float(info.total_assets),
            "cash_available": float(info.cash_available),
            "market_value": float(info.market_value),
            "updated_at": info.updated_at,
        },
        "positions": [
            {
                "code": p.code,
                "name": p.name or "",
                "shares": int(p.shares_total),
                "avg_cost": float(p.avg_cost),
                "market_price": float(getattr(p, "market_price", 0.0) or 0.0),
                "market_value": float(getattr(p, "market_value", 0.0) or 0.0),
            }
            for p in positions if int(p.shares_total) > 0
        ],
    }


def fetch_qmt_snapshot(ecs_user: str, ecs_host: str) -> Dict[str, Any]:
    """SSH to ECS, run scripts/qmt_snapshot.py via .venv python, return parsed JSON.

    Assumes ECS has the repo pulled at C:\\money-printer with scripts/qmt_snapshot.py
    present (committed alongside this file). No stdin-piped snippet — the script
    lives in the repo so SSH command is simple.

    Use --local flag (or fetch_qmt_snapshot_local() directly) when running on ECS
    itself, to avoid SSH self-loop hang.
    """
    # Pass the entire ssh command as a single string to subprocess to avoid
    # double-shell-escape (Mac → ssh → cmd.exe → powershell). Empirically the
    # list-form ssh command was being echoed not executed.
    remote_cmd = (
        r'cd C:\money-printer; '
        r'$env:PYTHONIOENCODING="utf-8"; '
        r'.venv\Scripts\python.exe -X utf8 scripts\qmt_snapshot.py'
    )
    ssh_cmd = f'ssh -o ConnectTimeout=15 {ecs_user}@{ecs_host} \'powershell -Command "{remote_cmd}"\''
    result = subprocess.run(
        ssh_cmd, shell=True, capture_output=True, text=True,
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


# round 168 sanity bounds (matches scripts/daily_report.py and
# scripts/arm_b_stop_monitor.py). Any account value outside [1e3, 1e9] OR
# matching a known QMT mock/未初始化哨兵 is rejected before write — better
# to refuse to overwrite portfolio.yaml than to write 100亿 over real data.
_TOTAL_ASSETS_MIN = 1_000.0
_TOTAL_ASSETS_MAX = 1_000_000_000.0
_SENTINEL_VALUES = {9_999_999_999.0, 10_000_000_000.0}


def validate_snapshot(snapshot: Dict[str, Any]) -> None:
    """Raise ValueError if the QMT snapshot looks polluted (round 168 incident).

    5/30 incident: sync wrote total_assets=10000071328 / cash=9999999999 over
    real ~28 万 — likely QMT had not connected / returned defaults. Guard at
    the writer so 哨兵值 cannot reach disk.
    """
    acc = snapshot.get("account") or {}
    ta = float(acc.get("total_assets") or 0)
    cash = float(acc.get("cash_available") or 0)
    mv = float(acc.get("market_value") or 0)
    if ta <= _TOTAL_ASSETS_MIN or ta >= _TOTAL_ASSETS_MAX:
        raise ValueError(
            f"snapshot.total_assets={ta} outside sane bounds "
            f"[{_TOTAL_ASSETS_MIN}, {_TOTAL_ASSETS_MAX}] — refusing to write. "
            "Likely QMT 未连接 / mock fallback."
        )
    if ta in _SENTINEL_VALUES or cash in _SENTINEL_VALUES:
        raise ValueError(
            f"snapshot has known QMT sentinel value (total={ta}, cash={cash}). "
            "Refusing — re-run after confirming QMT desktop is connected + logged in."
        )
    if cash < 0:
        raise ValueError(f"snapshot.cash_available={cash} is negative — refusing.")
    if mv > 0 and ta < mv:
        raise ValueError(
            f"snapshot.total_assets ({ta}) < market_value ({mv}) — impossible, refusing."
        )


# round 168 (c) — the placeholder comment line above `account:` MUST be unique.
# extract_header() prior to this fix kept everything above `account:`, so each
# sync iteration prepended a fresh comment while preserving the prior one —
# the file accumulated 4 copies before the incident exposed the bug.
_ACCOUNT_HEADER_COMMENT = "# 账户快照（结构化，供 4 点日报订单清单生成用）"


def render_portfolio_yaml(snapshot: Dict[str, Any], names: Dict[str, str],
                          header: str) -> str:
    """Render the new yaml string. Header is preserved verbatim above
    `account:`."""
    validate_snapshot(snapshot)
    acc = snapshot["account"]
    pos = snapshot["positions"]
    today = datetime.now().strftime("%Y-%m-%d")

    pos_pct = acc["market_value"] / acc["total_assets"] if acc["total_assets"] > 0 else 0.0

    lines: List[str] = []
    lines.append(header.rstrip())
    lines.append("")
    lines.append(_ACCOUNT_HEADER_COMMENT)
    lines.append("account:")
    lines.append(f"  total_assets: {acc['total_assets']:.2f}      # 总资产 (QMT auto-sync {today})")
    lines.append(f"  market_value: {acc['market_value']:.2f}       # 持仓市值")
    lines.append(f"  cash_available: {acc['cash_available']:.2f}     # 可用资金")
    lines.append(f"  position_pct: {pos_pct:.3f}          # 仓位")
    lines.append(f"  target_position_pct: 0.85    # 订单生成目标仓位（可调; user 2026-06-08 70%->85%）")
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
    """Everything above the first `account:` line (or whole file if absent).

    round 168 (c) fix: strip trailing blank lines AND any trailing copy of
    `_ACCOUNT_HEADER_COMMENT` so re-syncs don't accumulate duplicate header
    comments. Pre-fix portfolio.yaml had 4 copies of the comment line.
    """
    out = []
    for line in text.splitlines():
        if line.strip().startswith("account:"):
            break
        out.append(line)
    # Drop trailing blank-or-duplicate-comment lines so render can re-add
    # exactly one comment + blank line below the header.
    while out and (out[-1].strip() == "" or out[-1].strip() == _ACCOUNT_HEADER_COMMENT):
        out.pop()
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecs-user", default="Administrator")
    ap.add_argument("--ecs-host", default="14.103.49.51")
    ap.add_argument("--portfolio", default="config/portfolio.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="print rendered yaml to stdout, do not overwrite")
    ap.add_argument("--local", action="store_true",
                    help="ECS-local mode: call qmt_snapshot in-process, no SSH "
                         "(use when running ON ECS, avoids self-SSH hang)")
    args = ap.parse_args()

    portfolio_path = Path(args.portfolio)
    if not portfolio_path.exists():
        sys.stderr.write(f"[sync_portfolio] {portfolio_path} not found\n")
        sys.exit(2)

    if args.local:
        print(f"[sync_portfolio] ECS-local mode (in-process qmt_snapshot, no SSH)")
        snapshot = fetch_qmt_snapshot_local()
    else:
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
