"""P6-X3 paper_trade NAV vs walk_forward Sharpe divergence monitor.

What this monitors
------------------
Two production "Sharpe" numbers should track each other but can quietly
diverge:

  1. walk_forward_backtest's expanding-window Sharpe (from the latest
     ``data/reports/backtest_history.json`` snapshot)
  2. paper_trade's realized rolling Sharpe (computed from the last 20
     daily NAV returns in ``data/paper_trade/state.json::nav_history``)

If (2) starts running materially below (1), it usually means execution
drift — slippage assumptions wrong, fills happening late, wrong universe,
data quality gate firing on real days, etc.  ``walk_forward`` is the
idealized model; ``paper_trade`` is the realized one.  Their delta is
the execution gap, and a sudden widening should be visible to the
operator before they over-trust the backtest.

Why NOT real-time / daily
-------------------------
Single-day NAV is too noisy: A-share daily jitter ≈ ±2-3 %, so even a
true zero-edge paper_trade looks like ±100 % annualized for a few days.
Weekly cadence + 20-day rolling window gives enough denoising.

Why a separate file with no walk_forward / paper_trade imports
--------------------------------------------------------------
Same dependency-loop reason as ``weekly_heartbeat.py`` / ``cron_drift_detect.py``:
if those modules break, the monitor must still fire.  This script ONLY
reads two JSON files; no Python imports of trading code.

Min N=15 floor (cold-start protection)
--------------------------------------
``nav_history`` has to be ≥ 15 entries before the monitor will fire any
alert — that's enough for 14 returns + small std signal.  Below that we
log ``insufficient NAV history (N=…)`` and exit 0.  Avoids false
positives during the first two weeks of paper-trading.

Schedule
--------
Saturday 06:30 (heartbeat 06:00 + 30 min — independent fail domain but
same batch).  See ``docs/cron_setup.md``.

Thresholds (initial; will need σ-grounding in deferred P8)
-----------
- YELLOW: ``|Δ Sharpe| > 0.5``
- RED:    ``|Δ Sharpe| > 1.0  AND  paper_trade rolling Sharpe < 0``
  (a large divergence is only RED-worthy if paper is actually losing —
  otherwise it's "we got lucky" not "execution is broken")

Exit codes
----------
0 — always (even on RED / YELLOW alert sent; cron retry can't fix)
1 — script-level config failure (e.g. couldn't resolve repo root)
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Resolve repo root robustly
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent.parent
if not (_REPO / "scripts").is_dir():
    print(f"paper_trade_drift_detect: cannot resolve repo root from {_THIS}",
          file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(_REPO))


# ───────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────

STATE_PATH = _REPO / "data" / "paper_trade" / "state.json"
HISTORY_PATH = _REPO / "data" / "reports" / "backtest_history.json"

MIN_NAV_HISTORY = 15        # cold-start floor (>= 14 returns + 1 std signal)
ROLLING_WINDOW = 20         # daily NAV returns for rolling Sharpe
TRADING_DAYS_PER_YEAR = 252

YELLOW_THRESHOLD = 0.5      # |Δ Sharpe|
RED_THRESHOLD = 1.0         # combined with paper_rolling < 0

WALK_FORWARD_STALE_DAYS = 21  # if history latest > 21 d old, skip — heartbeat
                              # would already have alerted; don't double-fire


# ───────────────────────────────────────────────────────────────────────
# Rolling Sharpe
# ───────────────────────────────────────────────────────────────────────

def rolling_sharpe(navs: list[float], window: int = ROLLING_WINDOW) -> Optional[float]:
    """Annualized Sharpe from the last ``window`` daily NAV returns.

    Returns ``None`` if there are fewer than ``window + 1`` NAV points
    (need at least ``window`` returns) or the realized return std is
    zero.
    """
    if len(navs) < window + 1:
        return None
    # daily returns over the most recent window
    tail = navs[-(window + 1):]
    rets = []
    for i in range(1, len(tail)):
        prev = tail[i - 1]
        if prev <= 0:
            return None
        rets.append(tail[i] / prev - 1.0)
    if not rets:
        return None
    mean = sum(rets) / len(rets)
    # sample std with ddof=1
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    if var <= 0:
        return None
    std = math.sqrt(var)
    return mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)


# ───────────────────────────────────────────────────────────────────────
# Data loaders (json-only; no walk_forward / paper_trade imports)
# ───────────────────────────────────────────────────────────────────────

def load_nav_history() -> list[dict]:
    if not STATE_PATH.exists():
        return []
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return list(data.get("nav_history", []))


def load_latest_walk_forward() -> Optional[dict]:
    """Return the most-recent backtest_history.json entry, or None."""
    if not HISTORY_PATH.exists():
        return None
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data:
        return None
    return data[-1]


def _parse_sharpe_field(metrics: dict) -> Optional[float]:
    """Pull ``bt_metrics.sharpe_ratio`` out and coerce to float.

    The walk_forward writer renders Sharpe as a string like ``"1.21"``;
    older snapshots may carry a raw float.  Strip any trailing chars
    (no `%` expected but be defensive) and tolerate both shapes.
    """
    bt = metrics.get("bt_metrics") if isinstance(metrics, dict) else None
    if not isinstance(bt, dict):
        return None
    raw = bt.get("sharpe_ratio")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_history_date(entry: dict) -> Optional[datetime]:
    raw = entry.get("date")
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d")
    except ValueError:
        return None


# ───────────────────────────────────────────────────────────────────────
# Top-level check
# ───────────────────────────────────────────────────────────────────────

def check(now: Optional[datetime] = None) -> dict:
    """Compose the divergence check; no I/O side effects beyond reading
    the two JSON files (and only when they exist).

    ``now`` injectable for tests.  When None, uses ``datetime.now()``.
    """
    now = now or datetime.now()
    nav_history = load_nav_history()
    nav_n = len(nav_history)

    # ── Cold start floor ────────────────────────────────────────
    if nav_n < MIN_NAV_HISTORY:
        return {
            "healthy": True,
            "level": "OK",
            "msg": (f"insufficient NAV history (N={nav_n}, need "
                    f">={MIN_NAV_HISTORY}); skipping alert during "
                    f"cold-start period"),
            "paper_sharpe": None,
            "wf_sharpe": None,
            "delta": None,
            "nav_n": nav_n,
            "wf_date": None,
        }

    # ── Extract NAV floats ──────────────────────────────────────
    navs: list[float] = []
    for entry in nav_history:
        try:
            navs.append(float(entry.get("nav")))
        except (TypeError, ValueError):
            navs.append(float("nan"))
    if any(math.isnan(v) for v in navs):
        return {
            "healthy": True,
            "level": "OK",
            "msg": "NAV history contains non-numeric entries; skipping",
            "paper_sharpe": None, "wf_sharpe": None, "delta": None,
            "nav_n": nav_n, "wf_date": None,
        }

    paper = rolling_sharpe(navs, ROLLING_WINDOW)
    if paper is None:
        return {
            "healthy": True,
            "level": "OK",
            "msg": (f"rolling_sharpe returned None (N={nav_n}, "
                    f"window={ROLLING_WINDOW}); insufficient or zero-std data"),
            "paper_sharpe": None, "wf_sharpe": None, "delta": None,
            "nav_n": nav_n, "wf_date": None,
        }

    # ── Walk-forward latest ─────────────────────────────────────
    latest = load_latest_walk_forward()
    if latest is None:
        return {
            "healthy": True,
            "level": "OK",
            "msg": ("backtest_history.json missing or empty; "
                    "heartbeat would have alerted — skipping"),
            "paper_sharpe": paper, "wf_sharpe": None, "delta": None,
            "nav_n": nav_n, "wf_date": None,
        }

    wf_sharpe = _parse_sharpe_field(latest)
    wf_date = _parse_history_date(latest)

    if wf_sharpe is None:
        return {
            "healthy": True,
            "level": "OK",
            "msg": (f"latest backtest_history entry has no parseable "
                    f"sharpe_ratio (raw="
                    f"{(latest.get('bt_metrics') or {}).get('sharpe_ratio')!r}); "
                    f"skipping"),
            "paper_sharpe": paper, "wf_sharpe": None, "delta": None,
            "nav_n": nav_n,
            "wf_date": wf_date.strftime("%Y-%m-%d") if wf_date else None,
        }

    # ── Walk-forward stale check ────────────────────────────────
    if wf_date is not None:
        age = now - wf_date
        if age > timedelta(days=WALK_FORWARD_STALE_DAYS):
            return {
                "healthy": True,
                "level": "OK",
                "msg": (f"walk_forward Sharpe data is "
                        f"{age.days}d old (> {WALK_FORWARD_STALE_DAYS}); "
                        f"heartbeat should have alerted — skipping drift "
                        f"check until that's resolved"),
                "paper_sharpe": paper, "wf_sharpe": wf_sharpe,
                "delta": None, "nav_n": nav_n,
                "wf_date": wf_date.strftime("%Y-%m-%d"),
            }

    # ── Decision ────────────────────────────────────────────────
    delta = paper - wf_sharpe
    abs_delta = abs(delta)
    wf_date_s = wf_date.strftime("%Y-%m-%d") if wf_date else "(no date)"

    base_msg = (
        f"paper_trade rolling Sharpe (N={ROLLING_WINDOW}) = {paper:.3f}; "
        f"walk_forward latest Sharpe ({wf_date_s}) = {wf_sharpe:.3f}; "
        f"Δ = {delta:+.3f}; NAV history length = {nav_n}"
    )

    if abs_delta > RED_THRESHOLD and paper < 0:
        return {
            "healthy": False,
            "level": "RED",
            "msg": (f"{base_msg}. |Δ| > {RED_THRESHOLD} AND paper "
                    f"Sharpe negative — execution drift causing real "
                    f"losses. INVESTIGATE."),
            "paper_sharpe": paper, "wf_sharpe": wf_sharpe, "delta": delta,
            "nav_n": nav_n, "wf_date": wf_date_s,
        }
    if abs_delta > YELLOW_THRESHOLD:
        return {
            "healthy": False,
            "level": "YELLOW",
            "msg": (f"{base_msg}. |Δ| > {YELLOW_THRESHOLD} — execution "
                    f"drift; review fills / slippage / data quality gate "
                    f"frequency"),
            "paper_sharpe": paper, "wf_sharpe": wf_sharpe, "delta": delta,
            "nav_n": nav_n, "wf_date": wf_date_s,
        }
    return {
        "healthy": True,
        "level": "OK",
        "msg": (f"{base_msg}. Within ±{YELLOW_THRESHOLD} tolerance — "
                f"execution tracking walk_forward"),
        "paper_sharpe": paper, "wf_sharpe": wf_sharpe, "delta": delta,
        "nav_n": nav_n, "wf_date": wf_date_s,
    }


# ───────────────────────────────────────────────────────────────────────
# Render
# ───────────────────────────────────────────────────────────────────────

def format_for_feishu(status: dict) -> str:
    if status["healthy"]:
        return ""
    emoji = "🚨" if status["level"] == "RED" else "⚠"

    def _fmt(val, spec):
        return format(val, spec) if val is not None else "n/a"

    paper_line = f"- paper_trade rolling Sharpe: {_fmt(status.get('paper_sharpe'), '.3f')}"
    wf_line = (f"- walk_forward Sharpe ({status.get('wf_date') or 'n/a'}): "
               f"{_fmt(status.get('wf_sharpe'), '.3f')}")
    delta_line = f"- Δ = {_fmt(status.get('delta'), '+.3f')}"
    nav_line = f"- NAV history length: {status.get('nav_n')}"

    return "\n".join([
        f"# {emoji} {status['level']} ALERT: paper_trade Sharpe drift",
        "",
        status["msg"],
        "",
        paper_line,
        wf_line,
        delta_line,
        nav_line,
        "",
        "Diagnostics:",
        "1. Read paper_trade reports for last 4 weeks; compare hit-rate vs walk_forward",
        "2. Check `data/logs/paper_trade.log` for data-quality-gate / pending-skip events",
        "3. Compare slippage / commission assumptions in `paper_trade.py` vs realized fills",
        "4. If divergence is recent (last week), check if a market regime shifted",
        "",
        "Source: `scripts/monitor/paper_trade_drift_detect.py` (P6-X3, round 47)",
    ])


# ───────────────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    dry_run = "--dry-run" in argv

    status = check()
    print(f"[paper_trade_drift_detect] {status['level']}: {status['msg']}")

    if status["healthy"]:
        return 0

    block = format_for_feishu(status)
    if dry_run:
        print("=== DRY RUN — would send to Feishu ===")
        print(block)
        return 0

    try:
        from scripts.daily_report import send_to_feishu
    except Exception as e:
        print(f"[paper_trade_drift_detect] cannot import send_to_feishu: {e}",
              file=sys.stderr)
        return 0

    ok = send_to_feishu(block)
    print(f"[paper_trade_drift_detect] Feishu send returned {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
