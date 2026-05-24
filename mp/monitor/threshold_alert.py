"""Performance-threshold alert generation for weekly walk-forward.

P4-1C (docs/dialog/ rounds 38-39): until this module existed BASELINE.md
§4 thresholds were purely documentary — a Sharpe/DD breach would sit in
walk_forward_result.md until someone read it (advisor estimated up to 5
trading days of silent drift). This module is the source of truth for
those 3 numeric thresholds and produces the markdown block that
`send_model_update_report` injects into its Feishu message.

Source of truth contract
------------------------
Thresholds here MIRROR `data/reports/BASELINE.md` §4.1 (current
production: hs300+zz500 + 64-feature + winsorize + BlendRanker conviction,
seed=42 → Sharpe 1.90 / annual 60.42% / Max DD -36.30%). If BASELINE §4
changes, update this module and bump the comment date below.

  BASELINE last synced: 2026-05-24 (commit a947303, P2-#2 re-baseline)

3 indicators only
-----------------
Sharpe, annual return, max drawdown — the headline trio. Win rate, IC
health, style drift etc. (BASELINE §4.2-4.4) remain manual review per
BASELINE §4 intent ("看这些"). Adding more here is a P5 decision; do
not extend without advisor sign-off (round-39 spec was explicit on
keeping this minimal).

Threshold rationale (P5-A-light, docs/dialog/ round 41)
-------------------------------------------------------
The YELLOW / RED constants below are **absolute pain levels** copied
from BASELINE.md §4.1 (commit a947303 era). They are NOT statistically
grounded against a weekly walk-forward Sharpe distribution.

Specifically:
- Cross-seed σ from β0 spike (round 36, commit b73834a) = 0.13, but
  this measures **seed lottery**, not weekly walk-forward drift
- **Weekly walk-forward time-series σ has never been measured**.
  Production runs deterministic ``LGBM_SEED=42``; weekly drift comes
  from data-window shifting only (one more week of training data per
  weekly cron + any new factor-cache invalidations)
- Therefore framing "RED Sharpe 0.9 = -7σ from cross-seed mean" is a
  type error — the right σ for tuning these thresholds doesn't exist
  in this repo yet

Known limitations of the current calibration:
- RED Sharpe < 0.9 may rarely trigger if true weekly σ is small —
  Sharpe halving is a catastrophic regime change, not a typical
  weekly fluctuation
- YELLOW Sharpe < 1.4 may be too lax if true weekly σ is small enough
  that real degradation manifests as Δ ~ -0.2 Sharpe, not Δ ~ -0.5
- Proper grounding requires a "P5-A-mid" follow-up (currently NOT
  scheduled): rerun weekly walk-forward N weeks back, measure
  time-series σ, then re-derive YELLOW/RED bands as e.g. (mean −kσ).
  Cost ~4-6 hr per N=8-12 weeks. Deferred to a separate research chain
  per advisor round 41.

For now: treat the alerts here as **"absolute pain level" gates**, not
"statistically significant departure" gates. Manual review of weekly
``data/reports/walk_forward_result.md`` remains the primary monitoring
path; the Feishu auto-alert is a backstop for severe breaches, not the
sole detector of model drift.
"""
from __future__ import annotations

from typing import Optional


# YELLOW alert threshold (≈ 50% degradation from BASELINE)
YELLOW = {
    "sharpe_ratio":        1.40,
    "annual_return_pct":   30.0,
    "max_drawdown_pct":   -42.0,    # less negative than this = pass (i.e. > -42% is healthy)
}

# RED alert threshold (immediate paper-trade halt per BASELINE §4)
RED = {
    "sharpe_ratio":        0.90,
    "annual_return_pct":   15.0,
    "max_drawdown_pct":   -50.0,
}


def _to_pct(v: Optional[float]) -> Optional[float]:
    """Heuristic: accept either fraction (0.60) or percent (60.0)."""
    if v is None:
        return None
    return float(v) if abs(v) > 1.0 else float(v) * 100.0


def check_thresholds(bt_metrics: dict) -> list[dict]:
    """Return list of breach alert dicts; empty if all healthy.

    Each alert::

        {"level": "RED" | "YELLOW",
         "indicator": str,   # human label
         "actual": float,
         "threshold": float,
         "comparator": "<" | ">",
         "msg": str}         # one-line human summary
    """
    alerts: list[dict] = []

    # Sharpe (lower = worse)
    sharpe = bt_metrics.get("sharpe_ratio")
    if sharpe is not None:
        sharpe = float(sharpe)
        if sharpe < RED["sharpe_ratio"]:
            alerts.append({"level": "RED", "indicator": "Sharpe",
                           "actual": sharpe, "threshold": RED["sharpe_ratio"],
                           "comparator": "<",
                           "msg": f"Sharpe {sharpe:.2f} < {RED['sharpe_ratio']:.1f} (RED: 即停模拟交易)"})
        elif sharpe < YELLOW["sharpe_ratio"]:
            alerts.append({"level": "YELLOW", "indicator": "Sharpe",
                           "actual": sharpe, "threshold": YELLOW["sharpe_ratio"],
                           "comparator": "<",
                           "msg": f"Sharpe {sharpe:.2f} < {YELLOW['sharpe_ratio']:.1f} (黄色)"})

    # Annual return (lower = worse)
    annual_pct = _to_pct(bt_metrics.get("annual_return"))
    if annual_pct is not None:
        if annual_pct < RED["annual_return_pct"]:
            alerts.append({"level": "RED", "indicator": "annual_return",
                           "actual": annual_pct, "threshold": RED["annual_return_pct"],
                           "comparator": "<",
                           "msg": f"年化 {annual_pct:.1f}% < {RED['annual_return_pct']:.0f}% (RED)"})
        elif annual_pct < YELLOW["annual_return_pct"]:
            alerts.append({"level": "YELLOW", "indicator": "annual_return",
                           "actual": annual_pct, "threshold": YELLOW["annual_return_pct"],
                           "comparator": "<",
                           "msg": f"年化 {annual_pct:.1f}% < {YELLOW['annual_return_pct']:.0f}% (黄色)"})

    # Max drawdown (more negative = worse)
    dd_pct = _to_pct(bt_metrics.get("max_drawdown"))
    if dd_pct is not None:
        if dd_pct < RED["max_drawdown_pct"]:
            alerts.append({"level": "RED", "indicator": "max_drawdown",
                           "actual": dd_pct, "threshold": RED["max_drawdown_pct"],
                           "comparator": "<",
                           "msg": f"Max DD {dd_pct:.1f}% < {RED['max_drawdown_pct']:.0f}% (RED)"})
        elif dd_pct < YELLOW["max_drawdown_pct"]:
            alerts.append({"level": "YELLOW", "indicator": "max_drawdown",
                           "actual": dd_pct, "threshold": YELLOW["max_drawdown_pct"],
                           "comparator": "<",
                           "msg": f"Max DD {dd_pct:.1f}% < {YELLOW['max_drawdown_pct']:.0f}% (黄色)"})

    return alerts


def format_for_feishu(alerts: list[dict]) -> str:
    """Render alert list as a markdown block for the Feishu message.

    Returns empty string when ``alerts`` is empty (caller can omit the
    block entirely in that case).
    """
    if not alerts:
        return ""
    red = [a for a in alerts if a["level"] == "RED"]
    yel = [a for a in alerts if a["level"] == "YELLOW"]
    parts = []
    if red:
        parts.append("## 🚨 RED ALERT — 即停模拟交易")
        for a in red:
            parts.append(f"- {a['msg']}")
    if yel:
        parts.append("## ⚠ YELLOW ALERT")
        for a in yel:
            parts.append(f"- {a['msg']}")
    parts.append("")
    parts.append(
        f"thresholds source-of-truth: `mp/monitor/threshold_alert.py` "
        f"(mirrors BASELINE.md §4.1; see docs/dialog/ round 39 for design)"
    )
    return "\n".join(parts)
