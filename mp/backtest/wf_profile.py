"""Walk-forward backtest profiles + pure strategy-rule helpers (no side effects).

Rule (user, 2026-09-23): every backtest must run under the PRODUCTION strategy
口径 by default.  ``WF_PROFILE=prod`` (default) mirrors what
``scripts/daily_report.py`` actually does; ``WF_PROFILE=research`` restores the
historical walk-forward defaults so older experiments stay reproducible.

A profile only changes *defaults*: any knob set explicitly in the environment
wins over the profile value.

Production rules mirrored (verified against scripts/daily_report.py 2026-09-23):

  recommend_stocks(n_recommend=22)      → TOP_K=22 conviction targets.  The
      "25 holds" in the round-245 comment is EMPIRICAL (n=22 → ~25 holds):
      22 sized names + old holdings ranked 23..30 that are silently kept.
  688/689/300/301 dropped               → EXCL_CHINEXT=1
  HS300 included (2026-07-17 fix)       → EXCL_HS300=0
  LOW_LIQUIDITY_FILTER_AMOUNT = 1e8     → EXCL_ILLIQ=1, LIVE_ILLIQ_AMOUNT=1e8
  target_position_pct (portfolio.yaml)  → GROSS_EXPOSURE=0.85
  hard_single_weight default 0.40       → HARD_CAP=0.40 (min(investable×w, 0.40×NAV);
      truncation WITHOUT redistribution — so ``conviction_softcap`` is NOT
      equivalent and is not used)
  rebalance_tolerance default 0.02      → REBALANCE_TOLERANCE=0.02 (skip
      |delta| < 2% NAV), evaluated EVERY day (ORDER_PASS=prod)
  held & not in top-N: rank ≤ 30 keep; 30 < rank ≤ 100 sell half (lot-rounded,
      repeated daily); rank > 100 clear
                                        → HOLD_RANK_BAND=30, CLEAR_RANK_BAND=100
  straight top-N (no cost-aware swap)   → COST_AWARE_REBALANCE=0
  EOD plan → 9:25 next-day execution    → ENTRY_TIME=t_plus_1_open (already default)
  weekly gated auto-retrain             → RETRAIN_FREQ=weekly (first trading day
      of each ISO week; the verify gate cannot be reproduced offline)

Anything the framework cannot mirror is listed in :data:`UNALIGNED_ITEMS` and
printed in the report header (never silent).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

PROFILE_ENV = "WF_PROFILE"
PROFILES = ("prod", "research")
DEFAULT_PROFILE = "prod"


def _bool(v: str) -> bool:
    return str(v).strip() == "1"


def _freq(v: str) -> str:
    v = str(v).strip().lower()
    if v not in ("weekly", "monthly"):
        raise ValueError(f"RETRAIN_FREQ must be weekly|monthly, got {v!r}")
    return v


def _order_pass(v: str) -> str:
    v = str(v).strip().lower()
    if v not in ("prod", "legacy"):
        raise ValueError(f"ORDER_PASS must be prod|legacy, got {v!r}")
    return v


# name → (parser, prod default, research default)
# research defaults == the walk_forward_backtest.py defaults before profiles.
KNOB_SPECS: Dict[str, tuple] = {
    "TOP_K":                (int,   22,        10),
    "LIVE_UNIVERSE":        (_bool, True,      False),
    "EXCL_CHINEXT":         (_bool, True,      True),
    "EXCL_HS300":           (_bool, False,     True),
    "EXCL_ILLIQ":           (_bool, True,      True),
    "LIVE_ILLIQ_AMOUNT":    (float, 1e8,       1e8),
    "GROSS_EXPOSURE":       (float, 0.85,      1.0),
    "LIMIT_LOCK":           (_bool, True,      True),
    "POSITION_SIZING":      (str,   "conviction", "conviction"),
    "HARD_CAP":             (float, 0.40,      0.0),     # 0 = off
    "HOLD_RANK_BAND":       (int,   30,        0),       # 0 = off (legacy: sell on drop-out)
    "CLEAR_RANK_BAND":      (int,   100,       0),       # only used when HOLD_RANK_BAND > 0
    "REBALANCE_TOLERANCE":  (float, 0.02,      0.0),     # fraction of NAV; only in ORDER_PASS=prod
    "ORDER_PASS":           (_order_pass, "prod", "legacy"),
    "COST_AWARE_REBALANCE": (_bool, False,     True),
    "RETRAIN_FREQ":         (_freq, "weekly",  "monthly"),
}

# Production behaviours the framework does NOT reproduce under WF_PROFILE=prod.
UNALIGNED_ITEMS: List[str] = [
    "模型重训: 生产为周度 auto_retrain + verify gate(不过门不换模型); 框架每个交易周第一天无条件重训, 离线不可复现门控结果",
    "限价成交假设: 生产挂 close×1.01 买 / close×0.99 卖 的限价单(高开>1% 不成交); 框架按 T+1 开盘价 + 滑点/冲击成本 100% 成交(仅 LIMIT_LOCK 拦一字板/停牌)",
    "现金约束: 生产买单 ≤ 0.95×(可用现金+卖出回款) 按比例缩量再整手取整; 框架同样按比例缩量, 但价格用开盘价而非限价, 且不含 '1 手 ≤ 2×缺口 则买 1 手' 的小账户规则",
    "生产对不在打分池(rank=None)的持仓静默保留; 框架按清仓处理, 避免出池僵尸持仓",
    "生产 26-30 名(不在 top-22)静默保留是基于当日 full_scored 排名; 框架排名口径同为 live 过滤前的全池排名, 但打分池是 PIT 成分快照而非当日实时成分",
    "生产的 conviction 权重来自 predicted_excess(pp, 2 位小数四舍五入); 框架用未取整的 predict_raw, 权重存在 ≤0.005pp 量级差异",
    "生产 alert-only 场景(1 手 > 2×缺口 且 超额≥3% → 只提示不下单)在框架中同样不下单, 但不产生提示统计",
]


def resolve(env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Return {"profile": str, "explicit": set[str], <KNOB>: value, ...}.

    Profile only sets defaults; explicit env values override.  Pure function
    of ``env`` (defaults to ``os.environ``) — no I/O.
    """
    env = os.environ if env is None else env
    profile = str(env.get(PROFILE_ENV, DEFAULT_PROFILE)).strip().lower()
    if profile not in PROFILES:
        raise ValueError(f"{PROFILE_ENV} must be one of {PROFILES}, got {profile!r}")
    out: Dict[str, Any] = {"profile": profile, "explicit": set()}
    for name, (parser, prod_default, research_default) in KNOB_SPECS.items():
        if name in env and str(env[name]) != "":
            out[name] = parser(env[name])
            out["explicit"].add(name)
        else:
            out[name] = prod_default if profile == "prod" else research_default
    return out


def knob_names() -> List[str]:
    return list(KNOB_SPECS.keys())


def profile_line(p: Mapping[str, Any]) -> str:
    """One-line human description for the report header / startup log."""
    prof = p["profile"]
    if prof == "prod":
        desc = (
            f"prod ({p['TOP_K']} recs → ~25 holds, live filters"
            f"[chinext={int(p['EXCL_CHINEXT'])} hs300={int(p['EXCL_HS300'])} "
            f"illiq={int(p['EXCL_ILLIQ'])}@{p['LIVE_ILLIQ_AMOUNT']:.0e}], "
            f"cap {p['HARD_CAP']:.0%}, exposure {p['GROSS_EXPOSURE']:.0%}, "
            f"hold band {p['HOLD_RANK_BAND']}/clear {p['CLEAR_RANK_BAND']}, "
            f"tol {p['REBALANCE_TOLERANCE']:.0%}, {p['RETRAIN_FREQ']} retrain; "
            f"生产周度重训带 verify gate, 离线不可复现)"
        )
    else:
        desc = (
            f"research (legacy defaults: top-{p['TOP_K']}, live_universe={int(p['LIVE_UNIVERSE'])}, "
            f"sizing={p['POSITION_SIZING']}, exposure {p['GROSS_EXPOSURE']:.0%}, "
            f"no hold band, cost-aware swap={int(p['COST_AWARE_REBALANCE'])}, "
            f"{p['RETRAIN_FREQ']} retrain)"
        )
    if p.get("explicit"):
        desc += f" | env overrides: {', '.join(sorted(p['explicit']))}"
    return desc


# ──────────────────────────────────────────────────────────────────────
# Pure strategy-rule helpers (mirror daily_report.generate_order_list)
# ──────────────────────────────────────────────────────────────────────

def hold_band_action(rank: Optional[int], hold_band: int, clear_band: int) -> str:
    """Decision for a HELD name that is NOT in today's top-K selection.

    Mirrors daily_report Pass 2:
      rank ≤ hold_band            → "hold"  (silent hold)
      hold_band < rank ≤ clear    → "half"  (减半仓, lot-rounded)
      rank > clear / unknown      → "sell"  (清仓)
    hold_band ≤ 0 disables the band (legacy: always sell on drop-out).
    """
    if hold_band <= 0:
        return "sell"
    if rank is None:
        return "sell"   # unaligned by design: prod holds silently (see UNALIGNED_ITEMS)
    if rank <= hold_band:
        return "hold"
    if clear_band > hold_band and rank <= clear_band:
        return "half"
    return "sell"


def holding_decision(is_held: bool, rank: Optional[int], top_k: int,
                     hold_band: int, clear_band: int) -> str:
    """Full per-name decision.

      "target" → in top-K: sized to conviction target (buy / top-up / trim)
      "none"   → not held and not in top-K: never bought
      "hold" / "half" / "sell" → held & outside top-K (see hold_band_action)
    """
    if rank is not None and rank <= top_k:
        return "target"
    if not is_held:
        return "none"
    return hold_band_action(rank, hold_band, clear_band)


def half_lot_shares(shares: int, lot: int = 100) -> int:
    """Prod 减半仓: (shares // 2 // lot) * lot; 0 means 'skip' (< 1 lot)."""
    return (int(shares) // 2 // lot) * lot


def retrain_dates(dates: Iterable, freq: str) -> List[pd.Timestamp]:
    """First trading day of each period in ``dates`` (any iterable of dates).

    freq="monthly" → first trading day per calendar month (legacy).
    freq="weekly"  → first trading day per ISO week (Mon..Sun).
    """
    freq = _freq(freq)
    s = pd.Series(sorted({pd.Timestamp(d) for d in dates}))
    if s.empty:
        return []
    key = s.dt.to_period("M" if freq == "monthly" else "W")
    return s.groupby(key).first().tolist()


def scale_buys_to_caps(buy_deltas: Mapping[str, float], projected_total: float,
                       position_cap: float, cash_available: float,
                       cash_buffer: float = 0.95) -> Dict[str, float]:
    """Mirror daily_report budget + total-position reconciliation (sequential).

    1. buys ≤ cash_buffer × cash_available  → proportional scale-down
    2. projected_total (positions after all orders) ≤ position_cap
       → scale remaining buys so the total lands on the cap
    Returns {code: scaled_delta}; entries scaled to ≤ 0 are dropped.
    """
    if not buy_deltas:
        return {}
    buys = {c: float(v) for c, v in buy_deltas.items() if v > 0}
    total = sum(buys.values())
    if total <= 0:
        return {}
    cash_cap = max(0.0, cash_available) * cash_buffer
    if total > cash_cap:
        s1 = cash_cap / total
        buys = {c: v * s1 for c, v in buys.items()}
        projected_total = projected_total - total + total * s1
        total = total * s1
    if position_cap > 0 and projected_total > position_cap and total > 0:
        excess = projected_total - position_cap
        s2 = max(0.0, total - excess) / total
        buys = {c: v * s2 for c, v in buys.items()}
    return {c: v for c, v in buys.items() if v > 0}


# Module-level snapshot (importlib.reload picks up a patched environment).
CURRENT = resolve()
PROFILE = CURRENT["profile"]
