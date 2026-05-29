"""P11 Phase 3a — real-14:30-fill + per-stock realistic-cost analysis.

Round 131/132: with the real 14:30–15:00 execution bars (data/intraday_1430_exec/,
fetched on ECS via p11_4_fetch_intraday.py --exec-window) now local, estimate
how far Arm B's headline (2.12 3-seed mean, already after sqrt-impact+commission+
stamp) drops under (a) a REAL 14:30 fill and (b) per-stock realistic transaction
cost — then map that cost onto the round-127 flat-bps sweep.

No real money, no production touch (Rule #4). Reads only research parquets.

Findings (8mo window 2025-09~2026-04, ZZ500∪hs300 minus 创业板/科创板):
  [A] 14:30 open vs 14:29 close: median |diff| ≈ 0 bps → fill-price swap is
      negligible vs the ~15bps breakeven; the headline is driven by COST, not
      the 1-bar price change.
  [B] one-tick half-spread (0.5·0.01/price): median ≈ 2.9 bps (price median ~17).
  [C] Corwin-Schultz empirical half-spread (1m 14:30–15:00): median ≈ 3.1 bps —
      agrees with [B] → these names trade at ~1-tick spread.
  [D] realistic one-way ≈ half-spread (~3bps) + sqrt-impact (~1-7bps @100k AUM)
      ≈ ~4-10 bps → flat-bps sweep ⇒ Arm B Sharpe ~1.9 (5bps) .. ~1.4 (10bps),
      both > baseline A (1.00) and > market (25.8% / Sharpe ~0.8).

Gate (round 129): 3a PASS (Arm B still ≫ baseline at realistic cost). 3b: median
one-way ~5-6bps < 10bps gate (PASS); p90 illiquid/high-price tail ~10-17bps
(marginal → needs a liquidity filter). Caveats: small-AUM only (impact scales
with size), 8mo window, spread is a 1m-OHLC proxy (real L1/L2 not yet pulled).
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

EXEC_DIR = "data/intraday_1430_exec"
MORNING_DIR = "data/intraday_1m"
# round-127 flat-bps sweep (3-seed mean Sharpe, Arm B inject+14:30):
SWEEP = {0: 2.40, 5: 1.90, 10: 1.41, 20: 0.64, 40: -0.36}
BASELINE_A = 1.00   # 3-seed mean, no-inject T+1-open
MARKET_ANNUAL = 25.8


def _load(dir_: str) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{dir_}/*.parquet")],
                   ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.normalize()
    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def _corwin_schultz_halfspread_bps(g: pd.DataFrame) -> float:
    """Corwin-Schultz proportional spread from consecutive 1m high/low; return
    median HALF-spread in bps (negative estimates dropped)."""
    g = g.sort_values("datetime")
    h, l = g["high"].to_numpy(float), g["low"].to_numpy(float)
    if len(h) < 2:
        return np.nan
    h2 = np.maximum(h[:-1], h[1:])
    l2 = np.minimum(l[:-1], l[1:])
    with np.errstate(all="ignore"):
        beta = np.log(h[:-1] / l[:-1]) ** 2 + np.log(h[1:] / l[1:]) ** 2
        gamma = np.log(h2 / l2) ** 2
        k = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = s[np.isfinite(s)]
    s = s[s > 0]
    return float(np.median(s) * 1e4 / 2) if len(s) else np.nan


def _sweep_sharpe(bps: float) -> float:
    """Linear-interpolate Arm B net Sharpe at a flat one-way cost (bps)."""
    xs = sorted(SWEEP)
    if bps <= xs[0]:
        return SWEEP[xs[0]]
    if bps >= xs[-1]:
        return SWEEP[xs[-1]]
    for a, b in zip(xs, xs[1:]):
        if a <= bps <= b:
            return SWEEP[a] + (SWEEP[b] - SWEEP[a]) * (bps - a) / (b - a)
    return float("nan")


def main() -> None:
    ex = _load(EXEC_DIR)
    mo = _load(MORNING_DIR)

    o1430 = ex[ex["datetime"].dt.time == pd.Timestamp("14:30").time()].set_index(["code", "date"])["open"]
    c1429 = mo.sort_values("datetime").groupby(["code", "date"]).tail(1).set_index(["code", "date"])["close"]
    j = pd.concat([o1430.rename("o"), c1429.rename("c")], axis=1).dropna()
    j = j[j["c"] > 0]
    rel = (j["o"] - j["c"]).abs() / j["c"] * 1e4
    print(f"[A] 14:30 open vs 14:29 close ({len(j)} pairs): "
          f"|diff| median={rel.median():.2f}bps mean={rel.mean():.2f}bps p90={rel.quantile(.9):.2f}bps")

    px = o1430.dropna()
    px = px[px > 0]
    half_tick = 0.5 * 0.01 / px * 1e4
    print(f"[B] one-tick half-spread: median={half_tick.median():.2f}bps mean={half_tick.mean():.2f}bps "
          f"p90={half_tick.quantile(.9):.2f}bps  (price median={px.median():.2f})")

    cs = ex.groupby(["code", "date"]).apply(_corwin_schultz_halfspread_bps, include_groups=False).dropna()
    print(f"[C] Corwin-Schultz half-spread: median={cs.median():.2f}bps mean={cs.mean():.2f}bps p90={cs.quantile(.9):.2f}bps")

    for c in (5, 6, 10):
        print(f"[D] Arm B net Sharpe @ {c}bps one-way (sweep): {_sweep_sharpe(c):.2f}  (baseline A={BASELINE_A})")


if __name__ == "__main__":
    main()
