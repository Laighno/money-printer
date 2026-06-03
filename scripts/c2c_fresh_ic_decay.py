"""IC decay analysis for c2c FRESH model.

For each trading date D in the WF test window (2025-09-01 → 2026-04-28):
  1. Score all codes using c2c FRESH on factor[D-1] panel
  2. Compute realized forward returns at horizons 1, 5, 10, 20 days
  3. Cross-sectional Spearman IC per D × per horizon

Aggregate distribution → mean / std / pct of positive IC days.

Tells us: is c2c FRESH's alpha really 20-day, or front-loaded in 1-3 days?
If 1-day IC > 5-day IC > 20-day IC, then dual mode's 18h capture is the
sweet spot — model nominally 20d but signal decays fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mp.ml.model import BlendRanker  # noqa: E402

WINDOW_START = "2025-09-01"
WINDOW_END = "2026-04-28"
HORIZONS = [1, 3, 5, 10, 20]


def main() -> int:
    logger.info("Loading c2c FRESH model")
    # 现在 intraday_blend_*.lgb == c2c FRESH (round 207 swap)
    ranker = BlendRanker()
    ok = ranker.load("data/blend_c2c_FRESH")
    assert ok, "load blend_c2c_FRESH failed"
    feature_cols = ranker.primary.feature_cols
    logger.info(f"Model: {len(feature_cols)} features, primary trees={ranker.primary.model.num_trees()}")

    logger.info("Loading c2c factor cache (Apr 28 build)")
    factors = pd.read_parquet("data/wf_cache/factors.parquet")
    factors["date"] = pd.to_datetime(factors["date"])
    factors["code"] = factors["code"].astype(str).str.zfill(6)

    logger.info("Loading bars for forward returns")
    bars = pd.read_parquet("data/wf_cache/bars.parquet")
    bars["date"] = pd.to_datetime(bars["date"])
    bars["code"] = bars["code"].astype(str).str.zfill(6)
    bars = bars[["date", "code", "close"]].sort_values(["code", "date"])

    # Build close panel keyed by (code, date)
    close_map = bars.set_index(["code", "date"])["close"].to_dict()

    # Get all trading dates in test window from bars
    start = pd.Timestamp(WINDOW_START)
    end = pd.Timestamp(WINDOW_END)
    trading_dates = sorted(bars["date"].unique())
    trading_dates = [d for d in trading_dates if start <= d <= end]
    logger.info(f"Test dates: {len(trading_dates)} ({trading_dates[0]} → {trading_dates[-1]})")

    # Pre-build factor-by-date dict for fast lookup
    factor_by_date = {d: g for d, g in factors.groupby("date")}

    ic_records = []  # one row per (date, horizon)

    for i, D in enumerate(trading_dates):
        # Pick factor row at D-1
        prev_dates_all = [d for d in factor_by_date.keys() if d < D]
        if not prev_dates_all:
            continue
        d_minus_1 = max(prev_dates_all)
        panel = factor_by_date[d_minus_1].copy()
        if panel.empty:
            continue

        # Score
        for c in feature_cols:
            if c not in panel.columns:
                panel[c] = 0.0
        sub = panel.dropna(subset=feature_cols).copy()
        if sub.empty:
            continue
        if "fwd_ret" not in sub.columns:
            sub["fwd_ret"] = 0.0
        if "excess_ret" not in sub.columns:
            sub["excess_ret"] = 0.0

        try:
            sub["score"] = ranker.predict(sub)
        except Exception as e:
            logger.warning(f"{D}: predict failed: {e}")
            continue

        # Compute realized forward returns at multiple horizons
        # We use anchor = D's close (consistent with c2c label: anchor at close[D], future close[D+H])
        # For walk-forward 18h scalp, the anchor is close[D-1] → open[D] → close[D+1] etc., but for
        # apples-to-apples with the model's training (c2c at close[D-1]), we'll anchor at close[D-1]
        # and measure H days forward (since model was trained on c2c starting at close[i]).
        # Actually, the model's input is factor[D-1], so prediction is for "starting from close[D-1]".
        # We compute forward returns starting from close[D-1].
        anchor_date = d_minus_1

        for H in HORIZONS:
            # Find date at trading-day index + H from anchor_date
            future_dates = [d for d in trading_dates if d > anchor_date]
            if len(future_dates) < H:
                continue
            target_date = future_dates[H - 1]  # H trading days forward

            # Build return vector
            rets = []
            for _, row in sub.iterrows():
                c = str(row["code"]).zfill(6)
                p_now = close_map.get((c, anchor_date))
                p_fwd = close_map.get((c, target_date))
                if p_now is None or p_fwd is None or p_now <= 0:
                    rets.append(np.nan)
                else:
                    rets.append(p_fwd / p_now - 1.0)
            sub_with_ret = sub.copy()
            sub_with_ret["fwd_realized"] = rets
            sub_clean = sub_with_ret.dropna(subset=["score", "fwd_realized"])
            if len(sub_clean) < 30:
                continue

            ic, _ = spearmanr(sub_clean["score"], sub_clean["fwd_realized"])
            ic_records.append({"date": D, "horizon": H, "ic": ic, "n": len(sub_clean)})

        if (i + 1) % 20 == 0:
            logger.info(f"Progress {i+1}/{len(trading_dates)}")

    df = pd.DataFrame(ic_records)
    if df.empty:
        logger.error("No IC records!")
        return 1

    print()
    print("=" * 70)
    print("c2c FRESH IC decay (test window 2025-09-01 → 2026-04-28)")
    print("=" * 70)
    print(f"{'Horizon':<10}{'mean IC':>10}{'std IC':>10}{'ICIR':>10}{'%>0':>10}{'#dates':>10}")
    print("-" * 70)
    rows = []
    for H in HORIZONS:
        sub = df[df.horizon == H]
        if sub.empty:
            continue
        mean = sub.ic.mean()
        std = sub.ic.std()
        icir = mean / std if std > 0 else 0.0
        pct_pos = (sub.ic > 0).mean() * 100
        n = len(sub)
        rows.append({"H": H, "mean": mean, "std": std, "icir": icir, "pct_pos": pct_pos, "n": n})
        print(f"{H:<10}{mean:>10.4f}{std:>10.4f}{icir:>10.3f}{pct_pos:>10.1f}{n:>10d}")
    print("=" * 70)

    out_path = Path("data/reports/c2c_fresh_ic_decay.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# c2c FRESH IC decay\n\n")
        f.write(f"Test window: {WINDOW_START} → {WINDOW_END}, {len(trading_dates)} dates\n")
        f.write(f"Anchor: close[D-1] (matches model training c2c label anchor)\n\n")
        f.write(f"| Horizon (trading days) | Mean IC | Std IC | ICIR | %>0 | #dates |\n")
        f.write(f"|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['H']} | {r['mean']:.4f} | {r['std']:.4f} | {r['icir']:.3f} | "
                    f"{r['pct_pos']:.1f}% | {r['n']} |\n")
    logger.info(f"Report → {out_path}")
    json_path = Path("data/c2c_fresh_ic_decay.json")
    df.to_json(json_path, orient="records", date_format="iso")
    logger.info(f"Raw IC → {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
