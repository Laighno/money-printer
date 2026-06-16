"""Verify a freshly-trained BlendRanker against the current production model
before any swap. Round 274-275 (advisor, auto-retrain pipeline stage 2; user B).

WHY: the naive "primary IC ≥ 0.07 absolute floor" gate (advisor round 266) is
REGIME-DEPENDENT and FALSELY ABORTS healthy models. Concretely (round 274):
  - new model trained to cutoff 5/18 reported train_fast IC=0.000
  - but it is genuinely healthy: in-sample IC 0.14-0.21 on 2024/2025H1/2026Q1,
    and per-date IC corr(new, prod) = 0.957 on the recent window
  - the 0.000 is an artifact: train_fast's val split landed on the recent
    ADVERSE regime (2026-04-01~05-18, where prod itself scores IC -0.147)

So the gate is RELATIVE + non-degeneracy + regime-tracking, not an absolute
floor (advisor round 275 ACK):

  Gate 1 — non-degeneracy: new-model in-sample IC on a FAVORABLE, label-complete
           window (default 2025-H1, where healthy models score ~0.20) ≥ 0.10.
           Catches a truly broken model (e.g. the build_dataset 0.001 trap:
           its historical-window IC also collapses to ~0).
  Gate 2 — relative not-worse: new OOS IC ≥ prod OOS IC − margin (default 0.02)
           on the most recent labeled holdout. Regime-robust because both
           models see the SAME (possibly adverse) regime.
  Gate 3 — regime tracking: per-date IC corr(new, prod) ≥ 0.8 on the holdout.
           A degenerate/random model cannot track prod's daily IC; a real model
           that learned the same factor→return structure does.

Plus advisor round 266 pick-sanity (latest-date top-K):
  - top-K overlap(new, prod) ≥ min (sudden total churn = alarm)
  - no single industry > cap in new picks (degenerate sector pile-up)
  - no NaN / all-identical scores

IMPORTANT (advisor round 275 caveat): because the n2c label has a 20-day lag,
there is almost no clean post-cutoff OOS window — the "in-sample IC 0.14-0.21"
numbers are IN-SAMPLE and CANNOT prove the new model is *better*, only that it
is *non-degenerate, same-structure, and at-least-not-worse*. The swap rationale
is "non-degenerate + fresher + low-risk (high corr)", never "better".

This script does NOT swap — it only reports a verdict for human review.
swap_model.py (separate) does the swap after advisor + user confirm.

Usage:
  python -m scripts.verify_retrain_quality \\
      --new-prefix data/blend_new_20260518 \\
      --prod-prefix data/blend \\
      [--json-out data/retrain_verify_<date>.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LABEL_KIND", "next_open_to_close")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "wf_cache" / "factors_label_next_open_to_close.parquet"

# Gate thresholds (advisor round 275).
GATE1_NONDEGEN_IC = 0.10        # new in-sample IC on favorable window must clear this
GATE2_REL_MARGIN = 0.02         # new OOS IC ≥ prod OOS IC − margin
GATE3_REGIME_CORR = 0.80        # per-date IC corr(new, prod)
FAVORABLE_START = "2025-01-01"  # label-complete + favorable regime window for gate 1
FAVORABLE_END = "2025-06-30"
HOLDOUT_DATES = 20              # most recent N labeled dates = relative holdout
PICK_TOPK = 25
PICK_MIN_OVERLAP = 0.40         # < this fraction overlap = churn alarm
PICK_MAX_INDUSTRY = 0.40        # single-industry share cap in new picks


def _per_date_ic(scores: np.ndarray, y: np.ndarray, dates: pd.Series) -> pd.Series:
    df = pd.DataFrame({"date": dates.values, "s": scores, "y": y})
    return df.groupby("date").apply(
        lambda g: spearmanr(g["s"], g["y"])[0] if len(g) > 5 else np.nan
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-prefix", required=True,
                    help="new model prefix, e.g. data/blend_new_20260518")
    ap.add_argument("--prod-prefix", default="data/blend",
                    help="production model prefix (default: data/blend)")
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from mp.ml.model import BlendRanker

    cache = pd.read_parquet(args.cache)
    cache["date"] = pd.to_datetime(cache["date"])

    new = BlendRanker()
    if not new.load(args.new_prefix):
        logger.error("Failed to load new model: {}", args.new_prefix)
        return 2
    prod = BlendRanker()
    if not prod.load(args.prod_prefix):
        logger.error("Failed to load prod model: {}", args.prod_prefix)
        return 2

    verdict: dict = {"new_prefix": args.new_prefix, "prod_prefix": args.prod_prefix,
                     "gates": {}, "pick_sanity": {}, "abort": False, "reasons": []}

    # ---- Gate 1: non-degeneracy (new in-sample IC on favorable window) ----
    fav = cache[(cache["date"] >= FAVORABLE_START) & (cache["date"] <= FAVORABLE_END)] \
        .dropna(subset=["fwd_ret"]).copy()
    new_fav_ic = float(np.nanmean(_per_date_ic(
        new.predict(fav), fav["fwd_ret"].to_numpy(), fav["date"])))
    g1_pass = new_fav_ic >= GATE1_NONDEGEN_IC
    verdict["gates"]["gate1_nondegeneracy"] = {
        "window": f"{FAVORABLE_START}..{FAVORABLE_END}", "new_in_sample_ic": new_fav_ic,
        "threshold": GATE1_NONDEGEN_IC, "pass": g1_pass,
        "note": "IN-SAMPLE (≤cutoff); proves non-degenerate, NOT 'better'",
    }
    if not g1_pass:
        verdict["abort"] = True
        verdict["reasons"].append(
            f"gate1 FAIL: new in-sample IC {new_fav_ic:.4f} < {GATE1_NONDEGEN_IC} "
            "(model likely degenerate — build_dataset-style collapse)")

    # ---- recent labeled holdout (last N labeled dates) ----
    labeled = cache.dropna(subset=["fwd_ret"])
    hold_dates = sorted(labeled["date"].unique())[-HOLDOUT_DATES:]
    hold = labeled[labeled["date"].isin(hold_dates)].copy()
    new_ic_series = _per_date_ic(new.predict(hold), hold["fwd_ret"].to_numpy(), hold["date"])
    prod_ic_series = _per_date_ic(prod.predict(hold), hold["fwd_ret"].to_numpy(), hold["date"])
    new_oos_ic = float(np.nanmean(new_ic_series))
    prod_oos_ic = float(np.nanmean(prod_ic_series))

    # ---- Gate 2: relative not-worse ----
    g2_pass = new_oos_ic >= prod_oos_ic - GATE2_REL_MARGIN
    verdict["gates"]["gate2_relative"] = {
        "holdout": f"{pd.Timestamp(hold_dates[0]).date()}..{pd.Timestamp(hold_dates[-1]).date()}",
        "new_oos_ic": new_oos_ic, "prod_oos_ic": prod_oos_ic,
        "margin": GATE2_REL_MARGIN, "pass": g2_pass,
        "note": "regime-robust: both models see same (maybe adverse) regime",
    }
    if not g2_pass:
        verdict["abort"] = True
        verdict["reasons"].append(
            f"gate2 FAIL: new OOS IC {new_oos_ic:.4f} < prod {prod_oos_ic:.4f} "
            f"− {GATE2_REL_MARGIN} (meaningfully worse than prod)")

    # ---- Gate 3: regime tracking ----
    both = pd.DataFrame({"n": new_ic_series, "p": prod_ic_series}).dropna()
    regime_corr = float(spearmanr(both["n"], both["p"])[0]) if len(both) > 3 else float("nan")
    g3_pass = (not np.isnan(regime_corr)) and regime_corr >= GATE3_REGIME_CORR
    verdict["gates"]["gate3_regime_tracking"] = {
        "per_date_ic_corr": regime_corr, "threshold": GATE3_REGIME_CORR, "pass": g3_pass,
        "note": "degenerate/random model cannot track prod's daily IC",
    }
    if not g3_pass:
        verdict["abort"] = True
        verdict["reasons"].append(
            f"gate3 FAIL: corr(new,prod) {regime_corr:.3f} < {GATE3_REGIME_CORR} "
            "(new not tracking prod regime — suspect)")

    # ---- pick-sanity (latest-date top-K) ----
    latest = cache["date"].max()
    last = cache[cache["date"] == latest].copy()
    last = last[~last["code"].isna()]
    ns = new.predict(last)
    ps = prod.predict(last)
    nan_scores = int(np.isnan(ns).sum())
    degenerate = bool(np.nanstd(ns) < 1e-9)
    last = last.assign(ns=ns, ps=ps)
    new_top = set(last.nlargest(PICK_TOPK, "ns")["code"].astype(str).str.zfill(6))
    prod_top = set(last.nlargest(PICK_TOPK, "ps")["code"].astype(str).str.zfill(6))
    overlap = len(new_top & prod_top) / PICK_TOPK

    industry_share = None
    try:
        from mp.data.fetcher import get_industry_mapping
        from collections import Counter
        imap = get_industry_mapping(list(new_top))
        # Concentration only over MAPPED picks — unmapped ('?') is a coverage
        # gap, not a sector pile-up (round 275 self-fix: don't false-alarm on
        # missing industry data).
        mapped = {c: imap[c] for c in new_top if c in imap and imap[c]}
        coverage = len(mapped) / PICK_TOPK
        if mapped:
            cnt = Counter(mapped.values())
            top_ind, top_n = cnt.most_common(1)[0]
            industry_share = {"top_industry": top_ind,
                              "share": top_n / len(mapped),
                              "coverage": coverage}
        else:
            industry_share = {"top_industry": None, "share": 0.0,
                              "coverage": coverage}
    except Exception as e:
        logger.warning("industry mapping failed: {}", e)

    verdict["pick_sanity"] = {
        "latest_date": str(pd.Timestamp(latest).date()),
        "topk": PICK_TOPK, "overlap": overlap, "nan_scores": nan_scores,
        "degenerate_scores": degenerate, "industry": industry_share,
    }
    if nan_scores > 0 or degenerate:
        verdict["abort"] = True
        verdict["reasons"].append(
            f"pick-sanity FAIL: nan_scores={nan_scores}, degenerate={degenerate}")
    if overlap < PICK_MIN_OVERLAP:
        verdict["reasons"].append(
            f"pick-sanity WARN: top-{PICK_TOPK} overlap {overlap:.2f} < "
            f"{PICK_MIN_OVERLAP} (high churn — review, not abort)")
    if (industry_share and industry_share.get("coverage", 0) >= 0.5
            and industry_share["share"] > PICK_MAX_INDUSTRY):
        verdict["reasons"].append(
            f"pick-sanity WARN: {industry_share['top_industry']} "
            f"{industry_share['share']:.0%} of mapped picks > {PICK_MAX_INDUSTRY:.0%} "
            "(sector pile-up — review)")
    elif industry_share and industry_share.get("coverage", 0) < 0.5:
        verdict["reasons"].append(
            f"pick-sanity NOTE: industry coverage only "
            f"{industry_share['coverage']:.0%} — concentration check skipped "
            "(SW mapping cache cold; not an alarm)")

    # ---- summary ----
    logger.info("=" * 56)
    logger.info("RETRAIN VERIFY: {} vs {}", args.new_prefix, args.prod_prefix)
    logger.info("=" * 56)
    for gname, g in verdict["gates"].items():
        logger.info("  [{}] {}: {}", "PASS" if g["pass"] else "FAIL", gname,
                    {k: v for k, v in g.items() if k not in ("pass", "note")})
    ps_ = verdict["pick_sanity"]
    logger.info("  pick-sanity: overlap={:.2f} nan={} degen={} industry={}",
                ps_["overlap"], ps_["nan_scores"], ps_["degenerate_scores"],
                ps_["industry"])
    logger.info("-" * 56)
    if verdict["abort"]:
        logger.error("VERDICT: ABORT (do NOT swap)")
        for r in verdict["reasons"]:
            logger.error("  ✗ {}", r)
    else:
        logger.info("VERDICT: PASS gates (non-degenerate + not-worse + tracks prod)")
        logger.info("  → eligible for swap pending advisor + user review")
        for r in verdict["reasons"]:
            logger.warning("  ⚠ {}", r)

    out_path = args.json_out or str(
        ROOT / "data" / f"retrain_verify_{pd.Timestamp(latest).strftime('%Y%m%d')}.json")
    Path(out_path).write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    logger.info("Verdict JSON → {}", out_path)

    return 1 if verdict["abort"] else 0


if __name__ == "__main__":
    sys.exit(main())
