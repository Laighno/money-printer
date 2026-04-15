"""Model validation suite — 5 core analyses for production readiness.

Analyses:
  1. Yearly split performance (2020-2025)
  2. Market regime split (bull / sideways / bear)
  3. Top-K stability (K = 5, 10, 20, 30)
  4. After-fee returns (with realistic turnover & costs)
  5. Single model vs Blend comparison

Usage:
    python scripts/model_validation.py [--experiments blend_best,deep64,baseline]
    python scripts/model_validation.py --all         # run on all diagnostics_*.parquet
    python scripts/model_validation.py --top 5       # top 5 by relative hit rate
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Ensure project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from mp.regime.detector import ROUND_TRIP_COST

# ── Constants ────────────────────────────────────────────────────────────────

CACHE_DIR = Path("data/wf_cache")
REPORT_DIR = Path("data/reports")
DEFAULT_EXPERIMENTS = ["blend_best", "excess_deep64", "deep64", "baseline"]
TOP_K_DEFAULT = 10
HORIZONS_K = [5, 10, 20, 30]

# Realistic transaction costs (per round-trip)
SLIPPAGE_BPS = 5  # one-way
COMMISSION_BPS = 3  # one-way
ONE_WAY_COST = (SLIPPAGE_BPS + COMMISSION_BPS) / 10000  # 0.08%
# stamp duty only on sell: 0.05%
STAMP_DUTY = 0.0005


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_zz500_benchmark() -> pd.Series:
    """Load ZZ500 index daily close → forward 20d return series."""
    import akshare as ak
    idx_df = ak.stock_zh_index_daily(symbol="sh000905")
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_df = idx_df.sort_values("date").set_index("date")
    close = idx_df["close"].astype(float)
    fwd = close.shift(-20) / close - 1.0
    return fwd


def load_experiment(name: str) -> pd.DataFrame | None:
    """Load a diagnostics parquet by experiment name."""
    if name == "baseline":
        path = CACHE_DIR / "diagnostics.parquet"
    else:
        path = CACHE_DIR / f"diagnostics_{name}.parquet"
    if not path.exists():
        print(f"  SKIP {name}: file not found ({path})")
        return None
    df = pd.read_parquet(path)
    df["score_date"] = pd.to_datetime(df["score_date"])
    return df


# ── Analysis 1: Yearly Split ────────────────────────────────────────────────

def yearly_split(df: pd.DataFrame, zz500_fwd: pd.Series,
                 top_k: int = TOP_K_DEFAULT) -> pd.DataFrame:
    """Per-year Top-K hit rate, IC, long-short spread, excess over ZZ500."""
    df = df.copy()
    df["year"] = df["score_date"].dt.year
    rows = []
    for year, g in df.groupby("year"):
        months = g.groupby(g["score_date"].dt.to_period("M"))
        monthly_ic, monthly_hit, monthly_excess, monthly_spread = [], [], [], []

        for _, mg in months:
            if len(mg) < top_k * 2:
                continue
            top = mg.nlargest(top_k, "pred_score")
            bot = mg.nsmallest(top_k, "pred_score")

            # ZZ500 benchmark
            bench = zz500_fwd.asof(mg["score_date"].iloc[0])
            if pd.isna(bench):
                continue

            monthly_hit.append((top["fwd_ret"] > bench).mean())
            monthly_excess.append(top["fwd_ret"].mean() - bench)
            monthly_spread.append(top["fwd_ret"].mean() - bot["fwd_ret"].mean())

            valid = mg.dropna(subset=["pred_score", "fwd_ret"])
            if len(valid) > 10:
                ic, _ = spearmanr(valid["pred_score"], valid["fwd_ret"])
                monthly_ic.append(ic)

        if not monthly_hit:
            continue
        rows.append({
            "year": year,
            "rel_hit": np.mean(monthly_hit),
            "excess_mo": np.mean(monthly_excess),
            "spread": np.mean(monthly_spread),
            "ic_mean": np.mean(monthly_ic) if monthly_ic else np.nan,
            "ic_ir": (np.mean(monthly_ic) / np.std(monthly_ic)
                      if monthly_ic and np.std(monthly_ic) > 0 else np.nan),
            "n_months": len(monthly_hit),
        })
    return pd.DataFrame(rows)


# ── Analysis 2: Market Regime Split ──────────────────────────────────────────

def regime_split(df: pd.DataFrame, zz500_fwd: pd.Series,
                 top_k: int = TOP_K_DEFAULT) -> pd.DataFrame:
    """Performance by market regime (bull/sideways/bear based on ZZ500)."""
    df = df.copy()
    df["month"] = df["score_date"].dt.to_period("M")
    rows = []

    for month, mg in df.groupby("month"):
        if len(mg) < top_k * 2:
            continue
        top = mg.nlargest(top_k, "pred_score")
        bench = zz500_fwd.asof(mg["score_date"].iloc[0])
        if pd.isna(bench):
            continue

        # Regime classification
        if bench > 0.02:
            regime = "bull"
        elif bench < -0.02:
            regime = "bear"
        else:
            regime = "sideways"

        rel_hit = (top["fwd_ret"] > bench).mean()
        excess = top["fwd_ret"].mean() - bench

        valid = mg.dropna(subset=["pred_score", "fwd_ret"])
        ic = np.nan
        if len(valid) > 10:
            ic, _ = spearmanr(valid["pred_score"], valid["fwd_ret"])

        rows.append({
            "month": str(month), "regime": regime,
            "rel_hit": rel_hit, "excess": excess, "ic": ic,
            "bench_ret": bench,
        })

    mdf = pd.DataFrame(rows)
    if mdf.empty:
        return pd.DataFrame()

    summary = []
    for regime in ["bull", "sideways", "bear"]:
        sub = mdf[mdf["regime"] == regime]
        if sub.empty:
            continue
        summary.append({
            "regime": regime,
            "n_months": len(sub),
            "rel_hit": sub["rel_hit"].mean(),
            "excess_mo": sub["excess"].mean(),
            "ic_mean": sub["ic"].mean(),
            "bench_avg": sub["bench_ret"].mean(),
        })
    return pd.DataFrame(summary)


# ── Analysis 3: Top-K Stability ──────────────────────────────────────────────

def topk_stability(df: pd.DataFrame, zz500_fwd: pd.Series,
                   ks: list[int] | None = None) -> pd.DataFrame:
    """Hit rate and excess for varying K = 5, 10, 20, 30."""
    if ks is None:
        ks = HORIZONS_K
    df = df.copy()
    df["month"] = df["score_date"].dt.to_period("M")
    rows = []

    for k in ks:
        monthly_hit, monthly_excess = [], []
        for _, mg in df.groupby("month"):
            if len(mg) < k * 2:
                continue
            top = mg.nlargest(k, "pred_score")
            bench = zz500_fwd.asof(mg["score_date"].iloc[0])
            if pd.isna(bench):
                continue
            monthly_hit.append((top["fwd_ret"] > bench).mean())
            monthly_excess.append(top["fwd_ret"].mean() - bench)

        if monthly_hit:
            rows.append({
                "top_k": k,
                "rel_hit": np.mean(monthly_hit),
                "excess_mo": np.mean(monthly_excess),
                "hit_std": np.std(monthly_hit),
                "n_months": len(monthly_hit),
            })
    return pd.DataFrame(rows)


# ── Analysis 4: After-Fee Returns ────────────────────────────────────────────

def after_fee_returns(df: pd.DataFrame, zz500_fwd: pd.Series,
                      top_k: int = TOP_K_DEFAULT) -> pd.DataFrame:
    """Simulate monthly turnover and deduct realistic transaction costs.

    Assumes equal-weight monthly rebalance. Turnover = fraction of portfolio
    replaced each month. Cost = turnover × round_trip_cost.
    """
    df = df.copy()
    df["month"] = df["score_date"].dt.to_period("M")
    months = sorted(df["month"].unique())
    rows = []
    prev_codes: set = set()

    for month in months:
        mg = df[df["month"] == month]
        if len(mg) < top_k * 2:
            continue
        top = mg.nlargest(top_k, "pred_score")
        top_codes = set(top["code"].values)
        bench = zz500_fwd.asof(mg["score_date"].iloc[0])
        if pd.isna(bench):
            continue

        # Turnover
        if prev_codes:
            n_changed = len(top_codes - prev_codes)
            turnover = n_changed / top_k
        else:
            turnover = 1.0  # first month: buy everything

        # Costs: buy-side + sell-side for changed positions
        # Buy: commission + slippage; Sell: commission + slippage + stamp duty
        cost_per_trade = ONE_WAY_COST * 2 + STAMP_DUTY  # round-trip for changed
        total_cost = turnover * cost_per_trade

        gross_ret = top["fwd_ret"].mean()
        net_ret = gross_ret - total_cost
        excess_gross = gross_ret - bench
        excess_net = net_ret - bench

        rows.append({
            "month": str(month),
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "turnover": turnover,
            "cost": total_cost,
            "excess_gross": excess_gross,
            "excess_net": excess_net,
            "bench": bench,
        })
        prev_codes = top_codes

    mdf = pd.DataFrame(rows)
    if mdf.empty:
        return pd.DataFrame()

    # Summary row
    summary = pd.DataFrame([{
        "avg_turnover": mdf["turnover"].mean(),
        "avg_cost_mo": mdf["cost"].mean(),
        "gross_excess_mo": mdf["excess_gross"].mean(),
        "net_excess_mo": mdf["excess_net"].mean(),
        "gross_excess_ann": mdf["excess_gross"].mean() * 12,
        "net_excess_ann": mdf["excess_net"].mean() * 12,
        "cost_drag_ann": mdf["cost"].mean() * 12,
        "n_months": len(mdf),
    }])
    return summary, mdf


# ── Analysis 5: Single vs Blend Comparison ───────────────────────────────────

def single_vs_blend(experiments: dict[str, pd.DataFrame],
                    zz500_fwd: pd.Series,
                    top_k: int = TOP_K_DEFAULT) -> pd.DataFrame:
    """Compare all loaded experiments on unified metrics."""
    rows = []
    for name, df in experiments.items():
        df = df.copy()
        df["month"] = df["score_date"].dt.to_period("M")
        monthly_hit, monthly_excess, monthly_ic = [], [], []

        for _, mg in df.groupby("month"):
            if len(mg) < top_k * 2:
                continue
            top = mg.nlargest(top_k, "pred_score")
            bench = zz500_fwd.asof(mg["score_date"].iloc[0])
            if pd.isna(bench):
                continue
            monthly_hit.append((top["fwd_ret"] > bench).mean())
            monthly_excess.append(top["fwd_ret"].mean() - bench)
            valid = mg.dropna(subset=["pred_score", "fwd_ret"])
            if len(valid) > 10:
                ic, _ = spearmanr(valid["pred_score"], valid["fwd_ret"])
                monthly_ic.append(ic)

        if not monthly_hit:
            continue

        is_blend = "blend" in name.lower()
        rows.append({
            "experiment": name,
            "type": "blend" if is_blend else "single",
            "rel_hit": np.mean(monthly_hit),
            "excess_mo": np.mean(monthly_excess),
            "ic_mean": np.mean(monthly_ic) if monthly_ic else np.nan,
            "ic_ir": (np.mean(monthly_ic) / np.std(monthly_ic)
                      if monthly_ic and len(monthly_ic) > 1 and np.std(monthly_ic) > 0
                      else np.nan),
            "n_months": len(monthly_hit),
        })

    return pd.DataFrame(rows).sort_values("rel_hit", ascending=False)


# ── Report Formatting ────────────────────────────────────────────────────────

def format_report(name: str,
                  yearly: pd.DataFrame,
                  regime: pd.DataFrame,
                  topk: pd.DataFrame,
                  fee_summary: pd.DataFrame,
                  fee_monthly: pd.DataFrame,
                  comparison: pd.DataFrame) -> str:
    """Generate markdown validation report."""
    lines = [
        f"# 模型验证报告 — {name}",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # 1. Yearly split
    lines += [
        "## 1. 年度拆分表现",
        "",
        f"| 年份 | 相对胜率 | 月均超额 | 多空价差 | IC均值 | ICIR | 月数 |",
        f"|------|---------|---------|---------|--------|------|------|",
    ]
    for _, r in yearly.iterrows():
        lines.append(
            f"| {int(r['year'])} | {r['rel_hit']:.1%} | {r['excess_mo']:+.2%} "
            f"| {r['spread']:+.2%} | {r['ic_mean']:.3f} "
            f"| {r['ic_ir']:.2f} | {int(r['n_months'])} |"
        )
    avg = yearly.mean(numeric_only=True)
    lines.append(
        f"| **平均** | **{avg['rel_hit']:.1%}** | **{avg['excess_mo']:+.2%}** "
        f"| **{avg['spread']:+.2%}** | **{avg['ic_mean']:.3f}** "
        f"| **{avg['ic_ir']:.2f}** | |"
    )
    # Year-to-year std
    lines.append(f"\n年间标准差: 胜率 σ={yearly['rel_hit'].std():.1%}, "
                 f"超额 σ={yearly['excess_mo'].std():.2%}")
    lines.append("")

    # 2. Regime split
    lines += [
        "## 2. 市场状态拆分",
        "",
        "| 状态 | 月数 | 相对胜率 | 月均超额 | IC均值 | ZZ500月均 |",
        "|------|------|---------|---------|--------|----------|",
    ]
    regime_cn = {"bull": "牛市", "sideways": "震荡", "bear": "熊市"}
    for _, r in regime.iterrows():
        lines.append(
            f"| {regime_cn.get(r['regime'], r['regime'])} | {int(r['n_months'])} "
            f"| {r['rel_hit']:.1%} | {r['excess_mo']:+.2%} "
            f"| {r['ic_mean']:.3f} | {r['bench_avg']:+.2%} |"
        )
    lines.append("")

    # 3. Top-K stability
    lines += [
        "## 3. Top-K 稳定性",
        "",
        "| K | 相对胜率 | 月均超额 | 胜率波动 | 月数 |",
        "|---|---------|---------|---------|------|",
    ]
    for _, r in topk.iterrows():
        marker = " ←" if int(r["top_k"]) == TOP_K_DEFAULT else ""
        lines.append(
            f"| {int(r['top_k'])} | {r['rel_hit']:.1%} | {r['excess_mo']:+.2%} "
            f"| {r['hit_std']:.1%} | {int(r['n_months'])}{marker} |"
        )
    # Monotonicity check
    if len(topk) >= 3:
        hits = topk.sort_values("top_k")["rel_hit"].values
        if all(hits[i] >= hits[i + 1] - 0.02 for i in range(len(hits) - 1)):
            lines.append("\n✓ Top-K 收敛：K 增大时胜率平稳下降，信号集中在头部")
        else:
            lines.append("\n⚠ Top-K 不稳定：K 变化时胜率波动大，信号可能不稳健")
    lines.append("")

    # 4. After-fee returns
    lines += [
        "## 4. 扣费后收益",
        "",
    ]
    if not fee_summary.empty:
        fs = fee_summary.iloc[0]
        lines += [
            f"- 平均月换手率: {fs['avg_turnover']:.1%}",
            f"- 月均交易成本: {fs['avg_cost_mo']:.3%}",
            f"- **年化成本拖累**: {fs['cost_drag_ann']:.2%}",
            f"- 毛超额年化: {fs['gross_excess_ann']:+.2%}",
            f"- **净超额年化**: {fs['net_excess_ann']:+.2%}",
            f"- 成本吞噬比: {fs['cost_drag_ann'] / abs(fs['gross_excess_ann']) * 100:.1f}% "
            f"(成本占毛超额的比例)" if fs['gross_excess_ann'] != 0 else "- 成本吞噬比: N/A",
            "",
        ]
        # Quarterly net excess
        fee_monthly["quarter"] = pd.PeriodIndex(fee_monthly["month"], freq="M").to_timestamp().to_period("Q")
        qdf = fee_monthly.groupby("quarter").agg(
            net_excess=("excess_net", "mean"),
            turnover=("turnover", "mean"),
        )
        lines += [
            "### 季度净超额",
            "",
            "| 季度 | 净超额/月 | 换手率 |",
            "|------|----------|--------|",
        ]
        for q, r in qdf.iterrows():
            lines.append(f"| {q} | {r['net_excess']:+.2%} | {r['turnover']:.1%} |")
    lines.append("")

    # 5. Single vs Blend
    lines += [
        "## 5. 单模型 vs Blend 对照",
        "",
        "| 实验 | 类型 | 相对胜率 | 月均超额 | IC均值 | ICIR | 月数 |",
        "|------|------|---------|---------|--------|------|------|",
    ]
    for _, r in comparison.iterrows():
        marker = " **" if r["rel_hit"] == comparison["rel_hit"].max() else ""
        lines.append(
            f"| {r['experiment']} | {r['type']} | {r['rel_hit']:.1%} "
            f"| {r['excess_mo']:+.2%} | {r['ic_mean']:.3f} "
            f"| {r['ic_ir']:.2f} | {int(r['n_months'])}{marker} |"
        )

    # Summary verdict
    lines += ["", "---", "", "## 综合诊断", ""]

    # Auto-generated verdict
    best = comparison.iloc[0] if not comparison.empty else None
    if best is not None:
        verdict_parts = []

        # Yearly stability
        if yearly["rel_hit"].std() < 0.05:
            verdict_parts.append("✓ 年间稳定性好（σ < 5pp）")
        else:
            verdict_parts.append(f"⚠ 年间波动偏大（σ = {yearly['rel_hit'].std():.1%}）")

        # Bear market
        bear_row = regime[regime["regime"] == "bear"]
        if not bear_row.empty and bear_row.iloc[0]["rel_hit"] > 0.50:
            verdict_parts.append(f"✓ 熊市仍有超额（胜率 {bear_row.iloc[0]['rel_hit']:.1%}）")
        elif not bear_row.empty:
            verdict_parts.append(f"⚠ 熊市表现弱（胜率 {bear_row.iloc[0]['rel_hit']:.1%}）")

        # Cost impact
        if not fee_summary.empty:
            fs = fee_summary.iloc[0]
            if fs["net_excess_ann"] > 0.03:
                verdict_parts.append(f"✓ 扣费后年化超额 {fs['net_excess_ann']:+.1%} 可观")
            elif fs["net_excess_ann"] > 0:
                verdict_parts.append(f"△ 扣费后年化超额仅 {fs['net_excess_ann']:+.2%}")
            else:
                verdict_parts.append(f"✗ 扣费后年化超额为负 {fs['net_excess_ann']:+.2%}")

        # K stability
        if len(topk) >= 2:
            k5_hit = topk[topk["top_k"] == 5]["rel_hit"].values
            k30_hit = topk[topk["top_k"] == 30]["rel_hit"].values
            if len(k5_hit) and len(k30_hit):
                diff = k5_hit[0] - k30_hit[0]
                if diff > 0.03:
                    verdict_parts.append(f"✓ 头部集中度好（K5-K30 = {diff:+.1%}）")
                else:
                    verdict_parts.append(f"△ 头部集中度一般（K5-K30 = {diff:+.1%}）")

        for v in verdict_parts:
            lines.append(f"- {v}")

    lines.append("")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model validation suite")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated experiment names")
    parser.add_argument("--all", action="store_true",
                        help="Run on all diagnostics files")
    parser.add_argument("--top", type=int, default=None,
                        help="Only validate top N experiments by rel_hit")
    parser.add_argument("--focus", type=str, default=None,
                        help="Primary experiment for detailed report (default: blend_best)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output markdown path")
    args = parser.parse_args()

    print("Loading ZZ500 benchmark...")
    zz500_fwd = load_zz500_benchmark()
    print(f"  ZZ500 fwd_ret range: {zz500_fwd.index.min().date()} → {zz500_fwd.index.max().date()}")

    # Determine which experiments to load
    if args.all:
        exp_names = ["baseline"]
        for f in sorted(CACHE_DIR.glob("diagnostics_*.parquet")):
            exp_names.append(f.stem.replace("diagnostics_", ""))
    elif args.experiments:
        exp_names = [e.strip() for e in args.experiments.split(",")]
    else:
        exp_names = DEFAULT_EXPERIMENTS

    # Load experiments
    experiments: dict[str, pd.DataFrame] = {}
    for name in exp_names:
        df = load_experiment(name)
        if df is not None:
            experiments[name] = df

    if not experiments:
        print("No experiments loaded. Exiting.")
        return

    print(f"Loaded {len(experiments)} experiments: {list(experiments.keys())}")

    # If --top, pre-screen by relative hit rate
    if args.top and len(experiments) > args.top:
        quick = single_vs_blend(experiments, zz500_fwd)
        keep = set(quick.head(args.top)["experiment"].values)
        experiments = {k: v for k, v in experiments.items() if k in keep}
        print(f"Filtered to top {args.top}: {list(experiments.keys())}")

    # Focus experiment for detailed report
    focus = args.focus or "blend_best"
    if focus not in experiments:
        focus = list(experiments.keys())[0]
    focus_df = experiments[focus]

    print(f"\n{'='*80}")
    print(f"DETAILED VALIDATION: {focus}")
    print(f"{'='*80}")

    # ── Analysis 1: Yearly ────
    print("\n[1/5] Yearly split...")
    yearly = yearly_split(focus_df, zz500_fwd)
    print(yearly.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── Analysis 2: Regime ────
    print("\n[2/5] Market regime split...")
    regime = regime_split(focus_df, zz500_fwd)
    print(regime.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── Analysis 3: Top-K ────
    print("\n[3/5] Top-K stability...")
    topk = topk_stability(focus_df, zz500_fwd)
    print(topk.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── Analysis 4: After-fee ────
    print("\n[4/5] After-fee returns...")
    fee_result = after_fee_returns(focus_df, zz500_fwd)
    if fee_result is not None:
        fee_summary, fee_monthly = fee_result
        print(fee_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        fee_summary, fee_monthly = pd.DataFrame(), pd.DataFrame()

    # ── Analysis 5: Comparison ────
    print("\n[5/5] Single vs Blend comparison...")
    comparison = single_vs_blend(experiments, zz500_fwd)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── Generate Report ────
    report = format_report(focus, yearly, regime, topk, fee_summary, fee_monthly, comparison)

    out_path = args.output or str(REPORT_DIR / "model_validation.md")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(report, encoding="utf-8")
    print(f"\n{'='*80}")
    print(f"Report saved to {out_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
