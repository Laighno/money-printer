"""Batch IC analysis across ZZ500 sample stocks.

Computes cross-sectional mean IC, mean |IR|, IC sign, and positive ratio
for each of the 13 technical factors at the 20-day horizon.
"""

import sys
import random
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/laighno/laighno/money-printer")

from mp.data.fetcher import get_index_constituents
from mp.backtest.ic_analysis import run_ic_analysis

random.seed(42)

print("Fetching ZZ500 constituents...")
codes = get_index_constituents("zz500")
print(f"Total constituents: {len(codes)}")

sample = random.sample(codes, min(100, len(codes)))
print(f"Sampled {len(sample)} stocks for analysis\n")

results = {}  # factor_name -> list of (ic_20d, ir_20d)
failed = 0
empty = 0
t0 = time.time()

for i, code in enumerate(sample):
    try:
        df = run_ic_analysis(code, start="20230101", horizons=(20,))
        if df is None or df.empty:
            empty += 1
            print(f"  [{i+1:3d}] {code} — empty result (insufficient data)")
            continue
        for factor_name in df.index:
            if factor_name not in results:
                results[factor_name] = []
            ic_val = df.loc[factor_name, "IC(20d)"]
            ir_val = df.loc[factor_name, "IR(20d)"]
            if not np.isnan(ic_val):
                results[factor_name].append((ic_val, ir_val))
    except Exception as e:
        failed += 1
        print(f"  [{i+1:3d}] {code} failed: {e}")
        continue

    if (i + 1) % 20 == 0:
        elapsed = time.time() - t0
        rate = elapsed / (i + 1)
        eta = rate * (len(sample) - i - 1)
        print(f"Progress: {i+1}/{len(sample)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

elapsed_total = time.time() - t0
print(f"\nDone. {len(sample)} stocks processed in {elapsed_total:.1f}s")
print(f"  Failed: {failed}, Empty: {empty}, Successful: {len(sample) - failed - empty}")

print("\n" + "=" * 75)
print(f" ZZ500 Cross-Sectional IC Analysis (20d horizon, N={len(sample)} sample)")
print("=" * 75)
print()
print(f"{'Factor':<14} {'Mean IC':>10} {'Std IC':>10} {'Mean |IR|':>10} {'IC Sign':>8} {'IC>0 %':>8} {'N':>6}")
print("-" * 72)

summary_rows = []
for fname, vals in results.items():
    ics = np.array([v[0] for v in vals])
    irs = np.array([v[1] for v in vals])
    mean_ic = np.nanmean(ics)
    std_ic = np.nanstd(ics)
    mean_abs_ir = np.nanmean(np.abs(irs))
    ic_sign = +1 if mean_ic > 0 else -1
    ic_positive_pct = np.mean(ics > 0) * 100
    summary_rows.append((fname, mean_ic, std_ic, mean_abs_ir, ic_sign, ic_positive_pct, len(vals)))

# Sort by |mean IR| descending
summary_rows.sort(key=lambda x: -x[3])

for fname, mean_ic, std_ic, mean_abs_ir, ic_sign, ic_pos, n in summary_rows:
    print(f"{fname:<14} {mean_ic:>+10.4f} {std_ic:>10.4f} {mean_abs_ir:>10.3f} {ic_sign:>+8d} {ic_pos:>7.1f}% {n:>6d}")

print()
print("Legend: IC Sign = direction to use in screener (+1 = higher factor -> higher return)")
print("        Mean |IR| > 0.5 is considered a strong factor")
print("        IC>0 % > 55% or < 45% indicates directional consistency")
