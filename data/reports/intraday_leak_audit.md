# Intraday (Midday) Report Leak Audit

**Audit date**: 2026-04-21
**Scope**: `scripts/daily_report.py` midday path + `mp/ml/dataset.build_latest_features(intraday_bars=...)` + valuation/industry joins.

## TL;DR

**No look-ahead leak found.** The midday report uses only data known at the moment of the snapshot. However, there's a **distribution shift** risk: features computed from "morning-so-far" prices are fed into a model trained on end-of-day data. This is not a bug but an architectural note — it explains why rankings can shift between midday and afternoon runs even when reported bars look similar (e.g. 粤电力A 2026-04-21 ranking jump).

## Data flow at midday

```
1. holdings + ZZ500 universe
    ↓
2. _fetch_realtime_prices() → Sina live quote
    • open = today's open (EOD-equivalent, known)
    • high/low/close = morning-session so-far (known at t=snapshot)
    • volume/amount = morning cumulative (known)
    ↓
3. intraday_bars dict passed to build_latest_features()
    ↓
4. _process_single_stock() does:
      df = get_daily_bars()              ← DB up to yesterday ✅
      df = df[df.date < today]           ← drops any stale today row ✅
      df = concat(df, intraday_row)      ← appends morning partial as "today"
    ↓
5. Factors (mom_20d, rsi_14, ...) computed on [..., yesterday, morning-partial]
6. Valuation:
      snap = get_valuation_snapshot()    ← stock_zh_a_spot_em (LIVE quote)
      PE/PB/total_mv all = f(current_price)  — current, known
7. Industry mapping: get_industry_mapping() — current snapshot
8. StockRanker.predict() scores the single latest row per stock
```

## Per-source verdict

| Source | Content at midday | Leak? | Note |
|---|---|---|---|
| `get_daily_bars` | DB-cached EOD bars up to t-1 | ✅ safe | Explicit `df < today` filter |
| `_fetch_realtime_prices` (Sina) | morning O/H/L/C/V/A | ✅ safe | Known at snapshot time |
| `get_valuation_snapshot()` | current-price-based PE/PB/mv | ✅ safe | Derived from current price, which IS known |
| `get_financial_data()` | quarterly `publish_date`-aligned | ✅ safe | PIT by construction |
| `get_industry_mapping()` | current industry assignments | ⚠ non-PIT but not a leak at live-predict time | Only matters for BACKTEST, where `merge_asof(backward)` path is used |
| `_compute_benchmark_fwd_ret` | future bench return | 🚫 not invoked at midday | Training-only |

## Distribution-shift note (not a leak, but worth documenting)

The model is trained on factor values computed from **closing prices**:
- `mom_20d[t] = close[t]/close[t-20] - 1`, where `close[t]` is day-end.

At midday, we substitute `close[today]` with the **latest intraday print** (e.g. 11:25 AM price). So:
- A stock up 3% at lunch but closing flat will look "strong" on the midday scan but average by afternoon re-score.
- This is PIT-consistent (no future info used) but represents a **different distribution** from training.

### What the user saw with 粤电力A (2026-04-21)

- Midday report scored based on morning price (+1.8% up on light volume).
- Afternoon session saw heavy buying → 涨停.
- Afternoon re-score used a very different `mom_5d/rsi_14/turnover` snapshot → higher rank.
- **Cause**: distribution shift, not a leak. The morning features simply under-represented the stock's next-hour trajectory.

## Recommendations

**Accept as-is.** Building a separate model on "morning-only" features would:
- Double the training burden.
- Fight against the fact that morning/EOD features are highly correlated on most days (afternoon just adds noise to what the morning already told you).

If the distribution shift becomes a problem (e.g. consistently different midday vs EOD rankings):
1. **Cheap mitigation**: append an `intraday_confidence` column that shows time-of-day (e.g. drop from 1.0 to 0.6 before 11:00, rising to 0.95 at 14:30). Downstream consumers can weight accordingly.
2. **Heavy mitigation**: train an auxiliary `morning_ranker` on (morning factors → full-day fwd_ret) and blend with the main ranker during intraday scans. Not warranted by current evidence.

## Files covered

- `scripts/daily_report.py` (lines 1590-1730 — `run_midday_report`)
- `mp/ml/dataset.py::_process_single_stock` (lines 645-748)
- `mp/ml/dataset.py::build_latest_features` (lines 1077-1183)
- `mp/ml/dataset.py::_fetch_valuation_snapshot_map` (lines 444-495)
- `mp/data/fetcher.py::get_valuation_snapshot` (lines 656-687) + `_get_valuation_snapshot_em` (lines 689-702)

## Regression guard (recommendation)

Add a test `tests/test_intraday_pit.py` that:

1. Mocks `get_daily_bars` to return history up to t-1.
2. Mocks `_fetch_realtime_prices` to return a morning bar dated t.
3. Mocks `get_valuation_snapshot` to return a fresh snapshot.
4. Runs `build_latest_features(..., intraday_bars=...)` and asserts the last row's `date == t` AND that no factor value could only be produced with `close[t']` for `t' > t`.

Low priority but would lock in the current PIT guarantee.
