# P11-4 design spike: real intraday data via xtdata on ECS

Status: **DESIGN ONLY** (no code).  Round 77 deliverable.

Goal: replace EOD-proxy training (current P11-2 path) with **real**
9:30-14:30 1m bars sourced from `xtdata.get_local_data` on the ECS QMT
install.  Concrete cost + path numbers to inform "is it worth the effort
to go from proxy → real for P11 retraining."

---

## 1 · Data size estimate

Universe: HS300 + ZZ500 ≈ 800 codes.  Trading days/year: ~245.  1m bars
per session: 240 (4 hours × 60 min, A股 9:30–11:30 + 13:00–15:00).

| Slice | Bars | Cols (OHLCV + amount = 6) | Size (raw f64) | Parquet (~10× zstd) |
|---|---|---|---|---|
| 1 code × 1 day | 240 | 6 | 11.5 KB | ~1.2 KB |
| 1 code × 1 yr | 58,800 | 6 | 2.8 MB | ~0.3 MB |
| 1 code × 5 yr | 294,000 | 6 | 14 MB | ~1.5 MB |
| **800 codes × 1 yr** | 47.0M | 6 | **2.2 GB** | **~225 MB** |
| **800 codes × 5 yr** | 235M | 6 | **11.2 GB** | **~1.1 GB** |

Round-73-supplement probe showed xtdata's local cache plateau at ~1 year
deep on the current ECS install (5-yr query returns same 58,407 bars as
1-yr).  Need to run `xtdata.download_history_data(symbols, period='1m',
start_time='20200101')` to backfill before 5-year training is feasible.

Backfill time estimate (sequential, conservative): 800 codes × ~30 s/code
for 1-yr download = **~7 hours one-shot**, then incremental nightly.
Parallelism via xtdata is unclear — may need rate-limit testing.

---

## 2 · ECS xtdata stability test plan (before bulk download)

| Test | Method | Pass criteria |
|---|---|---|
| Rate limit | Pull 800 codes × 1 day in single thread; measure WPS + failures | ≥ 95% success, no auth lock-out |
| Cache idempotence | Re-pull same day twice; compare bytes-equal | byte-identical |
| Connection drop | Kill QMT mid-pull, restart, resume | no double-count, no missing bars |
| Backfill window | `download_history_data` for 1 specific code, 5-yr range; verify get_local_data returns the deep history | bar count ≥ 252×5×240 = 302k |
| Bar timing | Sample 5 codes × 5 days; verify timestamps are exchange-time (Asia/Shanghai 9:30:00 etc) | no UTC drift |

---

## 3 · Training path: Mac vs ECS

| Option | Pros | Cons |
|---|---|---|
| **A. ECS local train** — install LightGBM + project on Windows ECS, train in-place | No data transfer; matches inference env exactly | ECS is Windows, project's `mp/` is mac-tested; LightGBM on Windows works but venv setup is extra; cron + monitoring needs Windows Task Scheduler not crontab |
| **B. ECS dump parquet → Mac train** | Mac toolchain unchanged; existing `.venv/bin/python scripts/train_intraday.py` keeps working | One-time scp/git-lfs of 225 MB-1.1 GB; needs nightly refresh sync; Mac↔ECS clock-skew check for incremental |
| **C. Hybrid** — ECS pre-aggregates 1m bars into per-day per-code 14:30 snapshots (≤6 cols × 800 codes × 245 days = ~12 MB/yr); Mac trains on the snapshots | Minimal data transfer; snapshots are the only thing the model actually needs; isolation between data layer (ECS) and ML layer (Mac) | Slight info loss vs full 1m bars; need to redo aggregation if extras' window changes (e.g. switch from 14:30 to 14:00) |

Recommend **C** as default — snapshots are the model's actual input
contract, and aggregating on ECS keeps the slow path local to where the
data lives.

---

## 4 · Decision gate: when to actually run P11-4

Three orderings under consideration:

| Order | Logic | Risk |
|---|---|---|
| **P11-2b → P11-3 → P11-4** | Confirm proxy training is healthy first; if walk-fwd Sharpe gain ≥ 0.15, real data only worth it for productionization | Risk if proxy gives false positive (model trained on synthetic patterns that don't exist in real intraday) |
| **P11-2b → P11-4 → P11-3** | Replace synthetic features with real before any walk-fwd verdict | 1-2 week delay before any verdict; xtdata + backfill cost unknown until tested |
| **P11-4 + P11-2b parallel** | Engineer runs control A/B (P11-2b) on Mac while ECS spike runs in background | Coordination cost; both paths burn time but different bottlenecks |

The right ordering depends on **P11-2b's verdict**:
- If P11-2b shows control IC ~0.03 (4 extras add noise, no signal):
  → schema is the problem; try `overnight_gap` only re-train.
  → if THAT still doesn't help, then EOD-proxy is fundamentally inadequate → P11-4 mandatory.
- If P11-2b shows control IC also ~0.008:
  → problem is NOT the 4 extras; investigate label horizon / train_fast hyperparams first.
  → P11-4 won't help until base setup verified.

---

## 5 · Cost summary

| Item | Effort | Calendar |
|---|---|---|
| xtdata backfill 800 codes × 1 yr | ~7 hr ECS bandwidth, 1 line code | 1 day |
| Stability test plan execution | 4 tests × ~30 min each + analysis | 1 day |
| ECS-Mac transport (option C aggregator) | small Python script, ~150 lines | 1 day |
| `train_intraday.py --real-data` flag | add data-source switch | 0.5 day |
| Retrain + dump importance | 5-10 min compute | same day |
| P11-3 walk-forward with real data | 6 runs × ~10 min + 6-num table | 1 day |
| **Total** | | **~5 working days** |

This is roughly the P11-4 budget from round 73's 2-4 week estimate (~25%).
If P11-2b makes proxy look viable, P11-4 can be back-burnered behind γ.

---

*Round 77 deliverable. No code emitted — engineer to wait for advisor green
light on P11-4 ordering before implementing.*
