# Decision Log

Single-page summary of advisor decisions made during the
`collab/advisor-dialog` branch P0 → P2 chain (2026-05-23 → 2026-05-24).
For full reasoning behind each row, follow the `Round` link to
[docs/dialog/to_engineer.md](dialog/to_engineer.md) and
[docs/dialog/to_advisor.md](dialog/to_advisor.md).

For active follow-ups not yet decided, see [docs/TODO.md](TODO.md).

## Decision table

| # | Decision | Round | Commit | Status | Notes |
|---|---|---|---|---|---|
| 1 | Bug 1: `cross_sectional_ic.py` ICIR formula `mean/std×√N` → standard `mean/std` (was t-stat, not ICIR) | 1-2 | `b023ba4` | active | √N ≈ 28 inflated every factor past 0.15 threshold |
| 2 | Bug 2: populate `feature_importance` in `StockRanker.train_fast` (was silent zero) | 1-2 | `b023ba4` | active | Audit's "REAL CONTRIBUTOR" was permanently broken until this fix |
| 3 | New helper `mp/ml/ic_utils.py` consolidates IC math | 2 | `b023ba4` | active | Single source of truth; `blend_regime_sweep.py` switched to import |
| 4 | DryRunBroker docstring clarifies "NOT a backtest shadow / NOT for PnL reconciliation" + connect() warning | 2 | `b023ba4` (warn_once added `26010bf`) | active | Prevents PnL reconciliation misuse |
| 5 | `WF_FEATURE_PRESET` env hook + `mp/ml/feature_presets.py` (W0/W1/W2 frozen) | 5-7 | `0cba000` | active | Replaces source-mutation experiments with named presets |
| 6 | Add `W_BASELINE` preset = frozen 64-feature FACTOR_COLUMNS snapshot | 7-8 | `26f7d6c` | active | Reproducibility anchor against FACTOR_COLUMNS drift |
| 7 | Walk-forward A/B (W_BASELINE/W0/W1/W2) determines `FACTOR_COLUMNS` 64 全量 > any precomputed subset | 11-14 | (experiments only) | active | Production CURATED 32 → 64 decision |
| 8 | Ranker default `feature_cols` fallback → `list(FACTOR_COLUMNS)` | 15 | `a3cb98c` | active | Behavior-contract change for `BlendRanker()` / `StockRanker()` defaults |
| 9 | `CURATED_COLUMNS` deprecated in `mp/ml/dataset.py` (HEAD restored to 23-item version + header) | 15-16 | `05be047` | active (deprecated tag) | Working-tree 32-version WIP discarded — caused 80-line loss (see #11) |
| 10 | Production `.lgb` retrained on FACTOR_COLUMNS 64 | 14 | `89515cb` | active | First time production benefits from #8/#9 (Sharpe ~1.53 at this point) |
| 11 | Q16 lesson: `git checkout HEAD --` destroyed 80 lines of prior-session `dataset.py` WIP (incl. winsorize) | 16-17 | (incident) | rule | "Always `git diff` before destroying uncommitted content" — permanent rule, see docs/TODO.md |
| 12 | Recovery: `EXCESS_CAP=0.50` winsorize cherry-picked from prior-session Claude Code transcript | 20-22 | `1674e69` | active | +0.37 Sharpe on hs300+zz500 (conditional on 64-feature; reversed on 28-feature, see #14) |
| 13 | Production `.lgb` retrained with winsorize active — Sharpe 1.90 bit-perfect reproduced | 22 | `5be2856` | active | Net P0+P1+P2 lift ≈ +0.47 Sharpe (single-seed; ensemble path was stale, see #15) |
| 14 | Conditional reversal documented: winsorize HELPS 64-feature / HURTS 28-feature; same pattern for max_drawdown_20d | 21 | (finding) | active | Feature-contribution is conditional on baseline; see TODO multi-counterfactual P2 |
| 15 | Ensemble deprecated: `data/ensemble/` was stale 32-feature CURATED, daily_report preferred it over single BlendRanker → hid all P0/P1/P2 gains from production | 25-26 | `mv data/ensemble → data/ensemble.deprecated_*` (dir-move only; HEAD daily_report has no ensemble path) | active | EnsembleBlendRanker class WIP kept in working tree with DEPRECATED docstring + num_feature gate; HEAD stays clean |
| 16 | Re-baseline `BASELINE.md` + `framework_evaluation.md` to hs300+zz500 + 64+winsorize numbers (1.90 era), keep zz500-era snapshots tagged | 23 | `a947303` | active | β admonition + α `<sub>` tags + explicit per-event attribution table |
| 17 | P2-#1 audit methodology rewrite: add `mp/ml/wf_gate.py` mini walk-forward Δ Sharpe gate | 25-28 | `c9c3415` | active (standalone module only) | Module kept; wiring into audit reverted (see #18) |
| 18 | P2-#1 wf_gate integration into audit: REVERTED after calibration failed | 30 | `26010bf` (reverts `e71b722`) | superseded | wf_gate LOO from 64-feature ≠ W1/W2 28-feature counterfactual; no shared ground truth |
| 19 | P3-1a: StockRanker fallback (`model.lgb`) retrained to 64-feature + winsorize | 32 | `7079b5f` | active | 60d StockRanker (`model_60d.lgb`) was deterministically byte-identical to 89515cb so no commit needed |
| 20 | P3-1b: framework_evaluation.md §3 counterfactual specification section + docs/TODO.md cleanup (close 2 P2s + add 2 P3s + add P4) + decision_log update (this row) | 32 | `34a270a` (eaa4cc9 amended) | active | TODO.md now reflects current state with status tags |
| 21 | Bug discovered in P3-1a: `update_production_models()` clobbers blend_*.lgb when `RANKER_KIND=stock`; manually rolled back via `git show 5be2856:...`; new P3 TODO opened | 32 | (rollback only, not committed) | rule | Permanent rule #4 added: "any script that retrains data/*.lgb must `cp` backup first" |
| 22 | P3-1c: `update_production_models()` 3-way dispatch — `ranker_is_blend` saves walk-forward, `ranker_20d is None` (`--update-only`) refreshes via train_fast, else SKIP + warn-loudly | 34 | `14f7dbc` (amended from `37ebfa8` for loguru `%s`→`{}` fix) | active | Regression-tested by rerunning `RANKER_KIND=stock`: blend_*.lgb bytes 81620/250686 unchanged ✓. `--update-only` mode still uses train_fast (separate follow-up, not silent-clobber risk) |
| 23 | P3-1d (β0 spike): 3-seed sweep (42/43/44) reveals seed-stability spread 0.23 (tier 3 per round-35 table). seed 42/43 cluster 1.90/1.89, seed 44 outlier 1.67. BASELINE.md ★ table keeps 1.90 (production realized truth, deterministic LGBM_SEED=42) + adds seed-stability caveat subsection with 3-seed table. New P3 TODO for seed 44 attribution | 36 | `b73834a` | active | Advisor declined (α accept 1.82±0.13 as figure) and (β) pad-to-5-seed; chose (ε) keep 1.90 + caveat. n=3 sufficient to surface seed 44 issue without committing to its root cause |
| 24 | P4-1A (seed 44 catalyst attribution): per-month NAV breakdown locates **2023-03 single-month +17pp** as the main fluke; top 3 months only 19% of total gap, top 10 only 37.7% — mixed (not pure fluke, not uniform structural). 2020-2022 NAV ratio stable ~1.22 (structural ~0.05-0.08 Sharpe), 2023-03 catalyst jumps it to 1.40+, compounding amplifies to 1.50× by 2024-2025. BASELINE.md "Single-month catalyst attribution" subsection + TODO update | 39 | `f6dc5f4` | active | Production seed=42 unaffected (deterministic). Stock-level attribution of 2023-03 deferred — required only if future seed-switch / multi-seed work |
| 25 | P4-1C (production monitoring wiring): minimal threshold-breach alerts wired into weekly walk_forward → Feishu via new `mp/monitor/threshold_alert.py` (Sharpe / annual_return / max_drawdown only, 2 levels each); `send_model_update_report` injects "🚨 RED" / "⚠ YELLOW" block when `bt_metrics` breaches BASELINE §4.1 thresholds. Other indicators (win rate, IC health, style drift, §4.2-4.4) remain manual review as documented | 39 | `12c477e` (self-referential; final post-amend may differ by 1 amend) | active | 9/9 mock-breach tests pass. Wire pattern: alert dispatch wrapped in try/except so report-send never breaks even if monitor module errors. Source-of-truth: `mp/monitor/threshold_alert.py` docstring + BASELINE §4.1 cross-ref |
| 26 | P5-A-light: docstring clarifies YELLOW/RED thresholds in `mp/monitor/threshold_alert.py` are **absolute pain levels**, NOT σ-grounded against weekly walk-forward time-series. Cross-seed σ ≠ weekly drift σ (type error to mix). Proper grounding deferred (P5-A-mid candidate, ~4-6 hr, not scheduled). BASELINE §4.1 caveat block cross-ref | 41 | `7026b82` | active | Rebuts external-reviewer concern that "RED Sharpe 0.9 = -7σ → never fires" by clarifying which σ is right |
| 27 | P5 chain (rounds 41-44): cron fix + dead-man-switch. (i) docs/cron_setup.md tracks production crontab source-of-truth; old `--update-only` entry triggered P3-1c residual train_fast weak-blend bug every Friday — must be replaced with full `walk_forward_backtest.py`. (ii) `scripts/monitor/weekly_heartbeat.py` new independent script: reads `backtest_history.json` mtime, > 7d12h YELLOW / > 14d RED / missing RED, sends Feishu via `daily_report.send_to_feishu` with NO `walk_forward_backtest` import (dependency-loop safe per round-43). 10/10 tests pass. **code-close** but production deploy needs **user manual apply** of the new crontab and weekly_heartbeat schedule (macOS FDA blocked Claude shell from `crontab` modification) | 43 | `f5b5255` + `_THIS_FOLLOWUP_` | active | Until user runs the 2 terminal commands, Friday cron still hits the P3-1c residual `--update-only` bug. After apply, P4-1C threshold alerts also start firing weekly for the first time (the `--update-only` path never called `send_model_update_report`) |
| 28 | P6-A3: regression test `tests/test_model_60d_feature_count.py` locks `data/model_60d.lgb` num_feature=64. Closes long-deferred P3 TODO ("60d StockRanker fallback consistency") and guards against silent feature-count regression if any future cron path retrains on stale CURATED 32 | 47 | `80f8a64` | active | pytest.skip if `.lgb` absent (bootstrap-friendly); fails loudly otherwise with pointer to docs/dialog round 46 |
| 29 | P6-A2: `walk_forward_backtest.py --update-only` deprecated via explicit `raise SystemExit(...)` with 3-key-point stderr message (why deprecated / replacement cmd / commit ref). argparse flag kept so old scripts get a clear error instead of cryptic "unrecognized argument"; help text marked `[DEPRECATED 2026-05-24 P6-A2] DO NOT USE.` Closes the P3-1c residual `train_fast` weak-blend path (IC ≈ -0.005 vs walk-forward +0.038) that would otherwise re-introduce silent production clobber if anyone ever called `--update-only` again | 47 | `feac3c6` | active | Latent bug path explicitly killed at boundary; full walk-forward retrain is now the only sanctioned model-update entry point |
| 30 | P6-X2: trading-day-aware heartbeat. New `mp/data/trading_calendar.py` (164 LOC) centralizes `is_trading_day` (moved verbatim from `paper_trade.py:646` — preserves 2026-04-30 ZZ500-EOD-probe history context) and adds `trading_days_between(start, end) -> int` (closed interval; module-level akshare-cal cache; weekday-count fallback). `weekly_heartbeat.py` thresholds switched from wall-clock 7d12h/14d to trading-day-aware >5/>10 (kills holiday false-positives across CNY / National Day). 21-day wall-clock RED safety net retained so calendar-API failure can't suppress real "no cron for 3 weeks" alerts. 29/29 tests pass (12 new + 17 updated heartbeat tests) | 47 | `610e466` | active | Original 2026-04-30 ZZ500-EOD-probe bug fix context preserved in new module docstring per round-47 "don't delete history" instruction |
| 31 | P6-X1: cron drift detect. New `scripts/monitor/cron_drift_detect.py` (307 LOC) daily 07:00 cron. SHA256-compares the live `crontab -l` output to the fenced ```cron``` block under "## Current crontab" in `docs/cron_setup.md`. Normalizes by stripping blank lines + full-line comments + trailing whitespace before hashing (cosmetic edits don't false-positive). Fail-fast on missing anchor / missing fence (silent-fallback "" would defeat the script). Defensive subprocess.run(timeout=10) handles macOS FDA prompt + crontab binary missing + nonzero rc. 25/25 tests pass | 47 | `bdc8a89` | active | Catches the failure mode that already happened in P5-B: docs updated, manual apply forgotten, production schedule silently drifting. Independent fail domain (07:00, after 06:00 heartbeat) so a drift-detect crash doesn't take the heartbeat out |
| 32 | P6-X3: paper_trade vs walk_forward Sharpe drift monitor. New `scripts/monitor/paper_trade_drift_detect.py` (399 LOC) weekly Sat 06:30. Computes annualized rolling 20d Sharpe from `state.json::nav_history` and compares to `backtest_history.json[-1]::bt_metrics.sharpe_ratio`. Initial thresholds YELLOW \|Δ\| > 0.5, RED \|Δ\| > 1.0 + paper Sharpe < 0. Cold-start floor N=15. Skip if walk_forward stale > 21d (heartbeat would already have alerted). 24/24 tests pass | 47 | `b46f2e3` | superseded (initial thresholds) | Closes "execution drift unmonitored" gap (slippage / cash leakage / fill timing / QMT disconnect). Initial 0.5/1.0 thresholds superseded by row 33 (P7-α σ-anchor fix) |
| 33 | P7-α: P6 closeout + X3 σ-anchor fix + advisor rule #6. (i) X3 thresholds loosened 0.5/1.0 → 1.0/1.5; root cause documented in `paper_trade_drift_detect.py` module docstring "THRESHOLD CALIBRATION HISTORY" section: round-47 X3 spec used cross-seed σ ≈ 0.13 as scale anchor for paper_trade *time-series* rolling Sharpe σ — different distribution, different magnitude. Same anchoring error advisor had just caught engineer making in P5-A-light (round 41) failed to transfer to advisor's own spec writing. Caught by external reviewer in P6 evaluation pass. (ii) New permanent rule #6: σ-anchor cross-check before scale-matching thresholds. (iii) TODO.md closes P3 60d / P3-1c --update-only / P5 cross-chain monitoring gaps (cron + execution + holiday); 3 new P8 tickets opened (real σ grounding / alert channel diversification / 2023-03 catalyst stock-level investigation) | 48 | `638726c` | active | Initial 0.5/1.0 thresholds (row 32) superseded by 1.0/1.5; full statistical grounding deferred to P8 (8-12 week empirical NAV backfit, or walk_forward synthetic simulation) |
| 34 | P7-β STOP + Walk B spike: P7-β intended to regenerate `walk_forward_result.md` to reproduce production Sharpe 1.90, but the run produced **1.22** instead. Per advisor's STOP rule (round 49: "数字不是 1.90 立刻停下报回来"), engineer halted before commit and investigated. Findings: (i) `.lgb` byte-identical to `5be2856` ✓; (ii) walk_forward code 5be2856..HEAD only wrapper changes ✓; (iii) wf_cache pre-dates 5be2856 by 1 month ✓; (iv) `data/external` doesn't enter walk_forward panel ✓; (v) **smoking gun**: `backtest_history.json` HEAD already contained 10 entries with Sharpe spanning **[1.15, 1.90]** — same code, same data, same `LGBM_SEED=42`, walk_forward was **nondeterministic** (cross-thread LightGBM gradient/histogram + Python set hash). The "1.90 production truth" was a single cherry-picked rerun (lucky tail) out of N=10. Walk B spike (round 50 spec) tested three flags together: LightGBM `deterministic=True / num_threads=1 / force_row_wise=True` + `PYTHONHASHSEED=0` + already-sorted universe codes → two fresh-process reruns produced byte-identical metric blocks (Sharpe both **1.20**, `backtest_history.json` md5 identical). Deterministic baseline = 1.20, not 1.90. | 49-51 | (no commit — spike only; closure in row 35) | superseded by row 35 | Per rule #4 `cp`-backed up all `.lgb` files before run (4 files in `data/*.lgb.pre_p7bspike_20260524_2237`). Per rule #1 saved 1.22 forensic snapshots to `/tmp/p7b_post_run_1.22.{md,json}`. Production `.lgb` never modified — deterministic gate via `--skip-update` honored throughout. Triggered new permanent rule #7 (deterministic verification) |
| 35 | P7-γ: deterministic re-baseline 1.20 closeout (single commit). (i) `mp/ml/model.py::StockRanker.train_fast` LightGBM `deterministic=True num_threads=1 force_row_wise=True` promoted from env opt-in to **in-code default**, with `WF_NONDETERMINISTIC=1` env escape hatch for legacy reproduction; (ii) `scripts/walk_forward_backtest.py` emits stderr WARNING if `PYTHONHASHSEED != 0` at startup; (iii) `mp/ml/model.py::StockRanker` class docstring gains `PRODUCTION ARTIFACT PROVENANCE (P7-3)` section explaining the existing `.lgb` is the nondet lucky-tail sample (intentionally NOT retrained per advisor + user round-51 decision); (iv) `mp/monitor/threshold_alert.py` docstring gains `THRESHOLD ANCHOR STATUS (P7-3 update)` explaining 1.4/0.9 anchors lost meaning post-deterministic re-baseline but remain AS-IS per operator-pain semantics (awaits operator re-anchoring per new P8 ticket); (v) `data/reports/BASELINE.md` ★ table downgrades 1.90 row to `<sub>` and adds new P7-3 ★ row at Sharpe **1.20 / annual 38.74% / Max DD -32.74%**; new "Deterministic Baseline History (P7-3)" section provides full narrative; seed-stability sub-section marked superseded; (vi) `data/reports/walk_forward_result.md` regenerated under deterministic setup; (vii) new permanent **rule #7** (deterministic vs nondeterministic claims) added to docs/TODO.md with rule #2 cross-link update; (viii) two new P8 tickets: multi-seed ensemble + operator re-anchor threshold_alert | 51 | `_THIS_COMMIT_` | active | Single commit closes P7 chain (rounds 41-51, 11 rounds). Production `.lgb` deliberately NOT retrained — both lucky-draw and deterministic-draw are single samples; real remediation is P8 multi-seed ensemble. Forward: weekly walk_forward now byte-perfect reproducible at Sharpe 1.20 (will trip YELLOW threshold every Friday until operator re-anchors). |

## Status meanings

- **active** — currently in HEAD, intended to stay
- **active (deprecated tag)** — present in HEAD but marked deprecated; do not extend
- **rule** — process rule, not a code change
- **superseded** — overruled by a later decision; kept for git history
- **reverted** — removed from HEAD by a later commit

## P9 chain · winsorize A/B re-evaluation (2026-05-26)

### Triggering claim (P9-0)

Advisor baseline: "winsorize lift +0.37 Sharpe (OLD seed 42 = 1.54 / NEW seed 42 = 1.20)" — used as starting point for P9 revert chain. Later proven phantom (see Catch #8 below).

### Final finding

N=3 deterministic A/B (seeds 42/43/44, default `walk_forward_backtest.py` path = `RANKER_KIND=stock` / StockRanker / `fwd_ret` label, post-commit `6eef98e`):

| Config | seed 42 | seed 43 | seed 44 | mean (N=3) | std |
|---|---:|---:|---:|---:|---:|
| OLD (winsorize OFF, `EXCESS_CAP=999.0`) | 1.20 | 1.29 | 1.06 | 1.183 | 0.117 |
| NEW (winsorize ON, `EXCESS_CAP=0.50`) | 1.20 | 1.29 | 1.06 | 1.183 | 0.117 |
| OLD − NEW | 0.00 | 0.00 | 0.00 | 0.00 | — |

annual / vol metrics also byte-identical OLD vs NEW for all 3 seeds. Winsorize on/off has **no effect** under StockRanker walk_forward path.

`deterministic=True, PYTHONHASHSEED=0, LGBM_SEED ∈ {42,43,44}, num_threads=1, WF_FEATURE_PRESET=W_BASELINE`. EXCESS_CAP env honored confirmed via `grep -c Winsorized` == 0 for OLD runs vs == 1 for NEW runs.

### Permanent rules / catches added

- **Catch #7** (engineer): `EXCESS_CAP` env var override was silently ignored before commit `6eef98e` — `mp/ml/dataset.py:198` had `EXCESS_CAP = 0.50` hard-coded with no `os.getenv` reader. Round 60 spec attempted to disable winsorize via env; all 4 runs ended up with default winsorize on. Detected via byte-identical compare of intended-OLD vs intended-NEW runs (identical Sharpe/annual/vol = override ignored).
- **Catch #8** (advisor): P9-0 "OLD seed 42 = 1.54" not reproducible. Public retraction of the baseline number. Hypothesis (II) data-refresh ruled out (parquet files don't enter walk_forward training pipeline per `grep` of `fund_flow|margin|northbound` in dataset.py/model.py). Hypothesis (IV) — data entry error — likeliest, since `540630d` already locked "P7-γ deterministic re-baseline 1.20" which is inconsistent with P9-0 "OLD=1.54".
- **Rule #9** (new permanent rule): any env var / CLI flag override used for A/B testing must verify the override is actually consumed before reporting deterministic numbers. 3-tier verification:
  1. **grep behavior log** for the override's downstream side-effect (e.g. `grep Winsorized` should be empty when `EXCESS_CAP=999.0`).
  2. **byte-identical compare** two runs across different configs — same metrics = override ignored (strong signal).
  3. **code audit** for `os.environ.get` / `argparse` chain reading the override.

### Open question (P10 candidate)

walk_forward (default `RANKER_KIND=stock`) measures StockRanker on `fwd_ret` label. **Production** uses BlendRanker on `excess_ret` label (`scripts/paper_trade.py:56`, `data/blend_primary.lgb`, `data/blend_extreme.lgb`) — winsorize **is** active on production training data. P9 finding "winsorize no-op" therefore only proven for the measurement path, NOT for production. Measurement-to-production gap is the P10-1 candidate chain (queued, see below).

### Decision: no production change

- `data/blend_primary.lgb` / `data/blend_extreme.lgb` / `data/model.lgb` / `data/model_60d.lgb` **not retrained** (P9-0 phantom; winsorize finding only covers measurement, not production)
- Threshold alerts (`mp/monitor/threshold_alert.py`) **not re-anchored** (baseline 1.18 still in YELLOW=0.9 / RED=0.5 alert region — pre-P9 status preserved)
- γ live-trading path **unblock**: previous "winsorize is known worse" pause reason is now invalid. β-prep commits (65fe669 / 659c26b / f3e7055) provide fidelity test + emergency liquidate; β-3 user-action (Windows VNC QMT-paper, 1-case Approach B) remains the next gate.

### Commits

- `6eef98e` — P9-2: EXCESS_CAP env-readable for A/B testing (`mp/ml/dataset.py` +1 line `import os` +1 line `os.getenv` body). 1 commit total for P9 code change.

### Audit trail (rounds 60-63, docs/dialog/to_engineer.md + to_advisor.md)

| Round | Trigger | Engineer action |
|---|---|---|
| 60 | Advisor B+ spec (4 deterministic runs) | Ran Run 1/2/3; detected Catch #7 via Run 1 ≡ Run 3 byte-identical; halted Run 4; wrote emergency report |
| 61 | Advisor ACK Catch #7 + chose (B) env fix | Commit 6eef98e; ran 4 new runs (A/B/C/D); detected Catch #8 (OLD seed 42 ≠ 1.54); wrote 6-number table report |
| 62 | Advisor ACK Catch #8 + chose (α)(γ) closure | Ran NEW seed 42 v2 verify — byte-identical to OLD seed 42; wrote P9-close report |
| 63 | Advisor ACK P9 close + decision_log spec | This decision_log section + P10-1 queued section (THIS COMMIT) |

## P10-1 candidate · measurement-to-production gap (queued)

### Problem

walk_forward Sharpe baseline = 1.18 (StockRanker, `fwd_ret` label) does not directly proxy production behavior. Production uses BlendRanker on `excess_ret` label with winsorize active. Winsorize impact on the BlendRanker path is untested — P9 closed without measuring it.

### Proposed minimal P10-1 (N=3 deterministic, walk_forward)

- Run `RANKER_KIND=blend EXCESS_CAP=999.0 LGBM_SEED=42/43/44` (3 runs, no-winsorize blend)
- Run `RANKER_KIND=blend` (default `EXCESS_CAP=0.50`) `LGBM_SEED=42/43/44` (3 runs, winsorize blend)
- 6-number table: OLD blend Sharpe vs NEW blend Sharpe
- If significant spread → winsorize **is** material on the production-path measurement
- If byte-identical → winsorize is a no-op irrespective of ranker (stronger conclusion than P9)

### Time estimate

6 runs × 8 min ≈ 48 min walk_forward runtime + 10 min report. ~1 hour wall-clock.

### Not blocking γ

γ uses `paper_trade.py` as production-path measurement (BlendRanker live). P10-1 is supplementary walk_forward coverage of BlendRanker. Two paths independent.

### Defer trigger

Queued, not started. Will activate after β-prep finishes (user QMT-paper Approach B) or by explicit advisor green light. P9 chain is closed.

## P10 chain · BlendRanker A/B winsorize lift confirmed (2026-05-26)

### Trigger

Activated by user round 66 ACK after reading `data/reports/framework_evaluation.md` (P9 chain post-mortem). Engineer (C) grep confirmed `RANKER_KIND=blend` switches `scripts/walk_forward_backtest.py:721-741` to `BlendRanker(primary_label="excess_ret")` — winsorize impact would actually be visible (unlike P9 StockRanker measurement which proved no-op).

### Final finding

N=3 deterministic A/B (seeds 42/43/44, `RANKER_KIND=blend` walk_forward, post-`6eef98e`):

| Config | seed 42 | seed 43 | seed 44 | mean (N=3) | std |
|---|---:|---:|---:|---:|---:|
| OLD (winsorize OFF, `EXCESS_CAP=999.0`) | 1.54 | 1.52 | 1.61 | 1.557 | 0.047 |
| NEW (winsorize ON, `EXCESS_CAP=0.50`) | 1.90 | 1.89 | 1.67 | 1.820 | 0.130 |
| NEW − OLD spread | +0.36 | +0.37 | +0.06 | **+0.263** | — |

3/3 seed directionally consistent (NEW > OLD). **Winsorize HELPS BlendRanker** (opposite direction to P9-0's original "hurts" claim — see Catch #8 revision below). Production `data/blend_primary.lgb` is the better config (NEW seed 42 = 1.90 is the production-realized number).

annual / vol / max_dd matrix:

| 指标 | OLD seed 42 | OLD seed 43 | OLD seed 44 | NEW seed 42 | NEW seed 43 | NEW seed 44 |
|---|---:|---:|---:|---:|---:|---:|
| annual_return | 52.90% | 52.90% | 54.05% | 60.42% | 60.32% | 51.66% |
| annual_volatility | 34.38% | 34.88% | 33.59% | 31.85% | 31.86% | 30.93% |
| max_drawdown | -39.08% | -35.34% | -26.47% | -36.30% | -33.31% | -38.16% |

Seed 44 anomaly: NEW − OLD spread = +0.06 (vs ~+0.37 for seeds 42/43). NEW config cross-seed σ = 0.130 (3× OLD's 0.047). Seed 44 retains its outlier behavior previously documented in P3-1d (row 23) and P7-β (row 34).

### Catch #8 revision (P10-1 finding)

P9-CLOSE (row 36 in `docs/decision_log.md ## P9 chain` section) flagged "P9-0 OLD seed 42 = 1.54 not reproducible / phantom (likely IV data entry)". **Wrong**. P10-1 reproduces OLD blend seed 42 = 1.54 exactly. The phantom was the **comparison frame**, not the data:

- P9-0 "OLD seed 42 = 1.54" was BlendRanker OLD config — verified reproducible at P10-1
- P9-0 "NEW seed 42 = 1.20" was StockRanker NEW config — verified at P9-CLOSE as `RANKER_KIND=stock` deterministic baseline
- Computing spread (1.54 − 1.20) attributed the difference to winsorize but actually mixed two changes: (i) ranker_kind: blend→stock, (ii) winsorize: off→on. Real attribution requires isolating one variable.

Catch #8 corrected wording: **P9-0 baseline was correct data; the error was apples-to-oranges comparison across different ranker paths.** Not a data entry phantom.

### Permanent rules / catches added

- **Catch #10** (advisor, P10-1): P9-0 conclusion was wrong because two config variables changed simultaneously (ranker_kind + winsorize). The implied spread therefore can't be attributed to either variable alone. Same class of error as P5-A-light σ-anchor type-mismatch (row 26): the methodology was unsound, not the data.
- **Rule #10** (new permanent rule, candidate per advisor round 67): A/B tests must hold all other variables constant and the report must explicitly list the "holding constant" clause. Multi-variable spread cannot be attributed to a single variable. Companion to Rule #9 — where #9 verifies that the override is consumed, #10 verifies that nothing else changed.

### Decision: no production change, no threshold re-anchor

- `data/blend_primary.lgb` / `data/blend_extreme.lgb` retained at current winsorize-ON state (verified +0.26 mean Sharpe better than winsorize-OFF). No retrain.
- Production-realized Sharpe 1.90 (NEW seed 42) is within N=3 distribution top: distribution mean 1.82, worst seed 1.67. Not a lucky tail outside the cone.
- Threshold anchors (`mp/monitor/threshold_alert.py`) unchanged: 0.9 YELLOW / 0.5 RED still pre-P9 status, and 1.67 (worst-seed) >> 0.9 keeps RED dormant.
- γ live-trading path: winsorize is **helpful**, not "known worse"; β-prep commits (65fe669 / 659c26b / f3e7055) + β-3 user-action (Windows VNC QMT-paper) remain the gating sequence.

### Open question (P10-2 candidate, queued)

Default `RANKER_KIND` in `scripts/walk_forward_backtest.py` is `stock`. Production uses `blend`. Future walk_forward A/B work risks repeating Catch #10 (measurement layer mismatch) unless either:

- (a) Default `RANKER_KIND` flipped to `blend` so weekly walk_forward measures production-path Sharpe, OR
- (b) New `--ranker-kind` CLI flag forces explicit declaration (no silent default mismatch)

P10-2 candidate: pick one of (a)/(b), commit it, update weekly cron + threshold anchors accordingly. ~30 min wall. Queued.

### Commits

- `_THIS_COMMIT_` — P10-CLOSE: 6-run BlendRanker A/B + winsorize lift +0.26 + Catch #10 + Rule #10. Decision log update only; no code change.

### Audit trail (rounds 66-67)

| Round | Trigger | Engineer action |
|---|---|---|
| 66 | Advisor green light (after user reads `framework_evaluation.md`) | Ran 6 BlendRanker A/B (OLD/NEW × seeds 42/43/44), verified env per Rule #9 |
| 67 | Engineer reports +0.26 spread + direction reversal of P9-0 + Catch #8 partial retraction | Advisor (a) decision: production correct, P10-CLOSE, write Catch #10 + Rule #10 |
| 68 | This commit | Decision log update (~40 lines added to ## P10 chain section) |

## P10-2 chain · measurement default + threshold re-anchor + Rule #11 (2026-05-26)

### Trigger

User round 70 follow-up: 5 nuances after reading P10-1 outcome.
1. seed 44 NEW Sharpe lift (1.61→1.67) is 100% vol-compression (annual went DOWN); winsorize on seed 44 is vol-smoother only, not alpha-adder. NEW config cross-seed σ (0.130) is ~3× OLD (0.047).
2. BASELINE.md has been walked back 3 times (1.90 P2 → 1.20 P7-γ → 1.18 P9-CLOSE) — each time the measurement layer changed but the doc lagged. Need definitive N=3 BlendRanker distribution lock.
3. Root cause: `RANKER_KIND` default = `"stock"` in `scripts/walk_forward_backtest.py:148`. Weekly cron measures StockRanker (1.18), but production loads BlendRanker (1.82). All threshold anchoring + report narrative read the wrong measurement.
4. `framework_evaluation.md` retraction (advisor `a60ab4c` footer) needs a stronger conclusion-header marker.
5. New permanent **Rule #11** required: measurement ranker must equal production ranker.

### Action (single bundle, 1 commit)

- **(A) walk_forward default fixed**: `scripts/walk_forward_backtest.py:148` `RANKER_KIND` default changed from `"stock"` → `"blend"`. Single line + comment cross-ref. Weekly cron from now on measures the same path as production.
- **(A-verify)** Re-ran walk_forward with no `RANKER_KIND` env to confirm default works: Sharpe 1.90 / annual 60.42% / vol 31.85% — byte-identical to P10-1 NEW seed 42 ✓ → P10-2 fix produces the production-realized baseline as default measurement.
- **(B) `data/reports/BASELINE.md` ★ table rewritten** for the 3rd time: replaced single-point 1.20 row with N=3 BlendRanker distribution (1.90 / 1.89 / 1.67, mean 1.82, std 0.13). Annual / vol / max_dd 3×3 matrix included. Old Deterministic Baseline History sections preserved (1.90 lucky tail + 1.20 P7-γ re-baseline rationale) — they are historically valid but superseded.
- **(B-bis) BASELINE §4.1 threshold row** updated: Sharpe baseline `1.90 → 1.82 (N=3 mean; 1.67 worst-seed)`, YELLOW `1.4 → 1.0`, RED `0.9 → 0.5`.
- **(C) `mp/monitor/threshold_alert.py`** YELLOW Sharpe re-anchored `0.90 → 1.00` (per N=3 worst-seed 1.67 anomaly threshold). RED stays `0.50` (severe degrade ≈ worst-case / 3). MaxDD thresholds unchanged.
- **(C-test) `tests/test_threshold_alert.py`** anchor lock test rewritten: `test_thresholds_anchored_to_120 → test_thresholds_anchored_to_p10_distribution`. Asserts YELLOW Sharpe = 1.00 + RED Sharpe = 0.50 per P10-2 round 70. 10/10 tests pass.
- **(D) `data/reports/framework_evaluation.md`** added "🚨 retraction notice" block at the head of "## 十、结论" pointing readers to P10-1 reversal before they read the now-superseded narrative.
- **(E)** This decision_log section (THIS COMMIT).

### Rule #11 (new permanent)

walk_forward measurement's `RANKER_KIND` MUST equal what production (`paper_trade.py`, `daily_report.py`) actually loads. Current production = **single BlendRanker** (since `data/ensemble/` is empty; `paper_trade.log` `Using single BlendRanker (no ensemble found)`). Current measurement default (post-P10-2) = BlendRanker. ✓ aligned.

**If future work re-enables the EnsembleBlendRanker path** (re-populating `data/ensemble/seed_X/blend_*.lgb`), walk_forward must add an equivalent `RANKER_KIND=ensemble` mode (or in-script averaging) to keep alignment. The 2026-05-24 `data/ensemble.deprecated_*` directory move documents one direction of this; the inverse must trigger Rule #11 re-check.

**Until alignment is verified**, any A/B / regression / drift report must prefix:

> *(measurement ranker = X, production ranker = Y, results not directly comparable; see Rule #11)*

Companion to Rule #9 (env consume verify) and Rule #10 (A/B single-variable isolation). The pattern: catches #7 / #8 / #10 all rooted in **measurement layer != claimed measurement layer**; Rule #11 makes the alignment explicit and checkable.

### Catch #11 (advisor, P10-2 round 70 衍生)

P10-1 NEW seed 44: annual 51.66% (lower than OLD seed 44 annual 54.05% by 2.4pp), vol 30.93% (8% compression vs OLD's 33.59%). Sharpe lift 1.61→1.67 (+0.06) is **entirely vol-denominator compression**, not return improvement. Winsorize on seed 44 only smooths the ride; on seeds 42/43 it does both (return + vol).

NEW config cross-seed σ (0.130) is roughly 3× OLD's (0.047). Seed 44 outlier behavior is consistent with the P3-1d β0 spike (BASELINE seed-stability caveat, row 23, 36) and P7-β nondeterminism investigation (row 34) — same model, same seed family, different sample-level interaction with the rare 2023-03 catalyst (row 24).

**Implication**: production locks `LGBM_SEED=42 = 1.90` Sharpe — the top of the N=3 distribution. If a future panel/multi-seed averaging exercise picks any seed in the 42/43/44 family, ~33% probability of landing near 1.67 (worst-seed). 1.67 is still > OLD's mean 1.557, so winsorize is net-positive; but the realized 1.90 is not robust to seed-resampling without ensemble averaging.

### Decision: lock new measurement default + new thresholds; no production .lgb change

- `data/blend_*.lgb` retained at current state (P10-1 already confirmed; reaffirmed by P10-2 verify run byte-perfect to NEW seed 42 = 1.90)
- Weekly cron now measures the production path by default (no `RANKER_KIND` env needed)
- Threshold alerts re-anchored against the N=3 worst-seed (`1.67 → YELLOW 1.0 (anomaly: below worst-seed normal by 0.67)`; `RED 0.5` ≈ severe degrade)
- γ live-trading path: same as P10-CLOSE — winsorize HELPS (not "known worse" or "no-op"); β-prep landed; β-3 user-action is the remaining external gate

### Commits

- `_THIS_COMMIT_` — P10-2: walk_forward default RANKER_KIND=blend + BASELINE.md 3rd rewrite + threshold_alert re-anchor + framework_evaluation retraction + Rule #11 + Catch #11. Single bundle commit covering all 5 actions A-E + test update.

### Audit trail (round 70)

| Round | Trigger | Engineer action |
|---|---|---|
| 70 | Advisor 5-item bundle spec (P10-2/3) | (A) 1-line default fix + verify byte-identical, (B) BASELINE.md 3rd rewrite, (C) threshold_alert re-anchor + test update, (D) framework_evaluation retraction marker, (E) this decision_log section |

## P11 candidate · intraday re-prediction at 14:30 (queued, 2026-05-27)

### Trigger

User noted (2026-05-27 08:5x, pre-9:25 first-live-execute) that executing at
T 09:30 captures the full ~1% buy-side limit buffer cost (limit = close × 1.01)
while a 14:30 intraday re-prediction would in principle improve accuracy: by
then ~90% of T's price action has happened, so the residual 19-day prediction
is presumably cleaner than the 20-day prediction made T-1 17:30.

### Hypothesis

If we re-score the universe at T 14:30 using intraday features (9:30→14:30
OHLCV in addition to T-1 close), and execute via 集合竞价收盘 (14:55-15:00)
撮合 at close, three things happen:

1. **Eliminate 1% limit buffer cost** — 集合竞价 撮合 fills at actual close
   price regardless of submitted limit (per A股 rules).
2. **Reduce prediction noise** — model conditions on more recent data.
3. **Sacrifice T-day drift** — top-N picks have empirical T-day positive drift
   (~+0.3-0.5%); executing at close means we miss this.

Net Sharpe impact: unknown without walk_forward verification.

### Why not just flip a flag

Current BlendRanker trained on T-1 close features predicting "T-1 close →
T+19 close" returns. Using it on T 14:30 features is OOD; both the feature
distribution (intraday VWAP vs EOD close) and the prediction target horizon
(20d → ~19.0625d) differ from training distribution.

This is a *new model*, not a different execution time on the same model.

### Required work (~2-4 weeks)

1. Build intraday feature pipeline (mp/ml/intraday_features.py): compute
   features from 9:30-14:30 OHLCV per stock per day, ~800 stocks.
2. Train BlendRanker variant on "T 14:30 features → T 14:30 → T+19 close"
   labels. Walk-forward 2020-01 ~ 2025-12.
3. Compare 9:30-entry walk_forward Sharpe (P10-CLOSE baseline 1.82) vs
   14:30-entry walk_forward Sharpe. Decision rule:
   - If 14:30 Sharpe > 9:30 Sharpe by ≥ +0.15 → migrate to 14:30 execution.
   - Else → archive this branch, document negative result.
4. ECS data freshness: 9:30-14:30 intraday OHLCV must be retrievable on ECS.
   Sina/EM rate-limiting on Aliyun/火山云 IP is currently brittle; will need
   either proxy / multi-source / pre-fetch cache.
5. New ECS Windows Task Scheduler entry: T 14:30 trigger → score + plan →
   14:50 dispatch → 14:55 集合竞价 撮合.

### Open questions

- Does T-day drift dominate the limit-buffer saving? Empirically unknown.
- 集合竞价收盘 depth: large-cap stocks have decent depth at close, but
  small caps may have thin 撮合; 4000+ share orders may only partial-fill.
- Walk_forward Sharpe of 19d-horizon prediction vs 20d? Probably similar
  but verify.

### Status

**Queued. Not started.** Activated by explicit advisor green light AFTER:
- 1+ week of stable 9:30-entry live execution (β-3 N=1 already passed)
- BlendRanker 9:30 baseline confirmed in production (not just walk_forward)

### Cross-references

- Limit buffer slippage discussion: docs/dialog/ (around 2026-05-27 09:00)
- Live execute baseline: `scripts/ecs_auto_execute.ps1` (commit 9b3c275)
- B' ×1.03 hybrid as short-term alternative (not yet implemented as of
  2026-05-27 09:00)

## P11-3 chain · intraday_blend research → MIGRATE confirmed (2026-05-27)

### Trigger

User round 73 activated the P11 candidate (see "P11 candidate" entry above)
right after first successful live execute at T 09:30 on account 8886933837
(7/7 fills, ¥104,798 → ¥104,154 with ¥841 actual slippage savings vs feared
1% buffer cost).  Advisor scoped research into 5 phases P11-1 → P11-5 (~2-4
weeks total estimate) with the strict decision rule:

- mean N walk-forward intraday Sharpe ≥ mean EOD Sharpe + **0.15** → migrate
- ∈ [-0.10, +0.15] → archive intraday model, queue P11-4 (real intraday data)
- < -0.10 → kill P11

### Phase summary (P11-1 → P11-3 N=6)

| Phase | Round(s) | Commit(s) | What landed |
|---|---|---|---|
| **P11-1** schema | 73-74 | `26e90e6` + `132ae27` | `mp/ml/intraday_features.py` (207 L) + 20 tests; 67-feature contract (later bumped to 68) |
| **P11-1 bump** overnight_gap | 75-76 | `6ea01c9` | INTRADAY_EXTRA_COLUMNS bumped 3→4 (added `overnight_gap` clean / no-leak) |
| **P11-2** train | 75-76 | `20c4b8e` + `372f8d6` | `scripts/train_intraday.py` (302 L) + `data/intraday_blend_{primary,extreme}.lgb`. 786k rows × 5yr, 5.1 min total. Primary IC 0.008 flagged as concern. |
| **P11-2b** control + spike | 77-78 | `dbc71c0` + `fb6c4c5` | `--no-extras` flag; control IC -0.003 (extras NOT the root cause); `morning_vwap_dev` ↔ `vwap_dev` collinearity (delta -581 gain); extras +91% for extreme model; `docs/p11_4_spike.md` (5-day budget design) |
| **P11-3 N=3** | 79-80 | `4d64de2` + `f4e3c5f` | `scripts/walk_forward_backtest.py` extended with `RANKER_KIND=intraday_blend`; 3-seed runs (42/43/44); N=3 mean Sharpe Δ=+0.13, borderline ±0.15 threshold |
| **P11-3 N=6** | 81-82 | `2574a85` + `9dff3d4` | 6 additional seed runs (45-47 × {blend, intraday_blend}); N=6 mean Δ=+0.132; per-seed directional 5/6 positive |
| **P11-3 MIGRATE** | 83 | `_THIS_COMMIT_` | This audit-trail entry |

### Final N=6 results (research conclusion)

**Hold-constant (Rule #10)**: window 2020-01 ~ 2026-04, hs300+zz500 universe,
Top-K=10, conviction sizing, EXCESS_CAP=0.50 winsorize ON, deterministic
config (PYTHONHASHSEED=0, num_threads=1, force_row_wise=True), --skip-update
(Rule #4).  Only diff: RANKER_KIND.

| Config | mean Sharpe | mean Annual | mean Vol | mean Max DD |
|---|---:|---:|---:|---:|
| EOD blend (baseline) | 1.858 | 61.18% | 32.88% | -36.79% |
| Intraday_blend | 1.990 | 62.19% | 31.29% | -29.04% |
| **Delta** | **+0.132** | +1.01 pp | **-1.59 pp 优** | **-7.75 pp 优** |

Per-seed Sharpe delta {42, 43, 44, 45, 46, 47} = {+0.05, +0.03, +0.31, +0.35,
+0.08, -0.03} — **5/6 directional positive** (only seed 47 marginally
negative).  Bootstrap 95% CI on mean delta: [+0.04, +0.23] (0 excluded with
margin).  SE ≈ 0.053.

### Borderline-case secondary rule (round 81)

Mean delta +0.132 falls in [+0.10, +0.15] borderline range.  Advisor round 81
added a borderline-only secondary rule:

> If mean Δ Sharpe ∈ [+0.10, +0.15] AND per-seed directional ≥ 5/N positive → migrate.

This rule is **one-shot for this case** and is NOT promoted to a permanent
Rule.  Round 83 explicitly declined to add a "Catch #12" — the current
strict-threshold logic + measurement discipline is unchanged; only this
specific borderline judgement is documented here.

5/6 directional ≥ 5/6 → **MIGRATE triggered**.

### Sanity checks (Q1 + Q2 from round 81)

**Q1 fold-level MDD: broad-based, not regime-concentrated**.  Pooling
seeds 45-47 monthly returns:

- EOD blend worst-8 monthly returns include 2024-01 -14.72% AND -13.05%
  (two seeds both hit by the same Jan-2024 China-US rate stress).
- Intraday_blend worst-8 has **none** of the 2024-01 outliers, max is
  2022-12 -12.13%.
- Months < -10%: EOD 10/225 vs intraday 7/225 (30% reduction).
- Months < -14%: EOD 3/225 vs intraday 0/225.

Confirms MDD improvement spans multiple regimes, not concentrated on one
specific stress period.

**Q2 fold 3 IC=0.001 outlier (round 80)**.  Seed 43 intraday_blend retrain
schedule from log: fold 3 = **2020-03-02 retrain** = A股 COVID-19 crash +
US-market circuit-breakers (4× halts that month).  LightGBM cross-sectional
IC drops to near-zero on regime-shift training data; the very next fold
(2020-04-01) recovers to primary IC=0.070.  Standard regime-shift artifact,
not a schema bug.

### Walk-forward per-fold IC trajectory (sanity)

Seed 43 intraday_blend: primary IC across folds = {0.063, 0.059, 0.001 (Q2),
0.070, 0.068, 0.089, 0.075, 0.083, ...}, **mean ≈ 0.06-0.07**.  Production
blend's typical per-fold primary IC is 0.03-0.05.  So intraday_blend's
walk-forward IC is **slightly better than production**.

This confirms the round-78 hypothesis that **train_fast single-split IC=0.008
was metric noise**, not a model-quality issue — the model is healthy at
every fold except the one COVID-crash outlier.

### Production state — UNCHANGED

- `data/blend_primary.lgb` + `data/blend_extreme.lgb` timestamps remain
  May-24 17:45 (Rule #4 satisfied across all 12 walk-forward runs in this
  chain via `--skip-update`).
- Production daily_report.py / paper_trade.py still load EOD blend.
- `data/intraday_blend_{primary,extreme}.lgb` exist as parallel artifacts
  (P11-2 commit `20c4b8e`) but no production loader code path references
  them yet (P11-5 scope).

### Rule compliance summary

| Rule | Status |
|---|---|
| #1 stage diff | ✓ — every commit in this chain staged explicit files only |
| #4 production .lgb sacrosanct | ✓ — `data/blend_*.lgb` timestamps unchanged across the whole chain |
| #7 deterministic N report | ✓ — all 12 walk-forward runs N=1 seed each, fully reproducible config |
| #9 env/flag verify | ✓ — RANKER_KIND, EXCESS_CAP, LGBM_SEED, PYTHONHASHSEED all consume-verified per run log |
| #10 single-variable A/B | ✓ — hold-constant clause explicit in rounds 80 + 82, only RANKER_KIND differs |
| #11 walk_forward = production | ✓ schema-level — both walk_forward fresh-train and production-loaded `data/intraday_blend_*.lgb` use INTRADAY_FEATURE_COLS 68-col schema. Internal walk-forward-fresh-train vs P11-2-single-train difference is the same kind of internal precedent set by P10-2. |

### Decision: MIGRATE confirmed; P11-5 user-gated

P11-5 (live execution path change, T 14:30 entry replacing T 09:30) is a
user-gated decision because it touches real-money production.  Three forks
require user choice (round 83 outline):

1. **Cutover style**: full cutover vs 2-week paper-trade parallel (engineer
   leans paper-trade — walk-forward T_close × scaling proxy for fill is not
   the same as real 集合竞价收盘 撮合).
2. **Model version**: deploy current EOD-proxy-trained `data/intraday_blend_*.lgb`
   immediately on paper, or wait for P11-4 (real intraday data, ~5 working
   days per spike, then retrain).  Engineer leans "deploy proxy now, queue
   P11-4 behind paper-trade".
3. **14:50 broker order type**: limit vs market; if limit, anchor at
   T_close ± buffer or live last_price?  (P11-5 spec scope.)

### Commits (audit trail, this chain only)

- `26e90e6` P11-1 module + 20 tests (3 extras initial)
- `132ae27` round 74 report
- `6ea01c9` P11-1 schema bump (overnight_gap → 4 extras = 68 cols)
- `20c4b8e` P11-2 training + `data/intraday_blend_*.lgb` artifacts
- `372f8d6` round 76 report
- `dbc71c0` P11-2b control + `docs/p11_4_spike.md`
- `fb6c4c5` round 78 report
- `4d64de2` P11-3 walk_forward extension + 3-seed runs (42/43/44)
- `f4e3c5f` round 80 report
- `2574a85` P11-3 N=6 expand (3 additional seeds 45/46/47 × 2 configs)
- `9dff3d4` round 82 report
- `_THIS_COMMIT_` decision_log P11-3 chain MIGRATE confirmed (audit trail)

### Audit trail (rounds 73-83)

| Round | Trigger | Engineer action |
|---|---|---|
| 73 | Advisor P11-START + scope | ACK + read P11-1 spec |
| 74 | (engineer) | P11-1 schema + tests `26e90e6` |
| 75 | Advisor schema bump + P11-2 green light | (engineer reply only) |
| 76 | (engineer) | Schema bump `6ea01c9` + P11-2 train `20c4b8e` |
| 77 | Advisor "B route: root-cause IC 0.008" | (engineer reply only) |
| 78 | (engineer) | P11-2b control `dbc71c0` (IC also low → extras not root) |
| 79 | Advisor "(I) P11-3 first, Option C" | (engineer reply only) |
| 80 | (engineer) | P11-3 N=3 walk_forward `4d64de2`, delta +0.13 borderline |
| 81 | Advisor "(C) N=6 expand + secondary rule" | (engineer reply only) |
| 82 | (engineer) | P11-3 N=6 `2574a85`, delta +0.132, 5/6 directional → MIGRATE per secondary rule |
| 83 | Advisor "MIGRATE confirmed + hold for user" | (engineer reply only — this commit) |

