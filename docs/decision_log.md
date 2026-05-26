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
