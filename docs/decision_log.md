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

## Status meanings

- **active** — currently in HEAD, intended to stay
- **active (deprecated tag)** — present in HEAD but marked deprecated; do not extend
- **rule** — process rule, not a code change
- **superseded** — overruled by a later decision; kept for git history
- **reverted** — removed from HEAD by a later commit
