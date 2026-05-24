# Project TODO

Active follow-ups from the 2026-05-23 → 2026-05-24 P0/P1/P2/P3 research chain
on branch `collab/advisor-dialog` (commits b023ba4 → 7079b5f). See
[docs/dialog/](dialog/) for the 32-round narrative and [decision_log.md](decision_log.md)
for the single-page decision summary.

---

## ✅ ~~P2 — audit 方法学评估~~ — PARTIALLY RESOLVED 2026-05-24

P2-#1 chain (rounds 25-30) implemented and then reverted `--wf-gate`
integration into audit (commit `26010bf`). Audit verdicts no longer claim
"REAL CONTRIBUTOR" without WF validation — the in-sample-only header warning
now points readers at the multi-counterfactual P2 below. `mp/ml/wf_gate.py`
kept as standalone ad-hoc tool only.

Remaining if anyone wants to revive a binding gate: a true full-LOO
walk-forward (~8.5 hr on 64 features) would give a same-baseline ground
truth, but P2 multi-counterfactual finding suggests the answer would still
be conditional on the baseline. See `docs/dialog/` rounds 25-30.

---

## ✅ ~~P2 — BASELINE.md + framework_evaluation.md re-baseline~~ — RESOLVED 2026-05-24

Commits `a947303` (initial re-baseline) + `P3-1b` (counterfactual specification
section added to framework_evaluation.md §3) close this item. Both docs now
carry hs300+zz500 + 64-feature + winsorize numbers in their ★ tables and
preserve zz500-era snapshots as quoted history with `<sub>` tags.

---

## ✅ ~~P2 — dataset.py prior-session 80 行改动遗失~~ — RESOLVED 2026-05-24

通过 `.claude/projects/-Users-laighno-laighno-money-printer/006a1a75-...jsonl`
transcript 取证找回，cherry-pick `excess_ret` winsorize（commit `1674e69` + `5be2856`）。
P2-verify-1 跑 Sharpe **1.90 bit-perfect 复现** round-11。0.37 Sharpe 完全回收。

详见 docs/dialog/ rounds 20-22。

---

## ✅ ~~P3 — StockRanker fallback `.lgb` 一致性~~ — RESOLVED 2026-05-24

P3-1a (commit `7079b5f`) ran `RANKER_KIND=stock WF_FEATURE_PRESET=W_BASELINE
LGBM_SEED=42 python scripts/walk_forward_backtest.py` and:

- `data/model.lgb` (20d StockRanker fallback) → retrained walk-forward,
  64-feature + winsorize. Sharpe 1.15 (expected — StockRanker < Blend conviction).
- `data/model_60d.lgb` (60d StockRanker) → already 64-feature + winsorize
  in 89515cb; same seed produced byte-identical output, no commit needed.

Production fallback chain (`scripts/daily_report.py:2519`) now uses
consistent 64-feature + winsorize models.

**P6-A3 (commit `80f8a64`)** added `tests/test_model_60d_feature_count.py` as
a regression guard: any future retrain on stale CURATED 32 features will
fail the test in CI rather than silently going into production. See
decision_log.md #28.

---

## ✅ ~~P3 — update_production_models() clobbers blend_*.lgb when RANKER_KIND=stock~~ — RESOLVED 2026-05-24

P3-1c (commit `14f7dbc`) added the 3-way dispatch (`ranker_is_blend` /
`ranker_20d is None` / else SKIP+warn).  P6-A2 (commit `feac3c6`) then
killed the latent `--update-only` path entirely via `raise SystemExit`,
closing the remaining residual clobber surface.

Production blend `.lgb` files are now only overwritten by the legitimate
Friday walk-forward retrain path.  `RANKER_KIND=stock` runs hit the
SKIP+warn branch.  `--update-only` runs hit a clear deprecation error
instead of silently producing IC ≈ -0.005 weak models.

**参考**：docs/dialog/ rounds 34 (P3-1c dispatch) + 47 (P6-A2 SystemExit) +
decision_log.md #22, #29

---

## P2 — feature 评估的多 counterfactual 问题（2026-05-24 P2-#1 calibration 失败发现）

**问题**：同一个 feature 的"贡献"在不同 counterfactual 下方向可能完全相反：

- W2 vs W1（28-feature universe 上加 max_drawdown_20d）：Sharpe **-0.18**
- wf_gate LOO（64-feature universe 上砍 max_drawdown_20d）：Sharpe **+0.11**

方向相反。这不是 bug，是因子贡献 **conditional on feature set 容量**（与
docs/dialog/ round 21 的 winsorize conditional 反转同款）。
amount_ratio 的 wf_gate Δ Sharpe |-0.12| 也超出 0.10 noise band，
没有 universal "noise feature" 的判定。

**implication**：没有 single audit tool 能给出"feature X 是否应该用"
的 universal answer。任何 audit / 校准 / 决策**必须显式声明 counterfactual
baseline**。

**待办（P2 但低优先级，无 production 影响）**：
- 在 `data/reports/framework_evaluation.md` §3 因子表里加一段
  "counterfactual specification" 说明（哪些数字是 LOO from W_BASELINE
  64 / 哪些是 W1-vs-W2 add/remove / 哪些是 univariate 截面 IR）
- 任何未来的 feature 选择决策都必须先定义 "vs which baseline"
- 如果以后真要做"binding 二级 gate"：需要 64-feature full LOO walk-forward
  (~8.5 hr) 作为 ground truth，**没有快捷路径**
- `mp/ml/wf_gate.py` 模块保留作 ad-hoc 工具但禁止当 binding decision
  output（其 docstring 已显式说明）

**参考**：docs/dialog/ rounds 25-30（wf_gate 设计 + calibration 失败 +
Y1 revert 决策）

---

## P3 — seed 切换前的 2023-03 catalyst attribution（updated 2026-05-24 P4-1A）

**已完成（P4-1A round 39）**：β0 3-seed monthly gap breakdown 定位完成。
seed 44 落后 seed 42 = `[2020-2022 累积 structural variance ~0.05-0.08 Sharpe]`
+ `[2023-03 single-month +17 pp catalyst gap]` + `[compounding 在 2024-2025 进一步放大到 ~1.50×]`。
mixed 诊断 (top 3 月只占 19%, top 10 月占 37.7%)。

**剩余 prerequisite**：如未来要换 `LGBM_SEED` 或上 multi-seed averaging，
**必须先做 2023-03 catalyst stock-level attribution**——per-day portfolio
dump 比对 seed 42 vs 44 在那一月的 stock picks，弄清是 specific stock pick
fluke 还是 systematic market regime call。否则可能丢 ~0.10 Sharpe 来源不明的优势。

**预算**：~80 min（脚本 dump + 跑 + 分析）。
**优先级 P3**：production 锁 seed=42 不阻塞 daily ops；要切 seed 才解禁。

**参考**：docs/dialog/ rounds 35-36 (β0 spike) + 38-39 (1A monthly attribution +
BASELINE.md "Single-month catalyst attribution" 段)

---

## P7 (followup) — manual apply of P6-X1/X3 crontab entries

**未完事项（user action required）**：P5 manual apply 已完成（用户已在 Terminal
跑 `crontab /tmp/cron`，验证 2 条 entries 在）。P6 chain 加了 **2 条新 entry**，
需要 user 再做一次手动 apply：

- **Daily 07:00** `cron_drift_detect.py`（P6-X1）
- **Saturday 06:30** `paper_trade_drift_detect.py`（P6-X3）

**Command** — 抄 `docs/cron_setup.md "## Current crontab"` block 到 /tmp/cron
然后在 Terminal.app（**不是** Claude shell — FDA 拦截写操作）：

```bash
crontab /tmp/cron
crontab -l   # verify 4 条 entries 都在
```

**Until apply 完成**：
- `cron_drift_detect` 跑会 RED alert（live missing 07:00 entry vs docs has it）
  — 这是 expected state，确认 X1 wired up end-to-end
- `paper_trade_drift_detect` 没 cron 跑；ad-hoc 跑会"insufficient NAV (N=14)"
  cold-start 信息

**Apply 后**：
- 4 监控 weekly batch（Friday walk_forward / Sat heartbeat / Sat drift / daily cron_drift）
  全部真 firing
- cron_drift_detect 转 OK（live ≡ docs）

**参考**：`docs/cron_setup.md`（source-of-truth）+ docs/dialog/ rounds 41-44 (P5) +
47-48 (P6+P7-α)

---

## P4 — 6 个月后 review CURATED_COLUMNS 是否可物理删除

时间：2026-11-24（6 个月后）。

**问题**：`mp/ml/dataset.py::CURATED_COLUMNS` 当前保留 HEAD 23-item 版本作为
deprecation marker（commit `05be047`）。

**待办**：grep 整个仓库（含 production cron / 备份脚本）确认无 new
caller 显式引用 `CURATED_COLUMNS`，如确认 6 个月内无新调用，物理删
list 本体 + 同时清掉所有 `from mp.ml.dataset import CURATED_COLUMNS`
import 语句。

如有新引用，**评估是否合理**：6 个月内任何 commit 引入 CURATED 都应被
当作"未读 docs/dialog/ 14-15 决策档案"flag 出来质询。

**参考**：decision_log.md #9（CURATED 弃用决策）

---

## P8 — real σ grounding for X3 thresholds（opened 2026-05-24 P7-α）

**问题**：`scripts/monitor/paper_trade_drift_detect.py` 当前阈值
YELLOW=1.0 / RED=1.5（P7-α loosen from 0.5/1.0 after external reviewer
catch — round-47 spec used cross-seed σ ≈ 0.13 as anchor，跟 paper_trade
rolling 20d Sharpe 的 time-series σ 不是同一 distribution）。

**当前 1.0/1.5 是 reviewer 凭直觉建议的 conservative loosen 值**——
比 v1 的 0.5/1.0 更 conservative 但**仍然没有真 grounding**。

**待办**：择一：
- **(a) 8-12 周 paper_trade NAV 实测 backfit**: 等积累 8-12 周
  daily NAV，算 rolling 20d Sharpe time-series σ；用真分布定 ±2σ 黄 / ±3σ 红
- **(b) Synthetic NAV simulation from walk_forward backtest**: 拿
  walk_forward 期间 daily NAV 历史（已存在 backtest snapshot），做 rolling
  20d Sharpe；用其 time-series σ 作 no-execution-drift 假设下的 baseline

**优先级 P8**：当前 1.0/1.5 conservative，false-positive 风险已降；真
grounding 是 1-2 hr 工作但只能在 (a) 路径累积足够 NAV 后才有意义。

**参考**：`paper_trade_drift_detect.py::THRESHOLD CALIBRATION HISTORY` +
docs/dialog/ round 48 (P7-α)

---

## P8 — alert channel diversification（opened 2026-05-24 P7-α）

**问题**：当前 4 监控 (model_update / heartbeat / cron_drift / paper_drift)
**全走 lark-cli 飞书** = single point of failure。如果 webhook secret
revoke / lark-cli binary 缺失 / 网络分区 → 所有 alert silently drop。

**待办**：加 **file-based fallback**：
- 每个 alert 除发 Feishu 外，append 一行到 `data/logs/alerts.jsonl`
  （timestamp + level + source + body hash）
- Cron stderr capture（已经 `2>&1 >> log` 写入）保证最低限度 audit
- Optional: weekly heartbeat 同时检测 `alerts.jsonl` mtime / line count，
  无新 alert 一段时间 + walk_forward 时间正常 → log "alert pipeline 静默"

**优先级 P8**：production 已经有飞书一路 alert + cron log；fallback 是
defense-in-depth 不阻塞 daily ops。

**参考**：docs/dialog/ round 48 (P7-α)

---

## P8 — 2023-03 catalyst stock-level investigation（carry from P3 round 39）

**问题**（P4-1A 已 narrow）：seed 44 落后 seed 42 主要来自 **2023-03
single-month +17pp catalyst gap**。当前 production 锁 seed=42 deterministic，
不阻塞 daily ops，但**如未来切 seed 或上 multi-seed averaging，必须先
做 stock-level attribution**（per-day portfolio dump 比对 seed 42 vs 44
在那一月的 stock picks）。

**预算**：~80 min（脚本 dump + 跑 + 分析）。

**触发条件**：考虑切 `LGBM_SEED` 或上 multi-seed ensemble 时解禁。

**参考**：docs/dialog/ rounds 35-36 (β0 spike) + 38-39 (1A monthly attribution)

---

## 教训（永久规则）

以下规则按发现顺序累积，每条**对应一次具体 incident**，违反成本明确：

1. **任何 "销毁工作树未提交内容" 的决策必须先做 `git diff HEAD -- <file>`**
   看当前 diff 是多少行 + 是什么，确认无价值再执行。Q16 教训
   （成本：0.37 Sharpe 归因丢失，靠 Claude Code transcript 取证才捞回）
2. **任何 "production +X Sharpe" 声明前必须 grep production entry points
   end-to-end**，穷举所有 ranker 类型（StockRanker / BlendRanker /
   EnsembleBlendRanker / TwoStageRanker / 任何 future 类型）。
   P2-7 教训（advisor 漏 grep EnsembleBlendRanker, P0/P1/P2 所有 lift
   差点白做）
3. **任何需要 calibration 的工具，先做 1-2 hr spike 验证 calibration 可行性
   再投入完整实现**。P2-#1 教训
   （成本：~3 hr 工程时间最终 revert，wf_gate.py 留作 standalone tool）
4. **任何重训 / 覆盖 `data/*.lgb` 的脚本必须先 `cp` 备份**，即使是已
   ACK 的"安全"动作。Q16 + P3-1a 教训
   （P3-1a 漏备份 blend_*.lgb，update_production_models bug 直接覆盖了
   1.90 Sharpe 模型，靠 git show recover）
5. **统计判定 framework 不该预设 outlier 位置** — 当 spread > threshold
   时，distribution shape（uniform vs cluster + 1 outlier）必须先看再
   决定哪个 seed/sample 是 lucky/unlucky。round-35 教训
   （advisor 3-档判定表 implicitly 假设 "seed 42 outlier"，round-36
   3-seed 数据出来 seed 44 才是 outlier，险些把 1.90 production figure
   误降。修复：(ε) 路径 keep 1.90 + 加 seed-stability caveat）
6. **σ-anchor cross-check before scale-matching thresholds** — 用某个
   distribution 的 σ 作为 scale anchor 给另一个 distribution 的 threshold
   时，必须 verify 两个 σ 测的是同一 underlying quantity。常见混淆：
     - cross-seed σ (training noise, same data different seed)
     - time-series σ (drift noise, same model different time window)
     - cross-stock σ (dispersion noise, same date different stock)
     - rolling-window realized σ (e.g. rolling 20d Sharpe across NAV time series)
   这些 σ **量级不同 + 不可互替**。Spec docs + commit messages **必须显式
   写出 anchor type**。

   **Why**: P7-α (round 48) 教训。Round-47 X3 threshold spec 用 cross-seed
   σ ≈ 0.13 当 anchor 给 paper_trade rolling 20d Sharpe σ 的 threshold
   （0.5 / 1.0）—— cross-seed σ 测 LGBM seed lottery，rolling Sharpe σ 测
   time-series drift，substitution 给出 wrongly-calibrated (likely too tight)
   阈值。external reviewer P6 review pass 捕获。**同款 anchor 错 advisor
   刚在 P5-A-light (round 41) catch 过 engineer 做**，lesson 没 transfer 到
   advisor 自己的 spec writing。

   **How to apply**: spec 写 "threshold X, anchored to σ=Y" 时必须 name
   what Y measures, what the threshold target measures, 并 assert 两者是
   same distribution type（或 document why they're close enough）。Reviewer
   side: 任何 spec 提 σ anchor 但 missing this clarification → push back
   before implementation.
