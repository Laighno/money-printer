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

---

## P3 — update_production_models() clobbers blend_*.lgb when RANKER_KIND=stock

**问题**（2026-05-24 P3-1a 发现）：`scripts/walk_forward_backtest.py::update_production_models()`
line 1167-1191 在 `ranker_is_blend == False` 路径下 unconditionally retrains
BlendRanker via `train_fast(ds_20)` on full panel — produces a much worse
model (val IC ≈ -0.005) than walk-forward expanding-window training, and
silently overwrites `data/blend_*.lgb`.

**实测**：P3-1a 跑 `RANKER_KIND=stock ...` 期间，blend_primary.lgb 被
覆盖为 285KB / IC=-0.005 模型（vs P2-verify-1 walk-forward 训出的 81KB /
Sharpe 1.90 模型）。手动 rollback：`git show 5be2856:data/blend_*.lgb >`。

**后果**：任何人跑 `RANKER_KIND=stock` 不带 `--skip-update`（包括
cron / ad-hoc training）都会 silently nuke production blend model。

**待办**：修 `update_production_models()` 在 `ranker_is_blend == False`
路径下：
- 要么 skip blend retrain 完全（保持现有 blend_*.lgb 不变）
- 要么 fail-loudly：`raise RuntimeError("Cannot retrain BlendRanker via
  RANKER_KIND=stock walk-forward path. Pass --skip-update if running stock
  for the StockRanker fallback only.")`

我倾向第一个（skip + warning），与 README / BASELINE.md 当前"blend is the
primary production model"语义一致。**P3 但建议优先做**——production 一行
误操作就 nuke 1.90 Sharpe 模型。

**参考**：docs/dialog/ round 32-33 + commit `7079b5f` 末段披露

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

## P3 — seed 44 BlendRanker outlier 归因

**问题**：2026-05-24 β0 3-seed spike（docs/dialog/ round 36）显示 seed 44
BlendRanker walk-forward Sharpe 1.67 vs seed 42/43 的 1.89-1.90，spread 0.23 / -1.5σ。

**关键观察**：win_rate 几乎不变（51.36% vs 52.28%，spread 0.92pp），但 NAV
compounding 量级差距大（1106% vs 1601%）—— 不是"选股能力不一致"，是"compounding 关键日 hit 与否"
的 lottery-ticket 性质差异。

**production 影响**：production 锁 `LGBM_SEED=42` deterministic，**当前 1.90 Sharpe
是稳定数字**。但如果未来需要换 seed（例如 seed 42 模型 drift），不知道 seed 44
那种 outlier 会不会重现。

**待办**：
- 跑 seed 44 + dump per-month NAV breakdown，看哪几个月（or 哪些 trade）
  拉跨 cumulative NAV
- 比对 seed 42 同月 selection diff
- 如果是 1-2 个 large-loss event → 评估是否风控漏洞
- 如果是 cumulative drift → BlendRanker stability 问题（model architecture issue）

**不阻塞 production**（production 锁 seed=42）。要换 seed / 加 multi-seed 平均
前必须解。

**参考**：docs/dialog/ rounds 35-36（β0 spike 设计 + 数据）

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
