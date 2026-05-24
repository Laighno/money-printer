# Project TODO

Active follow-ups from the 2026-05-24 P0/P1 research chain on branch
`collab/advisor-dialog` (commits b023ba4 → 05be047). See
`docs/dialog/to_advisor.md` and `docs/dialog/to_engineer.md` for the
17-round narrative behind each item.

---

## P2 — audit 方法学评估

**问题**：`scripts/feature_importance_audit.py` 基于 80/20 时间分割 val IC drop
判定"REAL CONTRIBUTOR"，与 walk-forward out-of-sample Sharpe Δ 实证不一致
（2026-05-24 W2 实验：audit 推荐保留的 max_drawdown_20d / roe_qoq 实际加回
使 Sharpe -0.18 / Max DD 恶化 3.36 pp）。

**待办**：
- 改造 audit gold standard：用 walk-forward Δ Sharpe 替代 val IC drop，或
  叠加 walk-forward 作为二级 gate
- 重新审视所有 audit 推荐的"REAL CONTRIBUTOR"是否经过 walk-forward 验证
- 在 audit 输出里显式标注"in-sample only"vs"walk-forward verified"

**参考**：docs/dialog/ rounds 12-14（W1/W2 实验和 audit 反向证据）

---

## P2 — BASELINE.md + framework_evaluation.md re-baseline

**问题**：两文件当前数字基于 zz500 universe（2026-05-14 之前），production
已切到 hs300+zz500 但文档未更新。具体过时项：
- BASELINE.md Sharpe 2.01 / 年化 69.84% / Max DD -22.74% → 新 production
  实测 ~1.53 / ~52% / ~-38%（见 `data/reports/wf_experiments_20260524/`
  特别是 `wf_production_retrain_20260524_1305.log`）
- framework_evaluation.md 因子表 ICIR 排序基于错公式（Bug 1，已修
  commit b023ba4）和 zz500 universe，应在 hs300+zz500 + 修复后 ICIR
  公式下重做
- L221 "57 个因子中 24 个" 等所有数字都基于旧 universe / 旧公式

**待办**：
- BASELINE.md L25-65 表格重算（用 `data/reports/wf_experiments_20260524/`
  里的实际数字）
- framework_evaluation.md §3.2 §3.3 因子表重做（用 `mp/ml/ic_utils.py`
  + 新 universe + `data/ic_curated.json` 的修复后 IC 表）
- 加 "zz500 era (pre-2026-05-14)" tag 保留历史数字，不删

**参考**：docs/dialog/ rounds 9-11（universe widening 分析）+
commit b023ba4 (Bug 1 ICIR 修复)

---

## ✅ ~~P2 — dataset.py prior-session 80 行改动遗失~~ — RESOLVED 2026-05-24

通过 `.claude/projects/-Users-laighno-laighno-money-printer/006a1a75-...jsonl`
transcript 取证找回，cherry-pick `excess_ret` winsorize（commit `1674e69` + `5be2856`）。
P2-verify-1 跑 Sharpe **1.90 bit-perfect 复现** round-11。0.37 Sharpe 完全回收。

详见 docs/dialog/ rounds 20-22。

---

## P3 — StockRanker fallback `.lgb` 一致性

**问题**：2026-05-24 P2-fix-1 + P2-verify-1（commit `1674e69`+`5be2856`）
重训了 BlendRanker (`data/blend_*.lgb`) 用新 winsorize 配置，但
`data/model.lgb` (20d StockRanker fallback) 和 `data/model_60d.lgb`
(60d StockRanker) 仍是 P1 close-out commit `89515cb` 时的 winsorize-less 版本。
原因：`walk_forward_backtest.py::update_production_models()` 在
`RANKER_KIND=blend` 路径下只重训 BlendRanker，不重训 StockRanker。

**后果**：
- production 主路径走 BlendRanker，**不受影响**
- 但 daily_report 的 fallback 链路（`scripts/daily_report.py:2519`）触发时
  会用 winsorize-less 的 StockRanker → 行为不一致

**待办**：
- 跑一次 `RANKER_KIND=stock WF_FEATURE_PRESET=W_BASELINE LGBM_SEED=42 \
   python scripts/walk_forward_backtest.py`（不带 --skip-update）重训
  StockRanker `data/model.lgb`。预计 5 min
- 60d StockRanker 当前 walk-forward 无对应 HORIZON 切换，可能需要 ad-hoc
  retrain or 修 `update_production_models()` 让它在任何 RANKER_KIND 下
  都跑一遍 60d 重训

**优先级 P3**：fallback 路径触发概率低，不阻塞日常 production。

**参考**：docs/dialog/ round 22

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

## 教训（永久规则）

任何 `git checkout HEAD -- <file>` / `git reset --hard` / `git stash drop`
等 "销毁工作树未提交内容" 的决策**必须先做 `git diff HEAD -- <file>` 看
当前 diff 是多少行 + 是什么**，确认无价值再执行。Q16 教训。
