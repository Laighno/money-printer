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

## P2 — dataset.py prior-session 80 行改动遗失（2026-05-24 收尾时发现）

**问题**：2026-05-23（上一会话）在 `mp/ml/dataset.py` 工作树里有 80 行
改动（包括 `_add_industry_relative_features` / `_align_fundamentals_to_dates`
等可能的因子计算改进），从未 commit。2026-05-24 `collab/advisor-dialog`
分支 Q16 commit 05be047 用 `git checkout HEAD -- mp/ml/dataset.py` 整文件
还原 → 那 80 行永久销毁。

**量化影响**：用同样 W_BASELINE preset + LGBM_SEED=42 + RANKER_KIND=blend
跑 walk_forward：
- 改动 still in working tree（round-11，2026-05-24 12:04）：Sharpe **1.90** /
  年化 60.41% / Max DD -36.30%
- 改动被销毁后（round-16，2026-05-24 13:04）：Sharpe **1.53** /
  年化 52.49% / Max DD -38.49%
- 净损失：**-0.37 Sharpe / -7.9 pp 年化 / -2.2 pp Max DD**

**已知线索**：
- prior-session 在 `CURATED_COLUMNS` 注释里写了 "Added 2026-05-23 after
  permutation audit"
- 涉及 `pb_ind_rank` / `pe_ind_rank` / `mom_20d_ind_rank` 这些因子的
  计算实现 / PIT 对齐 / 行业相对 ranking 的某些细节
- HEAD 的 FACTOR_COLUMNS 里这些 factor 名都存在，所以 prior-session 改的
  是**计算逻辑**，不是新增 factor

**待办**：
- 通过 git log / 上一会话 transcript（如果存档）尝试找回那 80 行
- 如果找不回 → 重新实现：基于 round-11 vs round-16 跑出来的 Sharpe 差异
  反推可能的改进方向
- 实现后跑 walk_forward 验证 Sharpe 是否回到 1.90 量级

**关联**：这个修复路径可能与"audit 方法学评估"重叠 —— 如果未来用
walk-forward Δ 作为新 audit gold standard，扫一遍 `_add_industry_relative_features`
/ `_align_fundamentals_to_dates` 等可能被改良的函数，借机找回部分丢失逻辑。
两条 P2 一起做可以省工。

**参考**：docs/dialog/ rounds 16-17（gap 发现 + 不可逆销毁确诊）

---

## 教训（永久规则）

任何 `git checkout HEAD -- <file>` / `git reset --hard` / `git stash drop`
等 "销毁工作树未提交内容" 的决策**必须先做 `git diff HEAD -- <file>` 看
当前 diff 是多少行 + 是什么**，确认无价值再执行。Q16 教训。
