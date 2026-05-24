# 冻结基线（Frozen Baseline）

**定稿日期**: 2026-04-22 · **2026-04-29 更新**：volume 单位 bug 修复 + BlendRanker 完整 walk_forward 验证 · **2026-05-24 re-baseline**：universe 扩到 hs300+zz500 + Bug 1/2 修复 + winsorize 入栈
**状态**: 研究收敛 — 进入"观察 + 监控退化"阶段，主线不再频繁改动。

本文档是单一事实来源（single source of truth）。如果未来出现"是不是该 X"的冲动，先查这里：X 可能已经被否决过。

---

> ## ⚠ 2026-05-14 universe 切换 + 2026-05-24 P0+P1+P2 链条后的关键变化
>
> 本文档下文的数字分两套：
> - **当前 production（hs300+zz500 + 64 features + winsorize + BlendRanker conviction）**：写在表格 ★ 标记的行
> - **历史 zz500 era（pre-2026-05-14，pre-P0/P1/P2 链）**：保留作为对照参考，标 `(zz500 era)` 字样
>
> ### 三段对比（按事件拆分）
>
> | 配置 | 时间 | Sharpe | 年化 | Max DD |
> |---|---|---:|---:|---:|
> | zz500 + Blend conviction + winsorize | 2026-04-29 | 2.01 | 69.84% | -22.74% |
> | hs300+zz500 + 同模型，**无** winsorize（Q16 销毁后） | 2026-05-24 P1 | 1.53 | 52.49% | -38.49% |
> | hs300+zz500 + 同模型，**有** winsorize（P2-fix-1 恢复） | 2026-05-24 P2 ★ | **1.90** | **60.42%** | **-36.30%** |
>
> ### 归因
> - **Universe widening alone**: **-0.48 Sharpe / -17 pp 年化 / +15.8 pp Max DD**（结构性损失，因子在大盘股上信号弱化；amihud_illiq ICIR 从 1.32 → 0.455）
> - **Winsorize 标签去噪**: **+0.37 Sharpe / +8 pp 年化 / -2.2 pp Max DD**（救回 universe 损失的 77%）
> - **净结果**: -0.11 Sharpe / -9.4 pp 年化 / +13.6 pp Max DD
>
> 这不是 bug。universe 扩 60% 是已决策的研究范围调整（覆盖大盘股，paper_trade 也用 hs300+zz500），winsorize 在 64-feature 上是 net positive（与 28-feature 的反向效应不同，见 docs/dialog/ round 21 conditional 分析）。
>
> 详见 docs/dialog/ rounds 9-22（universe widening 归因 + P0/P1/P2 决策链）。

---

## 一、当前生产默认配置

| 参数 | 值 | 位置 |
|---|---|---|
| Universe ★ | **`hs300+zz500`**（~800 只，2026-05-14 起）| `scripts/walk_forward_backtest.py::UNIVERSES` |
| TOP_K | 10 | env `TOP_K` / 默认 10 |
| HORIZON | 20 个交易日 | `HORIZON` 常量 |
| REBALANCE_POLICY | `on_change` | env `REBALANCE_POLICY` / 默认 on_change |
| MAX_WEIGHT_DRIFT | 0.10（仅 drift 模式下有效） | env `MAX_WEIGHT_DRIFT` |
| SLIPPAGE_BPS | 5 | env `SLIPPAGE_BPS` |
| COMMISSION_BPS | 3 | env `COMMISSION_BPS` |
| COST_AWARE_REBALANCE | True | `scripts/walk_forward_backtest.py` |
| **模型（生产）** | **BlendRanker(0.80 primary + 0.20 extreme)** | `mp/ml/model.py` · 训练 label = `excess_ret` |
| RANKER_KIND（walk_forward 验证用） | 默认 `stock`，`blend` 用于验证 BlendRanker | env `RANKER_KIND` |
| **POSITION_SIZING** | **`conviction`**（weight ∝ 模型超额预测） | env `POSITION_SIZING`，2026-04-29 起从 `equal` 切换 |
| 因子集 ★ | `FACTOR_COLUMNS` 全量 64（51 技术 + 6 基本面 + 4 行业相对 + 3 基本面趋势）。固化在 `mp/ml/feature_presets.py::W_BASELINE_PRESET` (sig=3000062054) | `mp/ml/dataset.py` |
| Winsorize ★ | `EXCESS_CAP = 0.50` 对 excess_ret label clip（2026-05-24 P2-fix-1 cherry-pick from prior-session WIP）| `mp/ml/dataset.py::EXCESS_CAP` |
| USE_REGIME_FEATURES | **OFF** (False) | env，保留开关仅作为否定实验 artefact |

### 实测基线绩效

**★ 当前 production：BlendRanker + Conviction + FACTOR_COLUMNS 64 + EXCESS_CAP winsorize**
（2020-01 ~ 2026-04，PIT-clean，hs300+zz500 universe，2026-05-24 P2-verify-1 retrain）

| Metric | Value |
|---|---:|
| 年化收益 | **60.42%** |
| Sharpe | **1.90** |
| Calmar | **1.66** |
| 最大回撤 | -36.30% |
| 年化波动率 | 31.85% |
| 月度胜率 | 52.28% |
| 总收益 | 1601.12% |
| ZZ500 年化 | 6.02% |
| **超额（α）** | **+54.39pp** |

数据源：`data/reports/wf_experiments_20260524/wf_p2fix_20260524_1510.log`
（commit `5be2856`：P2-verify-1）。

> **历史快照（zz500 era, pre-2026-05-14）** —— 保留作为对比参考，不再是当前 production 数字
>
> | Metric | Value (zz500 era) |
> |---|---:|
> | 年化收益 | 69.84% |
> | Sharpe | 2.01 |
> | Calmar | 3.07 |
> | 最大回撤 | -22.74% |
> | 年化波动率 | 34.72% |
> | 月度胜率 | 52.88% |
> | 总收益 | 2294.91% |
> | 超额（α）| +63.82pp |
>
> 数据源：`data/reports/walk_forward_blend_conviction.md`

#### Position-sizing 完整对比 <sub>(zz500 era, pre-2026-05-14)</sub>

> ⚠ 以下表是 zz500 universe 上 5 种 sizing 的完整对比，2026-05-24 后未在 hs300+zz500 上重做。
> conviction 仍是生产默认，但**绝对数字应参考 §一 新表，不要混用**。

| Sizing | 年化 | Sharpe | Calmar | Max DD | Vol |
|---|---:|---:|---:|---:|---:|
| **conviction（生产）** ⭐ | 69.84% | 2.01 | 3.07 | -22.74% | 34.72% |
| equal（旧默认）| 54.25% | 1.88 | 2.21 | -24.59% | 28.81% |
| inverse_vol | 36.71% | 1.49 | 1.94 | -18.96% | 24.68% |
| vol_target (25%) | 27.21% | 1.44 | 1.70 | -15.99% | 18.93% |
| _conviction_oracle_ (LEAK CHECK) | _366.38%_ | _6.77_ | _10.20_ | _-35.93%_ | _54.14%_ |

#### Oracle 泄漏审计（重要）

`POSITION_SIZING=conviction_oracle` 用**真实 forward 20d 收益**当 conviction 权重（intentional leak），衡量"完美预测"的天花板。
- Oracle 366% 年化 vs Conviction 70% 年化 → **5.2× 差距**
- 如果 conviction 在悄悄用未来数据，Oracle 不会跟它差这么多
- → **Conviction 的 +16pp 优势是模型 IC 0.06 真本事，不是泄漏**

> **口径校准**：这是"已经认真测试过的靠谱替代方案里最优"，不等于全空间全局最优。

> **历史口径修正（2026-04-29，zz500 era）**：
> 1. 之前 BASELINE 写的 1.62 Sharpe / 49.34% <sub>(zz500 era)</sub> 是**污染数据 + StockRanker** 的结果
> 2. 之前从未跑过 BlendRanker 的 walk_forward，但 paper_trade / daily_report 一直用 BlendRanker
> 3. 数据修复（volume × 100 单位 bug）后两个模型都重训：StockRanker 1.81/57.10% <sub>(zz500 era)</sub>、**BlendRanker 1.88/54.25%** <sub>(zz500 era)</sub>
> 4. **生产模型最终选 BlendRanker**：风险调整后全面占优
> 5. StockRanker 模型保留作 fallback，但生产只用 BlendRanker
>
> **追加（2026-05-24 P0/P1/P2 链）**：
> 6. **Bug 1**：`scripts/cross_sectional_ic.py` ICIR 公式实际是 t-stat × √N（commit `b023ba4` 修复）
> 7. **Bug 2**：`StockRanker.train_fast` 未 populate `feature_importance`（同 commit 修复）
> 8. **P1**：ranker default `feature_cols` fallback 改 `list(FACTOR_COLUMNS)`（commit `a3cb98c`+`05be047`）。production 实际训练特征集统一到 64-feature 全量
> 9. **P2**：cherry-pick prior-session 的 excess_ret winsorize (EXCESS_CAP=0.50) 回到 dataset.py（commit `1674e69`+`5be2856`）。Sharpe 从 1.53 → **1.90**（hs300+zz500 universe 下；与 zz500 era 的 2.01 仍差 0.11，但 universe 不可逆）

运行记录：
- StockRanker post-fix walk_forward: `data/reports/walk_forward_postfix.md`
- BlendRanker post-fix walk_forward: `data/reports/walk_forward_blend.md`

---

## 二、已否决的方向（不要再尝试，除非有新证据）

| # | 方向 | 为什么否决 | 实验记录 |
|---|---|---|---|
| 1 | **drift_threshold rebalance @ Top-K=10** | 年化 47.76% vs 49.34%，每个维度都差 | `data/reports/walk_forward_drift_10.md` |
| 2 | **Regime proxy 作为 per-date 训练特征（#3a）** | 年化 38.19% vs 48.52%，Sharpe 1.30 vs 1.60（全面恶化）。原因：per-date 常量对横截面排序 0 信息、只能过拟合日期 | `data/reports/ab_regime/{baseline,regime}.md` |
| 3 | **Regime-aware blend 切换（#3b）** | 数学上是 #2 的受限版本，同时 hindsight regime 标签带来 PIT 成本。无独立实验，基于 #2 的结果外推否决 | (衍生结论) |
| 4 | **scripts/blend_weight_sweep 搜最优权重** | PIT-clean 数据下 blend ∈ [0.60, 1.00] 是 flat plateau，差异 <0.3pp — 不是稳定最优点 | `data/reports/blend_weight_sweep.md` |
| 5 | **Extreme head 退役** | 按聚合 IC 看确实弱；但 per-regime 分析在牛市/熊市 ICIR > 1.0，震荡期稀释总体。保留 | `data/reports/blend_regime_sweep.md` |
| 6 | **ml_backtest 作为研究级回测** | 有 survivorship + 缺行业相对特征。已降级为 demo，带 runtime warning | `mp/backtest/ml_backtest.py` docstring |
| 7 | **简单规则做大盘择时空仓信号**（如"距 60 日高点 -15% 且最近 5 日跌 → 清仓"）| 2015-2026 实证：10 次中证500 大跌事件中**6 次卖在 V 型底部**（2020-03 / 2022-03 / 2024-01 / 2024-07 / 2025-01 等），误报率 59.6%。"完美按规则空仓"策略的 MDD **-73.5% 比 buy-and-hold 的 -65.2% 更差**。A 股的 -15% 经常是底部，不是末日信号。**风险管理不通过宏观择时，交给 broker 层止损** | `data/reports/timing_rule_validation.md`（本次内嵌） |
| 8 | **训练 ZZ500 forward 20d 预测模型来替换 trailing 20d**（早期方案 C）| 学术界共识：指数 forward return 预测 IC 通常 < 0.05，OOS 难稳定打败长期均值；同时模型化的"假精确"比"老实给均值"危害更大（用户会"信"它）。改用长期均值常数（0.5%/月）作为绝对收益参考的基准加项 | (基于 Welch-Goyal 2008 共识 + #7 实证) |
| 9 | **Equal-weight, inverse_vol, vol_target 三种仓位方案** | 2026-04-29 全 5 方案 A/B 显示 **conviction 全面胜出**（见 §一 表格）。equal 是次优（Sharpe 1.88）；inverse_vol / vol_target 因削减高动量大幅伤年化（37% / 27%）。详细原因见运行记录。`POSITION_SIZING` env 开关保留可切换，默认 `conviction` | `data/reports/walk_forward_blend{_invvol,_voltarget}.md` |

---

## 三、已验证的稳健性

| 项目 | 结论 | 证据 |
|---|---|---|
| **费率 5→10 bps 敏感性** | 不敏感，差异落在 LGBM 种子噪声（~1% annual）内 | `data/reports/sensitivity/slippage10.md` |
| **Top-K × rebalance policy 矩阵** | Top-K=10 用 on_change 最优；Top-K=30 下 drift 反而更优。**当前 K=10 保持 on_change 不动摇** | `data/reports/sensitivity/topk30_{on_change,drift}.md` |
| **午间报告 PIT** | 无未来数据泄漏。存在 morning-partial vs EOD 分布漂移（解释粤电力A 午后排名跳变），不是 bug | `data/reports/intraday_leak_audit.md` |
| **PIT 护栏测试** | 9 条回归测试，每次运行都要求通过 | `tests/test_pit_filters.py` |
| **大盘规则择时（无效）** | 2015-2026 历史 10 次大跌事件实测，6 次"卖在底"，规则空仓策略 MDD 反比 buy-and-hold 更差 | 见上表 #7 |

## 三-bis、报告显示口径（2026-04-28 起）

**修复前的问题**：旧实现把"模型 forward 20d 超额预测"和"中证500 trailing 20d 实际涨跌"相加（forward + backward 时间窗错配），趋势市里所有票"绝对预测"被基准抬正/压负，导致：

- 大涨时几乎全部建议加仓
- 大跌时几乎全部建议清仓
- 实际"模型相对看空"的票（如海格通信 -1.74% 超额）被显示成 +6.19% → 🟢加仓

**修复后口径**：

| 字段 | 含义 | 用途 |
|---|---|---|
| `rank_pct` | 横截面排名分位（基于 BlendRanker 0.8 排名 + 0.2 排名）| **决策首要依据**（与回测换仓一致）|
| `predicted_excess` | 模型预测 forward 20d **超额**（vs ZZ500）| 决策辅助；suggestion 阈值用它 |
| `predicted_return` | predicted_excess + **0.5% 长期均值常数** | 仅作绝对收益**参考**，不参与决策 |

**suggestion 阈值改为基于 `predicted_excess`**：

| excess | 建议 |
|---|---|
| > +3% | 🟢 加仓 |
| 0% ~ +3% | 🟡 持有 |
| -3% ~ 0% | 🟠 减仓 |
| < -3% | 🔴 清仓 |

**为什么 0.5% 长期均值常数**：中证500 长期年化 ~6%，按 12 个月分摊 ≈ 0.5%/月。这个数字每天都一样、不会被短期行情扭曲，作为"绝对收益参考"诚实可信，但**不假装能预测短期市场方向**。

---

## 四、每周观察指标（看这些，不是去改参数）

每周五 18:00 cron 自动跑全量回测，产出 `data/reports/walk_forward_result.md` + 飞书通知。**每周看以下 4 项，如果任一项触发告警，回到本文档查是否需要重新打开已否决方向**：

### 4.1 性能退化告警

阈值基于 ★ 当前 BASELINE（年化 60.42% / Sharpe 1.90 / DD -36.30%，post-P2 2026-05-24）的 ~50% / 50% / 1.1× 安全裕度（DD 阈值放宽因为 hs300+zz500 固有 Max DD 已经偏深）。

| 指标 | BASELINE ★ | 黄色告警 | 红色告警（即停模拟交易）| 为什么 |
|---|---:|---:|---:|---|
| 最近 12 月滚动年化 | 60.42% | < 30% | < 15% 或 < 3× ZZ500 | 策略核心假设失效 |
| Sharpe（最近 252 日）| 1.90 | < 1.4 | < 0.9 | 风险调整后已无明显优势 |
| 最大回撤（新高）| -36.30% | > -42% | > -50% | 突破 hs300+zz500 实测最差 1.15× / 1.4× |
| 月度胜率（最近 12 月）| 52.28% | < 47% | < 42% | 系统性掉队 |
| paper_trade 1 月累计 | ≈ +4% | < 0% | < -5% | 实盘短期偏差 |
| paper_trade 3 月累计 | ≈ +12% | < +3% | < 0% 或 < ZZ500 | 实盘中期偏差 |
| paper_trade 6 月累计 | ≈ +30% | < +15% | < +5% 或 DD > -32% | 实盘长期偏差 |

### 4.2 模型健康度

| 指标 | 告警阈值 | 为什么这么定 |
|---|---|---|
| 20d primary IC 均值（最近 6 月）| < 0.04 或连续 2 月 < 0 | conviction sizing 对 IC 衰减敏感，IC 0.06 是历史均值，<0.04 表示信号显著弱化 |
| 20d primary IC（每月单期）| < 0 连续 2 月 | 模型方向性翻转 |
| Top-10 precision@K（最近 6 月）| < 0.55 | 回测里 Top-10 月平均命中"跑赢市场" >55% 的票，<0.55 = 选股能力退化 |
| 行业相对 rank 在选 Top-10 时的方差 | 突然变窄（< 0.1）| Top-10 全集中到一行业 = 模型在过度拟合某板块 |
| Primary 与 Extreme 子模型 IC 比较 | Extreme IC 持续 > Primary | 信号在尾部，平均效应失效，conviction 反而有害 |

### 4.3 选股风格漂移

- 候选池是不是越来越集中到某个行业（例如 80% 选股都在新能源）？
- 市值分布是不是越来越偏（全是小票 / 全是大票）？
- 成交额门槛是不是总被突破（选的票流动性越来越差）？

### 4.4 数据源健康

- 飞书"基本面数据大面积缺失"告警次数
- valuation 快照是不是连续几天没更新
- 某个股票长期取不到 bars

---

## 五、允许的小修小补

以下工作**可以做**，不需要重走 A/B 流程，因为只是清理、不影响决策路径：

1. 给 `prediction_diagnostics.py` 剩下 10 个非关键 scoring loops 补 PIT 过滤（机械工作）
2. 清理 `mp/backtest/ic_analysis.py` 里的 RuntimeWarning
3. 补充单元测试覆盖
4. 日志 / 报告格式改进
5. 文档修正

---

## 六、真正要重开研究的触发条件

**只有以下情况出现，才重新打开本文档"已否决"方向或启动大改**：

- 每周观察指标连续 2 周触发告警
- 数据源/市场结构发生根本变化（例如 T+0 开放、做市商制度变化）
- 发现新的、本文档没覆盖过的特征/想法，且有 off-the-shelf 证据证明它在 A 股有效
- 实盘持仓表现连续 3 个月 vs 回测基线偏差 > 15pp

**除此以外：保持不动，继续跑，每周看监控**。

---

## 六-bis、数据层重大修复（2026-04-28）

### 发现：Volume 单位混乱（持续 6 年）

**根因**：`mp/data/fetcher.py` 的两个 endpoint 单位不一致 ——
- `_get_daily_bars_sina` (akshare `stock_zh_a_daily`)：volume = **股**
- `_get_daily_bars_em` (akshare `stock_zh_a_hist`)：volume = **手** = 100 股

`get_daily_bars` 的逻辑是 Sina 优先 → 失败回退 EM。Sina 长期成功率 >95%，所以 daily_bars 表绝大多数行是「股」单位；但 6 年累积有 **419,440 行（16.6%）** 来自 EM fallback，单位是「手」 → 同表里两种单位混杂。

**影响范围**：所有用 volume 列计算的因子 ——
- `vwap_dev` (VWAP 偏离)
- `obv_slope` (OBV 斜率)
- `vol_price_corr` (量价相关)
- `vol_ratio_5_60`, `volume_volatility`, `volume_trend`, `intraday_intensity`
- `amihud_illiq` (流动性) 间接

**用户实际触发**：
- 2026-04-28 11:48 daily_report 显示福田汽车(600166) 模型超额 +14.61% → 排名 #1
- 同一只票，volume 修复后再算 → -2.43% → 排名跌出 Top
- 中间 17pp 差异里大约 12pp 是单位 bug 造成的污染特征，5pp 是真实的"日内 vs 收盘"分布漂移

### 修复（已上线）

1. **endpoint 归一化**：`_get_daily_bars_em` 和 `_get_daily_bars_etf` 在返回前 `volume *= 100`，统一到「股」
2. **存量数据修正**：扫描 daily_bars 表，找出 `amount/(volume*close)` 不在 [0.3, 3.0] 的行，全部 `volume *= 100`，已修正 419,440 行
3. **防御性检查**：`save_bars_upsert` 写入前对每行检查 ratio，明显错单位的拒收并 warning

测试：`tests/test_volume_units.py`（5 条）。

### 这意味着什么对历史回测的有效性

- **`data/wf_cache/factors.parquet` 现在是 stale 的**：基于污染过的 volume 算的因子，已经不能直接使用
- **`data/blend_*.lgb`（生产模型）也是 stale 的**：训练数据有 16% 的 volume 异常
- **过去回测 49.34% 年化 / Sharpe 1.62 是基于污染数据**

理论上需要：
1. 删 `data/wf_cache/factors.parquet`，让 walk_forward 用修复后的 daily_bars 重新计算
2. 重新跑 walk_forward → 重新训练 blend 模型
3. 验证修复后年化是否仍跑赢 ZZ500

实务上：模型从 5 万行（2020-01）成长到 200 万行的过程中，16% 的污染均匀分布，模型大概率已经「学会」对噪声免疫。但**正式做出"用"或"不用"的决策前，必须重训一次**。

**下次（最近的周五 18:00）的 cron `walk_forward_backtest.py --update-only` 自动会用修复后的数据重训** —— 等那次跑完看 IC 和 Sharpe 是否大幅波动。如果差异 <2pp 年化，证明修复后系统仍然 robust；如果差异 >5pp，需要重新评估全部下游决策。

---

## 七、模拟交易模式（2026-04-28 启动）

独立于研究主线的"实盘观察期"工具。`scripts/paper_trade.py` 每个工作日 16:00 自动跑：

| 项目 | 取值 | 备注 |
|---|---|---|
| 起始资金 | 300,000 | |
| 决策口径 | 完全等同 BASELINE §一（Top-K=10 / on_change / cost-aware）| 一致才有意义 |
| 模型 | `data/blend_*.lgb`（生产 BlendRanker） | |
| 执行 | T+1 开盘价（昨日收盘后决策、今日开盘兑现） | 与回测口径一致 |
| 真实成本 | 复用 `mp.account.broker.SimulatedBroker`（5 滑点 + 3 佣金 + 5 印花税） | |
| 数据降级门 | 基本面缺失 >50% → 当日不换仓 | 同 daily_report |
| 大盘择时 / 个股止损 | **不做** | BASELINE ❌#7 |
| 状态持久化 | `data/paper_trade/state.json` | |
| 日报 | `data/paper_trade/reports/{YYYYMMDD}.md` + 飞书 | |
| 起始日 | 2026-04-28（首次产出 10 笔买单计划，2026-04-29 开盘兑现） | |

### Cron 配置

加到 `crontab -e`：

```cron
# 每工作日 16:00 跑模拟交易（A股 15:00 收盘后）
0 16 * * 1-5 cd /Users/laighno/laighno/money-printer && .venv/bin/python scripts/paper_trade.py >> data/logs/paper_trade.log 2>&1
```

注意：脚本内部已检查 `is_trading_day(today)`，遇到节假日会自动跳过，cron 可以放心每周 1-5 触发。

### 监控建议

每周对账：

- `state.json` 里 `nav_history` 末项 NAV 与上周同位 → 周收益
- `trade_log` 长度增长 → 换仓频率
- 每日报告中"今日选 Top-10"的稳定性（同股反复进出 = 模型边际信号弱）

如出现以下任一现象，回到 BASELINE §四"每周观察指标"看是否触发告警：

- 累计 12 月 NAV 退坡 < ZZ500 × 2
- 月度胜率 < 45%
- Top-10 同股反复进出 ≥ 3 次/周（模型边际不稳定）

### 设计取舍

完整设计文档已直接写入脚本 docstring。摘要：

- 不做"模型驱动加减仓" —— 模型训练目标是横截面排序，不是仓位 sizing。任何"模型超额 +5% 就加大仓位"的规则都缺 OOS 支撑、容易引入新过拟合。
- 不做"分批建仓" —— 单股 30k = 远低于 ADV 1%，市场冲击微乎其微；分批只是推迟模型的有效作用。
- 不做"盘中调整" —— intraday 特征 vs 训练用 EOD 特征有分布漂移（见 `intraday_leak_audit.md`），无 OOS 支撑。

测试：`tests/test_paper_trade.py`（11 条），锁定上述决策口径。

---

## 附录：关键路径索引

| 想看什么 | 去哪 |
|---|---|
| 完整研究报告 | `data/reports/framework_evaluation.md` |
| 本次回测结果 | `data/reports/walk_forward_result.md` |
| 历史快照 | `data/reports/backtest_history.json` |
| 午间报告路径审计 | `data/reports/intraday_leak_audit.md` |
| A/B 原始数据 | `data/reports/{ab_regime,sensitivity}/` |
| 模拟交易状态 | `data/paper_trade/state.json` |
| 模拟交易日报 | `data/paper_trade/reports/{YYYYMMDD}.md` |
| PIT 护栏测试 | `tests/test_pit_filters.py` |
| 报告口径护栏测试 | `tests/test_report_display.py` |
| 模拟交易护栏测试 | `tests/test_paper_trade.py` |
| 默认参数定义 | `scripts/walk_forward_backtest.py` 顶部常量块 |
| 模拟交易参数 | `scripts/paper_trade.py` 顶部常量块 |
