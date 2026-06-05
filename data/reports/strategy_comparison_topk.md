# 交易策略优化对比报告 — TOP_K + Sizing

_生成 2026-06-05, advisor strategy-comparison_

## 方法

- **回测引擎**: `scripts/walk_forward_backtest.py` (月度滚动重训, expanding window, 每日 rebalance)
- **窗口**: 2022-01 → 2026-05 (4.4 年, 含 2022 熊市 / 2023-24 牛 / 2025-26)
- **Universe**: HS300 + ZZ500 (~800, 历史 PIT 成分股快照, 无 survivorship bias)
- **基准**: ZZ500 total +11.55% / annual +2.67%
- **现 prod 配置**: TOP_K=10, POSITION_SIZING=conviction

## 主结果 — conviction × equal × TOP_K grid

| TOP_K | CONVICTION (Sharpe/Total/MaxDD) | EQUAL (Sharpe/Total/MaxDD) |
|---:|---|---|
| **10** (现 prod) | **0.84 / 150.1% / -30.6%** | — |
| 15 | 1.11 / **192.7%** / -27.0% | 1.05 / 110.4% / -23.7% |
| 20 | 1.17 / 174.3% / -24.6% | 1.10 / 101.0% / -19.2% |
| 25 | 1.29 / 186.1% / -20.3% | **1.41** / 130.5% / -11.7% |
| 30 | 1.34 / 177.8% / -17.1% | 1.40 / 117.3% / **-9.1%** |
| 40 | 1.28 / 155.3% / -17.9% | — |

## 其它 sizing / overlay (TOP_K=10 baseline)

| 配置 | Total | Sharpe | MaxDD | 结论 |
|---|---:|---:|---:|---|
| conviction (prod) | 150.1% | 0.84 | -30.6% | baseline |
| equal | 104.0% | 0.82 | -26.0% | 收益降, SR 持平 |
| inverse_vol | 54.3% | 0.53 | -24.9% | ❌ 砍收益一半 |
| vol_target | 38.7% | 0.58 | -19.1% | ❌ 砍收益 |
| stop-loss -15% | 153.0% | 0.85 | -30.2% | ~无效 |
| stop-loss -10% | 137.5% | 0.80 | -30.6% | ❌ 更差 (whipsaw) |
| industry-cap 3 | 150.5% | 0.84 | -30.6% | 不 binding |

## 科学结论

1. **现 prod 的 conviction sizing 正确** — 每个 TOP_K 上 conviction 收益碾压 equal (top-25: 186% vs 130%). 信号加权抓到真 alpha. Risk-parity (inverse_vol/vol_target) 砍收益一半, 因为压低动量赢家 (A股动量是主要 alpha 源).

2. **现 prod 的 TOP_K=10 严重次优** — 这是最大优化空间.
   - 分散 (10→25/30): Sharpe 单调上升 (0.84→1.34), 回撤单调下降 (-30.6%→-17.1%), 收益维持高位.
   - 拐点 top-30 (conviction Sharpe 峰值 1.34), top-40 开始稀释 alpha.

3. **分散 vs 信号加权各管一头**:
   - conviction = 收益引擎 (信号加权保住高 return)
   - 分散 = 风险引擎 (摊薄单股 idiosyncratic risk → SR↑ DD↓)
   - "top-25/30 好" = 两者叠加.

4. **stop-loss 无效/有害** — 月度 rebalance 已出清亏损票, 额外止损只造成 whipsaw.

5. **industry-cap 不 binding** — model 的 top-K picks 本身已跨行业分散, 很少单板块 >3 只.

## 逐年稳定性 (top-25 conviction vs top-10 prod)

| 年份 | top-10 | top-25 | Δ |
|---|---:|---:|---:|
| 2022 (熊) | +3.3% | +11.2% | +7.9pp |
| 2023 (强牛) | +36.9% | +26.3% | -10.7pp |
| 2024 | +46.4% | +52.6% | +6.2pp |
| 2025 | +20.5% | +25.2% | +4.7pp |
| 2026 (partial) | +0.2% | +6.5% | +6.3pp |

top-25 跑赢 **4/5 年**. 唯一输的 2023 强牛仍 +26%. 2022 熊市分散优势最大 (下行保护). 优势跨 regime 稳定, 非单年驱动.

## 推荐

**改 prod TOP_K 从 10 → 25, 保持 conviction sizing**:
- 收益 150% → 186% (+36pp), Sharpe 0.84 → 1.29 (+54%), MaxDD -30.6% → -20.3% (砍 1/3)
- 三维度全面碾压现 prod, 零代码成本 (仅改 TOP_K 参数)
- 跨 4.4 年 (含熊市) + 逐年验证 robust

备选:
- 更厌恶回撤: equal + top-25 (DD -11.7%, Sharpe 1.41) 但收益降到 130%
- 偏稳: conviction + top-30 (DD -17.1%, Sharpe 1.34, 收益 178%)

## 实施注意

- TOP_K=25 时每只 ¥287k/25 = ¥11.5k, 远 > A股最小可行仓 (¥5k), 安全
- 多 15 只持仓 = 更多换手, 但 backtest 已含全套 friction (commission+slippage+stamp)
- 影响 EOD path (daily_report n_recommend). OOS Arm B path 有 ¥20k/日 cap, 单独考量
- OOS (14:30 c2c) 是否也加 TOP_K 待后续验证 (现 sweep 是 EOD n2c path)

---

## 附录: 权重方案 (POSITION_SIZING) 对比 — 2026-06-05

### 背景
现 prod 用 conviction sizing (权重 ∝ 模型预测超额 + 0.5pp floor). 用户问 "6.4%(#1) vs 2.3%(#25) 这个权重合理吗". 跑了 5 个 sizing 方案对比 (TOP_K=25 固定, 同 4.4 年 walk_forward_backtest 引擎).

### 聚合结果 (2022-2026, 4.4yr, TOP_K=25)

| 权重方案 | Total | Annual | Sharpe | MaxDD |
|---|---:|---:|---:|---:|
| conviction (现 prod) | 186.0% | 28.9% | 1.29 | -20.3% |
| rank-decay (1/rank) | 300.2% | 39.8% | 1.27 | **-33.5%** ❌ 太集中 |
| **rank-linear** | 201.1% | 30.5% | **1.40** | -19.8% |
| conviction+6%cap | 153.4% | 25.2% | 1.40 | -18.2% |
| equal | 130.5% | 22.3% | 1.41 | -11.7% |

### ⚠️ 聚合数字误导 — 必须看 t-stat

聚合上 rank-linear (SR 1.40) 看着完胜 conviction (1.29). 但**统计检验显示无显著差异**:

```
月度收益差 (rank-linear − conviction, 52 个月):
  均值:    +0.113%/月 (年化 +1.36%)
  标准差:  1.911%/月
  t-stat:  +0.43       ← |t|>2 才显著, 远不够
  月度胜率: 27/52 = 52% ← 基本随机
  IR:      0.21        ← 弱信号
```

→ **conviction vs rank-linear 统计上无法区分, 差异是噪声**. 聚合 SR 1.40 vs 1.29 是 4.4 年里一两个月差异被复利+波动放大的假象.

分 regime 也无故事: 2022-2023 (熊) rank-linear +0.132%/月, 2024-2026 (牛) +0.097%/月 — 都微正不显著, 连"熊市保护"都站不住.

### 逐年明细 (说明聚合 vs 逐年为何"反转")

| 年 | conv SR | rank-lin SR | 赢家 |
|---|---:|---:|---|
| 2022熊 | 0.54 | 0.78 | rank-lin (+0.24 大) |
| 2023 | 1.32 | 1.43 | rank-lin (+0.11) |
| 2024 | 1.72 | 1.60 | conv (+0.12 小) |
| 2025 | 3.22 | 3.05 | conv (+0.17 小) |
| 2026 | 2.43 | 2.10 | conv (+0.33) |

conviction 赢年份多 (3/5) 但赢得少; rank-linear 赢年份少 (2/5) 但 2022 赢得多 → 聚合被 2022 高波动主导 → 两个指标"反转". 本质都是噪声 (t=0.43).

### 结论

**保持现状 conviction sizing**. 不是因为它"更优", 而是因为**任何合理权重方案 (conviction/rank-linear) 统计上都差不多**.

- ✅ 决定业绩的是 **选股 (模型) + 持仓数 (分散)**, 不是权重精细调法
- ❌ rank-decay (1/rank): 太集中, -33.5% DD, 不可取
- ❌ equal: Sharpe 略高但收益砍半 (130% vs 186%)
- ⚠️ rank-linear / conviction+cap: 跟 conviction 无显著差异, 不值得改真钱

**别再调权重了** — t-stat 证明没区别. 精力放模型质量 + 持仓数.

### 留备用
`walk_forward_backtest.py` 已加 rank_decay / rank_linear / conviction_softcap sizing 模式 (env POSITION_SIZING + SOFT_CAP). 未来如果进入持续熊市 regime, rank-linear 的微弱下行保护可重新评估 (但当前数据不支持改).
