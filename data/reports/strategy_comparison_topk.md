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
