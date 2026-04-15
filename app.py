"""Streamlit app - Industry Rotation Signal Dashboard.

This is a DECISION SUPPORT tool, not an auto-trading bot.
It shows you the data signals; you make the final call.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import date

# Ensure project root is on path so scripts/ can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from mp.config import load_settings
from mp.data.fetcher import get_industry_list, get_industry_hist_batch, get_industry_constituents, get_index_constituents
from mp.data.store import DataStore
from mp.factor.screener import score_stocks
from mp.rotation.signals import (
    generate_rotation_signals, generate_reversal_signals,
    generate_deep_value_signals, generate_accumulation_signals,
    generate_rotation_shift_signals,
    calc_momentum, calc_volume_signal, calc_trend_strength,
)
from mp.risk.manager import RiskManager, RiskParams
from mp.backtest.engine import run_backtest, calc_performance, optimize_params
from mp.indicators.technical import compute_all_technical_signals
from mp.indicators.external import fetch_all_external_signals
from mp.screener.signal_screener import screen_stocks, screen_single_stock
from mp.ml.model import StockRanker, BlendRanker
from mp.ml.dataset import build_latest_features, build_dataset, FACTOR_COLUMNS
from mp.regime import RegimeDetector, MarketRegime

st.set_page_config(page_title="Money Printer", page_icon="💰", layout="wide")

settings = load_settings()
store = DataStore(settings.data.db_url)

# --- Sidebar ---
with st.sidebar:
    st.header("控制面板")

    if st.button("刷新行业数据"):
        with st.spinner("拉取行业列表..."):
            boards = get_industry_list()
            board_names = boards["board_name"].tolist()
        with st.spinner(f"拉取 {len(board_names)} 个行业历史K线..."):
            hist = get_industry_hist_batch(board_names, start="20240101")
            store.save_industry_bars(hist)
            st.success(f"已保存 {len(hist)} 条行业K线数据")

    st.divider()
    st.subheader("回测参数")
    rebal_freq = st.selectbox("再平衡频率", ["weekly", "monthly"], index=1)
    top_n = st.slider("持仓板块数", 2, 10, 5)
    stop_loss = st.slider("止损线", 3, 15, 8, 1, format="%d%%") / 100
    trailing_stop = st.slider("追踪止损", 5, 20, 12, 1, format="%d%%") / 100
    max_dd = st.slider("最大回撤熔断", 10, 30, 15, 1, format="%d%%") / 100

    run_bt = st.button("运行回测")
    run_optimize = st.button("自动寻优")

# --- Optimize (rendered above tabs so it's always visible) ---
if run_optimize:
    industry_bars = store.load_industry_bars()
    if industry_bars.empty:
        st.error("无行业数据，请先拉取")
    else:
        with st.spinner("正在遍历参数组合寻找最优解，请稍候（约1-3分钟）..."):
            opt = optimize_params(industry_bars, metric="sharpe")

        bp = opt["best_params"]
        bm = opt["best_metrics"]

        st.header("自动寻优结果")
        st.subheader("最优参数")
        param_cols = st.columns(5)
        param_cols[0].metric("再平衡频率", bp.get("rebalance_freq", "N/A"))
        param_cols[1].metric("持仓板块数", bp.get("top_n", "N/A"))
        param_cols[2].metric("止损线", f"{bp.get('stop_loss_pct', 0):.0%}")
        param_cols[3].metric("追踪止损", f"{bp.get('trailing_stop_pct', 0):.0%}")
        param_cols[4].metric("最大回撤熔断", f"{bp.get('max_drawdown_pct', 0):.0%}")

        st.subheader("最优参数下的回测表现")
        metric_cols = st.columns(5)
        metric_cols[0].metric("年化收益", bm.get("annual_return", "N/A"))
        metric_cols[1].metric("Sharpe", bm.get("sharpe_ratio", "N/A"))
        metric_cols[2].metric("Calmar", bm.get("calmar_ratio", "N/A"))
        metric_cols[3].metric("最大回撤", bm.get("max_drawdown", "N/A"))
        metric_cols[4].metric("胜率", bm.get("win_rate", "N/A"))

        # NAV chart with best params
        best_risk = RiskParams(
            max_sectors=bp["top_n"],
            stop_loss_pct=bp["stop_loss_pct"],
            trailing_stop_pct=bp["trailing_stop_pct"],
            max_drawdown_pct=bp["max_drawdown_pct"],
        )
        result = run_backtest(industry_bars, rebalance_freq=bp["rebalance_freq"], top_n=bp["top_n"], risk_params=best_risk)
        if not result.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result["date"], y=result["nav"], name="Strategy NAV", line=dict(color="#2196F3", width=2)))
            fig.update_layout(title="最优参数净值曲线", template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

            nav = result["nav"]
            peak = nav.cummax()
            dd = (nav - peak) / peak
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=result["date"], y=dd, fill="tozeroy", name="Drawdown", line=dict(color="#F44336", width=1)))
            fig2.update_layout(title="回撤", template="plotly_white", height=250)
            st.plotly_chart(fig2, use_container_width=True)

        # Top 10 combinations
        log = opt["results_log"]
        if not log.empty:
            st.subheader("参数组合排名 Top 10")
            log_sorted = log.sort_values("sharpe", key=lambda s: pd.to_numeric(s, errors="coerce"), ascending=False).head(10)
            log_sorted.columns = ["再平衡", "持仓数", "止损", "追踪止损", "回撤熔断", "Sharpe", "Calmar", "年化收益", "最大回撤"]
            st.dataframe(log_sorted, use_container_width=True)

        st.caption("注意：最优参数基于历史数据，不保证未来表现。多组参数都表现好说明策略鲁棒，只有一组好可能是过拟合。")
        st.divider()

# --- Main ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 行业信号", "📈 回测结果", "🔍 板块详情", "💼 持仓监控",
    "🎯 选股雷达", "🤖 ML模型", "📋 日报中心",
])

# Tab 1: Industry Signals
with tab1:
    industry_bars = store.load_industry_bars()
    if industry_bars.empty:
        st.warning("无行业数据，请点击左侧「刷新行业数据」")
    else:
        STRATEGIES = {
            "🚀 动量追涨": {
                "header": "行业轮动信号面板",
                "caption": "追涨逻辑：动量 + 资金流入 + 趋势确认，买强势板块",
                "func": generate_rotation_signals,
                "score_col": "composite_score",
                "score_label": "综合得分",
                "extra_cols": ["above_ma20"],
                "extra_names": ["站上MA20"],
                "extra_fmt": {"above_ma20": lambda x: "是" if x == 1 else "否"},
                "guide": """**因子权重：** mom_5d(0.25) + mom_20d(0.20) + mom_60d(0.10) + vol_ratio_5d(0.20) + amount_ratio_5d(0.10) + ma_alignment(0.10) + volatility_20d(-0.05)\n**入场条件：** 20日动量>0 + 量比>1.2 + 均线多头 + 认知确认\n**离场：** 20日动量转负 / 量比<0.8 / 均线转空头 / 止损""",
            },
            "🔄 超跌反弹": {
                "header": "超跌反弹信号面板",
                "caption": "抄底逻辑：中期深跌 + 短期企稳 + 资金进场",
                "func": generate_reversal_signals,
                "score_col": "reversal_score",
                "score_label": "反弹得分",
                "extra_cols": [],
                "extra_names": [],
                "extra_fmt": {},
                "guide": """**筛选条件：** 20日动量<-5%（超跌） + 5日动量>0（企稳）+ 量比>1.0（资金进场）\n**因子权重：** mom_5d(0.30) + mom_20d(-0.20) + mom_60d(-0.10) + vol_ratio_5d(0.20) + amount_ratio_5d(0.10) + volatility_20d(-0.10)\n**离场：** 反弹至20日均线 / 5日动量再转负\n**仓位：** 不超过30%""",
            },
            "🕳️ 超跌未涨": {
                "header": "超跌未涨信号面板",
                "caption": "埋伏逻辑：跌过头+卖盘枯竭+尚未反弹，极早期介入",
                "func": generate_deep_value_signals,
                "score_col": "deep_value_score",
                "score_label": "深度价值得分",
                "extra_cols": ["ma60_deviation", "deceleration"],
                "extra_names": ["MA60偏离%", "下跌减速度"],
                "extra_fmt": {},
                "guide": """**筛选条件：** 价格低于MA60超5% + 5日动量≤0（还没涨） + 20日动量<-5%\n**因子权重：** ma60_deviation(-0.30) + mom_20d(-0.15) + mom_5d(0.15) + vol_ratio_5d(-0.15) + volatility_20d(0.15) + deceleration(0.10)\n**认知确认：** 行业基本面未恶化，下跌是情绪超调\n**仓位：** 极轻仓或仅观察，等5日动量转正再加仓""",
            },
            "🔍 主力吸筹": {
                "header": "主力吸筹信号面板",
                "caption": "量价背离：价格不动但成交放大，有人悄悄建仓",
                "func": generate_accumulation_signals,
                "score_col": "accumulation_score",
                "score_label": "吸筹得分",
                "extra_cols": ["vol_price_div"],
                "extra_names": ["量价背离"],
                "extra_fmt": {},
                "guide": """**筛选条件：** |20日动量|<5%（价格横盘） + 量比>1.2（成交放大）\n**因子权重：** vol_ratio_5d(0.30) + amount_ratio_5d(0.25) + vol_price_div(0.20) + volatility_20d(-0.15) + mom_20d(-0.10, abs)\n**认知确认：** 行业有即将释放的利好预期\n**离场：** 放量突破后持有，缩量回落则撤""",
            },
            "🔀 强弱切换": {
                "header": "强弱切换信号面板",
                "caption": "轮动逻辑：从弱变强的第二梯队，资金正在切入",
                "func": generate_rotation_shift_signals,
                "score_col": "shift_score",
                "score_label": "切换得分",
                "extra_cols": ["acceleration"],
                "extra_names": ["动量加速度"],
                "extra_fmt": {},
                "guide": """**筛选条件：** 60日动量排名后50%（之前弱） + 20日&5日动量>0（正在变强） + 加速中\n**因子权重：** mom_5d(0.20) + mom_20d(0.20) + acceleration(0.20) + vol_ratio_5d(0.20) + ma_alignment(0.10) + mom_60d(-0.10)\n**离场：** 进入强势板块Top10后兑现""",
            },
        }

        strategy_mode = st.radio("策略模式", list(STRATEGIES.keys()), horizontal=True)
        cfg = STRATEGIES[strategy_mode]

        st.header(cfg["header"])
        st.caption(cfg["caption"])

        signals = cfg["func"](industry_bars)
        score_col = cfg["score_col"]

        with st.expander("信号解读指南"):
            st.markdown(cfg["guide"])

        st.subheader(f"Top 15")
        base_cols = ["mom_5d", "mom_10d", "mom_20d", "mom_60d", "vol_ratio_5d", "amount_ratio_5d", "ma_alignment", "volatility_20d"]
        base_names = ["5日动量", "10日动量", "20日动量", "60日动量", "量比(5d/20d)", "额比(5d/20d)", "均线排列", "波动率(20d)"]
        extra_cols = cfg.get("extra_cols", [])
        extra_names = cfg.get("extra_names", [])
        extra_fmt = cfg.get("extra_fmt", {})
        display_cols = base_cols + extra_cols + [score_col]
        col_names = base_names + extra_names + [cfg["score_label"]]

        # Display signals table
        available_cols = [c for c in display_cols if c in signals.columns]
        available_names = [col_names[display_cols.index(c)] for c in available_cols]
        top = signals.head(15)[available_cols].copy()

        def _fmt_col(top, col, score_col, extra_fmt):
            if col in extra_fmt:
                top[col] = top[col].apply(extra_fmt[col])
            elif col in ("vol_ratio_5d", "amount_ratio_5d"):
                top[col] = top[col].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "-")
            elif col == "ma_alignment":
                top[col] = top[col].map({1.0: "🟢多头", 0.0: "⚪震荡", -1.0: "🔴空头"}).fillna("-")
            elif col == score_col:
                top[col] = top[col].apply(lambda x: f"{x:.3f}")
            else:
                top[col] = top[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")

        for col in available_cols:
            _fmt_col(top, col, score_col, extra_fmt)

        top.columns = available_names
        st.dataframe(top, use_container_width=True)

        # Bottom / weak sectors (only for momentum mode)
        if strategy_mode == "🚀 动量追涨":
            st.subheader("💀 弱势板块 Bottom 10")
            bottom = signals.tail(10)[available_cols].copy()
            for col in available_cols:
                _fmt_col(bottom, col, score_col, extra_fmt)
            bottom.columns = available_names
            st.dataframe(bottom, use_container_width=True)

        # --- Level 2: Stock screening within selected sector ---
        st.divider()
        st.subheader("🎯 板块 → 个股二级筛选")
        st.caption("从信号板块中选一个，用基本面+量化信号给成分股打分排名")

        top_boards = signals.head(15).index.tolist()
        selected_board = st.selectbox("选择板块进行个股筛选", top_boards, key="screener_board")

        if selected_board:
            try:
                cons = get_industry_constituents(selected_board)
                scored = score_stocks(cons)

                # Basic fundamental view
                with st.expander("基本面排名 (PE/PB)", expanded=False):
                    factor_cols = ["code", "name", "close", "pe_ttm", "pb", "fundamental_score"]
                    factor_names = ["代码", "名称", "现价", "PE(TTM)", "PB", "基本面得分"]
                    for col, label in [("change_pct", "涨跌%"), ("turnover", "换手率")]:
                        if col in scored.columns and scored[col].notna().any():
                            idx = factor_cols.index("fundamental_score")
                            factor_cols.insert(idx, col)
                            factor_names.insert(idx, label)
                    avail = [c for c in factor_cols if c in scored.columns]
                    avail_names = [factor_names[factor_cols.index(c)] for c in avail]
                    display = scored[avail].head(20).copy()
                    display.columns = avail_names
                    fmt_map = {
                        "涨跌%": lambda x: f"{x:+.2f}%" if pd.notna(x) else "-",
                        "PE(TTM)": lambda x: f"{x:.1f}" if pd.notna(x) and x > 0 else "-",
                        "PB": lambda x: f"{x:.2f}" if pd.notna(x) and x > 0 else "-",
                        "基本面得分": lambda x: f"{x:.3f}" if pd.notna(x) else "-",
                    }
                    for col_name, fmt_fn in fmt_map.items():
                        if col_name in display.columns:
                            display[col_name] = display[col_name].apply(fmt_fn)
                    st.dataframe(display, use_container_width=True, hide_index=True)

                # Signal-based screening
                _has_model_t1 = os.path.exists("data/model.lgb") or (os.path.exists("data/blend_primary.lgb") and os.path.exists("data/blend_extreme.lgb"))
                _use_ml_t1 = st.checkbox("使用ML模型评分", value=_has_model_t1, key="tab1_ml",
                                         disabled=not _has_model_t1,
                                         help="模型文件: data/model.lgb" if _has_model_t1 else "请先在ML模型页训练")
                if st.button(f"📡 扫描 {selected_board} 信号评分", key="tab1_scan"):
                    codes = cons["code"].astype(str).str.zfill(6).tolist()
                    name_map = dict(zip(cons["code"].astype(str).str.zfill(6), cons["name"]))
                    progress = st.progress(0, text="扫描中...")
                    def _update(cur, tot):
                        progress.progress(cur / tot, text=f"扫描 {cur}/{tot}...")
                    screened = screen_stocks(codes, names=name_map, use_ml=_use_ml_t1,
                                             progress_callback=_update)
                    progress.empty()

                    if screened.empty:
                        st.warning("无数据")
                    else:
                        _scoring_label = "ML模型" if _use_ml_t1 else "IC/IR公式"
                        st.markdown(f"**{selected_board}** 量化信号排名 ({len(screened)}只) — {_scoring_label}")
                        disp = screened[["name", "price", "chg_5d", "chg_20d", "bull", "bear",
                                         "ic_score", "signal_score", "rating"]].copy()
                        disp.columns = ["名称", "现价", "5日%", "20日%", "看多", "看空",
                                        "IC分", "综合分", "评级"]
                        disp["5日%"] = disp["5日%"].apply(lambda x: f"{x:+.1f}%")
                        disp["20日%"] = disp["20日%"].apply(lambda x: f"{x:+.1f}%")
                        disp["现价"] = disp["现价"].apply(lambda x: f"{x:.2f}")
                        disp["IC分"] = disp["IC分"].apply(lambda x: f"{x:+.3f}")
                        disp["综合分"] = disp["综合分"].apply(lambda x: f"{x:+.3f}")
                        st.dataframe(disp, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"获取成分股失败: {e}")


# Tab 2: Backtest
with tab2:
    st.header("回测结果")
    if run_bt:
        industry_bars = store.load_industry_bars()
        if industry_bars.empty:
            st.error("无行业数据，请先拉取")
        else:
            risk_params = RiskParams(
                max_sectors=top_n,
                stop_loss_pct=stop_loss,
                trailing_stop_pct=trailing_stop,
                max_drawdown_pct=max_dd,
            )
            with st.spinner("正在运行回测..."):
                result = run_backtest(industry_bars, rebalance_freq=rebal_freq, top_n=top_n, risk_params=risk_params)
                metrics = calc_performance(result)

            if result.empty:
                st.warning("回测无结果")
            else:
                # Metrics
                cols = st.columns(5)
                cols[0].metric("年化收益", metrics.get("annual_return", "N/A"))
                cols[1].metric("Sharpe", metrics.get("sharpe_ratio", "N/A"))
                cols[2].metric("Calmar", metrics.get("calmar_ratio", "N/A"))
                cols[3].metric("最��回撤", metrics.get("max_drawdown", "N/A"))
                cols[4].metric("胜率", metrics.get("win_rate", "N/A"))

                # NAV chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=result["date"], y=result["nav"], name="Strategy NAV", line=dict(color="#2196F3", width=2)))
                fig.update_layout(title="净值曲线", template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Drawdown
                nav = result["nav"]
                peak = nav.cummax()
                dd = (nav - peak) / peak
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=result["date"], y=dd, fill="tozeroy", name="Drawdown", line=dict(color="#F44336", width=1)))
                fig2.update_layout(title="回撤", template="plotly_white", height=250)
                st.plotly_chart(fig2, use_container_width=True)

                st.json(metrics)
    else:
        st.info("调整左侧参数后，点击「运行回测」")


# Tab 3: Sector Details
with tab3:
    st.header("板块详情")
    industry_bars = store.load_industry_bars()
    if industry_bars.empty:
        st.warning("无行业数据")
    else:
        board_names = sorted(industry_bars["board_name"].unique())
        selected = st.selectbox("选择板块", board_names)

        if selected:
            board_data = industry_bars[industry_bars["board_name"] == selected].sort_values("date")

            # K-line chart
            fig = go.Figure(data=go.Candlestick(
                x=board_data["date"], open=board_data["open"],
                high=board_data["high"], low=board_data["low"], close=board_data["close"],
            ))
            fig.update_layout(title=f"{selected} K线", template="plotly_white", height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # Volume chart
            fig2 = go.Figure(data=go.Bar(x=board_data["date"], y=board_data["amount"], name="成交额"))
            fig2.update_layout(title="成交额", template="plotly_white", height=200)
            st.plotly_chart(fig2, use_container_width=True)

            # Constituents with fundamental scoring
            st.subheader(f"{selected} 成分股（按基本面排名）")
            try:
                cons = get_industry_constituents(selected)
                scored = score_stocks(cons)
                st.dataframe(
                    scored[["code", "name", "close", "change_pct", "pe_ttm", "pb", "turnover", "fundamental_score"]].head(20),
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as e:
                st.error(f"获取成分股失败: {e}")


# Tab 4: Portfolio Monitor
with tab4:
    st.header("持仓监控")
    st.caption("监控已购标的的卖出信号，红色=建议卖出，黄色=警告，绿色=继续持有")

    import yaml as _yaml
    import akshare as ak

    portfolio_path = Path(__file__).parent / "config" / "portfolio.yaml"

    # ---- OCR Upload Section (Claude Vision) ----
    with st.expander("📷 上传截图更新持仓"):
        _ocr_file = st.file_uploader("上传券商持仓截图", type=["png", "jpg", "jpeg"],
                                      key="ocr_upload", help="支持国盛证券等券商APP持仓页截图")

        if _ocr_file is not None:
            from PIL import Image
            import io
            import base64
            import json

            _img = Image.open(_ocr_file)
            st.image(_img, caption="上传的截图", use_container_width=True)

            if st.button("识别持仓", key="ocr_run"):
                # Load API key from .env
                from dotenv import load_dotenv
                load_dotenv()
                _api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                if not _api_key:
                    st.error("未找到 ANTHROPIC_API_KEY。请在项目根目录 .env 文件中设置:\n"
                             "ANTHROPIC_API_KEY=sk-ant-...")
                else:
                    with st.spinner("Claude Vision 识别中..."):
                        import anthropic

                        # Encode image to base64
                        _buf = io.BytesIO()
                        _img.save(_buf, format="PNG")
                        _b64 = base64.standard_b64encode(_buf.getvalue()).decode("utf-8")

                        _base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
                        _client_kwargs = {"api_key": _api_key}
                        if _base_url:
                            _client_kwargs["base_url"] = _base_url
                        _client = anthropic.Anthropic(**_client_kwargs)
                        _model_name = os.environ.get("ANTHROPIC_MODEL", "").strip() or "claude-sonnet-4-20250514"
                        _resp = _client.messages.create(
                            model=_model_name,
                            max_tokens=2048,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": _b64,
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": (
                                            "这是一张中国券商APP的持仓截图。请提取所有股票持仓信息，"
                                            "以纯JSON数组返回，每个元素包含以下字段：\n"
                                            '- "名称": 股票名称（中文）\n'
                                            '- "代码": 股票代码（6位数字字符串，保留前导零）\n'
                                            '- "数量": 持仓股数（整数）\n'
                                            '- "成本价": 成本价/买入均价（浮点数）\n\n'
                                            "只返回JSON数组，不要其他文字。如果某个字段无法识别，"
                                            "数量填0，成本价填0.0。示例格式：\n"
                                            '[{"名称":"军工ETF","代码":"512660","数量":28500,"成本价":1.455}]'
                                        ),
                                    },
                                ],
                            }],
                        )

                        _raw_text = _resp.content[0].text.strip()

                    # Show raw response for debugging
                    with st.expander("Claude 原始返回"):
                        st.code(_raw_text)

                    # Parse JSON from response (handle markdown code blocks)
                    _json_text = _raw_text
                    if "```" in _json_text:
                        # Extract content between ```json and ```
                        import re
                        _m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', _json_text, re.DOTALL)
                        if _m:
                            _json_text = _m.group(1).strip()

                    try:
                        _parsed = json.loads(_json_text)
                    except json.JSONDecodeError as _je:
                        st.error(f"JSON 解析失败: {_je}\n\n原始文本:\n{_raw_text}")
                        _parsed = []

                    if not _parsed:
                        st.warning("未能从截图中解析出持仓信息")
                    else:
                        st.success(f"识别到 {len(_parsed)} 只持仓")

                        # Normalize fields
                        for item in _parsed:
                            item["代码"] = str(item.get("代码", "")).zfill(6)
                            item["数量"] = int(item.get("数量", 0))
                            item["成本价"] = float(item.get("成本价", 0.0))

                        # Editable table
                        _edit_df = pd.DataFrame(_parsed)
                        st.caption("请校正识别结果（可直接编辑表格），确认无误后点击保存")
                        _edited = st.data_editor(_edit_df, num_rows="dynamic",
                                                  use_container_width=True, key="ocr_editor")

                        # Store in session state for the save button
                        st.session_state["_ocr_parsed"] = _edited

        # Save button (outside the OCR run block so it persists)
        if "_ocr_parsed" in st.session_state and st.session_state["_ocr_parsed"] is not None:
            _edited_data = st.session_state["_ocr_parsed"]
            if isinstance(_edited_data, pd.DataFrame) and not _edited_data.empty:
                # Show diff with current portfolio
                if portfolio_path.exists():
                    with open(portfolio_path) as _pf:
                        _cur_cfg = _yaml.safe_load(_pf)
                    _cur_holdings = {h["code"]: h for h in _cur_cfg.get("holdings", [])
                                     if h.get("type") == "stock" and h.get("code")}

                    st.markdown("**变更预览:**")
                    _diff_rows = []
                    for _, r in _edited_data.iterrows():
                        code = str(r["代码"]).zfill(6)
                        if code in _cur_holdings:
                            old = _cur_holdings[code]
                            changes = []
                            if int(r["数量"]) != old.get("shares", 0):
                                changes.append(f"数量 {old.get('shares', 0)} → {int(r['数量'])}")
                            if abs(float(r["成本价"]) - old.get("avg_cost", 0)) > 0.001:
                                changes.append(f"成本 {old.get('avg_cost', 0)} → {float(r['成本价']):.3f}")
                            if changes:
                                _diff_rows.append({"股票": f"{r['名称']}({code})", "变更": "修改",
                                                    "详情": ", ".join(changes)})
                            else:
                                _diff_rows.append({"股票": f"{r['名称']}({code})", "变更": "不变", "详情": "-"})
                        else:
                            _diff_rows.append({"股票": f"{r['名称']}({code})", "变更": "新增",
                                                "详情": f"数量={int(r['数量'])}, 成本={float(r['成本价']):.3f}"})
                    # Check for removed holdings
                    _new_codes = set(str(r["代码"]).zfill(6) for _, r in _edited_data.iterrows())
                    for code, old in _cur_holdings.items():
                        if code not in _new_codes:
                            _diff_rows.append({"股票": f"{old.get('name', code)}({code})",
                                                "变更": "截图中无", "详情": "将保留（不自动删除）"})

                    if _diff_rows:
                        st.dataframe(pd.DataFrame(_diff_rows), use_container_width=True, hide_index=True)

                if st.button("确认保存到 portfolio.yaml", key="ocr_save", type="primary"):
                    # Load existing config
                    with open(portfolio_path) as _pf:
                        _cfg = _yaml.safe_load(_pf)

                    _old_holdings = _cfg.get("holdings", [])
                    _old_by_code = {}
                    _non_stock = []
                    for h in _old_holdings:
                        if h.get("type") == "stock" and h.get("code"):
                            _old_by_code[h["code"]] = h
                        else:
                            _non_stock.append(h)

                    # Build new holdings list
                    _new_holdings = []
                    for _, r in _edited_data.iterrows():
                        code = str(r["代码"]).zfill(6)
                        name = str(r["名称"])
                        shares = int(r["数量"])
                        cost = float(r["成本价"])

                        if code in _old_by_code:
                            # Update existing, preserve board/entry_date
                            entry = _old_by_code[code].copy()
                            entry["name"] = name
                            entry["shares"] = shares
                            entry["avg_cost"] = cost
                        else:
                            entry = {
                                "name": name,
                                "code": code,
                                "type": "stock",
                                "board": "",
                                "shares": shares,
                                "avg_cost": cost,
                                "entry_date": "",
                            }
                        _new_holdings.append(entry)

                    # Keep old stock holdings not in screenshot (user can remove manually)
                    _new_codes = set(str(r["代码"]).zfill(6) for _, r in _edited_data.iterrows())
                    for code, old_h in _old_by_code.items():
                        if code not in _new_codes:
                            _new_holdings.append(old_h)

                    # Non-stock entries at the end
                    _new_holdings.extend(_non_stock)
                    _cfg["holdings"] = _new_holdings

                    # Write YAML
                    _header = (
                        "# 我的持仓 - 用于多维度信号监控 + 盈亏跟踪\n"
                        "# type: stock (A股个股) / board (行业板块) / us_stock (美股) / etf (场内ETF)\n"
                        "# shares / avg_cost / entry_date: 可选，填写后 Tab4 显示浮动盈亏\n"
                        "# 更新方式: 发券商截图给 Claude，自动识别并更新\n"
                        f"# 最后更新: {date.today().isoformat()} (OCR截图识别)\n\n"
                    )
                    _yaml_body = _yaml.dump(_cfg, allow_unicode=True, default_flow_style=False, sort_keys=False)
                    with open(portfolio_path, "w") as _wf:
                        _wf.write(_header)
                        _wf.write(_yaml_body)

                    st.success("持仓已更新! 刷新页面查看最新数据。")
                    # Clear OCR state
                    del st.session_state["_ocr_parsed"]
                    st.rerun()

    # ---- End OCR Upload Section ----

    if not portfolio_path.exists():
        st.warning("未找到持仓配置文件 config/portfolio.yaml")
    else:
        with open(portfolio_path) as _f:
            portfolio_cfg = _yaml.safe_load(_f)

        holdings = portfolio_cfg.get("holdings", [])
        industry_bars = store.load_industry_bars()

        for h in holdings:
            name = h["name"]
            code = h.get("code")
            htype = h.get("type", "stock")
            board = h.get("board")

            with st.container():
                st.subheader(f"{'📈' if htype == 'us_stock' else '🏭' if htype == 'board' else '📊'} {name}" + (f" ({code})" if code else ""))

                signals_list = []
                sell_signals = []
                hold_signals = []
                _stock_verdict_done = False

                try:
                    if htype == "board":
                        # Board-level: run all 5 strategies, show which ones flag it
                        if not industry_bars.empty:
                            rot_sig = generate_rotation_signals(industry_bars)
                            board_key = board if board else name
                            if board_key in rot_sig.index:
                                row = rot_sig.loc[board_key]
                                m5 = row.get("mom_5d", 0)
                                m20 = row.get("mom_20d", 0)
                                m60 = row.get("mom_60d", 0)
                                vr = row.get("vol_ratio_5d", 0)
                                ma = row.get("ma_alignment", 0)
                                vol = row.get("volatility_20d", 0)

                                c1, c2, c3, c4, c5, c6 = st.columns(6)
                                c1.metric("5日动量", f"{m5:+.1f}%")
                                c2.metric("20日动量", f"{m20:+.1f}%")
                                c3.metric("60日动量", f"{m60:+.1f}%")
                                c4.metric("量比", f"{vr:.2f}x")
                                c5.metric("均线", {1.0:"多头",0.0:"震荡",-1.0:"空头"}.get(ma, "?"))
                                c6.metric("波动率", f"{vol:.1f}%")

                                if m20 < 0: sell_signals.append(f"20日动量为负 ({m20:+.1f}%)")
                                else: hold_signals.append(f"20日动量为正 ({m20:+.1f}%)")
                                if vr < 0.8: sell_signals.append(f"量比萎缩 ({vr:.2f}x)")
                                elif vr > 1.2: hold_signals.append(f"量比放大 ({vr:.2f}x)")
                                if ma == -1: sell_signals.append("均线空头排列")
                                elif ma == 1: hold_signals.append("均线多头排列")
                                if m5 < -3: sell_signals.append(f"短期急跌 ({m5:+.1f}%)")

                            # Multi-strategy ranking
                            strat_results = []
                            for sname, sfunc in [("动量", generate_rotation_signals), ("超跌反弹", generate_reversal_signals),
                                                  ("超跌未涨", generate_deep_value_signals), ("主力吸筹", generate_accumulation_signals),
                                                  ("强弱切换", generate_rotation_shift_signals)]:
                                sig = sfunc(industry_bars)
                                score_col_name = [c for c in sig.columns if "score" in c][0]
                                if board_key in sig.index:
                                    sc = sig.loc[board_key, score_col_name]
                                    rank = int((sig[score_col_name] > sc).sum()) + 1
                                    strat_results.append({"策略": sname, "得分": f"{sc:.3f}", "排名": f"{rank}/{len(sig)}"})
                                else:
                                    strat_results.append({"策略": sname, "得分": "-", "排名": "未入池"})
                            st.caption("五策略综合评估")
                            st.dataframe(pd.DataFrame(strat_results), use_container_width=True, hide_index=True)

                    elif htype == "stock":
                        # Individual stock analysis (Sina source)
                        _sina_sym = f"sh{code}" if code.startswith(("6","9")) else f"sz{code}"
                        df = ak.stock_zh_a_daily(symbol=_sina_sym, start_date="20240101", adjust="qfq")
                        df = df.sort_values("date")
                        close = df["close"].values
                        high = df["high"].values
                        low = df["low"].values
                        open_arr = df["open"].values
                        volume = df["volume"].values.astype(float)

                        if len(close) > 20:
                            m5 = (close[-1] / close[-6] - 1) * 100
                            m20 = (close[-1] / close[-21] - 1) * 100
                            m60 = (close[-1] / close[-61] - 1) * 100 if len(close) >= 61 else 0
                            ma5 = close[-5:].mean()
                            ma20 = close[-20:].mean()
                            ma60 = close[-60:].mean() if len(close) >= 60 else ma20
                            vol5 = df["volume"].tail(5).mean()
                            vol20 = df["volume"].iloc[-25:-5].mean() if len(df) > 25 else 1
                            vr = vol5 / vol20 if vol20 > 0 else 0
                            trend = "多头" if ma5 > ma20 > ma60 else ("空头" if ma5 < ma20 < ma60 else "震荡")
                            returns = pd.Series(close).pct_change().dropna().tail(20)
                            vol_ann = returns.std() * (252 ** 0.5) * 100
                            ma60_dev = (close[-1] / ma60 - 1) * 100

                            c1, c2, c3, c4, c5, c6 = st.columns(6)
                            c1.metric("现价", f"{close[-1]:.2f}")
                            c2.metric("5日动量", f"{m5:+.1f}%")
                            c3.metric("20日动量", f"{m20:+.1f}%")
                            c4.metric("量比", f"{vr:.2f}x")
                            c5.metric("趋势", trend)
                            c6.metric("波动率", f"{vol_ann:.1f}%")

                            # P&L display when position data is available
                            _shares = h.get("shares", 0)
                            _avg_cost = h.get("avg_cost", 0.0)
                            _entry_date = h.get("entry_date", "")
                            if _shares and _avg_cost and _avg_cost > 0:
                                _cur = close[-1]
                                _pnl_pct = (_cur / _avg_cost - 1) * 100
                                _mkt_val = _shares * _cur
                                _profit = _mkt_val - _shares * _avg_cost
                                _p1, _p2, _p3, _p4 = st.columns(4)
                                _p1.metric("持仓", f"{_shares}股")
                                _p2.metric("成本", f"¥{_avg_cost:.2f}")
                                _p3.metric("市值", f"¥{_mkt_val:,.0f}")
                                _p4.metric("浮动盈亏", f"¥{_profit:+,.0f}", delta=f"{_pnl_pct:+.1f}%")

                            # Full factor detail in expander
                            with st.expander("详细因子"):
                                fc1, fc2, fc3, fc4 = st.columns(4)
                                fc1.metric("60日动量", f"{m60:+.1f}%")
                                fc2.metric("MA60偏离", f"{ma60_dev:+.1f}%")
                                fc3.metric("MA5", f"{ma5:.2f}")
                                fc4.metric("MA20", f"{ma20:.2f}")

                            # --- Signal-based analysis ---
                            all_signals = compute_all_technical_signals(close, high, low, open_arr, volume)

                            # Board context signal
                            if board and not industry_bars.empty:
                                rot_sig = generate_rotation_signals(industry_bars)
                                if board in rot_sig.index:
                                    b_m20 = rot_sig.loc[board, "mom_20d"]
                                    b_score = rot_sig.loc[board, "composite_score"]
                                    b_rank = int((rot_sig["composite_score"] > b_score).sum()) + 1
                                    if b_m20 < -5:
                                        all_signals.append({"dimension": "板块", "name": "行业动量", "value": round(b_m20, 1),
                                                            "signal": "bearish", "detail": f"{board}走弱 (20d:{b_m20:+.1f}%, 排名{b_rank}/90)"})
                                    elif b_m20 > 0:
                                        all_signals.append({"dimension": "板块", "name": "行业动量", "value": round(b_m20, 1),
                                                            "signal": "bullish", "detail": f"{board}走强 (20d:{b_m20:+.1f}%, 排名{b_rank}/90)"})
                                    else:
                                        all_signals.append({"dimension": "板块", "name": "行业动量", "value": round(b_m20, 1),
                                                            "signal": "neutral", "detail": f"{board}震荡 (排名{b_rank}/90)"})

                            # Display signals table
                            _sig_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
                            if all_signals:
                                sig_rows = [{"维度": s["dimension"], "指标": s["name"],
                                             "方向": _sig_emoji[s["signal"]], "说明": s["detail"]}
                                            for s in all_signals]
                                st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

                            # --- ML Model Scoring ---
                            with st.expander("🤖 ML模型评分", expanded=True):
                                _blend_ok_t4 = os.path.exists("data/blend_primary.lgb") and os.path.exists("data/blend_extreme.lgb")
                                _single_ok_t4 = os.path.exists("data/model.lgb")
                                if _blend_ok_t4 or _single_ok_t4:
                                    try:
                                        with st.spinner("计算ML评分..."):
                                            _ml_feat = build_latest_features([code], include_fundamentals=True)
                                        if not _ml_feat.empty:
                                            if _blend_ok_t4:
                                                _ml_ranker = BlendRanker()
                                                _ml_pred_raw = _ml_ranker.predict_raw(_ml_feat)[0]
                                                _ml_pred = _ml_pred_raw  # raw return for display
                                                _model_label = "BlendRanker"
                                            else:
                                                _ml_ranker = StockRanker()
                                                _ml_ranker.load()
                                                _ml_pred = _ml_ranker.predict(_ml_feat)[0]
                                                _model_label = "StockRanker"

                                            # Data quality warning
                                            _ml_warn = ""
                                            if "_data_warnings" in _ml_feat.columns:
                                                _ml_warn = _ml_feat["_data_warnings"].values[0]
                                            if _ml_warn:
                                                st.warning(f"⚠ 数据缺失({_ml_warn})，以下预测可能不准确")

                                            _ml_c1, _ml_c2, _ml_c3 = st.columns(3)
                                            _ml_c1.metric("ML预测20日收益", f"{_ml_pred*100:+.2f}%")
                                            if _ml_pred > 0.03:
                                                _ml_c2.metric("ML评级", "★★★")
                                            elif _ml_pred > 0.01:
                                                _ml_c2.metric("ML评级", "★★☆")
                                            elif _ml_pred > 0:
                                                _ml_c2.metric("ML评级", "★☆☆")
                                            else:
                                                _ml_c2.metric("ML评级", "⚠️")
                                            _ml_c3.metric("模型", _model_label)

                                            # Top contributing factors
                                            _fi = _ml_ranker.feature_importance_report()
                                            if not _fi.empty:
                                                _top_fi = _fi.head(8)
                                                _factor_detail = []
                                                for _, _fr in _top_fi.iterrows():
                                                    _fn = _fr["feature"]
                                                    if _fn in _ml_feat.columns:
                                                        _fv = _ml_feat[_fn].values[0]
                                                        _factor_detail.append({
                                                            "因子": _fn,
                                                            "值": f"{_fv:+.4f}" if pd.notna(_fv) else "N/A",
                                                            "重要性": f"{_fr['pct']:.1f}%",
                                                        })
                                                st.dataframe(pd.DataFrame(_factor_detail),
                                                             use_container_width=True, hide_index=True)
                                        else:
                                            st.warning("无法计算ML特征")
                                    except Exception as _ml_e:
                                        st.error(f"ML评分失败: {_ml_e}")
                                else:
                                    st.info("未训练ML模型 — 前往「ML模型」页面训练")

                            # External data signals
                            with st.expander("📡 外部数据信号"):
                                with st.spinner("加载外部数据..."):
                                    ext_signals = fetch_all_external_signals(code)
                                if ext_signals:
                                    ext_rows = [{"维度": s["dimension"], "指标": s["name"],
                                                 "方向": _sig_emoji[s["signal"]], "说明": s["detail"]}
                                                for s in ext_signals]
                                    st.dataframe(pd.DataFrame(ext_rows), use_container_width=True, hide_index=True)
                                    all_signals.extend(ext_signals)
                                else:
                                    st.info("无外部数据 (API不可用)")

                            # --- Factor IC Backtesting ---
                            with st.expander("📊 因子IC回测"):
                                with st.spinner("计算因子IC..."):
                                    try:
                                        from mp.backtest.ic_analysis import run_ic_analysis
                                        ic_df = run_ic_analysis(code, start="20230101")
                                        if ic_df.empty:
                                            st.warning("数据不足，无法计算IC")
                                        else:
                                            # Format for display
                                            display_cols = []
                                            for h in [5, 10, 20]:
                                                display_cols.extend([f"IC({h}d)", f"IR({h}d)"])
                                            display_cols.append("有效性")
                                            ic_display = ic_df[display_cols].copy()
                                            # Format numbers
                                            for col in ic_display.columns:
                                                if col.startswith("IC("):
                                                    ic_display[col] = ic_display[col].apply(
                                                        lambda v: f"{v:+.3f}" if pd.notna(v) else "-")
                                                elif col.startswith("IR("):
                                                    ic_display[col] = ic_display[col].apply(
                                                        lambda v: f"{v:.2f}" if pd.notna(v) else "-")
                                            st.dataframe(ic_display, use_container_width=True)
                                            st.caption("IC = 因子值与未来收益的Spearman相关系数 | IR = IC均值/IC标准差 | ★★★=高效因子, 噪音=无预测力")
                                    except Exception as e:
                                        st.error(f"IC计算失败: {e}")

                            # Verdict from all signals
                            n_bull = sum(1 for s in all_signals if s["signal"] == "bullish")
                            n_bear = sum(1 for s in all_signals if s["signal"] == "bearish")
                            n_total = len(all_signals)
                            if n_total > 0:
                                bear_pct = n_bear / n_total
                                bull_pct = n_bull / n_total
                                if bear_pct >= 0.6:
                                    st.error(f"🔴 **偏空** — {n_bear}/{n_total} 个信号看空 ({bear_pct:.0%})")
                                elif bull_pct >= 0.6:
                                    st.success(f"🟢 **偏多** — {n_bull}/{n_total} 个信号看多 ({bull_pct:.0%})")
                                elif bear_pct >= 0.4:
                                    st.warning(f"🟡 **偏空震荡** — 看空{n_bear} / 看多{n_bull} / 中性{n_total - n_bear - n_bull}")
                                elif bull_pct >= 0.4:
                                    st.info(f"🟠 **偏多震荡** — 看多{n_bull} / 看空{n_bear} / 中性{n_total - n_bear - n_bull}")
                                else:
                                    st.info(f"⚪ **中性** — 看多{n_bull} / 看空{n_bear} / 中性{n_total - n_bear - n_bull}")
                            _stock_verdict_done = True

                    elif htype == "us_stock":
                        df = ak.stock_us_daily(symbol=code, adjust="qfq")
                        df = df.sort_values("date")
                        close = df["close"].values

                        if len(close) > 20:
                            m5 = (close[-1] / close[-6] - 1) * 100
                            m20 = (close[-1] / close[-21] - 1) * 100
                            m60 = (close[-1] / close[-61] - 1) * 100 if len(close) >= 61 else 0
                            ma5 = close[-5:].mean()
                            ma20 = close[-20:].mean()
                            ma60 = close[-60:].mean() if len(close) >= 60 else ma20
                            trend = "多头" if ma5 > ma20 > ma60 else ("空头" if ma5 < ma20 < ma60 else "震荡")
                            returns = pd.Series(close).pct_change().dropna().tail(20)
                            vol_ann = returns.std() * (252 ** 0.5) * 100

                            c1, c2, c3, c4, c5, c6 = st.columns(6)
                            c1.metric("现价", f"${close[-1]:.2f}")
                            c2.metric("5日动量", f"{m5:+.1f}%")
                            c3.metric("20日动量", f"{m20:+.1f}%")
                            c4.metric("60日动量", f"{m60:+.1f}%")
                            c5.metric("趋势", trend)
                            c6.metric("波动率", f"{vol_ann:.1f}%")

                            # P&L display when position data is available
                            _shares = h.get("shares", 0)
                            _avg_cost = h.get("avg_cost", 0.0)
                            if _shares and _avg_cost and _avg_cost > 0:
                                _cur = close[-1]
                                _pnl_pct = (_cur / _avg_cost - 1) * 100
                                _mkt_val = _shares * _cur
                                _profit = _mkt_val - _shares * _avg_cost
                                _p1, _p2, _p3, _p4 = st.columns(4)
                                _p1.metric("持仓", f"{_shares}股")
                                _p2.metric("成本", f"${_avg_cost:.2f}")
                                _p3.metric("市值", f"${_mkt_val:,.0f}")
                                _p4.metric("浮动盈亏", f"${_profit:+,.0f}", delta=f"{_pnl_pct:+.1f}%")

                            if m20 < -5: sell_signals.append(f"20日动量为负 ({m20:+.1f}%)")
                            elif m20 > 0: hold_signals.append(f"20日动量为正 ({m20:+.1f}%)")
                            if trend == "空头": sell_signals.append("均线空头排列")
                            elif trend == "多头": hold_signals.append("均线多头排列")
                            if m5 < -5: sell_signals.append(f"短期急跌 ({m5:+.1f}%)")
                            if close[-1] < ma20: sell_signals.append(f"跌破MA20 (${ma20:.2f})")
                            if vol_ann > 60: sell_signals.append(f"波动率过高 ({vol_ann:.0f}%)")

                except Exception as e:
                    st.error(f"数据获取失败: {e}")
                    continue

                # Verdict (board / US stock — stock type renders its own above)
                if not _stock_verdict_done:
                    if len(sell_signals) >= 3:
                        st.error(f"🔴 **建议卖出** — {len(sell_signals)} 个卖出信号")
                    elif len(sell_signals) >= 2:
                        st.warning(f"🟡 **警告** — {len(sell_signals)} 个卖出信号，密切关注")
                    elif len(sell_signals) >= 1:
                        st.info(f"🟠 **留意** — {len(sell_signals)} 个弱势信号")
                    else:
                        st.success(f"🟢 **继续持有** — 暂无卖出信号")

                    col_s, col_h = st.columns(2)
                    with col_s:
                        if sell_signals:
                            st.markdown("**卖出信号：**")
                            for s in sell_signals:
                                st.markdown(f"- ❌ {s}")
                    with col_h:
                        if hold_signals:
                            st.markdown("**持有信号：**")
                            for s in hold_signals:
                                st.markdown(f"- ✅ {s}")

                st.divider()


# Tab 5: Stock Radar
with tab5:
    st.header("选股雷达")
    st.caption("从板块成分股中，用量化信号系统筛选最优个股")

    import akshare as _ak5

    industry_bars_t5 = store.load_industry_bars()

    # Board selection
    col_src, col_board = st.columns([1, 2])
    with col_src:
        radar_source = st.radio("选股来源", ["板块成分股", "自定义代码"], key="radar_src")
    with col_board:
        if radar_source == "板块成分股":
            if not industry_bars_t5.empty:
                rot = generate_rotation_signals(industry_bars_t5)
                radar_boards = rot.head(20).index.tolist()
            else:
                radar_boards = []
            radar_board = st.selectbox("选择板块", radar_boards, key="radar_board")
        else:
            radar_codes_input = st.text_input("输入股票代码（逗号分隔）", placeholder="603799,002466,002460", key="radar_codes")

    _has_model_t5 = os.path.exists("data/model.lgb") or (os.path.exists("data/blend_primary.lgb") and os.path.exists("data/blend_extreme.lgb"))
    col_ext, col_ml = st.columns(2)
    with col_ext:
        include_ext = st.checkbox("加载外部数据信号（融资/主力资金/北向等，较慢）", key="radar_ext")
    with col_ml:
        use_ml_scoring = st.checkbox("使用ML模型评分", value=_has_model_t5, key="radar_ml",
                                     disabled=not _has_model_t5,
                                     help="已加载模型" if _has_model_t5 else "请先在ML模型页训练")

    if st.button("🔍 开始扫描", key="radar_scan", type="primary"):
        # Resolve codes and names
        _radar_codes = []
        _radar_names = {}

        if radar_source == "板块成分股" and radar_board:
            with st.spinner(f"获取 {radar_board} 成分股..."):
                try:
                    _cons = get_industry_constituents(radar_board)
                    _radar_codes = _cons["code"].astype(str).str.zfill(6).tolist()
                    _radar_names = dict(zip(_cons["code"].astype(str).str.zfill(6), _cons["name"]))
                except Exception as e:
                    st.error(f"获取成分股失败: {e}")
        elif radar_source == "自定义代码" and radar_codes_input:
            _radar_codes = [c.strip().zfill(6) for c in radar_codes_input.split(",") if c.strip()]

        if _radar_codes:
            st.info(f"扫描 {len(_radar_codes)} 只股票...")
            progress = st.progress(0, text="扫描中...")

            def _radar_progress(cur, tot):
                progress.progress(cur / tot, text=f"扫描 {cur}/{tot}...")

            screened = screen_stocks(
                _radar_codes, names=_radar_names,
                include_external=include_ext,
                use_ml=use_ml_scoring,
                progress_callback=_radar_progress,
            )
            progress.empty()

            if screened.empty:
                st.warning("无有效数据")
            else:
                # Summary metrics
                n_recommend = len(screened[screened["rating"].isin(["★★★", "★★☆"])])
                n_watch = len(screened[screened["rating"] == "★☆☆"])
                n_avoid = len(screened[screened["rating"] == "⚠️"])

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("扫描数", len(screened))
                mc2.metric("推荐", n_recommend)
                mc3.metric("可关注", n_watch)
                mc4.metric("不推荐", n_avoid)

                # Main table
                st.subheader("排名")
                disp = screened[["name", "price", "chg_5d", "chg_20d", "bull", "bear",
                                 "ic_score", "signal_score", "rating"]].copy()
                disp.columns = ["名称", "现价", "5日%", "20日%", "看多", "看空",
                                "IC分", "综合分", "评级"]
                disp["5日%"] = disp["5日%"].apply(lambda x: f"{x:+.1f}%")
                disp["20日%"] = disp["20日%"].apply(lambda x: f"{x:+.1f}%")
                disp["现价"] = disp["现价"].apply(lambda x: f"{x:.2f}")
                disp["IC分"] = disp["IC分"].apply(lambda x: f"{x:+.3f}")
                disp["综合分"] = disp["综合分"].apply(lambda x: f"{x:+.3f}")
                st.dataframe(disp, use_container_width=True, hide_index=True)

                # Detail view for top stocks
                top_codes = screened.head(5)["code"].tolist()
                top_names = dict(zip(screened["code"], screened["name"]))
                st.subheader("Top 5 详情")
                for tc in top_codes:
                    tname = top_names.get(tc, tc)
                    trow = screened[screened["code"] == tc].iloc[0]
                    with st.expander(f"{tname} ({tc}) — {trow['rating']} 综合分{trow['signal_score']:+.3f}"):
                        # ML factor breakdown
                        if use_ml_scoring:
                            try:
                                _t5_feat = build_latest_features([tc], include_fundamentals=True)
                                if not _t5_feat.empty:
                                    _b5_ok = os.path.exists("data/blend_primary.lgb") and os.path.exists("data/blend_extreme.lgb")
                                    if _b5_ok:
                                        _t5_ranker = BlendRanker()
                                        _t5_pred = _t5_ranker.predict_raw(_t5_feat)[0]
                                    else:
                                        _t5_ranker = StockRanker()
                                        _t5_ranker.load()
                                        _t5_pred = _t5_ranker.predict(_t5_feat)[0]
                                    _t5_warn = ""
                                    if "_data_warnings" in _t5_feat.columns:
                                        _t5_warn = _t5_feat["_data_warnings"].values[0]
                                    if _t5_warn:
                                        st.warning(f"⚠ 数据缺失({_t5_warn})，预测可能不准确")
                                    st.metric("ML预测20日收益", f"{_t5_pred*100:+.2f}%")
                                    _t5_fi = _t5_ranker.feature_importance_report()
                                    if not _t5_fi.empty:
                                        _t5_rows = []
                                        for _, _t5r in _t5_fi.head(10).iterrows():
                                            _fn = _t5r["feature"]
                                            _fv = _t5_feat[_fn].values[0] if _fn in _t5_feat.columns else None
                                            _t5_rows.append({
                                                "因子": _fn,
                                                "当前值": f"{_fv:+.4f}" if pd.notna(_fv) else "N/A",
                                                "重要性": f"{_t5r['pct']:.1f}%",
                                            })
                                        st.dataframe(pd.DataFrame(_t5_rows),
                                                     use_container_width=True, hide_index=True)
                            except Exception as _t5e:
                                st.warning(f"ML详情加载失败: {_t5e}")

                        # Signal table (always show)
                        detail = screen_single_stock(tc, include_external=include_ext)
                        if detail and detail["signals"]:
                            _sig_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
                            sig_rows = [{"维度": s["dimension"], "指标": s["name"],
                                         "方向": _sig_emoji[s["signal"]], "说明": s["detail"]}
                                        for s in detail["signals"]]
                            st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)
                        else:
                            st.info("无信号数据")


# Tab 6: ML Model Training
with tab6:
    st.header("ML模型")
    st.caption("LightGBM股票评分模型 — 25个技术因子 + 6个基本面因子")

    from datetime import datetime as _dt6

    model_path = "data/model.lgb"
    model_exists = os.path.exists(model_path)
    model_60d_path = "data/model_60d.lgb"
    model_60d_exists = os.path.exists(model_60d_path)
    blend_primary_path = "data/blend_primary.lgb"
    blend_extreme_path = "data/blend_extreme.lgb"
    blend_exists = os.path.exists(blend_primary_path) and os.path.exists(blend_extreme_path)

    # --- Model status dashboard ---
    # BlendRanker status
    if blend_exists:
        st.success("BlendRanker 已就绪 (推荐)")
        _bp_mtime = os.path.getmtime(blend_primary_path)
        _bp_time = _dt6.fromtimestamp(_bp_mtime).strftime("%Y-%m-%d %H:%M")
        _bp_size = os.path.getsize(blend_primary_path) / 1024
        _be_size = os.path.getsize(blend_extreme_path) / 1024
        _bc1, _bc2, _bc3, _bc4 = st.columns(4)
        _bc1.metric("主模型", "blend_primary.lgb")
        _bc2.metric("极端模型", "blend_extreme.lgb")
        _bc3.metric("训练时间", _bp_time)
        _bc4.metric("总大小", f"{_bp_size + _be_size:.0f} KB")

        try:
            _blend6 = BlendRanker()
            fi6 = _blend6.feature_importance_report()
            if not fi6.empty:
                st.subheader("BlendRanker 因子重要性")
                _fi_top = fi6.head(15)
                _fi_chart = go.Figure(go.Bar(
                    x=_fi_top["pct"].values[::-1],
                    y=_fi_top["feature"].values[::-1],
                    orientation="h",
                    marker_color="#FF9800",
                ))
                _fi_chart.update_layout(
                    template="plotly_white", height=400,
                    xaxis_title="重要性 (%)", yaxis_title="",
                    margin=dict(l=120),
                )
                st.plotly_chart(_fi_chart, use_container_width=True)
                with st.expander("完整因子重要性表"):
                    st.dataframe(fi6, use_container_width=True, hide_index=True)
        except Exception:
            pass

    # StockRanker status
    if model_exists:
        st.success("StockRanker 已就绪")

        _model_mtime = os.path.getmtime(model_path)
        _model_time = _dt6.fromtimestamp(_model_mtime).strftime("%Y-%m-%d %H:%M")
        _model_size = os.path.getsize(model_path) / 1024

        _ms1, _ms2, _ms3 = st.columns(3)
        _ms1.metric("模型文件", "data/model.lgb")
        _ms2.metric("训练时间", _model_time)
        _ms3.metric("文件大小", f"{_model_size:.0f} KB")

        if model_60d_exists:
            _m60_mtime = os.path.getmtime(model_60d_path)
            _m60_time = _dt6.fromtimestamp(_m60_mtime).strftime("%Y-%m-%d %H:%M")
            _m60_size = os.path.getsize(model_60d_path) / 1024
            _ms60_1, _ms60_2, _ms60_3 = st.columns(3)
            _ms60_1.metric("60日模型", "data/model_60d.lgb")
            _ms60_2.metric("训练时间", _m60_time)
            _ms60_3.metric("文件大小", f"{_m60_size:.0f} KB")

        # Load model and show feature importance
        try:
            _ranker6 = StockRanker()
            if _ranker6.load():
                fi6 = _ranker6.feature_importance_report()
                if not fi6.empty:
                    st.subheader("因子重要性")

                    # Bar chart
                    _fi_top = fi6.head(15)
                    _fi_chart = go.Figure(go.Bar(
                        x=_fi_top["pct"].values[::-1],
                        y=_fi_top["feature"].values[::-1],
                        orientation="h",
                        marker_color="#2196F3",
                    ))
                    _fi_chart.update_layout(
                        template="plotly_white", height=400,
                        xaxis_title="重要性 (%)", yaxis_title="",
                        margin=dict(l=120),
                    )
                    st.plotly_chart(_fi_chart, use_container_width=True)

                    # Table
                    with st.expander("完整因子重要性表"):
                        st.dataframe(fi6, use_container_width=True, hide_index=True)
        except Exception:
            pass
    if not model_exists and not blend_exists:
        st.info("尚未训练模型，请配置参数后点击训练")

    # --- Training controls ---
    st.divider()
    st.subheader("训练 / 重新训练")

    col_idx, col_dates = st.columns(2)
    with col_idx:
        train_index = st.selectbox("训练股票池", ["hs300", "zz500", "zz1000"],
                                   index=1, key="ml_index",
                                   help="hs300=沪深300, zz500=中证500, zz1000=中证1000")
    with col_dates:
        train_start = st.text_input("起始日期", value="20220101", key="ml_start")

    col_horizon, col_splits, col_fund = st.columns(3)
    with col_horizon:
        train_horizon = st.number_input("预测窗口(交易日)", value=20, min_value=5,
                                        max_value=60, key="ml_horizon")
    with col_splits:
        train_splits = st.number_input("交叉验证折数", value=5, min_value=3,
                                       max_value=10, key="ml_splits")
    with col_fund:
        train_fund = st.checkbox("包含基本面因子", value=False, key="ml_fund",
                                 help="获取PE/PB/ROE等，每只股票额外+1s")

    train_also_60d = st.checkbox("同时训练60日模型", value=False, key="ml_also_60d",
                                  help="额外训练一个60日预测窗口的模型，用于日报60日收益预测")

    if st.button("🚀 开始训练", key="ml_train", type="primary"):
        # Step 1: Get stock universe
        with st.spinner(f"获取 {train_index} 成分股..."):
            try:
                codes = get_index_constituents(train_index)
                st.info(f"股票池: {len(codes)} 只, 因子数: {len(FACTOR_COLUMNS)}")
            except Exception as e:
                st.error(f"获取成分股失败: {e}")
                st.stop()

        # Step 2: Build dataset
        st.info(f"构建训练数据集（{len(codes)}只 × ~1s/只）...")
        data_progress = st.progress(0, text="构建数据...")

        def _ml_data_progress(cur, tot):
            data_progress.progress(cur / tot, text=f"获取数据 {cur}/{tot}...")

        dataset = build_dataset(codes, start=train_start, horizon=train_horizon,
                                include_fundamentals=train_fund,
                                progress_callback=_ml_data_progress)
        data_progress.empty()

        if dataset.empty:
            st.error("数据集为空，无法训练")
            st.stop()

        st.success(f"数据集: {len(dataset):,} 行, {dataset['code'].nunique()} 只股票, "
                   f"{dataset['date'].min().strftime('%Y-%m-%d')} ~ {dataset['date'].max().strftime('%Y-%m-%d')}")

        # Step 3: Train model
        with st.spinner("训练LightGBM模型..."):
            ranker = StockRanker()
            metrics = ranker.train(dataset, n_splits=train_splits)

        # Step 4: Display results
        st.subheader("训练结果")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("CV MAE", f"{metrics['cv_mae_mean']:.4f}")
        mc2.metric("CV IC", f"{metrics['cv_ic_mean']:.3f}")
        mc3.metric("最佳轮数", metrics["best_rounds"])
        mc4.metric("训练样本", f"{metrics['n_train_rows']:,}")

        st.caption(f"MAE std: {metrics['cv_mae_std']:.4f} | IC std: {metrics['cv_ic_std']:.3f} | "
                   f"CV折数: {metrics['n_folds']}")

        # Feature importance chart
        fi = metrics.get("feature_importance")
        if fi is not None and not fi.empty:
            st.subheader("因子重要性")
            _fi_top = fi.head(15)
            _fi_chart = go.Figure(go.Bar(
                x=_fi_top["pct"].values[::-1],
                y=_fi_top["feature"].values[::-1],
                orientation="h",
                marker_color="#4CAF50",
            ))
            _fi_chart.update_layout(
                template="plotly_white", height=400,
                xaxis_title="重要性 (%)", yaxis_title="",
                margin=dict(l=120),
            )
            st.plotly_chart(_fi_chart, use_container_width=True)

            with st.expander("完整因子重要性表"):
                st.dataframe(fi, use_container_width=True, hide_index=True)

        st.success("模型已保存，所有页面的ML评分已自动生效")

        # --- Optional 60-day model training ---
        if train_also_60d:
            st.divider()
            st.subheader("训练60日模型")

            st.info(f"构建60日训练数据集（{len(codes)}只 × ~1s/只）...")
            data_progress_60d = st.progress(0, text="构建60日数据...")

            def _ml_data_progress_60d(cur, tot):
                data_progress_60d.progress(cur / tot, text=f"获取60日数据 {cur}/{tot}...")

            dataset_60d = build_dataset(codes, start=train_start, horizon=60,
                                        include_fundamentals=train_fund,
                                        progress_callback=_ml_data_progress_60d)
            data_progress_60d.empty()

            if dataset_60d.empty:
                st.error("60日数据集为空，跳过60日模型训练")
            else:
                st.success(f"60日数据集: {len(dataset_60d):,} 行, {dataset_60d['code'].nunique()} 只股票, "
                           f"{dataset_60d['date'].min().strftime('%Y-%m-%d')} ~ "
                           f"{dataset_60d['date'].max().strftime('%Y-%m-%d')}")

                with st.spinner("训练60日LightGBM模型..."):
                    ranker_60d = StockRanker(model_path="data/model_60d.lgb")
                    metrics_60d = ranker_60d.train(dataset_60d, n_splits=train_splits)

                st.subheader("60日模型训练结果")
                mc60_1, mc60_2, mc60_3, mc60_4 = st.columns(4)
                mc60_1.metric("CV MAE", f"{metrics_60d['cv_mae_mean']:.4f}")
                mc60_2.metric("CV IC", f"{metrics_60d['cv_ic_mean']:.3f}")
                mc60_3.metric("最佳轮数", metrics_60d["best_rounds"])
                mc60_4.metric("训练样本", f"{metrics_60d['n_train_rows']:,}")

                st.caption(f"MAE std: {metrics_60d['cv_mae_std']:.4f} | "
                           f"IC std: {metrics_60d['cv_ic_std']:.3f} | "
                           f"CV折数: {metrics_60d['n_folds']}")

                st.success("60日模型已保存至 data/model_60d.lgb")


# =====================================================================
# Tab 7: Report Center (日报中心)
# =====================================================================
with tab7:
    st.header("日报中心")
    st.caption("一键生成完整持仓评估报告：市场状态 → ML 评估 → 60日技术面 → ZZ500 推荐 → 换仓建议")

    # --- Helper: fetch realtime prices from Sina ---
    def _fetch_realtime_prices_web(codes):
        import httpx
        result = {}
        symbols = [("sh" if c.startswith("6") else "sz") + c for c in codes]
        try:
            url = f"https://hq.sinajs.cn/list={','.join(symbols)}"
            resp = httpx.get(url, headers={"Referer": "https://finance.sina.com.cn"}, timeout=10)
            for line in resp.text.strip().split("\n"):
                if "=" not in line:
                    continue
                var_part, data_part = line.split("=", 1)
                code = var_part.split("_")[-1][2:]
                fields = data_part.strip('";\r').split(",")
                if len(fields) >= 10 and fields[0]:
                    try:
                        prev_close = float(fields[2])
                        current = float(fields[3])
                        if prev_close > 0 and current > 0:
                            result[code] = {
                                "name": fields[0],
                                "price": current,
                                "prev_close": prev_close,
                                "change_pct": (current / prev_close - 1) * 100,
                                "open": float(fields[1]),
                                "high": float(fields[4]),
                                "low": float(fields[5]),
                                "volume": float(fields[8]) * 100,
                                "amount": float(fields[9]),
                            }
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass
        return result

    # --- Helper: load holdings from portfolio.yaml ---
    def _load_holdings_web():
        import yaml
        p = Path(__file__).parent / "config" / "portfolio.yaml"
        if not p.exists():
            return []
        with open(p) as f:
            cfg = yaml.safe_load(f)
        return [h for h in cfg.get("holdings", []) if h.get("type") == "stock" and h.get("code")]

    # --- Helper: 60d analysis (same as daily_report.evaluate_holdings_60d) ---
    def _evaluate_holdings_60d_web(codes):
        from mp.data.fetcher import get_daily_bars
        import numpy as np
        start = (date.today() - pd.Timedelta(days=240)).strftime("%Y%m%d")
        results = []
        for code in codes:
            try:
                df = get_daily_bars(code, start=start)
            except Exception:
                results.append({"code": code, "error": "fetch failed"})
                continue
            if df is None or df.empty or "date" not in df.columns or len(df) < 60:
                results.append({"code": code, "error": "insufficient data"})
                continue
            df = df.sort_values("date").reset_index(drop=True)
            close = df["close"].values.astype(float)
            volume = df["volume"].values.astype(float)
            cur = close[-1]
            high_60 = np.max(close[-60:])
            low_60 = np.min(close[-60:])
            rng = high_60 - low_60
            pos = ((cur - low_60) / rng * 100) if rng > 0 else 50.0
            ma20 = np.mean(close[-20:])
            ma60 = np.mean(close[-60:])
            ma120 = np.mean(close[-120:]) if len(close) >= 120 else np.nan
            def pct_vs(p, m):
                return (p - m) / m * 100 if m != 0 and not np.isnan(m) else np.nan
            if len(close) >= 80:
                ma60_now = np.mean(close[-60:])
                ma60_prev = np.mean(close[-80:-20])
                ma60_trend = (ma60_now - ma60_prev) / ma60_prev * 100 if ma60_prev != 0 else 0.0
                ma60_rising = bool(ma60_now > ma60_prev)
            else:
                ma60_trend = np.nan
                ma60_rising = None
            p60 = close[-61] if len(close) >= 61 else close[0]
            p20 = close[-21] if len(close) >= 21 else close[0]
            m60 = (cur - p60) / p60 * 100 if p60 != 0 else 0.0
            m_first40 = (p20 - p60) / p60 * 100 if p60 != 0 else 0.0
            m_last20 = (cur - p20) / p20 * 100 if p20 != 0 else 0.0
            results.append({
                "code": code,
                "price_position": round(pos, 1),
                "pct_vs_ma20": round(pct_vs(cur, ma20), 1),
                "pct_vs_ma60": round(pct_vs(cur, ma60), 1),
                "pct_vs_ma120": round(pct_vs(cur, ma120), 1) if not np.isnan(ma120) else None,
                "above_ma20": bool(cur > ma20),
                "above_ma60": bool(cur > ma60),
                "above_ma120": bool(cur > ma120) if not np.isnan(ma120) else None,
                "ma60_rising": ma60_rising,
                "ma60_trend_pct": round(ma60_trend, 1) if not np.isnan(ma60_trend) else None,
                "momentum_60d": round(m60, 1),
                "momentum_first40": round(m_first40, 1),
                "momentum_last20": round(m_last20, 1),
            })
        return results

    # --- Helper: score universe for BlendRanker ---
    def _score_universe_web(ranker, holding_codes, intraday_bars=None):
        is_rank = getattr(ranker, "score_type", "predicted_return") == "rank_percentile"
        if not is_rank:
            features = build_latest_features(holding_codes, include_fundamentals=True,
                                             intraday_bars=intraday_bars)
            if features.empty:
                return pd.DataFrame()
            scores = ranker.predict(features)
            df = pd.DataFrame({
                "code": features["code"].values,
                "ml_score": scores,
                "rank_pct": pd.Series(scores).rank(pct=True).values,
                "raw_return": scores,
            })
            if "_data_warnings" in features.columns:
                df["_data_warnings"] = features["_data_warnings"].values
            return df
        try:
            zz500_codes = get_index_constituents("zz500")
        except Exception:
            zz500_codes = []
        all_codes = list(set(holding_codes + zz500_codes))
        features = build_latest_features(all_codes, include_fundamentals=True,
                                         intraday_bars=intraday_bars)
        if features.empty:
            return pd.DataFrame()

        # If universe data quality is bad, fall back to holdings-only with baidu
        dq = features.attrs.get("_data_quality", 1.0)
        if dq < 0.5:
            h_features = build_latest_features(holding_codes, include_fundamentals=True,
                                               intraday_bars=intraday_bars)
            if h_features.empty:
                return pd.DataFrame()
            raw = ranker.predict_raw(h_features)
            fallback = pd.DataFrame({
                "code": h_features["code"].values,
                "ml_score": raw,
                "rank_pct": float("nan"),
                "raw_return": raw,
            })
            if "_data_warnings" in h_features.columns:
                fallback["_data_warnings"] = h_features["_data_warnings"].values
            return fallback

        blend_scores = ranker.predict(features)
        raw_scores = ranker.predict_raw(features)
        full_df = pd.DataFrame({
            "code": features["code"].values,
            "ml_score": blend_scores,
            "rank_pct": blend_scores,
            "raw_return": raw_scores,
        })
        if "_data_warnings" in features.columns:
            full_df["_data_warnings"] = features["_data_warnings"].values
        return full_df[full_df["code"].isin(holding_codes)].reset_index(drop=True)

    # --- Helper: evaluate holdings (same logic as daily_report) ---
    def _evaluate_holdings_web(ranker, regime=None, intraday_bars=None):
        holdings = _load_holdings_web()
        if not holdings:
            return pd.DataFrame()
        codes = [h["code"] for h in holdings]
        name_map = {h["code"]: h["name"] for h in holdings}
        is_rank = getattr(ranker, "score_type", "predicted_return") == "rank_percentile"

        if is_rank:
            result = _score_universe_web(ranker, codes, intraday_bars=intraday_bars)
            if result.empty:
                return pd.DataFrame()
            features = build_latest_features(codes, include_fundamentals=True,
                                             intraday_bars=intraday_bars)
        else:
            features = build_latest_features(codes, include_fundamentals=True,
                                             intraday_bars=intraday_bars)
            if features.empty:
                return pd.DataFrame()
            scores = ranker.predict(features)
            result = pd.DataFrame({
                "code": features["code"].values,
                "ml_score": scores,
                "rank_pct": pd.Series(scores).rank(pct=True).values,
                "raw_return": scores,
            })
            if "_data_warnings" in features.columns:
                result["_data_warnings"] = features["_data_warnings"].values

        # Merge _data_warnings from features if not already present (is_rank branch)
        if "_data_warnings" not in result.columns and not features.empty and "_data_warnings" in features.columns:
            warn_map = dict(zip(features["code"].astype(str).str.zfill(6), features["_data_warnings"]))
            result["_data_warnings"] = result["code"].map(warn_map).fillna("")

        result["name"] = result["code"].map(name_map)
        result = result.sort_values("ml_score", ascending=False).reset_index(drop=True)

        if is_rank:
            shift = regime.score * 0.05 if regime else 0.0
            def suggest_rank(rank):
                if rank > 0.85 - shift: return "加仓"
                elif rank > 0.55 - shift: return "持有"
                elif rank > 0.25 - shift: return "减仓"
                else: return "清仓"
            result["suggestion"] = result["rank_pct"].apply(suggest_rank)
            result["predicted_return"] = (result["raw_return"] * 100).round(2)
        else:
            shift = regime.score * 0.02 if regime else 0.0
            def suggest_ret(score):
                if score > 0.03 - shift: return "加仓"
                elif score > 0.00 - shift: return "持有"
                elif score > -0.03 - shift: return "减仓"
                else: return "清仓"
            result["suggestion"] = result["ml_score"].apply(suggest_ret)
            result["predicted_return"] = (result["ml_score"] * 100).round(2)

        result["rank_pct"] = result["rank_pct"].round(4)

        # Top factors
        importance = ranker.feature_importance_report()
        top_factors = importance.head(5)["feature"].tolist() if not importance.empty else []
        result["top_factors"] = ""
        for idx, row in result.iterrows():
            feat_row = features[features["code"] == row["code"]]
            if not feat_row.empty and top_factors:
                parts = []
                for f in top_factors[:3]:
                    if f in feat_row.columns:
                        val = feat_row[f].values[0]
                        if pd.notna(val):
                            parts.append(f"{f}={val:.3f}")
                result.at[idx, "top_factors"] = ", ".join(parts)
        return result

    # --- Helper: recommend stocks (same logic as daily_report) ---
    def _recommend_stocks_web(ranker, n_recommend=5, intraday_bars=None):
        try:
            codes = get_index_constituents("zz500")
        except Exception:
            return pd.DataFrame(), [], {}
        features = build_latest_features(codes, include_fundamentals=True,
                                         intraday_bars=intraday_bars)
        if features.empty:
            return pd.DataFrame(), [], {}

        # Data quality gate: degrade if fundamentals largely missing
        dq = features.attrs.get("_data_quality", 1.0)
        if dq < 0.5:
            degraded = pd.DataFrame({"_degraded_reason": ["数据源异常：PE/PB等基本面数据大面积缺失，荐股结果不可信"]})
            return degraded, [], {}

        scores = ranker.predict(features)
        result = pd.DataFrame({"code": features["code"].values, "ml_score": scores})
        if "_data_warnings" in features.columns:
            result["_data_warnings"] = features["_data_warnings"].values
        is_rank = getattr(ranker, "score_type", "predicted_return") == "rank_percentile"
        if is_rank:
            raw = ranker.predict_raw(features)
            result["predicted_return"] = (pd.Series(raw) * 100).round(2)
            result["rank_pct"] = result["ml_score"].round(4)
        else:
            result["predicted_return"] = (result["ml_score"] * 100).round(2)
            result["rank_pct"] = pd.Series(scores).rank(pct=True).round(4)
        result = result.sort_values("ml_score", ascending=False).head(n_recommend).reset_index(drop=True)
        # Fetch names
        name_map = {}
        try:
            import httpx
            syms = [("sh" if c.startswith("6") else "sz") + c for c in result["code"].tolist()]
            url = f"https://hq.sinajs.cn/list={','.join(syms)}"
            resp = httpx.get(url, headers={"Referer": "https://finance.sina.com.cn"}, timeout=10)
            for line in resp.text.strip().split("\n"):
                if "=" not in line: continue
                vp, dp = line.split("=", 1)
                c = vp.split("_")[-1][2:]
                fs = dp.strip('";\r').split(",")
                if fs and fs[0]:
                    name_map[c] = fs[0]
        except Exception:
            pass
        for c in result["code"]:
            if c not in name_map:
                name_map[c] = c
        result["name"] = result["code"].map(name_map)

        # Top factors
        importance = ranker.feature_importance_report()
        top_factors = importance.head(5)["feature"].tolist() if not importance.empty else []
        result["top_factors"] = ""
        for idx, row in result.iterrows():
            feat_row = features[features["code"] == row["code"]]
            if not feat_row.empty and top_factors:
                parts = []
                for f in top_factors[:3]:
                    if f in feat_row.columns:
                        val = feat_row[f].values[0]
                        if pd.notna(val):
                            parts.append(f"{f}={val:.3f}")
                result.at[idx, "top_factors"] = ", ".join(parts)

        rec_60d = _evaluate_holdings_60d_web(result["code"].tolist())
        return result, rec_60d, name_map

    # --- Helper: rebalance advice (same logic as daily_report) ---
    def _generate_rebalance_advice_web(holdings_eval, recommendations, regime=None):
        from mp.regime import ROUND_TRIP_COST
        advice = []
        if holdings_eval.empty:
            return advice
        is_rank = "rank_pct" in holdings_eval.columns and holdings_eval["ml_score"].max() <= 1.05
        swap_edge = 0.05 if is_rank else ROUND_TRIP_COST + 0.005
        if regime and regime.regime == "bull":
            swap_edge *= 0.7
        elif regime and regime.regime == "bear":
            swap_edge *= 1.5
        held_codes = set(holdings_eval["code"].tolist())
        candidates = []
        if regime and regime.regime == "bear":
            pass
        elif not recommendations.empty:
            for _, r in recommendations.iterrows():
                if r["code"] not in held_codes:
                    candidates.append(r)
        sorted_h = holdings_eval.sort_values("ml_score", ascending=True).reset_index(drop=True)
        ci = 0
        for _, h in sorted_h.iterrows():
            h_score = h["ml_score"]
            h_ret = h["predicted_return"]
            sug = h["suggestion"]
            if sug == "清仓":
                entry = {"action": "清仓", "sell_code": h["code"], "sell_name": h["name"],
                         "sell_score": h_ret, "buy_code": None, "buy_name": None,
                         "buy_score": None, "reason": f"预测收益 {h_ret:+.2f}%，模型看空"}
                if ci < len(candidates):
                    c = candidates[ci]
                    if c["ml_score"] > h_score + swap_edge:
                        entry.update({"action": "换仓", "buy_code": c["code"], "buy_name": c["name"],
                                      "buy_score": c["predicted_return"],
                                      "reason": f"预测收益 {h_ret:+.2f}% → {c['predicted_return']:+.2f}%"})
                        ci += 1
                advice.append(entry)
            elif sug == "减仓":
                if ci < len(candidates) and candidates[ci]["ml_score"] > h_score + swap_edge:
                    c = candidates[ci]
                    advice.append({"action": "换仓", "sell_code": h["code"], "sell_name": h["name"],
                                   "sell_score": h_ret, "buy_code": c["code"], "buy_name": c["name"],
                                   "buy_score": c["predicted_return"],
                                   "reason": f"预测 {h_ret:+.2f}% → {c['predicted_return']:+.2f}%"})
                    ci += 1
                else:
                    advice.append({"action": "保持", "sell_code": h["code"], "sell_name": h["name"],
                                   "sell_score": h_ret, "buy_code": None, "buy_name": None,
                                   "buy_score": None, "reason": f"偏弱({h_ret:+.2f}%)，无更优替代"})
            else:
                if ci < len(candidates) and candidates[ci]["ml_score"] > h_score + swap_edge * 2:
                    c = candidates[ci]
                    advice.append({"action": "换仓", "sell_code": h["code"], "sell_name": h["name"],
                                   "sell_score": h_ret, "buy_code": c["code"], "buy_name": c["name"],
                                   "buy_score": c["predicted_return"],
                                   "reason": f"发现更优标的: {h_ret:+.2f}% → {c['predicted_return']:+.2f}%"})
                    ci += 1
                else:
                    advice.append({"action": "保持", "sell_code": h["code"], "sell_name": h["name"],
                                   "sell_score": h_ret, "buy_code": None, "buy_name": None,
                                   "buy_score": None, "reason": f"预测收益 {h_ret:+.2f}%，{sug}"})
        return advice

    # ---- Check model availability ----
    _blend_ok = os.path.exists("data/blend_primary.lgb") and os.path.exists("data/blend_extreme.lgb")
    _single_ok = os.path.exists("data/model.lgb")

    if not _blend_ok and not _single_ok:
        st.warning("未找到已训练模型。请先在「ML模型」页训练模型后再使用日报功能。")
    else:
        # Model selector
        _model_options = []
        if _blend_ok:
            _model_options.append("BlendRanker (推荐)")
        if _single_ok:
            _model_options.append("StockRanker (单模型)")
        _model_choice = st.radio("选择模型", _model_options, horizontal=True, key="report_model")

        col_rt, col_act = st.columns([1, 1])
        with col_rt:
            _use_realtime = st.checkbox("注入盘中实时数据 (交易时段)", value=True, key="report_rt",
                                         help="从 Sina 获取上午盘中数据注入 ML 特征，使预测更及时")
        with col_act:
            _n_rec = st.slider("推荐股票数", 3, 10, 5, key="report_nrec")

        if st.button("生成报告", key="report_gen", type="primary"):
            # --- Step 1: Load model ---
            with st.spinner("加载模型..."):
                if "BlendRanker" in _model_choice:
                    ranker = BlendRanker()
                else:
                    ranker = StockRanker()
                    ranker.load()

            # --- Step 2: Detect regime ---
            with st.spinner("检测市场状态..."):
                try:
                    regime = RegimeDetector().detect()
                except Exception as e:
                    st.warning(f"市场状态检测失败: {e}")
                    regime = None

            # --- Step 3: Fetch realtime data ---
            intraday_bars = None
            prices = {}
            if _use_realtime:
                with st.spinner("获取实时行情..."):
                    holdings = _load_holdings_web()
                    h_codes = [h["code"] for h in holdings]
                    try:
                        zz500_codes = get_index_constituents("zz500")
                    except Exception:
                        zz500_codes = []
                    all_codes = list(set(h_codes + zz500_codes))
                    prices = _fetch_realtime_prices_web(all_codes)
                    intraday_bars = {}
                    for code, p in prices.items():
                        if p.get("open") and p["open"] > 0:
                            intraday_bars[code] = {
                                "date": date.today(),
                                "open": p["open"],
                                "high": p["high"],
                                "low": p["low"],
                                "close": p["price"],
                                "volume": p["volume"],
                                "amount": p["amount"],
                            }
                    if intraday_bars:
                        st.info(f"已注入 {len(intraday_bars)} 只股票盘中数据")
                    else:
                        st.info("非交易时段或数据不可用，使用历史收盘数据")
                        intraday_bars = None

            # --- Step 4: Display regime ---
            if regime:
                st.subheader("市场状态")
                _regime_colors = {"bull": "green", "sideways": "orange", "bear": "red"}
                _rc = _regime_colors.get(regime.regime, "gray")
                st.markdown(f"### :{_rc}[{regime.label_cn}] (综合评分: {regime.score:+.2f})")
                st.caption(regime.summary_cn)

                # Signal table
                sig_rows = []
                for sig_name, sig_data in regime.signals.items():
                    if isinstance(sig_data, dict):
                        sig_rows.append({
                            "信号": sig_name,
                            "值": sig_data.get("value", ""),
                            "方向": sig_data.get("direction", ""),
                            "详情": sig_data.get("detail", ""),
                        })
                if sig_rows:
                    st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

            st.divider()

            # --- Step 5: Evaluate holdings ---
            with st.spinner("评估持仓 (含ZZ500打分)..."):
                holdings_eval = _evaluate_holdings_web(ranker, regime=regime, intraday_bars=intraday_bars)

            if not holdings_eval.empty:
                st.subheader("持仓评估")

                # Data quality warning
                if "_data_warnings" in holdings_eval.columns:
                    warned = holdings_eval[holdings_eval["_data_warnings"].fillna("") != ""]
                    if not warned.empty:
                        warn_names = ", ".join(warned["name"].tolist())
                        st.warning(f"⚠ 数据源异常: {warn_names} 缺少基本面数据(PE/PB等)，相关预测可能不准确")

                # Merge realtime prices
                if prices:
                    for idx, row in holdings_eval.iterrows():
                        code = row["code"]
                        if code in prices:
                            holdings_eval.at[idx, "realtime_price"] = prices[code]["price"]
                            holdings_eval.at[idx, "realtime_change_pct"] = prices[code]["change_pct"]

                is_rank = getattr(ranker, "score_type", "predicted_return") == "rank_percentile"
                # If rank_pct is all NaN (data degraded), don't show rank column
                _has_valid_rank = is_rank and "rank_pct" in holdings_eval.columns and holdings_eval["rank_pct"].notna().any()

                # Display table
                disp_cols = ["name", "predicted_return"]
                disp_names = ["名称", "预测收益%"]
                if _has_valid_rank:
                    disp_cols.insert(2, "rank_pct")
                    disp_names.insert(2, "排名百分位")
                if "realtime_change_pct" in holdings_eval.columns:
                    disp_cols.append("realtime_change_pct")
                    disp_names.append("今日涨跌%")
                disp_cols += ["suggestion", "top_factors"]
                disp_names += ["建议", "主要因子"]

                disp = holdings_eval[disp_cols].copy()
                disp.columns = disp_names
                # Mark predictions with missing data
                if "_data_warnings" in holdings_eval.columns:
                    warns = holdings_eval["_data_warnings"].fillna("").values
                    disp["预测收益%"] = [
                        f"{v:+.2f}% [缺]" if w else f"{v:+.2f}%"
                        for v, w in zip(holdings_eval["predicted_return"], warns)
                    ]
                else:
                    disp["预测收益%"] = disp["预测收益%"].apply(lambda x: f"{x:+.2f}%")
                if "排名百分位" in disp.columns:
                    disp["排名百分位"] = disp["排名百分位"].apply(lambda x: f"前{(1-x)*100:.0f}%")
                if "今日涨跌%" in disp.columns:
                    disp["今日涨跌%"] = disp["今日涨跌%"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")

                # Color-code suggestions
                def _sug_style(val):
                    colors = {"加仓": "background-color: #c8e6c9", "持有": "background-color: #fff9c4",
                              "减仓": "background-color: #ffccbc", "清仓": "background-color: #ef9a9a"}
                    return colors.get(val, "")
                st.dataframe(disp.style.applymap(_sug_style, subset=["建议"]),
                             use_container_width=True, hide_index=True)
            else:
                st.warning("持仓评估失败")

            # --- Step 6: 60d technical analysis ---
            if not holdings_eval.empty:
                with st.spinner("计算60日技术分析..."):
                    h60 = _evaluate_holdings_60d_web(holdings_eval["code"].tolist())

                h60_valid = [r for r in h60 if "error" not in r]
                if h60_valid:
                    st.subheader("60日技术分析")
                    h60_df = pd.DataFrame(h60_valid)
                    # Map code to name
                    _name_map = dict(zip(holdings_eval["code"], holdings_eval["name"]))
                    h60_df["名称"] = h60_df["code"].map(_name_map)
                    h60_disp = h60_df[["名称", "price_position", "pct_vs_ma20", "pct_vs_ma60",
                                       "above_ma60", "ma60_rising", "momentum_60d",
                                       "momentum_first40", "momentum_last20"]].copy()
                    h60_disp.columns = ["名称", "价格位置%", "vs MA20%", "vs MA60%",
                                        "站上MA60", "MA60上升", "60日动量%",
                                        "前40日%", "后20日%"]
                    h60_disp["站上MA60"] = h60_disp["站上MA60"].map({True: "是", False: "否"})
                    h60_disp["MA60上升"] = h60_disp["MA60上升"].map({True: "是", False: "否", None: "-"})
                    st.dataframe(h60_disp, use_container_width=True, hide_index=True)

            st.divider()

            # --- Step 7: ZZ500 recommendations ---
            with st.spinner(f"扫描ZZ500推荐 (Top {_n_rec})..."):
                recs, rec_60d, rec_name_map = _recommend_stocks_web(ranker, n_recommend=_n_rec,
                                                                      intraday_bars=intraday_bars)

            _recs_degraded = "_degraded_reason" in recs.columns if not recs.empty else False
            if _recs_degraded:
                st.subheader("推荐关注")
                st.error(f"🔸 荐股模块已降级: {recs['_degraded_reason'].values[0]}")
            elif not recs.empty:
                st.subheader("推荐关注")

                # Data quality warning for recommendations
                if "_data_warnings" in recs.columns:
                    rec_warned = recs[recs["_data_warnings"].fillna("") != ""]
                    if not rec_warned.empty:
                        rw_names = ", ".join(rec_warned["name"].dropna().tolist()[:5])
                        st.warning(f"⚠ 数据源异常: {rw_names} 等缺少基本面数据，预测可能不准确")

                # Merge realtime prices for recs
                if prices:
                    for idx, row in recs.iterrows():
                        code = row["code"]
                        if code in prices:
                            recs.at[idx, "realtime_change_pct"] = prices[code]["change_pct"]

                rec_disp_cols = ["name", "predicted_return"]
                rec_disp_names = ["名称", "预测收益%"]
                if is_rank:
                    rec_disp_cols.append("rank_pct")
                    rec_disp_names.append("排名百分位")
                if "realtime_change_pct" in recs.columns:
                    rec_disp_cols.append("realtime_change_pct")
                    rec_disp_names.append("今日涨跌%")
                rec_disp_cols.append("top_factors")
                rec_disp_names.append("主要因子")

                rec_disp = recs[rec_disp_cols].copy()
                rec_disp.columns = rec_disp_names
                # Mark predictions with missing data
                if "_data_warnings" in recs.columns:
                    rec_warns = recs["_data_warnings"].fillna("").values
                    rec_disp["预测收益%"] = [
                        f"{v:+.2f}% [缺]" if w else f"{v:+.2f}%"
                        for v, w in zip(recs["predicted_return"], rec_warns)
                    ]
                else:
                    rec_disp["预测收益%"] = rec_disp["预测收益%"].apply(lambda x: f"{x:+.2f}%")
                if "排名百分位" in rec_disp.columns:
                    rec_disp["排名百分位"] = rec_disp["排名百分位"].apply(lambda x: f"前{(1-x)*100:.0f}%")
                if "今日涨跌%" in rec_disp.columns:
                    rec_disp["今日涨跌%"] = rec_disp["今日涨跌%"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
                st.dataframe(rec_disp, use_container_width=True, hide_index=True)

                # 60d for recommendations
                rec60_valid = [r for r in rec_60d if "error" not in r]
                if rec60_valid:
                    with st.expander("推荐股票60日技术面"):
                        r60_df = pd.DataFrame(rec60_valid)
                        r60_df["名称"] = r60_df["code"].map(rec_name_map)
                        r60_disp = r60_df[["名称", "price_position", "pct_vs_ma60",
                                           "ma60_rising", "momentum_60d", "momentum_last20"]].copy()
                        r60_disp.columns = ["名称", "价格位置%", "vs MA60%", "MA60上升",
                                            "60日动量%", "近20日%"]
                        r60_disp["MA60上升"] = r60_disp["MA60上升"].map({True: "是", False: "否", None: "-"})
                        st.dataframe(r60_disp, use_container_width=True, hide_index=True)
            else:
                st.warning("ZZ500扫描失败")

            st.divider()

            # --- Step 8: Rebalance advice ---
            if not holdings_eval.empty:
                st.subheader("换仓建议")
                advice = _generate_rebalance_advice_web(holdings_eval, recs if not recs.empty else pd.DataFrame(), regime=regime)

                if advice:
                    _action_emoji = {"换仓": "🔄", "清仓": "🔴", "保持": "🟢"}
                    adv_rows = []
                    for a in advice:
                        row = {
                            "操作": f"{_action_emoji.get(a['action'], '')} {a['action']}",
                            "卖出": f"{a['sell_name']} ({a['sell_score']:+.2f}%)",
                        }
                        if a["buy_code"]:
                            row["买入"] = f"{a['buy_name']} ({a['buy_score']:+.2f}%)"
                        else:
                            row["买入"] = "-"
                        row["理由"] = a["reason"]
                        adv_rows.append(row)
                    st.dataframe(pd.DataFrame(adv_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("无换仓建议")

            # --- Step 9: Signal screening ---
            if not holdings_eval.empty:
                with st.expander("持仓信号评分详情"):
                    h_codes = holdings_eval["code"].tolist()
                    h_name_map = dict(zip(holdings_eval["code"], holdings_eval["name"]))
                    with st.spinner("扫描信号..."):
                        screened = screen_stocks(h_codes, names=h_name_map, use_ml=False)
                    if not screened.empty:
                        scr_disp = screened[["name", "price", "chg_5d", "chg_20d",
                                             "bull", "bear", "signal_score", "rating"]].copy()
                        scr_disp.columns = ["名称", "现价", "5日%", "20日%",
                                            "看多", "看空", "综合分", "评级"]
                        scr_disp["5日%"] = scr_disp["5日%"].apply(lambda x: f"{x:+.1f}%")
                        scr_disp["20日%"] = scr_disp["20日%"].apply(lambda x: f"{x:+.1f}%")
                        scr_disp["现价"] = scr_disp["现价"].apply(lambda x: f"{x:.2f}")
                        scr_disp["综合分"] = scr_disp["综合分"].apply(lambda x: f"{x:+.3f}")
                        st.dataframe(scr_disp, use_container_width=True, hide_index=True)

            st.divider()
            st.caption(f"报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | "
                       f"模型: {_model_choice} | "
                       f"{'已注入盘中数据' if intraday_bars else '使用历史收盘数据'}")
