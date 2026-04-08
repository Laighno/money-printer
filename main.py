"""CLI entry point - Industry Rotation Strategy."""

from loguru import logger

from mp.backtest.engine import calc_performance, run_backtest
from mp.config import load_settings
from mp.data.fetcher import get_industry_list, get_industry_hist_batch
from mp.data.store import DataStore
from mp.risk.manager import RiskParams
from mp.rotation.signals import generate_rotation_signals


def main():
    settings = load_settings()
    store = DataStore(settings.data.db_url)

    # 1. Fetch industry data
    if not store.has_industry_bars():
        logger.info("Fetching industry board list...")
        boards = get_industry_list()
        board_names = boards["board_name"].tolist()

        logger.info(f"Fetching K-line history for {len(board_names)} boards...")
        start = settings.strategy.start_date.replace("-", "")
        hist = get_industry_hist_batch(board_names, start)
        store.save_industry_bars(hist)
    else:
        logger.info("Using cached industry data")

    industry_bars = store.load_industry_bars(start=settings.strategy.start_date)
    logger.info(f"Loaded {len(industry_bars)} industry bar rows, {industry_bars['board_name'].nunique()} boards")

    # 2. Show current signals
    logger.info("=== Current Rotation Signals (Top 10) ===")
    signals = generate_rotation_signals(industry_bars)
    display_cols = ["mom_5d", "mom_20d", "mom_60d", "vol_ratio_5d", "ma_alignment", "composite_score"]
    available = [c for c in display_cols if c in signals.columns]
    top10 = signals.head(10)[available]
    for name, row in top10.iterrows():
        mom20 = f"{row.get('mom_20d', 0):.1f}%" if 'mom_20d' in row else "N/A"
        vol_r = f"{row.get('vol_ratio_5d', 0):.2f}x" if 'vol_ratio_5d' in row else "N/A"
        ma = {1.0: "多头", 0.0: "震荡", -1.0: "空头"}.get(row.get('ma_alignment'), "N/A")
        score = f"{row.get('composite_score', 0):.3f}"
        logger.info(f"  {name:8s} | 20d动量: {mom20:>8s} | 量比: {vol_r:>6s} | 趋势: {ma} | 得分: {score}")

    # 3. Run backtest
    logger.info("Running backtest...")
    risk_params = RiskParams(
        max_position_pct=settings.risk.max_position_pct,
        stop_loss_pct=settings.risk.stop_loss_pct,
        trailing_stop_pct=settings.risk.trailing_stop_pct,
        max_drawdown_pct=settings.risk.max_drawdown_pct,
        max_sectors=settings.risk.max_sectors,
        vol_target=settings.risk.vol_target,
    )

    result = run_backtest(
        industry_bars,
        rebalance_freq=settings.strategy.rebalance_freq,
        top_n=settings.strategy.top_n,
        risk_params=risk_params,
    )

    # 4. Report
    metrics = calc_performance(result)
    logger.info("=== Performance Report ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    result.to_csv("data/backtest_result.csv", index=False)
    logger.info("Result saved to data/backtest_result.csv")


if __name__ == "__main__":
    main()
