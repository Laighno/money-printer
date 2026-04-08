"""Backtest engine for industry rotation strategy.

Flow:
1. At each rebalance date, calculate rotation signals for all sectors
2. Select top-N sectors by composite score
3. Size positions using inverse-volatility weighting
4. Apply risk management (stop-loss, trailing stop, circuit breaker)
5. Track daily returns and performance
"""

import numpy as np
import pandas as pd
from loguru import logger

from ..risk.manager import RiskManager, RiskParams
from ..rotation.signals import generate_rotation_signals


def _get_rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    if freq == "daily":
        return dates.tolist()
    elif freq == "weekly":
        return dates.to_series().groupby(dates.isocalendar().week).last().tolist()
    elif freq == "monthly":
        return dates.to_series().groupby(dates.to_period("M")).last().tolist()
    else:
        raise ValueError(f"Unknown rebalance freq: {freq}")


def _build_close_map(industry_bars: pd.DataFrame) -> dict[tuple[str, any], float]:
    """Build close price lookup dict using vectorized zip instead of iterrows."""
    return dict(zip(
        zip(industry_bars["board_name"], industry_bars["date"]),
        industry_bars["close"],
    ))


def _precompute_signals(industry_bars: pd.DataFrame, rebalance_dates: set) -> dict:
    """Pre-compute rotation signals for each rebalance date."""
    signals_cache = {}
    sorted_dates = sorted(rebalance_dates)
    for dt in sorted_dates:
        hist = industry_bars[industry_bars["date"] <= dt]
        signals_cache[dt] = generate_rotation_signals(hist)
    return signals_cache


def run_backtest(
    industry_bars: pd.DataFrame,
    rebalance_freq: str = "monthly",
    top_n: int = 5,
    risk_params: RiskParams | None = None,
    close_map: dict | None = None,
    signals_cache: dict | None = None,
    silent: bool = False,
) -> pd.DataFrame:
    """Run industry rotation backtest.

    Args:
        industry_bars: board_name, date, open, high, low, close, volume, amount, turnover
        rebalance_freq: daily / weekly / monthly
        top_n: number of sectors to hold
        risk_params: risk management parameters
        close_map: pre-built close price lookup (optimization for repeated calls)
        signals_cache: pre-built signals per rebalance date (optimization for repeated calls)

    Returns:
        DataFrame with date, daily_return, nav, cumulative_return
    """
    if risk_params is None:
        risk_params = RiskParams(max_sectors=top_n)

    risk_mgr = RiskManager(risk_params, silent=silent)

    all_dates = sorted(industry_bars["date"].unique())
    rebalance_dates = set(_get_rebalance_dates(pd.DatetimeIndex(all_dates), rebalance_freq))

    if close_map is None:
        close_map = _build_close_map(industry_bars)

    if signals_cache is None:
        signals_cache = _precompute_signals(industry_bars, rebalance_dates)

    portfolio_returns = []
    prev_date = None

    for dt in all_dates:
        is_rebalance = dt in rebalance_dates

        # --- Daily: update prices and check stops ---
        current_prices = {}
        for board_name in list(risk_mgr.positions.keys()):
            price = close_map.get((board_name, dt))
            if price is not None:
                current_prices[board_name] = price

        if current_prices:
            risk_mgr.update_prices(current_prices)

        # Check stop-loss / trailing stop for each position
        for board_name in list(risk_mgr.positions.keys()):
            should_exit, reason = risk_mgr.check_exit(board_name)
            if should_exit:
                risk_mgr.exit_position(board_name, reason)

        # --- Rebalance: generate signals and rotate ---
        if is_rebalance and not risk_mgr.circuit_breaker_active:
            signals = signals_cache.get(dt)
            if signals is None:
                hist = industry_bars[industry_bars["date"] <= dt]
                signals = generate_rotation_signals(hist)

            # Target sectors: top N by composite score with positive score
            target_sectors = signals[signals["composite_score"] > 0].head(top_n).index.tolist()

            # Exit sectors no longer in target
            for board_name in list(risk_mgr.positions.keys()):
                if board_name not in target_sectors:
                    risk_mgr.exit_position(board_name, "Rotation: no longer in top-N")

            # Enter new sectors
            for board_name in target_sectors:
                can_enter, msg = risk_mgr.check_entry(board_name)
                if not can_enter:
                    continue

                price = close_map.get((board_name, dt))
                if price is None or price <= 0:
                    continue

                vol = signals.loc[board_name, "volatility_20d"] if "volatility_20d" in signals.columns else 20.0
                weight = risk_mgr.calc_position_size(board_name, vol)
                risk_mgr.enter_position(board_name, price, weight, str(dt.date()))

        # --- Calc daily return ---
        if prev_date is not None and risk_mgr.positions:
            day_return = 0.0
            for pos in risk_mgr.positions.values():
                prev_price = close_map.get((pos.board_name, prev_date))
                cur_price = close_map.get((pos.board_name, dt))
                if prev_price and cur_price and prev_price > 0:
                    stock_ret = (cur_price / prev_price - 1) * pos.weight
                    day_return += stock_ret

            portfolio_returns.append({"date": dt, "daily_return": day_return})

        prev_date = dt

    result = pd.DataFrame(portfolio_returns)
    if not result.empty:
        result["nav"] = (1 + result["daily_return"]).cumprod()
        result["cumulative_return"] = result["nav"] - 1
    return result


def optimize_params(
    industry_bars: pd.DataFrame,
    metric: str = "sharpe",
) -> dict:
    """Grid search over parameter combinations, return the best set.

    Pre-computes close_map and signals to avoid redundant work across runs.
    """
    param_grid = {
        "rebalance_freq": ["weekly", "monthly"],
        "top_n": [3, 5, 7],
        "stop_loss_pct": [0.05, 0.08, 0.12],
        "trailing_stop_pct": [0.10, 0.15, 0.20],
        "max_drawdown_pct": [0.12, 0.15, 0.20],
    }

    # Pre-compute shared data once
    close_map = _build_close_map(industry_bars)

    all_dates = sorted(industry_bars["date"].unique())
    all_signals_cache = {}
    for freq in param_grid["rebalance_freq"]:
        rebalance_dates = set(_get_rebalance_dates(pd.DatetimeIndex(all_dates), freq))
        all_signals_cache[freq] = _precompute_signals(industry_bars, rebalance_dates)

    best_score = -np.inf
    best_params = {}
    best_metrics = {}
    results_log = []

    total = 1
    for v in param_grid.values():
        total *= len(v)
    logger.info(f"Optimizing over ~{total} parameter combinations...")

    count = 0
    for freq in param_grid["rebalance_freq"]:
        signals_cache = all_signals_cache[freq]
        for top_n in param_grid["top_n"]:
            for sl in param_grid["stop_loss_pct"]:
                for ts in param_grid["trailing_stop_pct"]:
                    if ts <= sl:
                        continue
                    for mdd in param_grid["max_drawdown_pct"]:
                        count += 1
                        risk_params = RiskParams(
                            max_sectors=top_n,
                            stop_loss_pct=sl,
                            trailing_stop_pct=ts,
                            max_drawdown_pct=mdd,
                        )
                        result = run_backtest(
                            industry_bars,
                            rebalance_freq=freq,
                            top_n=top_n,
                            risk_params=risk_params,
                            close_map=close_map,
                            signals_cache=signals_cache,
                            silent=True,
                        )
                        if result.empty:
                            continue
                        perf = calc_performance(result)

                        if metric == "calmar":
                            score = float(perf.get("calmar_ratio", "0"))
                        else:
                            score = float(perf.get("sharpe_ratio", "0"))

                        combo = {
                            "rebalance_freq": freq,
                            "top_n": top_n,
                            "stop_loss_pct": sl,
                            "trailing_stop_pct": ts,
                            "max_drawdown_pct": mdd,
                        }
                        results_log.append({
                            **combo,
                            "sharpe": perf.get("sharpe_ratio"),
                            "calmar": perf.get("calmar_ratio"),
                            "annual_return": perf.get("annual_return"),
                            "max_drawdown": perf.get("max_drawdown"),
                        })

                        if score > best_score:
                            best_score = score
                            best_params = combo
                            best_metrics = perf

    logger.info(f"Tested {count} combinations, best {metric}: {best_score:.2f}")
    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "results_log": pd.DataFrame(results_log),
    }


def calc_performance(result: pd.DataFrame) -> dict:
    """Calculate key performance metrics."""
    if result.empty:
        return {}

    total_return = result["nav"].iloc[-1] - 1
    n_days = len(result)
    annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    daily_std = result["daily_return"].std()
    annual_vol = daily_std * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    nav = result["nav"]
    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    max_dd = drawdown.min()
    win_rate = (result["daily_return"] > 0).mean()

    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    return {
        "total_return": f"{total_return:.2%}",
        "annual_return": f"{annual_return:.2%}",
        "annual_volatility": f"{annual_vol:.2%}",
        "sharpe_ratio": f"{sharpe:.2f}",
        "calmar_ratio": f"{calmar:.2f}",
        "max_drawdown": f"{max_dd:.2%}",
        "win_rate": f"{win_rate:.2%}",
        "trading_days": n_days,
    }
