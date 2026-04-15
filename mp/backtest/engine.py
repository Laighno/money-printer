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

from ..account.broker import FeeSchedule, SimulatedBroker
from ..risk.manager import PositionState, RiskGuard, RiskParams
from ..rotation.signals import generate_rotation_signals


def _get_rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    if freq == "daily":
        return dates.tolist()
    elif freq == "weekly":
        iso = dates.isocalendar()
        return dates.to_series().groupby([iso.year, iso.week]).last().tolist()
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
    """Pre-compute rotation signals for each rebalance date.

    Uses data strictly before the rebalance date (date < dt) to avoid
    look-ahead bias — signals should not include the rebalance day's close.
    """
    signals_cache = {}
    sorted_dates = sorted(rebalance_dates)
    for dt in sorted_dates:
        hist = industry_bars[industry_bars["date"] < dt]
        signals_cache[dt] = generate_rotation_signals(hist)
    return signals_cache


def _build_risk_positions(broker: SimulatedBroker) -> dict[str, PositionState]:
    """Bridge BrokerPosition → PositionState for RiskGuard checks."""
    total = broker.total_value
    return {
        code: PositionState(
            board_name=code,
            entry_price=pos.avg_cost,
            entry_date=pos.entry_date,
            current_price=pos.current_price,
            peak_price=pos.peak_price,
            weight=pos.market_value / total if total > 0 else 0,
        )
        for code, pos in broker.positions.items()
    }


def run_backtest(
    industry_bars: pd.DataFrame,
    rebalance_freq: str = "monthly",
    top_n: int = 5,
    risk_params: RiskParams | None = None,
    close_map: dict | None = None,
    signals_cache: dict | None = None,
    silent: bool = False,
    initial_capital: float = 1_000_000,
    fees: FeeSchedule | None = None,
) -> pd.DataFrame:
    """Run industry rotation backtest.

    Args:
        industry_bars: board_name, date, open, high, low, close, volume, amount, turnover
        rebalance_freq: daily / weekly / monthly
        top_n: number of sectors to hold
        risk_params: risk management parameters
        close_map: pre-built close price lookup (optimization for repeated calls)
        signals_cache: pre-built signals per rebalance date (optimization for repeated calls)
        initial_capital: starting cash for the broker
        fees: fee schedule (defaults to zero fees for backward compatibility)

    Returns:
        DataFrame with date, daily_return, nav, cumulative_return
    """
    if risk_params is None:
        risk_params = RiskParams(max_sectors=top_n)

    if fees is None:
        fees = FeeSchedule(slippage_bps=0, commission_bps=0,
                           stamp_tax_bps_old=0, stamp_tax_bps_new=0)

    broker = SimulatedBroker(initial_capital, fees, silent=silent)
    guard = RiskGuard(risk_params)

    all_dates = sorted(industry_bars["date"].unique())
    rebalance_dates = set(_get_rebalance_dates(pd.DatetimeIndex(all_dates), rebalance_freq))

    if close_map is None:
        close_map = _build_close_map(industry_bars)

    if signals_cache is None:
        signals_cache = _precompute_signals(industry_bars, rebalance_dates)

    portfolio_returns = []
    prev_total: float | None = None
    nav_peak: float = initial_capital

    for dt in all_dates:
        is_rebalance = dt in rebalance_dates
        dt_str = str(dt.date()) if hasattr(dt, 'date') else str(dt)[:10]

        # --- 1. Update prices to today's close ---
        close_prices = {}
        for code in list(broker.positions.keys()):
            price = close_map.get((code, dt))
            if price is not None:
                close_prices[code] = price
        if close_prices:
            broker.update_prices(close_prices)

        # --- 2. Record daily return BEFORE any trading ---
        if prev_total is not None and prev_total > 0:
            day_return = broker.total_value / prev_total - 1
            portfolio_returns.append({"date": dt, "daily_return": day_return})

        # --- 3. Check stops ---
        risk_positions = _build_risk_positions(broker)
        for code in list(risk_positions.keys()):
            should_exit, reason = guard.check_exit(code, risk_positions)
            if should_exit:
                sell_price = close_prices.get(code)
                if sell_price and sell_price > 0:
                    broker.sell(code, sell_price, date=dt_str)

        # --- 4. Update nav peak and circuit breaker ---
        nav_peak = max(nav_peak, broker.total_value)
        drawdown = broker.total_value / nav_peak - 1 if nav_peak > 0 else 0
        guard.check_circuit_breaker(drawdown, silent)

        # --- 5. Rebalance: signals use data < dt (no look-ahead); all execution at close ---
        if is_rebalance and not guard.circuit_breaker_active:
            signals = signals_cache.get(dt)
            if signals is None:
                hist = industry_bars[industry_bars["date"] < dt]
                signals = generate_rotation_signals(hist)

            target_sectors = signals[signals["composite_score"] > 0].head(top_n).index.tolist()

            # Sell rotated-out positions at close
            for code in list(broker.positions.keys()):
                if code not in target_sectors:
                    sell_price = close_prices.get(code)
                    if sell_price and sell_price > 0:
                        broker.sell(code, sell_price, date=dt_str)

            # Buy new entries at close
            risk_positions = _build_risk_positions(broker)
            for board_name in target_sectors:
                can_enter, msg = guard.check_entry(board_name, risk_positions)
                if not can_enter:
                    continue

                price = close_map.get((board_name, dt))
                if price is None or price <= 0:
                    continue

                vol = signals.loc[board_name, "volatility_20d"] if "volatility_20d" in signals.columns else 20.0
                weight = guard.calc_position_size(len(broker.positions), vol)
                target_value = weight * broker.total_value
                broker.buy(board_name, price, target_value=target_value, date=dt_str)

        prev_total = broker.total_value

    result = pd.DataFrame(portfolio_returns)
    if not result.empty:
        result["nav"] = (1 + result["daily_return"]).cumprod()
        result["cumulative_return"] = result["nav"] - 1
    return result


def optimize_params(
    industry_bars: pd.DataFrame,
    metric: str = "sharpe",
    initial_capital: float = 1_000_000,
    fees: FeeSchedule | None = None,
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
                            initial_capital=initial_capital,
                            fees=fees,
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
