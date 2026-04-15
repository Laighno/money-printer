"""Individual-stock ML backtesting engine.

Runs a top-K stock-picking strategy driven by a pre-trained LightGBM model.
On each rebalance date the engine:

1. Computes rolling technical factors for every stock in the universe using
   data strictly up to that date (no look-ahead).
2. Feeds the features into ``StockRanker.predict()`` to obtain ML scores.
3. Selects the top-K highest-scoring stocks, applies position sizing, and
   generates buy/sell orders executed with slippage + commission.
4. Between rebalance dates, tracks daily NAV and checks per-stock stop-loss
   / trailing-stop triggers.

Usage::

    from mp.backtest.ml_backtest import run_ml_backtest

    result = run_ml_backtest(
        model_path="data/model.lgb",
        universe="zz500",
        start="2024-01-01",
        end="2026-01-01",
        top_k=10,
    )
    print(result["metrics"])
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from mp.account.broker import FeeSchedule, LOT_SIZE, SimulatedBroker
from mp.backtest.engine import calc_performance
from mp.data.fetcher import get_daily_bars, get_index_constituents
from mp.ml.dataset import (
    FACTOR_COLUMNS,
    TECHNICAL_COLUMNS,
    _compute_technical_factors,
)
from mp.ml.model import FEATURE_COLS, StockRanker

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Minimum number of price bars required for factor warm-up (longest rolling
# window is 60 days, but we add margin for safety).
_MIN_WARMUP_BARS = 80

# A-share lot size (imported from mp.account.broker)
# _LOT_SIZE is now LOT_SIZE from broker module


def _normalize_date(s: str) -> str:
    """Accept ``2024-01-01`` or ``20240101`` and return ``YYYYMMDD``."""
    return s.replace("-", "")


def _resolve_universe(universe: list[str] | str) -> list[str]:
    """Resolve *universe* to a concrete list of 6-digit stock codes."""
    if isinstance(universe, str):
        logger.info("Resolving universe from index: {}", universe)
        return get_index_constituents(universe)
    return list(universe)


def _get_rebalance_dates(
    trading_dates: list[pd.Timestamp],
    freq: str,
) -> list[pd.Timestamp]:
    """Return the subset of *trading_dates* that are rebalance triggers.

    ``"monthly"`` -> last trading day of each calendar month.
    ``"weekly"``  -> last trading day of each ISO week.
    """
    s = pd.Series(trading_dates, index=trading_dates)
    if freq == "monthly":
        return s.groupby(s.dt.to_period("M")).last().tolist()
    elif freq == "weekly":
        iso = s.dt.isocalendar()
        return s.groupby([iso.year, iso.week]).last().tolist()
    else:
        raise ValueError(f"Unsupported rebalance_freq: {freq!r}")


# ---------------------------------------------------------------------------
# Data pre-fetching (avoid look-ahead bias)
# ---------------------------------------------------------------------------

def _prefetch_bars(
    codes: list[str],
    start: str,
    end: str,
    progress_callback: Callable | None = None,
) -> dict[str, pd.DataFrame]:
    """Pre-fetch daily OHLCV bars for every stock in the universe.

    Returns ``{code: bars_df}`` where *bars_df* has columns
    ``[date, open, high, low, close, volume, amount, turnover]``
    sorted by date.
    """
    bars_map: dict[str, pd.DataFrame] = {}
    total = len(codes)
    for idx, code in enumerate(codes):
        try:
            df = get_daily_bars(code, start, end)
            if df is not None and not df.empty:
                df = df.sort_values("date").reset_index(drop=True)
                bars_map[code] = df
        except Exception:
            logger.debug("Failed to fetch bars for {}", code)
        if progress_callback:
            progress_callback(idx + 1, total, f"Fetching bars: {idx + 1}/{total}")
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            logger.info("Prefetch bars progress: {}/{}", idx + 1, total)
    logger.info("Prefetched bars for {}/{} stocks", len(bars_map), total)
    return bars_map


def _build_factor_panel(
    bars_map: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Compute rolling technical factors for every stock over the full date range.

    Returns ``{code: factors_df}`` where *factors_df* is indexed like the
    bars and contains columns ``[date] + TECHNICAL_COLUMNS``.

    Fundamental factors are filled with NaN here; they are quasi-static and
    the model was trained with them potentially missing (especially PE/PB
    which are snapshot-only).  For backtesting, the technical signal is the
    primary alpha driver and this avoids any forward-looking financial data.
    """
    factor_map: dict[str, pd.DataFrame] = {}
    for code, df in bars_map.items():
        if len(df) < _MIN_WARMUP_BARS:
            continue
        try:
            close = df["close"].values.astype(float)
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)
            volume = df["volume"].values.astype(float)
            open_arr = df["open"].values.astype(float)
            amount = (
                df["amount"].values.astype(float)
                if "amount" in df.columns
                else volume * close
            )
            turnover = (
                df["turnover"].values.astype(float)
                if "turnover" in df.columns
                else np.full(len(close), np.nan)
            )

            tech = _compute_technical_factors(
                close, high, low, volume, amount, turnover, open_arr,
            )

            fdf = pd.DataFrame({"date": df["date"].values})
            for col in TECHNICAL_COLUMNS:
                fdf[col] = tech[col]
            # Fill fundamental columns with NaN (model handles missing)
            for col in FACTOR_COLUMNS:
                if col not in fdf.columns:
                    fdf[col] = np.nan
            # Derive total_mv_log from daily bars (close * volume / turnover)
            valid_t = turnover > 0
            float_mv = np.where(valid_t, close * volume / turnover, np.nan)
            fdf["total_mv_log"] = np.where(float_mv > 0, np.log(float_mv), np.nan)
            fdf["code"] = code
            factor_map[code] = fdf
        except Exception:
            logger.debug("Factor computation failed for {}", code)
    logger.info("Built factor panel for {} stocks", len(factor_map))
    return factor_map


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def _calc_weights(
    scores: np.ndarray,
    sizing: str,
    volatilities: np.ndarray | None = None,
) -> np.ndarray:
    """Return target portfolio weights (summing to 1.0).

    Parameters
    ----------
    scores : array
        ML prediction scores for the selected stocks.
    sizing : str
        ``"equal"`` / ``"score_weighted"`` / ``"risk_parity"``.
    volatilities : array or None
        20-day realised volatility for each stock (needed for risk_parity).
    """
    n = len(scores)
    if n == 0:
        return np.array([])

    if sizing == "score_weighted":
        # Shift scores so they are all positive before weighting.
        shifted = scores - scores.min() + 1e-8
        weights = shifted / shifted.sum()
    elif sizing == "risk_parity" and volatilities is not None:
        inv_vol = np.where(
            (volatilities > 0) & np.isfinite(volatilities),
            1.0 / volatilities,
            0.0,
        )
        total = inv_vol.sum()
        weights = inv_vol / total if total > 0 else np.ones(n) / n
    else:
        # Default: equal weight
        weights = np.ones(n) / n

    return weights


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def run_ml_backtest(
    model_path: str | Path,
    universe: list[str] | str,
    start: str,
    end: str,
    rebalance_freq: str = "monthly",
    top_k: int = 10,
    initial_capital: float = 1_000_000,
    slippage_bps: float = 10,
    commission_bps: float = 5,
    stop_loss_pct: float | None = None,
    trailing_stop_pct: float | None = None,
    min_score: float | None = None,
    sizing: str = "equal",
    progress_callback: Callable | None = None,
) -> dict:
    """Run an individual-stock ML backtest.

    Parameters
    ----------
    model_path : str or Path
        Path to a pre-trained LightGBM model (``StockRanker.save()`` format).
    universe : list[str] or str
        Either a list of 6-digit stock codes or an index name recognised by
        ``get_index_constituents`` (e.g. ``"zz500"``).
    start, end : str
        Backtest window, e.g. ``"2024-01-01"`` or ``"20240101"``.
    rebalance_freq : str
        ``"weekly"`` or ``"monthly"``.
    top_k : int
        Number of stocks to hold after each rebalance.
    initial_capital : float
        Starting cash.
    slippage_bps : float
        One-way slippage in basis points.
    commission_bps : float
        One-way commission in basis points.
    stop_loss_pct : float or None
        If set, sell a stock when it drops this fraction from its entry price.
    trailing_stop_pct : float or None
        If set, sell a stock when it drops this fraction from its peak price
        since entry.
    min_score : float or None
        Minimum ML score required to enter a position.
    sizing : str
        Position sizing method: ``"equal"``, ``"score_weighted"``, or
        ``"risk_parity"``.
    progress_callback : callable or None
        ``callback(step, total, msg)`` for progress reporting.

    Returns
    -------
    dict with keys:
        - ``nav_series`` : pd.Series (date index -> NAV)
        - ``metrics``    : dict from ``calc_performance``
        - ``trade_log``  : list of trade dicts
        - ``turnover_stats`` : dict
        - ``positions_history`` : list of (date, {code: weight})
    """
    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    model_path = str(model_path)
    start_fmt = _normalize_date(start)
    end_fmt = _normalize_date(end)

    ranker = StockRanker(model_path=model_path)
    if not ranker.load(model_path):
        raise FileNotFoundError(f"Cannot load model from {model_path}")

    codes = _resolve_universe(universe)
    logger.info(
        "ML Backtest: {} stocks, {} ~ {}, freq={}, top_k={}, sizing={}",
        len(codes), start, end, rebalance_freq, top_k, sizing,
    )

    # ------------------------------------------------------------------
    # 1. Pre-fetch all bars (prevents any look-ahead in network calls)
    # ------------------------------------------------------------------
    # Extend fetch window back by 120 trading days for factor warm-up
    warmup_start = pd.Timestamp(start_fmt) - pd.offsets.BDay(150)
    warmup_start_str = warmup_start.strftime("%Y%m%d")

    bars_map = _prefetch_bars(codes, warmup_start_str, end_fmt, progress_callback)
    if not bars_map:
        raise RuntimeError("No price data fetched for any stock in the universe")

    # ------------------------------------------------------------------
    # 2. Pre-build full factor panel (all dates, all stocks)
    # ------------------------------------------------------------------
    factor_map = _build_factor_panel(bars_map)
    if not factor_map:
        raise RuntimeError("Factor computation failed for all stocks")

    # ------------------------------------------------------------------
    # 3. Determine trading calendar and rebalance dates
    # ------------------------------------------------------------------
    bt_start = pd.Timestamp(start_fmt)
    bt_end = pd.Timestamp(end_fmt)

    # Collect union of all trading dates within [start, end]
    all_dates_set: set[pd.Timestamp] = set()
    for df in bars_map.values():
        mask = (df["date"] >= bt_start) & (df["date"] <= bt_end)
        all_dates_set.update(df.loc[mask, "date"].tolist())
    trading_dates = sorted(all_dates_set)

    if len(trading_dates) < 2:
        raise RuntimeError(
            f"Fewer than 2 trading dates in [{start}, {end}] across universe"
        )

    rebalance_dates_set = set(
        _get_rebalance_dates(trading_dates, rebalance_freq)
    )
    logger.info(
        "Trading calendar: {} days, {} rebalance dates",
        len(trading_dates), len(rebalance_dates_set),
    )

    # ------------------------------------------------------------------
    # 4. Build close/open price lookup for fast access
    # ------------------------------------------------------------------
    # close_lookup[(code, date)] -> close price
    # open_lookup[(code, date)]  -> open price
    close_lookup: dict[tuple[str, pd.Timestamp], float] = {}
    open_lookup: dict[tuple[str, pd.Timestamp], float] = {}
    for code, df in bars_map.items():
        for _, row in df.iterrows():
            d = row["date"]
            close_lookup[(code, d)] = float(row["close"])
            open_lookup[(code, d)] = float(row["open"])

    # ------------------------------------------------------------------
    # 5. Main simulation loop
    # ------------------------------------------------------------------
    # Portfolio state — unified broker handles cash, positions, fees, trade log
    broker = SimulatedBroker(
        initial_capital,
        FeeSchedule(slippage_bps=slippage_bps, commission_bps=commission_bps),
        silent=True,
    )

    nav_records: list[dict] = []
    positions_history: list[tuple] = []
    prev_total_value = initial_capital
    pending_rebalance: list[tuple[str, float]] | None = None  # (code, target_weight)

    total_steps = len(trading_dates)

    for step, dt in enumerate(trading_dates):

        # --- 1. Execute pending rebalance from previous close at today's open ---
        if pending_rebalance is not None:
            target_map = dict(pending_rebalance)

            # Sell positions not in target at today's open
            for code in list(broker.positions.keys()):
                if code not in target_map:
                    price = open_lookup.get((code, dt), broker.positions[code].current_price)
                    broker.sell(code, price, date=dt, action="SELL (rebalance)")

            # Adjust / enter positions at today's open
            portfolio_value = broker.total_value

            for code, target_w in target_map.items():
                target_value = portfolio_value * target_w
                price = open_lookup.get((code, dt))
                if price is None or price <= 0:
                    continue

                if code in broker.positions:
                    current_value = broker.positions[code].shares * price
                    delta_value = target_value - current_value
                    if abs(delta_value) < price * LOT_SIZE * 0.5:
                        continue
                    if delta_value > 0:
                        broker.buy(code, price, target_value=target_value, date=dt, action="BUY (rebalance adjust)")
                    else:
                        sell_exec = broker.fees.sell_exec_price(price)
                        sell_shares = int(abs(delta_value) / sell_exec / LOT_SIZE) * LOT_SIZE
                        sell_shares = min(sell_shares, broker.positions[code].shares)
                        broker.sell(code, price, shares=sell_shares, date=dt, action="SELL (rebalance adjust)")
                else:
                    broker.buy(code, price, target_value=target_value, date=dt, action="BUY (rebalance)")

            pending_rebalance = None  # consumed

        # --- 2. Update current prices to today's close & peak prices ---
        day_prices: dict[str, float] = {}
        for code in broker.positions:
            price = close_lookup.get((code, dt))
            if price is not None:
                day_prices[code] = price
        broker.update_prices(day_prices)

        # --- 3. Check stop-loss / trailing-stop (between rebalances) ---
        if stop_loss_pct is not None or trailing_stop_pct is not None:
            codes_to_sell: list[str] = []
            for code, pos in broker.positions.items():
                cur = pos.current_price if pos.current_price > 0 else pos.avg_cost
                if stop_loss_pct is not None:
                    if cur <= pos.avg_cost * (1 - stop_loss_pct):
                        codes_to_sell.append(code)
                        continue
                if trailing_stop_pct is not None:
                    peak = pos.peak_price if pos.peak_price > 0 else pos.avg_cost
                    if peak > 0 and cur <= peak * (1 - trailing_stop_pct):
                        codes_to_sell.append(code)

            for code in codes_to_sell:
                pos = broker.positions[code]
                reason = "stop_loss" if (
                    stop_loss_pct is not None
                    and pos.current_price <= pos.avg_cost * (1 - stop_loss_pct)
                ) else "trailing_stop"
                broker.sell(code, pos.current_price, date=dt, action=f"SELL ({reason})")

        # --- 4. Generate signal at close, store as pending for T+1 open ---
        if dt in rebalance_dates_set:
            feature_rows: list[pd.DataFrame] = []
            vol_map: dict[str, float] = {}
            for code, fdf in factor_map.items():
                mask = fdf["date"] <= dt
                valid = fdf.loc[mask]
                core = TECHNICAL_COLUMNS[:13]
                valid = valid.dropna(subset=core)
                if valid.empty:
                    continue
                latest = valid.iloc[[-1]].copy()
                feature_rows.append(latest)
                vol_val = latest["volatility_20d"].values[0]
                vol_map[code] = float(vol_val) if np.isfinite(vol_val) else np.nan

            if not feature_rows:
                logger.warning("No valid features on {}, skipping rebalance", dt)
            else:
                features_df = pd.concat(feature_rows, ignore_index=True)
                codes_in_features = features_df["code"].tolist()

                scores = ranker.predict(features_df)

                scored = list(zip(codes_in_features, scores))
                if min_score is not None:
                    scored = [(c, s) for c, s in scored if s >= min_score]
                scored.sort(key=lambda x: -x[1])

                selected = scored[:top_k]
                sel_codes = [c for c, _ in selected]
                sel_scores = np.array([s for _, s in selected])

                sel_vols = np.array([vol_map.get(c, np.nan) for c in sel_codes])
                target_weights = _calc_weights(sel_scores, sizing, sel_vols)

                # Store as pending — execute at next day's open
                pending_rebalance = list(zip(sel_codes, target_weights))

        # --- Compute end-of-day NAV ---
        total_value = broker.total_value
        daily_return = (total_value / prev_total_value - 1) if prev_total_value > 0 else 0.0
        nav_records.append({"date": dt, "daily_return": daily_return})
        prev_total_value = total_value

        # --- Record position snapshot ---
        if broker.positions:
            snap: dict[str, float] = {}
            for code, pos in broker.positions.items():
                snap[code] = pos.market_value / total_value if total_value > 0 else 0.0
            positions_history.append((dt, snap))

        # --- Progress callback ---
        if progress_callback and (step + 1) % 20 == 0:
            progress_callback(step + 1, total_steps, f"Simulating: {step + 1}/{total_steps}")

    # ------------------------------------------------------------------
    # 6. Build output
    # ------------------------------------------------------------------
    nav_df = pd.DataFrame(nav_records)
    if nav_df.empty:
        raise RuntimeError("No NAV records produced; check date range and universe")

    nav_df["nav"] = (1 + nav_df["daily_return"]).cumprod()
    nav_df["cumulative_return"] = nav_df["nav"] - 1
    nav_series = nav_df.set_index("date")["nav"]

    # Performance metrics (reuse existing calc_performance)
    metrics = calc_performance(nav_df)

    # Turnover statistics
    trade_log = broker.trade_log
    turnover_stats = _compute_turnover_stats(trade_log, trading_dates, initial_capital)

    logger.info(
        "ML Backtest complete: {} trading days, {} trades | {}",
        len(trading_dates), len(trade_log),
        " | ".join(f"{k}={v}" for k, v in metrics.items()),
    )

    return {
        "nav_series": nav_series,
        "metrics": metrics,
        "trade_log": trade_log,
        "turnover_stats": turnover_stats,
        "positions_history": positions_history,
    }


def _compute_turnover_stats(
    trade_log: list[dict],
    trading_dates: list[pd.Timestamp],
    initial_capital: float,
) -> dict:
    """Derive turnover and holding-period statistics from the trade log."""
    if not trade_log:
        return {
            "avg_turnover": 0.0,
            "total_trades": 0,
            "avg_holding_days": 0.0,
        }

    total_trades = len(trade_log)

    # Sum absolute trade values by date to estimate daily turnover
    trade_df = pd.DataFrame(trade_log)
    daily_traded = trade_df.groupby("date")["value"].sum()
    n_days = max(len(trading_dates), 1)
    avg_turnover = float(daily_traded.sum() / n_days / initial_capital)

    # Estimate average holding days from buy/sell pairs per stock
    buys: dict[str, list[pd.Timestamp]] = {}
    sells: dict[str, list[pd.Timestamp]] = {}
    for t in trade_log:
        code = t["code"]
        if "BUY" in t["action"]:
            buys.setdefault(code, []).append(t["date"])
        else:
            sells.setdefault(code, []).append(t["date"])

    holding_days: list[float] = []
    for code in buys:
        buy_dates = sorted(buys[code])
        sell_dates = sorted(sells.get(code, []))
        for bd, sd in zip(buy_dates, sell_dates):
            delta = (pd.Timestamp(sd) - pd.Timestamp(bd)).days
            if delta >= 0:
                holding_days.append(delta)

    avg_holding = float(np.mean(holding_days)) if holding_days else 0.0

    return {
        "avg_turnover": round(avg_turnover, 4),
        "total_trades": total_trades,
        "avg_holding_days": round(avg_holding, 1),
    }


# ---------------------------------------------------------------------------
# Parameter optimisation
# ---------------------------------------------------------------------------

def optimize_ml_backtest(
    model_path: str | Path,
    universe: list[str] | str,
    start: str,
    end: str,
    initial_capital: float = 1_000_000,
    progress_callback: Callable | None = None,
) -> list[dict]:
    """Grid-search over ML backtest hyper-parameters.

    Searches over:
        - top_k: [5, 10, 15, 20]
        - rebalance_freq: ["weekly", "monthly"]
        - sizing: ["equal", "score_weighted"]
        - min_score: [None, 0.0, 0.01]
        - stop_loss_pct: [None, 0.05, 0.10]

    Returns a list of dicts sorted descending by Sharpe ratio.  Each dict
    contains the parameter combination and the resulting ``metrics``.
    """
    grid = {
        "top_k": [5, 10, 15, 20],
        "rebalance_freq": ["weekly", "monthly"],
        "sizing": ["equal", "score_weighted"],
        "min_score": [None, 0.0, 0.01],
        "stop_loss_pct": [None, 0.05, 0.10],
    }

    combos = list(itertools.product(*grid.values()))
    keys = list(grid.keys())
    total = len(combos)
    logger.info("optimize_ml_backtest: {} parameter combinations", total)

    results: list[dict] = []

    for idx, values in enumerate(combos):
        params = dict(zip(keys, values))
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        try:
            out = run_ml_backtest(
                model_path=model_path,
                universe=universe,
                start=start,
                end=end,
                initial_capital=initial_capital,
                **params,
            )
            sharpe_str = out["metrics"].get("sharpe_ratio", "0")
            sharpe = float(sharpe_str)
            results.append({
                "params": params,
                "metrics": out["metrics"],
                "sharpe": sharpe,
            })
            logger.info(
                "[{}/{}] {} => Sharpe={}", idx + 1, total, label, sharpe_str,
            )
        except Exception as exc:
            logger.warning("[{}/{}] {} => FAILED: {}", idx + 1, total, label, exc)

        if progress_callback:
            progress_callback(idx + 1, total, f"Optimizing: {idx + 1}/{total}")

    results.sort(key=lambda r: r["sharpe"], reverse=True)

    if results:
        best = results[0]
        logger.info(
            "Best params: {} => Sharpe={}",
            best["params"], best["metrics"].get("sharpe_ratio"),
        )

    return results
