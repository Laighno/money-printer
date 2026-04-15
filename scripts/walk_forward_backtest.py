"""Walk-Forward Backtest with monthly model retraining and daily rebalancing.

Monthly:
  1. Retrain LightGBM on expanding window (TRAIN_START → current month)
     Training labels (fwd_ret) are excluded for the last HORIZON trading days
     to prevent forward-return label leakage.

Daily:
  2. Score all universe stocks with current model at close, store as pending signal
  3. Next day: execute pending trades at T+1 open prices (no look-ahead)
  4. Track daily NAV

After the backtest, retrain the production model on the full 2015-2026 dataset.

Usage:
    python scripts/walk_forward_backtest.py
"""

from __future__ import annotations

import bisect
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from mp.account.broker import FeeSchedule, SimulatedBroker
from mp.backtest.engine import calc_performance
from mp.backtest.ml_backtest import _build_factor_panel, _prefetch_bars
from mp.data.fetcher import get_daily_bars, get_index_constituents, get_index_constituents_at
from mp.ml.dataset import (
    CURATED_COLUMNS,
    FACTOR_COLUMNS,
    FUNDAMENTAL_COLUMNS,
    FUNDAMENTAL_TREND_COLUMNS,
    INDUSTRY_RANK_COLUMNS,
    TECHNICAL_COLUMNS,
    _fetch_financial_history,
    _align_fundamentals_to_dates,
    _add_industry_relative_features,
)
from mp.ml.model import FEATURE_COLS, StockRanker

# ──────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────

INITIAL_CAPITAL = 100_000       # 10万元
UNIVERSE = "zz500"
TRAIN_START = "20150101"        # expanding window start
BT_START = "20200101"           # first trading month
BT_END = "20260401"             # last month (exclusive)
TOP_K = 10
HORIZON = 20                    # forward return horizon in trading days
SLIPPAGE_BPS = 5
COMMISSION_BPS = 3

CACHE_DIR = Path("data/wf_cache")
REPORT_PATH = Path("data/reports/walk_forward_result.md")

# ──────────────────────────────────────────────────────────────────────
# Data Preparation
# ──────────────────────────────────────────────────────────────────────

def _load_or_fetch_bars(codes: List[str]) -> Dict[str, pd.DataFrame]:
    """Pre-fetch all daily bars with parquet cache."""
    cache_path = CACHE_DIR / "bars.parquet"
    if cache_path.exists():
        logger.info("Loading cached bars from {}", cache_path)
        big = pd.read_parquet(cache_path)
        big["date"] = pd.to_datetime(big["date"])
        bars_map: Dict[str, pd.DataFrame] = {}
        for code, grp in big.groupby("code"):
            bars_map[str(code)] = grp.sort_values("date").reset_index(drop=True)
        logger.info("Loaded bars for {} stocks from cache", len(bars_map))
        return bars_map

    # Fetch from network — need extra warmup before TRAIN_START for factors
    warmup_start = "20140601"
    logger.info("Fetching bars for {} stocks from {}...", len(codes), warmup_start)
    bars_map = _prefetch_bars(codes, warmup_start, date.today().strftime("%Y%m%d"))

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for code, df in bars_map.items():
        df = df.copy()
        df["code"] = code
        frames.append(df)
    if frames:
        big = pd.concat(frames, ignore_index=True)
        big.to_parquet(cache_path, index=False)
        logger.info("Cached bars for {} stocks to {}", len(bars_map), cache_path)
    return bars_map


def _load_or_build_factors(
    bars_map: Dict[str, pd.DataFrame],
    codes: List[str],
) -> pd.DataFrame:
    """Build full factor panel with forward returns, with parquet cache."""
    cache_path = CACHE_DIR / "factors.parquet"
    if cache_path.exists():
        logger.info("Loading cached factor panel from {}", cache_path)
        panel = pd.read_parquet(cache_path)
        panel["date"] = pd.to_datetime(panel["date"])
        logger.info("Loaded factor panel: {} rows", len(panel))
        return panel

    # Build technical factors
    logger.info("Building factor panel for {} stocks...", len(bars_map))
    factor_map = _build_factor_panel(bars_map)

    # Merge into one big DataFrame
    frames = []
    for code, fdf in factor_map.items():
        frames.append(fdf)
    if not frames:
        raise RuntimeError("Factor computation failed for all stocks")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])

    # Add fundamental factors (time-aligned)
    logger.info("Fetching and aligning fundamental data...")
    for code in codes:
        mask = panel["code"] == code
        if mask.sum() == 0:
            continue
        fin_hist = _fetch_financial_history(code)
        code_dates = panel.loc[mask, "date"]
        fund = _align_fundamentals_to_dates(code_dates, fin_hist, None)
        for col in FUNDAMENTAL_COLUMNS + FUNDAMENTAL_TREND_COLUMNS:
            panel.loc[mask, col] = fund[col]

    # Compute forward returns for each stock
    logger.info("Computing {}-day forward returns...", HORIZON)
    fwd_parts = []
    for code, grp in panel.groupby("code"):
        grp = grp.sort_values("date").copy()
        close_vals = grp.merge(
            bars_map.get(str(code), pd.DataFrame()),
            on="date", how="left",
        )
        if "close" in close_vals.columns:
            close_arr = close_vals["close"].values.astype(float)
            n = len(close_arr)
            fwd = np.full(n, np.nan)
            for i in range(n - HORIZON):
                if close_arr[i] > 0:
                    fwd[i] = close_arr[i + HORIZON] / close_arr[i] - 1.0
            grp["fwd_ret"] = fwd
        else:
            grp["fwd_ret"] = np.nan
        fwd_parts.append(grp)
    panel = pd.concat(fwd_parts, ignore_index=True)

    # Drop warm-up NaN rows
    core_cols = TECHNICAL_COLUMNS[:13]
    panel = panel.dropna(subset=core_cols)

    # Industry-relative ranking features
    logger.info("Computing industry-relative ranking features...")
    try:
        from mp.data.fetcher import get_industry_mapping
        code_to_industry = get_industry_mapping(universe=codes)
        panel = _add_industry_relative_features(panel, code_to_industry)
        n_ranked = panel["pe_ind_rank"].notna().sum()
        logger.info("Industry rank features: {}/{} rows", n_ranked, len(panel))
    except Exception as e:
        logger.warning("Industry rank features failed, skipping: {}", e)
        for col in INDUSTRY_RANK_COLUMNS:
            if col not in panel.columns:
                panel[col] = np.nan

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache_path, index=False)
    logger.info("Cached factor panel: {} rows to {}", len(panel), cache_path)
    return panel


# ──────────────────────────────────────────────────────────────────────
# Walk-Forward Backtest
# ──────────────────────────────────────────────────────────────────────

def _get_monthly_retrain_dates(panel: pd.DataFrame) -> List[pd.Timestamp]:
    """Get first trading day of each month in [BT_START, BT_END) for model retraining."""
    bt_start = pd.Timestamp(BT_START)
    bt_end = pd.Timestamp(BT_END)
    dates = panel["date"].drop_duplicates().sort_values()
    dates = dates[(dates >= bt_start) & (dates < bt_end)]
    # First trading day of each month
    monthly = dates.groupby(dates.dt.to_period("M")).first()
    return monthly.tolist()


_ADV_PERIOD = 20   # days for rolling average daily value


def _build_price_adv_lookup(
    bars_map: Dict[str, pd.DataFrame],
) -> tuple[Dict, Dict, Dict]:
    """Build (code,date)->{close,open,adv} lookup dicts.

    ADV (average daily value) is the 20-day rolling mean of ``amount``
    ending the day BEFORE the trade date (no look-ahead).
    """
    close_lk: Dict[tuple, float] = {}
    open_lk: Dict[tuple, float] = {}
    adv_lk: Dict[tuple, float] = {}
    for code, df in bars_map.items():
        df = df.sort_values("date").reset_index(drop=True)
        for _, row in df.iterrows():
            d = row["date"]
            close_lk[(code, d)] = float(row["close"])
            open_lk[(code, d)] = float(row["open"])
        if "amount" in df.columns:
            # shift(1) so that ADV on day D uses amounts from D-20..D-1
            rolling_adv = df["amount"].rolling(_ADV_PERIOD, min_periods=5).mean().shift(1)
            for i, row in df.iterrows():
                v = rolling_adv.iloc[i]
                if not pd.isna(v) and v > 0:
                    adv_lk[(code, row["date"])] = float(v)
    return close_lk, open_lk, adv_lk


def run_walk_forward():
    """Main walk-forward backtest."""
    t0 = time.time()

    # 1. Universe
    logger.info("=" * 60)
    logger.info("Walk-Forward Backtest: {} -> {}", BT_START, BT_END)
    logger.info("Capital: {:,.0f} | Universe: {} | Top-K: {}", INITIAL_CAPITAL, UNIVERSE, TOP_K)
    logger.info("=" * 60)

    codes = get_index_constituents(UNIVERSE)
    logger.info("Universe: {} stocks", len(codes))

    # 2. Data
    bars_map = _load_or_fetch_bars(codes)
    panel = _load_or_build_factors(bars_map, codes)
    close_lk, open_lk, adv_lk = _build_price_adv_lookup(bars_map)

    # All trading dates in the BT window
    bt_start_ts = pd.Timestamp(BT_START)
    bt_end_ts = pd.Timestamp(BT_END)
    all_dates_set: set[pd.Timestamp] = set()
    for df in bars_map.values():
        mask = (df["date"] >= bt_start_ts) & (df["date"] < bt_end_ts)
        all_dates_set.update(df.loc[mask, "date"].tolist())
    trading_dates = sorted(all_dates_set)

    retrain_dates = _get_monthly_retrain_dates(panel)
    retrain_set = set(retrain_dates)
    logger.info("{} trading days, {} retrain months", len(trading_dates), len(retrain_dates))

    # Pre-group panel by date for fast daily feature lookup
    core = TECHNICAL_COLUMNS[:13]
    panel_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    for dt_key, grp in panel.groupby("date"):
        panel_by_date[pd.Timestamp(dt_key)] = grp

    # Trading calendar from panel for label_cutoff (Fix #3: avoid calendar-day approximation)
    _all_trade_dates = sorted(panel["date"].unique())

    # ── Survivorship-bias mitigation: time-varying scoring universe ──────────
    # Build a mapping from each retrain month to the index constituents that
    # were known at that time (from accumulated constituent_snapshots in the DB).
    # On the first ever run only today's snapshot exists, so all historical
    # months fall back to the current list.  Over quarterly runs the DB
    # accumulates snapshots and the mapping becomes increasingly accurate.
    _current_codes_set = frozenset(codes)
    _month_universe: Dict[pd.Period, frozenset] = {}
    _snapshot_dates = []
    try:
        from mp.data.store import DataStore as _DS
        _snapshot_dates = _DS().list_constituent_snapshot_dates(UNIVERSE)
    except Exception:
        pass
    n_snapshots = len(_snapshot_dates)
    if n_snapshots > 1:
        logger.info("Found {} historical {} snapshots — using time-varying universe",
                    n_snapshots, UNIVERSE)
    else:
        logger.warning(
            "Only {} constituent snapshot(s) available for {} — "
            "scoring universe uses current members throughout.  "
            "Survivorship bias is present: stocks that left the index "
            "before today are excluded from all historical periods.  "
            "Re-run quarterly to accumulate snapshots.",
            n_snapshots, UNIVERSE,
        )
    for rd in retrain_dates:
        rd_str = rd.strftime("%Y-%m-%d")
        snap = get_index_constituents_at(UNIVERSE, rd_str)
        _month_universe[rd.to_period("M")] = frozenset(snap) if snap else _current_codes_set
    # ────────────────────────────────────────────────────────────────────────

    # 3. Simulation state
    fees = FeeSchedule(slippage_bps=SLIPPAGE_BPS, commission_bps=COMMISSION_BPS,
                       use_sqrt_impact=True, impact_alpha_bps=150.0, min_slippage_bps=3.0)
    logger.info("Slippage model: sqrt-impact α={} bps, floor={} bps (fallback linear={} bps)",
                fees.impact_alpha_bps, fees.min_slippage_bps, fees.slippage_bps)
    broker = SimulatedBroker(INITIAL_CAPITAL, fees, silent=True)
    nav_records: List[dict] = []
    monthly_returns: List[dict] = []
    prev_total_value = float(INITIAL_CAPITAL)
    month_start_nav = 1.0
    current_month = None
    current_ranker: Optional[StockRanker] = None
    pending_selection: Optional[List[tuple]] = None  # signal from previous close
    current_universe: frozenset = _current_codes_set  # updated at each retrain

    # 4. Main loop — retrain monthly, score & rebalance daily
    for step, dt in enumerate(trading_dates):
        # Track month transitions for monthly return logging
        dt_period = dt.to_period("M")
        if current_month is not None and dt_period != current_month:
            current_nav = nav_records[-1]["nav"] if nav_records else 1.0
            monthly_returns.append({
                "month": str(current_month),
                "return": current_nav / month_start_nav - 1,
                "nav": current_nav,
            })
            month_start_nav = current_nav
        current_month = dt_period

        # --- Monthly: retrain model on first trading day of each month ---
        if dt in retrain_set:
            logger.info("── Retrain: {} ──", dt.strftime("%Y-%m-%d"))
            # Update scoring universe for this month.
            current_universe = _month_universe.get(dt_period, _current_codes_set)

            # ── Label-cutoff logic (prevents fwd_ret leakage into test period) ──
            # fwd_ret[t] = close[t + HORIZON] / close[t] - 1.
            # A row at date D is safe to train on iff D + HORIZON < dt
            # (i.e. the label uses only prices that pre-date the test period start).
            # label_cutoff is the trading date at position (idx_of_dt - HORIZON),
            # so every row with date < label_cutoff satisfies D + HORIZON < dt.
            # The `< label_cutoff` strict inequality ensures the cutoff date
            # itself (whose fwd_ret uses close[dt]) is also excluded.
            _dt_idx = bisect.bisect_right(_all_trade_dates, dt) - 1
            _cutoff_idx = max(0, _dt_idx - HORIZON)
            label_cutoff = _all_trade_dates[_cutoff_idx]
            train_mask = (panel["date"] < label_cutoff) & panel["fwd_ret"].notna()
            train_df = panel.loc[train_mask].copy()

            if len(train_df) < 500:
                logger.warning("Too few training rows ({}), skipping retrain", len(train_df))
            else:
                ranker = StockRanker(feature_cols=FACTOR_COLUMNS)
                try:
                    metrics = ranker.train_fast(train_df)
                    logger.info("  Train: {} rows, MAE={:.4f}, IC={:.3f}, HitRate@10={:.2f}, Precision@10={:.2f}, rounds={}",
                                len(train_df), metrics["mae"], metrics["ic"],
                                metrics.get("hit_rate_at_k", float("nan")),
                                metrics.get("precision_at_k", float("nan")),
                                metrics["best_rounds"])
                    current_ranker = ranker
                except Exception as e:
                    logger.error("  Training failed: {}", e)

        # --- Step A: Execute pending signal from previous close at today's open ---
        if pending_selection is not None:
            sel_codes = {c for c, _ in pending_selection}
            held_codes = set(broker.positions.keys())
            if sel_codes != held_codes:
                # Sizing uses previous close (already in broker from yesterday's NAV step).
                # Using today's close here would be look-ahead bias.

                # (b) Sell positions not in new selection at today's open
                dt_str = dt.strftime("%Y-%m-%d")
                for code in list(broker.positions.keys()):
                    if code not in sel_codes:
                        sell_price = open_lk.get((code, dt), broker.positions[code].current_price)
                        broker.sell(code, sell_price, date=dt_str,
                                    adv=adv_lk.get((code, dt)))

                # (c) Buy new / adjust positions (equal weight) at today's open
                target_per_stock = broker.total_value / TOP_K
                for code, score in pending_selection:
                    price_raw = open_lk.get((code, dt))
                    if price_raw is None or price_raw <= 0:
                        continue
                    broker.buy(code, price_raw, target_value=target_per_stock,
                               date=dt_str,
                               action="BUY (add)" if code in broker.positions else "BUY",
                               adv=adv_lk.get((code, dt)))

                logger.info("  Day {}: rebalanced → {} stocks, cash: {:,.0f}",
                            dt.strftime("%Y-%m-%d"), len(broker.positions), broker.cash)
            pending_selection = None  # consumed

        # --- Step B: Score stocks at today's close, store as pending for tomorrow ---
        # Filter to the universe that was known on this date to avoid look-ahead
        # from index membership.  When only today's snapshot is available,
        # current_universe == all current codes (no filtering effect).
        if current_ranker is not None and current_ranker.model is not None:
            today_df = panel_by_date.get(dt)
            if today_df is not None:
                today_df = today_df[today_df["code"].isin(current_universe)]
                today_valid = today_df.dropna(subset=core)
                if not today_valid.empty:
                    codes_in = today_valid["code"].tolist()
                    scores = current_ranker.predict(today_valid)
                    scored = sorted(zip(codes_in, scores), key=lambda x: -x[1])
                    pending_selection = scored[:TOP_K]

        # --- Daily NAV ---
        close_snap = {}
        for code in broker.positions:
            price = close_lk.get((code, dt), broker.positions[code].current_price)
            if price > 0:
                close_snap[code] = price
        broker.update_prices(close_snap)
        total_value = broker.total_value

        daily_ret = (total_value / prev_total_value - 1) if prev_total_value > 0 else 0.0
        nav = (nav_records[-1]["nav"] * (1 + daily_ret)) if nav_records else (1 + daily_ret)
        nav_records.append({"date": dt, "daily_return": daily_ret, "nav": nav})
        prev_total_value = total_value

        if (step + 1) % 100 == 0:
            logger.info("  Day {}/{}: NAV={:.4f}, value={:,.0f}",
                        step + 1, len(trading_dates), nav, total_value)

    # Log last month
    if nav_records and current_month is not None:
        monthly_returns.append({
            "month": str(current_month),
            "return": nav_records[-1]["nav"] / month_start_nav - 1,
            "nav": nav_records[-1]["nav"],
        })

    # 5. Performance metrics
    nav_df = pd.DataFrame(nav_records)
    metrics = calc_performance(nav_df)

    # Fetch ZZ500 benchmark
    benchmark_ret = _calc_benchmark_return(trading_dates, close_lk, bars_map)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Walk-Forward Backtest Complete ({:.1f} min)", elapsed / 60)
    logger.info("=" * 60)

    # 6. Print results
    _print_results(metrics, monthly_returns, benchmark_ret, broker.trade_log, elapsed)

    # 7. Save report
    _save_report(metrics, monthly_returns, benchmark_ret, broker.trade_log, elapsed)

    return metrics, monthly_returns, current_ranker, benchmark_ret


def _calc_benchmark_return(
    trading_dates: List[pd.Timestamp],
    close_lk: Dict,
    bars_map: Dict[str, pd.DataFrame],
) -> Optional[float]:
    """Approximate ZZ500 index return over the backtest period.

    Uses akshare to fetch index data directly.
    """
    try:
        import akshare as ak
        idx_df = ak.stock_zh_index_daily(symbol="sh000905")
        idx_df["date"] = pd.to_datetime(idx_df["date"])
        start_dt = trading_dates[0]
        end_dt = trading_dates[-1]
        mask = (idx_df["date"] >= start_dt) & (idx_df["date"] <= end_dt)
        idx_sub = idx_df.loc[mask].sort_values("date")
        if len(idx_sub) >= 2:
            ret = float(idx_sub["close"].iloc[-1] / idx_sub["close"].iloc[0] - 1)
            n_days = len(idx_sub)
            annual_ret = (1 + ret) ** (252 / max(n_days, 1)) - 1
            logger.info("ZZ500 benchmark: total={:.2%}, annual={:.2%}", ret, annual_ret)
            return {"total_return": ret, "annual_return": annual_ret, "days": n_days}
    except Exception as e:
        logger.warning("Failed to fetch ZZ500 benchmark: {}", e)
    return None


def _print_results(metrics, monthly_returns, benchmark_ret, trade_log, elapsed):
    """Print results to console."""
    print("\n" + "=" * 60)
    print("  WALK-FORWARD BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Period:          {BT_START[:4]}-{BT_START[4:6]} ~ {BT_END[:4]}-{BT_END[4:6]}")
    print(f"  Initial Capital: {INITIAL_CAPITAL:,.0f}")
    print(f"  Universe:        {UNIVERSE} | Top-K: {TOP_K}")
    print(f"  Runtime:         {elapsed/60:.1f} min")
    print("-" * 60)
    for k, v in metrics.items():
        if k != "trading_days":
            print(f"  {k:20s}: {v}")
    print(f"  {'trading_days':20s}: {metrics.get('trading_days', 'N/A')}")

    if benchmark_ret:
        print("-" * 60)
        print(f"  ZZ500 Total Return:  {benchmark_ret['total_return']:.2%}")
        print(f"  ZZ500 Annual Return: {benchmark_ret['annual_return']:.2%}")
        excess = float(metrics.get('annual_return', '0%').rstrip('%')) / 100 - benchmark_ret['annual_return']
        print(f"  Excess (alpha):      {excess:.2%}")

    print("-" * 60)
    print(f"  Total trades: {len(trade_log)}")
    print()

    # Monthly returns table
    print("  Monthly Returns:")
    print("  " + "-" * 50)
    for mr in monthly_returns:
        bar = "+" * int(max(0, mr["return"] * 100)) + "-" * int(max(0, -mr["return"] * 100))
        print(f"  {mr['month']:8s}  {mr['return']:+7.2%}  NAV={mr['nav']:.4f}  {bar}")
    print("=" * 60 + "\n")


def _save_report(metrics, monthly_returns, benchmark_ret, trade_log, elapsed):
    """Save markdown report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Walk-Forward Backtest Report",
        f"**Period**: {BT_START[:4]}-{BT_START[4:6]} ~ {BT_END[:4]}-{BT_END[4:6]}",
        f"**Initial Capital**: {INITIAL_CAPITAL:,.0f} | **Universe**: {UNIVERSE} | **Top-K**: {TOP_K}",
        f"**Model**: LightGBM monthly retrain, daily rebalance | **Horizon**: {HORIZON}d",
        f"**Runtime**: {elapsed/60:.1f} min",
        "",
        "## Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")

    if benchmark_ret:
        lines.extend([
            "",
            "## Benchmark (ZZ500)",
            "",
            f"| Total Return | {benchmark_ret['total_return']:.2%} |",
            f"| Annual Return | {benchmark_ret['annual_return']:.2%} |",
        ])

    lines.extend(["", "## Monthly Returns", "", "| Month | Return | NAV |", "|-------|--------|-----|"])
    for mr in monthly_returns:
        lines.append(f"| {mr['month']} | {mr['return']:+.2%} | {mr['nav']:.4f} |")

    lines.extend([
        "",
        f"## Trades ({len(trade_log)} total)",
        "",
    ])
    # Show first 20 and last 20 trades
    if len(trade_log) <= 40:
        show_trades = trade_log
    else:
        show_trades = trade_log[:20] + [{"date": "...", "code": "...", "action": "...", "shares": "...", "price": "...", "value": "..."}] + trade_log[-20:]

    lines.append("| Date | Code | Action | Shares | Price | Value |")
    lines.append("|------|------|--------|--------|-------|-------|")
    for t in show_trades:
        d = t["date"].strftime("%Y-%m-%d") if hasattr(t["date"], "strftime") else str(t["date"])
        lines.append(f"| {d} | {t['code']} | {t['action']} | {t['shares']} | {t.get('price', ''):.2f} | {t.get('value', ''):.0f} |" if isinstance(t.get("value"), (int, float)) else f"| {d} | {t['code']} | {t['action']} | ... | ... | ... |")

    lines.extend(["", "---", f"*Generated by Walk-Forward Backtest | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*"])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved to {}", REPORT_PATH)


# ──────────────────────────────────────────────────────────────────────
# Production Model Update
# ──────────────────────────────────────────────────────────────────────

def update_production_models(
    codes: List[str],
    ranker_20d: Optional[StockRanker] = None,
) -> dict:
    """Save/retrain production models.

    - 20d model: if ranker_20d is provided (from walk-forward), just save it;
      otherwise retrain from scratch with full CV.
    - 60d model: always retrain (walk-forward doesn't train 60d).

    Returns a dict with training metrics for both models.
    """
    from mp.ml.dataset import build_dataset

    logger.info("=" * 60)
    logger.info("Updating production models...")
    logger.info("=" * 60)

    results: dict = {}

    # 20-day horizon model — reuse walk-forward's last model if available
    if ranker_20d is not None and ranker_20d.model is not None:
        ranker_20d.model_path = Path("data/model.lgb")
        ranker_20d.save()
        logger.info("20d model: saved from walk-forward (no retrain needed)")
        results["20d"] = {"source": "walk-forward", "saved": True}
    else:
        logger.info("Building 20d dataset from scratch...")
        ds_20 = build_dataset(codes, TRAIN_START, horizon=20)
        if not ds_20.empty:
            ranker = StockRanker(model_path="data/model.lgb")
            m = ranker.train(ds_20)
            logger.info("20d model: MAE={:.4f}±{:.4f}, IC={:.3f}±{:.3f}, rounds={}",
                         m["cv_mae_mean"], m["cv_mae_std"],
                         m["cv_ic_mean"], m["cv_ic_std"], m["best_rounds"])
            results["20d"] = m
        else:
            logger.error("Failed to build 20d dataset")
            results["20d"] = {"error": "dataset build failed"}

    # 60-day horizon model — always retrain
    logger.info("Building 60d dataset...")
    ds_60 = build_dataset(codes, TRAIN_START, horizon=60)
    if not ds_60.empty:
        ranker_60 = StockRanker(model_path="data/model_60d.lgb")
        m = ranker_60.train(ds_60)
        logger.info("60d model: MAE={:.4f}±{:.4f}, IC={:.3f}±{:.3f}, rounds={}",
                     m["cv_mae_mean"], m["cv_mae_std"],
                     m["cv_ic_mean"], m["cv_ic_std"], m["best_rounds"])
        results["60d"] = m
    else:
        logger.error("Failed to build 60d dataset")
        results["60d"] = {"error": "dataset build failed"}

    return results


# ──────────────────────────────────────────────────────────────────────
# Backtest history (for run-over-run comparison)
# ──────────────────────────────────────────────────────────────────────

BACKTEST_HISTORY_PATH = Path("data/reports/backtest_history.json")
_MAX_HISTORY = 10  # keep at most this many snapshots


def _save_backtest_snapshot(
    bt_metrics: dict,
    bt_benchmark: Optional[dict],
    model_results: dict,
) -> None:
    """Append current run metrics to the history file (max _MAX_HISTORY entries)."""
    import json

    snapshot = {
        "date": date.today().isoformat(),
        "bt_metrics": bt_metrics,
        "bt_benchmark": bt_benchmark,
        "model_ic": {
            h: model_results.get(h, {}).get("cv_ic_mean")
            for h in ("20d", "60d")
        },
        "model_hit_rate": {
            h: model_results.get(h, {}).get("cv_hit_rate_mean")
            for h in ("20d", "60d")
        },
    }

    history: list = []
    if BACKTEST_HISTORY_PATH.exists():
        try:
            history = json.loads(BACKTEST_HISTORY_PATH.read_text())
        except Exception:
            history = []

    history.append(snapshot)
    history = history[-_MAX_HISTORY:]  # trim oldest

    BACKTEST_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    BACKTEST_HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False))
    logger.info("Backtest snapshot saved to {}", BACKTEST_HISTORY_PATH)


def _load_prev_backtest_snapshot() -> Optional[dict]:
    """Return the second-to-last snapshot (i.e. the previous run), or None."""
    import json

    if not BACKTEST_HISTORY_PATH.exists():
        return None
    try:
        history = json.loads(BACKTEST_HISTORY_PATH.read_text())
        # last entry is current run (already saved), second-to-last is previous
        if len(history) >= 2:
            return history[-2]
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────
# Feishu notification
# ──────────────────────────────────────────────────────────────────────

def send_model_update_report(
    model_results: dict,
    bt_metrics: Optional[dict] = None,
    bt_benchmark: Optional[dict] = None,
    elapsed_min: float = 0.0,
    update_only: bool = False,
) -> bool:
    """Send a model update summary to Feishu via lark-cli.

    Parameters
    ----------
    model_results:
        Return value of ``update_production_models()``.
    bt_metrics:
        Walk-forward backtest performance metrics (optional, from full run).
    bt_benchmark:
        ZZ500 benchmark return dict (optional).
    elapsed_min:
        Total runtime in minutes.
    update_only:
        True if this was a ``--update-only`` run (no backtest).
    """
    import subprocess
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    today = date.today().strftime("%Y-%m-%d")
    mode = "仅模型更新" if update_only else "完整回测 + 模型更新"

    lines = [
        f"# 📊 模型更新完成",
        f"**{today}**  |  模式: {mode}  |  耗时: {elapsed_min:.1f} 分钟",
        "",
    ]

    # --- Model training summary ---
    lines.append("## 模型训练结果")
    lines.append("")

    for horizon, label in [("20d", "20日预测模型"), ("60d", "60日预测模型")]:
        m = model_results.get(horizon, {})
        if "error" in m:
            lines.append(f"**{label}**: ❌ {m['error']}")
        elif m.get("source") == "walk-forward":
            lines.append(f"**{label}**: ✅ 沿用回测末期模型（无需重训）")
        else:
            mae  = m.get("cv_mae_mean", "N/A")
            ic   = m.get("cv_ic_mean",  "N/A")
            hr   = m.get("cv_hit_rate_mean")
            prec = m.get("cv_precision_mean")
            rds  = m.get("best_rounds", "N/A")
            mae_str = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
            ic_str  = f"{ic:.3f}"  if isinstance(ic,  float) else str(ic)
            hr_str  = f"{hr:.2f}"  if isinstance(hr,  float) else "N/A"
            prec_str = f"{prec:.2f}" if isinstance(prec, float) else "N/A"
            lines.append(f"**{label}**: ✅  MAE={mae_str}  IC={ic_str}  HitRate@10={hr_str}  Precision@10={prec_str}  迭代={rds}")
        lines.append("")

    # --- Backtest summary (only for full runs) ---
    if not update_only and bt_metrics:
        lines.append("## 回测绩效摘要")
        lines.append("")
        key_map = {
            "total_return":  "总收益",
            "annual_return": "年化收益",
            "sharpe_ratio":  "夏普比率",
            "max_drawdown":  "最大回撤",
            "win_rate":      "胜率",
        }
        for k, label in key_map.items():
            v = bt_metrics.get(k)
            if v is not None:
                lines.append(f"- **{label}**: {v}")
        if bt_benchmark:
            excess = None
            ar_str = bt_metrics.get("annual_return", "")
            try:
                ar = float(str(ar_str).rstrip("%")) / 100
                excess = ar - bt_benchmark["annual_return"]
            except Exception:
                pass
            lines.append(f"- **ZZ500基准年化**: {bt_benchmark['annual_return']:.2%}")
            if excess is not None:
                lines.append(f"- **超额收益(alpha)**: {excess:+.2%}")
        lines.append("")

        # --- Comparison with previous run ---
        prev = _load_prev_backtest_snapshot()
        if prev:
            prev_date = prev.get("date", "上次")
            lines.append(f"## 与上次回测对比（{prev_date}）")
            lines.append("")

            def _delta_str(cur_raw, prev_raw, is_pct: bool = False) -> str:
                """Format current value and delta vs previous."""
                try:
                    cur = float(str(cur_raw).rstrip("%")) / (100 if is_pct else 1)
                    prv = float(str(prev_raw).rstrip("%")) / (100 if is_pct else 1)
                    delta = cur - prv
                    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "↔")
                    cur_str = f"{cur:.2%}" if is_pct else f"{cur:.2f}"
                    prv_str = f"{prv:.2%}" if is_pct else f"{prv:.2f}"
                    delta_str = f"{delta:+.2%}" if is_pct else f"{delta:+.2f}"
                    return f"{prv_str} → {cur_str} ({arrow} {delta_str})"
                except Exception:
                    return str(cur_raw)

            prev_bt = prev.get("bt_metrics", {})
            cmp_map = [
                ("annual_return", "年化收益", True),
                ("sharpe_ratio",  "夏普比率", False),
                ("max_drawdown",  "最大回撤", True),
                ("win_rate",      "胜率",     True),
            ]
            for k, label, is_pct in cmp_map:
                cur_v = bt_metrics.get(k)
                prv_v = prev_bt.get(k)
                if cur_v is not None and prv_v is not None:
                    lines.append(f"- **{label}**: {_delta_str(cur_v, prv_v, is_pct)}")

            # Model IC & HitRate comparison
            for horizon, label in [("20d", "20日模型"), ("60d", "60日模型")]:
                m = model_results.get(horizon, {})
                cur_ic = m.get("cv_ic_mean")
                cur_hr = m.get("cv_hit_rate_mean")
                prv_ic = prev.get("model_ic", {}).get(horizon)
                prv_hr = prev.get("model_hit_rate", {}).get(horizon)
                parts = []
                if cur_ic is not None and prv_ic is not None:
                    parts.append(f"IC {_delta_str(cur_ic, prv_ic, False)}")
                if cur_hr is not None and prv_hr is not None:
                    parts.append(f"HitRate@10 {_delta_str(cur_hr, prv_hr, False)}")
                if parts:
                    lines.append(f"- **{label}**: {' | '.join(parts)}")
            lines.append("")

    # --- Footer ---
    lines.append("---")
    lines.append(f"*Money Printer 自动生成 | {now}*")

    markdown = "\n".join(lines)

    # Read default user ID from daily_report constants
    USER_ID = "ou_da792f0119461fb14c41b21b40834b09"
    cmd = ["lark-cli", "im", "+messages-send", "--as", "bot",
           "--user-id", USER_ID, "--markdown", markdown]

    logger.info("Sending model update report to Feishu...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("Model update report sent to Feishu")
            return True
        else:
            logger.error("Feishu send failed: {}", result.stderr or result.stdout)
            return False
    except Exception as e:
        logger.error("Feishu send error: {}", e)
        return False


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Walk-Forward Backtest (monthly retrain, daily rebalance)")
    parser.add_argument("--skip-update", action="store_true",
                        help="Skip production model update after backtest")
    parser.add_argument("--cache-only", action="store_true",
                        help="Only build data cache, don't run backtest")
    parser.add_argument("--update-only", action="store_true",
                        help="Only retrain production models (no backtest)")
    args = parser.parse_args()

    if args.update_only:
        t0 = time.time()
        codes = get_index_constituents(UNIVERSE)
        model_results = update_production_models(codes)
        elapsed_min = (time.time() - t0) / 60
        logger.info("Production models updated! ({:.1f} min)", elapsed_min)
        send_model_update_report(
            model_results=model_results,
            elapsed_min=elapsed_min,
            update_only=True,
        )
    elif args.cache_only:
        codes = get_index_constituents(UNIVERSE)
        bars_map = _load_or_fetch_bars(codes)
        _load_or_build_factors(bars_map, codes)
        logger.info("Cache built successfully. Run again without --cache-only to start backtest.")
    else:
        metrics, monthly, last_ranker, bt_benchmark = run_walk_forward()

        if not args.skip_update:
            t1 = time.time()
            codes = get_index_constituents(UNIVERSE)
            model_results = update_production_models(codes, ranker_20d=last_ranker)
            elapsed_min = (time.time() - t1) / 60
            logger.info("Production models updated! ({:.1f} min)", elapsed_min)
            _save_backtest_snapshot(metrics, bt_benchmark, model_results)
            send_model_update_report(
                model_results=model_results,
                bt_metrics=metrics,
                bt_benchmark=bt_benchmark,
                elapsed_min=elapsed_min,
                update_only=False,
            )
        else:
            logger.info("Skipping production model update (--skip-update)")
