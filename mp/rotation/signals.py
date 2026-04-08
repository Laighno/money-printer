"""Industry rotation signal calculation.

Core idea from the user's framework:
- Track sector price momentum (trend)
- Track capital flow (volume/amount acceleration)
- Track breadth (up_count ratio)
- Combine into a composite signal for sector rotation

Signals are data inputs for human decision, NOT automatic trading triggers.
"""

import numpy as np
import pandas as pd
from loguru import logger


def calc_momentum(industry_bars: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Calculate price momentum for each industry board.

    Args:
        industry_bars: board_name, date, close, ...
        windows: lookback windows in trading days, default [5, 10, 20, 60]

    Returns:
        DataFrame indexed by board_name with momentum columns
    """
    if windows is None:
        windows = [5, 10, 20, 60]

    results = {}
    for name, group in industry_bars.groupby("board_name"):
        group = group.sort_values("date")
        row = {"board_name": name}
        latest_close = group["close"].iloc[-1]
        for w in windows:
            if len(group) > w:
                past_close = group["close"].iloc[-(w + 1)]
                row[f"mom_{w}d"] = (latest_close / past_close - 1) * 100
            else:
                row[f"mom_{w}d"] = np.nan
        results[name] = row

    df = pd.DataFrame(results.values())
    return df.set_index("board_name")


def calc_volume_signal(industry_bars: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume/amount acceleration signals.

    Detects whether capital is flowing INTO a sector (volume expansion)
    or OUT of it (volume contraction).

    Signals:
    - vol_ratio_5d: recent 5d avg volume / prior 20d avg volume
    - amount_ratio_5d: same for turnover amount
    - turnover_avg_5d: average daily turnover rate (last 5 days)
    """
    results = {}
    for name, group in industry_bars.groupby("board_name"):
        group = group.sort_values("date")
        row = {"board_name": name}

        if len(group) < 25:
            row.update({"vol_ratio_5d": np.nan, "amount_ratio_5d": np.nan, "turnover_avg_5d": np.nan})
        else:
            recent_vol = group["volume"].tail(5).mean()
            prior_vol = group["volume"].iloc[-25:-5].mean()
            row["vol_ratio_5d"] = recent_vol / prior_vol if prior_vol > 0 else np.nan

            recent_amt = group["amount"].tail(5).mean()
            prior_amt = group["amount"].iloc[-25:-5].mean()
            row["amount_ratio_5d"] = recent_amt / prior_amt if prior_amt > 0 else np.nan

            row["turnover_avg_5d"] = group["turnover"].tail(5).mean()

        results[name] = row

    return pd.DataFrame(results.values()).set_index("board_name")


def calc_trend_strength(industry_bars: pd.DataFrame) -> pd.DataFrame:
    """Calculate trend strength indicators.

    - ma_alignment: whether short MA > mid MA > long MA (趋势多头排列)
    - above_ma20: close > MA20 (short-term trend)
    - volatility_20d: annualized volatility
    """
    results = {}
    for name, group in industry_bars.groupby("board_name"):
        group = group.sort_values("date")
        row = {"board_name": name}

        closes = group["close"]
        if len(closes) < 60:
            row.update({"ma_alignment": np.nan, "above_ma20": np.nan, "volatility_20d": np.nan})
        else:
            ma5 = closes.tail(5).mean()
            ma20 = closes.tail(20).mean()
            ma60 = closes.tail(60).mean()
            latest = closes.iloc[-1]

            # 1 = full bullish alignment, -1 = bearish, 0 = mixed
            if ma5 > ma20 > ma60:
                row["ma_alignment"] = 1.0
            elif ma5 < ma20 < ma60:
                row["ma_alignment"] = -1.0
            else:
                row["ma_alignment"] = 0.0

            row["above_ma20"] = 1.0 if latest > ma20 else 0.0

            returns = closes.pct_change().dropna().tail(20)
            row["volatility_20d"] = returns.std() * np.sqrt(252) * 100

        results[name] = row

    return pd.DataFrame(results.values()).set_index("board_name")


def generate_rotation_signals(industry_bars: pd.DataFrame) -> pd.DataFrame:
    """Generate composite rotation signals for all industry boards.

    Combines momentum, volume, and trend into a single signal dashboard.

    Returns:
        DataFrame indexed by board_name, sorted by composite_score desc.
        Columns: mom_5d, mom_10d, mom_20d, mom_60d, vol_ratio_5d, amount_ratio_5d,
                 ma_alignment, above_ma20, volatility_20d, composite_score
    """
    mom = calc_momentum(industry_bars)
    vol = calc_volume_signal(industry_bars)
    trend = calc_trend_strength(industry_bars)

    df = mom.join(vol, how="outer").join(trend, how="outer")

    # Composite score: z-score weighted combination
    # Favor: strong short+mid momentum, capital inflow, bullish trend, low volatility
    score_components = {
        "mom_5d": 0.10,
        "mom_20d": 0.25,
        "mom_60d": 0.15,
        "vol_ratio_5d": 0.15,     # capital flow
        "amount_ratio_5d": 0.10,
        "ma_alignment": 0.15,     # trend confirmation
        "volatility_20d": -0.10,  # lower vol is better (negative weight)
    }

    for col, weight in score_components.items():
        if col in df.columns:
            z = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            df[f"_z_{col}"] = z * weight

    z_cols = [c for c in df.columns if c.startswith("_z_")]
    df["composite_score"] = df[z_cols].sum(axis=1)
    df = df.drop(columns=z_cols)

    df = df.sort_values("composite_score", ascending=False)

    logger.info(f"Generated rotation signals for {len(df)} boards. Top 5: {df.head(5).index.tolist()}")
    return df


def generate_reversal_signals(industry_bars: pd.DataFrame) -> pd.DataFrame:
    """Generate oversold-bounce (超跌反弹) signals for all industry boards.

    Logic: find sectors that dropped heavily but are starting to recover.
    - 20d/60d momentum deeply negative (enough pain)
    - 5d momentum turning positive (止跌企稳)
    - Volume ratio expanding (capital stepping in)
    - Volatility high but starting to cool (panic fading)

    Returns:
        DataFrame indexed by board_name, sorted by reversal_score desc.
    """
    mom = calc_momentum(industry_bars)
    vol = calc_volume_signal(industry_bars)
    trend = calc_trend_strength(industry_bars)

    df = mom.join(vol, how="outer").join(trend, how="outer")

    # Filter: only consider sectors with meaningful decline
    # 20d momentum < -5% (has dropped enough)
    df["oversold"] = (df["mom_20d"] < -5).astype(float) if "mom_20d" in df.columns else 0.0

    # Reversal score components (opposite logic from momentum strategy)
    score_components = {
        "mom_5d": 0.30,          # short-term bounce (positive = recovering)
        "mom_20d": -0.20,        # deeper decline = more upside (negative weight: more negative mom = higher score)
        "mom_60d": -0.10,        # same logic for longer term
        "vol_ratio_5d": 0.20,    # volume expansion = capital entering
        "amount_ratio_5d": 0.10, # amount confirmation
        "volatility_20d": -0.10, # prefer vol cooling down from high
    }

    for col, weight in score_components.items():
        if col in df.columns:
            z = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            df[f"_z_{col}"] = z * weight

    z_cols = [c for c in df.columns if c.startswith("_z_")]
    df["reversal_score"] = df[z_cols].sum(axis=1)

    # Only keep oversold sectors (20d mom < -5%), set others to -999
    df.loc[df["oversold"] < 1, "reversal_score"] = -999

    df = df.drop(columns=z_cols + ["oversold"])
    df = df.sort_values("reversal_score", ascending=False)

    # Filter out non-candidates
    df = df[df["reversal_score"] > -999]

    logger.info(f"Generated reversal signals for {len(df)} oversold boards. Top 5: {df.head(5).index.tolist()}")
    return df


def generate_deep_value_signals(industry_bars: pd.DataFrame) -> pd.DataFrame:
    """Generate deep-value / oversold-not-yet-bounced (超跌未涨) signals.

    Find sectors that have dropped far beyond expectations but haven't started
    recovering yet. Earlier stage than reversal — for watchlist / light positions.

    Key indicators:
    - Price far below MA60 (extreme deviation)
    - Decline momentum decelerating (跌不动了)
    - Volume shrinking (selling exhaustion, 缩量筑底)
    - Volatility at extremes (panic peaking)
    """
    mom = calc_momentum(industry_bars)
    vol = calc_volume_signal(industry_bars)
    trend = calc_trend_strength(industry_bars)

    df = mom.join(vol, how="outer").join(trend, how="outer")

    # Price deviation from MA60
    deviation = {}
    for name, group in industry_bars.groupby("board_name"):
        group = group.sort_values("date")
        closes = group["close"]
        if len(closes) >= 60:
            ma60 = closes.tail(60).mean()
            latest = closes.iloc[-1]
            deviation[name] = (latest / ma60 - 1) * 100  # % below MA60
        else:
            deviation[name] = np.nan
    df["ma60_deviation"] = pd.Series(deviation)

    # Decline deceleration: |5d mom| < |20d mom| * 0.25 means slowing down
    df["deceleration"] = 0.0
    mask = (df["mom_20d"] < -5) & (df["mom_5d"].abs() < df["mom_20d"].abs() * 0.25)
    df.loc[mask, "deceleration"] = 1.0

    # Filter: must be significantly below MA60 and still declining (5d mom <= 0)
    df["candidate"] = (
        (df["ma60_deviation"] < -5) &  # 5%+ below MA60
        (df["mom_5d"] <= 0) &          # hasn't started bouncing yet
        (df["mom_20d"] < -5)           # meaningful decline
    ).astype(float)

    score_components = {
        "ma60_deviation": -0.30,   # further below MA60 = higher score (neg weight)
        "mom_20d": -0.15,          # deeper decline = higher score
        "mom_5d": 0.15,            # less negative 5d = closer to bottom
        "vol_ratio_5d": -0.15,     # lower volume = selling exhaustion (neg weight)
        "volatility_20d": 0.15,    # high vol = extreme fear = potential bottom
        "deceleration": 0.10,      # slowing decline
    }

    for col, weight in score_components.items():
        if col in df.columns:
            z = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            df[f"_z_{col}"] = z * weight

    z_cols = [c for c in df.columns if c.startswith("_z_")]
    df["deep_value_score"] = df[z_cols].sum(axis=1)
    df.loc[df["candidate"] < 1, "deep_value_score"] = -999

    df = df.drop(columns=z_cols + ["candidate", "deceleration"])
    df = df.sort_values("deep_value_score", ascending=False)
    df = df[df["deep_value_score"] > -999]

    logger.info(f"Generated deep-value signals for {len(df)} boards. Top 5: {df.head(5).index.tolist()}")
    return df


def generate_accumulation_signals(industry_bars: pd.DataFrame) -> pd.DataFrame:
    """Generate stealth accumulation (主力吸筹) signals.

    Find sectors where smart money is quietly building positions:
    price is flat/slightly down, but volume/amount is expanding.
    Classic volume-price divergence.

    Key indicators:
    - Price barely moving (small 20d momentum, near 0)
    - Volume expanding significantly (vol_ratio > 1.3)
    - Amount expanding (amount_ratio > 1.3)
    - Low volatility (quiet accumulation, not wild swings)
    """
    mom = calc_momentum(industry_bars)
    vol = calc_volume_signal(industry_bars)
    trend = calc_trend_strength(industry_bars)

    df = mom.join(vol, how="outer").join(trend, how="outer")

    # Price flatness: |20d momentum| < 3% means price is range-bound
    df["price_flat"] = (df["mom_20d"].abs() < 3).astype(float) if "mom_20d" in df.columns else 0.0

    # Volume-price divergence: volume up but price flat
    df["vol_price_div"] = 0.0
    mask = (df["price_flat"] > 0) & (df["vol_ratio_5d"] > 1.2)
    df.loc[mask, "vol_price_div"] = df.loc[mask, "vol_ratio_5d"]

    score_components = {
        "vol_ratio_5d": 0.30,       # volume expansion is the core signal
        "amount_ratio_5d": 0.25,    # amount confirmation
        "vol_price_div": 0.20,      # divergence bonus
        "volatility_20d": -0.15,    # low vol = quiet accumulation
        "mom_20d": -0.10,           # flatter price = more suspicious accumulation
    }

    # Abs-transform mom_20d so that near-zero gets highest score
    df["_abs_mom_20d"] = df["mom_20d"].abs()

    for col, weight in score_components.items():
        src = "_abs_mom_20d" if col == "mom_20d" else col
        if src in df.columns:
            z = (df[src] - df[src].mean()) / (df[src].std() + 1e-8)
            df[f"_z_{col}"] = z * weight

    z_cols = [c for c in df.columns if c.startswith("_z_")]
    df["accumulation_score"] = df[z_cols].sum(axis=1)

    # Filter: must have volume expansion and price flatness
    df.loc[
        ~((df["vol_ratio_5d"] > 1.2) & (df["mom_20d"].abs() < 5)),
        "accumulation_score"
    ] = -999

    df = df.drop(columns=z_cols + ["price_flat", "vol_price_div", "_abs_mom_20d"])
    df = df.sort_values("accumulation_score", ascending=False)
    df = df[df["accumulation_score"] > -999]

    logger.info(f"Generated accumulation signals for {len(df)} boards. Top 5: {df.head(5).index.tolist()}")
    return df


def generate_rotation_shift_signals(industry_bars: pd.DataFrame) -> pd.DataFrame:
    """Generate strong-weak rotation shift (强弱切换) signals.

    Find sectors that are transitioning from weak to strong — the "second tier"
    that is accelerating while leaders may be peaking.

    Logic:
    - 60d momentum is mediocre or negative (wasn't a recent winner)
    - 20d momentum is positive and accelerating (turning around)
    - 5d momentum > 20d momentum / 4 (short-term acceleration)
    - Volume expanding (capital rotating in)
    """
    mom = calc_momentum(industry_bars)
    vol = calc_volume_signal(industry_bars)
    trend = calc_trend_strength(industry_bars)

    df = mom.join(vol, how="outer").join(trend, how="outer")

    # Acceleration: 5d momentum is strong relative to 20d
    df["acceleration"] = 0.0
    mask_accel = (df["mom_5d"] > 0) & (df["mom_20d"] > 0)
    df.loc[mask_accel, "acceleration"] = df.loc[mask_accel, "mom_5d"] / (df.loc[mask_accel, "mom_20d"].abs() + 0.01)

    # Filter: was weak (60d rank in bottom half) but now strengthening (20d > 0)
    mom_60d_median = df["mom_60d"].median()
    df["was_weak"] = (df["mom_60d"] <= mom_60d_median).astype(float)

    score_components = {
        "mom_5d": 0.20,            # short-term strength
        "mom_20d": 0.20,           # medium-term turn
        "acceleration": 0.20,      # acceleration is key
        "vol_ratio_5d": 0.20,      # capital rotating in
        "ma_alignment": 0.10,      # trend starting to form
        "mom_60d": -0.10,          # was weak = more rotation potential
    }

    for col, weight in score_components.items():
        if col in df.columns:
            z = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            df[f"_z_{col}"] = z * weight

    z_cols = [c for c in df.columns if c.startswith("_z_")]
    df["shift_score"] = df[z_cols].sum(axis=1)

    # Filter: must be transitioning (was weak, now strengthening)
    df.loc[
        ~((df["was_weak"] > 0) & (df["mom_20d"] > 0) & (df["mom_5d"] > 0)),
        "shift_score"
    ] = -999

    df = df.drop(columns=z_cols + ["acceleration", "was_weak"])
    df = df.sort_values("shift_score", ascending=False)
    df = df[df["shift_score"] > -999]

    logger.info(f"Generated rotation-shift signals for {len(df)} boards. Top 5: {df.head(5).index.tolist()}")
    return df
