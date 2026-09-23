"""Unified simulated broker for A-share backtesting and paper trading.

Consolidates trade execution logic from three previous implementations:
- mp/execution/trader.py (SimulatedTrader)
- mp/backtest/ml_backtest.py (inline dicts)
- scripts/walk_forward_backtest.py (inline dicts + stamp tax)

Handles: slippage, commission, A-share stamp tax (sell-side, date-dependent),
lot rounding (100-share lots), cash management, and trade logging.
"""

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger


@dataclass
class BrokerPosition:
    code: str
    shares: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    peak_price: float = 0.0
    entry_date: str = ""
    # round 187 (user round 186 抓的 T+1 bug): A 股 T+1 锁仓 — 当日新增 buy
    # 的股数当日不可卖。production 靠 QMT shares_available 自动拒, backtest
    # 必须自模拟. 只锁当日 buy, 不锁 T-1 之前已有.
    last_buy_date: str = ""
    today_bought_shares: int = 0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    def available_shares(self, sim_date: str) -> int:
        """T+1 awareness: shares freely sellable today.

        ``today_bought_shares`` of the most recent buy day is locked until
        the next trading day. Old positions (T-1 or earlier) are unlocked.
        """
        if self.last_buy_date == sim_date:
            return max(0, self.shares - self.today_bought_shares)
        return self.shares

    @property
    def pnl_pct(self) -> float:
        if self.avg_cost <= 0:
            return 0.0
        return self.current_price / self.avg_cost - 1

    @property
    def drawdown_from_peak(self) -> float:
        if self.peak_price <= 0:
            return 0.0
        return self.current_price / self.peak_price - 1


@dataclass
class FeeSchedule:
    slippage_bps: float = 5             # linear fallback when ADV unavailable
    commission_bps: float = 3
    stamp_tax_bps_old: float = 10       # sell-side only, before cut date
    stamp_tax_bps_new: float = 5        # sell-side only, from cut date
    stamp_tax_cut_date: str = "2023-08-28"
    # ── Square-root market-impact model ──────────────────────────────────────
    # When ADV (average daily value) is provided at trade time, slippage is
    # estimated as:
    #   impact_bps = impact_alpha_bps * sqrt(notional / adv)
    # clamped to a minimum of min_slippage_bps (bid-ask spread floor).
    # Falls back to flat ``slippage_bps`` when ADV is unknown.
    #
    # Calibration for A-share ZZ500 (mid-caps, ~500M CNY ADV):
    #   10万 portfolio, 1万/stock → participation ~0.002% → ~0.7 bps impact
    #   10M  portfolio, 1M/stock  → participation ~0.2%  → ~6.7 bps impact
    use_sqrt_impact: bool = True
    impact_alpha_bps: float = 150.0     # α; typical range 100–300 for A-shares
    min_slippage_bps: float = 3.0       # bid-ask spread floor (always charged)

    def _slippage_bps(self, notional: float | None = None, adv: float | None = None) -> float:
        """Compute one-side slippage in bps for a given trade."""
        if self.use_sqrt_impact and notional is not None and adv is not None and adv > 0:
            impact = self.impact_alpha_bps * (notional / adv) ** 0.5
            return max(self.min_slippage_bps, impact)
        return self.slippage_bps

    def buy_exec_price(self, price: float, notional: float | None = None, adv: float | None = None) -> float:
        return price * (1 + self._slippage_bps(notional, adv) / 10_000)

    def sell_exec_price(self, price: float, notional: float | None = None, adv: float | None = None) -> float:
        return price * (1 - self._slippage_bps(notional, adv) / 10_000)

    def buy_fee(self, cost: float) -> float:
        return cost * self.commission_bps / 10_000

    def sell_fee(self, proceeds: float, date: str = "") -> float:
        commission = proceeds * self.commission_bps / 10_000
        stamp_rate = self._stamp_tax_bps(date)
        stamp = proceeds * stamp_rate / 10_000
        return commission + stamp

    def _stamp_tax_bps(self, date: str) -> float:
        if not date:
            return self.stamp_tax_bps_new
        date_str = str(date)[:10]
        if date_str >= self.stamp_tax_cut_date:
            return self.stamp_tax_bps_new
        return self.stamp_tax_bps_old

    def buy_cost_bps(self, notional: float | None = None, adv: float | None = None) -> float:
        """One-way buy cost in bps: slippage + commission."""
        return self._slippage_bps(notional, adv) + self.commission_bps

    def sell_cost_bps(
        self, date: str = "", notional: float | None = None, adv: float | None = None,
    ) -> float:
        """One-way sell cost in bps: slippage + commission + stamp tax."""
        return self._slippage_bps(notional, adv) + self.commission_bps + self._stamp_tax_bps(date)


LOT_SIZE = 100


class SimulatedBroker:
    """Simulated A-share broker with realistic cost modeling."""

    def __init__(self, initial_capital: float, fees: FeeSchedule | None = None, silent: bool = False):
        self.cash: float = initial_capital
        self.initial_capital: float = initial_capital
        self.fees: FeeSchedule = fees or FeeSchedule()
        self.positions: dict[str, BrokerPosition] = {}
        self.trade_log: list[dict] = []
        self.silent: bool = silent

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def buy(self, code: str, price: float, target_value: float | None = None,
            shares: int | None = None, date: str = "", action: str = "BUY",
            adv: float | None = None) -> dict | None:
        """Execute a buy order.

        Parameters
        ----------
        adv:
            Average daily trading value (amount) in CNY over the past N days,
            used to compute square-root market impact.  When ``None``, falls
            back to the flat ``FeeSchedule.slippage_bps``.
        """
        if price <= 0:
            return None

        # Estimate notional for impact model; target_value is a good proxy.
        notional_est = target_value or (shares or 1) * price
        exec_price = self.fees.buy_exec_price(price, notional=notional_est, adv=adv)

        if shares is not None:
            buy_shares = int(shares / LOT_SIZE) * LOT_SIZE
        elif target_value is not None:
            if code in self.positions:
                current_value = self.positions[code].shares * price
                delta = target_value - current_value
                if delta <= exec_price * LOT_SIZE * 0.5:
                    return None
            else:
                delta = target_value
            buy_shares = int(delta / exec_price / LOT_SIZE) * LOT_SIZE
        else:
            return None

        if buy_shares <= 0:
            return None

        cost = buy_shares * exec_price
        fee = self.fees.buy_fee(cost)
        total_cost = cost + fee

        if total_cost > self.cash:
            buy_shares = int(self.cash / exec_price / LOT_SIZE) * LOT_SIZE
            if buy_shares <= 0:
                return None
            cost = buy_shares * exec_price
            fee = self.fees.buy_fee(cost)
            total_cost = cost + fee

        self.cash -= total_cost

        if code in self.positions:
            pos = self.positions[code]
            total_shares = pos.shares + buy_shares
            pos.avg_cost = (pos.avg_cost * pos.shares + exec_price * buy_shares) / total_shares
            pos.shares = total_shares
            pos.current_price = price
            pos.peak_price = max(pos.peak_price, price)
        else:
            pos = BrokerPosition(
                code=code,
                shares=buy_shares,
                avg_cost=exec_price,
                current_price=price,
                peak_price=price,
                entry_date=str(date),
            )
            self.positions[code] = pos
        # round 187 (user round 186 T+1 lock fix): record same-day-bought
        # shares so .sell() can refuse to sell what was bought today.
        # When buying again later the same day, accumulate today_bought_shares.
        if pos.last_buy_date == str(date):
            pos.today_bought_shares += buy_shares
        else:
            pos.last_buy_date = str(date)
            pos.today_bought_shares = buy_shares

        # Friction breakdown for transparent reporting
        slippage_cost = (exec_price - price) * buy_shares  # always >= 0 for buy
        trade = {
            "date": date, "code": code, "action": action,
            "shares": buy_shares, "price": exec_price, "value": total_cost,
            "raw_price": price,
            "slippage_cost": slippage_cost,
            "commission": fee,
            "stamp_tax": 0.0,
            "total_friction": slippage_cost + fee,
        }
        self.trade_log.append(trade)
        if not self.silent:
            logger.info("BUY {}: {}shares @ {:.2f}, cost={:.0f}", code, buy_shares, exec_price, total_cost)
        return trade

    def sell(self, code: str, price: float, shares: int | None = None,
             date: str = "", action: str = "SELL",
             adv: float | None = None) -> dict | None:
        """Execute a sell order.

        Parameters
        ----------
        adv:
            Average daily trading value in CNY; used for sqrt impact model.
        """
        if code not in self.positions:
            return None
        pos = self.positions[code]
        if pos.shares <= 0:
            return None
        if price <= 0:
            return None

        sell_shares_est = shares if shares is not None else pos.shares
        notional_est = sell_shares_est * price
        exec_price = self.fees.sell_exec_price(price, notional=notional_est, adv=adv)

        if shares is None:
            sell_shares = pos.shares
        else:
            sell_shares = int(shares / LOT_SIZE) * LOT_SIZE
            sell_shares = min(sell_shares, pos.shares)

        # round 187 (user round 186 T+1 lock fix): A-share T+1 rule — shares
        # bought TODAY are locked until next trading day. production relies
        # on QMT shares_available; backtest must enforce here.
        available = pos.available_shares(str(date))
        sell_shares = min(sell_shares, available)

        if sell_shares <= 0:
            return None

        proceeds = sell_shares * exec_price
        # Compute commission + stamp tax separately for friction breakdown
        commission = proceeds * self.fees.commission_bps / 10_000
        stamp_rate = self.fees._stamp_tax_bps(str(date))
        stamp_tax = proceeds * stamp_rate / 10_000
        fee = commission + stamp_tax  # equivalent to self.fees.sell_fee(proceeds, str(date))
        net_proceeds = proceeds - fee

        self.cash += net_proceeds
        pos.shares -= sell_shares

        if pos.shares <= 0:
            del self.positions[code]

        # Friction breakdown for transparent reporting
        slippage_cost = (price - exec_price) * sell_shares  # >= 0 for sell (exec < price)
        trade = {
            "date": date, "code": code, "action": action,
            "shares": sell_shares, "price": exec_price, "value": net_proceeds,
            "raw_price": price,
            "slippage_cost": slippage_cost,
            "commission": commission,
            "stamp_tax": stamp_tax,
            "total_friction": slippage_cost + commission + stamp_tax,
        }
        self.trade_log.append(trade)
        if not self.silent:
            logger.info("SELL {}: {}shares @ {:.2f}, net={:.0f}", code, sell_shares, exec_price, net_proceeds)
        return trade

    def update_prices(self, prices: dict[str, float]) -> None:
        for code, price in prices.items():
            if code in self.positions:
                pos = self.positions[code]
                pos.current_price = price
                if price > pos.peak_price:
                    pos.peak_price = price

    def get_holdings_df(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame(columns=["code", "shares", "avg_cost", "current_price", "market_value", "pnl_pct", "entry_date"])
        rows = []
        for pos in self.positions.values():
            rows.append({
                "code": pos.code,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "pnl_pct": pos.pnl_pct,
                "entry_date": pos.entry_date,
            })
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# A-share price-limit / suspension fill rules (2026-09-23 limit-lock)
# ──────────────────────────────────────────────────────────────────────
# Pure helpers shared by backtests. A Top-K momentum picker frequently
# selects names that open 一字涨停 next day; assuming 100% fills there
# systematically overstates returns. ``fill_blocked`` says whether a
# buy/sell at the modelled fill time could actually have been filled.

from decimal import Decimal, ROUND_HALF_UP  # noqa: E402

# 创业板 20% band started with the registration reform on 2020-08-24
# (10% before). 科创板 (688/689) has been 20% since inception (2019-07-22).
CHINEXT_20PCT_START = "2020-08-24"
STAR_20PCT_START = "2019-07-22"

LIMIT_PCT_MAIN = 0.10
LIMIT_PCT_GROWTH = 0.20
LIMIT_PCT_ST = 0.05


def board_of(code: str) -> str:
    """Map a 6-digit A-share code to its board: main / chinext / star / bse."""
    c = str(code).zfill(6)
    if c.startswith(("300", "301", "302")):
        return "chinext"
    if c.startswith(("688", "689")):
        return "star"
    if c.startswith(("4", "8", "92")):
        return "bse"
    return "main"


def limit_pct(board: str, dt=None, *, is_st: bool = False) -> float | None:
    """Daily price-limit band for ``board`` on date ``dt`` (None = latest).

    Returns None for boards we deliberately do not model (北交所 30%, not in
    the ZZ500/HS300 universe) — callers then skip the limit check.
    ``is_st`` (5%) is only honoured when the caller has an ST flag; the
    walk-forward bars carry no ST marker so it defaults to False.
    """
    d = None if dt is None else str(pd.Timestamp(dt).date())
    if board == "bse":
        return None
    if board == "star":
        return LIMIT_PCT_GROWTH if (d is None or d >= STAR_20PCT_START) else LIMIT_PCT_MAIN
    if board == "chinext":
        if d is None or d >= CHINEXT_20PCT_START:
            return LIMIT_PCT_GROWTH
        return LIMIT_PCT_ST if is_st else LIMIT_PCT_MAIN
    return LIMIT_PCT_ST if is_st else LIMIT_PCT_MAIN


def round_cent(x: float) -> float:
    """Round to 分 with 四舍五入 (exchange convention), immune to binary
    float artefacts such as 9.99 * 1.1 == 10.989000000000001."""
    return float(Decimal(repr(float(x) + 1e-9)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def fill_blocked(action: str, bar, prev_close: float | None, *,
                 board: str = "main", dt=None, price: float | None = None,
                 strict: bool = False, is_st: bool = False,
                 tol_cents: int = 1) -> str | None:
    """Return why a fill is impossible, or None when it can fill.

    Parameters
    ----------
    action : "buy" | "sell"
    bar : mapping with open/high/low (and optionally volume) for the fill
        day, or None when the day's bar is missing (= suspended).
    prev_close : previous trading day's close (limit reference). None →
        limit check skipped (only the suspension check applies).
    board : from :func:`board_of`; drives the band via :func:`limit_pct`.
    dt : fill date, for the 创业板 10%→20% regime switch.
    price : the modelled fill price. Defaults to ``bar["open"]`` (T+1 open
        entry). Pass the 14:29 close for ENTRY_TIME=14_30.
    strict : when True, a fill price AT the limit is blocked outright
        (LIMIT_STRICT=1). When False, "open at limit but the day traded a
        range (high > low)" is treated as fillable — only a 一字板
        (high == low) blocks.
    tol_cents : the cached bars are 前复权 and re-rounded to 分, so the
        reconstructed limit price can be off by one tick; a price within
        ``tol_cents`` of the band counts as at-limit.

    Returns "suspended" | "limit_up" | "limit_down" | None.
    """
    if action not in ("buy", "sell"):
        raise ValueError(f"action must be buy/sell, got {action!r}")
    if bar is None:
        return "suspended"
    vol = bar.get("volume") if hasattr(bar, "get") else None
    if vol is not None and not pd.isna(vol) and float(vol) <= 0:
        return "suspended"
    if price is None:
        price = bar.get("open") if hasattr(bar, "get") else None
    if price is None or pd.isna(price) or float(price) <= 0:
        return "suspended"
    if prev_close is None or pd.isna(prev_close) or float(prev_close) <= 0:
        return None
    pct = limit_pct(board, dt, is_st=is_st)
    if pct is None:
        return None

    p = round_cent(price)
    hi = bar.get("high", None)
    lo = bar.get("low", None)
    one_word = (hi is not None and lo is not None and not pd.isna(hi) and not pd.isna(lo)
                and round_cent(hi) == round_cent(lo))
    tol = tol_cents / 100.0 + 1e-9

    if action == "buy":
        up = round_cent(float(prev_close) * (1.0 + pct))
        if p >= up - tol and (strict or one_word):
            return "limit_up"
        return None
    down = round_cent(float(prev_close) * (1.0 - pct))
    if p <= down + tol and (strict or one_word):
        return "limit_down"
    return None
