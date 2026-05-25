"""QMT-mock broker for fidelity testing (P8-β-1a, docs/dialog/ round 56).

Purpose
-------
Subclass of :class:`DryRunBroker` that simulates xtquant's *async* order
lifecycle without depending on the real ``xtquant`` SDK (Windows-only).
Used by ``tests/test_execute_orders_fidelity.py`` (β-1c) as the
"ground-truth-ish" broker against which DryRunBroker is diffed under
the Rule #8 three-constraint fidelity assertion.

Differences from DryRunBroker
-----------------------------
- ``autofill`` is forced to ``False``: fills are managed manually via
  :meth:`process_pending_orders` so callers can interleave broker state
  inspection between ticks (matches the real QMT request → async callback
  → status-poll lifecycle).
- :meth:`place_limit_order` returns immediately with ``status='pending'``
  and a ``"MOCK-<8hex>"`` order id — *no* cash / position mutation at
  submit time (only freeze).
- Configurable partial-fill (50% default chance, split over 2-3 ticks)
  and reject (cash-insufficient / limit-up / limit-down / per-test
  ``force_reject``) — both deterministic given the config seed.
- :meth:`cancel_order` on a partially-filled order refunds the unfilled
  portion (buy: cash refund from frozen; sell: shares returned to
  ``shares_available``).

NOT a substitute for the real xtquant path
------------------------------------------
The async-fill / partial / reject models here are *behavioral* — they
match the *shape* of QMT's order lifecycle but not the exact tick
microstructure of any specific broker (国金 in our case).  The β-1c
fidelity test treats this broker as ground-truth-*ish* only to detect
divergence in DryRunBroker's accounting; β-3 (Approach B, future) is the
single-case validation against the real QMT-paper environment that
verifies *this* mock is realistic.
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

from loguru import logger

from .dryrun_broker import DryRunBroker
from .qmt_broker import (
    AccountInfo, EmergencyResult, OrderResult, OrderStatus, Position,
    _emergency_liquidate_impl,
)


# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────

@dataclass
class _QMTMockConfig:
    """Per-test override of mock fill / reject behavior.

    seed
        RNG seed for reproducible partial-fill splits.
    partial_fill_chance
        Probability that an order takes >1 tick to fully fill.
        Default 0.0 means "always full first tick" — keep test cases
        that don't care about async deterministic.
    partial_ticks_range
        ``(lo, hi)`` inclusive number of ticks when an order goes
        partial.  Per-tick allocation is roughly even in lots of 100,
        with the remainder added to the last tick.
    pre_close
        ``code → prev close`` table used for limit-up / limit-down
        reject checks.  If ``None``, no limit-band enforcement.
    limit_pct
        Symmetric trading band around ``pre_close`` (default ±10%).
        Note: 科创板 / 创业板 (20%) and ST (5%) are not modeled — caller
        must supply per-test config if needed.  Out of scope for β-1a.
    force_reject
        ``code → reject reason`` per-test override that supersedes
        cash / balance checks.  Useful for simulating "broker said no"
        without contriving the underlying account state.
    force_fill_plan
        ``code → list[shares-per-tick]`` per-test override that bypasses
        the random partial-fill RNG.  Sum must equal the submitted
        shares; each entry should be a multiple of 100.
    """
    seed: int = 42
    partial_fill_chance: float = 0.0
    partial_ticks_range: tuple[int, int] = (2, 3)
    pre_close: Optional[dict[str, float]] = None
    limit_pct: float = 0.10
    force_reject: Optional[dict[str, str]] = None
    force_fill_plan: Optional[dict[str, list[int]]] = None


@dataclass
class _PendingFill:
    """Internal per-order fill plan (one tick advance per
    :meth:`QMTMockBroker.process_pending_orders` call)."""
    order_id: str
    code: str
    action: Literal["buy", "sell"]
    shares_total: int
    limit_price: float
    fill_plan: list[int]
    ticks_processed: int = 0


# ──────────────────────────────────────────────────────────────────
# Main mock broker
# ──────────────────────────────────────────────────────────────────

class QMTMockBroker(DryRunBroker):
    """Mock QMT broker simulating async fill + partial + reject.

    See module docstring for rationale.  Public surface mirrors
    :class:`mp.execution.qmt_broker.QMTBroker` (connect / disconnect /
    get_* / place_limit_order / cancel_order) so a caller can swap
    between ``QMTBroker``, ``DryRunBroker``, and ``QMTMockBroker``
    without changing call sites.
    """

    def __init__(
        self,
        cash: float = 0.0,
        positions: list[Position] | None = None,
        config: _QMTMockConfig | None = None,
        account_id: str = "mock",
    ):
        super().__init__(
            cash=cash, positions=positions, autofill=False,
            account_id=account_id,
        )
        self._config = config or _QMTMockConfig()
        self._rng = random.Random(self._config.seed)
        self._cash_frozen: float = 0.0
        self._pending: list[_PendingFill] = []

    # ── account queries ─────────────────────────────────────────

    def get_account_info(self) -> AccountInfo:
        self._require_connected()
        mv = sum(p.market_value for p in self._positions.values())
        return AccountInfo(
            cash_available=self._cash,
            cash_frozen=self._cash_frozen,
            market_value=mv,
            total_assets=self._cash + self._cash_frozen + mv,
            updated_at=datetime.now().isoformat(timespec="seconds"),
        )

    # ── trading ─────────────────────────────────────────────────

    def place_limit_order(
        self,
        code: str,
        action: Literal["buy", "sell"],
        shares: int,
        limit_price: float,
        order_remark: str = "",
    ) -> OrderResult:
        self._require_connected()

        if shares < 100 or shares % 100 != 0:
            return OrderResult(success=False, error=f"shares={shares} not a valid lot")
        if limit_price <= 0:
            return OrderResult(success=False, error=f"limit_price={limit_price} invalid")

        if self._config.force_reject and code in self._config.force_reject:
            return self._reject(code, action, shares, limit_price,
                                self._config.force_reject[code])

        if self._config.pre_close and code in self._config.pre_close:
            pc = self._config.pre_close[code]
            up_lim = round(pc * (1 + self._config.limit_pct), 2)
            dn_lim = round(pc * (1 - self._config.limit_pct), 2)
            if limit_price > up_lim:
                return self._reject(code, action, shares, limit_price,
                                    f"limit_price {limit_price:.2f} > up_lim {up_lim:.2f} (涨停拒单)")
            if limit_price < dn_lim:
                return self._reject(code, action, shares, limit_price,
                                    f"limit_price {limit_price:.2f} < dn_lim {dn_lim:.2f} (跌停拒单)")

        if action == "buy":
            cost = shares * limit_price
            if cost > self._cash:
                return self._reject(code, action, shares, limit_price,
                                    f"insufficient cash: need {cost:.2f}, have {self._cash:.2f}")
        else:
            cur = self._positions.get(code)
            have = cur.shares_available if cur else 0
            if have < shares:
                return self._reject(code, action, shares, limit_price,
                                    f"insufficient shares: want sell {shares}, have {have}")

        order_id = f"MOCK-{uuid.uuid4().hex[:8]}"
        logger.info(
            "[QMT-MOCK] {} {} {} 股 @ ¥{:.2f}  (id={}, queued)",
            action.upper(), code, shares, limit_price, order_id,
        )

        if action == "buy":
            self._cash -= shares * limit_price
            self._cash_frozen += shares * limit_price
        else:
            p = self._positions[code]
            self._positions[code] = Position(
                code=p.code, name=p.name,
                shares_total=p.shares_total,
                shares_available=p.shares_available - shares,
                avg_cost=p.avg_cost,
                market_price=p.market_price,
                market_value=p.market_value,
            )

        self._orders.append(OrderStatus(
            order_id=order_id, code=code, action=action,
            shares_submitted=shares,
            shares_filled=0,
            avg_fill_price=0.0,
            status="pending",
        ))

        fill_plan = self._build_fill_plan(code, shares)
        self._pending.append(_PendingFill(
            order_id=order_id, code=code, action=action,
            shares_total=shares, limit_price=limit_price,
            fill_plan=fill_plan,
        ))

        return OrderResult(
            success=True, order_id=order_id,
            code=code, action=action,
            shares=shares, limit_price=limit_price,
        )

    def emergency_liquidate_all(
        self,
        confirm_string: str,
        mode: Literal["limit", "market"] = "limit",
        limit_offset_pct: float = -0.5,
        prev_close: dict[str, float] | None = None,
    ) -> EmergencyResult:
        """Sell all sell-able positions + cancel all pending orders.

        Delegates to :func:`mp.execution.qmt_broker._emergency_liquidate_impl`.
        Because this broker is async-fill (``autofill=False``),
        ``total_realized_cash`` will be ~0 immediately after return —
        sells are queued in the pending list.  Caller can advance the
        clock with :meth:`process_pending_orders` to observe fills.
        """
        return _emergency_liquidate_impl(
            self, confirm_string, mode, limit_offset_pct, prev_close,
        )

    def cancel_order(self, order_id: str) -> OrderResult:
        self._require_connected()
        o = self._find_order(order_id)
        if o is None:
            return OrderResult(success=False, order_id=order_id, error="not found")
        if o.status in ("filled", "cancelled", "rejected"):
            return OrderResult(success=False, order_id=order_id,
                                error=f"already {o.status}")
        unfilled = o.shares_submitted - o.shares_filled
        entry = next((e for e in self._pending if e.order_id == order_id), None)
        limit_price = entry.limit_price if entry is not None else (o.avg_fill_price or 0.0)
        if entry is not None:
            self._pending = [e for e in self._pending if e.order_id != order_id]

        if o.action == "buy":
            refund = unfilled * limit_price
            self._cash_frozen -= refund
            self._cash += refund
        else:
            p = self._positions.get(o.code)
            if p is not None:
                self._positions[o.code] = Position(
                    code=p.code, name=p.name,
                    shares_total=p.shares_total,
                    shares_available=p.shares_available + unfilled,
                    avg_cost=p.avg_cost,
                    market_price=p.market_price,
                    market_value=p.market_value,
                )
        o.status = "cancelled"
        logger.info("[QMT-MOCK] CANCEL {} (unfilled={})", order_id, unfilled)
        return OrderResult(success=True, order_id=order_id)

    # ── async fill processing ───────────────────────────────────

    def process_pending_orders(self, now: datetime | str | None = None) -> int:
        """Advance every still-pending order by one tick.

        Returns the number of orders that *completed* (transitioned to
        ``status='filled'``) on this call.  Caller may invoke
        repeatedly; once an order is fully filled or cancelled it is
        removed from the internal queue and no longer touched.
        """
        self._require_connected()
        completed = 0
        still_pending: list[_PendingFill] = []
        for entry in self._pending:
            if entry.ticks_processed >= len(entry.fill_plan):
                continue
            tick_shares = entry.fill_plan[entry.ticks_processed]
            entry.ticks_processed += 1
            self._apply_fill(entry, tick_shares)
            if entry.ticks_processed >= len(entry.fill_plan):
                completed += 1
                self._set_order_status(entry.order_id, "filled")
            else:
                self._set_order_status(entry.order_id, "partial")
                still_pending.append(entry)
        self._pending = still_pending
        return completed

    # ── internal helpers ────────────────────────────────────────

    def _build_fill_plan(self, code: str, shares: int) -> list[int]:
        if self._config.force_fill_plan and code in self._config.force_fill_plan:
            plan = list(self._config.force_fill_plan[code])
            if sum(plan) != shares:
                raise ValueError(
                    f"force_fill_plan[{code}]={plan} sums to {sum(plan)}, "
                    f"expected {shares}"
                )
            return plan
        if shares < 200 or self._rng.random() >= self._config.partial_fill_chance:
            return [shares]
        lo, hi = self._config.partial_ticks_range
        n_ticks = self._rng.randint(lo, hi)
        total_lots = shares // 100
        if total_lots < n_ticks:
            n_ticks = total_lots
        per_tick_lots = total_lots // n_ticks
        remainder_lots = total_lots - per_tick_lots * n_ticks
        plan = [per_tick_lots * 100] * n_ticks
        if remainder_lots:
            plan[-1] += remainder_lots * 100
        return plan

    def _apply_fill(self, entry: _PendingFill, tick_shares: int):
        order = self._find_order(entry.order_id)
        if order is None:
            return
        order.shares_filled += tick_shares
        order.avg_fill_price = entry.limit_price

        if entry.action == "buy":
            spent = tick_shares * entry.limit_price
            self._cash_frozen -= spent
            if entry.code in self._positions:
                p = self._positions[entry.code]
                new_total = p.shares_total + tick_shares
                new_avg_cost = (
                    p.avg_cost * p.shares_total + entry.limit_price * tick_shares
                ) / new_total
                self._positions[entry.code] = Position(
                    code=p.code, name=p.name,
                    shares_total=new_total,
                    shares_available=p.shares_available,
                    avg_cost=new_avg_cost,
                    market_price=entry.limit_price,
                    market_value=new_total * entry.limit_price,
                )
            else:
                self._positions[entry.code] = Position(
                    code=entry.code, name=entry.code,
                    shares_total=tick_shares,
                    shares_available=0,
                    avg_cost=entry.limit_price,
                    market_price=entry.limit_price,
                    market_value=tick_shares * entry.limit_price,
                )
        else:
            proceeds = tick_shares * entry.limit_price
            self._cash += proceeds
            p = self._positions[entry.code]
            new_total = p.shares_total - tick_shares
            if new_total == 0:
                del self._positions[entry.code]
            else:
                self._positions[entry.code] = Position(
                    code=p.code, name=p.name,
                    shares_total=new_total,
                    shares_available=p.shares_available,
                    avg_cost=p.avg_cost,
                    market_price=entry.limit_price,
                    market_value=new_total * entry.limit_price,
                )

    def _find_order(self, order_id: str) -> Optional[OrderStatus]:
        for o in self._orders:
            if o.order_id == order_id:
                return o
        return None

    def _set_order_status(self, order_id: str, status: str):
        o = self._find_order(order_id)
        if o is not None:
            o.status = status

    def _reject(
        self,
        code: str,
        action: str,
        shares: int,
        limit_price: float,
        reason: str,
    ) -> OrderResult:
        order_id = f"MOCK-{uuid.uuid4().hex[:8]}"
        self._orders.append(OrderStatus(
            order_id=order_id, code=code, action=action,
            shares_submitted=shares, shares_filled=0,
            avg_fill_price=0.0,
            status="rejected", error_msg=reason,
        ))
        logger.warning(
            "[QMT-MOCK] REJECT {} {} {} @ ¥{:.2f}: {}",
            action.upper(), code, shares, limit_price, reason,
        )
        return OrderResult(
            success=False, order_id=order_id, error=reason,
            code=code, action=action,
            shares=shares, limit_price=limit_price,
        )
