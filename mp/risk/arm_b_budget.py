"""Arm B (14:30 OOS bucket) daily budget tracker — guardrail (a), round 161.

Background
----------
The 14:30 pipeline (intraday_plan + execute_orders) is the OOS Arm B bucket
that round 159 unlocked for small-position real money. To bound blast radius
during the burn-in observation window, total *new* buying notional on Arm B
must stay ≤ ``ARM_B_BUDGET_MAX_DEFAULT`` (default 20,000 RMB) per trading day.

This module is the single source of truth: callers ask before submitting,
commit after a successful broker accept, release on cancel / sell.

Persistence
-----------
Daily state is stored as JSON at ``data/arm_b_budget_state.json`` so the
tracker survives process restarts within the same trading day. On a new
date the state is reset automatically. Only buy notional is tracked.
Sells reduce the spent counter (release).

Rule (round 159)
----------------
Any change that lets OOS bucket > 20,000 RMB needs explicit user approval.
The tracker enforces this by **rejecting** any buy that would push the
daily committed notional above ``budget_max``; callers cannot override.

The cap can be set via env ``ARM_B_BUDGET_MAX`` (must be ≤ 20000 unless
the calling caller is round 159's user-authorized escalation path —
the tracker still enforces; bumping the env above 20000 is the auditable
trace).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from loguru import logger

# round 190 (user 拍板, 2026-06-01): cap lifted from ¥20,000 → effectively
# unlimited (¥99,999,999) based on v4.2 true-OOS backtest evidence:
#   - v4.1 +¥139k dual−max(solo) was look-ahead bias (model trained on test
#     window), proved by user round 188 catch
#   - v4.2 cutoff=2025-08-31 true OOS: dual − max(solo) = +¥12k (-91% from
#     v4.1, but still positive) → OOS path真有互补 alpha 给 EOD
#   - cap=¥20k was set round 161 conservatively during burn-in; now v4.2
#     supports release (see docs/dialog/to_engineer.md round 190)
# The original ¥20k value retained here as audit anchor.
_ORIGINAL_CAP_ROUND_161 = 20_000.0  # historical anchor, do not use
ARM_B_BUDGET_MAX_DEFAULT = 99_999_999.0  # RMB; round 190 effectively no-cap
DEFAULT_STATE_PATH = Path("data/arm_b_budget_state.json")


@dataclass
class BudgetState:
    trading_date: str            # ISO YYYY-MM-DD
    committed_notional: float    # cumulative buy notional today
    budget_max: float            # cap in effect (snapshotted at first use)
    events: list                 # ordered audit log of commits/releases


def _today_iso() -> str:
    return date.today().isoformat()


def _cap_from_env(default: float = ARM_B_BUDGET_MAX_DEFAULT) -> float:
    raw = os.environ.get("ARM_B_BUDGET_MAX")
    if raw is None:
        return default
    try:
        v = float(raw)
    except ValueError:
        logger.warning("ARM_B_BUDGET_MAX={!r} not float; using default {}",
                       raw, default)
        return default
    if v <= 0:
        logger.warning("ARM_B_BUDGET_MAX={} non-positive; using default {}",
                       v, default)
        return default
    return v


class ArmBBudgetTracker:
    """Daily-keyed budget tracker for the 14:30 OSS Arm B bucket.

    Usage
    -----
        tracker = ArmBBudgetTracker.load()
        ok, msg = tracker.check_buy("000539", 1000, 6.39)  # 6390 RMB
        if ok:
            # ... submit to broker; on success:
            tracker.commit_buy("000539", 1000, 6.39, order_id="123")
        else:
            logger.warning("Arm B cap blocked buy: {}", msg)

    On sell or cancel:
        tracker.release_buy("000539", 1000, 6.39, reason="cancel:order_id=123")
    """

    def __init__(self, state: BudgetState, path: Path):
        self._state = state
        self._path = path

    # ── construction ──────────────────────────────────────────────

    @classmethod
    def load(cls, path: Optional[Path] = None,
             budget_max: Optional[float] = None) -> "ArmBBudgetTracker":
        """Load tracker; reset to new day if stored state is stale."""
        path = path or DEFAULT_STATE_PATH
        budget_max = budget_max if budget_max is not None else _cap_from_env()

        today = _today_iso()
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if raw.get("trading_date") == today:
                    state = BudgetState(
                        trading_date=raw["trading_date"],
                        committed_notional=float(raw.get("committed_notional", 0.0)),
                        budget_max=float(raw.get("budget_max", budget_max)),
                        events=list(raw.get("events", [])),
                    )
                    return cls(state, path)
                # Stale: archive old, start fresh
                logger.info("Arm B budget state stale ({} != {}); resetting",
                            raw.get("trading_date"), today)
            except Exception as e:
                logger.warning("Arm B budget state unreadable ({}); resetting", e)

        state = BudgetState(
            trading_date=today,
            committed_notional=0.0,
            budget_max=budget_max,
            events=[],
        )
        tracker = cls(state, path)
        tracker._persist()
        return tracker

    # ── public API ────────────────────────────────────────────────

    def available(self) -> float:
        return max(0.0, self._state.budget_max - self._state.committed_notional)

    def committed(self) -> float:
        return self._state.committed_notional

    def budget_max(self) -> float:
        return self._state.budget_max

    def state(self) -> dict:
        return {
            "trading_date": self._state.trading_date,
            "committed_notional": self._state.committed_notional,
            "budget_max": self._state.budget_max,
            "available": self.available(),
            "events_count": len(self._state.events),
        }

    def check_buy(self, code: str, shares: int,
                  limit_price: float) -> tuple[bool, str]:
        """Return (ok, reason). Read-only; no state change."""
        notional = float(shares) * float(limit_price)
        if notional <= 0:
            return False, f"notional={notional} non-positive"
        avail = self.available()
        if notional > avail:
            return False, (
                f"Arm B daily cap blocked: code={code} shares={shares} "
                f"price={limit_price:.2f} notional={notional:.0f} > "
                f"available={avail:.0f} (committed={self.committed():.0f}, "
                f"cap={self._state.budget_max:.0f})"
            )
        return True, (
            f"OK: notional={notional:.0f} ≤ available={avail:.0f}; "
            f"committed_after_would_be={self.committed()+notional:.0f}/"
            f"{self._state.budget_max:.0f}"
        )

    def commit_buy(self, code: str, shares: int, limit_price: float,
                   order_id: Optional[str] = None) -> None:
        """Commit notional after broker acceptance.

        Re-validates the cap (defense in depth — caller may have skipped
        check_buy). Raises ValueError if commit would exceed cap.
        """
        notional = float(shares) * float(limit_price)
        ok, msg = self.check_buy(code, shares, limit_price)
        if not ok:
            raise ValueError(f"commit_buy refused: {msg}")
        self._state.committed_notional += notional
        self._state.events.append({
            "action": "commit_buy",
            "code": code,
            "shares": int(shares),
            "price": float(limit_price),
            "notional": float(notional),
            "order_id": order_id,
            "after_committed": self._state.committed_notional,
        })
        self._persist()
        logger.info("Arm B cap commit: {} {}@{:.2f} notional={:.0f}; "
                    "committed={:.0f}/{:.0f}",
                    code, shares, limit_price, notional,
                    self._state.committed_notional, self._state.budget_max)

    def release_buy(self, code: str, shares: int, limit_price: float,
                    reason: str = "") -> None:
        """Release committed notional (cancel / partial fill correction / sell).

        Note: sells release based on the *original buy price*; pass the
        cancelled order's actual price for accurate accounting. For simplicity
        the tracker does not maintain per-order ledger — it's a notional
        bucket. If precision matters, query the events log.
        """
        notional = float(shares) * float(limit_price)
        if notional <= 0:
            return
        new_total = max(0.0, self._state.committed_notional - notional)
        delta = self._state.committed_notional - new_total
        self._state.committed_notional = new_total
        self._state.events.append({
            "action": "release_buy",
            "code": code,
            "shares": int(shares),
            "price": float(limit_price),
            "notional": float(notional),
            "released": float(delta),
            "reason": reason,
            "after_committed": self._state.committed_notional,
        })
        self._persist()
        logger.info("Arm B cap release: {} {}@{:.2f} released={:.0f}; "
                    "committed={:.0f}/{:.0f} ({})",
                    code, shares, limit_price, delta,
                    self._state.committed_notional, self._state.budget_max,
                    reason or "no-reason-given")

    # ── persistence ───────────────────────────────────────────────

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({
                "trading_date": self._state.trading_date,
                "committed_notional": self._state.committed_notional,
                "budget_max": self._state.budget_max,
                "events": self._state.events,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── for tests ─────────────────────────────────────────────────

    def _reset_for_tests(self) -> None:
        self._state.committed_notional = 0.0
        self._state.events = []
        self._persist()
