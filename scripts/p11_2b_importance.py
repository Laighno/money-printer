"""P11-2b: dump feature_importance for an intraday BlendRanker model.

Loads a saved BlendRanker pair (primary + extreme) and prints feature gains
sorted descending. For the round-77 attribution: how much of total gain do
the 4 INTRADAY_EXTRA_COLUMNS account for vs the 64 FACTOR_COLUMNS?

Usage
-----
  # Full P11-2 model (68 features)
  .venv/bin/python scripts/p11_2b_importance.py --prefix data/intraday_blend

  # Control P11-2b model (64 features)
  .venv/bin/python scripts/p11_2b_importance.py --prefix data/intraday_blend_control

  # Both side-by-side
  .venv/bin/python scripts/p11_2b_importance.py --prefix data/intraday_blend --compare-prefix data/intraday_blend_control
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mp.ml.dataset import FACTOR_COLUMNS  # noqa: E402
from mp.ml.intraday_features import INTRADAY_EXTRA_COLUMNS  # noqa: E402
from mp.ml.model import BlendRanker  # noqa: E402


def _dump_one(prefix: str) -> Dict[str, float]:
    """Load BlendRanker pair, return {feature: primary_gain}."""
    if not Path(f"{prefix}_primary.lgb").exists():
        logger.error("Not found: {}_primary.lgb", prefix)
        return {}
    ranker = BlendRanker()
    if not ranker.load(prefix):
        logger.error("Failed to load BlendRanker from prefix {}", prefix)
        return {}
    importance = ranker.primary.feature_importance
    logger.info("Loaded {}: {} features", prefix, len(importance))
    return dict(importance)


def _print_table(gains: Dict[str, float], label: str, n_top: int = 10) -> None:
    if not gains:
        return
    total = sum(gains.values()) or 1.0
    df = pd.DataFrame({"feature": list(gains.keys()), "gain": list(gains.values())})
    df["gain_pct"] = 100.0 * df["gain"] / total
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)

    print(f"\n=== {label} (primary model, {len(df)} features) ===")
    print(f"Total gain: {total:,.0f}")

    # Intraday extras (highlight)
    extras_mask = df["feature"].isin(INTRADAY_EXTRA_COLUMNS)
    if extras_mask.any():
        extras_df = df[extras_mask].copy()
        print(f"\nIntraday extras gain breakdown (sum {extras_df['gain'].sum():,.0f} = "
              f"{extras_df['gain_pct'].sum():.2f}% of total):")
        print(extras_df.to_string(index=False, formatters={
            "gain": lambda v: f"{v:>12,.0f}",
            "gain_pct": lambda v: f"{v:>6.2f}%",
        }))

    print(f"\nTop {n_top} features by gain:")
    print(df.head(n_top).to_string(index=False, formatters={
        "gain": lambda v: f"{v:>12,.0f}",
        "gain_pct": lambda v: f"{v:>6.2f}%",
    }))


def _compare(gains_a: Dict[str, float], label_a: str,
             gains_b: Dict[str, float], label_b: str) -> None:
    """Side-by-side rank comparison of shared features."""
    if not gains_a or not gains_b:
        return
    shared = set(gains_a) & set(gains_b)
    print(f"\n=== Compare {label_a} vs {label_b} (shared {len(shared)} features) ===")

    rows = []
    for f in shared:
        rows.append({
            "feature": f,
            f"{label_a}_gain": gains_a[f],
            f"{label_b}_gain": gains_b[f],
            "delta": gains_a[f] - gains_b[f],
        })
    df = pd.DataFrame(rows).sort_values("delta", key=abs, ascending=False).head(15)
    print(df.to_string(index=False, formatters={
        f"{label_a}_gain": lambda v: f"{v:>12,.0f}",
        f"{label_b}_gain": lambda v: f"{v:>12,.0f}",
        "delta": lambda v: f"{v:>+12,.0f}",
    }))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="data/intraday_blend",
                    help="Primary blend model prefix (default: data/intraday_blend)")
    ap.add_argument("--compare-prefix", default=None,
                    help="Optional second prefix for side-by-side compare")
    ap.add_argument("--top", type=int, default=10, help="Top N features to print")
    args = ap.parse_args()

    gains = _dump_one(args.prefix)
    label = Path(args.prefix).name
    _print_table(gains, label, n_top=args.top)

    if args.compare_prefix:
        gains2 = _dump_one(args.compare_prefix)
        label2 = Path(args.compare_prefix).name
        _print_table(gains2, label2, n_top=args.top)
        _compare(gains, label, gains2, label2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
