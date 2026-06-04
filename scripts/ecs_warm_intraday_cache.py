"""Warm intraday 1m cache via xtdata.download_history_data2.

Run BEFORE intraday_plan.py (called from ecs_intraday_execute.ps1 Step 2a).

Background (round 194, user 拍板 2026-06-02):
  XtMiniQmt 重启后默认只起交易, 不订阅行情. 行情 cache 不再被自动填充,
  intraday_plan.py:380-420 的 cache-read assumption broken.
  6/2 14:30 OOS task failed: "xtdata 1m returned 0 rows for all fields"

Fix:
  此脚本 explicit 调 download_history_data2 强制下载 universe 的当天 1m,
  ECS PS1 在 sleep_to_trigger 之前 (14:28-14:30 window) 跑此脚本.
  实测 615 codes ~30-60s, 在 sleep_to_trigger deadline 之内.

Exit codes:
  0 = OK
  1 = xtdata not connected / download failed
  2 = universe empty (mp.data 问题)
"""
from __future__ import annotations
import sys
import time
from datetime import date

from loguru import logger
from xtquant import xtdata

# Universe load 用 intraday_plan 同样 logic
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from scripts.intraday_plan import load_universe, _code_to_xtquant


def main() -> int:
    asof = date.today()
    asof_str = asof.strftime("%Y%m%d")

    logger.info("Warm cache: asof={}, loading universe...", asof_str)
    try:
        codes = load_universe()
    except Exception as e:
        logger.exception("Failed to load universe: {}", e)
        return 2

    if not codes:
        logger.error("Universe empty — refusing to warm cache")
        return 2

    xt_codes = [_code_to_xtquant(c) for c in codes]
    logger.info("Universe loaded: {} codes; calling download_history_data2 for 1m...",
                len(xt_codes))

    start = time.time()
    try:
        xtdata.download_history_data2(
            stock_list=xt_codes,
            period='1m',
            start_time=asof_str,
            end_time=asof_str,
            callback=None,
        )
    except Exception as e:
        logger.exception("download_history_data2 failed: {}", e)
        return 1
    elapsed = time.time() - start
    logger.info("download_history_data2 done in {:.1f}s for {} codes", elapsed, len(xt_codes))

    # Round 219 (advisor 218 spec B): full-universe coverage check, not 2-sample.
    # Today's 14:30 ABORT root cause was 12-min xtdata fetch in intraday_plan
    # despite warm cache declaring exit 0 — suggests warm cache didn't actually
    # populate. Sample of 2 codes can't detect this. Read full universe back
    # and require coverage ≥80% before declaring success.
    full_check = xtdata.get_market_data(
        field_list=['close'],
        stock_list=xt_codes,
        period='1m',
        start_time=f"{asof_str}093000",
        end_time=f"{asof_str}143000",
        count=-1,
        dividend_type='none',
        fill_data=False,
    )
    close_df = full_check.get('close')
    if close_df is None or close_df.empty:
        logger.error("Cache STILL empty after download for full universe — abort")
        return 1
    # close_df: rows = stock codes (xt_codes), cols = minute timestamps.
    # A code has coverage if it has ≥1 non-NaN value over the trading window.
    has_data = close_df.notna().sum(axis=1) > 0
    n_covered = int(has_data.sum())
    n_total = len(close_df)
    coverage = n_covered / n_total if n_total else 0.0
    logger.info("Warm cache coverage: {}/{} codes = {:.1%} (latest col={})",
                n_covered, n_total, coverage,
                list(close_df.columns)[-1] if len(close_df.columns) > 0 else None)
    if coverage < 0.80:
        logger.error("Warm cache coverage {:.1%} < 80% — downstream intraday_plan "
                     "fetch will fall back to network, causing multi-min stall",
                     coverage)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
