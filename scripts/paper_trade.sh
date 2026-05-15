#!/bin/bash
# Paper trading daily run: rebalance simulated portfolio + send Feishu report.
# Run via launchd (com.moneyprinter.papertrade) at 16:30 Mon-Fri.
# launchd handles missed runs (catch-up) when Mac was asleep — unlike cron.

cd /Users/laighno/laighno/money-printer
source .venv/bin/activate

echo "$(date): Starting paper_trade run"
python scripts/paper_trade.py 2>&1 | tee -a data/logs/paper_trade.log
echo "$(date): paper_trade complete (exit=$?)"
