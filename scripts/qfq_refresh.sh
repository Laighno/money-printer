#!/bin/bash
# Weekly full-history qfq refresh
# Run via launchd (com.moneyprinter.qfq) Saturday 10:00.
# Re-pulls ZZ500 + holdings + indices to repair qfq adjustment artefacts
# from the week's corporate actions (splits, dividends, rights offerings).

cd /Users/laighno/laighno/money-printer
source .venv/bin/activate

echo "$(date): Starting qfq refresh"
python scripts/qfq_refresh.py --feishu 2>&1 | tee -a data/logs/qfq_refresh.log
echo "$(date): qfq refresh complete (exit=$?)"
