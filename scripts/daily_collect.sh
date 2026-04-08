#!/bin/bash
# Daily external data collector
# Run via cron: 0 18 * * 1-5 /Users/laighno/laighno/money-printer/scripts/daily_collect.sh
# Runs at 18:00 Mon-Fri (after A-share market close)

cd /Users/laighno/laighno/money-printer
source .venv/bin/activate

echo "$(date): Starting daily data collection"
python -m mp.data.collector 2>&1 | tee -a data/external/collect.log
echo "$(date): Collection complete"
