#!/bin/bash
# Daily portfolio report: collect data + evaluate + send to Feishu
# Run via launchd at 18:00 Mon-Fri

cd /Users/laighno/laighno/money-printer
source .venv/bin/activate

echo "$(date): Starting daily pipeline"

# Step 1: Collect external data
echo "$(date): Collecting external data..."
python -m mp.data.collector 2>&1 | tee -a data/external/collect.log

# Step 2: Generate report + send to Feishu
echo "$(date): Generating daily report..."
python scripts/daily_report.py 2>&1 | tee -a data/reports/report.log

echo "$(date): Pipeline complete"
