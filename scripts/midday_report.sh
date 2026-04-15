#!/bin/bash
# Midday report: realtime prices + ML reminders, sent to Feishu
# Run via launchd at 12:00 Mon-Fri

cd /Users/laighno/laighno/money-printer
source .venv/bin/activate

echo "$(date): Starting midday report"
python scripts/daily_report.py --midday 2>&1 | tee -a data/reports/report.log
echo "$(date): Midday report complete"
