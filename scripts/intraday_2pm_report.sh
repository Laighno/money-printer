#!/bin/bash
# 14:00 intraday report: same flow as midday, with session_scale ≈ 1.33×.
# Run via launchd at 14:00 Mon-Fri.  Captures the post-lunch leg of the
# session before the 16:00 EOD report locks in.

cd /Users/laighno/laighno/money-printer
source .venv/bin/activate

echo "$(date): Starting 14:00 intraday report"
python scripts/daily_report.py --intraday-2pm 2>&1 | tee -a data/reports/report.log
echo "$(date): 14:00 intraday report complete"
