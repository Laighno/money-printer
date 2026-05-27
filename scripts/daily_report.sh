#!/bin/bash
# Daily portfolio report: collect data + evaluate + send to Feishu
# Run via launchd at 17:00 Mon-Fri.
#
# After report generation, pushes data/orders/latest.json to GitHub so ECS
# can git pull at 09:25 and execute orders against the QMT broker. This
# lets Mac shut down anytime between 17:30 and the next 16:00 — ECS runs
# autonomously on the latest plan.

cd /Users/laighno/laighno/money-printer
source .venv/bin/activate

echo "$(date): Starting daily pipeline"

# Step 1: Collect external data
echo "$(date): Collecting external data..."
python -m mp.data.collector 2>&1 | tee -a data/external/collect.log

# Step 2: Generate report + send to Feishu
echo "$(date): Generating daily report..."
python scripts/daily_report.py 2>&1 | tee -a data/reports/report.log

# Step 3: Push plan to GitHub so ECS can pull + execute tomorrow 09:25.
# Only commits if latest.json actually changed (idempotent on weekends /
# market closure days when daily_report writes the same plan).
echo "$(date): Pushing latest plan to GitHub..."
PLAN_PATH="data/orders/latest.json"
if [ -f "$PLAN_PATH" ]; then
    git add "$PLAN_PATH"
    if ! git diff --cached --quiet "$PLAN_PATH"; then
        git commit -m "auto: daily plan $(date +%Y-%m-%d) (push for ECS auto-execute)" \
            --no-verify \
            2>&1 | tee -a data/orders/push.log
        git push origin HEAD 2>&1 | tee -a data/orders/push.log
    else
        echo "$(date): latest.json unchanged — skipping commit/push"
    fi
fi

echo "$(date): Pipeline complete"
