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

# Step 0: Re-sync portfolio.yaml from QMT (prevents stale-yaml burns).
# If ECS / QMT is unreachable, fall through to the stale yaml so the
# pipeline still produces *some* plan — but tee the failure to the log.
echo "$(date): Syncing portfolio.yaml from QMT..."
python scripts/sync_portfolio_from_qmt.py 2>&1 | tee -a data/orders/portfolio_sync.log \
    || echo "$(date): WARNING — QMT sync failed; using existing portfolio.yaml"

# Step 1: Collect external data
echo "$(date): Collecting external data..."
python -m mp.data.collector 2>&1 | tee -a data/external/collect.log

# Step 2: Generate report + send to Feishu
# Tier 1 (round 217): --allow-prod-write authorizes the scheduled run to
# write protected prod-state files (data/orders/latest.json + report .md).
# Ad-hoc invocations omit the flag → hard-fail at write_plan_json.
echo "$(date): Generating daily report..."
python scripts/daily_report.py --allow-prod-write 2>&1 | tee -a data/reports/report.log

# Step 2b: Arm B shadow recorder (P11-5 round 101 live A/B).
# 9:30+intraday-model simulation, no real trades. Runs AFTER daily_report
# so today's EOD bars are in the DB (collector ran in step 1). Non-fatal:
# a shadow failure must never block the real-account push in step 3.
echo "$(date): Recording 9:30-intraday shadow (Arm B)..."
mkdir -p data/shadow_930
python scripts/shadow_930_intraday.py 2>&1 | tee -a data/shadow_930/shadow.log \
    || echo "$(date): WARNING — shadow recorder failed (non-fatal; real path unaffected)"

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
