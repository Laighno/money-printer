#!/bin/bash
# Mac fallback: generate EOD top-25 plan locally + upload to ECS.
#
# WHY (2026-06-08): ECS (8GB RAM) daily_report.py ProcessPool scoring can OOM /
# get killed when bars DB is stale (heavy API fetch) or ECS is resource-stressed.
# 6/8 the 17:00 task + 3 manual retries all died before writing the plan. Mac has
# 48GB RAM + identical prod models (md5-verified) + same network data sources
# (Sina/akshare, NOT QMT — QMT is only for order execution). So Mac can generate
# the SAME plan reliably and push it to ECS for 9:25 auto-execute.
#
# Degradation trigger: run this when ECS 17:00 daily_report fails to write a fresh
# data/orders/latest.json (check: latest.json mtime < today 17:00, or task result
# != 0). Produces an identical plan because:
#   - models: blend_primary/extreme.lgb md5 == ECS (verify in step 1)
#   - code:   n_recommend=22 (top-25) from git
#   - data:   live Sina/akshare fetch (same as ECS)
#   - holds:  ECS's QMT-synced portfolio.yaml (scp'd in step 2)
#
# Usage:  bash scripts/mac_fallback_plan.sh
#         bash scripts/mac_fallback_plan.sh --dry-run   # generate on Mac, do NOT upload
#
# Rule #4.1: writes data/orders/latest.json on Mac (gated --allow-prod-write),
# then scp's to ECS prod path. This is the deliberate fallback path.
set -euo pipefail

ECS="Administrator@14.103.49.51"
ECS_REPO="C:/money-printer"
REPO="/Users/laighno/laighno/money-printer"
DRY_RUN="${1:-}"
TODAY=$(date +%Y%m%d)

cd "$REPO"
echo "================================================================"
echo " Mac fallback EOD plan — $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# ── Step 0: latest code (n_recommend=22 etc.) ──
echo "[0] git pull (Mac) ..."
git pull origin main 2>&1 | tail -2

# ── Step 1: verify Mac models == ECS models (plan would diverge otherwise) ──
echo "[1] verify Mac prod models == ECS ..."
mac_p=$(md5 -q data/blend_primary.lgb)
mac_e=$(md5 -q data/blend_extreme.lgb)
ecs_p=$(ssh -o ConnectTimeout=45 "$ECS" "powershell -Command \"(Get-FileHash $ECS_REPO/data/blend_primary.lgb -Algorithm MD5).Hash\"" 2>/dev/null | tr -d '\r' | tr 'A-Z' 'a-z' | grep -oE '^[0-9a-f]{32}')
ecs_e=$(ssh -o ConnectTimeout=45 "$ECS" "powershell -Command \"(Get-FileHash $ECS_REPO/data/blend_extreme.lgb -Algorithm MD5).Hash\"" 2>/dev/null | tr -d '\r' | tr 'A-Z' 'a-z' | grep -oE '^[0-9a-f]{32}')
if [ "$mac_p" != "$ecs_p" ] || [ "$mac_e" != "$ecs_e" ]; then
  echo "  ✗ MODEL MISMATCH — Mac plan would diverge from ECS. ABORT."
  echo "    primary: mac=$mac_p ecs=$ecs_p"
  echo "    extreme: mac=$mac_e ecs=$ecs_e"
  echo "    Fix: sync prod models Mac<->ECS first."
  exit 2
fi
echo "  ✓ models match (primary=$mac_p extreme=$mac_e)"

# ── Step 2: pull ECS's QMT-synced portfolio.yaml (current holdings) ──
# ECS Step 2 (sync_portfolio_from_qmt --local) runs even when daily_report.py
# later dies, so portfolio.yaml on ECS reflects live QMT. Mac needs it for
# accurate position sizing.
echo "[2] scp ECS portfolio.yaml → Mac (current holdings) ..."
cp config/portfolio.yaml "config/portfolio.yaml.mac_bak_$TODAY" 2>/dev/null || true
scp -o ConnectTimeout=45 "$ECS:$ECS_REPO/config/portfolio.yaml" config/portfolio.yaml
n_holds=$(grep -cE "code:" config/portfolio.yaml || echo 0)
echo "  ✓ portfolio.yaml from ECS ($n_holds holdings)"

# ── Step 3: generate plan on Mac (48GB, no OOM; live Sina/akshare fetch) ──
echo "[3] Mac daily_report.py --allow-prod-write (generating top-25 plan) ..."
MP_ALLOW_PROD_WRITE=1 .venv/bin/python -u scripts/daily_report.py --allow-prod-write \
  > "data/logs/mac_fallback_${TODAY}.log" 2>&1
n_orders=$(.venv/bin/python -c "import json; print(len(json.load(open('data/orders/latest.json')).get('orders',[])))" 2>/dev/null || echo "?")
gen_at=$(.venv/bin/python -c "import json; print(json.load(open('data/orders/latest.json')).get('source',{}).get('generated_at','?'))" 2>/dev/null || echo "?")
echo "  ✓ Mac plan generated: $n_orders orders, generated_at=$gen_at"
echo "    (log: data/logs/mac_fallback_${TODAY}.log)"

if [ "$DRY_RUN" = "--dry-run" ]; then
  echo "[--dry-run] generated on Mac, NOT uploading. Inspect data/orders/latest.json."
  exit 0
fi

# ── Step 4: upload plan to ECS prod path ──
echo "[4] scp Mac plan → ECS ..."
scp -o ConnectTimeout=45 data/orders/latest.json "$ECS:$ECS_REPO/data/orders/latest.json"
scp -o ConnectTimeout=45 "data/orders/orders_${TODAY}.json" "$ECS:$ECS_REPO/data/orders/orders_${TODAY}.json" 2>/dev/null || true
scp -o ConnectTimeout=45 "data/reports/daily_${TODAY}.md" "$ECS:$ECS_REPO/data/reports/daily_${TODAY}.md" 2>/dev/null || true
echo "  ✓ uploaded"

# ── Step 5: verify ECS latest.json updated ──
echo "[5] verify ECS latest.json ..."
ssh -o ConnectTimeout=45 "$ECS" "powershell -Command \"\$d = Get-Content $ECS_REPO/data/orders/latest.json -Raw | ConvertFrom-Json; Write-Host ('  ECS latest.json: orders=' + \$d.orders.Count + ' generated=' + \$d.source.generated_at + ' is_prod=' + \$d.source.is_prod)\"" 2>&1 | tail -2

echo "================================================================"
echo " DONE — ECS latest.json = Mac-generated top-25 plan."
echo " 9:25 ecs_auto_execute will run it (git pull is no-op; latest.json"
echo " is uncommitted local, preserved by fast-forward)."
echo "================================================================"
