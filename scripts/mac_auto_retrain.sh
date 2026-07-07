#!/bin/bash
# Mac-side weekly auto-retrain orchestrator (advisor; user 拍 ①B Mac-compute hybrid).
#
# WHY Mac not ECS (2026-06-24): the n2c factor cache is ~1GB on disk → multi-GB
# in RAM to rebuild + train. ECS (8GB, ~3.2GB free) OOMs on it — the same wall
# that forced scripts/mac_fallback_plan.sh for the lighter daily scoring. Only
# Mac (48GB) can run refresh+train. The original root cause ("Mac slept → Friday
# cron never fired") is fixed by `pmset repeat wake` (see docs/cron_setup.md),
# NOT by relocating compute.
#
# Division of labour:
#   Mac (this script): refresh cache → train(cutoff) → verify gate  [heavy compute]
#   ECS              : hold prod model + swap into prod + always-on dead-man-switch
#
# Flow:
#   1. git pull (Mac has github auth).
#   2. md5-verify Mac prod blend == ECS prod blend — the verify gate compares the
#      new model against Mac's prod, but the swap REPLACES ECS's prod; they must
#      be the same baseline (same discipline as mac_fallback_plan.sh).
#   3. auto_retrain.py (NO --auto-swap / NO --allow-prod-write): refresh+train+
#      verify, stage retrain_pending_swap.json on gate PASS. Never touches prod.
#   4. scp data/auto_retrain_last.json → ECS ALWAYS (so MP-RetrainDeadman sees the
#      attempt's freshness + verdict, pass or fail).
#   5. On gate PASS: scp the candidate model → ECS, then
#        - ECS .first_swap_done ABSENT (first swap, ②B): scp pending marker + alert
#          the user to approve once manually on ECS. Do NOT swap.
#        - ECS .first_swap_done PRESENT: SSH-invoke ECS swap_model.py to swap prod
#          automatically (gate already passed).
#
# Usage:  bash scripts/mac_auto_retrain.sh            # full run (cron entry)
#         bash scripts/mac_auto_retrain.sh --no-swap  # compute+stage only, never swap ECS
#
# Exit non-zero on compute/gate failure so the cron log + ECS dead-man-switch
# both surface it.
set -uo pipefail

ECS="Administrator@14.103.49.51"
ECS_REPO="C:/money-printer"
REPO="/Users/laighno/laighno/money-printer"
NO_SWAP="${1:-}"
TODAY=$(date +%Y%m%d)
LOG="data/logs/mac_auto_retrain_${TODAY}.log"

cd "$REPO"
mkdir -p data/logs
echo "================================================================"
echo " Mac auto-retrain — $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# ── Step 1: latest code ──
echo "[1] git pull (Mac) ..."
git pull origin main 2>&1 | tail -2 || echo "  WARN: git pull failed, proceeding on local code"

# ── Step 2: Mac prod blend == ECS prod blend (swap baseline must match) ──
# 2026-07-07 fix: the ECS md5 fetch is over flaky SSH. The old code read a
# single empty/dropped result as "mismatch" and ABORTED the whole retrain —
# 6/26 + 7/3 both died here (one md5 came back empty) → model frozen at 6/24.
# Now: retry 3x; and if ECS is genuinely unreachable, DON'T abort — still
# train+verify (compute isn't wasted) and only DEFER the swap (which needs ECS
# anyway). Only a real reachable-but-different md5 aborts.
echo "[2] verify Mac prod blend == ECS prod blend ..."
mac_p=$(md5 -q data/blend_primary.lgb 2>/dev/null || echo MAC_MISSING)
mac_e=$(md5 -q data/blend_extreme.lgb 2>/dev/null || echo MAC_MISSING)
ecs_md5() {  # $1 = data-relative filename; echoes lowercase md5 or empty
  local f="$1" h="" i
  for i in 1 2 3; do
    h=$(ssh -o ConnectTimeout=45 -o ServerAliveInterval=10 "$ECS" "powershell -Command \"(Get-FileHash $ECS_REPO/data/$f -Algorithm MD5).Hash\"" 2>/dev/null | tr -d '\r' | tr 'A-Z' 'a-z' | grep -oE '^[0-9a-f]{32}')
    [ -n "$h" ] && break
    sleep 5
  done
  echo "$h"
}
ecs_p=$(ecs_md5 blend_primary.lgb)
ecs_e=$(ecs_md5 blend_extreme.lgb)
SKIP_SWAP=0
if [ -z "$ecs_p" ] || [ -z "$ecs_e" ]; then
  echo "  ⚠ ECS unreachable (md5 empty after 3 retries) — train+verify anyway, DEFER swap."
  SKIP_SWAP=1
elif [ "$mac_p" != "$ecs_p" ] || [ "$mac_e" != "$ecs_e" ]; then
  echo "  ⚠ PROD MISMATCH Mac vs ECS (reachable but different) — gate baseline != swap target."
  echo "    primary: mac=$mac_p ecs=$ecs_p"
  echo "    extreme: mac=$mac_e ecs=$ecs_e"
  echo "    Sync prod blend Mac<->ECS before trusting the swap. ABORT."
  exit 2
else
  echo "  ✓ prod blend matches (primary=$mac_p extreme=$mac_e)"
fi

# ── Step 3: compute + gate (never touches prod) ──
echo "[3] auto_retrain.py (refresh → train(cutoff) → verify gate; stage only) ..."
.venv/bin/python -u scripts/auto_retrain.py 2>&1 | tee "data/logs/auto_retrain_run_${TODAY}.log"
retrain_rc=${PIPESTATUS[0]}
echo "  auto_retrain exit = $retrain_rc"

# ── Step 4: push freshness/verdict to ECS ALWAYS (dead-man-switch input) ──
echo "[4] scp auto_retrain_last.json → ECS (watchdog freshness) ..."
scp -o ConnectTimeout=45 data/auto_retrain_last.json "$ECS:$ECS_REPO/data/auto_retrain_last.json" 2>/dev/null \
  && echo "  ✓ pushed" || echo "  WARN: scp freshness failed (deadman may RED next morning)"

# read verdict from summary
verify_pass=$(.venv/bin/python -c "import json;print(json.load(open('data/auto_retrain_last.json')).get('verify_pass'))" 2>/dev/null || echo "None")
new_prefix=$(.venv/bin/python -c "import json;print(json.load(open('data/auto_retrain_last.json')).get('new_prefix') or '')" 2>/dev/null || echo "")
cutoff=$(.venv/bin/python -c "import json;print(json.load(open('data/auto_retrain_last.json')).get('cutoff') or '')" 2>/dev/null || echo "")

if [ "$verify_pass" != "True" ]; then
  echo "  gate NOT passed (verify_pass=$verify_pass, retrain_rc=$retrain_rc) — prod unchanged."
  echo "  ECS dead-man-switch will surface staleness/failure. Done."
  exit "${retrain_rc:-1}"
fi

# ── Step 5: gate PASS → deliver candidate to ECS + swap (or stage first-swap) ──
echo "[5] gate PASS: new=$new_prefix cutoff=$cutoff"

# ECS was unreachable in Step 2 → can't scp/swap now. Candidate is trained +
# gate-passed + staged on Mac; the NEXT reachable run swaps the latest. Not an
# error — prod just stays put this cycle (dead-man-switch tracks staleness).
if [ "$SKIP_SWAP" = "1" ]; then
  echo "  ECS unreachable this run — candidate $new_prefix staged on Mac, swap DEFERRED."
  echo "  (next reachable run, or manual: scp candidate + swap_model.py on ECS)"
  exit 0
fi

echo "  scp candidate model → ECS ..."
scp -o ConnectTimeout=60 "${new_prefix}_primary.lgb" "$ECS:$ECS_REPO/${new_prefix}_primary.lgb"
scp -o ConnectTimeout=60 "${new_prefix}_extreme.lgb" "$ECS:$ECS_REPO/${new_prefix}_extreme.lgb"
echo "  ✓ model uploaded"

first_done=$(ssh -o ConnectTimeout=45 "$ECS" "powershell -Command \"Test-Path $ECS_REPO/data/.first_swap_done\"" 2>/dev/null | tr -d '\r' | tail -1)

if [ "$NO_SWAP" = "--no-swap" ]; then
  echo "  --no-swap: candidate on ECS, NOT swapping."
  exit 0
fi

if [ "$first_done" = "True" ]; then
  echo "  ECS first-swap sentinel present → AUTO-SWAP on ECS ..."
  ssh -o ConnectTimeout=60 "$ECS" "powershell -Command \"cd C:\\money-printer; \$env:MP_ALLOW_PROD_WRITE='1'; .venv\\Scripts\\python.exe -X utf8 scripts\\swap_model.py --new-prefix $new_prefix --prod-prefix data/blend --allow-prod-write --reason 'mac_auto_retrain cutoff $cutoff'\"" 2>&1 | tail -8
  echo "  (verify swapped: ECS blend md5 should now match candidate)"
else
  echo "  ECS first-swap sentinel ABSENT → ②B: first swap needs MANUAL approval."
  scp -o ConnectTimeout=45 data/retrain_pending_swap.json "$ECS:$ECS_REPO/data/retrain_pending_swap.json" 2>/dev/null || true
  echo "  Approve once on ECS:"
  echo "    ssh $ECS"
  echo "    cd C:\\money-printer; \$env:MP_ALLOW_PROD_WRITE='1'; .venv\\Scripts\\python.exe scripts\\swap_model.py --new-prefix $new_prefix --allow-prod-write --reason 'first manual swap'"
fi

echo "================================================================"
echo " DONE — candidate cutoff $cutoff delivered to ECS."
echo "================================================================"
