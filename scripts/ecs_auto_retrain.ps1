# ECS weekly auto-retrain (replaces the dead Mac Friday cron).
#
# Schedule: Windows Task Scheduler "MP-AutoRetrain", Friday 18:30 (after the
# 17:xx daily_report so bars/cache are fresh; clear of the 9:25 trading path).
#
# WHY ECS not Mac: the old retrain (Mac cron `0 18 * * 5 walk_forward_backtest.py`)
# silently died ~4/24 when the laptop slept; prod blend froze at 6/2. ECS is
# always-on. This wrapper runs the gated pipeline scripts/auto_retrain.py:
#   refresh_n2c_cache -> train_blend_cutoff(current cutoff) -> verify gate
#   -> swap (only on gate PASS, and only after the first manual swap; --auto-swap).
#
# Governance (user 2B): the FIRST swap is staged as data/retrain_pending_swap.json
# and NOT applied until a human runs swap_model.py once. After that, the sentinel
# data/.first_swap_done exists and weekly gate-PASS swaps auto-apply here.
#
# ASCII-only Log strings (GBK parse safety, round 235/253 lesson: no em-dash /
# CJK inside double-quoted strings the parser scans).

$ErrorActionPreference = "Continue"
$REPO = "C:\money-printer"
$BRANCH = "main"
$LogPath = "C:\money-printer\data\logs\auto_retrain.log"

function Log {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
}

$LogDir = Split-Path $LogPath
if (-not (Test-Path $LogDir)) { New-Item -Type Directory -Path $LogDir -Force | Out-Null }

Log "==================== ECS auto-retrain start ===================="

# Allow the gated prod-model swap (swap_model.py writes data/blend_*.lgb, now
# in PROTECTED_PROD_PATHS). This scheduled task is the authoritative path.
$env:MP_ALLOW_PROD_WRITE = "1"

Set-Location $REPO
Log "Step 1: git pull origin $BRANCH"
$pullOutput = & git pull origin $BRANCH 2>&1 | Out-String
$pullOutput.Trim().Split("`n") | ForEach-Object { Log "  git: $_" }
if ($LASTEXITCODE -ne 0) { Log "ABORT: git pull failed (exit $LASTEXITCODE)"; exit 1 }

$pythonExe = "$REPO\.venv\Scripts\python.exe"

# Step 2: run the orchestrator. --auto-swap honours the first-swap sentinel:
# before the human approves once, gate PASS only stages a pending marker.
Log "Step 2: auto_retrain.py --auto-swap --allow-prod-write"
$out = & $pythonExe -X utf8 scripts\auto_retrain.py --auto-swap --allow-prod-write 2>&1 | Out-String
$out.Trim().Split("`n") | ForEach-Object { Log "  retrain: $_" }
$rc = $LASTEXITCODE
Log "Step 2: auto_retrain exit = $rc"

# Exit-code map (from auto_retrain.py): 0 = OK (swapped, or staged pending, or
# nothing to do); 3/4/5 = refresh/cutoff/train failed; 6 = gate FAIL (prod kept);
# 7 = swap failed (prod rolled back). Non-zero -> the Saturday dead-man-switch
# (retrain_freshness_check.py) will also surface it; we just record here.
if ($rc -eq 0) {
    Log "==================== ECS auto-retrain DONE ===================="
    exit 0
} else {
    Log "ECS auto-retrain finished NON-ZERO (exit $rc) -- prod model unchanged unless swap logged above"
    exit $rc
}
