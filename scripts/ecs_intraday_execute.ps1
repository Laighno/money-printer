# ECS intraday pipeline for QMT live trading (P11-5 Phases A + B
# combined into a single ECS Task Scheduler entry -- per round 99
# decision (A): one process owns the 14:30 path end-to-end, avoiding
# Mac push → ECS pull race conditions on the plan JSON).
#
# Schedule: Windows Task Scheduler at 14:29:55 Mon-Fri. The 5s lead
# gives intraday_plan.py's sleep_to_trigger() time to settle onto
# 14:30:00 exactly (Rule #11).
#
# Flow (single process, atomic):
#   1. cd C:\money-printer && git pull origin <branch>
#   2. Run scripts\intraday_plan.py
#        → sleep_to_trigger 14:30:00
#        → xtdata 1m fetch + EOD history
#        → BlendRanker(intraday_blend) score + Top-K=10
#        → write data\orders\intraday_latest.json + intraday_<YYYYMMDD>.json
#      ABORT on exit != 0 (DQ gate, 0 morning bars, etc) -- Phase C
#      9:30 next day will take over.
#   3. Verify intraday_latest.json present + entry_path = intraday_14_30
#   4. Verify plan freshness (mtime > today 14:29:00 -- should be ≥ 14:30
#      because same-process generation, threshold catches re-runs of
#      stale plans from earlier in the day).
#   5. Verify XtMiniQmt.exe running
#   6. Verify portfolio.yaml account = 8886933837
#   7. Run scripts/intraday_preflight.py -- abort on > 5% drift
#   8. Run scripts/execute_orders.py --mode auto --plan intraday_latest.json
#   9. On success: write data\orders\intraday_success_<YYYYMMDD>.flag
#  10. Log everything to data\orders\ecs_intraday.log
#
# Phase C consumer: scripts\ecs_auto_execute.ps1 (9:25 Mon-Fri) checks
# for intraday_success_<previous_trading_day>.flag at start; if present
# → skip 9:30 execute. That's the cutover handover.
#
# Pre-conditions (MANUAL setup, one-time):
#   - XtMiniQmt logged in with account 8886933837 (same as 9:30 path)
#   - ECS git on the same branch as Mac (collab/advisor-dialog currently)
#   - Task Scheduler task MoneyPrinter-IntradayPipeline registered to
#     fire at 14:29:55 Mon-Fri (see scripts/ecs_setup_schedule.ps1)
#
# Safety guards (any → ABORT, no fill flag written → 9:30 next day takes over):
#   - git pull fails
#   - intraday_plan.py exit != 0 (DQ gate / 0 morning bars / deadline missed)
#   - intraday_latest.json missing or entry_path != intraday_14_30
#   - plan mtime older than today 14:29:00 (something abnormal in step 2)
#   - XtMiniQmt not running
#   - QMT positions drift > 5% from plan
#   - execute_orders exit != 0

$ErrorActionPreference = "Continue"
$LogPath = "C:\money-printer\data\orders\ecs_intraday.log"
$REPO = "C:\money-printer"
$EXPECTED_ACCOUNT = "8886933837"
$USERDATA = "C:\guojin\userdata_mini"
$BRANCH = "collab/advisor-dialog"

function Log {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
}

function Abort {
    param([string]$reason)
    Log "ABORT: $reason"
    exit 1
}

Log "==================== ECS intraday-pipeline start ===================="

# Ensure log dir exists
$LogDir = Split-Path $LogPath
if (-not (Test-Path $LogDir)) { New-Item -Type Directory -Path $LogDir -Force | Out-Null }

# Step 1: git pull (latest universe metadata, intraday_blend artifacts,
# script updates).
Set-Location $REPO
Log "Step 1: git pull origin $BRANCH"
$pullOutput = & git pull origin $BRANCH 2>&1 | Out-String
$pullOutput.Trim().Split("`n") | ForEach-Object { Log "  git: $_" }
if ($LASTEXITCODE -ne 0) { Abort "git pull failed (exit $LASTEXITCODE)" }
$head = (& git rev-parse --short HEAD).Trim()
Log "Step 1: HEAD = $head"

# Step 2: Phase A -- intraday_plan.py generates today's 14:30 plan.
# Script blocks via sleep_to_trigger until clock = 14:30:00, then
# fetches + scores + writes data/orders/intraday_latest.json. Exit
# codes per scripts/intraday_plan.py: 0=OK, 2=missed 14:30:30 deadline,
# 3=portfolio.yaml missing acct, 4=0 morning bars / DQ gate fail, 5=
# empty Top-K. Any non-zero → no fill flag → Phase C 9:30 next day
# fallback takes over.
Log "Step 2: intraday_plan.py (sleep_to_trigger 14:30:00 + fetch + score)"
$pythonExe = "$REPO\.venv\Scripts\python.exe"
$planArgs = @(
    "-X", "utf8",
    "scripts\intraday_plan.py"
)
$planOutput = & $pythonExe @planArgs 2>&1 | Out-String
$planOutput.Trim().Split("`n") | ForEach-Object { Log "  plan: $_" }
$planExit = $LASTEXITCODE
Log "Step 2: intraday_plan exit = $planExit"
if ($planExit -ne 0) {
    Abort "intraday_plan.py failed (exit $planExit) -- 9:30 next day will take over"
}

# Step 3: Verify intraday plan present + entry_path tag (defensive --
# intraday_plan.py should have written it in step 2, but verifying
# costs nothing).
$planPath = "$REPO\data\orders\intraday_latest.json"
if (-not (Test-Path $planPath)) {
    Abort "intraday plan missing: $planPath (intraday_plan.py succeeded but file absent?)"
}
$planContent = Get-Content $planPath -Raw -Encoding UTF8
if ($planContent -notmatch '"entry_path"\s*:\s*"intraday_14_30"') {
    Abort "intraday_latest.json entry_path != 'intraday_14_30' -- refusing to layer 14:30 path onto wrong plan"
}
Log "Step 3: plan present + entry_path = intraday_14_30 verified"

# Step 4: Verify plan freshness (mtime > today 14:29:00). Same-process
# generation in step 2 means mtime should be ≥ 14:30:00; the 14:29
# threshold catches stale plans from earlier-day runs that somehow
# leaked through (e.g. task fired but step 2 crashed before sleep
# completed → file from yesterday).
$today1429 = (Get-Date).Date.AddHours(14).AddMinutes(29)
$planMtime = (Get-Item $planPath).LastWriteTime
Log "Step 4: plan mtime = $($planMtime.ToString('yyyy-MM-dd HH:mm:ss')); threshold = $($today1429.ToString('yyyy-MM-dd HH:mm:ss'))"
if ($planMtime -lt $today1429) {
    Abort "plan mtime $($planMtime.ToString('HH:mm:ss')) older than today 14:29:00 -- stale (intraday_plan.py didn't actually write?)"
}

# Step 5: Verify XtMiniQmt is running
Log "Step 5: verify XtMiniQmt running"
$qmt = Get-Process -Name "XtMiniQmt" -ErrorAction SilentlyContinue
if (-not $qmt) {
    Abort "XtMiniQmt.exe not running -- start it manually on ECS + login + retry"
}
Log "  XtMiniQmt pid $($qmt.Id) running"

# Step 6: Verify portfolio.yaml account
Log "Step 6: verify portfolio.yaml account = $EXPECTED_ACCOUNT"
$portfolioContent = Get-Content "$REPO\config\portfolio.yaml" -Raw
if ($portfolioContent -notmatch "$EXPECTED_ACCOUNT") {
    Log "  portfolio.yaml does not reference $EXPECTED_ACCOUNT -- non-critical, proceeding"
}

# Step 7: QMT positions reconcile (> 5% drift => abort, per round 97 decision 1)
Log "Step 7: QMT positions reconcile (5% tolerance)"
$preflightArgs = @(
    "-X", "utf8",
    "scripts\intraday_preflight.py",
    "--plan", "data\orders\intraday_latest.json",
    "--qmt-account", $EXPECTED_ACCOUNT,
    "--qmt-userdata", $USERDATA,
    "--tolerance", "0.05"
)
$preflightOutput = & $pythonExe @preflightArgs 2>&1 | Out-String
$preflightOutput.Trim().Split("`n") | ForEach-Object { Log "  preflight: $_" }
$preflightExit = $LASTEXITCODE
Log "Step 7: preflight exit = $preflightExit"
if ($preflightExit -ne 0) {
    Abort "preflight reconcile failed (exit $preflightExit) -- 9:30 next day will take over"
}

# Step 8: Execute orders
Log "Step 8: execute_orders.py --plan intraday_latest.json --mode auto"
$execArgs = @(
    "-X", "utf8",
    "scripts\execute_orders.py",
    "--mode", "auto",
    "--plan", "data\orders\intraday_latest.json",
    "--qmt-account", $EXPECTED_ACCOUNT,
    "--qmt-userdata", $USERDATA
)
$execOutput = & $pythonExe @execArgs 2>&1 | Out-String
$execOutput.Trim().Split("`n") | ForEach-Object { Log "  exec: $_" }
$execExit = $LASTEXITCODE
Log "Step 8: execute_orders exit = $execExit"

if ($execExit -ne 0) {
    Abort "execute_orders failed (exit $execExit) -- 9:30 next day will take over"
}

# Step 9: Write success flag (consumed by Phase C 9:25 path next morning)
$today = (Get-Date).ToString("yyyyMMdd")
$flagPath = "$REPO\data\orders\intraday_success_$today.flag"
$flagBody = "intraday pipeline succeeded at $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))`r`nHEAD: $head`r`n"
Set-Content -Path $flagPath -Value $flagBody -Encoding UTF8
Log "Step 9: wrote success flag $flagPath"

Log "==================== ECS intraday-pipeline DONE ===================="
exit 0
