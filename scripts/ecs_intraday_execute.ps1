# ECS intraday execute pipeline for QMT live trading (P11-5 Phase B).
#
# Schedule: Windows Task Scheduler at 14:30:30 Mon-Fri (5s buffer for
# Mac to push intraday_latest.json + ECS to git-pull it).
#
# Flow:
#   1. cd C:\money-printer && git pull origin <branch>   (5s buffer)
#   2. Verify intraday_latest.json present + entry_path = intraday_14_30
#   3. Verify plan freshness (mtime > today 14:25:00 — caught stale plan)
#   4. Verify XtMiniQmt.exe running
#   5. Verify portfolio.yaml account = 8886933837
#   6. Run scripts/intraday_preflight.py — abort on > 5% drift
#   7. Run scripts/execute_orders.py --mode auto --plan intraday_latest.json
#   8. On success: write data\orders\intraday_success_<YYYYMMDD>.flag
#   9. Log everything to data\orders\ecs_intraday.log
#
# Phase C consumer: scripts\ecs_auto_execute.ps1 (9:25 Mon-Fri) checks
# for intraday_success_<yesterday>.flag at start; if present → skip
# 9:30 execute. That's the cutover handover.
#
# Pre-conditions (MANUAL setup, one-time):
#   - XtMiniQmt logged in with account 8886933837 (same as 9:30 path)
#   - ECS git on the same branch as Mac (collab/advisor-dialog currently)
#   - Task Scheduler task MoneyPrinter-IntradayExecute registered to
#     fire at 14:30:30 Mon-Fri (see scripts/ecs_setup_schedule.ps1)
#
# Safety guards (any → ABORT, no fill flag written → 9:30 next day takes over):
#   - git pull fails
#   - intraday_latest.json missing or entry_path != intraday_14_30
#   - plan mtime older than today 14:25:00 (Mac push didn't land)
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

Log "==================== ECS intraday-execute start ===================="

# Ensure log dir exists
$LogDir = Split-Path $LogPath
if (-not (Test-Path $LogDir)) { New-Item -Type Directory -Path $LogDir -Force | Out-Null }

# Step 1: git pull (gives Mac a 5s window to push the freshly-generated plan)
Set-Location $REPO
Log "Step 1: git pull origin $BRANCH (5s Mac push buffer)"
$pullOutput = & git pull origin $BRANCH 2>&1 | Out-String
$pullOutput.Trim().Split("`n") | ForEach-Object { Log "  git: $_" }
if ($LASTEXITCODE -ne 0) { Abort "git pull failed (exit $LASTEXITCODE)" }
$head = (& git rev-parse --short HEAD).Trim()
Log "Step 1: HEAD = $head"

# Step 2: Verify intraday plan present + entry_path tag
$planPath = "$REPO\data\orders\intraday_latest.json"
if (-not (Test-Path $planPath)) {
    Abort "intraday plan missing: $planPath (Mac scripts/intraday_plan.py didn't push?)"
}
$planContent = Get-Content $planPath -Raw -Encoding UTF8
if ($planContent -notmatch '"entry_path"\s*:\s*"intraday_14_30"') {
    Abort "intraday_latest.json entry_path != 'intraday_14_30' — wrong file? aborting to avoid stomping 9:30 plan"
}
Log "Step 2: plan present + entry_path = intraday_14_30 verified"

# Step 3: Verify plan freshness (mtime > today 14:25:00)
$today1425 = (Get-Date).Date.AddHours(14).AddMinutes(25)
$planMtime = (Get-Item $planPath).LastWriteTime
Log "Step 3: plan mtime = $($planMtime.ToString('yyyy-MM-dd HH:mm:ss')); threshold = $($today1425.ToString('yyyy-MM-dd HH:mm:ss'))"
if ($planMtime -lt $today1425) {
    Abort "plan mtime $($planMtime.ToString('HH:mm:ss')) older than today 14:25:00 — stale (Mac push didn't land?)"
}

# Step 4: Verify XtMiniQmt is running
Log "Step 4: verify XtMiniQmt running"
$qmt = Get-Process -Name "XtMiniQmt" -ErrorAction SilentlyContinue
if (-not $qmt) {
    Abort "XtMiniQmt.exe not running -- start it manually on ECS + login + retry"
}
Log "  XtMiniQmt pid $($qmt.Id) running"

# Step 5: Verify portfolio.yaml account
Log "Step 5: verify portfolio.yaml account = $EXPECTED_ACCOUNT"
$portfolioContent = Get-Content "$REPO\config\portfolio.yaml" -Raw
if ($portfolioContent -notmatch "$EXPECTED_ACCOUNT") {
    Log "  portfolio.yaml does not reference $EXPECTED_ACCOUNT -- non-critical, proceeding"
}

# Step 6: QMT positions reconcile (> 5% drift => abort, per round 97 decision 1)
Log "Step 6: QMT positions reconcile (5% tolerance)"
$pythonExe = "$REPO\.venv\Scripts\python.exe"
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
Log "Step 6: preflight exit = $preflightExit"
if ($preflightExit -ne 0) {
    Abort "preflight reconcile failed (exit $preflightExit) — 9:30 next day will take over"
}

# Step 7: Execute orders
Log "Step 7: execute_orders.py --plan intraday_latest.json --mode auto"
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
Log "Step 7: execute_orders exit = $execExit"

if ($execExit -ne 0) {
    Abort "execute_orders failed (exit $execExit) — 9:30 next day will take over"
}

# Step 8: Write success flag (consumed by Phase C 9:25 path next morning)
$today = (Get-Date).ToString("yyyyMMdd")
$flagPath = "$REPO\data\orders\intraday_success_$today.flag"
$flagBody = "intraday execute succeeded at $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))`r`nHEAD: $head`r`n"
Set-Content -Path $flagPath -Value $flagBody -Encoding UTF8
Log "Step 8: wrote success flag $flagPath"

Log "==================== ECS intraday-execute DONE ===================="
exit 0
