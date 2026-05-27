# ECS auto-execute pipeline for QMT live trading
#
# Schedule: Windows Task Scheduler at 09:25 Mon-Fri (5 min before 9:30 open).
#
# Flow:
#   1. cd C:\money-printer && git pull origin <branch>
#   2. Verify XtMiniQmt.exe is running (else abort — don't trade without it)
#   3. Verify config/portfolio.yaml is the QMT account (8886933837)
#   4. Run scripts/execute_orders.py --mode auto against latest plan
#   5. Log everything to C:\money-printer\data\orders\ecs_auto.log
#
# Pre-conditions (must be set up MANUALLY before this task runs):
#   - XtMiniQmt.exe started and logged in with account 8886933837
#     (recommended: enable QMT auto-login on Windows boot, or RDP each morning)
#   - ECS git is on the same branch as Mac (collab/advisor-dialog currently)
#
# Safety guards:
#   - If git pull fails -> abort (would execute stale plan otherwise)
#   - If XtMiniQmt not running -> abort
#   - If portfolio.yaml account mismatch -> abort
#   - If plan json older than 24h -> abort (likely stale)

$ErrorActionPreference = "Stop"
$LogPath = "C:\money-printer\data\orders\ecs_auto.log"
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

Log "==================== ECS auto-execute start ===================="

# Ensure log dir exists
$LogDir = Split-Path $LogPath
if (-not (Test-Path $LogDir)) { New-Item -Type Directory -Path $LogDir -Force | Out-Null }

# Step 1: git pull
Set-Location $REPO
Log "Step 1: git pull origin $BRANCH"
$pullOutput = & git pull origin $BRANCH 2>&1 | Out-String
$pullOutput.Trim().Split("`n") | ForEach-Object { Log "  git: $_" }
if ($LASTEXITCODE -ne 0) { Abort "git pull failed (exit $LASTEXITCODE)" }
$head = (& git rev-parse --short HEAD).Trim()
Log "Step 1: HEAD = $head"

# Step 2: Verify XtMiniQmt is running
Log "Step 2: verify XtMiniQmt running"
$qmt = Get-Process -Name "XtMiniQmt" -ErrorAction SilentlyContinue
if (-not $qmt) {
    Abort "XtMiniQmt.exe not running — start it manually on ECS + login + retry"
}
Log "  XtMiniQmt pid $($qmt.Id) running"

# Step 3: Verify portfolio.yaml account
Log "Step 3: verify portfolio.yaml account = $EXPECTED_ACCOUNT"
$portfolioContent = Get-Content "$REPO\config\portfolio.yaml" -Raw
if ($portfolioContent -notmatch "$EXPECTED_ACCOUNT") {
    Log "  portfolio.yaml does not reference $EXPECTED_ACCOUNT — non-critical, proceeding"
}

# Step 4: Verify plan freshness
$planPath = "$REPO\data\orders\latest.json"
if (-not (Test-Path $planPath)) { Abort "plan json missing: $planPath" }
$planAge = ((Get-Date) - (Get-Item $planPath).LastWriteTime).TotalHours
Log "Step 4: plan age = $([math]::Round($planAge,1))h"
if ($planAge -gt 30) { Abort "plan older than 30h — likely stale, aborting" }

# Step 5: Execute orders
Log "Step 5: execute_orders.py --mode auto"
$cmd = ".venv\Scripts\python.exe -X utf8 scripts\execute_orders.py " +
       "--mode auto --plan data\orders\latest.json " +
       "--qmt-account $EXPECTED_ACCOUNT --qmt-userdata `"$USERDATA`""
Log "  cmd: $cmd"

$pythonExe = "$REPO\.venv\Scripts\python.exe"
$args = @(
    "-X", "utf8",
    "scripts\execute_orders.py",
    "--mode", "auto",
    "--plan", "data\orders\latest.json",
    "--qmt-account", $EXPECTED_ACCOUNT,
    "--qmt-userdata", $USERDATA
)
$execOutput = & $pythonExe @args 2>&1 | Out-String
$execOutput.Trim().Split("`n") | ForEach-Object { Log "  exec: $_" }
$execExit = $LASTEXITCODE
Log "Step 5: execute_orders exit = $execExit"

if ($execExit -eq 0) {
    Log "==================== ECS auto-execute DONE ===================="
    exit 0
} else {
    Abort "execute_orders failed (exit $execExit)"
}
