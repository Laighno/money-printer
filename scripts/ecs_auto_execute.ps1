# ECS auto-execute pipeline for QMT live trading (9:25 path).
#
# Schedule: Windows Task Scheduler at 09:25 Mon-Fri (5 min before 9:30 open).
#
# P11-5 round 103: this path is now a diff-RECONCILE against the last
# 14:30 intraday target, NOT an independent EOD-blend execution. It fills
# only the residual the 14:30 path couldn't (e.g. 涨停/跌停 names whose
# limit orders queued but never filled). Normal days → residual empty →
# no-op. The old success-flag gate is gone (发单≠成交 made it skip
# unfilled residuals — silent orphan bug the user caught).
#
# Flow:
#   1. cd C:\money-printer && git pull origin <branch>
#   2. Verify XtMiniQmt.exe is running (else abort -- don't trade without it)
#   3. Verify config/portfolio.yaml is the QMT account (8886933837)
#   4. reconcile_plan.py: live QMT vs 14:30 target → residual plan
#        exit 0  → execute data/orders/reconcile_latest.json
#        exit 10 → deep-fallback: execute EOD blend data/orders/latest.json
#                  (14:30 infra down ≥2 trading days)
#        exit 2/3 → abort (QMT/other failure)
#   5. Run scripts/execute_orders.py --mode auto against the chosen plan
#   6. Log everything to C:\money-printer\data\orders\ecs_auto.log
#
# Pre-conditions (must be set up MANUALLY before this task runs):
#   - XtMiniQmt.exe started and logged in with account 8886933837
#     (recommended: enable QMT auto-login on Windows boot, or RDP each morning)
#   - ECS git is on the same branch as Mac (main, since round 193 merged collab/advisor-dialog)
#
# Safety guards:
#   - If git pull fails -> abort (would reconcile against stale scripts)
#   - If XtMiniQmt not running -> abort
#   - If portfolio.yaml account mismatch -> warn (non-critical)
#   - reconcile_plan handles target staleness via trading-day count
#   - Deep-fallback plan older than 90h -> abort (likely stale)

$ErrorActionPreference = "Continue"
$LogPath = "C:\money-printer\data\orders\ecs_auto.log"
$REPO = "C:\money-printer"
$EXPECTED_ACCOUNT = "8886933837"
$USERDATA = "C:\guojin\userdata_mini"
$BRANCH = "main"

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

# Step 1: git pull (latest reconcile_plan.py + scripts + plans)
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
    Abort "XtMiniQmt.exe not running -- start it manually on ECS + login + retry"
}
Log "  XtMiniQmt pid $($qmt.Id) running"

# Step 3: Verify portfolio.yaml account
Log "Step 3: verify portfolio.yaml account = $EXPECTED_ACCOUNT"
$portfolioContent = Get-Content "$REPO\config\portfolio.yaml" -Raw
if ($portfolioContent -notmatch "$EXPECTED_ACCOUNT") {
    Log "  portfolio.yaml does not reference $EXPECTED_ACCOUNT -- non-critical, proceeding"
}

$pythonExe = "$REPO\.venv\Scripts\python.exe"

# Step 4: diff-reconcile against the last 14:30 intraday target (round 103).
# REPLACES the old flag-gate. reconcile_plan.py compares the live QMT
# portfolio to the 14:30 intended target and emits ONLY the residual:
#   - normal day  → 14:30 fully filled → residual empty → executor no-ops
#   - 涨停/跌停 day → 14:30 didn't fill → residual fills the gap at open
# Exit codes: 0 = reconcile_latest.json written (execute it);
#             10 = target missing/stale (14:30 infra down ≥2 trading days)
#                  → deep-fallback to EOD blend latest.json;
#             2/3 = QMT/other failure → abort.
Log "Step 4: reconcile_plan.py (live vs 14:30 target diff)"
$reconArgs = @(
    "-X", "utf8",
    "scripts\reconcile_plan.py",
    "--target-plan", "data\orders\intraday_latest.json",
    "--out", "data\orders\reconcile_latest.json",
    "--qmt-account", $EXPECTED_ACCOUNT,
    "--qmt-userdata", $USERDATA
)
$reconOutput = & $pythonExe @reconArgs 2>&1 | Out-String
$reconOutput.Trim().Split("`n") | ForEach-Object { Log "  reconcile: $_" }
$reconExit = $LASTEXITCODE
Log "Step 4: reconcile exit = $reconExit"

if ($reconExit -eq 0) {
    $planPath = "data\orders\reconcile_latest.json"
    Log "Step 4: using reconcile plan (residual补 of 14:30 target)"
} elseif ($reconExit -eq 10 -or $reconExit -eq 11) {
    # Deep fallback:
    #   exit 10 = 14:30 target missing or stale ≥2 trading days
    #   exit 11 = 14:30 target source not prod-authoritative (round 213 Tier 0)
    # In both cases execute the EOD blend plan (daily_report 17:00 still
    # regenerates it). Exit 11 specifically defends against ad-hoc replay
    # overwriting intraday_latest.json — the 6/4 9:25 incident root cause.
    $planPath = "data\orders\latest.json"
    $fbReason = if ($reconExit -eq 10) { "target missing/stale" }
                else { "target source non-authoritative (replay/ad-hoc)" }
    Log "Step 4: reconcile signalled deep-fallback ($fbReason) -> EOD blend latest.json"
    if (-not (Test-Path "$REPO\$planPath")) { Abort "fallback plan missing: $planPath" }
    $planAge = ((Get-Date) - (Get-Item "$REPO\$planPath").LastWriteTime).TotalHours
    Log "Step 4: fallback plan age = $([math]::Round($planAge,1))h"
    if ($planAge -gt 90) { Abort "fallback plan older than 90h -- stale, aborting" }
} else {
    Abort "reconcile_plan failed (exit $reconExit) -- QMT/other error, not trading blind"
}

# Step 5: Execute the chosen plan
Log "Step 5: execute_orders.py --mode auto --plan $planPath"
$args = @(
    "-X", "utf8",
    "scripts\execute_orders.py",
    "--mode", "auto",
    "--plan", $planPath,
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
