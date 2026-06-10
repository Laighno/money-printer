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

# Round 217 Tier 1: this scheduled task is the authoritative live-trading
# entry point; allow downstream prod-state writes (reconcile_latest.json,
# exec_*.json). Ad-hoc CLI / ssh invocations don't set this env, so the
# Tier 1 hard-fail in mp.common.paths.assert_prod_write_allowed protects
# every protected path globally.
$env:MP_ALLOW_PROD_WRITE = "1"

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

# Step 4: diff-reconcile against the EOD daily_report top-25 plan (round 258,
# advisor round 256/257 ask, user 拍板 6/9 'A: EOD primary').
# WAS reconciling against 14:30 OOS target (intraday_latest.json) -- but that
# decoupled 9:25 execute from the EOD plan user sees in Feishu (6/9 incident:
# executed 11 orders vs stale 6/5 OOS target while user expected top-25 85%).
# Now: reconcile latest.json (EOD top-25) vs live QMT -> residual.
#   - residual non-empty -> execute the residual (closes the gap to 85% top-25)
#   - residual empty     -> executor no-ops (already in target shape)
# Exit codes: 0  = reconcile_latest.json written (execute it);
#             10 = target missing/stale (EOD plan failed >2 trading days) -> ABORT
#                  (target IS the EOD plan, no separate fallback exists);
#             11 = target source non-authoritative (Tier 0 replay defense) -> ABORT;
#             2/3 = QMT/other failure -> abort.
Log "Step 4: reconcile_plan.py (live vs EOD top-25 target diff, --target-kind eod)"
$reconArgs = @(
    "-X", "utf8",
    "scripts\reconcile_plan.py",
    "--target-plan", "data\orders\latest.json",
    "--target-kind", "eod",
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
    Log "Step 4: using reconcile plan (residual to close gap to EOD top-25)"
} elseif ($reconExit -eq 10) {
    # EOD target missing or stale. Since target IS the EOD plan there's no
    # separate fallback file -- abort rather than trade blind.
    Abort "reconcile signalled exit 10 (EOD target missing/stale) -- no fallback target, aborting (do NOT trade blind)"
} elseif ($reconExit -eq 11) {
    # EOD target source not prod-authoritative (round 213 Tier 0 defense).
    Abort "reconcile signalled exit 11 (EOD target source non-authoritative -- replay/ad-hoc) -- aborting"
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
    "--max-orders", "35",
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
