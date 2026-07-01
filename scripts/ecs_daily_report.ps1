# ECS daily report pipeline (Mac launchd com.moneyprinter.collect 等价)
#
# Mon-Fri 17:00 fire (round 195+ schedule, feat/ecs-standalone P0-A migration):
#   - sync portfolio.yaml from QMT
#   - collect external data (northbound / margin / fund_flow)
#   - run daily_report.py (generates plan + Feishu notify)
#   - shadow_930_intraday.py (Arm B research, non-fatal)
#   - git commit + push (latest.json, orders/*, daily_*.md, portfolio.yaml)
#
# Pre-conditions (MANUAL setup, one-time):
#   - XtMiniQmt logged in (for portfolio sync; daily_report uses
#     daily data from mp.data, not 1m, so cache-read assumption ok)
#   - git credentials configured (ECS already pushes commits, see
#     6/1 commit `98e1175 auto: daily plan ...` came from ECS path)
#   - Python venv at C:\money-printer\.venv

$ErrorActionPreference = "Continue"
$LogPath = "C:\money-printer\data\logs\ecs_daily_report.log"
$REPO = "C:\money-printer"
$BRANCH = "main"

function Log { param([string]$msg)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content $LogPath $line
}

function Abort { param([string]$msg)
    Log "ABORT: $msg"
    exit 1
}

Log "==================== ECS daily-report start ===================="
Set-Location $REPO

# Cap scoring parallelism so daily_report.py fits ECS 8GB RAM (2026-07-01 fix).
# Default in code is 8 ProcessPool workers -> OOM-killed daily_report on 6/22,
# 6/25, 6/30 (each worker loads ~800 stocks' data; 8x peak > free RAM). 3
# workers fit; slower (~2.5x) but the 17:00 batch has time. Mac keeps default 8.
$env:MP_SCORE_WORKERS = "3"

$pythonExe = "$REPO\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) { Abort "python not found: $pythonExe" }

# Step 1: git pull (latest models, scripts, config)
Log "Step 1: git pull origin $BRANCH"
$pullOutput = & git pull origin $BRANCH 2>&1 | Out-String
$pullOutput.Trim().Split("`n") | ForEach-Object { Log "  git: $_" }
if ($LASTEXITCODE -ne 0) { Abort "git pull failed (exit $LASTEXITCODE)" }
$head = (& git rev-parse HEAD 2>&1).Trim()
Log "Step 1: HEAD = $head"

# Step 2: sync portfolio.yaml from QMT (ECS-local mode, round 197 fix)
# sync_portfolio_from_qmt.py --local: in-process qmt_snapshot + write yaml,
# no SSH self-loop. Replaces D2.5 stop-gap that only printed JSON.
Log "Step 2: sync_portfolio_from_qmt.py --local (ECS-local mode)"
$syncOutput = & $pythonExe -X utf8 scripts\sync_portfolio_from_qmt.py --local 2>&1 | Out-String
$syncOutput.Trim().Split("`n") | ForEach-Object { Log "  sync: $_" }
if ($LASTEXITCODE -ne 0) {
    Log "  WARNING: ECS-local sync failed (exit $LASTEXITCODE); proceeding with existing yaml"
}

# Step 3: collect external data (northbound + margin + fund_flow)
Log "Step 3: mp.data.collector"
$collectOutput = & $pythonExe -X utf8 -m mp.data.collector 2>&1 | Out-String
$collectOutput.Trim().Split("`n") | ForEach-Object { Log "  collect: $_" }
if ($LASTEXITCODE -ne 0) {
    Log "  WARNING: collector failed (exit $LASTEXITCODE); proceeding (daily_report may be partial)"
}

# Step 4: daily_report.py (generates plan + Feishu)
Log "Step 4: daily_report.py"
$reportOutput = & $pythonExe -X utf8 scripts\daily_report.py --allow-prod-write 2>&1 | Out-String
$reportOutput.Trim().Split("`n") | ForEach-Object { Log "  report: $_" }
if ($LASTEXITCODE -ne 0) { Abort "daily_report.py failed (exit $LASTEXITCODE)" }

# Step 5: Arm B shadow recorder (non-fatal, research; 10 min timeout)
# Shadow re-fetches 60-day daily bars for 615 codes (~30-60 min on slow
# network). Cap at 10 min: if not done, kill + skip (research data is
# nice-to-have, prod execute path unaffected). round 197 fix.
Log "Step 5: shadow_930_intraday.py (Arm B shadow, 10 min timeout, non-fatal)"
if (-not (Test-Path "$REPO\data\shadow_930")) {
    New-Item -ItemType Directory -Path "$REPO\data\shadow_930" -Force | Out-Null
}
$shadowJob = Start-Job -ScriptBlock {
    param($py, $repo)
    Set-Location $repo
    & $py -X utf8 scripts\shadow_930_intraday.py 2>&1
} -ArgumentList $pythonExe, $REPO
$completed = Wait-Job $shadowJob -Timeout 600
if ($completed) {
    $shadowOutput = Receive-Job $shadowJob | Out-String
    $shadowExit = if ($shadowJob.State -eq 'Failed') { 1 } else { 0 }
    $shadowOutput.Trim().Split("`n") | Select-Object -Last 5 | ForEach-Object { Log "  shadow: $_" }
    if ($shadowExit -ne 0) {
        Log "  WARNING: shadow failed (non-fatal)"
    } else {
        Log "  shadow OK"
    }
} else {
    Stop-Job $shadowJob
    Log "  WARNING: shadow timeout (>10 min) -- killed, research data skipped (non-fatal)"
}
Remove-Job $shadowJob -Force -ErrorAction SilentlyContinue

# Step 6: git commit plan files LOCALLY (round 195 C-arch: no push)
# Background: ECS GitHub deploy key is read-only by design (security).
# User round 195 拍板 option C: ECS writes runtime state locally; prod
# execute_orders (ecs_auto_execute.ps1) reads from local file system, not
# git. Mac launchd com.moneyprinter.collect to be disabled at P0 D3 step
# to avoid double-write conflict.
# Local commit kept for audit trail (ECS git log shows daily commits even
# without push). ECS local main diverges from origin/main by N commits but
# never pushes — D7 reconciliation: squash or reset.
Log "Step 6: git commit plan files LOCALLY (no push, P0 C-arch)"
& git add data/orders/latest.json data/orders/orders_*.json data/reports/daily_*.md config/portfolio.yaml data/external/*.parquet 2>&1 | Out-Null
& git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    $commitMsg = "auto: daily plan $(Get-Date -Format 'yyyy-MM-dd') (ECS-local, no push by P0 C-arch design)"
    & git commit -m $commitMsg 2>&1 | Out-String | ForEach-Object { Log "  git: $_" }
    Log "Step 6: committed locally (ECS write-only; deploy key read-only by design)"
} else {
    Log "Step 6: no changes to commit (idempotent / weekend / market closed)"
}

Log "==================== ECS daily-report DONE ===================="
exit 0
