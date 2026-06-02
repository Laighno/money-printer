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

$pythonExe = "$REPO\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) { Abort "python not found: $pythonExe" }

# Step 1: git pull (latest models, scripts, config)
Log "Step 1: git pull origin $BRANCH"
$pullOutput = & git pull origin $BRANCH 2>&1 | Out-String
$pullOutput.Trim().Split("`n") | ForEach-Object { Log "  git: $_" }
if ($LASTEXITCODE -ne 0) { Abort "git pull failed (exit $LASTEXITCODE)" }
$head = (& git rev-parse HEAD 2>&1).Trim()
Log "Step 1: HEAD = $head"

# Step 2: sync portfolio.yaml from QMT (ECS-local, no SSH self-loop)
# round 195 fix: original sync_portfolio_from_qmt.py SSH-es to ECS (designed
# for Mac caller); on ECS that's self-SSH → hang. Call qmt_snapshot.py
# directly + post-process to rewrite yaml.
Log "Step 2: qmt_snapshot.py (ECS-local portfolio sync)"
$snapOutput = & $pythonExe -X utf8 scripts\qmt_snapshot.py 2>&1 | Out-String
$snapOutput.Trim().Split("`n") | ForEach-Object { Log "  snapshot: $_" }
if ($LASTEXITCODE -ne 0) {
    Log "  WARNING: qmt_snapshot failed (exit $LASTEXITCODE); proceeding with existing yaml"
}
# Note: D2.5 doesn't write portfolio.yaml from snapshot — that's a separate
# code path in sync_portfolio_from_qmt.py post-SSH. For D2.5 dry-run, we just
# confirm QMT is reachable; yaml stays as-is. D7 follow-up: extract yaml-write
# logic from sync_portfolio_from_qmt.py and run it locally on ECS.

# Step 3: collect external data (northbound + margin + fund_flow)
Log "Step 3: mp.data.collector"
$collectOutput = & $pythonExe -X utf8 -m mp.data.collector 2>&1 | Out-String
$collectOutput.Trim().Split("`n") | ForEach-Object { Log "  collect: $_" }
if ($LASTEXITCODE -ne 0) {
    Log "  WARNING: collector failed (exit $LASTEXITCODE); proceeding (daily_report may be partial)"
}

# Step 4: daily_report.py (generates plan + Feishu)
Log "Step 4: daily_report.py"
$reportOutput = & $pythonExe -X utf8 scripts\daily_report.py 2>&1 | Out-String
$reportOutput.Trim().Split("`n") | ForEach-Object { Log "  report: $_" }
if ($LASTEXITCODE -ne 0) { Abort "daily_report.py failed (exit $LASTEXITCODE)" }

# Step 5: Arm B shadow recorder (non-fatal, research)
Log "Step 5: shadow_930_intraday.py (Arm B shadow, non-fatal)"
if (-not (Test-Path "$REPO\data\shadow_930")) {
    New-Item -ItemType Directory -Path "$REPO\data\shadow_930" -Force | Out-Null
}
$shadowOutput = & $pythonExe -X utf8 scripts\shadow_930_intraday.py 2>&1 | Out-String
$shadowOutput.Trim().Split("`n") | ForEach-Object { Log "  shadow: $_" }
if ($LASTEXITCODE -ne 0) {
    Log "  WARNING: shadow recorder failed (non-fatal; real path unaffected)"
}

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
