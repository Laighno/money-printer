# ECS qfq refresh (Mac launchd com.moneyprinter.qfq 等价)
#
# Saturday 10:00 weekly fire (feat/ecs-standalone P0-B migration):
#   - git pull
#   - python scripts/qfq_refresh.py --feishu
#   - git commit + push (data/db/* qfq-refreshed parquet)
#
# qfq_refresh.py 是重操作 (~30-60 min), 周末跑不影响 trading day pipeline.

$ErrorActionPreference = "Continue"
$LogPath = "C:\money-printer\data\logs\ecs_qfq_refresh.log"
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

Log "==================== ECS qfq-refresh start ===================="
Set-Location $REPO

$pythonExe = "$REPO\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) { Abort "python not found: $pythonExe" }

# Step 1: git pull
Log "Step 1: git pull origin $BRANCH"
$pullOutput = & git pull origin $BRANCH 2>&1 | Out-String
$pullOutput.Trim().Split("`n") | ForEach-Object { Log "  git: $_" }
if ($LASTEXITCODE -ne 0) { Abort "git pull failed (exit $LASTEXITCODE)" }

# Step 2: qfq_refresh.py --feishu
Log "Step 2: qfq_refresh.py --feishu (heavy, ~30-60 min)"
$refreshOutput = & $pythonExe -X utf8 scripts\qfq_refresh.py --feishu 2>&1 | Out-String
$refreshOutput.Trim().Split("`n") | ForEach-Object { Log "  refresh: $_" }
if ($LASTEXITCODE -ne 0) { Abort "qfq_refresh.py failed (exit $LASTEXITCODE)" }

# Step 3: git commit + push (data/db parquet updates)
Log "Step 3: git commit + push qfq-refreshed data/db"
& git add data/db/ 2>&1 | Out-Null
& git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    $commitMsg = "auto: qfq refresh $(Get-Date -Format 'yyyy-MM-dd') (ECS-side weekly maintenance)"
    & git commit -m $commitMsg 2>&1 | Out-String | ForEach-Object { Log "  git: $_" }
    & git push origin $BRANCH 2>&1 | Out-String | ForEach-Object { Log "  git: $_" }
    if ($LASTEXITCODE -ne 0) { Abort "git push failed (exit $LASTEXITCODE)" }
} else {
    Log "Step 3: no changes to commit (qfq already up-to-date)"
}

Log "==================== ECS qfq-refresh DONE ===================="
exit 0
