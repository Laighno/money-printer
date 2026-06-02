# Register Windows Task Scheduler entries for the Money Printer ECS
# pipeline. Idempotent -- re-running drops and re-registers the tasks
# (existing fills / flags untouched).
#
# Run ONCE on ECS to set up, and again whenever you change either of
# the runners or their cadence. Requires Administrator.
#
# Tasks registered:
#   1. MoneyPrinter-AutoExecute        09:25:00 Mon-Fri
#      → scripts\ecs_auto_execute.ps1
#      → 9:30 entry path (Phase C fallback gate baked in).
#   2. MoneyPrinter-IntradayPipeline   14:29:55 Mon-Fri    (P11-5)
#      → scripts\ecs_intraday_execute.ps1
#      → 14:30 entry path. Combined Phase A + B in a single process
#        (round 99 decision): the runner invokes scripts/intraday_plan.py
#        which sleep-to-snapshots 14:30:00, then preflights + executes.
#        5s lead time = sleep_to_trigger landing buffer.

$ErrorActionPreference = "Stop"

$REPO = "C:\money-printer"

function Register-MPTask {
    param(
        [Parameter(Mandatory=$true)] [string]$TaskName,
        [Parameter(Mandatory=$true)] [string]$Script,
        [Parameter(Mandatory=$true)] [string]$RunTime,        # "HH:mm:ss"
        [Parameter(Mandatory=$true)] [string]$Description,
        [int]$ExecutionLimitMinutes = 30
    )

    if (-not (Test-Path $Script)) { throw "script not found: $Script" }

    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Script`""

    $trigger = New-ScheduledTaskTrigger `
        -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
        -At $RunTime

    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -DontStopOnIdleEnd `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes $ExecutionLimitMinutes) `
        -RestartCount 2 `
        -RestartInterval (New-TimeSpan -Minutes 2)

    $principal = New-ScheduledTaskPrincipal `
        -UserId $env:USERNAME `
        -LogonType Interactive `
        -RunLevel Highest

    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Removing existing task: $TaskName"
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    Write-Host "Registering task: $TaskName (run $RunTime Mon-Fri)"
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description $Description
}

# ── Task 1: 09:25 auto-execute (9:30 entry path, Phase C gate inside) ──
Register-MPTask `
    -TaskName "MoneyPrinter-AutoExecute" `
    -Script "$REPO\scripts\ecs_auto_execute.ps1" `
    -RunTime "09:25:00" `
    -Description "Money Printer: 9:30 entry path (git pull + execute_orders --mode auto). Skips when previous trading day's 14:30 path succeeded (intraday_success flag)." `
    -ExecutionLimitMinutes 30

# ── Task 2: 14:28:00 intraday pipeline (14:30 entry path, P11-5) ────
# round 185 fix (advisor 184): trigger 90s before 14:30:00, NOT 5s.
# Old 14:29:55 left only 5s for the task scheduler + `git pull` to
# complete before intraday_plan.py's sleep_to_trigger 14:30:30 deadline.
# 6/1 14:30 run actually fired exit 2 because `git pull` took 33s (cold
# cache after weekend) and missed the deadline by ~3s — see round 184.
# 14:25:00 (round 194 fix): git pull (~10-30s) + Step 2a warm cache (~141s
# for 615 codes, measured 6/2) + intraday_plan setup needs ~3 min total.
# sleep_to_trigger deadline = 14:30:30, so fire 14:25:00 gives 5:30 buffer
# (300s+30s safety). Previous 14:28:00 schedule (round 185) was only 2:30
# which fit git pull but NOT post-round-194 warm step.
# sleep_to_trigger still anchors execution to 14:30:00 (Rule #11 unchanged).
Register-MPTask `
    -TaskName "MoneyPrinter-IntradayPipeline" `
    -Script "$REPO\scripts\ecs_intraday_execute.ps1" `
    -RunTime "14:25:00" `
    -Description "Money Printer: 14:30 entry path (intraday_plan + preflight + execute_orders). Writes intraday_success flag on success -- next morning's 9:25 task skips." `
    -ExecutionLimitMinutes 35

# ── Task 3: 17:00 daily report (P0-A migration from Mac launchd ─────
# com.moneyprinter.collect, feat/ecs-standalone branch)
# Replaces Mac launchd 17:00 daily that became single point of failure
# when Mac shut down. Runs collect + daily_report + push so ECS 9:25
# next-day task has a fresh plan.
# Note: registered DISABLED by default; D3 in P0-A migration plan enables
# this AND disables Mac launchd `com.moneyprinter.collect` simultaneously.
Register-MPTask `
    -TaskName "MoneyPrinter-DailyReport" `
    -Script "$REPO\scripts\ecs_daily_report.ps1" `
    -RunTime "17:00:00" `
    -Description "Money Printer: 17:00 daily report (sync portfolio + collect + daily_report + push plan). P0-A migration from Mac launchd com.moneyprinter.collect." `
    -ExecutionLimitMinutes 30

# ── Task 4: Saturday 10:00 qfq refresh (P0-B migration from Mac ──
# launchd com.moneyprinter.qfq, feat/ecs-standalone branch)
# Replaces Mac launchd weekly Saturday qfq refresh. Heavy (~30-60 min).
# Note: registered DISABLED by default; D3 in P0-B migration plan enables.
Register-MPTask `
    -TaskName "MoneyPrinter-QfqRefresh" `
    -Script "$REPO\scripts\ecs_qfq_refresh.ps1" `
    -RunTime "10:00:00" `
    -Description "Money Printer: Saturday 10:00 weekly qfq refresh. P0-B migration from Mac launchd com.moneyprinter.qfq." `
    -ExecutionLimitMinutes 90

# Disable Task 3 + 4 by default (P0 D2 stage; enable manually D3 step
# when ready to migrate from Mac launchd).
Disable-ScheduledTask -TaskName "MoneyPrinter-DailyReport" | Out-Null
Disable-ScheduledTask -TaskName "MoneyPrinter-QfqRefresh" | Out-Null
Write-Host "Task 3 (DailyReport) and Task 4 (QfqRefresh) registered DISABLED (P0 D2)."

Write-Host ""
Write-Host "Done. Verify with:"
Write-Host "  Get-ScheduledTask -TaskName MoneyPrinter-AutoExecute"
Write-Host "  Get-ScheduledTask -TaskName MoneyPrinter-IntradayPipeline"
Write-Host "  Get-ScheduledTask -TaskName MoneyPrinter-DailyReport       # P0-A, default Disabled"
Write-Host "  Get-ScheduledTask -TaskName MoneyPrinter-QfqRefresh        # P0-B, default Disabled"
Write-Host "  Get-ScheduledTaskInfo -TaskName MoneyPrinter-IntradayPipeline"
