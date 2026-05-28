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

# ── Task 2: 14:29:55 intraday pipeline (14:30 entry path, P11-5) ────
# 14:29:55 (not 14:30:00) gives intraday_plan.py 5s lead time. Inside
# it, sleep_to_trigger() blocks until 14:30:00 exactly (Rule #11).
# Hard deadline 14:30:30 inside the script means there's still a 25s
# operational buffer for the task scheduler itself to fire on time.
# Execution limit is 35min: plan generation typically 1-3min + execute
# ~2-5min, but xtdata fetch can occasionally take longer; 35min keeps
# us safely under the 15:00 close.
Register-MPTask `
    -TaskName "MoneyPrinter-IntradayPipeline" `
    -Script "$REPO\scripts\ecs_intraday_execute.ps1" `
    -RunTime "14:29:55" `
    -Description "Money Printer: 14:30 entry path (intraday_plan + preflight + execute_orders). Writes intraday_success flag on success -- next morning's 9:25 task skips." `
    -ExecutionLimitMinutes 35

Write-Host ""
Write-Host "Done. Verify with:"
Write-Host "  Get-ScheduledTask -TaskName MoneyPrinter-AutoExecute"
Write-Host "  Get-ScheduledTask -TaskName MoneyPrinter-IntradayPipeline"
Write-Host "  Get-ScheduledTaskInfo -TaskName MoneyPrinter-IntradayPipeline"
