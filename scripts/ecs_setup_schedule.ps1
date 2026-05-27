# Register Windows Task Scheduler entry to run ECS auto-execute pipeline
# at 09:25 Mon-Fri.
#
# Run ONCE on ECS to set up. Requires Administrator.

$ErrorActionPreference = "Stop"

$TaskName = "MoneyPrinter-AutoExecute"
$Script = "C:\money-printer\scripts\ecs_auto_execute.ps1"
$RunTime = "09:25:00"

# Sanity check: script must exist
if (-not (Test-Path $Script)) { throw "script not found: $Script" }

# Build action
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Script`""

# Build trigger: weekdays at 09:25
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $RunTime

# Build settings: run if missed, allow hard stop, allow on AC + battery
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 2)

# Build principal: run as current user with highest privileges
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

# Remove existing task with same name (idempotent)
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task: $TaskName"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Register new task
Write-Host "Registering task: $TaskName"
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Money Printer: auto-execute QMT orders at 09:25 Mon-Fri (git pull + execute_orders --mode auto)"

Write-Host ""
Write-Host "Done. Verify with:"
Write-Host "  Get-ScheduledTask -TaskName $TaskName"
Write-Host "  Get-ScheduledTaskInfo -TaskName $TaskName"
