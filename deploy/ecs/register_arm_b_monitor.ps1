# Register (or re-register) ONLY the Arm B stop monitor task on ECS.
# Safe to run on a live box: it does not touch the other MoneyPrinter-* tasks
# (scripts\ecs_setup_schedule.ps1 re-registers everything and DISABLES
# DailyReport/QfqRefresh, so do not use that on a box that is already live).
#
# (audit 2026-09-23) mp/risk/freeze.guard_or_raise is fail-closed on the
# monitor heartbeat (data\.arm_b_monitor_heartbeat, max age 36h). The monitor
# MUST run on the same host as execute_orders, i.e. here on ECS.
# Exit codes 0/1/2 write the heartbeat; 3 (internal error) does not.
#
# Usage (Administrator):  powershell -ExecutionPolicy Bypass -File deploy\ecs\register_arm_b_monitor.ps1
$ErrorActionPreference = "Stop"
$REPO = "C:\money-printer"
$monitorName = "MoneyPrinter-ArmBStopMonitor"
$monitorScript = "$REPO\scripts\arm_b_stop_monitor.py"
if (-not (Test-Path $monitorScript)) { throw "script not found: $monitorScript" }
$monitorAction = New-ScheduledTaskAction `
    -Execute "$REPO\.venv\Scripts\python.exe" `
    -Argument "-X utf8 `"$monitorScript`"" `
    -WorkingDirectory $REPO
$monitorTrigger = New-ScheduledTaskTrigger `
    -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At "09:00:00"
$monitorTrigger.Repetition = (New-ScheduledTaskTrigger -Once -At "09:00:00" `
    -RepetitionInterval (New-TimeSpan -Minutes 15) `
    -RepetitionDuration (New-TimeSpan -Hours 7)).Repetition
$monitorSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew
$monitorPrincipal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest
$existingMonitor = Get-ScheduledTask -TaskName $monitorName -ErrorAction SilentlyContinue
if ($existingMonitor) {
    Write-Host "Removing existing task: $monitorName"
    Unregister-ScheduledTask -TaskName $monitorName -Confirm:$false
}
Write-Host "Registering task: $monitorName (every 15 min 09:00-16:00 Mon-Fri)"
Register-ScheduledTask `
    -TaskName $monitorName `
    -Action $monitorAction `
    -Trigger $monitorTrigger `
    -Settings $monitorSettings `
    -Principal $monitorPrincipal `
    -Description "Money Printer: Arm B -5pp hard-stop monitor + liveness heartbeat for execute_orders freeze guard (fail-closed, 36h)." | Out-Null

Write-Host "Done. Verify: Get-ScheduledTask -TaskName $monitorName"
