# Phase 3 launch — enable 14:30 OOS Arm B intraday pipeline on ECS.
#
# Run ONCE in an Administrator PowerShell on the Money Printer ECS
# Windows host (RDP in). Idempotent — re-running is safe (just
# re-enables / queries).
#
# Pre-flight (engineer already verified locally before sending this):
#   - n2c BlendRanker + StockRanker swapped (round 162 commit 94a7002)
#   - 4 guardrails wired + smoke-tested (round 162)
#   - final dryrun picks sane (round 164 commit 3b44851)
#   - launchd monitor + monthly report plists loaded on Mac side
#   - data/.real_money_frozen flag cleared / non-existent
#   - ARM_B_BUDGET_MAX env not set (defaults to ¥20,000 in tracker)
#
# After running this script, the next eligible 14:30 will fire the OOS
# pipeline; first execution = next Mon 14:30:00 (after launch).
#
# Rollback (any time, no user approval needed):
#     Disable-ScheduledTask -TaskName "MoneyPrinter-IntradayPipeline"
#
# Recovery after hard-stop (-5pp) auto-freeze: user must explicitly
# approve, then run BOTH:
#     1. (Mac side) unfreeze flag via mp.risk.freeze.unfreeze(...)
#     2. (ECS, this file)  Enable-ScheduledTask -TaskName "MoneyPrinter-IntradayPipeline"

$ErrorActionPreference = "Stop"

$TaskName = "MoneyPrinter-IntradayPipeline"

Write-Host "=== Phase 3 launch: Enable $TaskName ===" -ForegroundColor Cyan

# (1) Show current state first
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $existing) {
    Write-Host "ERROR: ScheduledTask '$TaskName' not found." -ForegroundColor Red
    Write-Host "       Run scripts\ecs_setup_schedule.ps1 first to register it." -ForegroundColor Yellow
    exit 1
}

Write-Host "`nBefore:"
$existing | Select-Object TaskName, State, @{N="NextRunTime";E={(Get-ScheduledTaskInfo $_).NextRunTime}}, LastRunTime, LastTaskResult | Format-Table -AutoSize

# (2) Enable the task
Enable-ScheduledTask -TaskName $TaskName

# (3) Show after-state
$after = Get-ScheduledTask -TaskName $TaskName
Write-Host "`nAfter:"
$after | Select-Object TaskName, State, @{N="NextRunTime";E={(Get-ScheduledTaskInfo $_).NextRunTime}}, LastRunTime, LastTaskResult | Format-Table -AutoSize

if ($after.State -eq "Ready") {
    Write-Host "`n✓ $TaskName enabled. First run at the next 14:29:55 trigger (Mon-Fri)." -ForegroundColor Green
    Write-Host "  Monitor: tail logs on Mac side (data/logs/launchd_arm_b_monitor.err.log)" -ForegroundColor DarkGray
} else {
    Write-Host "`n⚠ State is $($after.State) — expected 'Ready'. Investigate." -ForegroundColor Yellow
    exit 1
}
