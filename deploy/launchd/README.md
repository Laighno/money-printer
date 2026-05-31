# macOS launchd Deployment — Phase 4 Monitoring (round 165)

Two LaunchAgent plists for the Phase 3 OOS Arm B monitoring:

| Plist | Purpose | Cadence |
|---|---|---|
| `com.moneyprinter.arm_b_monitor.plist` | `arm_b_stop_monitor.py` — round 161 guardrail (d) -5pp hard-stop | every 15 min |
| `com.moneyprinter.arm_b_report.plist` | `oos_arm_b_report.py` — round 161 guardrail (c) monthly OOS vs EOD compare | 1st of month, 08:00 |

## Why launchd not cron

macOS deprecated `cron` for desktop use; `launchd` survives reboots and respects sleep/wake (re-fires on next valid slot). The 15-min monitor is harmless if it fires off-market (script exits 0 when Arm B has no executions in history — verified in round 162 smoke test).

## Load

```bash
# from the repo root (paths are absolute inside the plist, so cwd doesn't matter)
launchctl bootstrap gui/$UID deploy/launchd/com.moneyprinter.arm_b_monitor.plist
launchctl bootstrap gui/$UID deploy/launchd/com.moneyprinter.arm_b_report.plist
```

## Verify

```bash
launchctl list | grep moneyprinter
# Expect 2 lines, PID = "-" between fires, Status = 0 (last exit)
```

## Logs

```bash
tail -f data/logs/launchd_arm_b_monitor.out.log
tail -f data/logs/launchd_arm_b_monitor.err.log
tail -f data/logs/launchd_arm_b_report.out.log
tail -f data/logs/launchd_arm_b_report.err.log
```

## Unload (recovery / cleanup)

```bash
launchctl bootout gui/$UID/com.moneyprinter.arm_b_monitor
launchctl bootout gui/$UID/com.moneyprinter.arm_b_report
```

## On freeze trigger

If `arm_b_stop_monitor.py` writes `data/.real_money_frozen` (hard-stop ≤ -5pp), the monitor will keep firing every 15 min and exit 2 each time (state propagates: `is_frozen() = True` → run_check returns 2 before doing anything else). To recover after user explicit approval:

```bash
.venv/bin/python -c "
from mp.risk.freeze import unfreeze
unfreeze(by='<your-name>', approval_token='<fresh-uuid-or-issue-id>')
"
```

Then on the ECS Windows side:

```powershell
Enable-ScheduledTask -TaskName 'MoneyPrinter-IntradayPipeline'
```

(See `deploy/ecs/` for the packaged PowerShell snippet.)
