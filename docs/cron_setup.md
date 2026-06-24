# crontab / launchd Setup

**Source of truth for production scheduler entries.** When this file is
updated, manually apply via `crontab path/to/new` and `launchctl load`.
Reason for not auto-applying: macOS requires Full Disk Access for
`crontab` modification and we don't want to grant that to Claude Code.

---

## ⚠️ RETRAIN MIGRATED Mac → ECS (2026-06-24, advisor; user 拍 ①B/②B)

**The Mac `Friday 18:00 walk_forward_backtest.py` retrain below is RETIRED.**
Root cause: it silently died ~4/24 when the laptop slept → prod blend froze at
6/2 for 3 weeks (nobody noticed until the user asked "模型还在每周更吗"). It also
saved prod models with **no verify gate** (any weak retrain overwrote prod).

Replacement runs on **ECS (always-on)** via Task Scheduler, gated:

| Task | When | Script | Purpose |
|---|---|---|---|
| `MP-AutoRetrain` | Fri 18:30 | `scripts/ecs_auto_retrain.ps1` → `auto_retrain.py --auto-swap --allow-prod-write` | refresh cache → train(current cutoff) → **verify gate** → swap (only on PASS; first swap manual per ②B) |
| `MP-RetrainDeadman` | Sat 09:00 | `scripts/monitor/retrain_freshness_check.py` | RED Feishu if `data/auto_retrain_last.json` >8d stale or gate keeps failing |

Register on ECS (PowerShell, once):
```powershell
$py = "C:\money-printer\.venv\Scripts\python.exe"
# weekly retrain
schtasks /Create /TN "MP-AutoRetrain" /SC WEEKLY /D FRI /ST 18:30 /RU SYSTEM ^
  /TR "powershell -ExecutionPolicy Bypass -File C:\money-printer\scripts\ecs_auto_retrain.ps1"
# dead-man-switch (independent failure domain, next morning)
schtasks /Create /TN "MP-RetrainDeadman" /SC WEEKLY /D SAT /ST 09:00 /RU SYSTEM ^
  /TR "$py -X utf8 C:\money-printer\scripts\monitor\retrain_freshness_check.py"
```

First-swap approval (②B, run once on ECS after the first gate-PASS stages
`data/retrain_pending_swap.json`): `.venv\Scripts\python.exe scripts\swap_model.py
--new-prefix <staged> --allow-prod-write --reason "first manual swap"`. This
creates `data/.first_swap_done`; thereafter weekly gate-PASS swaps auto-apply.

**Mac action:** remove the `Friday 18:00 walk_forward_backtest.py` line from the
Mac crontab (block below kept for history/rollback only). The Saturday
`weekly_heartbeat.py` may stay (it watches backtest_history.json, now a
secondary signal) but the authoritative dead-man-switch is `MP-RetrainDeadman`.

---

## Current crontab (post P6-X3, 2026-05-24)

```cron
# Friday 18:00 — full walk-forward + retrain production models
# Dropped --update-only on 2026-05-24 (P5-2, docs/dialog/ round 43):
# --update-only used train_fast on full panel and produced IC≈-0.005
# weak BlendRanker every Friday, silently nuking 1.90 Sharpe model.
# Full walk_forward uses expanding-window training, producing the real
# 1.90 Sharpe blend then saving via update_production_models (which now
# correctly routes ranker_is_blend=True → save from walk-forward).
# Side effect: cron now also triggers send_model_update_report which
# fires P4-1C threshold alerts via Feishu (no-op silent before).
# Runtime: ~30 min (was ~5 min).
0 18 * * 5 /Users/laighno/laighno/money-printer/.venv/bin/python scripts/walk_forward_backtest.py >> data/logs/model_update.log 2>&1

# Saturday 06:00 — weekly heartbeat (P5-B dead-man-switch)
# Reads data/reports/backtest_history.json mtime; > 5 trading days →
# YELLOW, > 10 trading days → RED (P6-X2 round 47: was 7 calendar days,
# false-positives across CNY / National Day). Sends Feishu alert when
# weekly cron likely silently failed.  Independent of
# walk_forward_backtest.py — if walk_forward broke, this still fires.
0 6 * * 6 /Users/laighno/laighno/money-printer/.venv/bin/python scripts/monitor/weekly_heartbeat.py >> data/logs/heartbeat.log 2>&1

# Daily 07:00 — cron drift detect (P6-X1 round 47)
# SHA256-compares the live `crontab -l` output against the
# "## Current crontab" block in this very file (docs/cron_setup.md).
# Mismatch → RED Feishu alert. Catches: someone ran `crontab -e` outside
# this docs workflow, or the manual apply step was skipped after a docs
# update.  Runs 60 min after the heartbeat slot — independent failure
# domain. Reads crontab; never modifies it.
0 7 * * * /Users/laighno/laighno/money-printer/.venv/bin/python scripts/monitor/cron_drift_detect.py >> data/logs/cron_drift.log 2>&1

# Saturday 06:30 — paper_trade vs walk_forward Sharpe drift (P6-X3 round 47)
# Computes rolling 20-day Sharpe from data/paper_trade/state.json::nav_history
# and compares to the latest data/reports/backtest_history.json bt_metrics
# Sharpe.  |Δ|>0.5 → YELLOW; |Δ|>1.0 AND paper Sharpe negative → RED.
# Min N=15 NAV entries before any alert (cold-start guard). Skips if
# walk_forward data >21 days old (heartbeat would already RED).  Same
# batch as heartbeat 06:00 but independent failure domain (30 min offset).
30 6 * * 6 /Users/laighno/laighno/money-printer/.venv/bin/python scripts/monitor/paper_trade_drift_detect.py >> data/logs/paper_trade_drift.log 2>&1
```

## How to apply

```bash
# 1. Pull this repo (so the script files exist locally)
git pull

# 2. Compare with current crontab
crontab -l

# 3. Save the block above to /tmp/cron and apply
crontab /tmp/cron

# 4. Verify
crontab -l
```

## Previous crontab (pre P5-2, archived for rollback)

```cron
# Friday 18:00 — full backtest + retrain production models
0 18 * * 5 /Users/laighno/laighno/money-printer/.venv/bin/python scripts/walk_forward_backtest.py --update-only >> data/logs/model_update.log 2>&1
```

Rollback: replace current crontab with the above block via `crontab` command above.

## launchd plists (for daily_report etc)

The following launchd agents handle other scheduled tasks
(not modified by this P5 chain):

- `~/Library/LaunchAgents/com.moneyprinter.collect.plist` — external data collection
- `~/Library/LaunchAgents/com.moneyprinter.execute-live.plist.disabled` — live execution (disabled)
- `~/Library/LaunchAgents/com.moneyprinter.execute-preview.plist` — execution preview
- `~/Library/LaunchAgents/com.moneyprinter.intraday-2pm.plist` — 14:00 intraday report
- `~/Library/LaunchAgents/com.moneyprinter.midday.plist` — midday report
- `scripts/daily_report.sh` — 18:00 Mon-Fri daily report (probably launchd, plist not in standard location?)

If any of these change, add to the table above + commit.

## References

- docs/dialog/ round 41-43 — P5 chain background + design + crontab bug discovery
- docs/dialog/ round 47 — P6-X1/X2/X3 cron drift + trading-day-aware heartbeat + paper_trade Sharpe drift
- mp/monitor/threshold_alert.py — P4-1C breach alerts (now actually triggers because we removed --update-only)
- scripts/monitor/weekly_heartbeat.py — P5-B dead-man-switch implementation (trading-day-aware as of P6-X2)
- scripts/monitor/cron_drift_detect.py — P6-X1 drift detector
- scripts/monitor/paper_trade_drift_detect.py — P6-X3 paper/walk-forward Sharpe drift monitor
