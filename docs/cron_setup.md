# crontab / launchd Setup

**Source of truth for production scheduler entries.** When this file is
updated, manually apply via `crontab path/to/new` and `launchctl load`.
Reason for not auto-applying: macOS requires Full Disk Access for
`crontab` modification and we don't want to grant that to Claude Code.

---

## Current crontab (post P5-2, 2026-05-24)

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
# Reads data/reports/backtest_history.json mtime; if > 7 days ago,
# sends RED ALERT to Feishu indicating weekly cron likely failed.
# Independent of walk_forward_backtest.py to avoid dependency loop
# (if walk_forward broke, this still fires).
0 6 * * 6 /Users/laighno/laighno/money-printer/.venv/bin/python scripts/monitor/weekly_heartbeat.py >> data/logs/heartbeat.log 2>&1
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
- mp/monitor/threshold_alert.py — P4-1C breach alerts (now actually triggers because we removed --update-only)
- scripts/monitor/weekly_heartbeat.py — P5-B dead-man-switch implementation
