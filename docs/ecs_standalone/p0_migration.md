# ECS-Standalone Migration — Phase 0 (P0) Spec

**Branch**: `feat/ecs-standalone`
**Goal**: Mac 关机后 ECS 可独立运行每日量化逻辑
**Phase 0 scope**: 迁移 2 个 P0 关键 launchd job 到 ECS Windows Task Scheduler
**Status**: SPEC, awaiting user 拍 implementation

## Phase 0 涉及的 Mac launchd jobs

| Job | Mac Schedule | Script | 影响 |
|---|---|---|---|
| `com.moneyprinter.collect` | **Mon-Fri 17:00** | `scripts/daily_report.sh` | Mac 关机 → 没人生成 next-day plan → ECS 09:25 EOD 没东西 execute |
| `com.moneyprinter.qfq` | **Sat 10:00 weekly** | `scripts/qfq_refresh.sh` | qfq 复权数据脏 → 后续所有报告 / backtest 偏 |

## P0-A: collect (daily_report.sh) 迁 ECS

### 原 Mac 行为 (`scripts/daily_report.sh`)

```bash
cd /Users/laighno/laighno/money-printer
source .venv/bin/activate
python scripts/sync_portfolio_from_qmt.py   # Step 0: re-sync portfolio.yaml from QMT
python -m mp.data.collector                  # Step 1: collect external data
python scripts/daily_report.py               # Step 2: generate plan + Feishu
python scripts/shadow_930_intraday.py        # Step 2b: Arm B shadow (non-fatal)
# Step 3: git add + commit + push latest.json (so ECS 09:25 can pull + execute)
```

### ECS 等价 (`scripts/ecs_daily_report.ps1`, 新增)

```powershell
# Register: 17:00 Mon-Fri, see ecs_setup_schedule.ps1 (P0-A 任务 3)
$ErrorActionPreference = "Continue"
$LogPath = "C:\money-printer\data\logs\ecs_daily_report.log"
$REPO = "C:\money-printer"

function Log { param([string]$msg)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line; Add-Content $LogPath $line
}

Log "==================== ECS daily-report start ===================="
cd $REPO

# Step 1: git pull (latest models, scripts)
Log "Step 1: git pull origin main"
$pullOutput = & git pull origin main 2>&1 | Out-String
$pullOutput.Trim().Split("`n") | ForEach-Object { Log "  git: $_" }
if ($LASTEXITCODE -ne 0) { Log "ABORT: git pull failed"; exit 1 }

$pythonExe = "$REPO\.venv\Scripts\python.exe"

# Step 2: sync portfolio.yaml from QMT (cap-aware, run on ECS where QMT lives)
Log "Step 2: sync_portfolio_from_qmt.py"
& $pythonExe -X utf8 scripts\sync_portfolio_from_qmt.py 2>&1 | Out-String | Tee-Object -Append $LogPath
# Non-fatal: stale yaml is fine for plan generation, but log it

# Step 3: collect external data (northbound + margin + fund_flow)
Log "Step 3: mp.data.collector"
& $pythonExe -X utf8 -m mp.data.collector 2>&1 | Out-String | Tee-Object -Append $LogPath

# Step 4: generate daily_report (plan + Feishu)
Log "Step 4: daily_report.py"
& $pythonExe -X utf8 scripts\daily_report.py 2>&1 | Out-String | Tee-Object -Append $LogPath
if ($LASTEXITCODE -ne 0) { Log "ABORT: daily_report.py failed"; exit 1 }

# Step 5: Arm B shadow (non-fatal, research)
Log "Step 5: shadow_930_intraday.py"
& $pythonExe -X utf8 scripts\shadow_930_intraday.py 2>&1 | Out-String | Tee-Object -Append $LogPath
# Don't abort on shadow failure

# Step 6: git commit + push plan files (latest.json, daily_*.md, portfolio.yaml updates)
Log "Step 6: git commit + push"
& git add data/orders/latest.json data/orders/orders_*.json data/reports/daily_*.md config/portfolio.yaml data/external/*.parquet 2>&1
$diffOutput = & git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    & git commit -m "auto: daily plan $(Get-Date -Format 'yyyy-MM-dd') (ECS-side, push for next-morning execute)" 2>&1 | Out-String | Tee-Object -Append $LogPath
    & git push origin main 2>&1 | Out-String | Tee-Object -Append $LogPath
} else {
    Log "Step 6: no changes to commit (idempotent)"
}

Log "==================== ECS daily-report DONE ===================="
exit 0
```

### Task 注册 (`ecs_setup_schedule.ps1` 增 entry)

```powershell
# Task 3: 17:00 daily report (P0-A migration)
Register-MPTask `
    -TaskName "MoneyPrinter-DailyReport" `
    -Script "$REPO\scripts\ecs_daily_report.ps1" `
    -RunTime "17:00:00" `
    -Description "Money Printer: collect + daily_report.py + push next-day plan" `
    -ExecutionLimitMinutes 30
```

### 风险点 (P0-A)

1. **ECS git credentials**: 已存在 (commit `98e1175` 之前 ECS push 过, 或将通过 ssh deploy key 配置)
2. **ECS Python venv 有 deps**: ✓ 已有 (intraday_plan 已跑通)
3. **ECS 能调 QMT**: ✓ 已有 (execute_orders 已跑通)
4. **Mac 端 double commit**: **必须先 disable Mac launchd `com.moneyprinter.collect`** 才 enable ECS task, 否则 17:00 时 ECS + Mac 都跑 daily_report → conflict push

### 测试 plan (P0-A)

| Day | 动作 | Mac launchd | ECS task |
|---|---|---|---|
| **D1 spec** | Write PS1 + register task disabled | enabled (跑) | Disabled |
| **D2 dry** | 手动 trigger ECS task, verify log + output | enabled (跑) | Disabled (manual fire only) |
| **D3 switch** | Disable Mac, Enable ECS | Disabled | Enabled |
| **D4-D7 观察** | 1 周观察, ECS daily 跑通 | Disabled | Enabled |
| **D8+ stable** | 1 周通过 → P0-A done | (留 disabled, 不删) | Enabled stable |

## P0-B: qfq_refresh 迁 ECS

### 原 Mac 行为 (`scripts/qfq_refresh.sh`)

```bash
cd /Users/laighno/laighno/money-printer
source .venv/bin/activate
python scripts/qfq_refresh.py --feishu
```

### ECS 等价 (`scripts/ecs_qfq_refresh.ps1`, 新增)

```powershell
$REPO = "C:\money-printer"
$LogPath = "C:\money-printer\data\logs\ecs_qfq_refresh.log"
function Log { ... 同上 ... }

Log "==================== ECS qfq-refresh start ===================="
cd $REPO
& git pull origin main 2>&1 | Out-String | Tee-Object -Append $LogPath

$pythonExe = "$REPO\.venv\Scripts\python.exe"
& $pythonExe -X utf8 scripts\qfq_refresh.py --feishu 2>&1 | Out-String | Tee-Object -Append $LogPath

# Commit qfq-refreshed data files (data/db/ parquet)
& git add data/db/ 2>&1
$diffOutput = & git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    & git commit -m "auto: qfq refresh $(Get-Date -Format 'yyyy-MM-dd') (ECS-side)" 2>&1
    & git push origin main 2>&1
}
Log "==================== ECS qfq-refresh DONE ===================="
```

### Task 注册

```powershell
# Task 4: Saturday 10:00 qfq refresh (P0-B migration)
Register-MPTask `
    -TaskName "MoneyPrinter-QfqRefresh" `
    -Script "$REPO\scripts\ecs_qfq_refresh.ps1" `
    -RunTime "10:00:00" `
    -DaysOfWeek Saturday `  # 注意: New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday
    -Description "Money Printer: weekly qfq refresh (Saturday 10:00)" `
    -ExecutionLimitMinutes 60
```

### P0-B 风险 (跟 P0-A 相同 + 额外):

- qfq_refresh.py 跑 30-60 分钟 (重型脚本), ExecutionLimitMinutes=60 必须给足
- 周末跑, 不影响 trading day pipeline
- Mac 关机后没人提示 (qfq 是 weekly, 不影响 daily)

## 总实施步骤 (P0-A + P0-B)

1. **本分支 `feat/ecs-standalone`** spec doc 写完 (本文) ← 你 review
2. **PS1 write**: 写 `scripts/ecs_daily_report.ps1` + `scripts/ecs_qfq_refresh.ps1`
3. **Modify `ecs_setup_schedule.ps1`**: 加 Task 3 + Task 4 register
4. **commit + push** feat/ecs-standalone 分支 (不 merge main 直到测试通过)
5. **ssh ECS**: git fetch + checkout feat/ecs-standalone branch + git pull (临时切换, 测试用)
6. **ssh ECS**: 跑 `ecs_setup_schedule.ps1` (注册 Task 3 + 4, default Disabled)
7. **D2 dry-run**: 手动 trigger Task 3 一次, verify log + output
8. **D3 switch**: ssh ECS Enable Task 3 + Mac launchctl unload `com.moneyprinter.collect`
9. **D4-D7 观察**: 1 周 daily 看 ECS log + commits
10. **D8 stable**: 跑通 → merge feat/ecs-standalone → main, P0 done
11. **P0-B 同样流程** (qfq 是 weekly, 可同时 spec 但实施时间错开)

## 关于今天 6/2 的特殊情况

- **6/2 17:00 仍由 Mac 端跑 daily_report** (P0 还没切换)
- 6/2 14:30 OOS task 已经 plan 好 (gate 仍 require user pre-approval)
- P0 是 "spec + 写代码 + 测试" 阶段, 不动 prod
- 真正切换最早 6/4 (D3), 6/2-6/3 是 spec + dry-run

## 暂不在 P0 scope

- arm_b_monitor / arm_b_report (Phase 1)
- midday / intraday-2pm / execute-preview / papertrade (Phase 2)
- 这些是 daily 真实运行, 但 Mac 关机后不影响 prod 真盘 (它们更多是研究 / 监控)
