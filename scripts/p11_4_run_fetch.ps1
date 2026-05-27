# Detached wrapper for P11-4 fetch. Designed to be run via Start-Process
# on ECS so the python process keeps running after SSH session closes.
# Usage (from ECS PowerShell):
#   cd C:\money-printer
#   Start-Process powershell -ArgumentList "-File","scripts\p11_4_run_fetch.ps1" -WindowStyle Hidden
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location C:\money-printer
$logDir = "C:\money-printer\data\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$outLog = "$logDir\p11_4_fetch.out.log"
$errLog = "$logDir\p11_4_fetch.err.log"
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $outLog -Value "=== p11_4_run_fetch.ps1 START $ts ==="
& "C:\money-printer\.venv\Scripts\python.exe" `
  -X utf8 `
  scripts\p11_4_fetch_intraday.py `
  --start 20250901 --end 20260430 `
  >> $outLog 2>> $errLog
$ts2 = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $outLog -Value "=== p11_4_run_fetch.ps1 END $ts2 (exit=$LASTEXITCODE) ==="
