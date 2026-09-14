# QMT auto-login nudger (2026-09-14). Runs resident in the Administrator
# INTERACTIVE session (registered under HKCU Run, same session as XtItClient).
#
# Problem: Guojin QMT has no "auto login" checkbox. Password is remembered, but
# after the client auto-restarts (e.g. weekly upgrade) it sits on the login
# dialog waiting for one Enter keypress, and the bridge strategy stays dead.
#
# Loop: if bridge heartbeat is stale (>180s) AND XtItClient process exists,
# activate its window and send ENTER (login dialog -> logs in; a modal upgrade
# prompt -> confirms it; the plain main window ignores a stray Enter). Then
# wait 5 min for login + strategy autorun before re-checking.
#
# Limits (honest): SendKeys needs this script to live in the interactive
# session. It keeps working while the RDP session is disconnected, but NOT if
# the session is fully logged off (Run key restarts it on next logon anyway).
# The 09:10 heartbeat alarm remains the safety net.
$ErrorActionPreference = "SilentlyContinue"
Add-Type -AssemblyName Microsoft.VisualBasic
$ws = New-Object -ComObject WScript.Shell
$LogF = "C:\money-printer\data\logs\qmt_autologin.log"
$HbF = "C:\money-printer\data\bridge\heartbeat.json"

function Log([string]$m) {
    Add-Content -Path $LogF -Value ((Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m)
}

Log "watcher started (pid $PID, session $([System.Diagnostics.Process]::GetCurrentProcess().SessionId))"

while ($true) {
    Start-Sleep -Seconds 90
    $stale = $true
    if (Test-Path $HbF) {
        try {
            $hb = Get-Content $HbF -Raw | ConvertFrom-Json
            $age = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [long]$hb.ts
            if ($age -lt 180) { $stale = $false }
        } catch { }
    }
    if (-not $stale) { continue }

    $p = Get-Process XtItClient -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $p) { continue }   # client not running; Run-key autostart handles next logon

    try {
        [Microsoft.VisualBasic.Interaction]::AppActivate($p.Id)
        Start-Sleep -Milliseconds 800
        $ws.SendKeys("{ENTER}")
        Log ("heartbeat stale -> sent ENTER to XtItClient pid " + $p.Id)
    } catch {
        Log ("AppActivate/SendKeys failed: " + $_.Exception.Message)
    }
    Start-Sleep -Seconds 300   # give login + strategy autorun time
}
