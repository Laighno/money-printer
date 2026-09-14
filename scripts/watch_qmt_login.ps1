# QMT auto-login watcher v2 (2026-09-14).
#
# v1 only pressed ENTER -- works when the password is still remembered. But a
# Windows sign-out wipes QMT's remembered password (observed in rehearsal), so
# v2 performs the full login: click password box -> type password (DPAPI
# decrypted from C:\guojin\.tradepwd, created BY THE USER with
#   Read-Host -AsSecureString | ConvertFrom-SecureString | Set-Content ...
# so the plaintext never leaves this machine) -> click login button.
#
# Login window vs main window is told apart by WINDOW SIZE: the login dialog is
# a fixed small window (~1200x790); the main terminal is much larger. On a big
# window we only send a bare ENTER (confirms a modal upgrade prompt, harmless
# otherwise). Runs resident in the interactive session via HKCU Run.
$ErrorActionPreference = "SilentlyContinue"
Add-Type -AssemblyName Microsoft.VisualBasic
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class W {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint dw, UIntPtr ex);
  public struct RECT { public int L; public int T; public int R; public int B; }
}
"@
$ws = New-Object -ComObject WScript.Shell
$LogF = "C:\money-printer\data\logs\qmt_autologin.log"
$HbF = "C:\money-printer\data\bridge\heartbeat.json"
$PwdF = "C:\guojin\.tradepwd"

function Log([string]$m) {
    Add-Content -Path $LogF -Value ((Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m)
}

function Click([int]$x, [int]$y) {
    [W]::SetCursorPos($x, $y) | Out-Null
    Start-Sleep -Milliseconds 200
    [W]::mouse_event(2, 0, 0, 0, [UIntPtr]::Zero)   # LEFTDOWN
    [W]::mouse_event(4, 0, 0, 0, [UIntPtr]::Zero)   # LEFTUP
}

function Get-TradePwd {
    if (-not (Test-Path $PwdF)) { return $null }
    try {
        $ss = Get-Content $PwdF | ConvertTo-SecureString
        $b = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($ss)
        $p = [Runtime.InteropServices.Marshal]::PtrToStringAuto($b)
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b)
        return $p
    } catch { Log ("pwd decrypt failed: " + $_.Exception.Message); return $null }
}

function Esc-SendKeys([string]$s) {
    # brace-wrap every char so SendKeys metacharacters (+^%~(){}) are literal
    ($s.ToCharArray() | ForEach-Object {
        if ($_ -eq "{") { "{{}" } elseif ($_ -eq "}") { "{}}" } else { "{" + $_ + "}" }
    }) -join ""
}

Log "watcher v2 started (pid $PID, session $([System.Diagnostics.Process]::GetCurrentProcess().SessionId))"

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

    $p = Get-Process XtItClient -ErrorAction SilentlyContinue |
         Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
    if (-not $p) { continue }

    $r = New-Object "W+RECT"
    [W]::GetWindowRect($p.MainWindowHandle, [ref]$r) | Out-Null
    $wdt = $r.R - $r.L
    $hgt = $r.B - $r.T

    try { [Microsoft.VisualBasic.Interaction]::AppActivate($p.Id) } catch { }
    Start-Sleep -Milliseconds 600

    if ($wdt -ge 700 -and $wdt -le 1600 -and $hgt -ge 450 -and $hgt -le 1050) {
        # login dialog geometry
        $pwd = Get-TradePwd
        if ($pwd) {
            # password box center ~ (50% w, 70% h); login button ~ (38% w, 85% h)
            Click ($r.L + [int]($wdt * 0.50)) ($r.T + [int]($hgt * 0.70))
            Start-Sleep -Milliseconds 400
            $ws.SendKeys((Esc-SendKeys $pwd))
            Start-Sleep -Milliseconds 400
            Click ($r.L + [int]($wdt * 0.38)) ($r.T + [int]($hgt * 0.85))
            Log ("login sequence executed on ${wdt}x${hgt} window")
            $pwd = $null
        } else {
            $ws.SendKeys("{ENTER}")
            Log ("no pwd file -> ENTER only (${wdt}x${hgt})")
        }
    } else {
        $ws.SendKeys("{ENTER}")
        Log ("non-login window ${wdt}x${hgt} -> bare ENTER (modal confirm)")
    }
    Start-Sleep -Seconds 300
}
