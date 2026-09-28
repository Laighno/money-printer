# QMT login watcher v3 (2026-09-28).
#
# v3 STOPS AUTO-LOGIN. The 9/25 broker client update added an image CAPTCHA
# ("input the calculation result") to the login dialog -- that is what turned
# the 624x419 dialog into 624x445. v2 kept clicking at the old coordinates,
# typing the trade password into what is now the ACCOUNT box, and pressing a
# login button that can never succeed without the captcha: 730 blind attempts
# over 3.5 days (9/25-9/28), three trading days lost, and the trade password
# repeatedly typed into an unverified field.
#
# Defeating an image captcha would mean circumventing a control the broker put
# there on purpose, so v3 does not try. It now only DETECTS and ESCALATES:
# when the bridge heartbeat is stale and a QMT window is up, it sends one
# Feishu alert telling the user to RDP in and log in by hand, then backs off.
#
# What it still does automatically: a bare ENTER on a LARGE (main terminal)
# window, which only dismisses a modal upgrade/notice prompt -- no credentials
# are typed anywhere, ever.
#
# Runs resident in the interactive session via HKCU Run.
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
$AlertF = "C:\money-printer\data\logs\.qmt_login_alert_sent"
$AlertCooldownMin = 60

function Log([string]$m) {
    Add-Content -Path $LogF -Value ((Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $m)
}

function Click([int]$x, [int]$y) {
    [W]::SetCursorPos($x, $y) | Out-Null
    Start-Sleep -Milliseconds 200
    [W]::mouse_event(2, 0, 0, 0, [UIntPtr]::Zero)   # LEFTDOWN
    [W]::mouse_event(4, 0, 0, 0, [UIntPtr]::Zero)   # LEFTUP
}

function Send-Alert([string]$md) {
    # One alert per $AlertCooldownMin so a multi-day outage does not spam, but
    # every new trading morning still gets a fresh nudge.
    try {
        if (Test-Path $AlertF) {
            $last = (Get-Item $AlertF).LastWriteTime
            if (((Get-Date) - $last).TotalMinutes -lt $AlertCooldownMin) { return $false }
        }
        & C:\money-printer\.venv\Scripts\python.exe -X utf8 -c "import sys; sys.path.insert(0, r'C:\money-printer'); from scripts.daily_report import send_to_feishu; send_to_feishu(sys.argv[1])" $md 2>&1 | Out-Null
        Set-Content -Path $AlertF -Value (Get-Date -Format s)
        return $true
    } catch {
        Log ("alert send failed: " + $_.Exception.Message)
        return $false
    }
}

Log "watcher v3 started (pid $PID, session $([System.Diagnostics.Process]::GetCurrentProcess().SessionId))"

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

    # Small window = the login dialog. Since the 9/25 client update it carries
    # an image captcha, so there is nothing safe to automate here: escalate.
    if ($wdt -ge 450 -and $wdt -le 1600 -and $hgt -ge 300 -and $hgt -le 1050) {
        $sent = Send-Alert ("RED - QMT needs a MANUAL login`n`n" +
            "The bridge heartbeat is stale and a ${wdt}x${hgt} login dialog is up on ECS. " +
            "Since the 2026-09-25 client update the dialog has an image captcha, so the " +
            "watcher can no longer log in for you (it will NOT type the password blindly).`n`n" +
            "RDP to 14.103.49.51, enter the trade password + captcha, then start the MONEY " +
            "strategy under Model Trading. 9:25 execution aborts while the bridge is down.")
        if ($sent) {
            Log ("login dialog ${wdt}x${hgt} -> Feishu alert sent (manual login required; captcha since 2026-09-25)")
        } else {
            Log ("login dialog ${wdt}x${hgt} -> alert suppressed (cooldown)")
        }
    } else {
        # Large window: a modal upgrade/notice prompt. ENTER dismisses it and
        # types no credentials, so this stays automatic.
        $ws.SendKeys("{ENTER}")
        Log ("non-login window ${wdt}x${hgt} -> bare ENTER (modal confirm)")
    }
    Start-Sleep -Seconds 300
}
