<#
  Wazuh Active Response - automated containment for AD identity attacks.
  Triggered by the Wazuh manager when a detection rule fires; runs on DC01
  (location "local"), reads the alert JSON from stdin, extracts the abusing
  account, and disables it with Disable-ADAccount.

  Placed in:  C:\Program Files (x86)\ossec-agent\active-response\bin\
  Launched by: disable-ad-account.cmd
#>
$ErrorActionPreference = "Stop"
$log = "C:\Program Files (x86)\ossec-agent\active-response\active-responses.log"

function Write-Log($msg) {
    $ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
    Add-Content -Path $log -Value "$ts disable-ad-account: $msg"
}

try {
    # Wazuh sends the alert as one JSON message on stdin
    $raw = [Console]::In.ReadToEnd()
    if (-not $raw) { Write-Log "no stdin received"; exit 1 }
    $msg = $raw | ConvertFrom-Json
    $command = $msg.command                      # "add" (contain) or "delete" (rollback)
    $alert   = $msg.parameters.alert
    $ruleId  = $alert.rule.id
    $ed      = $alert.data.win.eventdata

    # Pick the abusing account. DCSync/Kerberoast/AS-REP identify it in different fields;
    # prefer the subject (actor), fall back to the target.
    $user = $ed.subjectUserName
    if (-not $user -or $user -eq "-") { $user = $ed.targetUserName }
    if (-not $user) { Write-Log "rule $ruleId fired but no username in alert"; exit 1 }

    # sanitise: strip a trailing '$' (machine accounts) and any domain prefix
    $user = $user.Trim().TrimEnd('$')
    if ($user -match '\\') { $user = $user.Split('\')[-1] }

    # safety: never disable built-in / critical accounts
    $protected = @("Administrator","krbtgt","DC01$","Guest")
    if ($protected -contains $user) { Write-Log "refusing to disable protected account '$user'"; exit 0 }

    if ($command -eq "add") {
        Disable-ADAccount -Identity $user -ErrorAction Stop
        Write-Log "CONTAINED: disabled '$user' in response to rule $ruleId"
        # leave a Windows event so the response itself is auditable / shows in Wazuh
        $src = "WazuhActiveResponse"
        if (-not [System.Diagnostics.EventLog]::SourceExists($src)) {
            New-EventLog -LogName Application -Source $src -ErrorAction SilentlyContinue
        }
        Write-EventLog -LogName Application -Source $src -EventId 9100 -EntryType Warning `
            -Message "Active Response: disabled AD account '$user' after Wazuh rule $ruleId." -ErrorAction SilentlyContinue
    }
    elseif ($command -eq "delete") {
        Enable-ADAccount -Identity $user -ErrorAction Stop
        Write-Log "ROLLBACK: re-enabled '$user'"
    }
    exit 0
}
catch {
    Write-Log "ERROR: $($_.Exception.Message)"
    exit 1
}
