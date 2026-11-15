<#
  Wazuh Active Response - guarded containment for AD identity attacks.

  Runs on DC01 (location "local"), reads the alert JSON from stdin and, ONLY for
  rules explicitly allow-listed below, disables one validated, non-privileged
  AD *user* account.

  Design rules (see docs/SECURITY_REVIEW_FIXES.md):
    1. The alert is untrusted input. Nothing from it reaches an AD cmdlet until
       it has passed strict validation AND resolved to exactly one user object.
    2. Machine accounts (trailing '$') are rejected BEFORE any normalisation.
    3. Privileged / critical identities are protected by SID/RID, adminCount and
       (recursive) group membership - never by name alone.
    4. The rule allow-list lives HERE, not only in the manager XML.
    5. Only DCSync (100514) auto-contains. Every other rule is alert-only.
    6. DRY-RUN is the default. Create the file  <ar-dir>\enforce.flag  to enable
       real containment (explicit operator decision, easy to revoke).

  Placed in:  C:\Program Files (x86)\ossec-agent\active-response\bin\
  Launched by: disable-ad-account.cmd
#>
$ErrorActionPreference = "Stop"
$arRoot  = "C:\Program Files (x86)\ossec-agent\active-response"
$log     = Join-Path $arRoot "active-responses.log"
$enforce = Test-Path (Join-Path $arRoot "enforce.flag")   # absent => dry-run

# rule id -> field that names the account to contain ($null = alert-only).
# DCSync: the actor (subject) is the account that used replication rights.
# AS-REP / Kerberoast / spraying identify TARGETS or many accounts, which is not
# evidence that one specific account is compromised -> never auto-disable.
$RuleActions = @{
    "100514" = "subjectUserName"
    "100510" = $null
    "100511" = $null
    "100512" = $null
    "100515" = $null
}

# SAM account name: letters/digits . _ - only, 1-64 chars, must NOT end in '$'.
$SamPattern = '^[A-Za-z0-9._-]{1,64}$'

# Groups whose members must never be auto-disabled (well-known RIDs / SIDs).
$ProtectedGroupRids = @(512, 518, 519, 520, 544, 548, 549, 550, 551, 498, 521)  # DA, Schema, Enterprise, GPO owners, Admins, Operators, ERODC

function Write-Log([string]$msg) {
    # strip control chars so alert-supplied text cannot forge log lines
    $safe = ($msg -replace '[\x00-\x1F\x7F]', ' ')
    if ($safe.Length -gt 400) { $safe = $safe.Substring(0, 400) }
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Add-Content -Path $log -Value "$ts disable-ad-account: $safe"
}

function Test-Protected($adUser, $domainSid) {
    # Returns a reason string if the account must not be touched, else $null.
    $rid = [int]($adUser.SID.Value.Split('-')[-1])
    if ($rid -lt 1000) { return "built-in/well-known RID $rid" }          # 500 Administrator, 501 Guest, 502 krbtgt...
    if ($adUser.adminCount -eq 1) { return "adminCount=1" }               # a signal, NOT sufficient on its own
    if ($adUser.SamAccountName -eq $env:USERNAME) { return "self" }
    # recursive membership via LDAP_MATCHING_RULE_IN_CHAIN against protected groups
    foreach ($grid in $ProtectedGroupRids) {
        $gsid = if ($grid -ge 544 -and $grid -le 551) { "S-1-5-32-$grid" } else { "$domainSid-$grid" }   # BUILTIN vs domain groups
        try {
            $g = Get-ADGroup -Identity $gsid -ErrorAction Stop
            $hit = Get-ADUser -LDAPFilter "(&(objectSid=$($adUser.SID.Value))(memberOf:1.2.840.113556.1.4.1941:=$($g.DistinguishedName)))" -ErrorAction Stop
            if ($hit) { return "member of protected group $($g.Name)" }
        } catch { continue }   # group not present in this domain
    }
    return $null
}

try {
    $raw = [Console]::In.ReadToEnd()
    if (-not $raw -or $raw.Length -gt 262144) { Write-Log "no/oversized stdin - ignored"; exit 1 }
    $msg = $raw | ConvertFrom-Json

    $command = [string]$msg.command
    if ($command -notin @("add", "delete")) { Write-Log "unknown command '$command' - ignored"; exit 1 }

    $alert  = $msg.parameters.alert
    $ruleId = [string]$alert.rule.id

    # --- 1. rule allow-list (defence in depth: independent of ossec.conf) ----
    if (-not $RuleActions.ContainsKey($ruleId)) { Write-Log "rule '$ruleId' not allow-listed - ignored"; exit 0 }
    $field = $RuleActions[$ruleId]
    if (-not $field) { Write-Log "rule $ruleId is ALERT-ONLY (analyst review required) - no action"; exit 0 }

    # --- 2. extract + validate the identity ---------------------------------
    $user = [string]$alert.data.win.eventdata.$field
    if (-not $user) { Write-Log "rule $ruleId: no '$field' in alert"; exit 1 }
    $user = $user.Trim()

    # accept DOMAIN\sam, reject everything else exotic (UPN, DN, SID, wildcard)
    if ($user -match '^[A-Za-z0-9._-]{1,15}\\(.+)$') { $user = $Matches[1] }

    if ($user.EndsWith('$'))        { Write-Log "rule $ruleId: REJECT machine account '$user'"; exit 0 }
    if ($user -notmatch $SamPattern){ Write-Log "rule $ruleId: REJECT unsafe identity '$user'"; exit 1 }

    # --- 3. resolve to EXACTLY ONE user object (no -Identity from raw text) --
    Import-Module ActiveDirectory -ErrorAction Stop
    $domainSid = (Get-ADDomain).DomainSID.Value
    $found = @(Get-ADUser -Filter "SamAccountName -eq '$user'" -Properties adminCount, Enabled -ErrorAction Stop)
    if ($found.Count -ne 1) { Write-Log "rule $ruleId: '$user' resolved to $($found.Count) objects - abort"; exit 1 }
    $adUser = $found[0]

    # --- 4. protected-identity check on the RESOLVED object ------------------
    $why = Test-Protected $adUser $domainSid
    if ($why) { Write-Log "rule $ruleId: REFUSE '$user' ($why)"; exit 0 }

    # --- 5. act (by ObjectGUID of the resolved object, not by text) ----------
    if ($command -eq "add") {
        if (-not $adUser.Enabled) { Write-Log "'$user' already disabled - no action"; exit 0 }
        if (-not $enforce) { Write-Log "DRY-RUN: would disable '$user' (rule $ruleId). Create enforce.flag to enable."; exit 0 }
        Disable-ADAccount -Identity $adUser.ObjectGUID.ToString() -ErrorAction Stop
        Write-Log "CONTAINED: disabled '$user' (rule $ruleId)"
        $src = "WazuhActiveResponse"
        if (-not [System.Diagnostics.EventLog]::SourceExists($src)) {
            New-EventLog -LogName Application -Source $src -ErrorAction SilentlyContinue
        }
        Write-EventLog -LogName Application -Source $src -EventId 9100 -EntryType Warning `
            -Message "Active Response: disabled AD account '$user' after Wazuh rule $ruleId." -ErrorAction SilentlyContinue
    }
    else {  # delete = rollback, same validation applies
        if (-not $enforce) { Write-Log "DRY-RUN: would re-enable '$user'"; exit 0 }
        Enable-ADAccount -Identity $adUser.ObjectGUID.ToString() -ErrorAction Stop
        Write-Log "ROLLBACK: re-enabled '$user'"
    }
    exit 0
}
catch {
    Write-Log "ERROR: $($_.Exception.Message)"
    exit 1
}
