# Wazuh Active Response — automated AD account containment

Closes the **"and Response"** half of the project title: when a detection rule fires,
Wazuh can run `Disable-ADAccount` on DC01 for a **validated DCSync detection (rule 100514)**.
All other detections are alert-only. Containment is **dry-run by default**.

## Files
- `disable-ad-account.ps1` — the response script (reads the alert, disables the account, logs it).
- `disable-ad-account.cmd` — launcher Wazuh calls (pipes the alert JSON into PowerShell).
- `ossec-manager-snippet.xml` — the `<command>` + `<active-response>` blocks for the manager.

## Deploy (one-time)

**1. On DC01** — copy both scripts into the agent's active-response bin:
```
C:\Program Files (x86)\ossec-agent\active-response\bin\disable-ad-account.cmd
C:\Program Files (x86)\ossec-agent\active-response\bin\disable-ad-account.ps1
```
The agent service runs as SYSTEM, which on a DC can run `Disable-ADAccount`. Confirm the
AD module is present: `Get-Command Disable-ADAccount` in an elevated PowerShell.

**2. On the Wazuh manager (wazuh-siem)** — paste the two blocks from `ossec-manager-snippet.xml`
inside `<ossec_config>` in `/var/ossec/etc/ossec.conf`, then:
```bash
xmllint --noout /var/ossec/etc/ossec.conf && systemctl restart wazuh-manager
```

**3. Restart the DC01 agent** (Services → WazuhSvc → Restart, or `Restart-Service WazuhSvc`).

## Demo flow (rehearse this for the viva)

1. Show the account enabled: on DC01, `Get-ADUser adm.rthapa | Select Enabled` → `True`.
2. From Kali, run DCSync:
   ```
   impacket-secretsdump PROJECTLAB.AJ.LOCAL/adm.rthapa:'<pass>'@192.168.100.10 -just-dc
   ```
3. In the Wazuh dashboard, rule **100514 (DCSync)** fires.
4. Within a second or two the Active Response runs. Verify:
   ```powershell
   Get-ADUser adm.rthapa | Select Enabled          # -> False
   Get-Content "C:\Program Files (x86)\ossec-agent\active-response\active-responses.log" -Tail 3
   ```
   The attacker's account is now disabled — containment complete, automatically.

**One-line narration:** *"The attack is detected, and the same platform responds — it disables
the compromised account without any analyst action."*

## Reset between demo runs
```powershell
Enable-ADAccount -Identity adm.rthapa
```

## Safety model (after the security review)
- **Dry-run by default.** Real disabling happens only if
  `C:\Program Files (x86)\ossec-agent\active-response\enforce.flag` exists. For the demo:
  `New-Item "...\active-response\enforce.flag" -ItemType File`; delete it to go back to dry-run.
- **Rule allow-list inside the script** (only 100514 acts); the manager XML binds only 100514 too.
- **Strict identity validation**: SAM-name regex, machine accounts (`$`) rejected *before* any
  normalisation, account resolved to exactly **one** AD user, then acted on by ObjectGUID.
- **Protected identities**: RID < 1000 (Administrator/Guest/krbtgt), `adminCount=1`, and recursive
  membership of Domain/Enterprise/Schema Admins, Administrators and operator groups are all refused.
- Spraying / AS-REP / Kerberoasting never auto-disable: they do not identify one compromised account.

## Notes
- The older line below that says the script protects only a fixed name list is superseded by the above.
- It writes an Application event (ID 9100) on each action, so the **response itself is auditable**
  and shows up in Wazuh — a nice closing-the-loop detail for the demo.
- For spraying (100515) there is no single abusing account; prefer IP-based blocking there,
  or leave it out of Active Response and mention that distinction in the viva (shows judgement).
