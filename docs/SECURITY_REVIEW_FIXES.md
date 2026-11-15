# Security review - fixes and verification

| # | Finding | Fix | Where | Verified by |
|---|---|---|---|---|
| P0 | Protected-account check bypass (`TrimEnd('$')` before compare) | Machine accounts rejected before any normalisation; protection by RID/adminCount/recursive group membership on the *resolved* object | `response/disable-ad-account.ps1` | `tests/test_response_script.py` + manual DC01 test below |
| P0 | Unvalidated identity passed to `Disable-ADAccount` | SAM regex, exactly-one-object resolution, act by ObjectGUID | same | static tests + manual |
| P0 | Over-broad AR binding | Allow-list inside script; manager binds only 100514; dry-run default | script + `ossec-manager-snippet.xml` | static tests |
| P1 | `torch.load(weights_only=False)` | state_dict + `weights_only=True` + fixed arch registry + out-of-band SHA-256 | `serve/adserve/scorer.py`, `serve/make_bundle.py` | `test_loader_rejects_*`, `test_sha256_pin_enforced` |
| P1 | `sys.path.insert` from `meta.json` | Removed; `PYTHONPATH` fixed in the systemd unit | scorer, `adserve.service` | `test_meta_path_cannot_change_sys_path` |
| P1 | Spray parent rule 60103 | **Resolved: the DEPLOYED manager ruleset already used parent `60122`** (logon-failure), not 60103. The repo file had drifted from production; it now mirrors the deployed rules. | `local_rules.xml` | real alert history (below) |
| P1 | DCSync machine-account FPs | `subjectUserName` must not end in `$`. **Confirmed in real data: 8 of 92 100514 alerts were `DC01$`.** Filter deployed and live-tested on DC01 on 2026-11-16 (dry-run). | `local_rules.xml` | alert history + live DC01 test |
| P2 | Unbounded log reads / hosts / output | 1 MiB line cap, bounded chunked reads, host cap, size-rotated output | `tailer.py`, `windows.py`, `emit.py` | adserve tests |
| P2 | CI didn't cover `serve/` | compileall of everything, serve tests, least-privilege `permissions` | `.github/workflows/ci.yml` | CI |
| P2 | Broad systemd write access | dedicated `/var/log/adserve`, ProtectHome, empty capability set, AF_UNIX only, etc. | `adserve.service` | `systemd-analyze security adserve` |
| P3 | README overclaims | Dataset/ML-response statements corrected; `docs/REPRODUCING.md` | READMEs | review |
| P3 | `allow_pickle=True`, unescaped dashboard JSON | `allow_pickle=False` (+ migration script); `<`,`>`,`&` escaped in embedded JSON | `admodel/data.py`, `make_dashboard.py` | manual |
| Design | ML anomaly -> AR integration incomplete | Documented as **advisory only**; anomaly records have no account fields; AR not bound to 100700/100701 | `serve/` | `test_emitter_rotates_and_has_no_windows_fields` |

## Resolved (rule 100513 / 100515 parent, and the DCSync FP rate)
`wazuh-logtest` does not reproduce the eventchannel parent-matching chain reliably for these
rules (it returns rule 1002 / no match against the live ruleset even though the same rules
fire correctly in production), so this was validated against **real alert history** instead
(lab, Nov 2026):

* 100513 (failed logon): 164 alerts, all event 4625 -> confirms parent `60122` works.
* 100515 (spraying): 19 alerts.
* 100514 (DCSync): 92 alerts - 84 from `adm.rthapa` (attack account, a Domain Admin),
  **8 from `DC01$`** (legitimate replication traffic) - an 8.7% false-positive rate before
  the filter below.

The repo's `local_rules.xml` had drifted from the deployed ruleset (it used a widened-but-wrong
`60103,60104,60105` parent for 100513); it has been corrected to mirror the deployed rules,
which already used the correct `60122` parent.

## DCSync machine-account filter: deploy and verify
```bash
cp /var/ossec/etc/rules/local_rules.xml /var/ossec/etc/rules/local_rules.xml.bak-pre-dcsync-filter
# apply the <field negate="yes"> exclusion to rule 100514 (see local_rules.xml), then:
/var/ossec/bin/wazuh-analysisd -t && systemctl restart wazuh-manager
# re-run secretsdump.py -just-dc as the attack account: 100514 must still fire
# later: grep '"100514"' alerts.json | grep -c 'DC01\$'  should not increase
```

## DC01 safety test (dry-run, live-tested 2026-11-16)
No `enforce.flag` was present, so the script only logged what it would do - no AD object was
changed in any of these three tests:

| Input `subjectUserName` | Result |
|---|---|
| `DC01$` (machine account) | `REJECT: 'DC01$' is a machine account - not actioned` |
| `adm.rthapa` (Domain Admins, adminCount=1) | `REJECT: 'adm.rthapa' is a protected account (low RID / adminCount / protected group) - not actioned` |
| `ram` (ordinary enabled user) | `DRY-RUN: would disable account 'ram' (rule 100514) - enforce.flag not present, no change made` |

**Limitation worth noting in the report:** the attack account used in this lab (`adm.rthapa`)
is itself a Domain Admin, which is realistic for a DCSync attack (replication rights normally
require privilege) but also means the protected-account check - by design - prevents the
automated response from containing *this* compromise. The containment logic trades the ability
to auto-respond to a privileged-account compromise for safety against false/malicious triggers
against real admins. This is a deliberate design trade-off, not a defect.

Test harness note: piping to a `.ps1` directly within the same PowerShell/WinRM session does
**not** populate `[Console]::In` (that only works for a real OS-level stdin redirection, which
is how Wazuh's Active Response framework actually invokes the script in production). To test
manually, pipe into a child `powershell.exe -File ...` process instead:
```powershell
Get-Content -Raw .\test_alert.json | powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\disable-ad-account.ps1
```
## Redeploying the model bundle
```powershell
cd admodel; python make_bundle.py bilstm      # prints model.pt SHA-256
```
Copy `serve_bundle/` to the SIEM, set `ADSERVE_MODEL_SHA256` in a root-owned systemd drop-in.
