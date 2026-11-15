# Security review - fixes and verification

| # | Finding | Fix | Where | Verified by |
|---|---|---|---|---|
| P0 | Protected-account check bypass (`TrimEnd('$')` before compare) | Machine accounts rejected before any normalisation; protection by RID/adminCount/recursive group membership on the *resolved* object | `response/disable-ad-account.ps1` | `tests/test_response_script.py` + manual DC01 test below |
| P0 | Unvalidated identity passed to `Disable-ADAccount` | SAM regex, exactly-one-object resolution, act by ObjectGUID | same | static tests + manual |
| P0 | Over-broad AR binding | Allow-list inside script; manager binds only 100514; dry-run default | script + `ossec-manager-snippet.xml` | static tests |
| P1 | `torch.load(weights_only=False)` | state_dict + `weights_only=True` + fixed arch registry + out-of-band SHA-256 | `serve/adserve/scorer.py`, `serve/make_bundle.py` | `test_loader_rejects_*`, `test_sha256_pin_enforced` |
| P1 | `sys.path.insert` from `meta.json` | Removed; `PYTHONPATH` fixed in the systemd unit | scorer, `adserve.service` | `test_meta_path_cannot_change_sys_path` |
| P1 | Spray parent rule 60103 | Parent widened to `60103,60104,60105`; **must be confirmed with wazuh-logtest** | `local_rules.xml` | logtest (below) |
| P1 | DCSync machine-account FPs | `subjectUserName` must not end in `$` | `local_rules.xml` | logtest (below) |
| P2 | Unbounded log reads / hosts / output | 1 MiB line cap, bounded chunked reads, host cap, size-rotated output | `tailer.py`, `windows.py`, `emit.py` | adserve tests |
| P2 | CI didn't cover `serve/` | compileall of everything, serve tests, least-privilege `permissions` | `.github/workflows/ci.yml` | CI |
| P2 | Broad systemd write access | dedicated `/var/log/adserve`, ProtectHome, empty capability set, AF_UNIX only, etc. | `adserve.service` | `systemd-analyze security adserve` |
| P3 | README overclaims | Dataset/ML-response statements corrected; `docs/REPRODUCING.md` | READMEs | review |
| P3 | `allow_pickle=True`, unescaped dashboard JSON | `allow_pickle=False` (+ migration script); `<`,`>`,`&` escaped in embedded JSON | `admodel/data.py`, `make_dashboard.py` | manual |
| Design | ML anomaly -> AR integration incomplete | Documented as **advisory only**; anomaly records have no account fields; AR not bound to 100700/100701 | `serve/` | `test_emitter_rotates_and_has_no_windows_fields` |

## A point to reconcile before the viva (rule 100513 / 100515)
The review says 4625 events cannot match parent 60103. Your own telemetry shows rule 100515
(password spraying) firing 19 times, so in *your* Wazuh version the chain evidently matched.
Wazuh's classification of 4625 differs between versions, so do not take either claim on trust:
run `wazuh-logtest` on a real 4625 event and record which parent rule it hits. The widened
parent list is safe either way, and the logtest output becomes dissertation evidence.

## Verification commands (run on the Wazuh manager)
```bash
sudo cp detections/wazuh/local_rules.xml /var/ossec/etc/rules/local_rules.xml
sudo /var/ossec/bin/wazuh-logtest          # paste a real 4625 / 4662 JSON event (win.* fields)
#   expect: 4625 -> 100513 (then 100515 after 8 distinct users/120 s)
#   expect: 4662 + replication GUID + subjectUserName "attacker.user" -> 100514
#   expect: 4662 + replication GUID + subjectUserName "DC01$"          -> NOT 100514
sudo systemctl restart wazuh-manager
```
## DC01 safety test (dry-run first)
```powershell
# 1. no enforce.flag => script only logs DRY-RUN lines
# 2. feed it a fake alert for a machine account and a Domain Admin; both must be REFUSED
'{"command":"add","parameters":{"alert":{"rule":{"id":"100514"},"data":{"win":{"eventdata":{"subjectUserName":"DC01$"}}}}}}' |
  powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Program Files (x86)\ossec-agent\active-response\bin\disable-ad-account.ps1"
Get-Content "C:\Program Files (x86)\ossec-agent\active-response\active-responses.log" -Tail 3   # REJECT machine account
```
## Redeploying the model bundle
```powershell
cd admodel; python make_bundle.py bilstm      # prints model.pt SHA-256
```
Copy `serve_bundle/` to the SIEM, set `ADSERVE_MODEL_SHA256` in a root-owned systemd drop-in.
