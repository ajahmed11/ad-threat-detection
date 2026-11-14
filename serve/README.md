# adserve — Live AD Anomaly Scoring Service

Turns the offline sequence models (LSTM / BiLSTM / LogBERT) into a **real-time
detector**. It tails the Wazuh alert stream, keeps a sliding window of log keys
per host, scores each window with the same **max-surprisal** logic used in
`admodel/evaluate.py`, and emits anomalies as JSON that Wazuh re-ingests to
drive the existing `Disable-ADAccount` Active Response.

This is the proposal's *"secondary integration"* delivered as a deployable
service, not a design.

```
 Wazuh alerts.json ──tail──▶ featurize ──▶ per-host window ──▶ model (max-surprisal)
                                                                    │ score ≥ threshold
                                                                    ▼
 Active Response ◀── rule 100701 ◀── <localfile> ◀── ml_anomalies.json ◀── emit
```

## Faithful to the research — verified

adserve's scorer was checked against the `admodel/evaluate.py` harness on the
real 192 test windows: **max|difference| = 0.000000, 192/192 identical
decisions** at the trained threshold. It deploys the exact model behaviour, it
does not re-implement it:

- **No architecture drift.** The service loads the *whole* `nn.Module` you
  trained (`admodel.models.MaskedRNN` / `LogBERT`).
- **Identical log key.** The featurizer rebuilds the pipeline's **5-part
  composite key** — `event_id|logon_type|ticket_enc|preauth|status_class` —
  from each alert's `data.win` fields, mirroring `adpipe/parse_archives.py` and
  `build_dataset.make_log_key`. The spec lives in the bundle's `meta.json`, so
  inference matches training by construction. (It is **not** the Wazuh rule id.)
- **Identical vocab.** It reuses the frozen training vocab (list form, index =
  id, built from the train split only); unseen keys → `<unk>`.
- **Identical scoring.** Mask each position, take the single most anomalous
  (max surprisal), compare to the F1-optimal threshold frozen at train time.
- **Advisory, not autonomous.** It raises a signal; the SIEM rule decides
  whether to contain. The ML engine stays auditable.

## 1. Build a serving bundle

Your training harness scored models in memory without saving them, so there is
no checkpoint to package. `make_bundle.py` trains the chosen model (seed 0),
captures the F1-optimal threshold exactly as `evaluate.py` does, and writes the
bundle in one step. **Run it from your admodel project root** (where
`import admodel` works and the adpipe dataset sits beside it):

```powershell
# copy make_bundle.py next to run_all.py, then:
cd D:\Final_Year\CSP\Lab_Project\data\raw\admodel
python make_bundle.py bilstm
```

It writes `serve_bundle/` containing:
- `model.pt` — the whole trained BiLSTM module
- `vocab.json` — the frozen training vocabulary
- `meta.json` — threshold, window size, mask id, the featurizer key spec, and
  `admodel_path` (so the service can unpickle the module)

Swap `bilstm` for `lstm` or `logbert` to bundle a different model. (LogBERT is
served on masked-surprisal only; its hypersphere term is not applied live.)

## 2. Demo offline (no live attack) — replay mode

Point it at a historical alert capture and watch it score:

```powershell
cd D:\Final_Year\CSP\ad-threat-detection\ad-threat-detection\serve
python -m adserve.service `
    --bundle-dir D:\Final_Year\CSP\Lab_Project\data\raw\admodel\serve_bundle `
    --replay <your_nov9_alerts.json> `
    --out anoms.json
Get-Content anoms.json | ConvertFrom-Json | Format-List
```

Use your **real** Wazuh capture (e.g. the Nov 9 `alerts.json`), not synthetic
data — the models judge event *sequences*, so only realistic traffic separates
cleanly. On the real test set, benign windows stay below threshold 83.8% of the
time and the DCSync burst scores highest.

## 3. Run live on the SIEM node

```bash
python -m adserve.service \
    --bundle-dir /opt/.../serve_bundle \
    --alerts /var/ossec/logs/alerts/alerts.json \
    --out    /var/ossec/logs/ml_anomalies.json
```

As a managed daemon: `deploy/adserve.service` (systemd). Logs via
`journalctl -u adserve -f`.

## 4. Close the loop into Wazuh

Add the `<localfile>` from `deploy/ossec-localfile-snippet.xml` so the manager
tails `ml_anomalies.json`; the rules in that file lift the score and
(optionally) bind the existing `disable-ad-account` command to the
high-severity ML rule. A model-raised anomaly then flows through the same
containment path as a rule-based detection.

## Emitted record

```json
{
  "timestamp": "2026-11-14T12:00:00+00:00",
  "integration": "adserve",
  "model": "bilstm",
  "host": "DC01",
  "anomaly_score": 13.82,
  "threshold": 9.68,
  "severity": "medium",
  "window": ["4624|2|na|na|na", "4769|na|0x17|na|ok", "4662|na|na|na|na", "..."],
  "detail": { "peak_position": 7, "peak_token": "<unk>",
              "per_position_surprisal": [ ... ] }
}
```

`detail` names the single event that drove the score — triage context, and a
concrete talking point for the demo.

## Configuration

Every setting takes a CLI flag or an `ADSERVE_*` env var (see `config.py`):
`--bundle-dir`, `--alerts`, `--out`, `--threshold`, `--window-size`,
`--host-field`, `--device`. The composite-key spec comes from the bundle's
`meta.json`, so you don't configure it by hand.

## Tests

```bash
pip install -r requirements-serve.txt
set PYTHONPATH=.   &&   python -m pytest adserve/tests -q   # 7 tests
```

The suite uses a tiny stand-in model and list-form vocab, so it runs in CI
without the trained weights.

## Limitations (state honestly in the report)

- Scoring quality is bounded by the offline models: strong on burst attacks
  (DCSync), weaker on single-event signatures where the rule layer wins. The
  service deploys the models, it does not change them.
- The live log key must match the training definition; the featurizer enforces
  this from `meta.json`, but a pipeline change means rebuilding the bundle.
- Upstream telemetry gaps (e.g. the Sysmon 17/18 / 5145 forwarding gap noted as
  a threat to validity) limit what any downstream consumer, this service
  included, can see.
