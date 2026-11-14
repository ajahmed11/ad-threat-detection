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

## Why it is faithful to the research

- **No architecture drift.** The service pickles and loads the *whole*
  `nn.Module` you trained (`admodel.models.MaskedRNN` / `LogBERT`) — it never
  re-implements the network.
- **Identical tokenisation.** It reuses the frozen training vocabulary
  (`vocab.json`, built from the train split only) and maps unseen keys to
  `<unk>`, so the live token stream matches the evaluation stream.
- **Identical scoring.** `scorer.score()` is the max-surprisal reduction from
  the evaluation harness: mask each position, take the single most anomalous.
- **Advisory, not autonomous.** The service only raises a signal; the SIEM rule
  decides whether to contain. That keeps the ML engine auditable.

## 1. Build a serving bundle

A bundle is a folder with `model.pt`, `vocab.json`, `meta.json`. Produce one
from a trained checkpoint:

```bash
python export_bundle.py \
    --checkpoint ../models/artifacts/bilstm.pt \
    --vocab      ../models/artifacts/vocab.json \
    --threshold  7.3 \
    --model      bilstm \
    --out        serve_bundle
```

`--threshold` is the decision boundary you froze at train time (e.g. the 99th
percentile of benign-window scores — the same value behind the reported FPR).

If your training saved a `state_dict` rather than a whole module, add
`--state-dict --arch bilstm` and run it from the repo root so `admodel` is
importable.

**Or** write the bundle directly at the end of `train.py`, where the live
objects are already in scope:

```python
import json, torch
from pathlib import Path
out = Path("serve/serve_bundle"); out.mkdir(parents=True, exist_ok=True)
torch.save(model.eval(), out / "model.pt")
(out / "vocab.json").write_text(json.dumps(vocab.token_to_id))
(out / "meta.json").write_text(json.dumps({
    "model": "bilstm", "threshold": thr_99pct,
    "window_size": cfg.window_size, "log_key_field": "rule.id",
}))
```

> **Match the log key.** `log_key_field` must be the *same* field your
> `build_dataset.py` used to form each event token (default `rule.id`). If the
> pipeline keyed on something else (e.g. Windows `data.win.system.eventID`),
> set `--log-key-field` to that dotted path.

## 2. Demo offline (no live attack) — replay mode

Point it at a historical alert capture and watch it score:

```bash
python -m adserve.service \
    --bundle-dir serve_bundle \
    --replay ../data/alerts_2025-11-09.json \
    --out /tmp/anoms.json
cat /tmp/anoms.json | python -m json.tool
```

Ideal for the viva: deterministic, repeatable, and it exercises the exact
production code path.

## 3. Run live on the SIEM node

```bash
python -m adserve.service \
    --bundle-dir serve_bundle \
    --alerts /var/ossec/logs/alerts/alerts.json \
    --out    /var/ossec/logs/ml_anomalies.json
```

As a managed daemon:

```bash
sudo cp deploy/adserve.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now adserve
journalctl -u adserve -f
```

## 4. Close the loop into Wazuh

Add the `<localfile>` from `deploy/ossec-localfile-snippet.xml` so the manager
tails `ml_anomalies.json`, then the rules in that file lift the score and
(optionally) bind the existing `disable-ad-account` command to the
high-severity ML rule. A model-raised anomaly now flows through the same
containment path as a rule-based detection.

## Configuration

Every setting takes a CLI flag or an `ADSERVE_*` env var (see `config.py`).
Key ones: `--bundle-dir`, `--alerts`, `--out`, `--threshold`, `--window-size`,
`--log-key-field`, `--host-field`, `--device`.

## Emitted record

```json
{
  "timestamp": "2026-11-14T12:00:00+00:00",
  "integration": "adserve",
  "model": "bilstm",
  "host": "DC01",
  "anomaly_score": 9.42,
  "threshold": 7.30,
  "severity": "high",
  "window": ["4624", "4769", "4662", "..."],
  "detail": { "peak_position": 12, "peak_token": "4662",
              "per_position_surprisal": [ ... ] }
}
```

The `detail` block gives the analyst the single event that drove the score —
useful triage context, and a concrete talking point for the demo.

## Tests

```bash
pip install -r requirements-serve.txt
python -m pytest adserve/tests -q
```

The suite uses a tiny stand-in model, so it runs in CI without the trained
weights.

## Limitations (state honestly in the report)

- Scoring quality is bounded by the offline models: strong on burst attacks
  (DCSync), weaker on single-event signatures where the rule layer wins. The
  service does not change the models — it deploys them.
- The log key must match the training definition exactly; a mismatch silently
  degrades to `<unk>`-heavy windows.
- Windows forwarding gaps upstream (e.g. the Sysmon 17/18 / 5145 gap noted as a
  threat to validity) limit what any downstream consumer, this service
  included, can see.
