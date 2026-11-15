# Reproducing the results

The Git repository holds **code, rules, tests and documentation**. It does *not* hold the raw
Wazuh archives, `windows.npz`, trained models or `results/` (they are large, contain lab
telemetry and are excluded by `.gitignore`). A fresh clone therefore cannot re-run the
experiment on its own - this is intentional and stated here rather than hidden.

## What you need (outside Git)
| Item | Used by | How to provide |
|---|---|---|
| Wazuh archive JSON (2026-Nov-ossec-archive-09, 2026-Oct-ossec-archive-22) | `pipeline/` | folder pointed to by `ADPIPE_RAW_DIR` (`archives_clean/`) |
| `runs_frozen_20261114.jsonl`, `runs_all.jsonl` (attack run manifest) | `pipeline/label_events.py` | same folder |
| Built dataset `data/processed/windows.npz`, `vocab.json` | `models/admodel` | produced by the pipeline below |

Publish the frozen data as a **versioned release asset / archive (e.g. Zenodo or a GitHub
Release)** and record its SHA-256 here once uploaded:

```
windows.npz  sha256: <fill in after upload>
```

## Steps
```powershell
$env:ADPIPE_RAW_DIR = "D:\Final_Year\CSP\raw"      # folder with archives_clean\ and the runs_*.jsonl files
python pipeline/parse_archives.py
python pipeline/label_events.py
python pipeline/build_dataset.py                   # writes data/processed/windows.npz (safe str dtype)
python pipeline/score_rules_windows.py             # rule baseline
python models/run_all.py                           # LSTM / BiLSTM / LogBERT (3 seeds)
python models/run_supervised.py
```
If you hold an older `windows.npz` (pickled object array), convert once with
`python pipeline/convert_npz_safe.py <path>` or simply rebuild it.
`models/admodel/config.py` expects the processed data at `<repo>/data/processed`
(or set the paths there).
