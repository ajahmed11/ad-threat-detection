# Active Directory Identity-Attack Detection — Rules vs Deep Learning

![CI](https://github.com/USER/ad-threat-detection/actions/workflows/ci.yml/badge.svg)

A controlled, reproducible study comparing **custom Wazuh SIEM rules** with **LSTM, BiLSTM
and LogBERT** sequence models for detecting Active Directory identity attacks — with
**ATT&CK-mapped, detection-as-code** rules and **automated response**.

> Level 6 Cyber Security Project (UFCE8A-30-3, UWE Bristol). CyBOK: SOIM (primary), AAA (secondary).

## Headline result

Custom rules decisively outperform unsupervised deep sequence models for identity attacks
whose signature is a single field value; the models only reliably detect multi-event
(burst) attacks such as DCSync. This is a nuanced, reproducible finding — not "deep learning
wins" — backed by a leakage-free dataset and per-attack analysis.

| Detector | F1 | Recall | FPR | ROC-AUC |
| --- | --- | --- | --- | --- |
| Custom Wazuh rules | — | 45–88% (per attack) | low | — |
| LogBERT (best model) | 0.27 | 0.36 | 0.10 | 0.64 |
| LSTM | 0.23 | 0.33 | 0.12 | 0.65 |
| BiLSTM | 0.21 | 0.38 | 0.17 | 0.63 |

(See `models/results/results.json` after running; full write-up in the project report.)

## What's in here

```
detections/
  wazuh/local_rules.xml      # custom Wazuh rules (ATT&CK-tagged)
  sigma/*.yml                # same detections as portable Sigma (detection-as-code)
attack/                      # MITRE ATT&CK Navigator coverage layer
response/                    # Wazuh Active Response — auto-disable the abused AD account
pipeline/                    # parse -> label (leak-free) -> split -> window
models/                      # LSTM / BiLSTM / LogBERT (anomaly + supervised) + live dashboard
tests/                       # CI: detection validity + leakage-free pipeline
docs/                        # architecture figures
.github/workflows/ci.yml     # xmllint + py_compile + pytest on every push
```

## Reproduce

```bash
pip install -r requirements.txt            # pipeline/eval deps
# training also needs torch (Python 3.11/3.12): pip install torch

# 1. build the leakage-free dataset from Wazuh archives + ground-truth runs
cd pipeline && python parse_archives.py && python label_events.py && python build_dataset.py

# 2. train + evaluate the models
cd ../models && python run_all.py          # anomaly detectors (LSTM/BiLSTM/LogBERT)
python run_supervised.py                   # supervised LSTM/BiLSTM
python make_dashboard.py                   # live anomaly-monitor dashboard.html

# 3. score the rules on the same windows (apples-to-apples)
cd ../pipeline && python score_rules_windows.py
```

## Methodology highlights (what makes it rigorous)

- **Leakage-free split**: whole attack runs assigned to one split; vocabulary built from
  training data only; thresholds chosen on validation; the test set is scored once.
- **Honest metrics**: precision / recall / F1 / FPR / ROC-AUC / PR-AUC, mean ± std over 3
  seeds, per attack type, with inference latency.
- **Frozen dataset**: the run manifest and dataset hashes are recorded for reproducibility; the raw Wazuh archives and the built `windows.npz` are **not stored in Git** (see `docs/REPRODUCING.md` for how to obtain/rebuild them).
- **Detection-as-code**: rules are ATT&CK-tagged and mirrored in Sigma; CI validates them.
- **Response**: a *guarded* Wazuh Active Response (DCSync only, non-privileged user accounts only, dry-run by default). All other detections and the ML service are **alert-only / advisory** - see `response/README.md` and `docs/SECURITY_REVIEW_FIXES.md`.

## Deploy to GitHub

```bash
git init && git add . && git commit -m "AD identity-attack detection: rules vs deep learning"
git branch -M main
git remote add origin https://github.com/USER/ad-threat-detection.git
git push -u origin main
```
CI runs automatically on push (free for public repos). Replace `USER` in the badge URL above.
