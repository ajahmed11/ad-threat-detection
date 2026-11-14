# Alert-Fatigue / Cost Analysis

Quantifies the operational burden of running the BiLSTM anomaly model as a live
detector, using one real day of telemetry (`2026-Nov-ossec-archive-09.json`).

## Run

```bash
python alert_fatigue.py \
    --bundle-dir ../serve_bundle \
    --archive  <path>/2026-Nov-ossec-archive-09.json \
    --windows  <path>/windows.npz \
    --min-per-alert 5 --hourly-rate 30 \
    --out-dir results
```

Outputs `results/alert_fatigue_report.json` plus three figures.

## Headline findings (one lab day, BiLSTM @ F1-optimal threshold)

| Measure | Value |
| --- | --- |
| Windows scored | 12,713 |
| Raw window-alerts | 1,222 (9.6% of windows) |
| Deduplicated incidents | 54 |
| Precision at this point | ~0.15 (≈85% of alerts are false) |
| Recall at this point | 0.38 |
| Busiest hour | 544 alerts in one hour |
| Alerts on `DC01` | 1,077 of 1,222 (88%) |

**There is no usable operating point.** No threshold gives even 30% recall at
under ~100 alerts/day: raising the threshold to cut volume drops recall to near
zero before the volume becomes tolerable (`tradeoff.png`). This is the
quantified case for why the unsupervised deep model is not deployable alone on
this telemetry, and why the precise rule layer carries the load for known
attacks.

## Figures

- `tradeoff.png` — recall vs alerts/day across thresholds (the core result)
- `by_host.png` — alert concentration on the DC
- `by_hour.png` — the hourly burst an analyst would face

## Cost model

Analyst load = alerts × triage minutes. At 5 min/alert the raw window-alerts
imply ~100 analyst-hours/day (an upper bound that counts overlapping windows
separately); collapsing consecutive windows into 54 incidents gives ~4.5
hours/day — but at ~15% precision most of that time is spent on false alarms.
Both `--min-per-alert` and `--hourly-rate` are parameters; set them to your
own assumptions.

## Honest caveats

- Window-alerts overlap (stride-1 windows), so raw counts overstate distinct
  work; the deduplicated incident count is the fairer unit.
- The rule layer's window-level numbers (`rules_window_scores.json`: recall
  0.77, FPR 0.35) are noisier than often assumed — rules catch more but also
  over-fire at window granularity. Neither layer is clean; the point is the
  trade-off, not a winner.
