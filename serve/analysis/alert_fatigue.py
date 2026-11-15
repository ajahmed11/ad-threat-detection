#!/usr/bin/env python3
"""Alert-fatigue / cost analysis for the AD anomaly detector.

Quantifies the operational burden of running the BiLSTM anomaly model as a
live detector, and the volume/quality trade-off of the decision threshold.

Three views:
  A. Operational volume  - replay one real day (Wazuh archive) through the
     exact adserve path and count alerts per host / per hour / per event type.
  B. Threshold sweep     - on the LABELLED test windows (ground truth), measure
     precision / recall / FPR / F1 across thresholds, and pair each with the
     daily alert volume it would produce on the real stream.
  C. Cost model          - translate alert volume into analyst-hours/day and an
     annualised figure, at a few operating points, next to the rule layer.

Run from a place where `admodel` and `adserve` import (see --help).
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from adserve.featurize import Featurizer, Vocab, status_class  # noqa: F401
from adserve.scorer import ModelBundle


# ----------------------------------------------------------------------------
def _open_any(path: Path):
    p = str(path)
    if p.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(p, "rb"), encoding="utf-8", errors="ignore")
    return open(p, encoding="utf-8", errors="ignore")


def parse_hour(ts: str):
    """Return the UTC hour (0-23) from a Wazuh ISO timestamp, or None."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).hour
    except ValueError:
        return None


def score_archive(bundle: ModelBundle, featurizer: Featurizer, archive: Path):
    """Replay the archive exactly as adserve does (per-host stride-1 windows).

    Returns a list of dicts, one per scored window:
      {host, hour, score, event_id}  (event_id = the peak/most-anomalous event)
    """
    W = bundle.window_size
    buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=W))
    times: dict[str, deque] = defaultdict(lambda: deque(maxlen=W))
    rows = []
    with _open_any(archive) as fh:
        for line in fh:
            line = line.replace("\x00", "").strip()
            if not line:
                continue
            try:
                alert = json.loads(line)
            except json.JSONDecodeError:
                continue
            feat = featurizer(alert)
            if feat is None:
                continue
            host, tok = feat
            buffers[host].append(tok)
            times[host].append(alert.get("timestamp"))
            if len(buffers[host]) < W:
                continue
            window = list(buffers[host])
            score = bundle.score(window)
            rows.append({"host": host, "hour": parse_hour(times[host][-1]),
                         "score": score, "_window": window})
    return rows


def peak_event_id(bundle: ModelBundle, window) -> str:
    """Event ID of the single most anomalous position in a window."""
    tok = bundle.explain(window)["peak_token"]
    return tok.split("|")[0] if tok != "<unk>" else "<unk>"


def count_episodes(rows, thr):
    """Collapse consecutive above-threshold windows on a host into one 'incident'.

    Windows slide by one event, so a single burst produces many overlapping
    alerts. An analyst triages the burst once, so episodes is the fairer unit
    of operational load than raw window-alerts.
    """
    prev_alert = defaultdict(bool)
    episodes = 0
    for r in rows:
        a = r["score"] >= thr
        if a and not prev_alert[r["host"]]:
            episodes += 1
        prev_alert[r["host"]] = a
    return episodes


# ----------------------------------------------------------------------------
def labelled_scores(bundle: ModelBundle, npz_path: Path):
    d = np.load(npz_path, allow_pickle=False)
    X, y = d["X_test"], d["y_test"].astype(int)
    scores = np.array([bundle.score([int(t) for t in w]) for w in X], dtype=np.float32)
    return scores, y


def sweep(scores_lbl, y, raw_scores, thresholds, min_per_alert, hourly_rate):
    """Precision/recall/FPR/F1 (labelled) + daily alert volume + cost per threshold."""
    out = []
    raw = np.asarray(raw_scores)
    n_days = 1  # the archive is one calendar day
    for thr in thresholds:
        pred = (scores_lbl >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        alerts_day = int((raw >= thr).sum()) / n_days
        hours_day = alerts_day * min_per_alert / 60.0
        out.append({
            "threshold": round(float(thr), 3),
            "precision": round(prec, 3), "recall": round(rec, 3),
            "fpr": round(fpr, 3), "f1": round(f1, 3),
            "alerts_per_day": round(alerts_day, 1),
            "analyst_hours_per_day": round(hours_day, 2),
            "analyst_cost_per_year": round(hours_day * 365 * hourly_rate, 0),
        })
    return out


def first_threshold_under_fpr(rows, max_fpr):
    cand = [r for r in rows if r["fpr"] <= max_fpr]
    return min(cand, key=lambda r: r["threshold"]) if cand else None


# ----------------------------------------------------------------------------
def _charts(out_dir: Path, sweep_rows, operational, op_thr):
    """Three figures: the volume/recall trade-off, per-host, per-hour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thr = [r["threshold"] for r in sweep_rows]
    rec = [r["recall"] for r in sweep_rows]
    apd = [max(r["alerts_per_day"], 0.5) for r in sweep_rows]

    # 1) trade-off: recall vs alerts/day across thresholds
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(thr, rec, color="#2563eb", marker="o", ms=3, label="recall (test)")
    ax1.set_xlabel("anomaly threshold"); ax1.set_ylabel("recall", color="#2563eb")
    ax1.set_ylim(-0.03, 1.03)
    ax2 = ax1.twinx()
    ax2.plot(thr, apd, color="#dc2626", marker="s", ms=3, label="alerts/day (real stream)")
    ax2.set_yscale("log"); ax2.set_ylabel("alerts / day (log)", color="#dc2626")
    ax1.axvline(op_thr, ls="--", lw=.8, color="#555")
    ax1.set_title("No usable operating point: recall collapses before alert volume does")
    fig.tight_layout(); fig.savefig(out_dir / "tradeoff.png", dpi=130); plt.close(fig)

    # 2) alerts by host
    bh = operational["by_host"]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(list(bh.keys()), list(bh.values()), color="#2563eb")
    ax.set_ylabel("alerts"); ax.set_title(f"Alerts by host @thr={op_thr:.1f} (one day)")
    for i, v in enumerate(bh.values()):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "by_host.png", dpi=130); plt.close(fig)

    # 3) alerts by hour
    by_hour = operational["by_hour"]
    hours = list(range(24)); vals = [by_hour.get(str(h), 0) for h in hours]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(hours, vals, color="#dc2626")
    ax.set_xlabel("hour of day (UTC)"); ax.set_ylabel("alerts")
    ax.set_title("Alert volume by hour — the burst an analyst would face")
    fig.tight_layout(); fig.savefig(out_dir / "by_hour.png", dpi=130); plt.close(fig)
    print(f"wrote 3 charts -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--archive", type=Path, required=True, help="one day's Wazuh archive")
    ap.add_argument("--windows", type=Path, required=True, help="labelled windows.npz")
    ap.add_argument("--out-dir", type=Path, default=Path("analysis_out"))
    ap.add_argument("--min-per-alert", type=float, default=5.0,
                    help="analyst triage minutes per alert (assumption)")
    ap.add_argument("--hourly-rate", type=float, default=30.0,
                    help="loaded analyst cost per hour, same currency throughout")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bundle = ModelBundle.load(args.bundle_dir, device="cpu")
    featurizer = Featurizer.from_meta(bundle.vocab, bundle.meta)
    op_thr = float(bundle.meta["threshold"])

    print("[1/4] scoring the labelled test windows ...")
    s_lbl, y = labelled_scores(bundle, args.windows)

    print("[2/4] replaying the day's archive (this is the slow part) ...")
    rows = score_archive(bundle, featurizer, args.archive)
    raw_scores = np.array([r["score"] for r in rows], dtype=np.float32)
    n_windows = len(rows)

    # --- A. operational volume at the operating threshold --------------------
    alerts = [r for r in rows if r["score"] >= op_thr]
    for r in alerts:
        r["event_id"] = peak_event_id(bundle, r["_window"])
    by_host = Counter(r["host"] for r in alerts)
    by_hour = Counter(r["hour"] for r in alerts if r["hour"] is not None)
    by_event = Counter(r["event_id"] for r in alerts)
    peak_hour = max(by_hour.items(), key=lambda kv: kv[1]) if by_hour else (None, 0)

    episodes = count_episodes(rows, op_thr)
    operational = {
        "windows_scored": n_windows,
        "alerts": len(alerts),
        "alert_rate": round(len(alerts) / n_windows, 4) if n_windows else 0.0,
        "incidents_deduped": episodes,
        "operating_threshold": round(op_thr, 3),
        "by_host": dict(by_host.most_common()),
        "alerts_per_hour_peak": {"hour_utc": peak_hour[0], "alerts": peak_hour[1]},
        "by_hour": {str(h): by_hour.get(h, 0) for h in range(24)},
        "top_event_ids": dict(by_event.most_common(10)),
    }

    # --- B. threshold sweep ---------------------------------------------------
    print("[3/4] threshold sweep ...")
    lo, hi = float(raw_scores.min()), float(raw_scores.max())
    grid = np.linspace(lo, hi, 40)
    rows_sweep = sweep(s_lbl, y, raw_scores, grid, args.min_per_alert, args.hourly_rate)

    # operating points
    f1_opt = min(rows_sweep, key=lambda r: abs(r["threshold"] - op_thr))
    low_fpr = first_threshold_under_fpr(rows_sweep, 0.01) or first_threshold_under_fpr(rows_sweep, 0.05)

    # --- C. cost model + rules comparison ------------------------------------
    print("[4/4] cost model ...")
    rules_path = args.windows.parent / "rules_window_scores.json"
    rules = json.loads(rules_path.read_text()) if rules_path.exists() else None

    report = {
        "inputs": {
            "archive": args.archive.name,
            "bundle": bundle.meta.get("model"),
            "min_per_alert": args.min_per_alert,
            "hourly_rate": args.hourly_rate,
        },
        "operational": operational,
        "operating_points": {
            "f1_optimal": f1_opt,
            "low_fpr": low_fpr,
        },
        "threshold_sweep": rows_sweep,
        "rules_window_level": rules,
    }
    (args.out_dir / "alert_fatigue_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out_dir/'alert_fatigue_report.json'}")
    _charts(args.out_dir, rows_sweep, operational, op_thr)

    # concise console summary
    print("\n===== ALERT-FATIGUE SUMMARY =====")
    print(f"day replayed:        {args.archive.name}")
    print(f"windows scored:      {n_windows:,}")
    print(f"alerts @F1-opt thr {op_thr:.2f}:  {len(alerts):,} "
          f"({operational['alert_rate']*100:.1f}% of windows)")
    print(f"  busiest hour:      {peak_hour[1]} alerts at {peak_hour[0]:02d}:00 UTC"
          if peak_hour[0] is not None else "")
    print(f"  by host:           {dict(by_host.most_common())}")
    print(f"\n-- operating points --")
    print(f"F1-optimal  thr={f1_opt['threshold']:.2f}  P={f1_opt['precision']} "
          f"R={f1_opt['recall']} FPR={f1_opt['fpr']}  "
          f"{f1_opt['alerts_per_day']:.0f} alerts/day  "
          f"{f1_opt['analyst_hours_per_day']:.1f} h/day")
    if low_fpr:
        print(f"Low-FPR     thr={low_fpr['threshold']:.2f}  P={low_fpr['precision']} "
              f"R={low_fpr['recall']} FPR={low_fpr['fpr']}  "
              f"{low_fpr['alerts_per_day']:.0f} alerts/day  "
              f"{low_fpr['analyst_hours_per_day']:.1f} h/day")
    if rules:
        print(f"\nrules (window-level): P={rules['precision']} R={rules['recall']} "
              f"FPR={rules['fpr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
