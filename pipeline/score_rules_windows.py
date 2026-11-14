"""
score_rules_windows.py - score the custom Wazuh rules on the SAME windows the ML models
were tested on, so the rules-vs-ML comparison is apples-to-apples (window level).

A test window (host, t_start..t_end) is "rule-detected" if any custom-rule alert
(rule id >= 100500) for that host fires inside the window's time span (+/- a small
tolerance for Wazuh's forwarding delay). Reports precision / recall / F1 / FPR and
per-attack recall on the test split, directly comparable with results.json from admodel.

Needs: windows_meta.parquet (produced by build_dataset.py) and the dated alert files
staged under <RAW_DIR>/alerts/2026/<Mon>/ossec-alerts-*.json(.gz).

Run:  python src/score_rules_windows.py
"""
from __future__ import annotations
import glob, gzip, io, json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import config

CUSTOM_MIN_ID = 100500
TOLERANCE_S = 5          # a little slack on each side for forwarding delay
ALERT_GLOB = str(config.RAW_DIR / "alerts" / "2026" / "*" / "ossec-alerts-*.json*")


def _open_any(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


def load_custom_alerts():
    rows, nfiles = [], 0
    for path in sorted(glob.glob(ALERT_GLOB)):
        nfiles += 1
        with _open_any(path) as f:
            for line in f:
                line = line.replace("\x00", "").strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    rid = int(d["rule"]["id"])
                    if rid < CUSTOM_MIN_ID:
                        continue
                    rows.append({"t": d["timestamp"],
                                 "host": d.get("agent", {}).get("name"),
                                 "rule_id": rid})
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    continue
    if not rows:
        raise SystemExit(f"No custom-rule alerts found under {ALERT_GLOB} ({nfiles} files).")
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce").astype("int64") // 10**9
    print(f"[rules] {nfiles} alert files, {len(df)} custom-rule alerts")
    return df.dropna(subset=["t"])


def main():
    meta = pd.read_parquet(config.OUT_DIR / "windows_meta.parquet")
    test = meta[meta.split == "test"].reset_index(drop=True)
    alerts = load_custom_alerts()

    tol = TOLERANCE_S
    detected = np.zeros(len(test), dtype=int)
    for host, ha in alerts.groupby("host"):
        times = ha["t"].to_numpy()
        m = test["host"] == host
        for i in test[m].index:
            t0, t1 = test.at[i, "t_start"] - tol, test.at[i, "t_end"] + tol
            if np.any((times >= t0) & (times <= t1)):
                detected[i] = 1

    y = test["label"].to_numpy()
    p, r, f1, _ = precision_recall_fscore_support(y, detected, average="binary", zero_division=0)
    tn = int(((detected == 0) & (y == 0)).sum()); fp = int(((detected == 1) & (y == 0)).sum())
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print("\n===== CUSTOM RULES @ WINDOW LEVEL (test split) =====")
    print(f"windows: {len(test)}  (attack {int(y.sum())} / benign {int((y==0).sum())})")
    print(f"precision {p:.3f} | recall {r:.3f} | F1 {f1:.3f} | FPR {fpr:.3f}")
    print("\nper-attack recall:")
    for atk in sorted(set(test["attack_type"]) - {"benign"}):
        mm = test["attack_type"] == atk
        print(f"  {atk:12s} {detected[mm].mean():.3f}  (n={int(mm.sum())})")

    out = {"windows": int(len(test)), "attack_windows": int(y.sum()),
           "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
           "fpr": round(fpr, 4),
           "per_attack_recall": {atk: round(float(detected[test.attack_type == atk].mean()), 4)
                                 for atk in sorted(set(test.attack_type) - {"benign"})}}
    (config.OUT_DIR / "rules_window_scores.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {config.OUT_DIR / 'rules_window_scores.json'}")
    print("Compare these directly with admodel/results.json (same test windows).")


if __name__ == "__main__":
    main()
