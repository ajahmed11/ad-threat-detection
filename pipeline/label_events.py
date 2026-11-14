"""
label_events.py - attach ground-truth labels to the parsed events, with attack-window masking.

Logic (leak-safe):
  * An event inside a FROZEN (verified-clean) run window  -> label 1, attack_type set, run_id set.
  * An event inside ANY run window from the full log that is NOT a clean positive
    (a failed/dropped/other attack) -> DROPPED, so it never pollutes the benign class.
  * Everything else -> label 0 (benign).

All times are handled in UTC. Windows are padded by config.PAD_SECONDS on each side.
Writes data/processed/labelled_events.parquet.

Run:  python src/label_events.py
"""
from __future__ import annotations
import pandas as pd
import config


def _load_runs(path):
    r = pd.read_json(path, lines=True)
    if "attack" not in r.columns:
        raise SystemExit(f"{path} has no 'attack' column - wrong run file?")
    r = r[r["attack"].astype(str).str.lower() != "test"].copy()
    r["start"] = pd.to_datetime(r["start"], utc=True)
    r["end"]   = pd.to_datetime(r["end"], utc=True)
    r["attack"] = r["attack"].astype(str).str.lower()
    pad = pd.Timedelta(seconds=config.PAD_SECONDS)
    r["w0"] = r["start"] - pad
    r["w1"] = r["end"] + pad
    return r.reset_index(drop=True)


def main():
    df = pd.read_parquet(config.EVENTS_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    frozen = _load_runs(config.FROZEN_RUNS)
    allr   = _load_runs(config.ALL_RUNS)

    df["label"] = 0
    df["attack_type"] = "benign"
    df["run_id"] = pd.NA
    df["_masked"] = False

    ts = df["timestamp"].values

    # 1) mask every attack window from the full log (will drop any that aren't clean positives)
    for _, r in allr.iterrows():
        m = (df["timestamp"] >= r["w0"]) & (df["timestamp"] <= r["w1"])
        df.loc[m, "_masked"] = True

    # 2) positives from the frozen clean runs (overrides the mask for these windows)
    for _, r in frozen.iterrows():
        m = (df["timestamp"] >= r["w0"]) & (df["timestamp"] <= r["w1"])
        df.loc[m, "label"] = 1
        df.loc[m, "attack_type"] = r["attack"]
        df.loc[m, "run_id"] = r.get("run_id")
        df.loc[m, "_masked"] = False  # a clean positive is kept, not dropped

    before = len(df)
    dropped = int(df["_masked"].sum())
    df = df[~df["_masked"]].drop(columns="_masked").reset_index(drop=True)

    print(f"[label] events in: {before:,}")
    print(f"[label] dropped (unlabelled-attack windows, masked out of benign): {dropped:,}")
    print(f"[label] kept: {len(df):,}")
    print("[label] class balance:\n", df["attack_type"].value_counts().to_string())
    print("[label] labelled attack runs seen:",
          df.loc[df.label == 1, "run_id"].nunique())

    df.to_parquet(config.LABELLED_PARQUET, index=False)
    print(f"[label] wrote -> {config.LABELLED_PARQUET}")


if __name__ == "__main__":
    main()
