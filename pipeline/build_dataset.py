"""
build_dataset.py - log-key encoding, leak-free split, and windowing.

Produces the sequence windows the LSTM / BiLSTM / LogBERT models consume.

Leak-free design:
  * Attack RUNS are assigned whole to train/val/test, stratified per attack type,
    so overlapping windows from one run never straddle a split.
  * Benign events are split by TIME block per host (70/15/15), and windows are built
    AFTER the split, per host, never crossing a split boundary or a long time gap.
  * The log-key VOCABULARY is built from the TRAIN split only; unseen keys -> <unk>.

Outputs windows.npz (X_train/val/test int sequences, y_*, atk_*), vocab.json, dataset_report.json.

Run:  python src/build_dataset.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import config

rng = np.random.default_rng(config.SEED)


def _status_class(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "na"
    s = str(s).lower()
    return "ok" if s in ("0x0", "0") else "fail" if s.startswith("0x") else "other"


def make_log_key(row):
    parts = []
    for f in config.KEY_FIELDS:
        if f == "status_class":
            parts.append(_status_class(row.get("status")))
        else:
            v = row.get(f)
            parts.append("na" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    return "|".join(parts)


def assign_run_splits(runs_by_type):
    """Return dict run_id -> split, stratified per attack type."""
    assign = {}
    for atk, run_ids in runs_by_type.items():
        ids = list(run_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * config.TEST_FRAC)) if n >= 3 else 0
        n_val = max(1, round(n * config.VAL_FRAC)) if n >= 3 else 0
        for i, rid in enumerate(ids):
            if i < n_test:
                assign[rid] = "test"
            elif i < n_test + n_val:
                assign[rid] = "val"
            else:
                assign[rid] = "train"
    return assign


def time_split_benign(bdf):
    """Split benign events by contiguous time block per host (70/15/15, train earliest)."""
    out = {}
    for host, g in bdf.groupby("agent"):
        g = g.sort_values("timestamp")
        n = len(g)
        i_tr = int(n * (1 - config.VAL_FRAC - config.TEST_FRAC))
        i_va = int(n * (1 - config.TEST_FRAC))
        split = np.array(["train"] * n, dtype=object)
        split[i_tr:i_va] = "val"
        split[i_va:] = "test"
        out[host] = pd.Series(split, index=g.index)
    s = pd.concat(out.values()) if out else pd.Series(dtype=object)
    return s.reindex(bdf.index)


def build_windows(df_split, key2id):
    """Sliding windows per host within one split. Returns X (int seqs), y, atk."""
    X, y, atk, meta = [], [], [], []
    unk = key2id[config.UNK]
    W, S, maxgap = config.WINDOW_LEN, config.WINDOW_STRIDE, config.MAX_GAP_SECONDS
    for host, g in df_split.groupby("agent"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        ids = [key2id.get(k, unk) for k in g["logkey"]]
        labels = g["label"].tolist()
        atypes = g["attack_type"].tolist()
        times = g["timestamp"].astype("int64").to_numpy() // 10**9  # seconds
        # break into contiguous segments (no gap > maxgap)
        seg_start = 0
        segments = []
        for i in range(1, len(g)):
            if times[i] - times[i - 1] > maxgap:
                segments.append((seg_start, i))
                seg_start = i
        segments.append((seg_start, len(g)))
        for a, b in segments:
            if b - a < W:
                continue
            for st in range(a, b - W + 1, S):
                sl = slice(st, st + W)
                X.append(ids[sl])
                win_lab = int(any(labels[sl]))
                y.append(win_lab)
                # window attack type = the attack present, else benign
                present = {atypes[j] for j in range(st, st + W) if labels[j] == 1}
                atk.append(next(iter(present)) if present else "benign")
                # metadata: host + time span, so the Wazuh rules can be scored on the
                # SAME windows (window-level rules-vs-ML comparison).
                meta.append((host, int(times[st]), int(times[st + W - 1])))
    return (np.array(X, dtype=np.int64),
            np.array(y, dtype=np.int64),
            np.array(atk, dtype=str), meta)   # str, not object: loadable with allow_pickle=False


def main():
    df = pd.read_parquet(config.LABELLED_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["logkey"] = df.apply(make_log_key, axis=1)

    # ---- split assignment ----
    atk_df = df[df.label == 1]
    ben_df = df[df.label == 0].copy()
    runs_by_type = (atk_df.dropna(subset=["run_id"])
                    .groupby("attack_type")["run_id"].unique().to_dict())
    run_split = assign_run_splits(runs_by_type)
    df["split"] = None
    df.loc[atk_df.index, "split"] = atk_df["run_id"].map(run_split)
    df.loc[ben_df.index, "split"] = time_split_benign(ben_df)
    df = df.dropna(subset=["split"])

    # ---- vocabulary from TRAIN only ----
    train_keys = sorted(df.loc[df.split == "train", "logkey"].unique())
    vocab = [config.PAD, config.UNK] + train_keys
    key2id = {k: i for i, k in enumerate(vocab)}

    # ---- windows per split ----
    data, report = {}, {"vocab_size": len(vocab), "splits": {}, "runs_per_split": {}}
    meta_rows = []
    for sp in ("train", "val", "test"):
        X, y, atk, meta = build_windows(df[df.split == sp], key2id)
        data[f"X_{sp}"], data[f"y_{sp}"], data[f"atk_{sp}"] = X, y, atk
        for (host, t0, t1), lab, at in zip(meta, y, atk):
            meta_rows.append({"split": sp, "host": host, "t_start": t0, "t_end": t1,
                              "label": int(lab), "attack_type": at})
        report["splits"][sp] = {
            "windows": int(len(X)),
            "attack_windows": int(y.sum()) if len(y) else 0,
            "benign_windows": int((y == 0).sum()) if len(y) else 0,
            "events": int((df.split == sp).sum()),
            "by_attack": {k: int(v) for k, v in
                          pd.Series(atk).value_counts().items()} if len(atk) else {},
        }
    # runs per split per attack (leakage check: a run_id appears in one split only)
    rs = (df[df.label == 1].dropna(subset=["run_id"])
          .groupby(["attack_type", "split"])["run_id"].nunique().unstack(fill_value=0))
    report["runs_per_split"] = json.loads(rs.to_json())

    np.savez_compressed(config.DATASET_NPZ, **data)
    config.VOCAB_JSON.write_text(json.dumps(vocab, indent=0))
    config.REPORT_JSON.write_text(json.dumps(report, indent=2))
    pd.DataFrame(meta_rows).to_parquet(config.OUT_DIR / "windows_meta.parquet", index=False)

    print(f"[build] vocab size: {len(vocab)}")
    for sp in ("train", "val", "test"):
        s = report["splits"][sp]
        print(f"[build] {sp:5s}: {s['windows']:6d} windows "
              f"({s['attack_windows']} attack / {s['benign_windows']} benign)")
    print("[build] runs per split (leakage check - each run in exactly one split):")
    print(rs.to_string())
    print(f"[build] wrote {config.DATASET_NPZ.name}, {config.VOCAB_JSON.name}, {config.REPORT_JSON.name}")


if __name__ == "__main__":
    main()
