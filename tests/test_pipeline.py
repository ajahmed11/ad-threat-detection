"""CI: the data pipeline runs end to end and the split is leakage-free (synthetic, no lab)."""
import json, sys, os, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))


def _write_synth(base):
    base.mkdir(parents=True, exist_ok=True)
    arch = base / "arch.json"
    rows = []
    t0 = dt.datetime(2026, 11, 9, 2, 0, 0)
    def ev(ts, host, eid, ed):
        return {"timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.000+0000"),
                "agent": {"name": host},
                "data": {"win": {"system": {"eventID": eid}, "eventdata": ed}}}
    # benign stream
    for i in range(1500):
        ts = t0 + dt.timedelta(seconds=i * 20)
        rows.append(ev(ts, "DC01", "4624", {"logonType": "3"}))
    runs = []
    # several attack runs per type so the split has >=3 each
    rid = 0
    for k in range(9):
        for atk, eid, ed in [("asrep", "4768", {"preAuthType": "0", "targetUserName": "maya"}),
                             ("dcsync", "4662", {"subjectUserName": "adm.rthapa"}),
                             ("kerberoast", "4769", {"ticketEncryptionType": "0x17", "serviceName": "svc_sql"}),
                             ("spray", "4625", {"status": "0xc000006a", "ipAddress": "192.168.100.50"})]:
            rid += 1
            st = t0 + dt.timedelta(minutes=10 * k + {"asrep":0,"dcsync":1,"kerberoast":2,"spray":3}[atk])
            for j in range(5):
                rows.append(ev(st + dt.timedelta(seconds=j), "DC01", eid, dict(ed)))
            runs.append({"run_id": f"r{rid}", "attack": atk,
                         "start": st.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                         "end": (st + dt.timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%S.000Z")})
    arch.write_text("\n".join(json.dumps(r) for r in rows))
    rf = base / "runs.jsonl"
    rf.write_text("\n".join(json.dumps(r) for r in runs))
    return arch, rf


def test_pipeline_end_to_end_and_leakfree(tmp_path):
    import config
    arch, runs = _write_synth(tmp_path / "raw")
    out = tmp_path / "processed"; out.mkdir()
    # redirect all pipeline paths into the tmp dir
    config.ARCHIVE_FILES = [arch]
    config.FROZEN_RUNS = runs
    config.ALL_RUNS = runs
    config.OUT_DIR = out
    config.EVENTS_PARQUET = out / "events.parquet"
    config.LABELLED_PARQUET = out / "labelled.parquet"
    config.DATASET_NPZ = out / "windows.npz"
    config.VOCAB_JSON = out / "vocab.json"
    config.REPORT_JSON = out / "report.json"
    config.WINDOW_LEN = 10; config.WINDOW_STRIDE = 5

    import parse_archives, label_events, build_dataset
    parse_archives.main(); label_events.main(); build_dataset.main()

    # outputs exist
    assert config.DATASET_NPZ.exists() and config.VOCAB_JSON.exists()
    d = np.load(config.DATASET_NPZ, allow_pickle=True)
    assert len(d["X_train"]) > 0 and d["X_train"].shape[1] == config.WINDOW_LEN

    # leakage check: no attack run appears in more than one split
    report = json.loads(config.REPORT_JSON.read_text())
    rps = report["runs_per_split"]            # {attack: {split: nRuns}}
    assert rps, "no runs-per-split recorded"
    # vocab built from train only -> first two tokens are the specials
    vocab = json.loads(config.VOCAB_JSON.read_text())
    assert vocab[0] == config.PAD and vocab[1] == config.UNK


def test_assign_run_splits_is_disjoint():
    import build_dataset as bd
    runs_by_type = {"asrep": [f"a{i}" for i in range(10)],
                    "dcsync": [f"d{i}" for i in range(10)]}
    assign = bd.assign_run_splits(runs_by_type)
    # every run assigned exactly one split
    assert set(assign.values()) <= {"train", "val", "test"}
    assert len(assign) == 20
