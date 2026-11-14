#!/usr/bin/env python3
"""Train a model and package it as a serving bundle for adserve.

Run this from the admodel project root (where `import admodel` works and the
adpipe dataset exists), e.g.:

    D:\\Final_Year\\CSP\\Lab_Project\\data\\raw\\admodel> python make_bundle.py bilstm

It trains the chosen model on the benign windows (seed 0), picks the
F1-optimal threshold on the validation set exactly as evaluate.py does, then
writes serve_bundle/{model.pt, vocab.json, meta.json}. Point adserve at it:

    python -m adserve.service --bundle-dir serve_bundle --replay <alerts.json> --out anoms.json

Why this exists: the training harness scored models in-memory without saving
them. This adds model serialization so the trained network can be *deployed* —
part of the ML-ops story the live service tells.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

from admodel import config, data as D, train, evaluate


# How each composite-key field is read from a live Wazuh alert. These paths
# mirror adpipe/src/parse_archives.py so the live log key == the training key.
FIELD_PATHS = {
    "event_id":   "data.win.system.eventID",
    "logon_type": "data.win.eventdata.logonType",
    "ticket_enc": "data.win.eventdata.ticketEncryptionType",
    "preauth":    "data.win.eventdata.preAuthType",
    "status":     "data.win.eventdata.status",
}
KEY_FIELDS = ["event_id", "logon_type", "ticket_enc", "preauth", "status_class"]


def main() -> int:
    name = (sys.argv[1] if len(sys.argv) > 1 else "bilstm").lower()
    if name not in ("lstm", "bilstm", "logbert"):
        raise SystemExit("usage: python make_bundle.py [lstm|bilstm|logbert]")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("serve_bundle")
    out.mkdir(parents=True, exist_ok=True)

    ds = D.load_dataset()
    vs, mask_id = ds["vocab_size"], ds["mask_id"]
    Xtr = ds["X_train"][ds["y_train"] == 0]
    Xva = ds["X_val"][ds["y_val"] == 0]
    print(f"[bundle] training {name}: device={config.DEVICE} vocab={vs} "
          f"benign_train={len(Xtr)} benign_val={len(Xva)}")

    window_len = int(ds["X_train"].shape[1])  # authoritative: the built window width
    model, lc = train.train_model(name, Xtr, Xva, vs, mask_id, seed=0)
    res, _ = evaluate.evaluate(model, name, ds, mask_id)
    thr = float(res["threshold"])
    print(f"[bundle] trained {lc['epochs']} epochs | "
          f"F1={res['f1']} P={res['precision']} R={res['recall']} "
          f"FPR={res['fpr']} threshold={thr:.4f}")

    model.eval()
    torch.save(model, out / "model.pt")
    shutil.copyfile(config.VOCAB_JSON, out / "vocab.json")

    # Where the admodel package lives, so adserve can unpickle the module.
    admodel_root = str(Path(train.__file__).resolve().parent.parent)
    meta = {
        "model": name,
        "threshold": thr,
        "window_size": window_len,
        "mask_id": int(mask_id),
        "vocab_size": int(vs),
        "admodel_path": admodel_root,
        "featurizer": {
            "key_fields": KEY_FIELDS,
            "field_paths": FIELD_PATHS,
            "require_path": "data.win",
            "host_field": "agent.name",
        },
        "val_metrics": {k: res[k] for k in ("f1", "precision", "recall", "fpr")},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[bundle] wrote -> {out.resolve()}")
    print(f"[bundle] files: model.pt, vocab.json, meta.json")
    if name == "logbert":
        print("[bundle] NOTE: the live scorer uses masked-surprisal only; the "
              "LogBERT hypersphere term is not applied at serve time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
