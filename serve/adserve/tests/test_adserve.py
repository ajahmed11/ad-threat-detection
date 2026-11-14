"""Self-contained tests: no trained model or admodel needed.

A tiny stand-in nn.Module lets us exercise the full featurize -> window ->
score -> emit path and the replay loop in CI, independent of the real
BiLSTM/LogBERT weights. The vocab and log keys mirror the adpipe pipeline:
a list vocab (index = id) and 5-part composite keys.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

from adserve.config import ServeConfig
from adserve.featurize import Featurizer, Vocab, status_class
from adserve.scorer import ModelBundle
from adserve.service import run
from adserve.tailer import iter_json_lines
from adserve.windows import WindowStore


class TinyMasked(nn.Module):
    """Embedding (vocab_size+1 for the mask slot) -> linear; logits [B,L,V]."""
    def __init__(self, vocab_size, dim=8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, dim, padding_idx=0)
        self.head = nn.Linear(dim, vocab_size + 1)

    def forward(self, x):
        return self.head(self.emb(x))


# list vocab exactly like build_dataset.py writes it (index = id)
VOCAB = ["<pad>", "<unk>", "4624|2|na|na|ok", "4769|na|0x17|na|ok", "4662|na|na|na|na"]


def _alert(eid, host="DC01", logon=None, tenc=None, preauth=None, status=None):
    ed = {}
    if logon is not None:   ed["logonType"] = logon
    if tenc is not None:    ed["ticketEncryptionType"] = tenc
    if preauth is not None: ed["preAuthType"] = preauth
    if status is not None:  ed["status"] = status
    return {"agent": {"name": host},
            "data": {"win": {"system": {"eventID": eid}, "eventdata": ed}}}


def _make_bundle(tmp: Path, threshold=0.0) -> Path:
    d = tmp / "bundle"
    d.mkdir()
    torch.save(TinyMasked(len(VOCAB)), d / "model.pt")
    (d / "vocab.json").write_text(json.dumps(VOCAB))
    (d / "meta.json").write_text(json.dumps({
        "model": "bilstm", "threshold": threshold, "window_size": 4,
        "mask_id": len(VOCAB),
    }))
    return d


def test_status_class_matches_pipeline():
    assert status_class("0x0") == "ok"
    assert status_class("0") == "ok"
    assert status_class("0xC000006D") == "fail"
    assert status_class(None) == "na"
    assert status_class("weird") == "other"


def test_vocab_list_form_and_unk():
    v = Vocab(VOCAB)
    assert v.encode("4624|2|na|na|ok") == 2
    assert v.encode("never-seen") == v.unk_id == 1
    assert v.pad_id == 0


def test_featurizer_builds_composite_key():
    f = Featurizer(Vocab(VOCAB))
    # 4624 logon type 2, status 0x0 -> ok  => "4624|2|na|na|ok" (id 2)
    host, tok = f(_alert("4624", logon="2", status="0x0"))
    assert host == "DC01" and tok == 2
    # 4769 with 0x17 encryption, ok  => "4769|na|0x17|na|ok" (id 3)
    _, tok2 = f(_alert("4769", tenc="0x17", status="0x0"))
    assert tok2 == 3
    # non-Windows alert (no data.win) is skipped
    assert f({"agent": {"name": "DC01"}, "rule": {"id": "5502"}}) is None


def test_window_fills_and_slides():
    s = WindowStore(window_size=3, idle_evict_s=999)
    assert s.push("h", 2) is None
    assert s.push("h", 3) is None
    assert s.push("h", 4) == [2, 3, 4]
    assert s.push("h", 2) == [3, 4, 2]


def test_scorer_runs_and_is_deterministic(tmp_path):
    bundle = ModelBundle.load(_make_bundle(tmp_path), device="cpu")
    assert bundle.mask_id == len(VOCAB)
    a = bundle.score([2, 3, 4, 2])
    b = bundle.score([2, 3, 4, 2])
    assert a == b and isinstance(a, float)
    exp = bundle.explain([2, 3, 4, 2])
    assert 0 <= exp["peak_position"] < 4
    assert len(exp["per_position_surprisal"]) == 4


def test_iter_json_lines_skips_malformed():
    rows = list(iter_json_lines(['{"a":1}', "not json", "", '{"b":2}']))
    assert rows == [{"a": 1}, {"b": 2}]


def test_replay_end_to_end_emits(tmp_path):
    bundle_dir = _make_bundle(tmp_path, threshold=-1.0)  # low thr -> always fires
    alerts = tmp_path / "alerts.json"
    seq = [("4624", "2", None, "0x0"), ("4769", None, "0x17", "0x0"),
           ("4662", None, None, None), ("4624", "2", None, "0x0"),
           ("4769", None, "0x17", "0x0")]
    with open(alerts, "w") as fh:
        for eid, logon, tenc, status in seq:
            fh.write(json.dumps(_alert(eid, logon=logon, tenc=tenc, status=status)) + "\n")
    out = tmp_path / "anoms.json"
    cfg = ServeConfig()
    cfg.bundle_dir = bundle_dir
    cfg.out_path = out
    cfg.threshold = float("nan")  # use meta
    rc = run(cfg, replay_path=alerts)
    assert rc == 0
    recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert recs and recs[0]["host"] == "DC01"
    assert recs[0]["integration"] == "adserve"
