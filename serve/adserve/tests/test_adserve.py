"""Self-contained tests: no trained model or admodel needed.

A tiny stand-in nn.Module lets us exercise the full featurize -> window ->
score -> emit path and the replay loop in CI, independent of the real
BiLSTM/LogBERT weights.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

from adserve.config import ServeConfig
from adserve.featurize import Featurizer, Vocab
from adserve.scorer import ModelBundle
from adserve.service import run
from adserve.tailer import iter_json_lines
from adserve.windows import WindowStore


class TinyMasked(nn.Module):
    """Predicts each position from an embedding sum; returns logits [B,L,V]."""
    def __init__(self, vocab_size, dim=8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):            # x: [B, L]
        return self.head(self.emb(x))  # [B, L, V]


VOCAB = {"<pad>": 0, "<unk>": 1, "<mask>": 2, "4624": 3, "4769": 4, "4662": 5}


def _make_bundle(tmp: Path, threshold=0.0) -> Path:
    d = tmp / "bundle"
    d.mkdir()
    torch.save(TinyMasked(len(VOCAB)), d / "model.pt")
    (d / "vocab.json").write_text(json.dumps(VOCAB))
    (d / "meta.json").write_text(json.dumps({
        "model": "tiny", "threshold": threshold, "window_size": 4,
        "log_key_field": "rule.id",
    }))
    return d


def test_vocab_maps_unknown_to_unk():
    v = Vocab(VOCAB)
    assert v.encode("4624") == 3
    assert v.encode("does-not-exist") == v.unk_id


def test_featurizer_extracts_host_and_key():
    f = Featurizer(Vocab(VOCAB), "rule.id", "agent.name")
    host, tok = f({"rule": {"id": "4769"}, "agent": {"name": "DC01"}})
    assert host == "DC01" and tok == 4
    assert f({"agent": {"name": "DC01"}}) is None  # no key -> skipped


def test_window_fills_and_slides():
    s = WindowStore(window_size=3, idle_evict_s=999)
    assert s.push("h", 3) is None
    assert s.push("h", 4) is None
    assert s.push("h", 5) == [3, 4, 5]
    assert s.push("h", 3) == [4, 5, 3]  # slid by one


def test_scorer_runs_and_is_deterministic(tmp_path):
    bundle = ModelBundle.load(_make_bundle(tmp_path), device="cpu")
    a = bundle.score([3, 4, 5, 3])
    b = bundle.score([3, 4, 5, 3])
    assert a == b and isinstance(a, float)
    exp = bundle.explain([3, 4, 5, 3])
    assert 0 <= exp["peak_position"] < 4
    assert len(exp["per_position_surprisal"]) == 4


def test_iter_json_lines_skips_malformed():
    rows = list(iter_json_lines(['{"a":1}', "not json", "", '{"b":2}']))
    assert rows == [{"a": 1}, {"b": 2}]


def test_replay_end_to_end_emits(tmp_path):
    bundle_dir = _make_bundle(tmp_path, threshold=-1.0)  # low thr -> always fires
    alerts = tmp_path / "alerts.json"
    with open(alerts, "w") as fh:
        for ev in ["4624", "4769", "4662", "4624", "4769"]:
            fh.write(json.dumps({"rule": {"id": ev}, "agent": {"name": "DC01"}}) + "\n")
    out = tmp_path / "anoms.json"
    cfg = ServeConfig()
    cfg.bundle_dir = bundle_dir
    cfg.out_path = out
    cfg.threshold = float("nan")  # use meta
    rc = run(cfg, replay_path=alerts)
    assert rc == 0
    assert out.exists()
    recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert recs and recs[0]["host"] == "DC01"
    assert recs[0]["integration"] == "adserve"
