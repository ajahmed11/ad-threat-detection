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
from adserve.scorer import ARCH_REGISTRY, ModelBundle, register_arch, sha256_file
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


# Architectures are registered in CODE (never from bundle data).
register_arch("tiny", lambda vs: TinyMasked(vs))

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
    torch.manual_seed(0)
    torch.save(TinyMasked(len(VOCAB)).state_dict(), d / "model.pt")   # state_dict only
    (d / "vocab.json").write_text(json.dumps(VOCAB))
    (d / "meta.json").write_text(json.dumps({
        "model": "tiny", "arch": "tiny", "threshold": threshold, "window_size": 4,
        "mask_id": len(VOCAB), "vocab_size": len(VOCAB),
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


# ---------------------------------------------------------------- security tests
import pytest


def test_loader_rejects_pickled_module(tmp_path):
    """A full pickled nn.Module (the old, unsafe format) must be refused."""
    d = _make_bundle(tmp_path)
    torch.save(TinyMasked(len(VOCAB)), d / "model.pt")      # whole object, not state_dict
    with pytest.raises(Exception):
        ModelBundle.load(d, device="cpu")


def test_loader_rejects_malicious_pickle_payload(tmp_path):
    """weights_only=True must refuse a payload that would run code on unpickle."""
    import pickle, os

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned > " + str(tmp_path / "pwned"),))

    d = _make_bundle(tmp_path)
    with open(d / "model.pt", "wb") as fh:
        pickle.dump(Evil(), fh)
    with pytest.raises(Exception):
        ModelBundle.load(d, device="cpu")
    assert not (tmp_path / "pwned").exists()


def test_unknown_arch_is_refused(tmp_path):
    d = _make_bundle(tmp_path)
    meta = json.loads((d / "meta.json").read_text())
    meta["arch"] = "os.system"          # meta cannot name arbitrary callables
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="unknown architecture"):
        ModelBundle.load(d, device="cpu")


def test_sha256_pin_enforced(tmp_path):
    d = _make_bundle(tmp_path)
    good = sha256_file(d / "model.pt")
    ModelBundle.load(d, device="cpu", expected_sha256=good)       # ok
    with pytest.raises(RuntimeError, match="mismatch"):
        ModelBundle.load(d, device="cpu", expected_sha256="0" * 64)


def test_meta_path_cannot_change_sys_path(tmp_path):
    import sys
    d = _make_bundle(tmp_path)
    meta = json.loads((d / "meta.json").read_text())
    meta["admodel_path"] = "/tmp/evil"
    (d / "meta.json").write_text(json.dumps(meta))
    ModelBundle.load(d, device="cpu")
    assert "/tmp/evil" not in sys.path


def test_oversized_line_is_dropped(tmp_path):
    from adserve.tailer import replay
    p = tmp_path / "a.json"
    big = '{"x":"' + "A" * 5000 + '"}'
    p.write_text('{"ok":1}\n' + big + '\n{"ok":2}\n')
    rows = list(replay(p, max_line_bytes=1000))
    assert rows == [{"ok": 1}, {"ok": 2}]


def test_non_object_json_is_skipped():
    assert list(iter_json_lines(["[1,2]", "42", '{"a":1}'])) == [{"a": 1}]


def test_host_cap_is_enforced():
    s = WindowStore(window_size=3, idle_evict_s=999, max_hosts=3)
    for i in range(10):
        s.push(f"h{i}", 2)
    assert s.active_hosts == 3


def test_emitter_rotates_and_has_no_windows_fields(tmp_path):
    from adserve.emit import AnomalyEmitter
    out = tmp_path / "ml.json"
    em = AnomalyEmitter(out, max_bytes=300, backups=2)
    for _ in range(10):
        rec = em.emit(host="DC01", score=9.0, threshold=1.0, model="m",
                      window_tokens=["a"] * 5, detail={"peak_token": "a"})
    assert "data" not in rec                         # advisory only: no data.win.eventdata
    assert out.with_name("ml.json.1").exists()
    assert not out.with_name("ml.json.3").exists()   # bounded retention
