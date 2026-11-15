"""Load the trained model bundle and score a window with max-surprisal.

Scoring mirrors admodel/evaluate.py exactly: mask each position in turn,
read the model's log-probability for the token that was really there, and take
the single most surprising position as the window score. A window is anomalous
when its score crosses the threshold frozen at training time.

SECURITY MODEL (see docs/SECURITY_REVIEW_FIXES.md)
  * The bundle holds DATA only: a tensor state_dict (model.pt), vocab.json and
    meta.json. It is loaded with torch.load(weights_only=True), so no pickled
    Python object / code is ever executed.
  * The architecture is chosen from a FIXED registry in this file. meta.json can
    only select a name from that registry; it cannot name a module or a path.
  * sys.path is never modified. `admodel` must be importable as an installed /
    PYTHONPATH-configured dependency (set by the systemd unit, not by the bundle).
  * model.pt is verified against a SHA-256 that must come from OUTSIDE the bundle
    (env ADSERVE_MODEL_SHA256 / --expect-sha256). A hash stored beside the file
    protects against nothing, so meta.json's copy is informational only.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from .featurize import Vocab

log = logging.getLogger("adserve.scorer")

# name -> factory(vocab_size) -> nn.Module.  Trusted application code only.
ARCH_REGISTRY: dict[str, Callable[[int], nn.Module]] = {}


def register_arch(name: str, factory: Callable[[int], nn.Module]) -> None:
    """Register a model architecture in code (never from bundle data)."""
    ARCH_REGISTRY[name] = factory


def _register_admodel() -> None:
    try:
        from admodel import models as _m   # fixed import; location is deployment config
    except ImportError:                     # service can still run with test archs
        return
    for n in ("lstm", "bilstm", "logbert"):
        register_arch(n, lambda vs, _n=n: _m.build(_n, vs))


_register_admodel()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelBundle:
    def __init__(self, model, vocab: Vocab, meta: dict, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vocab = vocab
        self.meta = meta
        self.window_size = int(meta.get("window_size", 20))
        # mask id: the training pipeline uses an extra embedding slot at index
        # len(vocab), recorded in meta. Fall back to that, else <unk>.
        self.mask_id = int(meta.get("mask_id", len(vocab)))

    @classmethod
    def load(cls, bundle_dir: Path, device: str = "cpu",
             expected_sha256: str | None = None) -> "ModelBundle":
        bundle_dir = Path(bundle_dir)
        weights = bundle_dir / "model.pt"
        meta = json.loads((bundle_dir / "meta.json").read_text())
        vocab = Vocab(json.loads((bundle_dir / "vocab.json").read_text()))

        actual = sha256_file(weights)
        if expected_sha256:
            if actual.lower() != expected_sha256.strip().lower():
                raise RuntimeError(
                    f"model.pt SHA-256 mismatch: expected {expected_sha256}, got {actual}")
            log.info("model.pt SHA-256 verified against trusted value")
        else:
            log.warning("no trusted SHA-256 supplied (ADSERVE_MODEL_SHA256); model.pt=%s "
                        "- integrity NOT verified", actual)

        arch = meta.get("arch") or meta.get("model")
        if arch not in ARCH_REGISTRY:
            raise RuntimeError(
                f"unknown architecture {arch!r}; allowed: {sorted(ARCH_REGISTRY)}. "
                "Is the 'admodel' package importable (PYTHONPATH / pip install)?")
        vocab_size = int(meta.get("vocab_size", len(vocab)))
        model = ARCH_REGISTRY[arch](vocab_size)

        # weights_only=True: tensors only, no arbitrary object deserialisation.
        state = torch.load(weights, map_location=device, weights_only=True)
        if not isinstance(state, dict):
            raise RuntimeError("model.pt must be a state_dict (re-export with make_bundle.py)")
        model.load_state_dict(state)           # strict: shape/key mismatch fails loudly
        return cls(model, vocab, meta, device=device)

    def _logits(self, batch: torch.Tensor) -> torch.Tensor:
        """Call the model and normalise output to logits [B, L, V].

        Handles models that return a bare tensor or a (logits, ...) tuple.
        """
        out = self.model(batch)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    @torch.no_grad()
    def score(self, window_ids: list[int]) -> float:
        """Max-surprisal over a full window (higher = more anomalous)."""
        L = len(window_ids)
        base = torch.tensor(window_ids, dtype=torch.long)
        # L copies of the window, one masked position each.
        batch = base.unsqueeze(0).repeat(L, 1).to(self.device)
        idx = torch.arange(L, device=self.device)
        true_tok = batch[idx, idx].clone()
        batch[idx, idx] = self.mask_id
        logits = self._logits(batch)                      # [L, L, V]
        logp = torch.log_softmax(logits, dim=-1)
        surpr = -logp[idx, idx, true_tok]                 # [L]
        return float(surpr.max().item())

    def explain(self, window_ids: list[int]) -> dict:
        """Per-position surprisal plus the peak token, for alert context."""
        L = len(window_ids)
        base = torch.tensor(window_ids, dtype=torch.long)
        batch = base.unsqueeze(0).repeat(L, 1).to(self.device)
        idx = torch.arange(L, device=self.device)
        true_tok = batch[idx, idx].clone()
        batch[idx, idx] = self.mask_id
        with torch.no_grad():
            logp = torch.log_softmax(self._logits(batch), dim=-1)
        surpr = (-logp[idx, idx, true_tok]).tolist()
        peak = int(max(range(L), key=lambda i: surpr[i]))
        return {
            "score": float(surpr[peak]),
            "peak_position": peak,
            "peak_token": self.vocab.id_to_token.get(window_ids[peak], "<unk>"),
            "per_position_surprisal": [round(s, 4) for s in surpr],
        }
