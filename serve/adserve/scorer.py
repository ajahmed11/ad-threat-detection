"""Load the trained model bundle and score a window with max-surprisal.

Scoring mirrors admodel/evaluate.py exactly: mask each position in turn,
read the model's log-probability for the token that was really there, and take
the single most surprising position as the window score. A window is anomalous
when its score crosses the threshold frozen at training time.

The model architecture is NOT re-implemented here. export_bundle.py pickles
the already-constructed module (admodel.models.MaskedRNN / LogBERT) into
model.pt, so the service stays in lock-step with whatever was trained.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .featurize import Vocab


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
    def load(cls, bundle_dir: Path, device: str = "cpu") -> "ModelBundle":
        import sys
        bundle_dir = Path(bundle_dir)
        meta = json.loads((bundle_dir / "meta.json").read_text())
        vocab = Vocab(json.loads((bundle_dir / "vocab.json").read_text()))
        # The pickled module references admodel.models.{MaskedRNN,LogBERT}; make
        # that package importable. meta records where it lives at build time.
        ap = meta.get("admodel_path")
        if ap and ap not in sys.path:
            sys.path.insert(0, ap)
        # weights_only=False: the bundle is a trusted artifact we produced.
        model = torch.load(bundle_dir / "model.pt", map_location=device, weights_only=False)
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
