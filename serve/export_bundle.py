#!/usr/bin/env python3
"""Package a trained admodel checkpoint into a serving bundle.

Produces serve_bundle/{model.pt, meta.json, vocab.json} — the exact shape
adserve.scorer.ModelBundle.load() expects. Run it once per model you want to
serve (BiLSTM is usually the pick).

The service pickles the *whole* nn.Module so there is zero architecture drift.
Two input modes:

  (A) Whole-module checkpoint  (training did `torch.save(model, path)`)
        python export_bundle.py --checkpoint models/artifacts/bilstm.pt \
            --vocab models/artifacts/vocab.json --threshold 7.3 \
            --model bilstm --out serve_bundle

  (B) state_dict checkpoint    (training saved model.state_dict())
        Rebuilds the module from admodel first, so run this inside the repo
        with admodel importable and pass the builder args your training used:
        python export_bundle.py --checkpoint models/artifacts/bilstm_sd.pt \
            --state-dict --arch bilstm --vocab ... --threshold 7.3 --out serve_bundle

If you would rather not use this script, the README has a 6-line snippet to
drop at the end of train.py that writes the bundle directly from the live
objects already in scope.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def _load_whole_module(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "eval") and callable(getattr(obj, "eval")):
        return obj
    if isinstance(obj, dict) and hasattr(obj.get("model"), "eval"):
        return obj["model"]
    raise SystemExit(
        f"{path} is not a whole nn.Module. If it is a state_dict, pass "
        "--state-dict --arch <lstm|bilstm|logbert> so I can rebuild the module "
        "from admodel."
    )


def _rebuild_from_state_dict(path: Path, arch: str, vocab_size: int):
    """Reconstruct the module from admodel, then load weights into it."""
    try:
        from admodel.config import Config  # type: ignore
        from admodel import models as M     # type: ignore
    except Exception as e:  # pragma: no cover - env-specific
        raise SystemExit(
            "Could not import admodel to rebuild the module. Run this from the "
            f"repo root with the admodel package on PYTHONPATH. ({e})"
        )
    cfg = Config()
    setattr(cfg, "vocab_size", vocab_size)
    arch = arch.lower()
    if arch in ("lstm", "bilstm"):
        model = M.MaskedRNN(cfg, bidirectional=(arch == "bilstm"))
    elif arch == "logbert":
        model = M.LogBERT(cfg)
    else:
        raise SystemExit(f"unknown --arch {arch!r}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True, help="vocab.json (token->id)")
    ap.add_argument("--out", type=Path, default=Path("serve_bundle"))
    ap.add_argument("--model", default="model", help="label stored in meta.json")
    ap.add_argument("--threshold", type=float, required=True,
                    help="decision threshold frozen at train time (e.g. 99th pct benign)")
    ap.add_argument("--window-size", type=int, default=20)
    ap.add_argument("--log-key-field", default="rule.id",
                    help="MUST match build_dataset.py's event-key definition")
    ap.add_argument("--state-dict", action="store_true",
                    help="checkpoint is a state_dict, not a whole module")
    ap.add_argument("--arch", default=None, help="lstm|bilstm|logbert (with --state-dict)")
    args = ap.parse_args()

    vocab = json.loads(args.vocab.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    if args.state_dict:
        if not args.arch:
            raise SystemExit("--state-dict requires --arch")
        model = _rebuild_from_state_dict(args.checkpoint, args.arch, len(vocab))
    else:
        model = _load_whole_module(args.checkpoint)

    model.eval()
    torch.save(model, args.out / "model.pt")
    shutil.copyfile(args.vocab, args.out / "vocab.json")
    meta = {
        "model": args.model,
        "threshold": args.threshold,
        "window_size": args.window_size,
        "log_key_field": args.log_key_field,
        "vocab_size": len(vocab),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote bundle -> {args.out}")
    print(f"[export] meta: {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
