"""Dataset loading and masking for benign-only sequence modelling."""
from __future__ import annotations
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from . import config


def load_dataset():
    d = np.load(config.WINDOWS_NPZ, allow_pickle=True)
    vocab = json.loads(config.VOCAB_JSON.read_text())
    out = {k: d[k] for k in d.files}
    out["vocab"] = vocab
    out["vocab_size"] = len(vocab)
    out["mask_id"] = len(vocab)        # extra token id for <mask>
    out["pad_id"] = 0                  # index 0 is <pad> by construction
    return out


class WindowDS(Dataset):
    """Holds integer windows (N x W). Returns a LongTensor per window."""
    def __init__(self, X):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]


def mask_batch(batch, mask_id, vocab_size, mask_prob=config.MASK_PROB):
    """BERT-style masking for training. Returns (inputs, targets, mask) where
    targets hold the original key at masked positions and -100 elsewhere."""
    x = batch.clone()
    targets = torch.full_like(x, -100)
    probs = torch.rand(x.shape)
    sel = (probs < mask_prob) & (x != config.__dict__.get("PAD_ID", 0))
    # guarantee at least one masked position per row
    for r in range(x.shape[0]):
        if sel[r].sum() == 0:
            sel[r, torch.randint(0, x.shape[1], (1,))] = True
    targets[sel] = x[sel]
    # 80% -> <mask>, 10% -> random, 10% -> unchanged (BERT recipe)
    r = torch.rand(x.shape)
    x[sel & (r < 0.8)] = mask_id
    rand_sel = sel & (r >= 0.9)
    x[rand_sel] = torch.randint(2, vocab_size, (int(rand_sel.sum()),))
    return x, targets, sel


def eval_mask_each_position(window, mask_id):
    """For scoring: produce W copies of a window, each with one position masked.
    Returns (inputs W x W, true_keys W)."""
    W = window.shape[0]
    inp = window.unsqueeze(0).repeat(W, 1).clone()
    idx = torch.arange(W)
    true_keys = window.clone()
    inp[idx, idx] = mask_id
    return inp, true_keys
