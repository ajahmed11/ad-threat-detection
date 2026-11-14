"""Training loop: masked log-key prediction on BENIGN windows only.
Early stopping on benign validation loss. Returns the best model and its loss curves.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from . import config, data as D


def _step(model, name, x, mask_id, vocab_size, ce):
    x = x.to(config.DEVICE)
    inp, tgt, _ = D.mask_batch(x, mask_id, vocab_size)
    inp, tgt = inp.to(config.DEVICE), tgt.to(config.DEVICE)
    if name == "logbert":
        logits, dist_h = model(inp)
        loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        vhm = (dist_h - model.center).pow(2).sum(1).mean()
        loss = loss + config.VHM_WEIGHT * vhm
    else:
        logits = model(inp)
        loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    return loss


def train_model(name, X_benign_train, X_benign_val, vocab_size, mask_id, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    from . import models
    model = models.build(name, vocab_size).to(config.DEVICE)

    # initialise LogBERT centre from an untrained forward pass over benign train
    if name == "logbert":
        model.eval()
        with torch.no_grad():
            xb = torch.as_tensor(X_benign_train[:512], dtype=torch.long, device=config.DEVICE)
            _, dh = model(xb)
            model.center.copy_(dh.mean(0))

    tr = DataLoader(D.WindowDS(X_benign_train), batch_size=config.BATCH, shuffle=True)
    va = DataLoader(D.WindowDS(X_benign_val), batch_size=config.BATCH)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    best, best_val, bad = None, float("inf"), 0
    tr_curve, va_curve = [], []
    for ep in range(config.EPOCHS):
        model.train(); tl = []
        for x in tr:
            opt.zero_grad()
            loss = _step(model, name, x, mask_id, vocab_size, ce)
            loss.backward(); opt.step(); tl.append(loss.item())
        model.eval(); vl = []
        with torch.no_grad():
            for x in va:
                vl.append(_step(model, name, x, mask_id, vocab_size, ce).item())
        tr_curve.append(float(np.mean(tl))); va_curve.append(float(np.mean(vl)))
        if va_curve[-1] < best_val - 1e-4:
            best_val, best, bad = va_curve[-1], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= config.PATIENCE:
                break
    model.load_state_dict(best)
    return model, {"train_loss": tr_curve, "val_loss": va_curve, "epochs": len(tr_curve)}
