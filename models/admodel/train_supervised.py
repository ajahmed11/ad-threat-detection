"""Supervised training for the LSTM/BiLSTM classifiers (proposal's primary framing).
Class-weighted BCE on all training windows; early stopping on validation F1."""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from . import config, models


def _best_val_f1(y, s):
    best = -1.0
    for t in np.unique(s):
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > best:
            best = f
    return best


def train_supervised(name, X_train, y_train, X_val, y_val, vocab_size, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = models.SupervisedRNN(vocab_size, bidirectional=(name == "bilstm")).to(config.DEVICE)
    Xtr = torch.as_tensor(np.asarray(X_train), dtype=torch.long)
    ytr = torch.as_tensor(np.asarray(y_train), dtype=torch.float)
    Xva = torch.as_tensor(np.asarray(X_val), dtype=torch.long).to(config.DEVICE)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=config.BATCH, shuffle=True)

    pos = max(int(np.sum(y_train)), 1)
    neg = len(y_train) - pos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=config.DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    best, best_f1, bad = None, -1.0, 0
    tr_curve, va_curve = [], []
    for ep in range(config.EPOCHS):
        model.train(); tl = []
        for xb, yb in loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step(); tl.append(loss.item())
        model.eval()
        with torch.no_grad():
            sv = torch.sigmoid(model(Xva)).cpu().numpy()
        vf1 = _best_val_f1(np.asarray(y_val).astype(int), sv)
        tr_curve.append(float(np.mean(tl))); va_curve.append(float(vf1))
        if vf1 > best_f1 + 1e-4:
            best_f1, best, bad = vf1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= config.PATIENCE:
                break
    model.load_state_dict(best)
    return model, {"train_loss": tr_curve, "val_f1": va_curve, "epochs": len(tr_curve)}
