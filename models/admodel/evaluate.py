"""Anomaly scoring + metrics.

Score of a window = number of positions whose true log-key is NOT in the model's
top-g predictions when that position is masked (DeepLog/LogBERT style). For LogBERT the
normalised hypersphere distance is added. Threshold is picked on validation; the test set
is touched once.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, roc_curve, precision_recall_curve)
from . import config, data as D


@torch.no_grad()
def window_scores(model, name, X, mask_id, topg=config.TOPG, batch_windows=64):
    """Return a continuous anomaly score per window in X (N,)."""
    model.eval()
    X = torch.as_tensor(np.asarray(X), dtype=torch.long)
    N, W = X.shape
    scores = np.zeros(N, dtype=np.float32)
    for s in range(0, N, batch_windows):
        chunk = X[s:s + batch_windows]
        # expand each window into W masked copies -> (chunk*W, W)
        inps, trues = [], []
        for w in chunk:
            inp, tk = D.eval_mask_each_position(w, mask_id)
            inps.append(inp); trues.append(tk)
        inp = torch.cat(inps).to(config.DEVICE)
        true = torch.cat(trues).to(config.DEVICE)
        if name == "logbert":
            logits, dist_h = model(inp)
        else:
            logits = model(inp); dist_h = None
        # each row's masked position is row-index % W
        rows = torch.arange(inp.size(0), device=config.DEVICE)
        pos = rows % W
        masked_logits = logits[rows, pos]                   # (chunk*W, V)
        masked_true = true                                   # already (chunk*W,), aligned with rows
        # surprisal of the true key at each masked position = -log p(true)
        logp = torch.log_softmax(masked_logits, dim=-1)
        surpr = -logp[torch.arange(len(masked_true), device=config.DEVICE), masked_true]
        surpr = surpr.view(len(chunk), W)
        # window score = MOST anomalous event (max surprisal), so a single attack
        # event is not diluted by the surrounding benign events in the window.
        sc = surpr.max(dim=1).values.cpu().numpy()
        if name == "logbert":
            d = (dist_h - model.center).pow(2).sum(1).view(len(chunk), W).mean(1).cpu().numpy()
            sc = sc + d / (d.mean() + 1e-6)                 # add normalised hypersphere distance
        scores[s:s + len(chunk)] = sc
    return scores


def _metrics_at(y, score, thr):
    pred = (score >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "fpr": round(fpr, 4), "threshold": float(thr)}


def pick_threshold(y_val, s_val):
    thrs = np.unique(s_val)
    best, best_f1 = thrs.min(), -1
    for t in thrs:
        m = _metrics_at(y_val, s_val, t)
        if m["f1"] > best_f1:
            best_f1, best = m["f1"], t
    return float(best)


def evaluate(model, name, ds, mask_id):
    s_val = window_scores(model, name, ds["X_val"], mask_id)
    s_test = window_scores(model, name, ds["X_test"], mask_id)
    y_val, y_test = ds["y_val"].astype(int), ds["y_test"].astype(int)
    thr = pick_threshold(y_val, s_val)

    res = _metrics_at(y_test, s_test, thr)
    res["roc_auc"] = round(roc_auc_score(y_test, s_test), 4) if y_test.sum() else None
    res["pr_auc"] = round(average_precision_score(y_test, s_test), 4) if y_test.sum() else None

    # per-attack recall on test
    pred = (s_test >= thr).astype(int)
    per = {}
    for atk in set(ds["atk_test"]):
        if atk == "benign":
            continue
        m = ds["atk_test"] == atk
        per[atk] = round(float(pred[m].mean()), 4) if m.sum() else None
    res["per_attack_recall"] = per

    # latency (per window), single and batched
    res["latency_ms"] = _latency(model, name, ds["X_test"], mask_id)
    curves = {"roc": roc_curve(y_test, s_test) if y_test.sum() else None,
              "pr": precision_recall_curve(y_test, s_test) if y_test.sum() else None,
              "scores_test": s_test, "y_test": y_test}
    return res, curves


@torch.no_grad()
def _latency(model, name, X, mask_id, n=50):
    model.eval()
    X = torch.as_tensor(np.asarray(X[:n]), dtype=torch.long)
    # single-window
    t0 = time.perf_counter()
    for w in X:
        window_scores(model, name, w.unsqueeze(0).numpy(), mask_id, batch_windows=1)
    single = (time.perf_counter() - t0) / len(X) * 1000
    # batched
    t0 = time.perf_counter()
    window_scores(model, name, X.numpy(), mask_id, batch_windows=len(X))
    batched = (time.perf_counter() - t0) / len(X) * 1000
    return {"single_ms": round(single, 3), "batched_ms": round(batched, 3)}
