"""Train and evaluate the SUPERVISED LSTM and BiLSTM classifiers (proposal's primary
framing), for comparison with the benign-trained anomaly detectors from run_all.py.

    python run_supervised.py

Writes results_supervised.json + supervised_curves.png to results/. Run run_all.py too,
then compare: the supervised models are expected to struggle on the tiny positive class,
which is exactly why the benign-trained anomaly framing is the stronger choice here.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, roc_curve, precision_recall_curve)
from admodel import config, data as D, train_supervised as TS


def agg(v):
    v = [x for x in v if x is not None]
    return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} if v else None


def pick_thr(y, s):
    best, bf = float(np.min(s)), -1.0
    for t in np.unique(s):
        p, r, f, _ = precision_recall_fscore_support(y, (s >= t).astype(int),
                                                     average="binary", zero_division=0)
        if f > bf:
            bf, best = f, t
    return float(best)


@torch.no_grad()
def _scores(model, X):
    model.eval()
    xb = torch.as_tensor(np.asarray(X), dtype=torch.long, device=config.DEVICE)
    return torch.sigmoid(model(xb)).cpu().numpy()


def main():
    ds = D.load_dataset()
    vs = ds["vocab_size"]
    yv, yt = ds["y_val"].astype(int), ds["y_test"].astype(int)
    atk_test = [str(a) for a in ds["atk_test"]]
    print(f"device={config.DEVICE} train_windows={len(ds['X_train'])} "
          f"(attack {int(ds['y_train'].sum())}) test_attacks={int(yt.sum())}")

    summary, curves = {}, {}
    for name in ("lstm", "bilstm"):
        runs, last = [], None
        for seed in config.SEEDS:
            model, lc = TS.train_supervised(name, ds["X_train"], ds["y_train"],
                                            ds["X_val"], ds["y_val"], vs, seed)
            sv, st = _scores(model, ds["X_val"]), _scores(model, ds["X_test"])
            thr = pick_thr(yv, sv)
            pred = (st >= thr).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(yt, pred, average="binary", zero_division=0)
            tn = int(((pred == 0) & (yt == 0)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            per = {}
            for a in sorted(set(atk_test) - {"benign"}):
                m = np.array([x == a for x in atk_test])
                per[a] = round(float(pred[m].mean()), 3) if m.sum() else None
            runs.append({"f1": round(f1, 4), "precision": round(p, 4), "recall": round(r, 4),
                         "fpr": round(fpr, 4),
                         "roc_auc": round(roc_auc_score(yt, st), 4) if yt.sum() else None,
                         "pr_auc": round(average_precision_score(yt, st), 4) if yt.sum() else None,
                         "per_attack": per, "train_f1_gap": round(lc["val_f1"][-1], 4)})
            last = (lc, roc_curve(yt, st), precision_recall_curve(yt, st))
            print(f"  supervised {name} seed{seed}: F1={runs[-1]['f1']} P={runs[-1]['precision']} "
                  f"R={runs[-1]['recall']} FPR={runs[-1]['fpr']} ROC-AUC={runs[-1]['roc_auc']}")
        summary[name] = {k: agg([r[k] for r in runs]) for k in
                         ("f1", "precision", "recall", "fpr", "roc_auc", "pr_auc")}
        summary[name]["per_attack_recall_last"] = runs[-1]["per_attack"]
        summary[name]["params"] = sum(p.numel() for p in model.parameters())
        curves[name] = last

    (config.RESULTS_DIR / "results_supervised.json").write_text(json.dumps(summary, indent=2))
    # curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for name, (_, roc, _) in curves.items():
        plt.plot(roc[0], roc[1], label=f"{name} (sup)")
    plt.plot([0, 1], [0, 1], "k:"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (supervised, test)"); plt.legend()
    plt.subplot(1, 2, 2)
    for name, (_, _, pr) in curves.items():
        plt.plot(pr[1], pr[0], label=f"{name} (sup)")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (supervised, test)"); plt.legend()
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "supervised_curves.png", dpi=130); plt.close()

    print("\n===== SUPERVISED SUMMARY (mean +/- std over seeds) =====")
    for name, s in summary.items():
        print(f"{name:8s} F1 {s['f1']} | P {s['precision']} | R {s['recall']} | "
              f"FPR {s['fpr']} | ROC-AUC {s['roc_auc']} | PR-AUC {s['pr_auc']}")
    print(f"\nWrote results_supervised.json + supervised_curves.png to {config.RESULTS_DIR}")
    print("Compare with results.json (anomaly framing) for the full 5-way picture.")


if __name__ == "__main__":
    main()
