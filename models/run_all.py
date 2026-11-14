"""Train and evaluate LSTM, BiLSTM and LogBERT over several seeds, then write a
results table and loss/ROC/PR curves. Run on the laptop (GPU recommended).

    python run_all.py
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from admodel import config, data as D, train, evaluate


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}


def main():
    ds = D.load_dataset()
    vs, mask_id = ds["vocab_size"], ds["mask_id"]
    Xtr_ben = ds["X_train"][ds["y_train"] == 0]
    Xva_ben = ds["X_val"][ds["y_val"] == 0]
    print(f"device={config.DEVICE} vocab={vs} benign_train={len(Xtr_ben)} "
          f"benign_val={len(Xva_ben)} test_windows={len(ds['X_test'])} "
          f"(attacks {int(ds['y_test'].sum())})")

    # run all three by default, or just the model(s) named on the command line
    # e.g.  python run_all.py lstm      /  python run_all.py bilstm logbert
    chosen = tuple(a for a in sys.argv[1:] if a in ("lstm", "bilstm", "logbert")) \
             or ("lstm", "bilstm", "logbert")
    summary, all_curves = {}, {}
    for name in chosen:
        runs, last_curves = [], None
        for seed in config.SEEDS:
            model, lc = train.train_model(name, Xtr_ben, Xva_ben, vs, mask_id, seed)
            res, curves = evaluate.evaluate(model, name, ds, mask_id)
            res["epochs"] = lc["epochs"]
            runs.append(res)
            last_curves = (lc, curves)
            print(f"  {name} seed{seed}: F1={res['f1']} P={res['precision']} "
                  f"R={res['recall']} FPR={res['fpr']} PR-AUC={res['pr_auc']}")
        summary[name] = {
            "f1": agg([r["f1"] for r in runs]),
            "precision": agg([r["precision"] for r in runs]),
            "recall": agg([r["recall"] for r in runs]),
            "fpr": agg([r["fpr"] for r in runs]),
            "roc_auc": agg([r["roc_auc"] for r in runs]),
            "pr_auc": agg([r["pr_auc"] for r in runs]),
            "latency_single_ms": agg([r["latency_ms"]["single_ms"] for r in runs]),
            "latency_batched_ms": agg([r["latency_ms"]["batched_ms"] for r in runs]),
            "per_attack_recall_last": runs[-1]["per_attack_recall"],
            "params": sum(p.numel() for p in model.parameters()),
        }
        all_curves[name] = last_curves

    (config.RESULTS_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    _plots(all_curves)
    print("\n===== SUMMARY (mean +/- std over seeds) =====")
    for name, s in summary.items():
        print(f"{name:8s} F1 {s['f1']} | P {s['precision']} | R {s['recall']} | "
              f"FPR {s['fpr']} | PR-AUC {s['pr_auc']} | {s['params']:,} params")
    print(f"\nWrote results.json + curves to {config.RESULTS_DIR}")


def _plots(all_curves):
    # loss curves
    plt.figure(figsize=(7, 4))
    for name, (lc, _) in all_curves.items():
        plt.plot(lc["train_loss"], label=f"{name} train")
        plt.plot(lc["val_loss"], "--", label=f"{name} val")
    plt.xlabel("epoch"); plt.ylabel("masked-CE loss"); plt.legend(fontsize=7); plt.title("Loss curves")
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "loss_curves.png", dpi=130); plt.close()
    # ROC + PR
    for kind, idx in (("roc", "roc"), ("pr", "pr")):
        plt.figure(figsize=(5, 5))
        for name, (_, cv) in all_curves.items():
            c = cv[kind]
            if c is None:
                continue
            if kind == "roc":
                fpr, tpr, _ = c; plt.plot(fpr, tpr, label=name)
            else:
                pr, rc, _ = c; plt.plot(rc, pr, label=name)
        if kind == "roc":
            plt.plot([0, 1], [0, 1], "k:", lw=.8); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (test)")
        else:
            plt.xlabel("recall"); plt.ylabel("precision"); plt.title("Precision-Recall (test)")
        plt.legend(); plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / f"{kind}_curve.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
