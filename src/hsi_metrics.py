import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, hamming_loss


def compute_metrics(y_true, y_pred, multilabel=False):
    if not multilabel:
        acc = accuracy_score(y_true, y_pred)
        return {"accuracy": acc,
                "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "kappa": cohen_kappa_score(y_true, y_pred),
                "exact_match": acc}

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    kappas = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) > 1 or len(np.unique(y_pred[:, j])) > 1:
            kappas.append(cohen_kappa_score(y_true[:, j], y_pred[:, j]))
    return {"accuracy": 1.0 - hamming_loss(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
            "kappa": float(np.mean(kappas)) if kappas else 0.0,
            "exact_match": accuracy_score(y_true, y_pred)}


def summarise(rows, group_keys, metrics=("accuracy", "f1", "kappa")):
    df = pd.DataFrame(rows)
    out = df.groupby(list(group_keys)).agg({m: ["mean", "std"] for m in metrics})
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


def print_results(summary, group_keys, metrics=("accuracy", "f1", "kappa")):
    header = "".join(f"{k:<20}" for k in group_keys) + "".join(f"{m:>18}" for m in metrics)
    print("\n" + header)
    print("-" * len(header))
    for _, r in summary.iterrows():
        line = "".join(f"{str(r[k]):<20}" for k in group_keys)
        for m in metrics:
            line += f"{r[f'{m}_mean']:.3f}±{r[f'{m}_std']:.3f}".rjust(18)
        print(line)
