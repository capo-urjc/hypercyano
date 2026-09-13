"""OOD corruption sweep: evaluates Baseline vs Robust NN across techniques and severities."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hsi_augmentations as aug
import hsi_data as H
import hsi_metrics as M
import hsi_models as Mo
import hsi_preprocessing as P
import hsi_splits as S

MODELS_DIR  = Path("../models/04")
RESULTS_DIR = Path("../results/04")
VARIANTS   = ["baseline", "robust"]
SEVERITIES = np.linspace(0.0, 1.0, 11)
THRESHOLD  = 0.5
COLORS     = {"baseline": "#e74c3c", "robust": "#2ecc71"}
SEEDS      = range(42, 52)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="../data/cyanobacterial_spectra_processed_clean.xlsx")
    args = p.parse_args()

    d   = H.load_data(args.file, "6class")
    mlb = MultiLabelBinarizer(classes=H.GENERA).fit([H.GENERA])

    y_ml = []
    for genus, pure in zip(d.y, d.is_pure):
        parts = [genus] if pure else genus.split("_")
        y_ml.append(sorted({g for g in parts if g in H.GENERA}))
    Y = mlb.transform(y_ml).astype("float32")

    out = RESULTS_DIR / "multilabel"
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in SEEDS:
        folds = S.make_folds(d, "6class", seed)

        for k, f in enumerate(folds):
            X_tr_raw = d.X[f["train"]]
            X_te_raw = d.X[f["test"]]
            bio_tr   = d.bio_types[f["train"]]
            bio_te   = d.bio_types[f["test"]]
            Y_te     = Y[f["test"]]

            prep   = P.Preprocessor.load(MODELS_DIR / "multilabel" / f"seed{seed}" / "baseline" / f"fold{k}" / "preprocessor.joblib")
            models = {v: Mo.load_nn(MODELS_DIR / "multilabel" / f"seed{seed}" / v / f"fold{k}" / "nn.keras") for v in VARIANTS}

            for tech_idx, technique in enumerate(aug.TECHNIQUES):
                for sev_idx, severity in enumerate(SEVERITIES):
                    rng = np.random.default_rng(
                        np.random.SeedSequence([seed, k, tech_idx, sev_idx]).generate_state(1)[0]
                    )
                    X_corr = np.array([
                        aug.corrupt_one(X_te_raw[i], rng, technique, severity,
                                        X_tr_raw, bio_tr, bio_te[i])
                        for i in range(len(X_te_raw))
                    ])
                    X_te = prep.transform(X_corr)
                    for variant in VARIANTS:
                        Y_pred = (models[variant].predict(X_te, verbose=0) >= THRESHOLD).astype(int)
                        rows.append({"variant": variant, "technique": technique,
                                     "severity": severity, "seed": seed, "fold": k,
                                     **M.compute_metrics(Y_te, Y_pred, multilabel=True)})

            Mo.clear_session()
            print(f"  seed{seed} fold {k} done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "raw.csv", index=False)

    summary = df.groupby(["variant", "technique", "severity"]).agg(
        accuracy_mean=("accuracy", "mean"),    accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"),                f1_std=("f1", "std"),
        kappa_mean=("kappa", "mean"),          kappa_std=("kappa", "std"),
        exact_match_mean=("exact_match", "mean"), exact_match_std=("exact_match", "std"),
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    for technique in aug.TECHNIQUES:
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant in VARIANTS:
            sub = summary[(summary.technique == technique) & (summary.variant == variant)]
            ax.plot(sub.severity * 100, sub.exact_match_mean * 100, label=variant, color=COLORS[variant])
            ax.fill_between(sub.severity * 100,
                            (sub.exact_match_mean - sub.exact_match_std) * 100,
                            (sub.exact_match_mean + sub.exact_match_std) * 100,
                            color=COLORS[variant], alpha=0.15)
        ax.set_title(f"Robustness: {technique}")
        ax.set_xlabel("Severity (%)"); ax.set_ylabel("Exact match (%)")
        ax.set_ylim(0, 105); ax.legend(); ax.grid(linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(out / f"robustness_{technique}.png", dpi=150)
        plt.close(fig)
        print(f"plot saved: robustness_{technique}.png")
