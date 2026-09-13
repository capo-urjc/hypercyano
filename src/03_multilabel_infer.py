"""Evaluates the multi-label models from ../models/03/."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

import hsi_data as H
import hsi_metrics as M
import hsi_models as Mo
import hsi_preprocessing as P
import hsi_splits as S

MODELS_DIR  = Path("../models/03")
RESULTS_DIR = Path("../results/03")
CONFIGS   = ["fournier", "snv"]
SEEDS     = range(42, 52)
THRESHOLD = 0.5
METRICS   = ("accuracy", "f1", "kappa", "exact_match")


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

    rows = []
    for seed in SEEDS:
        folds = S.make_folds(d, "6class", seed)

        for config in CONFIGS:
            for k, f in enumerate(folds):
                path = MODELS_DIR / "multilabel" / f"seed{seed}" / config / f"fold{k}"
                prep = P.Preprocessor.load(path / "preprocessor.joblib")
                X_te = prep.transform(d.X[f["test"]])
                Y_te = Y[f["test"]]

                rf = Mo.load_rf(path / "rf.joblib")
                rows.append({"config": config, "model": "RF", "seed": seed, "fold": k,
                             **M.compute_metrics(Y_te, rf.predict(X_te), multilabel=True)})

                nn = Mo.load_nn(path / "nn.keras")
                Y_pred = (nn.predict(X_te, verbose=0) >= THRESHOLD).astype(int)
                rows.append({"config": config, "model": "NN", "seed": seed, "fold": k,
                             **M.compute_metrics(Y_te, Y_pred, multilabel=True)})
                Mo.clear_session()

    out = RESULTS_DIR / "multilabel"
    out.mkdir(parents=True, exist_ok=True)
    summary = M.summarise(rows, group_keys=["config", "model"], metrics=METRICS)
    M.print_results(summary, group_keys=["config", "model"], metrics=METRICS)
    summary.to_csv(out / "summary.csv", index=False)
