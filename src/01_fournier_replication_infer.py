"""Loads models from ../models/01/ and reports metrics for both split protocols."""

import argparse
import joblib
from pathlib import Path

import numpy as np

import hsi_data as H
import hsi_metrics as M
import hsi_models as Mo
import hsi_preprocessing as P
import hsi_splits as S

MODELS_DIR  = Path("../models/01")
RESULTS_DIR = Path("../results/01")
SPLITS = ["spectra", "images"]
TASKS  = ["6class", "12class"]
SEEDS  = range(42, 52)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="../data/cyanobacterial_spectra_processed_clean.xlsx")
    args = p.parse_args()

    for task in TASKS:
        d    = H.load_data(args.file, task)
        rows = []

        for seed in SEEDS:
            for split in SPLITS:
                folds = [S.split_spectra(d.y, seed)] if split == "spectra" else S.make_folds(d, task, seed)

                for k, f in enumerate(folds):
                    path = MODELS_DIR / task / f"seed{seed}" / split / f"fold{k}"
                    prep = P.Preprocessor.load(path / "preprocessor.joblib")
                    le   = joblib.load(path / "label_encoder.joblib")
                    X_te = prep.transform(d.X[f["test"]])
                    y_te = le.transform(d.y[f["test"]])

                    for name, model, predict in [
                        ("RF", Mo.load_rf(path / "rf.joblib"),  lambda m, X: m.predict(X)),
                        ("NN", Mo.load_nn(path / "nn.keras"),   lambda m, X: np.argmax(m.predict(X, verbose=0), axis=1)),
                    ]:
                        y_pred = predict(model, X_te)
                        rows.append({"split": split, "model": name, "seed": seed, "fold": k,
                                     **M.compute_metrics(y_te, y_pred)})
                    Mo.clear_session()

        out = RESULTS_DIR / task
        out.mkdir(parents=True, exist_ok=True)
        summary = M.summarise(rows, group_keys=["split", "model"])
        print(f"\n===== {task} =====")
        M.print_results(summary, group_keys=["split", "model"])
        summary.to_csv(out / "summary.csv", index=False)
