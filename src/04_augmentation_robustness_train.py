"""Robustness study: trains Baseline (SNV only) and Robust (augment -> SNV) NN on the multi-label task."""

import argparse
from pathlib import Path

from sklearn.preprocessing import MultiLabelBinarizer

import hsi_data as H
import hsi_models as Mo
import hsi_preprocessing as P
import hsi_splits as S

MODELS_DIR = Path("../models/04")
VARIANTS = ["baseline", "robust"]
SEEDS    = range(42, 52)


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

    for seed in SEEDS:
        H.set_seeds(seed)
        folds = S.make_folds(d, "6class", seed)

        for k, f in enumerate(folds):
            X_tr_raw = d.X[f["train"]]
            bio_tr   = d.bio_types[f["train"]]

            prep = P.Preprocessor("snv")
            X_tr = prep.fit_transform(X_tr_raw)
            X_va = prep.transform(d.X[f["val"]])

            for variant in VARIANTS:
                path = MODELS_DIR / "multilabel" / f"seed{seed}" / variant / f"fold{k}"
                path.mkdir(parents=True, exist_ok=True)
                prep.save(path / "preprocessor.joblib")

                nn = Mo.build_fournier_nn(X_tr.shape[1], len(H.GENERA), multilabel=True)

                if variant == "baseline":
                    Mo.train_nn(nn, X_tr, Y[f["train"]], X_va, Y[f["val"]], multilabel=True)
                else:
                    gen = Mo.AugmentedSequence(X_tr_raw, Y[f["train"]], bio_tr, seed=seed + k)
                    Mo.train_nn(nn, X_tr, Y[f["train"]], X_va, Y[f["val"]], multilabel=True, generator=gen)

                Mo.save_nn(nn, path / "nn.keras")
                Mo.clear_session()

            print(f"  multilabel seed{seed} fold {k} saved (baseline, robust)")
