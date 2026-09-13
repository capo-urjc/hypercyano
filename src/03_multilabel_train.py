"""Multi-label genus presence: trains RF and NN (sigmoid output) for fournier and snv configs."""

import argparse
from pathlib import Path

from sklearn.preprocessing import MultiLabelBinarizer

import hsi_data as H
import hsi_models as Mo
import hsi_preprocessing as P
import hsi_splits as S

MODELS_DIR = Path("../models/03")
CONFIGS = ["fournier", "snv"]
SEEDS   = range(42, 52)


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

        for config in CONFIGS:
            for k, f in enumerate(folds):
                prep = P.Preprocessor(config)
                X_tr = prep.fit_transform(d.X[f["train"]])
                X_va = prep.transform(d.X[f["val"]])

                path = MODELS_DIR / "multilabel" / f"seed{seed}" / config / f"fold{k}"
                path.mkdir(parents=True, exist_ok=True)
                prep.save(path / "preprocessor.joblib")

                rf = Mo.train_rf(X_tr, Y[f["train"]], seed)
                Mo.save_rf(rf, path / "rf.joblib")

                nn = Mo.build_fournier_nn(X_tr.shape[1], len(H.GENERA), multilabel=True)
                Mo.train_nn(nn, X_tr, Y[f["train"]], X_va, Y[f["val"]], multilabel=True)
                Mo.save_nn(nn, path / "nn.keras")
                Mo.clear_session()

                print(f"  multilabel seed{seed} {config} fold {k} saved")
