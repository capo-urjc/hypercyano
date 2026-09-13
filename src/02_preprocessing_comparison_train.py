"""Preprocessing comparison (fournier vs snv) at the image-level split: trains RF and NN for both configs."""

import argparse
import joblib
from pathlib import Path

from sklearn.preprocessing import LabelEncoder

import hsi_data as H
import hsi_models as Mo
import hsi_preprocessing as P
import hsi_splits as S

MODELS_DIR = Path("../models/02")
CONFIGS = ["fournier", "snv"]
TASKS   = ["6class", "12class"]
SEEDS   = range(42, 52)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="../data/cyanobacterial_spectra_processed_clean.xlsx")
    args = p.parse_args()

    for task in TASKS:
        d  = H.load_data(args.file, task)
        le = LabelEncoder().fit(d.class_names)

        for seed in SEEDS:
            H.set_seeds(seed)
            folds = S.make_folds(d, task, seed)

            for config in CONFIGS:
                for k, f in enumerate(folds):
                    prep = P.Preprocessor(config)
                    X_tr = prep.fit_transform(d.X[f["train"]])
                    X_va = prep.transform(d.X[f["val"]])
                    y_tr = le.transform(d.y[f["train"]])
                    y_va = le.transform(d.y[f["val"]])

                    path = MODELS_DIR / task / f"seed{seed}" / config / f"fold{k}"
                    path.mkdir(parents=True, exist_ok=True)
                    prep.save(path / "preprocessor.joblib")
                    joblib.dump(le, path / "label_encoder.joblib")

                    rf = Mo.train_rf(X_tr, y_tr, seed)
                    Mo.save_rf(rf, path / "rf.joblib")

                    nn = Mo.build_fournier_nn(X_tr.shape[1], len(le.classes_))
                    Mo.train_nn(nn, X_tr, y_tr, X_va, y_va,
                                epochs=Mo.NN_EPOCHS_FOURNIER, batch_size=Mo.NN_BATCH_FOURNIER, callbacks=[])
                    Mo.save_nn(nn, path / "nn.keras")
                    Mo.clear_session()

                    print(f"  {task} seed{seed} {config} fold {k} saved")
