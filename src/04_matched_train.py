"""Matched augmentation: trains one NN per (technique, value, seed, fold).

Each model only ever sees its own (technique, value), applied to each training
sample with probability P_AUG (MatchedAugmentedSequence). The grid comes from
hsi_augmentations.grid_for.

The baseline (never-augmented NN) is NOT trained here: 04_matched_infer.py loads it from ../models/03 (03_multilabel_train.py, config "snv", same architecture and folds).

Models already on disk are skipped, so an interrupted run can be relaunched. Saves to ../models/04/{technique}/value{X}/seed{seed}/fold{k}/
  preprocessor.joblib   (SNV)
  nn.keras
"""

import argparse
from pathlib import Path

from joblib import Parallel, delayed
from sklearn.preprocessing import MultiLabelBinarizer

import hsi_augmentations as aug
import hsi_data as H
import hsi_models as Mo
import hsi_preprocessing as P
import hsi_splits as S

MODELS_DIR = Path("../models/04")
SEEDS      = range(42, 52)
P_AUG      = 0.5


def model_dir(technique, value, seed, k):
    return MODELS_DIR / technique / f"value{aug.value_str(technique, value)}" / f"seed{seed}" / f"fold{k}"


def _train_one(technique, value, seed, k, train_idx, val_idx, X_raw, Y, bio):
    H.set_seeds(seed)

    X_tr_raw = X_raw[train_idx]
    prep = P.Preprocessor("snv")
    X_tr = prep.fit_transform(X_tr_raw)
    X_va = prep.transform(X_raw[val_idx])

    nn  = Mo.build_fournier_nn(X_tr.shape[1], Y.shape[1], multilabel=True)
    gen = Mo.MatchedAugmentedSequence(X_tr_raw, Y[train_idx], bio[train_idx],
                                      technique, value, seed=seed + k, p_aug=P_AUG)
    Mo.train_nn(nn, X_tr, Y[train_idx], X_va, Y[val_idx], multilabel=True, generator=gen)

    out_dir = model_dir(technique, value, seed, k)
    out_dir.mkdir(parents=True, exist_ok=True)
    prep.save(out_dir / "preprocessor.joblib")
    Mo.save_nn(nn, out_dir / "nn.keras")
    Mo.clear_session()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="../data/cyanobacterial_spectra_processed_clean.xlsx")
    p.add_argument("--n_jobs", type=int, default=-1)
    args = p.parse_args()

    d   = H.load_data(args.file, "6class")
    mlb = MultiLabelBinarizer(classes=H.GENERA).fit([H.GENERA])

    y_ml = []
    for genus, pure in zip(d.y, d.is_pure):
        parts = [genus] if pure else genus.split("_")
        y_ml.append(sorted({g for g in parts if g in H.GENERA}))
    Y = mlb.transform(y_ml).astype("float32")

    n_bands = len(d.wavelengths)
    print(f"{n_bands} spectral bands after trimming")
    grids = {t: aug.grid_for(t, n_bands) for t in aug.TECHNIQUES}

    tasks = [(technique, value, seed, k, f["train"], f["val"])
             for seed in SEEDS
             for k, f in enumerate(S.make_folds(d, "6class", seed))
             for technique in aug.TECHNIQUES
             for value in grids[technique]]
    todo = [t for t in tasks if not (model_dir(*t[:4]) / "nn.keras").exists()]

    print(f"{len(todo)} models to train, {len(tasks) - len(todo)} already on disk "
          f"({sum(aug.GRID_SIZE.values())} (technique, value) x {len(SEEDS)} seeds x {S.N_FOLDS} folds), "
          f"n_jobs={args.n_jobs}")

    Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_train_one)(technique, value, seed, k, train_idx, val_idx, d.X, Y, d.bio_types)
        for technique, value, seed, k, train_idx, val_idx in todo
    )

    print(f"\nDone. Models in {MODELS_DIR}")
