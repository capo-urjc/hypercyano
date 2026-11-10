"""
Distortion, real variability and accuracy of the baseline (../models/03) and matched (../models/04) NNs.

Writes CSVs to ../results/04/
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import MultiLabelBinarizer

import hsi_augmentations as aug
import hsi_data as H
import hsi_metrics as M
import hsi_models as Mo
import hsi_preprocessing as P
import hsi_splits as S

MODELS_DIR   = Path("../models/04")
BASELINE_DIR = Path("../models/03")
RESULTS_DIR  = Path("../results/04")
SEEDS     = range(42, 52)
THRESHOLD = 0.5
METRICS   = ("accuracy", "f1", "kappa", "exact_match")
N_REPEATS = 10
SEED      = 42


def model_dir(technique, value, seed, k):
    return MODELS_DIR / technique / f"value{aug.value_str(technique, value)}" / f"seed{seed}" / f"fold{k}"



def _distortion_worker(tech_idx, technique, val_idx, value, X, bio_types, wavelengths):
    rows = []
    for i in range(len(X)):
        x, bio_type = X[i], bio_types[i]
        x_snv = P.snv(x[None, :])[0]
        for rep in range(N_REPEATS):
            rng = np.random.default_rng(
                np.random.SeedSequence([SEED, tech_idx, val_idx, i, rep]).generate_state(1)[0]
            )
            x_p = aug.apply(x.copy(), rng, technique, value, X, bio_types, bio_type)
            rows.append({
                "technique": technique, "value": value,
                "spectrum_idx": i, "repeat": rep,
                "grad_rmse": M.grad_rmse(x_snv, P.snv(x_p[None, :])[0], wavelengths),
            })
    return rows


def compute_distortion(d, grids, n_jobs):
    tasks = [(tech_idx, technique, val_idx, value)
             for tech_idx, technique in enumerate(aug.TECHNIQUES)
             for val_idx, value in enumerate(grids[technique])]
    print(f"  distortion: {len(tasks)} (technique, value) tasks, n_jobs={n_jobs}")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_distortion_worker)(tech_idx, technique, val_idx, value,
                                    d.X, d.bio_types, d.wavelengths)
        for tech_idx, technique, val_idx, value in tasks
    )
    return pd.DataFrame([row for chunk in results for row in chunk])


def _variability_worker(cls_idx, X, image_ids, wavelengths):
    """grad_rmse between every pair of spectra of one class: within the same
    image (pdist) and between different images (cdist). Euclidean distance of
    the derivatives / sqrt(n_bands) == M.grad_rmse of each pair."""
    by_image = {}
    for pos in cls_idx:
        by_image.setdefault(int(image_ids[pos]), []).append(int(pos))

    same, diff = [], []
    for members in by_image.values():
        if len(members) < 2:
            continue
        gA = np.gradient(P.snv(X[members]), wavelengths, axis=1)
        same.append(pdist(gA) / np.sqrt(gA.shape[1]))

    for a, b in itertools.combinations(by_image, 2):
        gA = np.gradient(P.snv(X[by_image[a]]), wavelengths, axis=1)
        gB = np.gradient(P.snv(X[by_image[b]]), wavelengths, axis=1)
        diff.append(cdist(gA, gB).ravel() / np.sqrt(gA.shape[1]))

    return (np.concatenate(same) if same else np.array([]),
            np.concatenate(diff) if diff else np.array([]))


def compute_real_variability(d, n_jobs):
    # classes = 12class labels (d.bio_types)
    classes = [c for c in np.unique(d.bio_types) if (d.bio_types == c).sum() >= 2]
    print(f"  variability: {len(classes)} classes, n_jobs={n_jobs}")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_variability_worker)(np.where(d.bio_types == c)[0], d.X, d.image_ids, d.wavelengths)
        for c in classes
    )
    same = np.concatenate([s for s, _ in results]) if results else np.array([])
    diff = np.concatenate([x for _, x in results]) if results else np.array([])
    return same, diff


def summarise_variability(same, diff):
    return pd.DataFrame([
        {"pair_type": pair_type, "n": int(len(vals)),
         "mean": float(np.mean(vals)) if len(vals) else np.nan,
         "std":  float(np.std(vals))  if len(vals) else np.nan}
        for pair_type, vals in [("same_image", same), ("different_image", diff)]
    ])


def _predict(model, prep, X):
    return (model.predict(prep.transform(X), verbose=0) >= THRESHOLD).astype(int)


def _accuracy_one_seed(seed, d, Y, grids):
    rows = []
    for k, f in enumerate(S.make_folds(d, "6class", seed)):
        base_dir = BASELINE_DIR / "multilabel" / f"seed{seed}" / "snv" / f"fold{k}"
        baseline_model = Mo.load_nn(base_dir / "nn.keras")
        baseline_prep  = P.Preprocessor.load(base_dir / "preprocessor.joblib")

        X_tr_raw = d.X[f["train"]]
        X_te_raw = d.X[f["test"]]
        bio_tr   = d.bio_types[f["train"]]
        bio_te   = d.bio_types[f["test"]]
        Y_te     = Y[f["test"]]

        for tech_idx, technique in enumerate(aug.TECHNIQUES):
            for val_idx, value in enumerate(grids[technique]):
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, k, tech_idx, val_idx]).generate_state(1)[0]
                )
                X_corr = np.array([
                    aug.apply(X_te_raw[i], rng, technique, value, X_tr_raw, bio_tr, bio_te[i])
                    for i in range(len(X_te_raw))
                ])

                mdir = model_dir(technique, value, seed, k)
                matched_model = Mo.load_nn(mdir / "nn.keras")
                matched_prep  = P.Preprocessor.load(mdir / "preprocessor.joblib")

                for variant, model, prep in [("baseline", baseline_model, baseline_prep),
                                             ("matched",  matched_model,  matched_prep)]:
                    rows.append({"variant": variant, "technique": technique, "value": value,
                                 "seed": seed, "fold": k,
                                 **M.compute_metrics(Y_te, _predict(model, prep, X_corr), multilabel=True)})
                del matched_model

        del baseline_model
        Mo.clear_session()
    print(f"  accuracy: seed{seed} done")
    return rows


def compute_accuracy(d, Y, grids, n_jobs):
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_accuracy_one_seed)(seed, d, Y, grids) for seed in SEEDS
    )
    return pd.DataFrame([row for chunk in results for row in chunk])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="../data/cyanobacterial_spectra_processed_clean.xlsx")
    p.add_argument("--n_jobs", type=int, default=-1,
                   help="Distortion and real variability.")
    p.add_argument("--n_jobs_accuracy", type=int, default=5,
                   help="One job per seed.")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    d = H.load_data(args.file, "6class")
    grids = {t: aug.grid_for(t, len(d.wavelengths)) for t in aug.TECHNIQUES}

    print("Distortion (grad_rmse after SNV)...")
    dist_raw = compute_distortion(d, grids, args.n_jobs)
    dist_raw.to_csv(RESULTS_DIR / "distortion_raw.csv", index=False)
    M.summarise(dist_raw, group_keys=["technique", "value"], metrics=("grad_rmse",)) \
        .to_csv(RESULTS_DIR / "distortion_summary.csv", index=False)

    print("Real variability (12class, grad_rmse, all pairs)...")
    same, diff = compute_real_variability(d, args.n_jobs)
    var_summary = summarise_variability(same, diff)
    var_summary.to_csv(RESULTS_DIR / "variability_summary.csv", index=False)
    print(var_summary)

    mlb = MultiLabelBinarizer(classes=H.GENERA).fit([H.GENERA])
    y_ml = []
    for genus, pure in zip(d.y, d.is_pure):
        parts = [genus] if pure else genus.split("_")
        y_ml.append(sorted({g for g in parts if g in H.GENERA}))
    Y = mlb.transform(y_ml).astype("float32")

    print(f"Accuracy (baseline from {BASELINE_DIR}, matched from {MODELS_DIR}), "
          f"{len(SEEDS)} seeds, n_jobs={args.n_jobs_accuracy}...")
    acc_raw = compute_accuracy(d, Y, grids, args.n_jobs_accuracy)
    acc_raw.to_csv(RESULTS_DIR / "accuracy_raw.csv", index=False)
    group_keys = ["variant", "technique", "value"]
    acc_summary = M.summarise(acc_raw, group_keys=group_keys, metrics=METRICS)
    acc_summary.to_csv(RESULTS_DIR / "accuracy_summary.csv", index=False)
    M.print_results(acc_summary, group_keys=group_keys, metrics=METRICS)

    print("\nDone. Results in", RESULTS_DIR)
