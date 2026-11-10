"""
Hyperparameter sweep (W&B, Bayes) of the multi-label NN: SNV, no augmentation.

Each trial trains 10 seeds x 5 folds and logs the mean exact match on val and on test. A trial is stopped early when its running val exact match falls below ABORT_RULES.

"""

import argparse
import multiprocessing
import os

import numpy as np
import tensorflow as tf
import wandb
from sklearn.preprocessing import MultiLabelBinarizer

import hsi_data as H
import hsi_metrics as M
import hsi_models as Mo
import hsi_preprocessing as P
import hsi_splits as S

PROJECT    = "cyanobacterial-hsi-sweep"
SEEDS      = range(42, 52)
TOTAL_RUNS = 150
THRESHOLD  = 0.5

PATIENCE   = 50
MIN_DELTA  = 1e-4
MAX_EPOCHS = 10_000

# (folds done, minimum running val exact match): the trial stops if below it
ABORT_RULES = ((3, 0.03), (10, 0.50))

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "exact_match_val", "goal": "maximize"},
    "parameters": {
        "n_layers":         {"distribution": "int_uniform", "min": 1, "max": 8},
        "first_layer_size": {"values": [32, 64, 128, 256, 512, 1024]},
        "dropout":          {"distribution": "uniform", "min": 0.0, "max": 0.6},
        "batch_norm":       {"values": [True, False]},
        "l2_reg":           {"distribution": "log_uniform_values", "min": 1e-7, "max": 1e-2},
        "optimizer":        {"values": ["adam", "adamw", "sgd", "rmsprop"]},
        "learning_rate":    {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "batch_size":       {"values": [16, 32, 64, 128, 256, 512, 1024]},
    },
}


def exact_match(model, X, Y):
    pred = (model.predict(X, verbose=0) >= THRESHOLD).astype(int)
    return float(M.compute_metrics(Y, pred, multilabel=True)["exact_match"])


def train_fold(cfg, d, Y, f):
    prep = P.Preprocessor("snv")
    X_tr = prep.fit_transform(d.X[f["train"]])
    X_va = prep.transform(d.X[f["val"]])
    X_te = prep.transform(d.X[f["test"]])

    model = Mo.build_sweep_nn(cfg, X_tr.shape[1], Y.shape[1])
    model.fit(X_tr, Y[f["train"]], validation_data=(X_va, Y[f["val"]]),
              epochs=MAX_EPOCHS, batch_size=int(cfg["batch_size"]), verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=PATIENCE, min_delta=MIN_DELTA,
                  restore_best_weights=True)])

    em_val  = exact_match(model, X_va, Y[f["val"]])
    em_test = exact_match(model, X_te, Y[f["test"]])
    Mo.clear_session()
    return em_val, em_test


def train_eval(d, Y, folds):
    wandb.init()
    cfg = dict(wandb.config)

    em_val, em_test = [], []
    for seed in SEEDS:
        H.set_seeds(seed)
        for f in folds[seed]:
            val, test = train_fold(cfg, d, Y, f)
            em_val.append(val)
            em_test.append(test)

            n_done  = len(em_val)
            running = float(np.mean(em_val))
            wandb.log({"exact_match_val": running, "n_folds_done": n_done})

            for n_min, threshold in ABORT_RULES:
                if n_done == n_min and running < threshold:
                    reason = f"val exact-match {running:.3f} < {threshold} after {n_done} folds"
                    print(f"  trial stopped: {reason}", flush=True)
                    wandb.log({"exact_match_val": running, "early_aborted": True,
                               "abort_reason": reason, "n_folds_done": n_done})
                    return

    wandb.log({"exact_match_val": float(np.mean(em_val)),
               "exact_match_test": float(np.mean(em_test)),
               "early_aborted": False, "n_folds_done": len(em_val)})


def run_agent(sweep_id, count, n_agents, file):
    tf.config.threading.set_intra_op_parallelism_threads(max(1, (os.cpu_count() or 1) // n_agents))
    tf.config.threading.set_inter_op_parallelism_threads(1)

    d   = H.load_data(file, "6class")
    mlb = MultiLabelBinarizer(classes=H.GENERA).fit([H.GENERA])

    y_ml = []
    for genus, pure in zip(d.y, d.is_pure):
        parts = [genus] if pure else genus.split("_")
        y_ml.append(sorted({g for g in parts if g in H.GENERA}))
    Y = mlb.transform(y_ml).astype("float32")

    folds = {seed: S.make_folds(d, "6class", seed) for seed in SEEDS}
    wandb.agent(sweep_id, function=lambda: train_eval(d, Y, folds), project=PROJECT, count=count)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="../data/cyanobacterial_spectra_processed_clean.xlsx")
    p.add_argument("--agents", type=int, default=1, help="Parallel agent processes.")
    args = p.parse_args()

    sweep_id = wandb.sweep(SWEEP_CONFIG, project=PROJECT)
    print(f"Created sweep {sweep_id}")

    counts = [TOTAL_RUNS // args.agents + (i < TOTAL_RUNS % args.agents) for i in range(args.agents)]
    if args.agents == 1:
        run_agent(sweep_id, counts[0], 1, args.file)
    else:
        ctx   = multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=run_agent, args=(sweep_id, c, args.agents, args.file))
                 for c in counts]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
