from collections import namedtuple

import numpy as np
import pandas as pd
import os
import random

# Spectral range used by Fournier et al. (2025). Trimming removes noisy bands
TRIM_MIN_NM = 420
TRIM_MAX_NM = 1000
GENERA = ["Chrysosporum", "Dolichospermum", "Microcystis"]

Dataset = namedtuple("Dataset", "X y is_pure bio_types image_ids wavelengths class_names")


def load_data(file_path, task="6class"):
    df = pd.read_excel(file_path)

    spectral_cols = df.columns[3:]
    wavelengths, keep = [], []
    for i, col in enumerate(spectral_cols):
        try:
            wavelengths.append(float(str(col).replace("nm", "").strip()))
            keep.append(i)
        except ValueError:
            continue
    wavelengths = np.array(wavelengths)
    X = df[spectral_cols].values[:, np.array(keep)]
    mask = (wavelengths >= TRIM_MIN_NM) & (wavelengths <= TRIM_MAX_NM)
    X, wavelengths = X[:, mask], wavelengths[mask]

    genus_col = df.iloc[:, 0]
    type_col  = df.iloc[:, 1]
    bio_col   = df.iloc[:, 2]
    is_pure   = (type_col.astype(str) == "pure").values

    if task == "6class":
        y = []
        for genus, pure in zip(genus_col, is_pure):
            parts = str(genus).split("_")
            y.append(str(genus) if pure or len(parts) < 2 else f"{parts[0]}_{parts[1]}")
        y = np.array(y)
    elif task == "12class":
        y = bio_col.values.astype(str)
    else:
        raise ValueError("task must be '6class' or '12class'")

    image_ids = (bio_col != bio_col.shift()).cumsum().values.astype(int)
    return Dataset(X, y, is_pure, bio_col.values, image_ids, wavelengths, np.unique(y))


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
