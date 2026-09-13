import gc

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence

import hsi_augmentations as aug
from hsi_preprocessing import snv

RF_PARAMS = dict(n_estimators=200, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, class_weight="balanced", n_jobs=-1)

# Fournier et al. replication (scripts 01 and 02)
NN_EPOCHS_FOURNIER = 100
NN_BATCH_FOURNIER  = 32

# Adapted config (scripts 03 and 04)
NN_BATCH    = 2048
NN_EPOCHS   = 2000
NN_PATIENCE = 50


def build_fournier_nn(input_dim, n_outputs, multilabel=False):
    out_act = "sigmoid" if multilabel else "softmax"
    loss    = "binary_crossentropy" if multilabel else "sparse_categorical_crossentropy"
    metric  = keras.metrics.BinaryAccuracy(name="accuracy") if multilabel else "accuracy"

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_outputs, activation=out_act),
    ])
    model.compile(optimizer="adam", loss=loss, metrics=[metric])
    return model

# Scripts 03 and 04
def default_callbacks():
    return [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=NN_PATIENCE, restore_best_weights=True),
    ]


def train_rf(X_train, y_train, seed):
    rf = RandomForestClassifier(random_state=seed, **RF_PARAMS)
    rf.fit(X_train, y_train)
    return rf


def train_nn(model, X_train, y_train, X_val, y_val, multilabel=False,
             generator=None, epochs=NN_EPOCHS, batch_size=NN_BATCH, callbacks=None):
    if callbacks is None:
        callbacks = default_callbacks()
    class_weight = None
    if not multilabel:
        w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight = dict(enumerate(w))

    if generator is not None:
        model.fit(generator, validation_data=(X_val, y_val),
                  epochs=epochs, verbose=0,
                  class_weight=class_weight, callbacks=callbacks)
    else:
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  batch_size=batch_size, epochs=epochs, verbose=0,
                  class_weight=class_weight, callbacks=callbacks)
    return model


def save_rf(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_rf(path):
    return joblib.load(path)


def save_nn(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_nn(path):
    return keras.models.load_model(path)


def clear_session():
    keras.backend.clear_session()
    gc.collect()


class AugmentedSequence(Sequence):
    """Augments each RAW batch then applies SNV. Pool is the full training set."""

    def __init__(self, X_raw, y, bio, seed):
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=10)
        self.X   = np.asarray(X_raw)
        self.y   = np.asarray(y)
        self.bio = np.asarray(bio)
        self.rng         = np.random.default_rng(seed)
        self.shuffle_rng = np.random.default_rng(seed + 777)
        self.indices     = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / NN_BATCH))

    def __getitem__(self, idx):
        batch = self.indices[idx * NN_BATCH:(idx + 1) * NN_BATCH]
        Xb = aug.augment_batch(self.X[batch], self.bio[batch], self.X, self.bio, self.rng)
        return snv(Xb), self.y[batch]

    def on_epoch_end(self):
        self.shuffle_rng.shuffle(self.indices)
