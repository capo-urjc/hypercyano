import joblib

import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler


def snv(X):
    return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)


class Preprocessor:
    """
    fournier = StandardScaler[train] + Savitzky-Golay.
    snv = Standard Normal Variate.
    """

    def __init__(self, config):
        self.config = config
        self._scaler = None

    def fit(self, X_train):
        if self.config == "fournier":
            self._scaler = StandardScaler().fit(X_train)
        return self

    def transform(self, X):
        if self.config == "snv":
            return snv(X)
        return savgol_filter(self._scaler.transform(X), 11, 2, axis=1)

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
