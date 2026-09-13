import numpy as np

TECHNIQUES = ["AddN", "MultN", "Shift", "Mask", "Mixup_interclass", "Poisson"]

MAX = {"AddN": 0.01, "MultN": 0.012, "Shift": 5, "Mask": 75, "Mixup_interclass": 0.5, "Poisson": 1000.0}


# --- primitives ---

def noise_additive(x, rng, sigma):
    return x + rng.normal(0.0, sigma, size=x.shape)


def noise_multiplicative(x, rng, sigma):
    return x * rng.normal(1.0, sigma, size=x.shape)


def spectral_shift(x, rng, max_shift):
    n = int(rng.integers(-max_shift, max_shift + 1))
    if n == 0:
        return x.copy()
    out = np.empty_like(x)
    if n > 0:
        out[n:] = x[:-n]; out[:n] = x[0]
    else:
        out[:n] = x[-n:]; out[n:] = x[-1]
    return out


def band_masking(x, rng, n_bands):
    out = x.copy()
    for i in rng.choice(len(x), size=min(n_bands, len(x)), replace=False):
        if 0 < i < len(x) - 1:
            out[i] = 0.5 * (out[i - 1] + out[i + 1])
        elif i == 0:
            out[i] = out[1]
        else:
            out[i] = out[i - 1]
    return out


def mixup_interclass(x, rng, lam, X_pool, bio_pool, bio_type):
    """No exclude_idx: if the same sample is drawn, lam*x + (1-lam)*x = x (no-op)."""
    cand = np.where(bio_pool == bio_type)[0]
    if len(cand) == 0:
        return x.copy()
    return lam * x + (1.0 - lam) * X_pool[rng.choice(cand)]


def noise_poisson(x, rng, scale):
    return rng.poisson(np.clip(x, 0, None) * scale) / scale


# --- composed ---

def corrupt_one(x, rng, technique, severity, X_pool, bio_pool, bio_type):
    """Apply one technique at severity in [0, 1]. Poisson always runs at fixed scale."""
    if technique == "AddN":
        return noise_additive(x, rng, severity * MAX["AddN"])
    if technique == "MultN":
        return noise_multiplicative(x, rng, severity * MAX["MultN"])
    if technique == "Shift":
        n = int(round(severity * MAX["Shift"]))
        return spectral_shift(x, rng, n) if n > 0 else x.copy()
    if technique == "Mask":
        n = int(round(severity * MAX["Mask"]))
        return band_masking(x, rng, n) if n > 0 else x.copy()
    if technique == "Mixup_interclass":
        return mixup_interclass(x, rng, 1.0 - severity * MAX["Mixup_interclass"], X_pool, bio_pool, bio_type)
    if technique == "Poisson":
        return noise_poisson(x, rng, MAX["Poisson"])
    raise ValueError(f"unknown technique: {technique}")


def augment_batch(X, bio, X_pool, bio_pool, rng):
    """Pick one technique uniformly (including None) at a random severity."""
    options = TECHNIQUES + ["None"]
    out = X.copy()
    for i in range(len(X)):
        technique = rng.choice(options)
        if technique != "None":
            out[i] = corrupt_one(X[i], rng, technique, rng.uniform(0.0, 1.0),
                                 X_pool, bio_pool, bio[i])
    return out
