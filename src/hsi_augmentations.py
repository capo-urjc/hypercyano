import numpy as np

TECHNIQUES = ["AddN", "MultN", "Shift", "Mask", "Mixup_interclass", "Poisson"]


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
        out[n:] = x[:-n]
        out[:n] = x[0]
    else:
        out[:n] = x[-n:]
        out[n:] = x[-1]
    return out


def band_masking(x, rng, n_bands):
    n = len(x)
    n_bands = min(n_bands, n)
    if n_bands <= 0:
        return x.copy()

    masked_idx = rng.choice(n, size=n_bands, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[masked_idx] = True
    valid_idx = np.where(~mask)[0]

    if len(valid_idx) == 0:
        return np.full_like(x, np.mean(x))

    out = x.copy().astype(float, copy=True)
    positions = np.arange(n)
    out[mask] = np.interp(positions[mask], positions[valid_idx], x[valid_idx])
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

def apply(x, rng, technique, value, X_pool, bio_pool, bio_type):
    """Apply `technique` at `value`: sigma, number of bands, lambda or Poisson scale."""
    if technique == "AddN":
        return noise_additive(x, rng, value)
    if technique == "MultN":
        return noise_multiplicative(x, rng, value)
    if technique == "Shift":
        n = int(round(value))
        return spectral_shift(x, rng, n) if n > 0 else x.copy()
    if technique == "Mask":
        n = int(round(value))
        return band_masking(x, rng, n) if n > 0 else x.copy()
    if technique == "Mixup_interclass":
        return mixup_interclass(x, rng, value, X_pool, bio_pool, bio_type)
    if technique == "Poisson":
        return noise_poisson(x, rng, value)
    raise ValueError(f"unknown technique: {technique}")


# --- sweep grid ---

GRID_SIZE = {
    "AddN": 16, "MultN": 16, "Shift": 16, "Mask": 16,
    "Mixup_interclass": 6, "Poisson": 16,
}


def grid_for(technique, n_bands):
    if technique == "AddN":
        return np.linspace(0.0, 0.03, GRID_SIZE["AddN"])
    if technique == "MultN":
        return np.linspace(0.0, 0.036, GRID_SIZE["MultN"])
    if technique == "Shift":
        return np.arange(GRID_SIZE["Shift"])
    if technique == "Mask":
        return np.round(np.linspace(0, n_bands, GRID_SIZE["Mask"])).astype(int)
    if technique == "Mixup_interclass":
        return np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    if technique == "Poisson":
        return np.geomspace(10.0, 5000.0, GRID_SIZE["Poisson"])
    raise ValueError(f"unknown technique: {technique}")


def value_str(technique, value):
    """String of a grid value, used in the model folder names."""
    if technique in ("Shift", "Mask"):
        return str(int(round(value)))
    if technique == "Mixup_interclass":
        return f"{value:.2f}"
    if technique == "Poisson":
        return f"{value:.3f}"
    return f"{value:.6f}"
