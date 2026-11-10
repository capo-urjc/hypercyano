"""
00_aumentado_2x3_puros.py

Grafica 2x3 por cada genero PURO (Chrysosporum_pure, Dolichospermum_pure,
Microcystis_pure): firma original (reflectancia, SIN SNV) vs. cada una de
las 6 tecnicas de aumentado ACTUALES de hsi_augmentations.py -- las mismas
que usa el pipeline de entrenamiento y el 06_robustness_distance_analysis.py,
incluido el Mask arreglado (band_masking con interpolacion simultanea) y el
Poisson arreglado (escala de verdad con severity). No se reimplementa nada
a mano: se llama directamente a aug.corrupt_one, asi que si vuelves a tocar
hsi_augmentations.py esta figura se actualiza sola.

Para Mixup_interclass se pinta tambien la pareja (Partner), en reflectancia
SIN SNV igual que el resto -- no solo el resultado mezclado.

Salida: ../graficas_tfm/aug_2x3_original_vs_aug_{clase}.pdf
  (p.ej. aug_2x3_original_vs_aug_Chrysosporum_pure.pdf)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import hsi_augmentations as aug

# ==============================================================================
# CONFIGURACION
# ==============================================================================
FILE_PATH  = "../data/cyanobacterial_spectra_processed_clean.xlsx"
OUTPUT_DIR = "../graficas_tfm"
WL_MIN, WL_MAX = 420, 1000
SEED = 42
BASE_SEVERITY = 0.6   # severidad general por defecto (en [0,1])

# Severidades específicas por técnica (si no está aquí, usa BASE_SEVERITY)
TECHNIQUE_SEVERITIES = {
    "Mask": 1.2,  # <--- Severidad de Band Masking aún más alta
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT_COLOR = "#000000"
plt.rcParams.update({
    "font.family":                 "serif",
    "font.serif":                  ["cmr10", "CMU Serif",
                                    "Computer Modern Roman", "DejaVu Serif"],
    "text.color":                  TEXT_COLOR,
    "axes.labelcolor":             TEXT_COLOR,
    "axes.titlecolor":             TEXT_COLOR,
    "xtick.color":                 TEXT_COLOR,
    "ytick.color":                 TEXT_COLOR,
    "axes.edgecolor":              "#666666",
    "axes.linewidth":              1.0,
    "mathtext.fontset":            "cm",
    "axes.unicode_minus":          False,
    "axes.formatter.use_mathtext": True,
    "font.size":        15,
    "axes.titlesize":   17,
    "axes.titleweight": "bold",
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linewidth":   0.8,
})

COLOR_ORIG    = "#222222"
COLOR_AUG     = "#1f77b4"
COLOR_PARTNER = "#ff7f0e"

# Nombres "bonitos" (sin guion bajo) para el texto de la figura -- mismos
# nombres que en 00_aumentado_snv_3x3.py, consistentes en todo el paper.
TECHNIQUE_DISPLAY = {
    "AddN":             "Additive Noise",
    "MultN":            "Multiplicative Noise",
    "Shift":            "Spectral Shift",
    "Mask":             "Band Masking",
    "Mixup_interclass": "Spectral Mixup (Intra-Class)",
    "Poisson":          "Poisson Noise",
}
TECHNIQUES = aug.TECHNIQUES   # las 6 actuales del modulo, en su orden


# ==============================================================================
# HELPERS
# ==============================================================================
def load_data(file_path):
    df = pd.read_excel(file_path)
    spectral_cols = df.columns[3:]
    wavelengths, valid_idx = [], []
    for i, col in enumerate(spectral_cols):
        try:
            wl = float(str(col).replace("nm", "").strip())
            wavelengths.append(wl)
            valid_idx.append(i)
        except ValueError:
            continue
    X_full = df[spectral_cols].values[:, valid_idx].astype(float)
    wl = np.array(wavelengths)
    mask = (wl >= WL_MIN) & (wl <= WL_MAX)
    col1   = df.iloc[:, 1].values   # "pure" / tipo
    labels = df.iloc[:, 2].values   # bio_type completo (12class)
    return X_full[:, mask], wl[mask], col1, labels


def save_pdf(fig, path):
    with PdfPages(path) as pdf:
        pdf.savefig(fig, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  Guardado: {path}")


# ==============================================================================
# PLOT 2x3
# ==============================================================================
def plot_2x3_for_class(x_base, bio_type, wavelengths, X, bio_types, rng, abbr_name, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.patch.set_facecolor("white")

    for i, technique in enumerate(TECHNIQUES):
        ax = axes[i]

        # Obtenemos la severidad correspondiente a la técnica actual
        severity = TECHNIQUE_SEVERITIES.get(technique, BASE_SEVERITY)

        if technique == "Mixup_interclass":
            # Igual que hace aug.mixup_interclass, pero calculado aqui para
            # poder pintar tambien la pareja elegida (corrupt_one no la
            # expone, solo devuelve el resultado mezclado).
            lam = 1.0 - severity * aug.MAX["Mixup_interclass"]
            cand = np.where(bio_types == bio_type)[0]
            partner = X[rng.choice(cand)] if len(cand) else x_base
            mixed = lam * x_base + (1.0 - lam) * partner

            ax.plot(wavelengths, x_base, color=COLOR_ORIG, linestyle="--", linewidth=1.8,
                    label=f"Original ({abbr_name})", zorder=4)
            ax.plot(wavelengths, partner, color=COLOR_PARTNER, linestyle=":", linewidth=1.8,
                    label="Partner", zorder=5)
            ax.plot(wavelengths, mixed, color=COLOR_AUG, linewidth=2.0,
                    label="Mixup", zorder=6)
            ax.legend(frameon=True, loc="upper left")
        else:
            augmented = aug.corrupt_one(x_base.copy(), rng, technique, severity,
                                        X, bio_types, bio_type)
            ax.plot(wavelengths, x_base, color=COLOR_ORIG, linestyle="--", linewidth=2.0,
                    label=f"Original ({abbr_name})" if i == 0 else "", zorder=5)
            ax.plot(wavelengths, augmented, color=COLOR_AUG, linewidth=1.8,
                    label="Augmented" if i == 0 else "", alpha=0.9, zorder=6)
            if i == 0:
                ax.legend(frameon=True, loc="upper left")

        ax.set_title(TECHNIQUE_DISPLAY[technique], pad=8)
        ax.set_xlabel("Wavelength (nm)", labelpad=6)
        ax.set_ylabel("Reflectance", labelpad=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#666666")

    plt.tight_layout()
    save_pdf(fig, output_path)
    plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    rng = np.random.default_rng(SEED)
    print("Cargando datos ...")
    X, wavelengths, col1, labels = load_data(FILE_PATH)

    pure_idx = np.where(col1 == "pure")[0]
    if len(pure_idx) == 0:
        pure_idx = np.arange(len(X))
    pure_classes = np.unique(labels[pure_idx])
    print(f"Generos puros encontrados: {list(pure_classes)}")

    for cls in pure_classes:
        cls_samples = np.where((col1 == "pure") & (labels == cls))[0]
        if len(cls_samples) == 0:
            continue

        # Muestra base = mediana por intensidad, unica y fija para todo el genero
        intensities = np.mean(X[cls_samples], axis=1)
        sample_idx = cls_samples[np.argsort(intensities)[len(intensities) // 2]]
        x_base = X[sample_idx]
        bio_type = labels[sample_idx]

        abbr_name = str(cls)
        for key, val in {"Chrysosporum": "Chryso.", "Dolichospermum": "Dolicho.",
                         "Microcystis": "Micro."}.items():
            if key in str(cls):
                abbr_name = val
                break

        safe = str(cls).replace(" ", "_").replace("/", "_")
        output_path = os.path.join(OUTPUT_DIR, f"aug_2x3_original_vs_aug_{safe}.pdf")
        plot_2x3_for_class(x_base, bio_type, wavelengths, X, labels, rng, abbr_name, output_path)
        print(f"Generada grafica para puro '{cls}' -> Guardado en: {output_path}")

    print("\n¡Listo!")


if __name__ == "__main__":
    main()