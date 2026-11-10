"""Generates confusion matrices from the 03_multilabel models.

Two PDFs per (config, model) combination, saved to ../results/03/multilabel/:
  confusion_{config}_{model}.pdf        -- genus combination vs genus combination
  condition_{config}_{model}_{pair}.pdf -- preparation condition vs genus combination

No title on the figures (caption goes in the paper), except the condition
plots, which use the mixture name as their title.
"""

from pathlib import Path

import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MultiLabelBinarizer

import hsi_data as H
import hsi_models as Mo
import hsi_preprocessing as P
import hsi_splits as S

MODELS_DIR  = Path("../models/03")
RESULTS_DIR = Path("../results/03/multilabel")
CONFIGS     = ["snv"]
SEEDS       = range(42, 52)
THRESHOLD   = 0.5

CONDITIONS = ["cell-mL", "biovol", "chla"]

PAIRS = {
    "Chrysosporum+Dolichospermum": ("C-D", "Chryso.+Dolicho."),
    "Chrysosporum+Microcystis":    ("C-M", "Chryso.+Micro."),
    "Dolichospermum+Microcystis":  ("D-M", "Dolicho.+Micro."),
}

# Proportions from Fournier et al. (Table adapted).
# Structure: pair_key -> equalized_condition -> other_variable -> {genus: pct}
# For each pair, each preparation batch equalizes ONE variable to 50-50
# (cell-mL, biovol or chla) and the OTHER TWO variables end up split
# unevenly between the two genera. Each entry below gives that split for
# one of the two non-equalized variables. Values transcribed from the
# "Equalized" / "Variable" columns of the source table.
VAR_LABELS = {"cell-mL": "Cell-mL", "biovol": "Biovol", "chla": "Chla"}
VAR_ORDER  = ["cell-mL", "biovol", "chla"]  # fixed display order within a row label

PROPORTIONS = {
    "Chrysosporum+Microcystis": {
        "cell-mL": {
            "biovol": {"Microcystis": 40, "Chrysosporum": 60},
            "chla":   {"Microcystis": 22, "Chrysosporum": 78},
        },
        "biovol": {
            "cell-mL": {"Microcystis": 60, "Chrysosporum": 40},
            "chla":    {"Microcystis": 30, "Chrysosporum": 70},
        },
        "chla": {
            "cell-mL": {"Microcystis": 72, "Chrysosporum": 28},
            "biovol":  {"Microcystis": 63, "Chrysosporum": 37},
        },
    },
    "Chrysosporum+Dolichospermum": {
        "cell-mL": {
            "biovol": {"Dolichospermum": 84, "Chrysosporum": 16},
            "chla":   {"Dolichospermum": 82, "Chrysosporum": 18},
        },
        "biovol": {
            "cell-mL": {"Dolichospermum": 16, "Chrysosporum": 84},
            "chla":    {"Dolichospermum": 46, "Chrysosporum": 54},
        },
        "chla": {
            "cell-mL": {"Dolichospermum": 25, "Chrysosporum": 75},
            "biovol":  {"Dolichospermum": 64, "Chrysosporum": 36},
        },
    },
    "Dolichospermum+Microcystis": {
        "cell-mL": {
            "biovol": {"Microcystis": 11, "Dolichospermum": 89},
            "chla":   {"Microcystis": 6,  "Dolichospermum": 94},
        },
        "biovol": {
            "cell-mL": {"Microcystis": 89, "Dolichospermum": 11},
            "chla":    {"Microcystis": 33, "Dolichospermum": 67},
        },
        "chla": {
            "cell-mL": {"Microcystis": 88, "Dolichospermum": 12},
            "biovol":  {"Microcystis": 49, "Dolichospermum": 51},
        },
    },
}

# Fixed label order: "All three" first, then None, then singles, then pairs.
# "None" ("") is dropped dynamically at plot time if it never occurs.
KEY_ORDER = [
    "Chrysosporum+Dolichospermum+Microcystis",  # All three — always first
    "",                                           # None — dropped if unused
    "Chrysosporum",
    "Dolichospermum",
    "Microcystis",
    "Chrysosporum+Dolichospermum",
    "Chrysosporum+Microcystis",
    "Dolichospermum+Microcystis",
]

GENERA_LABELS = {
    "":                                        "None",
    "Chrysosporum":                            "Chryso.",
    "Dolichospermum":                          "Dolicho.",
    "Microcystis":                             "Micro.",
    "Chrysosporum+Dolichospermum":             "Chryso.+Dolicho.",
    "Chrysosporum+Microcystis":                "Chryso.+Micro.",
    "Dolichospermum+Microcystis":              "Dolicho.+Micro.",
    "Chrysosporum+Dolichospermum+Microcystis": "Chryso.+Dolicho.+Micro.",
}

# ── Typography ────────────────────────────────────────────────────────────────
_INSTALLED = {f.name for f in _fm.fontManager.ttflist}
FONT_STACK = [f for f in ("CMU Serif", "cmr10", "DejaVu Serif") if f in _INSTALLED] or ["serif"]
TEXT_COLOR = "#000000"

plt.rcParams.update({
    "font.family":        FONT_STACK,
    "mathtext.fontset":   "cm",
    "axes.unicode_minus": False,
    "text.color":         TEXT_COLOR,
    "axes.labelcolor":    TEXT_COLOR,
    "xtick.color":        TEXT_COLOR,
    "ytick.color":        TEXT_COLOR,
    "axes.edgecolor":     "#666666",
    "axes.linewidth":     1.0,
    "font.size":          18,
    "axes.labelsize":     21,
    "xtick.labelsize":    15,
    "ytick.labelsize":    15,
    "legend.fontsize":    15,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})
# ─────────────────────────────────────────────────────────────────────────────


def combo_key(row, classes):
    names = sorted(classes[row.astype(bool)])
    return "+".join(names) if names else ""


def condition_of(bio_type):
    for cond in CONDITIONS:
        if str(bio_type).endswith(cond):
            return cond
    return "unknown"


def collect_predictions(d, d12, Y, mlb, config):
    records = []
    for seed in SEEDS:
        folds = S.make_folds(d, "6class", seed)
        for k, f in enumerate(folds):
            path = MODELS_DIR / "multilabel" / f"seed{seed}" / config / f"fold{k}"
            prep = P.Preprocessor.load(path / "preprocessor.joblib")
            X_te = prep.transform(d.X[f["test"]])
            Y_te = Y[f["test"]]

            rf      = Mo.load_rf(path / "rf.joblib")
            rf_pred = rf.predict(X_te)

            nn      = Mo.load_nn(path / "nn.keras")
            nn_pred = (nn.predict(X_te, verbose=0) >= THRESHOLD).astype(int)
            Mo.clear_session()

            for i, idx in enumerate(f["test"]):
                records.append({
                    "true":     Y_te[i],
                    "rf":       rf_pred[i],
                    "nn":       nn_pred[i],
                    "bio_type": d12.bio_types[idx],
                    "is_pure":  d.is_pure[idx],
                })
    return records


def build_genus_cm(records, mlb, model_key):
    """Full label set fixed to KEY_ORDER; "None" is only included if it
    actually occurs among the true or predicted labels. Always includes
    C.+D.+M."""
    classes   = np.array(mlb.classes_)
    true_keys = [combo_key(r["true"], classes) for r in records]
    pred_keys = [combo_key(r[model_key], classes) for r in records]

    seen     = set(true_keys) | set(pred_keys)
    has_none = "" in seen

    # Drop "" from the fixed order if it never occurs
    base_order = [k for k in KEY_ORDER if k != "" or has_none]
    extra      = [k for k in seen if k not in base_order]
    all_keys   = base_order + sorted(extra)
    idx        = {k: i for i, k in enumerate(all_keys)}

    cm = np.zeros((len(all_keys), len(all_keys)), dtype=int)
    for t, p in zip(true_keys, pred_keys):
        cm[idx[t], idx[p]] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct   = np.where(row_sums > 0, 100.0 * cm / row_sums, 0.0)
    labels   = [GENERA_LABELS.get(k, k) for k in all_keys]
    return cm_pct, labels


def build_condition_cm(records, mlb, model_key, pair_key):
    """condition (rows) x predicted combination (columns), row-normalised %.
    Row order follows CONDITIONS (cell-mL, biovol, chla).
    Column order follows KEY_ORDER ("None" dropped if unused for this pair).

    Each row gets a 3-line label showing all three variables in the FIXED
    order Cell-mL / Biovolumen / Chla, e.g. for the "cell-mL" row:
        Cell-mL (50-50)      <- bold, this is the equalized variable
        Biovolumen (40-60)
        Chla (22-78)
    The two non-equalized lines come from PROPORTIONS[pair_key][cond][var];
    the equalized line is always "50-50". Numbers follow the genus order
    used in the pair's title. Returns row_labels as a list of
    (lines, bold_line_index) tuples instead of plain strings.
    """
    classes   = np.array(mlb.classes_)
    pair_recs = [r for r in records
                 if combo_key(r["true"], classes) == pair_key and not r["is_pure"]]
    if not pair_recs:
        return None, None, None

    true_keys = [combo_key(r["true"], classes) for r in pair_recs]
    pred_keys = [combo_key(r[model_key], classes) for r in pair_recs]
    has_none  = "" in (set(true_keys) | set(pred_keys))

    col_order = [k for k in KEY_ORDER if k != "" or has_none]
    col_idx   = {k: i for i, k in enumerate(col_order)}

    # Genus order for the numeric labels = same order as the pair's title
    genus_order = pair_key.split("+")
    props       = PROPORTIONS.get(pair_key, {})
    row_labels  = []  # list of (lines, bold_idx)
    for cond in CONDITIONS:
        cond_data = props.get(cond, {})
        lines = []
        for var in VAR_ORDER:
            if var == cond:
                pct_str = "50-50"
            else:
                pcts    = cond_data.get(var, {})
                pct_str = "-".join(f"{pcts.get(g, 0)}" for g in genus_order)
            lines.append(f"{VAR_LABELS[var]} ({pct_str})")
        row_labels.append((lines, VAR_ORDER.index(cond)))

    cm = np.zeros((len(CONDITIONS), len(col_order)), dtype=int)
    for r in pair_recs:
        cond = condition_of(str(r["bio_type"]))
        pred = combo_key(r[model_key], classes)
        if cond in CONDITIONS and pred in col_idx:
            cm[CONDITIONS.index(cond), col_idx[pred]] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct   = np.where(row_sums > 0, 100.0 * cm / row_sums, 0.0)
    col_labels = [GENERA_LABELS.get(k, k) for k in col_order]
    return cm_pct, row_labels, col_labels


# ── Plotting ──────────────────────────────────────────────────────────────────

def _log_heatmap(ax, cm_pct, row_labels, col_labels, ylabel, xlabel):
    """Heatmap with log-scale colour mapping and no colorbar.

    row_labels is normally a flat list of strings (one line per row), used
    for the standard seaborn yticklabels. If it is instead a list of
    (lines, bold_idx) tuples -- one per row, `lines` a list of strings and
    `bold_idx` the index of the line to render in bold -- the y-axis labels
    are drawn manually as multiline text, with that one line bold, instead
    of using yticklabels.
    """
    from matplotlib.colors import LogNorm

    multiline = bool(row_labels) and isinstance(row_labels[0], tuple)

    # Replace 0 with NaN so empty cells render as white (Blues cmap treats NaN as masked)
    data = np.where(cm_pct > 0, cm_pct, np.nan)

    sns.heatmap(
        data, annot=cm_pct, fmt=".1f", cmap="Blues",
        xticklabels=col_labels, yticklabels=(False if multiline else row_labels), ax=ax,
        annot_kws={"size": 15, "weight": "bold"},
        norm=LogNorm(vmin=0.1, vmax=100),
        cbar=False,
        linewidths=0.5, linecolor="#DDDDDD",
        mask=np.isnan(data),
    )

    # Cells that are truly 0: annotate manually with "0" on white background
    n_rows, n_cols = cm_pct.shape
    for r in range(n_rows):
        for c in range(n_cols):
            if cm_pct[r, c] == 0:
                ax.text(c + 0.5, r + 0.5, "0.0", ha="center", va="center",
                        fontsize=15, fontweight="bold", color=TEXT_COLOR)

    if multiline:
        ax.set_yticks([])
        trans = ax.get_yaxis_transform()  # x: axes fraction, y: data coords
        for r, (lines, bold_idx) in enumerate(row_labels):
            n_lines      = len(lines)
            line_spacing = 0.30
            start_offset = -line_spacing * (n_lines - 1) / 2
            for li, line in enumerate(lines):
                y = r + 0.5 + start_offset + li * line_spacing
                is_bold = (li == bold_idx)
                # cmr10 (the main body font) has no real bold face, so
                # matplotlib can't fake it: use DejaVu Serif bold just for
                # the emphasized (equalized) line instead.
                kwargs = (dict(family="DejaVu Serif", fontweight="bold")
                          if is_bold else dict(fontweight="normal"))
                ax.text(-0.02, y, line, ha="right", va="center",
                         transform=trans, fontsize=12,
                         color=TEXT_COLOR, clip_on=False, **kwargs)

    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right", fontsize=15)
    if not multiline:
        plt.yticks(rotation=0, fontsize=15)
    ax.tick_params(length=3, colors=TEXT_COLOR)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#666666")
        ax.spines[sp].set_linewidth(0.8)


def save_pdf(fig, path):
    with PdfPages(path) as pdf:
        pdf.savefig(fig, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  saved: {path}")


def plot_genus_cm(cm_pct, labels, output_path):
    n   = len(labels)
    fig = plt.figure(figsize=(max(10, n * 1.3), max(8, n * 1.1)))
    ax  = fig.add_subplot(111)
    fig.patch.set_facecolor("white")
    _log_heatmap(ax, cm_pct, labels, labels,
                 "True combination", "Predicted combination")
    plt.tight_layout()
    save_pdf(fig, output_path)


def plot_condition_cm(cm_pct, row_labels, col_labels, pair_label, output_path):
    n_cols = len(col_labels)
    fig    = plt.figure(figsize=(max(10, n_cols * 1.3), 5.5))
    ax     = fig.add_subplot(111)
    fig.patch.set_facecolor("white")
    _log_heatmap(ax, cm_pct, row_labels, col_labels,
                 "Condition", "Predicted combination")
    ax.set_title(pair_label, fontsize=21, pad=14)
    plt.tight_layout()
    save_pdf(fig, output_path)


def plot_master_condition_table(records, mlb, model_key, output_path):
    """Genera una única matriz combinada limpia, sin barra de color y con etiquetas ordenadas."""
    classes = np.array(mlb.classes_)

    all_col_keys = [k for k in KEY_ORDER if k != ""]
    col_labels = [GENERA_LABELS.get(k, k) for k in all_col_keys]
    col_idx = {k: i for i, k in enumerate(all_col_keys)}

    matrix_rows = []
    row_labels_info = []
    pair_boundary_indices = []

    for pair_key, (pair_short, pair_label) in PAIRS.items():
        pair_recs = [r for r in records if combo_key(r["true"], classes) == pair_key and not r["is_pure"]]
        if not pair_recs:
            continue

        genus_order = pair_key.split("+")
        props = PROPORTIONS.get(pair_key, {})

        for cond in CONDITIONS:
            cond_data = props.get(cond, {})
            lines = []
            for var in VAR_ORDER:
                if var == cond:
                    pct_str = "50-50"
                else:
                    pcts = cond_data.get(var, {})
                    pct_str = "-".join(f"{pcts.get(g, 0)}" for g in genus_order)
                lines.append(f"{VAR_LABELS[var]} ({pct_str})")

            bold_idx = VAR_ORDER.index(cond)
            row_labels_info.append((pair_label, lines, bold_idx))

            cond_recs = [r for r in pair_recs if condition_of(str(r["bio_type"])) == cond]
            row_counts = np.zeros(len(all_col_keys), dtype=int)
            for r in cond_recs:
                pred = combo_key(r[model_key], classes)
                if pred in col_idx:
                    row_counts[col_idx[pred]] += 1

            total = len(cond_recs)
            row_pct = (100.0 * row_counts / total) if total > 0 else np.zeros_like(row_counts, dtype=float)
            matrix_rows.append(row_pct)

        pair_boundary_indices.append(len(matrix_rows))

    matrix_data = np.array(matrix_rows)

    # Creamos la figura dejando espacio a la izquierda
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")

    data_masked = np.where(matrix_data > 0, matrix_data, np.nan)

    from matplotlib.colors import LogNorm
    sns.heatmap(
        data_masked, annot=matrix_data, fmt=".1f", cmap="Blues",
        xticklabels=col_labels, yticklabels=False, ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
        norm=LogNorm(vmin=0.1, vmax=100),
        cbar=False,  # <--- Barra de color eliminada
        linewidths=0.5, linecolor="#CCCCCC",
        mask=np.isnan(data_masked)
    )

    # Anotar ceros explícitamente
    n_rows, n_cols = matrix_data.shape
    for r in range(n_rows):
        for c in range(n_cols):
            if matrix_data[r, c] == 0:
                ax.text(c + 0.5, r + 0.5, "0.0", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=TEXT_COLOR)

    # Líneas divisorias horizontales entre los 3 pares
    for b_idx in pair_boundary_indices[:-1]:
        ax.axhline(b_idx, color="black", linewidth=2)

    ax.set_yticks([])
    trans = ax.get_yaxis_transform()

    # Dibujar las etiquetas organizadas a la izquierda sin solaparse
    for r, (pair_lbl, lines, bold_idx) in enumerate(row_labels_info):
        # Título del par de géneros centrado en su bloque respectivo
        if r in [0, 3, 6]:
            ax.text(-0.35, r + 1.5, pair_lbl, ha="center", va="center",
                    transform=trans, fontsize=12, fontweight="bold", color=TEXT_COLOR)

        # Líneas de condiciones (con la variable equiparada en negrita)
        n_lines = len(lines)
        line_spacing = 0.28
        start_offset = -line_spacing * (n_lines - 1) / 2
        for li, line in enumerate(lines):
            y = r + 0.5 + start_offset + li * line_spacing
            is_bold = (li == bold_idx)
            kwargs = (dict(family="DejaVu Serif", fontweight="bold")
                      if is_bold else dict(fontweight="normal"))
            ax.text(-0.03, y, line, ha="right", va="center",
                    transform=trans, fontsize=10,
                    color=TEXT_COLOR, clip_on=False, **kwargs)

    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right", fontsize=13)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Margen izquierdo amplio para acomodar todo el texto sin apretujar
    plt.subplots_adjust(left=0.38, right=0.98, top=0.95, bottom=0.18)
    save_pdf(fig, output_path)

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    d   = H.load_data("../data/cyanobacterial_spectra_processed_clean.xlsx", "6class")
    d12 = H.load_data("../data/cyanobacterial_spectra_processed_clean.xlsx", "12class")
    mlb = MultiLabelBinarizer(classes=H.GENERA).fit([H.GENERA])

    y_ml = []
    for genus, pure in zip(d.y, d.is_pure):
        parts = [genus] if pure else genus.split("_")
        y_ml.append(sorted({g for g in parts if g in H.GENERA}))
    Y = mlb.transform(y_ml).astype("float32")

    for config in CONFIGS:
        records = collect_predictions(d, d12, Y, mlb, config)

        for model_name in ("RF", "NN"):
            model_key = model_name.lower()

            # 1. Matriz de confusión general de géneros
            cm_pct, labels = build_genus_cm(records, mlb, model_key)
            plot_genus_cm(cm_pct, labels,
                          RESULTS_DIR / f"confusion_{config}_{model_name}.pdf")

            # 2. (Opcional) Si sigues queriendo los individuales por si acaso
            for pair_key, (pair_short, pair_label) in PAIRS.items():
                cm_pct, row_labels, col_labels = build_condition_cm(
                    records, mlb, model_key, pair_key)
                if cm_pct is None:
                    continue
                plot_condition_cm(cm_pct, row_labels, col_labels, pair_label,
                                  RESULTS_DIR / f"condition_{config}_{model_name}_{pair_short}.pdf")

            # 3. AQUÍ FUERA del bucle de pares: Genera la 3-en-1 combinada una sola vez por modelo
            plot_master_condition_table(records, mlb, model_key,
                                         RESULTS_DIR / f"conditions_combined_{config}_{model_name}.pdf")