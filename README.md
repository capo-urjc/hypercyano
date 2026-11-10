# HyperCyano

**Beyond spectral-wise evaluation: Generalization-aware multi-label learning for
hyperspectral cyanobacteria identification**

**HyperCyano** studies how the evaluation protocol, the preprocessing, the
label formulation and data augmentation affect the generalization of
hyperspectral classifiers for cyanobacterial genera to unseen acquisitions.

It builds on the Random Forest and feed-forward network of Fournier et al.,
whose architectures are kept unchanged in experiments 1–4, so differences in
the reported metrics come from the experimental setup and not from the model.

## Key Features

- **Image-wise evaluation.** Replicates the spectral-wise protocol of
  Fournier et al. and complements it with an image-wise one, in which all
  spectra of an acquisition fall in the same partition, to measure
  generalization to unseen acquisitions.
- **Preprocessing.** Compares Savitzky-Golay + StandardScaler (as in
  Fournier et al.) against Standard Normal Variate (SNV).
- **Multi-label identification.** Predicts which genera are present in a
  sample instead of a single composite class, so each species is represented
  independently and the formulation scales to mixtures.
- **Robustness to spectral perturbations.** Under progressive synthetic
  perturbations (additive and multiplicative noise, shift, band masking, mixup
  and Poisson shot noise), compares a never-augmented baseline (SNV only)
  against networks trained with each technique, revealing where the models are
  robust and where they are sensitive.
- **Hyperparameter search.** Bayesian sweep of the multi-label network without
  augmentation.

## Installation

Tested with Python 3.13.

```bash
git clone https://github.com/capo-urjc/hypercyano.git
cd hypercyano
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The spectra workbook is the dataset published with the paper. Place it at
`data/cyanobacterial_spectra_processed_clean.xlsx` (or pass `--file` with a
different path to any script).

## Usage

Experiments 1–4 are `train`/`infer` pairs: `train` fits the preprocessing and
the models and saves them; `infer` loads them, evaluates them and writes
tables. All scripts are run from `src/`.

```bash
cd src

# 1. Fournier et al. replication: spectral-wise split vs image-wise split.
python 01_fournier_replication_train.py
python 01_fournier_replication_infer.py

# 2. Preprocessing comparison with the image-wise split:
#    Savitzky-Golay + StandardScaler vs SNV.
python 02_preprocessing_comparison_train.py
python 02_preprocessing_comparison_infer.py

# 3. Multi-label genus presence instead of single composite classes.
python 03_multilabel_train.py
python 03_multilabel_infer.py

# 4. On that same multi-label task: Baseline (SNV) vs augmentation-trained
#    networks under synthetic sensor corruption, technique by technique.
#    Uses the baseline models of step 3.
python 04_matched_train.py
python 04_matched_infer.py

# 5. Hyperparameter sweep of the multi-label network (needs `wandb login`).
python 05_sweep_final.py
```

Models and fitted preprocessors go to `models/<experiment>/` and results to
`results/<experiment>/`. Run `python <script>.py --help` to see the options of
each script.

### Repository layout

The `hsi_*` modules hold everything the scripts share, so the splitter,
preprocessing, models and metrics are identical across experiments.

```
hypercyano/
├── data/
│   ├── cyanobacterial_spectra_processed_clean.xlsx
├── models/                        # saved models + preprocessors (not versioned)
├── results/                       # tables (not versioned)
├── src/
│   ├── hsi_data.py                # loading and labels
│   ├── hsi_splits.py              # image-wise and spectral-wise splits
│   ├── hsi_preprocessing.py       # fournier / snv pipelines, fitted and persisted
│   ├── hsi_models.py              # Random Forest and network: train, predict, save, load
│   ├── hsi_augmentations.py       # spectral augmentation primitives
│   ├── hsi_metrics.py             # metrics and summaries
│   ├── 01_fournier_replication_train.py / _infer.py
│   ├── 02_preprocessing_comparison_train.py / _infer.py
│   ├── 03_multilabel_train.py / _infer.py
│   ├── 04_matched_train.py / _infer.py
│   └── 05_sweep_final.py
└── requirements.txt
```

## License
HyperCyano offers two licensing options to suit different needs:

**AGPL-3.0 License**: This OSI-approved open-source license is perfect for students, researchers, and enthusiasts. It encourages open collaboration and knowledge sharing. See the [LICENSE](https://github.com/capo-urjc/hypercyano/tree/main?tab=AGPL-3.0-1-ov-file) file for full details.

**Enterprise License**: Designed for commercial use, this license permits using this software for commercial solutions, bypassing the open-source requirements of AGPL-3.0. Please contact us for more details.

## Acknowledgments

This work was supported by:

- R&D project PID2024-156033OB-C32 funded by MICIU/AEI/10.13039/501100011033 and FEDER, UE.
- R&D project PID2024-157492OB-I00 funded by MICIU/AEI/10.13039/501100011033 and FEDER, UE.

## Authors

- Tarek Elshami-Ahmed (corresponding author) - `tarek.elshami@urjc.es`
- Laura Llopis-Ibor - `laura.llopis@urjc.es`
- Juan J. Pantrigo - `juanjose.pantrigo@urjc.es`
- Alfredo Cuesta-Infante - `alfredo.cuesta@urjc.es`
- Raúl Cabido - `raul.cabido@urjc.es`

Universidad Rey Juan Carlos, Móstoles, Spain
