# Beyond spectral-wise evaluation: Generalization-aware multi-label learning for hyperspectral cyanobacteria identification

This repository provides the code and experimental framework developed to identify cyanobacterial species in hyperspectral images using machine learning. The study focuses on evaluating model generalization to unseen acquisitions, improving the scalability of mixture identification through multi-label classification, and assessing robustness to spectral variations.

The experiments build on the Random Forest and feed-forward neural network proposed by Fournier et al., allowing direct comparisons with previously reported results.

## Key Features

- **Image-wise evaluation.** Introduces an evaluation protocol that separates training, validation, and testing at the hyperspectral-image level, preventing information leakage between acquisitions and providing a more realistic assessment of model generalization.

- **Spectral preprocessing.** Compares the original StandardScaler and Savitzky-Golay preprocessing pipeline with Standard Normal Variate (SNV) normalization.

- **Multi-label classification.** Reformulates cyanobacterial mixture identification by independently predicting the presence of each species, avoiding the need to define every possible combination as a separate class.

- **Robustness assessment.** Evaluates model sensitivity to seven types of spectral perturbations at increasing intensity levels, including noise, illumination variations, spectral shifts, and band masking. It also examines whether perturbation-specific training improves robustness.

- **Hyperparameter optimization.** Explores neural network configurations through Bayesian optimization to assess potential improvements in multi-label classification performance.

## Installation

Tested with Python 3.13.

```bash
git clone https://github.com/capo-urjc/hypercyano.git
cd hypercyano
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset
 
The cleaned dataset is already included in this repository at `data/cyanobacterial_spectra_processed_clean.xlsx`, so you don't need to download anything to run the experiments.
 
It is derived from the public dataset by Fournier et al. ([10.5281/zenodo.14056022](https://doi.org/10.5281/zenodo.14056022), CC-BY-4.0), after removing duplicate rows. To regenerate it from the original source:
 
```bash
cd src
python download_dataset.py
```

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
│   ├── download_dataset.py        # downloads the Zenodo dataset and removes duplicate rows
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
