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

The code has been tested with Python 3.13.

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/capo-urjc/hypercyano.git
cd hypercyano
python -m venv .venv
```

Activate the virtual environment:

**Linux/macOS:**

```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The processed dataset is included in the repository at `data/cyanobacterial_spectra_processed_clean.xlsx`. No additional downloads are required to run the experiments.

The dataset is derived from the publicly available data published by Fournier et al. ([Zenodo, DOI: 10.5281/zenodo.14056022](https://doi.org/10.5281/zenodo.14056022), CC BY 4.0), with duplicate rows removed.

To regenerate the processed dataset from the original source, run:

```bash
cd src
python download_dataset.py
```

## Usage

The repository contains five experiments corresponding to the analyses presented in the paper. Experiments 1–4 use separate training (`train`) and evaluation (`infer`) scripts. Training scripts fit and save the models and preprocessing pipelines, while evaluation scripts load them and generate the results.

All scripts should be executed from the `src/` directory.

```bash
cd src

# 1. Image-wise evaluation
# Compare spectral-wise and image-wise evaluation protocols using the
# reference models from Fournier et al.
python 01_fournier_replication_train.py
python 01_fournier_replication_infer.py

# 2. Spectral preprocessing
# Compare StandardScaler + Savitzky-Golay with SNV normalization
# under the image-wise evaluation protocol.
python 02_preprocessing_comparison_train.py
python 02_preprocessing_comparison_infer.py

# 3. Multi-label classification
# Predict the presence of individual cyanobacterial species
# instead of treating each mixture as a separate class.
python 03_multilabel_train.py
python 03_multilabel_infer.py

# 4. Robustness assessment
# Evaluate model sensitivity to progressive spectral perturbations
# and compare baseline models with perturbation-trained networks.
# Uses the baseline models from experiment 3.
python 04_matched_train.py
python 04_matched_infer.py

# 5. Hyperparameter optimization
# Bayesian optimization of the multi-label neural network.
# Requires Weights & Biases authentication (wandb login).
python 05_sweep_final.py
```

Trained models and preprocessing pipelines are saved in `models/<experiment>/`, while evaluation results are stored in `results/<experiment>/`.

To view the available options for each script, run:

```bash
python <script>.py --help
```

### Repository layout

The repository is organized into experiment scripts and shared modules for data handling, preprocessing, model training, and evaluation.

```text
hypercyano/
├── data/
│   └── cyanobacterial_spectra_processed_clean.xlsx
├── models/                        # Saved models and preprocessors (not versioned)
├── results/                       # Experimental results (not versioned)
├── src/
│   ├── download_dataset.py        # Dataset download and duplicate removal
│   ├── hsi_data.py                # Data loading and label processing
│   ├── hsi_splits.py              # Spectral-wise and image-wise data splitting
│   ├── hsi_preprocessing.py       # Spectral preprocessing pipelines
│   ├── hsi_models.py              # Model training, prediction, saving, and loading
│   ├── hsi_augmentations.py       # Spectral perturbations
│   ├── hsi_metrics.py             # Evaluation metrics and result summaries
│   ├── 01_fournier_replication_train.py
│   ├── 01_fournier_replication_infer.py
│   ├── 02_preprocessing_comparison_train.py
│   ├── 02_preprocessing_comparison_infer.py
│   ├── 03_multilabel_train.py
│   ├── 03_multilabel_infer.py
│   ├── 04_matched_train.py
│   ├── 04_matched_infer.py
│   └── 05_sweep_final.py
└── requirements.txt
```

## License

This software is available under two licensing options:

- **AGPL-3.0 License:** The source code is distributed under the GNU Affero General Public License v3.0, allowing use, modification, and redistribution under its terms. See the [LICENSE](LICENSE) file for details.

- **Enterprise License:** A commercial licensing option is available for users who require different terms from those of the AGPL-3.0. Please contact the authors for further information.

## Acknowledgments

This work was supported by:

- R&D project PID2024-156033OB-C32 funded by MICIU/AEI/10.13039/501100011033 and FEDER, UE.
- R&D project PID2024-157492OB-I00 funded by MICIU/AEI/10.13039/501100011033 and FEDER, UE.

## Authors

- Tarek Elshami-Ahmed — `tarek.elshami@urjc.es`
- Laura Llopis-Ibor — `laura.llopis@urjc.es`
- Juan J. Pantrigo — `juanjose.pantrigo@urjc.es`
- Alfredo Cuesta-Infante — `alfredo.cuesta@urjc.es`
- Raúl Cabido — `raul.cabido@urjc.es`

Universidad Rey Juan Carlos, Móstoles, Spain.
