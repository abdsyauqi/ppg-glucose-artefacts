# Small-sample artefacts in PPG-based blood glucose estimation

Code, extracted features and numerical results for the manuscript
*Small-sample artefacts in photoplethysmography-based blood glucose estimation:
morphological feature analysis with independent-dataset replication*.

## Contents

| Path | What it is |
|---|---|
| `src/ppg_features.py` | Preprocessing and the 16 morphological features (Table 2 of the manuscript) |
| `scripts/01_extract_*.py` | Feature extraction for MUST and STU-BA |
| `scripts/02_within_dataset.py` | Correlations (pooled and repeated-measures), variance components, LOSO, within-subject and temporally held-out models, honest correlations and feature stability with permutation nulls |
| `scripts/03_cross_dataset.py` | Cross-dataset transfer and pooled LOSO over 47 subjects |
| `scripts/04_subgroup.py` | Subject-level subgroup analysis |
| `scripts/05_figures.py`, `06_tables.py` | Figures 2 to 7 and supplementary tables |
| `features/` | Extracted feature tables for both datasets |
| `results/` | All numerical results (JSON) and generated LaTeX tables |
| `figures/` | Generated figures |

## Reproducing

    pip install -r requirements.txt
    ./run_all.sh

Raw data are not included; see `data/README.md`. Every analysis from step 2 onwards runs
from `features/`. Random seeds are fixed; bootstrap and permutation counts are set in the scripts.

## Datasets

- MUST: Kermani A, Esmaeili H (2023), Mendeley Data, doi:10.17632/37pm7jk7jn.3
- STU-BA: Berki M, Mateasik A (2026), Zenodo, doi:10.5281/zenodo.22206256

## Licence

Code: MIT. Feature tables are derived from the datasets above and follow their licences.
