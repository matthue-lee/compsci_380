# Beta-Protein Mechanics Analysis

Tools for aggregating experimental/structural annotations, exploring feature
distributions, and training baseline models that predict
`Fmax_eps_per_A` for beta-rich proteins.

## Repository Overview

- `dataset.csv` – curated source table of 137 proteins with experimental
  measurements and categorical annotations.
- `build_master_table.py` – merges the curated table with sequence, DSSP, and
  structure-derived features inside `pdb_helpers/` to create
  `master_table.csv`.
- `filter_master_table.py` – selects beta-rich rows into `master_table_beta.csv`
  (default: `ss_fraction_strand >= 0.4`).
- `exploratory.py` – generates summary statistics and plots in `figures/`.
- `linear_baselines.py` – trains mean-imputed ridge/lasso baselines
- `nonlinear_models.py` – trains Random Forest and XGBoost models
  via scikit-learn, producing metrics and feature-importance plots.
- `pdb_helpers/` – helper scripts and downloaded assets (PDB files, FASTA,
  DSSP summaries) referenced by the master-table builder.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- `linear_baselines.py` only needs the Python standard library.
- `nonlinear_models.py` works without xgboost; the XGBoost pipeline is skipped
  automatically if the package is absent.

## Data Preparation Pipeline

1. **Curated dataset**: `dataset.csv` is included; adjust or replace it as
   needed (must contain `PDB`, `Fmax_eps_per_A`, etc.).
2. **Download supplemental inputs (optional)**:
   - Populate `pdb_helpers/pdb.list.to.download` and run
     `python pdb_helpers/download_pdb.py` to fetch any missing PDB files.
   - Provide FASTA sequences (`pdb_helpers/Fasta_Files`), DSSP outputs, and any
     secondary-structure files referenced by `pdb_helpers/`.
3. **Build master table**:
   ```bash
   python3 build_master_table.py \
       --dataset dataset.csv \
       --work-dir pdb_helpers \
       --output master_table.csv
   ```
   Additional flags allow overriding FASTA/DSSP directories.
4. **Filter beta-rich entries** (default threshold 0.4 on
   `ss_fraction_strand`):
   ```bash
   python3 filter_master_table.py \
       --input master_table.csv \
       --output master_table_beta.csv \
       --column ss_fraction_strand \
       --min-value 0.4
   ```

## Exploratory Analysis

```bash
python3 exploratory.py
```

- Reads `dataset.csv`, prints dataset summaries, and writes plots (histograms,
  correlations, pairplots, categorical boxplots) to `figures/`.
- Set `DATA_PATH` or pass a different CSV by editing the script if desired.

## Linear Baseline Models

```bash
python3 linear_baselines.py \
    --data master_table_beta.csv \
    --target Fmax_eps_per_A \
    --length-feature N \
    --test-size 0.2 \
    --seed 42
```

- Implements mean-imputation, standardization, and gradient-descent training
  for ridge and lasso regressors, plus a single-feature linear baseline on `N`.
- Outputs RMSE/R² for the holdout split and optional custom folds
  (see `--folds`, `--length-feature`, and `--seed`).

## Nonlinear Models

```bash
python3 nonlinear_models.py \
    --data master_table_beta.csv \
    --target Fmax_eps_per_A \
    --folds 5 \
    --seed 42 \
    --output-dir figures \
    --top-features 15
```

- Loads numeric features, applies per-column median imputation via scikit-learn
  pipelines, and evaluates each model with shuffled K-Fold (and optional LOOCV
  unless `--skip-loocv` is set).
- Saves feature-importance bar plots to `figures/` for models exposing
  `feature_importances_`.

## Tips

- All scripts default to the included CSVs/paths; override with CLI flags as
  needed.
- `figures/` is reused across exploratory and model scripts—clean it manually
  if you want a fresh set of outputs.
- Use `python3 -m py_compile <script>.py` to sanity-check syntax before running
  longer pipelines.

