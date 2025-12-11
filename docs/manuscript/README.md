# Manuscript Reproduction Guide

This directory contains a minimal set of scripts (in `scripts/`) to reproduce the manuscript results with LightGBM on matrices A/B/C, generate figures/tables, and run LOSO evaluation.

## 1. Environment
1. `cd /home/zhengkun-li/frost-risk-forecast-challenge`
2. (Recommended) create/activate venv  
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## 2. Where the scripts live
`docs/manuscript/scripts` (copied from `Supplementary_lighgbm_abc/scripts`):
- `run_all.sh` — one-click pipeline: collect metrics & feature importance, optional LOSO, generate all manuscript figures.
- `run_all_balanced_experiments.sh` / `run_balanced_experiments_sequential.sh` — class-balanced training (A/B/C; C sweeps radii). Sequential version is safer on memory.
- `train.py` — CLI entry to train a single model/horizon.
- `collect_all_metrics_for_supplementary.py`, `collect_all_feature_importance_for_supplementary.py` — aggregate metrics / feature importance to CSV.
- Figure generators: `generate_class_balance_comparison_figure.py`, `generate_matrix_a_feature_importance_figure.py`, `generate_matrix_b_feature_category_importance_cumulative.py`, `generate_matrix_c_feature_category_importance_cumulative.py`, `generate_loso_boxplot_comparison.py`.
- LOSO runner: `run_loso_for_abc_matrices.py`.

## 3. Training
### 3.1 Class-balanced (recommended, reproduces manuscript numbers)
Fast (may use more memory):
```bash
./docs/manuscript/scripts/run_all_balanced_experiments.sh
```
Memory-safe (sequential):
```bash
./docs/manuscript/scripts/run_balanced_experiments_sequential.sh
```
Tips:
- To limit to LightGBM only, edit `MODELS=("lightgbm")` in the script.
- To restrict to A/B/C, comment/remove Matrix D blocks (C already sweeps radii 20–200 km).

### 3.2 Baseline / no-balance runs
Use the CLI to launch an unbalanced run (example: Matrix A, 3h):
```bash
python -m src.cli train single \
  --model-name lightgbm \
  --matrix-cell A \
  --track raw \
  --horizon-h 3 \
  --output-dir experiments/lightgbm/raw/A/full_training_unbalance
```
For Matrix B:
```bash
python -m src.cli train single \
  --model-name lightgbm \
  --matrix-cell B \
  --track feature_engineering \
  --horizon-h 3 \
  --output-dir experiments/lightgbm/feature_engineering/B/full_training_unbalance
```
For Matrix C (set radius):
```bash
python -m src.cli train single \
  --model-name lightgbm \
  --matrix-cell C \
  --track raw \
  --horizon-h 3 \
  --radius-km 60 \
  --output-dir experiments/lightgbm/raw/C/full_training_unbalance_60km
```
Adjust `--horizon-h {3,6,12,24}` and radius (C) as needed.

## 4. Feature importance (A/B/C)
Assumes training outputs are under `experiments/` as above.
```bash
python docs/manuscript/scripts/collect_all_feature_importance_for_supplementary.py
```
- Matrix A: uses per-feature gain.
- Matrix B/C: computes 90% cumulative importance then aggregates by category.

## 5. LOSO (A, B; C optional — high memory)
```bash
python docs/manuscript/scripts/run_loso_for_abc_matrices.py
```
Notes:
- Uses best configs from `docs/manuscript/Supplementary_lighgbm_abc/summary_best_configurations.csv` if present; otherwise defaults (A/B balanced, C radii 60/100/200/200 km).
- To skip C (memory), edit the script to set `matrices = ["A","B"]`.

## 6. Generate manuscript figures/tables
One-click (after training artifacts exist):
```bash
./docs/manuscript/scripts/run_all.sh
```
Individual figures:
- Class-balance impact: `python docs/manuscript/scripts/generate_class_balance_comparison_figure.py`
- Matrix A importance: `python docs/manuscript/scripts/generate_matrix_a_feature_importance_figure.py`
- Matrix B category importance (90% cum.): `python docs/manuscript/scripts/generate_matrix_b_feature_category_importance_cumulative.py`
- Matrix C category importance (90% cum.): `python docs/manuscript/scripts/generate_matrix_c_feature_category_importance_cumulative.py`
- LOSO boxplots (A vs B): `python docs/manuscript/scripts/generate_loso_boxplot_comparison.py`

Aggregated metrics:
```bash
python docs/manuscript/scripts/collect_all_metrics_for_supplementary.py
```

## 7. Outputs
- Experiments: `experiments/<model>/<track>/<matrix>/...`
- Supplementary CSVs/figures: `docs/manuscript/Supplementary_lighgbm_abc/`
- Manuscript figures: `docs/manuscript/figures/`

## 8. Troubleshooting
- Memory: prefer `run_balanced_experiments_sequential.sh` and limit `MODELS` to lightgbm.
- GPU not required for LightGBM. Ensure `.venv` matches repo `requirements.txt`.
- If LOSO runs long, run per-matrix by editing `run_loso_for_abc_matrices.py`.

