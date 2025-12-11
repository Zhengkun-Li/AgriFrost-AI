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

### 3.2 Baseline / no-balance runs (not default)
LightGBM defaults to class-balanced training (`is_unbalance=True` in `src/training/model_config.py`). If you truly need unbalanced runs, you must **override `is_unbalance` to `False` in a custom config** before running. A simple way:
1) Copy `config/pipeline/matrix_a.yaml` (or `matrix_b.yaml`, `matrix_c.yaml`) to a new file, e.g. `config/pipeline/matrix_a_unbalance.yaml`.
2) Add under `training:`:
```
training:
  model_params:
    classification:
      is_unbalance: false
```
3) Run with the custom config (example: Matrix A, 3h):
```bash
python -m src.cli train single \
  --model-name lightgbm \
  --matrix-cell A \
  --track raw \
  --horizon-h 3 \
  --config config/pipeline/matrix_a_unbalance.yaml \
  --output-dir experiments/lightgbm/raw/A/full_training_unbalance
```
Repeat similarly for Matrix B/C, adjusting `matrix_cell`, `track`, `radius-km` (for C), and output dir. Recommendation: keep class-balanced training for manuscript reproduction.

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

