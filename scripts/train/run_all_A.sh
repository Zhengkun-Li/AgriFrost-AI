#!/usr/bin/env bash
set -euo pipefail

# Matrix A (raw-only + single-station) - run all models (full_training + LOSO) sequentially
# Output structure: experiments/A/{model}/raw/{full_training|loso}/...
#
# Usage:
#   bash scripts/train/run_all_A.sh
#
# Logs:
#   - Full training:   experiments/A/{model}/raw/training_root.log
#   - LOSO evaluation: experiments/A/{model}/raw/loso_root.log

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODELS=("lightgbm" "xgboost" "catboost" "random_forest" "extratrees" "linear_regression" "ridge" "elasticnet" "ensemble" "persistence")
HORIZONS=("3" "6" "12" "24")

run_full_training() {
  local model="$1"
  local outdir="experiments/A/${model}/raw"
  mkdir -p "${outdir}"
  echo "==== [A/raw] Full training: ${model} ===="
  PYTHONUNBUFFERED=1 python3 scripts/train/train_frost_forecast.py \
    --horizons ${HORIZONS[*]} \
    --model "${model}" \
    --output "${outdir}" \
    | tee -a "${outdir}/training_root.log"
}

run_loso() {
  local model="$1"
  local outdir="experiments/A/${model}/raw"
  mkdir -p "${outdir}"
  echo "==== [A/raw] LOSO: ${model} ===="
  PYTHONUNBUFFERED=1 python3 scripts/train/train_frost_forecast.py \
    --horizons ${HORIZONS[*]} \
    --model "${model}" \
    --loso \
    --save-loso-models \
    --output "${outdir}" \
    | tee -a "${outdir}/loso_root.log"
}

for model in "${MODELS[@]}"; do
  run_full_training "${model}"
  run_loso "${model}"
done

echo "✅ Matrix A (raw) - all models completed."


