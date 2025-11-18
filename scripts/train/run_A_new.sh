#!/usr/bin/env bash
set -euo pipefail

# Matrix A (raw-only + single-station) - New baseline models (ExtraTrees, Linear/Ridge/ElasticNet, Persistence)
# Runs full_training then LOSO for each model sequentially.
#
# Usage:
#   bash scripts/train/run_A_new.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ensure venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate || true
fi

MODELS=("extratrees" "linear_regression" "ridge" "elasticnet" "persistence")
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
  local outdir="experts/A/${model}/raw"
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

echo "✅ Matrix A (raw) - new baseline models completed."


