#!/usr/bin/env bash
set -euo pipefail

# Matrix A (raw-only + single-station) - Deep models (LSTM + LSTM Multi-task)
# Runs full_training then LOSO for each model sequentially.
# Output: experiments/A/{model}/raw/{full_training|loso}/...
#
# Usage:
#   bash scripts/train/run_A_deep.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ensure we use project venv if exists
if [ -d ".${VIRTUAL_ENV:-}" ]; then
  source .venv/bin/activate || true
fi

MODELS=("lstm" "lstm_multitask")
HORIZONS=("3" "6" "12" "24")

run_full_training() {
  local model="$1"
  local outdir="experiments/A/${model}/raw"
  mkdir -p "${outdir}" || true
  # Force retrain: clean previous full_training outputs
  if [ -d "${outdir}/full_training" ]; then
    echo "Forcing retrain: removing ${outdir}/full_training/horizon_*"
    rm -rf "${outdir}/full_training/horizon_"* || true
  fi
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
  # Force retrain: clean previous LOSO outputs
  if [ -d "${outdir}/loso" ]; then
    echo "Forcing retrain: removing ${outdir}/loso"
    rm -rf "${outdir}/loso" || true
  fi
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

echo "✅ Matrix A (raw) - deep models completed."


