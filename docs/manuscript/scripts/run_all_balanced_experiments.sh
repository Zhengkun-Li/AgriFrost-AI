#!/bin/bash
# Script to run all class-balanced training experiments
# Matrix C and D with all radius values

set -e

PROJECT_ROOT="/home/zhengkun-li/frost-risk-forecast-challenge"
cd "$PROJECT_ROOT"

source .venv/bin/activate

# Radius list for Matrix C and D
RADIUS_LIST=(20 40 60 80 100 120 140 160 180 200)

# Models to train
MODELS=("lightgbm" "xgboost" "catboost")

# Function to check if experiment is completed
check_completed() {
    local model=$1
    local matrix=$2
    local track=$3
    local radius=$4
    local output_dir="experiments/${model}/${track}/${matrix}/full_training_balance_${radius}km"
    local summary_file="${output_dir}/full_training/summary.json"
    
    if [ -f "$summary_file" ]; then
        return 0  # Completed
    else
        return 1  # Not completed
    fi
}

# Function to run training
run_training() {
    local model=$1
    local matrix=$2
    local track=$3
    local radius=$4
    local output_dir="experiments/${model}/${track}/${matrix}/full_training_balance_${radius}km"
    
    echo "=========================================="
    echo "Training: ${model} - Matrix ${matrix} - radius=${radius}km"
    echo "Output: ${output_dir}"
    echo "=========================================="
    
    # Create directory if not exists
    mkdir -p "$output_dir"
    
    # Run training
    python -m src.cli train single \
        --model-name "$model" \
        --matrix-cell "$matrix" \
        --track "$track" \
        --horizon-h 3 \
        --radius-km "$radius" \
        --output-dir "$output_dir" > "${output_dir}/experiment.log" 2>&1
    
    # Check if training was successful
    if [ -f "${output_dir}/full_training/summary.json" ]; then
        echo "✅ Training completed successfully"
        return 0
    else
        echo "❌ Training failed or incomplete"
        return 1
    fi
}

# Matrix C (raw track)
echo "=========================================="
echo "Starting Matrix C experiments"
echo "=========================================="

for model in "${MODELS[@]}"; do
    for radius in "${RADIUS_LIST[@]}"; do
        if check_completed "$model" "C" "raw" "$radius"; then
            echo "⏭️  Skipping ${model} - Matrix C - radius=${radius}km (already completed)"
            continue
        fi
        
        echo "🔄 Running ${model} - Matrix C - radius=${radius}km"
        if run_training "$model" "C" "raw" "$radius"; then
            echo "✅ Completed: ${model} - Matrix C - radius=${radius}km"
        else
            echo "❌ Failed: ${model} - Matrix C - radius=${radius}km"
            # Continue with next experiment
        fi
        
        # Wait a bit between experiments to avoid memory issues
        sleep 5
    done
done

# Matrix D (feature_engineering track)
echo "=========================================="
echo "Starting Matrix D experiments"
echo "=========================================="
echo "⚠️  Warning: Matrix D has 866 features - high memory usage!"
echo "Memory optimization is automatically applied"

for model in "${MODELS[@]}"; do
    for radius in "${RADIUS_LIST[@]}"; do
        if check_completed "$model" "D" "feature_engineering" "$radius"; then
            echo "⏭️  Skipping ${model} - Matrix D - radius=${radius}km (already completed)"
            continue
        fi
        
        echo "🔄 Running ${model} - Matrix D - radius=${radius}km"
        if run_training "$model" "D" "feature_engineering" "$radius"; then
            echo "✅ Completed: ${model} - Matrix D - radius=${radius}km"
        else
            echo "❌ Failed: ${model} - Matrix D - radius=${radius}km"
            # Continue with next experiment
        fi
        
        # Wait longer for Matrix D due to memory sensitivity
        sleep 10
    done
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

