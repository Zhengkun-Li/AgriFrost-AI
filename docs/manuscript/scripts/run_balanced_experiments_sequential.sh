#!/bin/bash
# Sequential training script for class-balanced experiments
# Runs one experiment at a time to avoid memory overflow (>50GB)
# Runs for 12 hours maximum
# Models: LightGBM, XGBoost, CatBoost
# Matrices: A, B, C (all radius), D (all radius)

# Note: We don't use 'set -e' because we want to continue even if one experiment fails

PROJECT_ROOT="/home/zhengkun-li/frost-risk-forecast-challenge"
cd "$PROJECT_ROOT"

source .venv/bin/activate

# Configuration
MAX_MEMORY_GB=50
MAX_RUNTIME_HOURS=12
START_TIME=$(date +%s)
MAX_RUNTIME_SECONDS=$((MAX_RUNTIME_HOURS * 3600))

# Models to train
MODELS=("lightgbm" "xgboost" "catboost")

# Radius list for Matrix C and D
RADIUS_LIST=(20 40 60 80 100 120 140 160 180 200)

# Log file
LOG_FILE="scripts/balanced_training_sequential.log"
mkdir -p scripts
echo "==========================================" >> "$LOG_FILE"
echo "Training started at: $(date)" >> "$LOG_FILE"
echo "Max runtime: ${MAX_RUNTIME_HOURS} hours" >> "$LOG_FILE"
echo "Max memory: ${MAX_MEMORY_GB}GB" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

# Function to check elapsed time
check_time_limit() {
    local current_time=$(date +%s)
    local elapsed=$((current_time - START_TIME))
    if [ $elapsed -ge $MAX_RUNTIME_SECONDS ]; then
        echo "⏰ Time limit reached (${MAX_RUNTIME_HOURS} hours). Stopping." | tee -a "$LOG_FILE"
        return 1
    fi
    return 0
}

# Function to check memory usage
check_memory() {
    local mem_used_gb=$(free -g | awk '/^Mem:/{print $3}')
    if [ "$mem_used_gb" -gt "$MAX_MEMORY_GB" ]; then
        echo "⚠️  Memory usage high: ${mem_used_gb}GB > ${MAX_MEMORY_GB}GB. Waiting..." | tee -a "$LOG_FILE"
        sleep 60
        return 1
    fi
    return 0
}

# Function to check if experiment is completed
is_completed() {
    local model=$1
    local matrix=$2
    local track=$3
    local radius=$4
    local output_dir="experiments/${model}/${track}/${matrix}/full_training_balance"
    
    if [ "$radius" != "none" ]; then
        output_dir="experiments/${model}/${track}/${matrix}/full_training_balance_${radius}km"
    fi
    
    local summary_file="${output_dir}/full_training/summary.json"
    [ -f "$summary_file" ]
}

# Function to run training
run_training() {
    local model=$1
    local matrix=$2
    local track=$3
    local radius=$4
    local output_dir="experiments/${model}/${track}/${matrix}/full_training_balance"
    
    if [ "$radius" != "none" ]; then
        output_dir="experiments/${model}/${track}/${matrix}/full_training_balance_${radius}km"
    fi
    
    # Create directory
    mkdir -p "$output_dir"
    
    local log_file="${output_dir}/experiment.log"
    local start_time=$(date +%s)
    
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Training: ${model} - Matrix ${matrix} - radius=${radius}km" | tee -a "$LOG_FILE"
    echo "Start time: $(date)" | tee -a "$LOG_FILE"
    echo "Output: ${output_dir}" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    
    # Build command
    local cmd="python -m src.cli train single --model-name ${model} --matrix-cell ${matrix} --track ${track} --horizon-h 3"
    
    if [ "$radius" != "none" ]; then
        cmd="${cmd} --radius-km ${radius}"
    fi
    
    cmd="${cmd} --output-dir ${output_dir}"
    
    # Run training
    eval "$cmd" > "$log_file" 2>&1
    local exit_code=$?
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))
    
    if [ $exit_code -eq 0 ] && [ -f "${output_dir}/full_training/summary.json" ]; then
        echo "✅ Training completed successfully in ${duration_min} minutes" | tee -a "$LOG_FILE"
        return 0
    else
        echo "❌ Training failed or incomplete (exit code: $exit_code, duration: ${duration_min} minutes)" | tee -a "$LOG_FILE"
        echo "Last 10 lines of log:" | tee -a "$LOG_FILE"
        tail -10 "$log_file" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Generate experiment list
generate_experiments() {
    local experiments=()
    
    # Matrix A (raw, no radius)
    for model in "${MODELS[@]}"; do
        experiments+=("${model}|A|raw|none")
    done
    
    # Matrix B (feature_engineering, no radius)
    for model in "${MODELS[@]}"; do
        experiments+=("${model}|B|feature_engineering|none")
    done
    
    # Matrix C (raw, all radius)
    for model in "${MODELS[@]}"; do
        for radius in "${RADIUS_LIST[@]}"; do
            experiments+=("${model}|C|raw|${radius}")
        done
    done
    
    # Matrix D (feature_engineering, all radius)
    for model in "${MODELS[@]}"; do
        for radius in "${RADIUS_LIST[@]}"; do
            experiments+=("${model}|D|feature_engineering|${radius}")
        done
    done
    
    printf '%s\n' "${experiments[@]}"
}

# Main execution
echo "=========================================="
echo "Sequential Class-Balanced Training Script"
echo "=========================================="
echo "Max runtime: ${MAX_RUNTIME_HOURS} hours"
echo "Max memory: ${MAX_MEMORY_GB}GB"
echo "Models: ${MODELS[*]}"
echo "Matrices: A, B, C (all radius), D (all radius)"
echo ""

# Generate and filter experiments
EXPERIMENTS=$(generate_experiments)
TOTAL_EXPERIMENTS=$(echo "$EXPERIMENTS" | wc -l)
COMPLETED_COUNT=0
FAILED_COUNT=0

echo "Total experiments: ${TOTAL_EXPERIMENTS}"
echo ""

# Process each experiment
while IFS='|' read -r model matrix track radius; do
    # Check time limit
    if ! check_time_limit; then
        break
    fi
    
    # Check memory before starting
    while ! check_memory; do
        if ! check_time_limit; then
            exit 0
        fi
        sleep 30
    done
    
    # Skip if already completed
    if is_completed "$model" "$matrix" "$track" "$radius"; then
        echo "⏭️  Skipping ${model} - Matrix ${matrix} - radius=${radius}km (already completed)"
        ((COMPLETED_COUNT++))
        continue
    fi
    
    # Run training
    if run_training "$model" "$matrix" "$track" "$radius"; then
        ((COMPLETED_COUNT++))
    else
        ((FAILED_COUNT++))
        # Continue with next experiment even if one fails
    fi
    
    # Wait a bit between experiments
    sleep 5
    
    # Progress update
    progress=$((COMPLETED_COUNT * 100 / TOTAL_EXPERIMENTS))
    echo ""
    echo "Progress: ${COMPLETED_COUNT}/${TOTAL_EXPERIMENTS} (${progress}%)" | tee -a "$LOG_FILE"
    echo ""
    
done <<< "$EXPERIMENTS"

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_DURATION_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_DURATION_MIN=$((TOTAL_DURATION % 3600 / 60))

echo "==========================================" | tee -a "$LOG_FILE"
echo "Training completed at: $(date)" | tee -a "$LOG_FILE"
echo "Total duration: ${TOTAL_DURATION_HOURS}h ${TOTAL_DURATION_MIN}m" | tee -a "$LOG_FILE"
echo "Completed: ${COMPLETED_COUNT}/${TOTAL_EXPERIMENTS}" | tee -a "$LOG_FILE"
echo "Failed: ${FAILED_COUNT}" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

