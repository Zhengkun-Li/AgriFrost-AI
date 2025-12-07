# AgriFrost-AI Training and Evaluation Complete Guide

<div align="center">

<img src="../logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

</div>

**Last Updated**: 2025-12-06

This document integrates all training-related content including training configuration, LOSO evaluation, training monitoring, and command details, providing a one-stop reference for model training.

## 📋 Table of Contents

1. [Environment Preparation](#environment-preparation)
2. [Training Command Details](#training-command-details)
3. [Training Configuration](#training-configuration)
4. [LOSO Evaluation](#loso-evaluation)
5. [Training Monitoring](#training-monitoring)
6. [Performance Comparison](#performance-comparison)
7. [Command Line Details](#command-line-details)
8. [Frequently Asked Questions](#frequently-asked-questions)

---

## Environment Preparation

### ⚠️ Important: Use Virtual Environment

Before starting training, ensure you have created and activated a virtual environment:

```bash
# Create virtual environment (if not already created)
python3 -m venv .venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate.bat

# Ensure all dependencies are installed
pip install -r requirements.txt
```

**Verify environment:**
```bash
# Check if CLI is available
python -m src.cli --help

# Check key dependencies
python -c "import lightgbm, xgboost, torch; print('✅ Environment ready!')"
```

For more environment setup instructions, see [Quick Start Guide](../guides/QUICK_START.md#1-environment-setup).

---

## Training Command Details

### Basic Command Format

```bash
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell A \
    --track raw \
    --horizon-h 12 \
    --config config/pipeline/train_with_loso.yaml \
    --data-path data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz
```

### ⚠️ **Key Issue: Actual Effect of `--horizon-h 12`**

#### Observation

Although `--horizon-h 12` is specified in the command, the actual training **includes all horizons** (3h, 6h, 12h, 24h).

From the experiment directory, you can see:
```
experiments/lightgbm/raw/A/full_training/full_training/
  ├── horizon_3h/
  ├── horizon_6h/
  ├── horizon_12h/
  └── horizon_24h/
```

#### Cause Analysis

##### 1. **Label Generation Stage**

In the `single()` function of `src/cli/commands/train.py`:

```python
# CRITICAL: Generate labels for all horizons [3, 6, 12, 24] even when training single horizon
# This ensures labeled_data.parquet contains all horizon labels
cli_overrides: Dict[str, Any] = {
    "model": model_name,
    "matrix_cell": matrix_cell,
    "horizons": [3, 6, 12, 24],  # Generate labels for all horizons
}
```

**Key point**: Even though `--horizon-h 12` is specified in the command line, the code **forces** `horizons: [3, 6, 12, 24]` to generate labels for all horizons.

**Reason**: This fixes a previous bug, ensuring `labeled_data.parquet` contains labels for all horizons (`frost_3h`, `frost_6h`, `frost_12h`, `frost_24h`).

##### 2. **Configuration File Override**

In `config/pipeline/train_with_loso.yaml`:

```yaml
labels:
  horizons: [3, 6, 12, 24]
```

The configuration file also specifies all horizons, which merges with CLI parameters.

##### 3. **Actual Training Stage**

In `src/training/pipeline_runner.py`:

```python
# Train only horizons that have labels
training_horizons = [h for h in self.config.labels.horizons if h in available_horizons]
```

`TrainingRunner` trains **all horizons with labels** in `config.labels.horizons`.

Since label generation creates labels for all horizons (`[3, 6, 12, 24]`), **all horizons will be trained**.

##### 4. **Actual Purpose of `--horizon-h 12`**

**The `--horizon-h 12` parameter is mainly used for**:
- Generating hint information in output paths
- Displaying horizon information in success messages
- **Does not actually limit training horizons**

### Complete Training Workflow

```
1. Command Line Parsing
   ├── --horizon-h 12 (for hint information)
   └── Other parameters

2. Configuration Merging
   ├── CLI overrides: horizons = [3, 6, 12, 24] (forced)
   ├── Config file: horizons: [3, 6, 12, 24]
   └── Final config: horizons = [3, 6, 12, 24]

3. Data Loading and Label Generation
   ├── DataPipeline.run() generates labels for all horizons
   └── labeled_data.parquet contains: frost_3h, frost_6h, frost_12h, frost_24h

4. Training Stage
   ├── TrainingRunner iterates through config.labels.horizons
   ├── Train 3h horizon → horizon_3h/
   ├── Train 6h horizon → horizon_6h/
   ├── Train 12h horizon → horizon_12h/
   └── Train 24h horizon → horizon_24h/

5. LOSO Evaluation (if enabled)
   └── Evaluate all horizons
```

### How to Train Only a Single Horizon?

#### Method 1: Modify Configuration File

Create or modify configuration file to specify only one horizon:

```yaml
labels:
  horizons: [12]  # Only train 12h
```

**Note**: Due to forced settings in code, this method **may not work**.

#### Method 2: Modify Code

If you need to train only a single horizon, modify `src/cli/commands/train.py`:

```python
# Before:
"horizons": [3, 6, 12, 24],  # Generate labels for all horizons

# After:
"horizons": [horizon_h],  # Only generate labels for specified horizon
```

**Note**: This may cause other issues (LOSO evaluation may fail).

### ✅ **Current Behavior Summary**

| Item | Description |
|------|-------------|
| **Command line parameter** | `--horizon-h 12` |
| **Label generation** | Generates labels for all horizons `[3, 6, 12, 24]` |
| **Actual training** | Trains all horizons `[3, 6, 12, 24]` |
| **Output directory** | Contains subdirectories for all horizons |
| **`--horizon-h` effect** | Mainly for hint information, does not limit training horizons |

---

## Training Configuration

### Hardware Configuration

- **GPU**: NVIDIA RTX 5090 (32GB)
- **CPU**: AMD 9950 (32 cores)
- **Memory**: 60GB

### Data Scale

- **Total data**: 2,367,360 rows
- **Number of stations**: 18
- **Time range**: 2010-09-28 to 2025-09-28 (15 years of data)

### Model Configuration Optimization

```python
{
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 63,
    "n_jobs": 8,  # Limit CPU core usage (avoid memory overflow)
    "force_col_wise": True,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}
```

### Start Training

Use the new CLI interface:

```bash
# Activate virtual environment
source .venv/bin/activate

# Train single model
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track feature_engineering \
    --horizon-h 12 \
    --output-dir experiments/lightgbm_B_12h

# Batch training (Matrix Experiments)
python -m src.cli train matrix \
    --config config/pipeline/matrix_experiments.yaml
```

### Estimated Training Time

- **Data loading**: ~2-5 minutes
- **Data cleaning**: ~5-10 minutes
- **Feature engineering**: ~30-60 minutes
- **Standard evaluation training** (4 time windows): ~40-80 minutes
- **LOSO evaluation training** (18 stations × 4 time windows): ~180-360 minutes (3-6 hours)

**Total estimated time** (including LOSO evaluation): **4-7 hours**

**Note**:
- If not running LOSO evaluation (without `--loso` parameter), total time is approximately **1.5-2.5 hours**
- If running LOSO evaluation, total time is approximately **4-7 hours** (LOSO evaluation requires additional 3-6 hours)

---

## LOSO Evaluation

### What is LOSO?

**LOSO (Leave-One-Station-Out)** is a cross-validation method used to evaluate model spatial generalization capability.

### LOSO Evaluation Workflow

1. **Select one station as test set**
   - Example: Select station "Davis" as test set
   - Other 17 stations as training set

2. **Train model using training set**
   - Train model using all station data except "Davis"

3. **Evaluate model using test set**
   - Evaluate model performance using "Davis" station data

4. **Repeat the above process**
   - Repeat for each station
   - Finally get evaluation results for 18 stations

5. **Summarize results**
   - Calculate average performance across all stations
   - Calculate standard deviation to assess performance stability

### Advantages of LOSO Evaluation

1. ✅ **Spatial generalization**: Evaluate model performance on unseen stations
2. ✅ **Robustness assessment**: Evaluate model adaptability to different microclimates
3. ✅ **Practical application value**: Closer to actual deployment scenarios

### Enable LOSO Evaluation

Enable LOSO evaluation in configuration file:

```yaml
# config/pipeline/train_with_loso.yaml
training:
  loso:
    enabled: true
    params:
      stations: null  # null means use all stations
      horizons: [3, 6, 12, 24]
```

Or use CLI parameter:

```bash
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell A \
    --track raw \
    --horizon-h 12 \
    --config config/pipeline/train_with_loso.yaml
```

### LOSO Evaluation Results

LOSO evaluation results are saved in:

```
experiments/lightgbm/raw/A/full_training/loso/
  ├── summary.json          # Summary statistics (mean ± std)
  └── station_metrics.json  # Detailed metrics for each station
```

**Summary statistics example**:
```json
{
  "summary": {
    "3h": {
      "frost_metrics": {
        "brier_score": {"mean": 0.1234, "std": 0.0123},
        "ece": {"mean": 0.0567, "std": 0.0045},
        "roc_auc": {"mean": 0.9876, "std": 0.0089},
        "pr_auc": {"mean": 0.8765, "std": 0.0234}
      },
      "temp_metrics": {
        "mae": {"mean": 1.23, "std": 0.45},
        "rmse": {"mean": 1.56, "std": 0.67},
        "r2": {"mean": 0.91, "std": 0.08}
      }
    },
    ...
  }
}
```

---

## Training Monitoring

### Log Files

Multiple log files are generated during training:

#### 1. **Experiment-Level Log** (`experiment.log`)

Location: `experiments/<model>/<track>/<cell>/<scope>/experiment.log`

**Content**:
- Data loading information (sample count, feature count, station count, date range, label statistics)
- Train/validation/test set split information
- Training result summary for each horizon
- LOSO evaluation result summary (mean ± std, detailed metrics for top 10 stations)
- Total experiment duration

**Example**:
```
[Data Loading]
  ✅ Data loaded successfully
  📊 Total samples: 2,367,360
  📊 Total features: 12
  📊 Stations: 18
  📊 Date range: 2010-09-28 to 2025-09-28

[Label Statistics]
  3h: 45,234 frost events (1.91%)
  6h: 89,567 frost events (3.78%)
  12h: 156,789 frost events (6.62%)
  24h: 234,567 frost events (9.90%)

[Training]
  Training horizon: 12h
    ✅ Training completed in 123.45 seconds (2.06 minutes)
    📊 Frost Metrics:
       ROC-AUC: 0.9892
       PR-AUC: 0.8765
       Brier Score: 0.1234
    📊 Temp Metrics:
       MAE: 1.84°C
       RMSE: 2.45°C
       R²: 0.9270
    📁 Model saved to: horizon_12h/

[LOSO Evaluation]
  ✅ LOSO evaluation completed in 1800.00 seconds (30.00 minutes)
  📊 LOSO Results Summary (across all stations):
    Horizon 12h:
      Frost Metrics:
        Brier Score: 0.1345 ± 0.0123
        Expected Calibration Error (ECE): 0.0567 ± 0.0045
        ROC-AUC (discrimination): 0.9876 ± 0.0089
        PR-AUC (discrimination): 0.8765 ± 0.0234
      Temp Metrics:
        MAE: 1.96°C ± 0.45°C
        RMSE: 2.56°C ± 0.67°C
        R²: 0.9167 ± 0.0800
```

#### 2. **Horizon-Level Log** (`training.log`)

Location: `experiments/<model>/<track>/<cell>/<scope>/horizon_<h>/training.log`

**Content**:
- Data preparation information (feature count, sample count, frost event count, **feature list**)
- Data split information (train/validation/test set sizes and percentages)
- Training process details (metrics for each epoch)
- Evaluation result details (calibration metrics, discrimination skill metrics)
- Model save location

**Example**:
```
📊 Data preparation:
   Features: 12
   Samples: 2,367,360
   Frost events: 156,789 (6.62%)
   Feature list: Hour (PST), Jul, ETo (mm), Precip (mm), ...

📊 Data split:
   Train: 1,657,152 (70.0%)
   Val: 355,104 (15.0%)
   Test: 355,104 (15.0%)

📊 Evaluation Results:
   Calibration & Reliability:
     Brier Score: 0.1234
     Expected Calibration Error (ECE): 0.0567
     Reliability Diagram: horizon_12h/reliability_diagram.png
   Discrimination Skill:
     ROC-AUC: 0.9892
     PR-AUC: 0.8765
   Temp Metrics:
     MAE: 1.84°C
     RMSE: 2.45°C
     R²: 0.9270
   Evaluation time: 12.34 seconds
   Model saved to: horizon_12h/
```

### Feature Importance Files

After training completes, feature importance is automatically saved:

```
experiments/lightgbm/raw/A/full_training/full_training/horizon_12h/
  ├── frost_feature_importance.csv    # Feature importance for frost classification model
  └── temp_feature_importance.csv     # Feature importance for temperature regression model
```

---

## Performance Comparison

### Standard Evaluation vs LOSO Evaluation

| Metric | Standard Evaluation | LOSO Evaluation | Difference |
|--------|---------------------|-----------------|------------|
| **Training data** | 70% of data | 94.4% of stations (17/18) | More training data |
| **Test data** | 15% of data | 5.6% of stations (1/18) | Less test data |
| **Performance** | Usually better | Usually slightly worse | More realistic |
| **Generalization** | Limited | Stronger | Spatial generalization |

### Performance Comparison Across Different Horizons

| Horizon | ROC-AUC | PR-AUC | MAE (°C) | RMSE (°C) | R² |
|---------|---------|--------|----------|-----------|-----|
| 3h | 0.9965 | 0.9543 | 1.15 | 1.45 | 0.9698 |
| 6h | 0.9928 | 0.9234 | 1.59 | 2.01 | 0.9458 |
| 12h | 0.9892 | 0.8765 | 1.84 | 2.45 | 0.9270 |
| 24h | 0.9827 | 0.8123 | 1.96 | 2.67 | 0.9171 |

**Trend**:
- Performance gradually decreases as horizon increases
- This is expected, as long-term prediction is more difficult

---

## Command Line Details

### Compound Command Example

```bash
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h \
    --top-k 12 2>&1 | grep -E "(Saved plot|Top.*Features)" | head -5
```

#### **Command Structure**

This is a **compound command** (Pipeline), using pipe `|` to connect multiple commands:

```
Command1 | Command2 | Command3
```

#### **Detailed Explanation by Part**

##### **1️⃣ Python CLI Command (Main Part)**

```bash
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h \
    --top-k 12 2>&1
```

- `python -m src.cli`: Run CLI as module
- `analysis feature-importance`: Feature importance analysis subcommand
- `--model-dir`: Specify model directory path
- `--top-k 12`: Only show top 12 most important features
- `2>&1`: Redirect standard error to standard output (allows `grep` to search all output)

##### **2️⃣ grep Filter (Middle Part)**

```bash
grep -E "(Saved plot|Top.*Features)"
```

- `grep`: Text search tool
- `-E`: Enable extended regular expressions
- `"(Saved plot|Top.*Features)"`: Search for lines containing "Saved plot" or "Top.*Features"

##### **3️⃣ head Limit Output (Final Part)**

```bash
head -5
```

- `head`: Display first N lines
- `-5`: Only show first 5 matching results

#### **Complete Execution Flow**

```
1. Python CLI command execution
   ↓ Output all logs (stdout + stderr)
   
2. grep filter
   ↓ Only keep lines containing "Saved plot" or "Top.*Features"
   
3. head limit
   ↓ Only show first 5 matching results
   
4. Terminal display
   ✅ Final output
```

---

## Frequently Asked Questions

### Q1: Why does training include all horizons even when only `--horizon-h 12` is specified?

**A**: This is a design decision for:
- Ensuring all labels are generated
- Supporting LOSO evaluation
- Avoiding label generation bugs

See [Training Command Details](#training-command-details) section for details.

### Q2: How to train only a single horizon?

**A**: Modify configuration file or code, see [How to Train Only a Single Horizon?](#how-to-train-only-a-single-horizon) section.

### Q3: How long does LOSO evaluation take?

**A**: Usually 3-6 hours, depending on data scale and model complexity.

### Q4: Where are training logs saved?

**A**: 
- Experiment level: `experiments/<model>/<track>/<cell>/<scope>/experiment.log`
- Horizon level: `experiments/<model>/<track>/<cell>/<scope>/horizon_<h>/training.log`

### Q5: How to view feature importance?

**A**: Use `analysis feature-importance` command, see [Command Line Details](#command-line-details) section.

---

## Related Documentation

- **[Feature Engineering Guide](../features/FEATURE_GUIDE.md)**: Complete feature engineering guide
- **[Feature Importance Guide](../features/FEATURE_IMPORTANCE.md)**: Feature importance analysis guide
- **[Models Guide](../models/MODELS_GUIDE.md)**: Detailed model descriptions
- **[Inference Guide](../inference/INFERENCE_GUIDE.md)**: Model inference guide

---

**Last Updated**: 2025-12-06  
**Document Version**: 3.0
