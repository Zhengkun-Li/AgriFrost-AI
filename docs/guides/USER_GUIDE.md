# AgriFrost-AI User Guide

<div align="center">

<img src="logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

</div>

**Last Updated**: 2025-12-06

This guide covers everything from environment setup, quick start to advanced usage.

## 📋 Table of Contents

1. [Environment Setup](#environment-setup)
2. [Quick Start](#quick-start)
3. [Data Preparation and Loading](#data-preparation-and-loading)
4. [Complete Workflow Guide](#complete-workflow-guide)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Model Inference](#model-inference)
8. [Result Interpretation](#result-interpretation)
9. [Frequently Asked Questions](#frequently-asked-questions)

---

## Environment Setup

### ⚠️ Important: Use Virtual Environment

**Strongly recommended to use a virtual environment** to install project dependencies, because:
- ✅ **Dependency isolation**: Avoid conflicts with system Python or other projects
- ✅ **Version consistency**: Ensure team members use the same dependency versions
- ✅ **Easy management**: Easy to delete and recreate environment
- ✅ **Avoid pollution**: Won't affect system Python environment

### Quick Setup (Recommended)

#### Step 1: Create Virtual Environment

```bash
# Create virtual environment (recommended: use .venv)
python3 -m venv .venv

# Alternative: use other names
# python3 -m venv venv
# python3 -m venv env
```

#### Step 2: Activate Virtual Environment

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
# PowerShell
.venv\Scripts\Activate.ps1

# Command Prompt
.venv\Scripts\activate.bat
```

**Verify activation:**
- Command prompt should show `(.venv)` or `(venv)` prefix
- Run `which python` (Linux/macOS) or `where python` (Windows) should show virtual environment path

#### Step 3: Install Dependencies

```bash
# Upgrade pip (important: ensure latest version to support latest packages)
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

#### Step 4: Verify Installation

```bash
# Verify key dependencies
python3 -c "import pandas, numpy, lightgbm, xgboost; print('✅ All packages installed')"

# Verify CLI is available
python -m src.cli --help
```

#### Deactivate Virtual Environment

When you're done working, you can deactivate the virtual environment:
```bash
deactivate
```

### Frequently Asked Questions

**Q: How do I know the virtual environment is activated?**
- A: The command prompt will show `(.venv)` or `(venv)` prefix

**Q: Do I need to activate it every time?**
- A: Yes, you need to reactivate the virtual environment each time you open a new terminal window

**Q: Can I delete the virtual environment?**
- A: Yes, simply delete the `.venv` directory, then recreate it

**Q: How much space does the virtual environment take?**
- A: Approximately 1-2GB, including all Python packages and dependencies

---

## Quick Start

### Simplest Usage

Use the unified CLI interface for training:

```bash
# Activate virtual environment
source .venv/bin/activate

# Train single model (LightGBM, Matrix Cell B, Top 175 Features, 12-hour prediction)
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --output-dir experiments/lightgbm_B_12h
```

This automatically completes:
1. Load data from data directory
2. Clean data and build features
3. Train model (classification + regression)
4. Evaluate on train/val/test
5. Save all results to output directory

### Batch Training (Matrix Experiments)

```bash
# Use configuration file to batch train multiple matrix cells
python -m src.cli train matrix \
    --config config/pipeline/matrix_experiments.yaml
```

### Using Different Models

```bash
# XGBoost
python -m src.cli train single \
    --model-name xgboost \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --output-dir experiments/xgboost_B_12h

# LSTM (requires GPU)
python -m src.cli train single \
    --model-name lstm \
    --matrix-cell A \
    --track raw \
    --horizon-h 12 \
    --output-dir experiments/lstm_A_12h \
    --config config/pipeline/lstm_config.yaml
```

---

## Data Preparation and Loading

### Data Location

Data is located at: `data/raw/frost-risk-forecast-challenge/stations/`

Contains 18 station CSV files (each ~14-15MB), system will automatically load and combine them.

### Automatic Loading (Recommended)

System automatically detects data in the following order:
1. `stations/` directory (preferred, 18 station files)
2. `cimis_all_stations.csv.gz` (alternative)
3. `cimis_all_stations.csv` (last resort)

**Currently using**: `stations/` directory automatic loading method

### Data Loading Process

```
Loading 18 station files from stations/...
  Loaded 18/18 files...
Combining 18 station DataFrames...
Combined data: 2367360 rows, 26 columns
Stations: 18
```

### Performance Notes

- **Loading time**: 18 files take approximately 10-30 seconds
- **Memory usage**: After combining ~2.36 million rows, memory usage ~500MB-1GB
- **File size**: Each station file ~14-15MB, total ~254MB

---

## Complete Workflow Guide

### Data Flow Diagram

```
Raw Data (CSV)
    ↓
[Data Loading] → DataFrame (2.36M rows, 26 columns)
    ↓
[QC Filtering] → Low-quality data marked as NaN
    ↓
[Sentinel Value Handling] → -6999, -9999 → NaN
    ↓
[Missing Value Imputation] → Forward fill
    ↓
[Feature Engineering] → DataFrame (2.36M rows, 300+ columns)
    ├─ Time features (hour, month, season, ...)
    ├─ Lag features (lag_1, lag_3, lag_6, ...)
    ├─ Rolling features (rolling_6h_mean, ...)
    ├─ Radiation features (Sol Rad related)
    ├─ Wind direction features (Wind Dir periodic encoding)
    └─ Derived features (temp_dew_diff, ...)
    ↓
[Label Generation] → DataFrame (2.36M rows, 300+ feature columns + 8 label columns)
    ├─ frost_3h, frost_6h, frost_12h, frost_24h
    └─ temp_3h, temp_6h, temp_12h, temp_24h
    ↓
[Data Splitting] → Train (70%) / Val (15%) / Test (15%)
    ↓
[Model Training] → Train 2 models for each time window
    ├─ Classification model (frost probability)
    └─ Regression model (temperature)
    ↓
[Model Evaluation] → Calculate all metrics
    ↓
[Model Saving] → Model files and metadata
```

### Key Steps Explanation

#### 1. Data Cleaning

System automatically executes the following cleaning steps:
- **QC Filtering**: Filter low-quality data based on QC flags (keep blank and `Y`, mark `M/R/S/Q/P` as NaN)
- **Sentinel Value Handling**: Replace sentinel values like `-6999`, `-9999` with `NaN`
- **Missing Value Imputation**: Use forward fill (grouped by station)

#### 2. Feature Engineering

System automatically creates the following features:
- **Time features**: hour, day_of_year, month, season, periodic encoding
- **Lag features**: Values from 1h, 3h, 6h, 12h, 24h ago
- **Rolling statistics**: mean, min, max, std for 6h, 12h, 24h windows
- **Radiation features**: Daily cumulative radiation, radiation change rate, nighttime cooling rate
- **Wind direction features**: Periodic encoding, categorical encoding
- **Derived features**: Temperature difference, wind chill index, heat index, etc.

For detailed explanation, see [Feature Engineering Guide](../features/FEATURE_GUIDE.md).

#### 3. Label Generation

For each prediction time window (3h, 6h, 12h, 24h), create:
- **Frost label** (`frost_{h}h`): Whether future temperature < 0°C (binary classification)
- **Temperature label** (`temp_{h}h`): Future temperature value (regression)

---

## Model Training

### Training with CLI (Recommended)

#### Single Model Training

```bash
# Basic training
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --output-dir experiments/lightgbm_B_12h

# Using configuration file
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --config config/pipeline/train.yaml
```

#### Batch Training (Matrix Experiments)

```bash
# Use configuration file to batch train multiple matrix cells
python -m src.cli train matrix \
    --config config/pipeline/matrix_experiments.yaml
```

### Training Parameters Explanation

#### CLI Parameters

- `--model-name`: Model type (lightgbm, xgboost, lstm, gru, tcn, etc.)
- `--matrix-cell`: Matrix cell (A/B/C/D/E)
- `--track`: Feature track (raw, top175_features, etc.)
- `--horizon-h`: Prediction time window (3, 6, 12, 24 hours)
- `--radius-km`: Spatial radius (required for C/D tracks)
- `--knn-k`: KNN k parameter (required for E track)
- `--config`: Configuration file path (YAML)
- `--output-dir`: Output directory
- `--data-path`: Input data path (optional)

#### Configuration Files

You can set detailed training parameters via YAML configuration files:

```yaml
data:
  source: "data/raw/frost-risk-forecast-challenge/stations/"
  matrix_cell: "B"
  
training:
  model: "lightgbm"
  horizons: [3, 6, 12, 24]
  
model_params:
  lightgbm:
    n_estimators: 200
    learning_rate: 0.05
    max_depth: 8
    num_leaves: 63
```

### Training Output

Two models are trained for each time window:
1. **Classification model** (`frost_classifier`): Predicts frost probability
2. **Regression model** (`temp_regressor`): Predicts future temperature

Training results are saved in the output directory:

```
experiments/lightgbm_B_12h/
├── horizon_12h/
│   ├── frost_classifier/
│   │   ├── model.pkl
│   │   ├── train_metrics.json
│   │   ├── val_metrics.json
│   │   ├── test_metrics.json
│   │   └── feature_importance.csv
│   ├── temp_regressor/
│   │   ├── model.pkl
│   │   ├── train_metrics.json
│   │   ├── val_metrics.json
│   │   └── test_metrics.json
│   └── run_metadata.json
```

---

## Model Evaluation

### Evaluate Single Model

```bash
# Evaluate trained model
python -m src.cli evaluate model \
    --model-dir experiments/lightgbm_B_12h/horizon_12h \
    --config config/evaluation.yaml \
    --output-dir evaluation_results/
```

### Compare Multiple Models

```bash
# Compare two or more models
python -m src.cli evaluate compare \
    --model-dirs experiments/model1 experiments/model2 \
    --output-dir comparison/
```

### Generate Matrix Summary

```bash
# Generate matrix summary for all experiments
python -m src.cli evaluate matrix \
    --experiments-dir experiments/ \
    --output-dir matrix_summary/
```

### Evaluation Metrics

Each model is automatically evaluated on train/val/test three datasets:

**Regression Metrics**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

**Classification Metrics**:
- Brier Score
- ROC-AUC
- PR-AUC (Precision-Recall AUC)
- ECE (Expected Calibration Error)

For detailed explanation, see [Training and Evaluation Documentation](../training/TRAINING_GUIDE.md).

---

## Model Inference

### Generate Predictions

```bash
# Use trained model to generate predictions
python -m src.cli inference predict \
    --model-dir experiments/lightgbm_B_12h/horizon_12h \
    --input data/test.csv \
    --output predictions.csv \
    --horizon-h 12
```

### Multi-Horizon Predictions

```bash
# Generate predictions for multiple time windows
python -m src.cli inference predict \
    --model-dir experiments/lightgbm_B_12h \
    --input data/test.csv \
    --output predictions.csv \
    --horizon-h 3 --horizon-h 6 --horizon-h 12 --horizon-h 24
```

For detailed explanation, see [Inference Guide](../inference/INFERENCE_GUIDE.md).

---

## Result Interpretation

### Result Organization Structure

```
experiments/lightgbm_B_12h/
├── horizon_12h/
│   ├── frost_classifier/
│   │   ├── model.pkl
│   │   ├── train_metrics.json
│   │   ├── val_metrics.json
│   │   ├── test_metrics.json
│   │   └── feature_importance.csv
│   ├── temp_regressor/
│   │   ├── model.pkl
│   │   ├── train_metrics.json
│   │   ├── val_metrics.json
│   │   └── test_metrics.json
│   └── run_metadata.json
```

### Key Metrics Location

**Key metrics for single model**:
- **Test MAE**: `test_metrics.json` → `mae` (lower is better)
- **Test R²**: `test_metrics.json` → `r2` (closer to 1 is better)
- **Test ROC-AUC**: `test_metrics.json` → `roc_auc` (classification model, closer to 1 is better)

### Result Quality Assessment

- **MAE < 1°C**: High prediction accuracy
- **R² > 0.9**: Good model fit
- **Small Train vs Test difference**: No overfitting
- **ROC-AUC > 0.95**: Excellent classification performance

---

## Frequently Asked Questions

### Q: How to view model results?

```bash
# View test metrics
cat experiments/lightgbm_B_12h/horizon_12h/frost_classifier/test_metrics.json

# View experiment metadata
cat experiments/lightgbm_B_12h/horizon_12h/run_metadata.json
```

### Q: How to get help?

```bash
# View all commands
python -m src.cli --help

# View training command help
python -m src.cli train --help

# View single training command help
python -m src.cli train single --help
```

### Q: How to customize model parameters?

Edit configuration files (YAML) or use configuration options in CLI. Example configuration files are in `config/pipeline/` directory.

### Q: Are results automatically saved?

**Yes!** All results are automatically saved:
- Model files (.pkl)
- Evaluation metrics (JSON)
- Experiment metadata (run_metadata.json)
- Feature importance (CSV, if supported)

### Q: How to switch between different matrix cells?

Use `--matrix-cell` parameter:
- `A`: Raw features, Single-station
- `B`: Feature-engineered, Single-station
- `C`: Raw features, Multi-station
- `D`: Feature-engineered, Multi-station
- `E`: Graph neural networks

---

## 📚 Related Documentation

- **[TECHNICAL_DOCUMENTATION.md](../technical/TECHNICAL_DOCUMENTATION.md)**: Technical documentation and API reference
- **[DATA_DOCUMENTATION.md](../technical/DATA_DOCUMENTATION.md)**: Data description and QC processing
- **[FEATURE_GUIDE.md](../features/FEATURE_GUIDE.md)**: Complete feature engineering guide
- **[TRAINING_GUIDE.md](../training/TRAINING_GUIDE.md)**: Detailed training and evaluation guide
- **[INFERENCE_GUIDE.md](../inference/INFERENCE_GUIDE.md)**: Inference usage guide

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-06  
**Author**: Zhengkun LI (TRIC Robotics / UF ABE)
