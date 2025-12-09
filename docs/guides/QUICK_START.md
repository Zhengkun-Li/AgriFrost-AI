# AgriFrost-AI: Quick Start Guide

<div align="center">

<img src="../logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

## 🌡️ AgriFrost-AI Quick Start

**AI-Powered Frost Risk Prediction System for California Agriculture**

*Get your first frost prediction model running from scratch in 15 minutes*

</div>

---

## 📋 Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Download](#2-data-download)
3. [First Model Training](#3-first-model-training)
4. [Model Evaluation](#4-model-evaluation)
5. [Generate Predictions](#5-generate-predictions)
6. [Next Steps](#6-next-steps)

---

## 1. Environment Setup

### 1.1 System Requirements

- **Python**: 3.10 - 3.14 (3.12 recommended)
- **Operating System**: Linux, macOS, Windows
- **GPU** (optional): NVIDIA GPU with CUDA 13.0+ (for deep learning models)
- **Memory**: 16GB+ RAM recommended
- **Storage**: At least 10GB free space

### 1.2 Installation Steps

#### Step 1: Clone the Repository

```bash
# Clone the repository (data can be downloaded later if needed)
git clone <your-repo-url>
cd frost-risk-forecast-challenge
```

#### Step 2: Create and Activate Virtual Environment

**⚠️ Important: Strongly recommended to use a virtual environment!**

Virtual environments provide:
- ✅ **Dependency isolation**: Avoid conflicts with system Python
- ✅ **Version consistency**: Ensure consistent dependency versions across team members
- ✅ **Easy management**: Easy to delete and recreate

**Create virtual environment:**

```bash
# Create virtual environment (recommended: use .venv)
python3 -m venv .venv

# Alternative: use other names
# python3 -m venv venv
# python3 -m venv env
```

**Activate virtual environment:**

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

**Deactivate virtual environment:**
```bash
deactivate
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

**Note:**
- For **CPU-only** installation (no GPU), modify `requirements.txt` to install CPU version of PyTorch:
  ```bash
  # Comment out CUDA version of PyTorch, install CPU version
  pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu
  ```

#### Step 4: Verify Installation

```bash
# Check if CLI is available
python -m src.cli --help

# Check key dependencies
python -c "import lightgbm, xgboost, torch, pandas; print('✅ All dependencies installed!')"
```

---

## 2. Data Download

### 2.1 Data Source

Data from **F3 Innovate Frost Risk Forecasting Challenge** official repository:
- **Repository**: https://github.com/CarlSaganPhD/frost-risk-forecast-challenge
- **Data Format**: CSV files (gzipped)
- **Size**: ~38 MB (compressed), ~200 MB (uncompressed)
- **Time Range**: 2010-09-28 to 2025-09-28
- **Stations**: 18 CIMIS weather stations

### 2.2 Download Methods

#### Method 1: Git Clone (Recommended)

```bash
# Create data directory
mkdir -p data/raw/frost-risk-forecast-challenge

# Clone data repository
git clone https://github.com/CarlSaganPhD/frost-risk-forecast-challenge.git data_repo_temp

# Copy data files
cp -r data_repo_temp/stations data/raw/frost-risk-forecast-challenge/
cp data_repo_temp/cimis_all_stations.csv.gz data/raw/frost-risk-forecast-challenge/

# Clean up temporary directory
rm -rf data_repo_temp

# Verify data
ls -lh data/raw/frost-risk-forecast-challenge/
# Should see:
# - stations/ (contains 18 CSV files)
# - cimis_all_stations.csv.gz
```

#### Method 2: Manual Download

1. Visit: https://github.com/CarlSaganPhD/frost-risk-forecast-challenge
2. Download `cimis_all_stations.csv.gz` file
3. Download `stations/` directory (or all CSV files within)
4. Place in `data/raw/frost-risk-forecast-challenge/` directory

#### Method 3: Using Python Script (if API available)

```bash
# If download script exists (adjust according to actual situation)
python scripts/tools/download_data.py
```

### 2.3 Verify Data

```bash
# Check data files
python -c "
from pathlib import Path
data_dir = Path('data/raw/frost-risk-forecast-challenge')
print(f'📁 Data directory: {data_dir}')
print(f'📊 Combined file: {data_dir / \"cimis_all_stations.csv.gz\"} exists: {(data_dir / \"cimis_all_stations.csv.gz\").exists()}')
print(f'📁 Stations directory: {data_dir / \"stations\"} exists: {(data_dir / \"stations\").exists()}')
if (data_dir / 'stations').exists():
    station_files = list((data_dir / 'stations').glob('*.csv'))
    print(f'📈 Number of station files: {len(station_files)}')
"
```

**Expected output:**
```
📁 Data directory: data/raw/frost-risk-forecast-challenge
📊 Combined file: exists: True
📁 Stations directory: exists: True
📈 Number of station files: 18
```

---

## 3. First Model Training

### 3.1 Simplest Training Command

Let's train a **LightGBM** model using **Top 175 features** to predict frost risk **12 hours** ahead:

```bash
# Activate virtual environment (if not already activated)
source .venv/bin/activate

# Train single model
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --output-dir experiments/my_first_model_12h
```

**Parameter explanation:**
- `--model-name lightgbm`: Use LightGBM model (fast and accurate)
- `--matrix-cell B`: Use feature engineering + single station (Matrix Cell B)
- `--track top175_features`: Use Top 175 selected features (best performance)
- `--horizon-h 12`: Predict frost 12 hours ahead
- `--output-dir`: Model save directory

**Estimated time:**
- **Data loading and preprocessing**: ~2-5 minutes
- **Feature engineering**: ~10-30 minutes
- **Model training**: ~5-10 minutes
- **Total**: ~20-45 minutes (depending on hardware)

### 3.2 Training Process Overview

The training process automatically executes the following steps:

1. **Data loading**: Load raw data from `data/raw/`
2. **Data cleaning**: QC filtering, outlier handling, missing value imputation
3. **Feature engineering**: Generate 175 selected features
4. **Label generation**: Generate frost labels for 12h horizon
5. **Data splitting**: 70% training, 15% validation, 15% test
6. **Model training**: 
   - Classification model (frost probability)
   - Regression model (temperature prediction)
7. **Model saving**: Save to `experiments/my_first_model_12h/horizon_12h/`

### 3.3 View Training Results

After training completes, check the output directory:

```bash
# View model files
ls -lh experiments/my_first_model_12h/horizon_12h/

# Should see:
# - frost_model.pkl (classification model)
# - temp_model.pkl (regression model)
# - run_metadata.json (experiment metadata)
# - train_metrics.json (training metrics)
# - validation_metrics.json (validation metrics)
# - test_metrics.json (test metrics)
```

**View training metrics:**

```bash
# View test set performance
cat experiments/my_first_model_12h/horizon_12h/test_metrics.json

# Or use Python
python -c "
import json
from pathlib import Path
metrics = json.load(open('experiments/my_first_model_12h/horizon_12h/test_metrics.json'))
print('📊 Test Set Performance:')
print(f'  ROC-AUC (Classification): {metrics[\"classification\"][\"roc_auc\"]:.4f}')
print(f'  Brier Score (Calibration): {metrics[\"classification\"][\"brier_score\"]:.4f}')
print(f'  MAE (Regression): {metrics[\"regression\"][\"mae\"]:.4f}°C')
print(f'  R² (Regression): {metrics[\"regression\"][\"r2\"]:.4f}')
"
```

**Expected performance** (LightGBM + Top 175 features, 12h):
- ROC-AUC: > 0.98
- Brier Score: < 0.01
- MAE: < 2°C
- R²: > 0.91

---

## 4. Model Evaluation

### 4.1 Standard Evaluation

Evaluate the model you just trained:

```bash
# Evaluate single model
python -m src.cli evaluate model \
    --model-dir experiments/my_first_model_12h \
    --config config/evaluation.yaml
```

This generates a detailed evaluation report including:
- Classification metrics (ROC-AUC, PR-AUC, Brier Score, ECE)
- Regression metrics (MAE, RMSE, R²)
- Calibration curves and reliability diagrams

### 4.2 LOSO Evaluation (Spatial Generalization)

To test model generalization across different stations, run LOSO (Leave-One-Station-Out) evaluation:

```bash
# LOSO evaluation (takes longer)
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --loso \
    --output-dir experiments/my_first_model_12h_loso
```

**Note:**
- LOSO evaluation requires training 18 models (one per station)
- Estimated time: **3-6 hours** (depending on hardware)
- Uses simplified model configuration (faster but slightly lower performance)

### 4.3 Multi-Horizon Evaluation

Train all horizons (3h, 6h, 12h, 24h):

```bash
# Train matrix experiments (all horizons)
python -m src.cli train matrix \
    --config config/pipeline/matrix_experiments.yaml
```

Or train individually:

```bash
for horizon in 3 6 12 24; do
    python -m src.cli train single \
        --model-name lightgbm \
        --matrix-cell B \
        --track top175_features \
        --horizon-h $horizon \
        --output-dir experiments/lightgbm_B_${horizon}h
done
```

---

## 5. Generate Predictions

### 5.1 Prepare Prediction Data

Prediction data should have the same format as training data. Example:

```bash
# Create test data directory (if not exists)
mkdir -p data/test

# Use part of historical data as test data
python -c "
import pandas as pd
from pathlib import Path

# Load data
data_path = Path('data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz')
df = pd.read_csv(data_path)

# Take last 1000 rows as test data
test_df = df.tail(1000)

# Save test data
test_df.to_csv('data/test/prediction_input.csv', index=False)
print(f'✅ Test data saved: {len(test_df)} rows')
"
```

### 5.2 Generate Predictions

```bash
# Use trained model to generate predictions
python -m src.cli inference predict \
    --model-dir experiments/my_first_model_12h \
    --input data/test/prediction_input.csv \
    --output predictions.csv
```

**Output format:**
```csv
Date,Stn Id,Frost Probability,Temperature Prediction
2025-09-28 12:00:00,2,0.0234,8.5
2025-09-28 12:00:00,7,0.0156,9.2
...
```

### 5.3 View Prediction Results

```bash
# View first few prediction rows
head -20 predictions.csv

# Analyze predictions using Python
python -c "
import pandas as pd
df = pd.read_csv('predictions.csv')
print('📊 Prediction Statistics:')
print(f'  Total predictions: {len(df)}')
print(f'  Average frost probability: {df[\"Frost Probability\"].mean():.4f}')
print(f'  High-risk predictions (>0.5): {(df[\"Frost Probability\"] > 0.5).sum()}')
print(f'  Average temperature prediction: {df[\"Temperature Prediction\"].mean():.2f}°C')
"
```

---

## 6. Next Steps

### 6.1 Explore More Features

1. **Try different models**:
   ```bash
   # XGBoost
   python -m src.cli train single --model-name xgboost --matrix-cell B --track top175_features --horizon-h 12 --output-dir experiments/xgboost_B_12h
   
   # LSTM (requires GPU)
   python -m src.cli train single --model-name lstm --matrix-cell B --track top175_features --horizon-h 12 --output-dir experiments/lstm_B_12h
   ```

2. **Try different matrix cells**:
   ```bash
   # Matrix Cell C (multi-station, raw features)
   python -m src.cli train single --model-name lightgbm --matrix-cell C --track raw_features --horizon-h 12 --output-dir experiments/lightgbm_C_12h
   
   # Matrix Cell D (multi-station, engineered features)
   python -m src.cli train single --model-name lightgbm --matrix-cell D --track top175_features --horizon-h 12 --output-dir experiments/lightgbm_D_12h
   ```

3. **Feature analysis**:
   ```bash
   # Full feature analysis
   python -m src.cli analysis full \
       --data-path data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz \
       --model-dir experiments/my_first_model_12h \
       --output-dir analysis/features
   ```

### 6.2 Deep Learning

- 📖 **User Guide**: `docs/guides/USER_GUIDE.md` - Complete usage instructions
- 🏗️ **Implementation Guide**: `docs/guides/IMPLEMENTATION_GUIDE.md` - System architecture and methodology
- 🔬 **Technical Documentation**: `docs/technical/TECHNICAL_DOCUMENTATION.md` - Technical details
- 🤖 **Models Guide**: `docs/models/MODELS_GUIDE.md` - Detailed descriptions of all models
- 📊 **Feature Guide**: `docs/features/FEATURE_GUIDE.md` - Feature engineering details

### 6.3 Command Quick Reference

```bash
# ===== Training =====
# Single model training
python -m src.cli train single --model-name lightgbm --matrix-cell B --track top175_features --horizon-h 12 --output-dir experiments/model

# Matrix batch training
python -m src.cli train matrix --config config/pipeline/matrix_experiments.yaml

# LOSO training
python -m src.cli train single --loso --output-dir experiments/loso_model

# ===== Evaluation =====
# Single model evaluation
python -m src.cli evaluate model --model-dir experiments/model

# Model comparison
python -m src.cli evaluate compare --model-dirs experiments/model1 experiments/model2 --output-dir comparison/

# Matrix summary
python -m src.cli evaluate matrix --experiments-dir experiments/ --output-dir matrix_summary/

# ===== Inference =====
# Generate predictions
python -m src.cli inference predict --model-dir experiments/model --input data/test.csv --output predictions.csv

# ===== Analysis =====
# Feature analysis
python -m src.cli analysis full --data-path data/train.csv --model-dir experiments/model --output-dir analysis/

# ===== Tools =====
# Generate station distribution map
python scripts/tools/generate_station_map.py

# Fetch station metadata
python scripts/tools/fetch_station_metadata.py
```

### 6.4 Troubleshooting

#### Issue 1: Data Not Found

```
FileNotFoundError: Data file not found: data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz
```

**Solution:**
1. Confirm data is downloaded (see [Data Download](#2-data-download))
2. Check path is correct: `ls -lh data/raw/frost-risk-forecast-challenge/`

#### Issue 2: Out of Memory

```
MemoryError: Unable to allocate array
```

**Solution:**
1. Reduce data size: Use `--sample-size` parameter
   ```bash
   python -m src.cli train single --model-name lightgbm --matrix-cell B --track top175_features --horizon-h 12 --output-dir experiments/model --sample-size 100000
   ```
2. Use Top 175 features (instead of full 298 features)
3. Increase system memory or use machine with more RAM

#### Issue 3: GPU Not Available (Deep Learning Models)

```
RuntimeError: CUDA error: no kernel image is available for execution
```

**Solution:**
1. Check CUDA version: `nvidia-smi`
2. Confirm PyTorch version matches: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
3. Use CPU version or reinstall correct PyTorch version
4. For deep learning models, consider using LightGBM/XGBoost (no GPU required)

#### Issue 4: Training Takes Too Long

**Solution:**
1. Use LightGBM (fastest)
2. Reduce `n_estimators` (number of trees)
3. Use Top 175 features (instead of full feature set)
4. Reduce data size (for quick testing)

---

## 📞 Get Help

- 📖 **Complete Documentation**: Check detailed documentation in `docs/` directory
- 🐛 **Issue Reports**: Report issues in GitHub Issues
- 💬 **Discussions**: Ask questions in GitHub Discussions

---

**Congratulations!** 🎉 You've completed the AgriFrost-AI quick start! Now you can:
- Train more models for experiments
- Explore different configurations and parameters
- Read detailed documentation for deeper learning
- Start your frost prediction research!

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-06  
**Author**: Zhengkun LI (TRIC Robotics / UF ABE)
