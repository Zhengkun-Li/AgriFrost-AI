<div align="center">

<img src="docs/logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="200"/>

# 🌡️ AgriFrost-AI

## F3 Innovate Frost Risk Forecasting Challenge (2025)

**AI-Powered Frost Risk Prediction System for California Agriculture**

</div>

---

**Author:** Zhengkun LI  
**Email:** zhengkun.li3969@gmail.com  
**Affiliation:** TRIC Robotics / UF ABE / F3 Innovate Participant  
**Last Updated:** 2025-12-06  
**Report Location:** [docs/manuscript](https://github.com/Zhengkun-Li/AgriFrost-AI/tree/main/docs/manuscript)

## 📋 Project Description

AgriFrost-AI is an end-to-end machine learning system for frost risk forecasting in California's Central Valley agriculture. The system processes hourly meteorological observations from 18 CIMIS (California Irrigation Management Information System) stations spanning 2010-2025 (~2.37 million records) to predict frost events and temperature drops across multiple forecast horizons (3h, 6h, 12h, 24h).

### Key Features & Experimental Scale

- **ABCD Feature Matrix Framework**: Systematic evaluation of spatial scope (single-station vs. multi-station) and feature complexity (raw vs. engineered features)
  - **4 Feature Matrices**: A (16 dim), B (278 dim), C (534 dim), D (818 dim)
  - **4 Forecast Horizons**: 3h, 6h, 12h, 24h
  - **Spatial Radius Range**: 0-200 km (20 km step)
- **471 Reproducible Experiments**: Comprehensive comparison across 7 model families (LightGBM, XGBoost, CatBoost, Random Forest, GRU, LSTM, TCN)
- **Best Performance**: Matrix C + LightGBM achieves ROC-AUC 0.9972 (3h) to 0.9877 (24h) with excellent spatial generalization (LOSO evaluation)
- **Production-Ready**: Complete pipeline from data processing to model deployment with strict temporal leakage protection and robust validation

### Experimental Results Summary

#### Best Performance (Matrix C + LightGBM)

| Horizon | Radius | ROC-AUC ↑ | PR-AUC ↑ | Brier ↓ | ECE ↓ | MAE ↓ | RMSE ↓ |
|---------|--------|-----------|----------|---------|-------|-------|--------|
| 3h      | 60 km  | 0.9972    | 0.7242   | 0.0027  | 0.0012| 1.16   | 1.58    |
| 6h      | 160 km | 0.9943    | 0.5871   | 0.0039  | 0.0021| 1.60   | 2.05    |
| 12h     | 200 km | 0.9901    | 0.4914   | 0.0043  | 0.0032| 1.85   | 2.42    |
| 24h     | 180 km | 0.9877    | 0.4671   | 0.0045  | 0.0034| 1.85   | 2.39    |

#### LOSO Spatial Generalization (Matrix C + LightGBM)

| Horizon | ROC-AUC (Standard) | ROC-AUC (LOSO) | Change | MAE (LOSO) |
|---------|---------------------|----------------|--------|------------|
| 3h      | 0.9965              | 0.9974         | +0.09  | 1.14       |
| 6h      | 0.9926              | 0.9938         | +0.12  | 1.55       |
| 12h     | 0.9892              | 0.9905         | +0.13  | 1.79       |
| 24h     | 0.9843              | 0.9878         | +0.35  | 1.93       |

#### Feature Selection Results (Matrix B, LightGBM, 12h Horizon)

- **90% Cumulative Importance**: 146 features (47.5% compression)
- **Performance**: ROC-AUC change < 0.01%, PR-AUC improvement +2.6%
- **Efficiency**: Training time reduced 35-40%, inference time reduced 30-35%

**Key Findings:**
- ✅ Excellent spatial generalization (LOSO ROC-AUC > 0.98 for all horizons, no performance degradation)
- ✅ Outstanding probability calibration (Brier Score < 0.005, ECE < 0.004)
- ✅ High-precision temperature prediction (MAE < 2°C, RMSE < 2.5°C)
- ✅ Effective feature selection (146 features maintain performance with 47.5% compression)

#### Complete Results Data

Complete experimental results are available in the Supplementary Materials:

- **[All Experiments](docs/manuscript/Supplementary/supplementary_table_S2_all_experiments.csv)**: Complete performance metrics for all 471 experiment configurations
- **[Best Configurations](docs/manuscript/Supplementary/supplementary_table_S3_best_configurations.csv)**: Optimal configurations for each feature matrix and horizon
- **[Matrix Summary](docs/manuscript/Supplementary/supplementary_table_S4_matrix_summary.csv)**: Statistical summary aggregated by matrix and horizon
- **[Feature Importance](docs/manuscript/Supplementary/supplementary_table_S5_feature_category_importance.csv)**: Cumulative importance of feature categories across horizons
- **[Top Features](docs/manuscript/Supplementary/supplementary_table_S6_top_features_by_category.csv)**: Importance of top features in each category

See [Supplementary Materials](docs/manuscript/Supplementary/) for complete documentation of all supplementary tables.

## 🚀 Quick Start

### ⭐ New Unified CLI (Recommended)

#### Step 1: Set Up Virtual Environment

```bash
# Create virtual environment (if not already created)
python3 -m venv .venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate.bat  (Command Prompt)
# .venv\Scripts\Activate.ps1   (PowerShell)

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **✅ Good News**: By default, `requirements.txt` does **NOT** include PyTorch/CUDA. You can install dependencies immediately without downloading ~2GB of PyTorch packages. PyTorch is only needed if you plan to use deep learning models (GRU, LSTM, TCN).

**Verify activation**: Command prompt should show `(.venv)` prefix

#### Step 2: Download Data

**Important**: This repository contains only code. Data files must be downloaded separately.

```bash
# Clone the data repository
git clone https://github.com/CarlSaganPhD/frost-risk-forecast-challenge.git data_repo

# Copy data to project directory
mkdir -p data/raw/frost-risk-forecast-challenge
cp -r data_repo/stations data/raw/frost-risk-forecast-challenge/
cp data_repo/cimis_all_stations.csv.gz data/raw/frost-risk-forecast-challenge/

# Or download manually from:
# https://github.com/CarlSaganPhD/frost-risk-forecast-challenge
```

The data includes:
- **18 CIMIS station files** (2010–2025, hourly observations)
- **Combined CSV** (`cimis_all_stations.csv.gz`, ~2.37M rows (2,367,360), 38 MB gzipped)

#### Step 3: Run Commands

> **⭐ Recommended**: Use the unified CLI (`python -m src.cli ...`) for training, evaluation, inference, and analysis.  
> **Note**: The `scripts/` directory contains additional tool scripts (e.g., `scripts/tools/fetch_station_metadata.py`) that can be used as needed.

**Quick Test (Recommended for first-time users):**
```bash
# Train with small dataset (100K samples) for quick testing
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --sample-size 100000 \
    --output-dir experiments/lightgbm_B_12h_test

# Generate predictions with trained model
# Note: Use the processed data file from training output, or any CSV with required columns
python -m src.cli inference predict \
    --model-dir experiments/lightgbm_B_12h_test \
    --input data/processed/labeled_data.parquet \
    --output predictions_test.csv \
    --horizon-h 12
```

**Full Training (Production):**
```bash
# Train with full dataset
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --output-dir experiments/lightgbm_B_12h

# Run matrix experiments (batch training) - Using CLI
python -m src.cli train matrix \
    --config config/pipeline/matrix_experiments.yaml

# Batch train multiple models (using script) - Alternative method
# Train all models (xgboost, catboost, random_forest) on all matrix cells
bash scripts/experiments/start_batch_training.sh

# Or customize which models to train
python scripts/experiments/batch_train_all_models.py \
    --models xgboost catboost \
    --matrix-cells A B C D \
    --skip-existing

# Evaluate single model
python -m src.cli evaluate model \
    --model-dir experiments/lightgbm_B_12h \
    --config config/evaluation.yaml

# Compare multiple models
python -m src.cli evaluate compare \
    --model-dirs experiments/model1 experiments/model2 \
    --output-dir comparison/

# Generate matrix summary
python -m src.cli evaluate matrix \
    --experiments-dir experiments/ \
    --output-dir matrix_summary/

# Generate predictions
python -m src.cli inference predict \
    --model-dir experiments/lightgbm_B_12h_test \
    --input data/test.csv \
    --output predictions.csv \
    --horizon-h 12

# Feature analysis
python -m src.cli analysis full \
    --data-path data/train.csv \
    --model-dir experiments/lightgbm_B_12h \
    --output-dir analysis/features

# Generate feature importance figures for manuscript
# Generate Matrix A feature importance figure (Figure 12)
python scripts/tools/generate_matrix_a_feature_importance.py

# Generate Matrix A supplementary table (Table S7)
python scripts/tools/generate_matrix_a_supplementary_table.py

# Generate feature category importance bar chart (Figure 13)
python scripts/tools/generate_feature_category_importance_bar.py
```


## 📚 Documentation

### Quick Links

- **🚀 [Quick Start Guide](docs/guides/QUICK_START.md)**: Get started in 15 minutes! (Recommended for new users)
- **📖 [User Guide](docs/guides/USER_GUIDE.md)**: Complete usage instructions
- **🏗️ [Implementation Guide](docs/guides/IMPLEMENTATION_GUIDE.md)**: System architecture and methodology
- **🤖 [Models Guide](docs/models/MODELS_GUIDE.md)**: Detailed model descriptions
- **📊 [Feature Guide](docs/features/FEATURE_GUIDE.md)**: Feature engineering guide
- **🔬 [Technical Documentation](docs/technical/TECHNICAL_DOCUMENTATION.md)**: Technical details

### CLI Documentation

- **[Unified CLI](scripts/README.md)**: Complete CLI documentation with all commands and detailed usage examples
  - **Main operations** (training, evaluation, inference, analysis): Use `python -m src.cli ...`
  - **Batch training scripts**: For training multiple models, see `scripts/experiments/`:
    - `batch_train_all_models.py` - Batch train multiple ML models (lightgbm, xgboost, catboost, random_forest)
    - `batch_train_deep_models.py` - Batch train deep learning models (GRU, LSTM, TCN)
    - `start_batch_training.sh` - Quick start script for batch training
  - **Tool scripts**: Additional utilities in `scripts/tools/` (e.g., `fetch_station_metadata.py`, `generate_station_map.py`)

## 🧾 Project Structure

```
frost-risk-forecast-challenge/
├── data/                    # Data directory (not in repo, download from GitHub)
│   └── raw/
│       └── frost-risk-forecast-challenge/
│           ├── stations/    # 18 CIMIS station CSV files (download required)
│           └── cimis_all_stations.csv.gz  # Combined data (download required)
│   └── (interim/, processed/, external/ created during processing)
├── src/                     # Source code (library code)
│   ├── data/                # Data processing pipeline
│   │   ├── loaders.py       # Data loading (CSV, Parquet, directories)
│   │   ├── cleaners.py      # Data cleaning (QC, outliers, imputation)
│   │   ├── features/        # Feature engineering modules
│   │   │   ├── temporal.py  # Time-based features
│   │   │   ├── lagging.py   # Lag and rolling window features
│   │   │   ├── derived.py   # Derived meteorological features
│   │   │   ├── station.py   # Station-level features
│   │   │   └── constants.py # Column name constants
│   │   ├── spatial/         # Spatial aggregation (C/D tracks)
│   │   ├── frost_labels.py  # Frost label generation
│   │   ├── preprocessors.py # Scaling, imputation
│   │   ├── feature_selection.py # Feature selection
│   │   └── pipeline.py      # Unified DataPipeline
│   ├── training/            # Training modules
│   │   ├── pipeline_runner.py # Training/Evaluation/Inference runners
│   │   ├── model_trainer.py   # Model training logic
│   │   ├── data_preparation.py # Data preparation utilities
│   │   └── loso_evaluator.py  # LOSO evaluation
│   ├── models/              # Model implementations
│   │   ├── base.py          # Base model interface
│   │   ├── registry.py      # Model registry
│   │   ├── ml/              # Machine learning models (LightGBM, XGBoost, etc.)
│   │   ├── deep/            # Deep learning models (LSTM, GRU, TCN)
│   │   └── graph/           # Graph neural networks (DCRNN, etc.)
│   ├── evaluation/          # Evaluation modules
│   │   ├── metrics.py       # Evaluation metrics (MAE, RMSE, ROC-AUC, ECE, etc.)
│   │   ├── validators.py    # Cross-validation strategies (time_split, LOSO, etc.)
│   │   ├── registry.py      # Evaluation strategy registry
│   │   ├── multi_horizon_evaluator.py  # Multi-horizon evaluation
│   │   ├── matrix_evaluator.py         # 2×2+1 matrix evaluation
│   │   └── spatial_sensitivity_evaluator.py  # Spatial parameter sensitivity
│   ├── cli/                 # ⭐ Unified CLI (Recommended)
│   │   ├── main.py          # CLI entry point
│   │   ├── common.py        # Common utilities
│   │   └── commands/        # CLI commands
│   │       ├── train.py      # Training commands
│   │       ├── evaluate.py   # Evaluation commands
│   │       ├── inference.py  # Inference commands
│   │       └── analysis.py   # Analysis commands
│   ├── visualization/       # Visualization utilities
│   │   └── plots.py         # Plotting functions (matplotlib, plotly)
│   └── utils/               # Utility functions
│       ├── calibration.py   # Probability calibration
│       ├── hyperopt.py       # Hyperparameter optimization
│       ├── losses.py         # Custom loss functions
│       └── path_utils.py    # Path utilities
├── scripts/                 # Scripts and tools
│   ├── README.md            # CLI usage guide
│   ├── MIGRATION.md         # Migration guide (from old scripts)
│   ├── tools/               # Independent tool scripts
│   └── test/                # Test scripts
├── config/                  # Configuration files
│   ├── data_cleaning*.yaml  # Data cleaning configurations
│   ├── feature_engineering/ # Feature engineering configs
│   └── pipeline/            # Pipeline configurations
├── experiments/             # Experiment results (created during training, not tracked in repo)
├── results/                 # Result summaries (not tracked in repo)
├── docs/                    # Documentation
│   ├── guides/              # User guides and tutorials
│   ├── technical/           # Technical documentation
│   ├── features/            # Feature engineering guides
│   ├── models/              # Model guides
│   ├── training/            # Training guides
│   ├── inference/           # Inference guides
│   └── manuscript/          # Academic manuscript
├── tests/                   # Test code
│   ├── data/                # Data module tests
│   ├── models/              # Model tests
│   └── evaluation/          # Evaluation tests
└── README.md                # This file
```

## 🔗 Links

- **Data Repository**: https://github.com/CarlSaganPhD/frost-risk-forecast-challenge
- **Manuscript**: [docs/manuscript/frost_risk_progress_cn.pdf](docs/manuscript/frost_risk_progress_cn.pdf)
- **Supplementary Materials**: [docs/manuscript/Supplementary/](docs/manuscript/Supplementary/)
- **Experiment Results**: Results are stored in `results/` directory (not tracked in repo). Run `python scripts/tools/update_results.py` to generate summaries.

## 📄 License

MIT License --- For research and educational use.

