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

**Verify activation**: Command prompt should show `(.venv)` prefix

#### Step 2: Run Commands

```bash
# Train single model
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --output-dir experiments/lightgbm_B_12h

# Run matrix experiments (batch training)
python -m src.cli train matrix \
    --config config/pipeline/matrix_experiments.yaml

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
    --model-dir experiments/lightgbm_B_12h \
    --input data/test.csv \
    --output predictions.csv

# Feature analysis
python -m src.cli analysis full \
    --data-path data/train.csv \
    --model-dir experiments/lightgbm_B_12h \
    --output-dir analysis/features
```


## 📥 Data Download

**Important**: This repository contains only code. Data files must be downloaded separately.

### Download Data

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

## 🧰 Environment Setup (CUDA 13.0, PyTorch cu130)

### Prerequisites
- **Python**: 3.12 (3.10–3.14 supported)
- **NVIDIA Driver**: r580+ (visible via `nvidia-smi`), optional system CUDA Toolkit 13.0 (recommended if installed)

### ⚠️ Important: Use Virtual Environment

**Strongly recommended to use a virtual environment** to avoid dependency conflicts and prevent polluting the system Python environment.

### Step 1: Create Virtual Environment

```bash
# Create virtual environment (recommended: use .venv)
python3 -m venv .venv

# Alternative: use 'venv' or 'env' as name
# python3 -m venv venv
# python3 -m venv env
```

### Step 2: Activate Virtual Environment

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

**Verify activation**: The command prompt should show `(.venv)` or `(venv)` prefix

### Step 3: Install Dependencies

```bash
# Upgrade pip (important for latest package compatibility)
python -m pip install -U pip

# Install project dependencies (includes cu130 extra-index for PyTorch)
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Verify GPU & capability
python - << 'PY'
import torch
print('torch=', torch.__version__, 'cuda=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device=', torch.cuda.get_device_name(0), 'cap=', torch.cuda.get_device_capability(0))
PY

# Verify other key packages
python -c "import pandas, numpy, lightgbm, xgboost; print('✅ All packages installed!')"
```

### Deactivate Virtual Environment

When you're done working:
```bash
deactivate
```

### Notes

**Notes:**
- The `requirements.txt` adds an extra index `https://download.pytorch.org/whl/cu130` and pins:
  - `torch==2.9.1+cu130`, `torchvision==0.24.1+cu130`, `torchaudio==2.9.1+cu130`
- For CPU-only installation, comment out the three PyTorch dependency lines in `requirements.txt`, then install separately:
  ```bash
  pip install torch==2.9.1
  ```
- If encountering unsupported GPU architectures (e.g., sm_120), switch to the cu130 combination above or compile PyTorch from source (set `TORCH_CUDA_ARCH_LIST="12.0"`).
- Training optimizations:
  - Both LSTM and LSTM-MT support AMP (mixed precision): enable with `model_params.use_amp: true` (LSTM already supports it, LSTM-MT is aligned and uses `GradScaler`).
  - Training scripts set `torch.set_float32_matmul_precision('high')` under CUDA to improve matmul performance and print GPU model and cuDNN version to logs at startup.

## 📊 Results Summary

### Experimental Scale

- **Total Experiments**: 471 reproducible experiments
- **Feature Matrices**: 4 (A, B, C, D)
- **Forecast Horizons**: 4 (3h, 6h, 12h, 24h)
- **Model Families**: 7 (LightGBM, XGBoost, CatBoost, Random Forest, GRU, LSTM, TCN)
- **Spatial Radius Range**: 0-200 km (20 km step)

### Best Performance (Matrix C + LightGBM)

| Horizon | Radius | ROC-AUC ↑ | PR-AUC ↑ | Brier ↓ | ECE ↓ | MAE ↓ | RMSE ↓ |
|---------|--------|-----------|----------|---------|-------|-------|--------|
| 3h      | 60 km  | 0.9972    | 0.7242   | 0.0027  | 0.0012| 1.16   | 1.58    |
| 6h      | 160 km | 0.9943    | 0.5871   | 0.0039  | 0.0021| 1.60   | 2.05    |
| 12h     | 200 km | 0.9901    | 0.4914   | 0.0043  | 0.0032| 1.85   | 2.42    |
| 24h     | 180 km | 0.9877    | 0.4671   | 0.0045  | 0.0034| 1.85   | 2.39    |

### LOSO Spatial Generalization (Matrix C + LightGBM)

| Horizon | ROC-AUC (Standard) | ROC-AUC (LOSO) | Change | MAE (LOSO) |
|---------|---------------------|----------------|--------|------------|
| 3h      | 0.9965              | 0.9974         | +0.09  | 1.14       |
| 6h      | 0.9926              | 0.9938         | +0.12  | 1.55       |
| 12h     | 0.9892              | 0.9905         | +0.13  | 1.79       |
| 24h     | 0.9843              | 0.9878         | +0.35  | 1.93       |

### Feature Selection Results (Matrix B, LightGBM, 12h Horizon)

- **90% Cumulative Importance**: 146 features (47.5% compression)
- **Performance**: ROC-AUC change < 0.01%, PR-AUC improvement +2.6%
- **Efficiency**: Training time reduced 35-40%, inference time reduced 30-35%

**Key Findings:**
- ✅ Excellent spatial generalization (LOSO ROC-AUC > 0.98 for all horizons, no performance degradation)
- ✅ Outstanding probability calibration (Brier Score < 0.005, ECE < 0.004)
- ✅ High-precision temperature prediction (MAE < 2°C, RMSE < 2.5°C)
- ✅ Effective feature selection (146 features maintain performance with 47.5% compression)

## 📚 Documentation

### Academic Manuscript

- **[Manuscript](docs/manuscript/frost_risk_progress_cn.pdf)**: Complete academic manuscript - methodology, results, and analysis (471 experiments, ABCD feature matrix framework)
- **[Supplementary Materials](docs/manuscript/Supplementary/)**: Detailed feature lists, station metadata, and additional analysis

### Main Documentation

- **[USER_GUIDE.md](docs/guides/USER_GUIDE.md)**: Complete user guide - setup, quick start, and advanced usage
- **[TECHNICAL_DOCUMENTATION.md](docs/technical/TECHNICAL_DOCUMENTATION.md)**: Technical documentation - architecture, API reference, configuration
- **[DATA_DOCUMENTATION.md](docs/technical/DATA_DOCUMENTATION.md)**: Data documentation - data overview, QC processing, variable usage
- **[FEATURE_GUIDE.md](docs/features/FEATURE_GUIDE.md)**: Complete feature engineering guide - 278 features (Matrix B), feature selection, and implementation
- **[MODELS_GUIDE.md](docs/models/MODELS_GUIDE.md)**: Comprehensive guide to all models - principles, advantages, disadvantages, and use cases
- **[TRAINING_GUIDE.md](docs/training/TRAINING_GUIDE.md)**: Training and evaluation - configuration, LOSO evaluation, performance comparison
- **[INFERENCE_GUIDE.md](docs/inference/INFERENCE_GUIDE.md)**: Inference guide - how to use trained models for prediction

### Key Improvements (2025)

**Data Module:**
- ✅ Strict temporal leakage protection for rolling/lagging features
- ✅ Spatial aggregation features with missing mask features (`neighbor_missing_count`, `feature_missing_mask`)
- ✅ Enhanced input validation and error handling

**Training Module:**
- ✅ Strict column validation (DATE_COL, STATION_ID_COL, feature columns)
- ✅ Temporal leakage protection in LOSO evaluation
- ✅ GPU memory management for multi-horizon training
- ✅ Improved track inference (RAW_CELLS vs FE_CELLS)

**Evaluation Module:**
- ✅ Multi-horizon evaluator for cross-horizon analysis
- ✅ Matrix evaluator for 2×2+1 framework comparison
- ✅ Spatial sensitivity evaluator for radius/k parameter optimization
- ✅ Multi-task model support (classification + regression structured metrics)
- ✅ Enhanced LOSO with temporal sorting and validation

### Module Documentation

- **[Data Module](src/data/README.md)**: Data processing pipeline - loaders, cleaners, feature engineering, labels, spatial aggregation
- **[Training Module](src/training/README.md)**: Training, evaluation, and inference runners with strict validation
- **[Models Module](src/models/README.md)**: Model interfaces and implementations (ML, deep learning, graph neural networks)
- **[Evaluation Module](src/evaluation/README.md)**: Metrics, cross-validation strategies, and advanced evaluators (multi-horizon, matrix, spatial sensitivity)
- **[Utils Module](src/utils/README.md)**: Utility functions (calibration, hyperopt, losses, path utilities)
- **[Visualization Module](src/visualization/README.md)**: Plotting utilities

### CLI Documentation

- **[Unified CLI](scripts/README.md)**: ⭐ **All commands** - Unified CLI interface
  - `python -m src.cli train single ...` - Train single model
  - `python -m src.cli train matrix ...` - Batch matrix experiments
  - `python -m src.cli evaluate model ...` - Evaluate model
  - `python -m src.cli evaluate compare ...` - Compare models
  - `python -m src.cli evaluate matrix ...` - Matrix summary
  - `python -m src.cli inference predict ...` - Generate predictions
  - `python -m src.cli analysis full ...` - Feature analysis

See [scripts/README.md](scripts/README.md) for detailed CLI usage and [scripts/MIGRATION.md](scripts/MIGRATION.md) for migration from old scripts.

### Reports

- **[MODELS_GUIDE.md](docs/models/MODELS_GUIDE.md)**: Comprehensive guide to all models (principles, advantages, disadvantages, use cases)

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
│   ├── visualization/       # Visualization utilities
│   │   └── plots.py         # Plotting functions (matplotlib, plotly)
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

## 🎯 Key Features

### Core Functionality

- **ABCD Feature Configuration Matrix**: Systematic framework for evaluating spatial scope (single-station vs. multi-station) and feature complexity (raw vs. engineered features)
  - **Matrix A**: Single-station + raw features (16 dimensions)
  - **Matrix B**: Single-station + engineered features (278 dimensions)
  - **Matrix C**: Multi-station aggregation + raw features (534 dimensions)
  - **Matrix D**: Multi-station aggregation + engineered features (818 dimensions)
- **278 Features (Matrix B)**: Raw variables (12) + engineered features: time-based (15), lagged (50), rolling statistics (180), derived meteorological (3), radiation (4), wind (6), humidity (4), trend (1), and station features (4)
- **Feature Selection**: Two-stage strategy based on cumulative importance (90% threshold = 146 features for 12h horizon, 47.5% compression)
- **Multi-Horizon Forecasting**: 3h, 6h, 12h, and 24h predictions
- **Probabilistic Outputs**: Calibrated frost probabilities with temperature predictions
- **Spatial Generalization**: LOSO evaluation across 18 CIMIS stations with temporal leakage protection
- **Model Comparison**: 7 model families (LightGBM, XGBoost, CatBoost, Random Forest, GRU, LSTM, TCN) across 471 experiments
- **Advanced Evaluation**: Multi-horizon, matrix, and spatial sensitivity evaluators
- **Multi-Task Models**: Classification + regression with structured metrics

### Code Quality & Architecture

- **Unified Data Pipeline**: Configurable `DataPipeline` for reproducible data processing with strict validation
- **Modular Feature Engineering**: Specialized modules (temporal, lagging, derived, station, spatial)
- **Temporal Leakage Protection**: Strict sorting and validation for rolling/lagging features
- **Spatial Aggregation**: Support for multi-station features (C/D/E tracks) with missing mask features
- **Configuration-Driven**: YAML-based configuration with CLI overrides
- **Standardized Logging**: Consistent logging across all modules using Python `logging`
- **Robust Error Handling**: Specific exception types with informative error messages
- **Input Validation**: Comprehensive parameter validation and boundary checks
- **GPU Memory Management**: Automatic GPU cache cleanup for multi-horizon training
- **Production-Ready**: Well-organized, maintainable, and tested codebase

## 📦 Status

| Item | Status | Location |
|------|--------|----------|
| Data acquisition | ✅ Complete | `data/raw/`, `data/external/` |
| Data pipeline | ✅ Complete | `src/data/pipeline.py`, unified `DataPipeline` with strict validation |
| Feature engineering | ✅ Complete | 278 features (Matrix B), 534 features (Matrix C), 818 features (Matrix D), modular structure, temporal leakage protection, `src/data/features/` |
| Spatial aggregation | ✅ Complete | Multi-station features (C/D/E tracks), missing mask features, `src/data/spatial/` |
| Model training | ✅ Complete | 471 experiments across 7 model families (LightGBM, XGBoost, CatBoost, Random Forest, GRU, LSTM, TCN), feature selection (Top-146 for 12h horizon) |
| LOSO evaluation | ✅ Complete | `src/training/loso_evaluator.py` with temporal leakage protection |
| Advanced evaluation | ✅ Complete | Multi-horizon, matrix, spatial sensitivity evaluators, `src/evaluation/` |
| Inference services | ✅ Complete | `python -m src.cli inference predict ...` |
| Code quality | ✅ Complete | Logging, error handling, input validation, GPU memory management |
| Configuration system | ✅ Complete | YAML-based configs with CLI overrides |
| Documentation | ✅ Complete | Module READMEs, API docs, user guides |

## 📚 Documentation

- **🚀 [Quick Start Guide](docs/guides/QUICK_START.md)**: Get started in 15 minutes! (Recommended for new users)
- **📖 [User Guide](docs/guides/USER_GUIDE.md)**: Complete usage instructions
- **🏗️ [Implementation Guide](docs/guides/IMPLEMENTATION_GUIDE.md)**: System architecture and methodology (English)
- **🏗️ [Implementation Guide (Chinese)](docs/guides/IMPLEMENTATION_GUIDE_CN.md)**: System architecture and methodology (中文/Chinese)
- **📓 [Jupyter Notebook Tutorial](notebooks/tutorial.ipynb)**: Interactive end-to-end tutorial
- **🤖 [Models Guide](docs/models/MODELS_GUIDE.md)**: Detailed model descriptions
- **📊 [Feature Guide](docs/features/FEATURE_GUIDE.md)**: Feature engineering guide
- **🔬 [Technical Documentation](docs/technical/TECHNICAL_DOCUMENTATION.md)**: Technical details

## 🔗 Links

- **Data Repository**: https://github.com/CarlSaganPhD/frost-risk-forecast-challenge
- **Manuscript**: [docs/manuscript/frost_risk_progress_cn.pdf](docs/manuscript/frost_risk_progress_cn.pdf)
- **Supplementary Materials**: [docs/manuscript/Supplementary/](docs/manuscript/Supplementary/)
- **Experiment Results**: Results are stored in `results/` directory (not tracked in repo). Run `python scripts/tools/update_results.py` to generate summaries.

## 📄 License

MIT License --- For research and educational use.

