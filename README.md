# 🌡️ F3 Innovate --- Frost Risk Forecasting Challenge (2025)

**Author:** Zhengkun LI  
**Email:** zhengkun.li3969@gmail.com  
**Affiliation:** TRIC Robotics / UF ABE / F3 Innovate Participant  
**Platform:** National Data Platform (NDP)  
**Last Updated:** 2025-11-16

## 🚀 Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Train LightGBM model with Top 175 features
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --output experiments/lightgbm/top175_features/lightgbm \
    --top-k-features 175

# Run LOSO evaluation
python3 scripts/train/train_frost_forecast.py \
    --horizons 3 6 12 24 \
    --model lightgbm \
    --loso \
    --save-loso-models \
    --output experiments/lightgbm/top175_features/lightgbm \
    --top-k-features 175

# Analyze feature importance
python3 scripts/analysis/analyze_feature_importance.py \
    --model-dir experiments/lightgbm/top175_features/full_training/horizon_3h \
    --model-type lightgbm \
    --task both

# Generate comprehensive feature report
python3 scripts/analysis/generate_feature_report.py \
    --data data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz \
    --model-dir experiments/lightgbm/top175_features \
    --output scripts/analysis/output
```

## 🧰 Environment Setup (CUDA 13.0, PyTorch cu130)

Prerequisites:
- Python 3.12 (3.10–3.14 supported)
- NVIDIA Driver r580+（`nvidia-smi` 可见）, 可选装系统 CUDA Toolkit 13.0（已安装更佳）

Setup:
```bash
# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install -U pip

# Install project deps (requirements includes cu130 extra-index for PyTorch)
pip install -r requirements.txt

# Verify GPU & capability
python - << 'PY'
import torch
print('torch=', torch.__version__, 'cuda=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device=', torch.cuda.get_device_name(0), 'cap=', torch.cuda.get_device_capability(0))
PY
```

Notes:
- The `requirements.txt` adds an extra index `https://download.pytorch.org/whl/cu130` and pins:
  - `torch==2.9.1+cu130`, `torchvision==0.24.1+cu130`, `torchaudio==2.9.1+cu130`
- 若仅需 CPU，可将 `requirements.txt` 中三行 PyTorch 依赖注释掉，然后单独安装：
  ```bash
  pip install torch==2.9.1
  ```
- 如果遇到新显卡架构（如 sm_120）不被某些轮子支持，请切换到以上 cu130 组合或从源码编译 PyTorch（设置 `TORCH_CUDA_ARCH_LIST="12.0"`）。
- 训练优化：
  - LSTM 与 LSTM-MT 均支持 AMP（混合精度）：可在 `model_params.use_amp: true` 开启（LSTM 早已支持，LSTM-MT 已对齐并使用 `GradScaler`）。
  - 训练脚本在 CUDA 下会设置 `torch.set_float32_matmul_precision('high')` 以提升 matmul 性能，并在开头打印 GPU 型号与 cuDNN 版本到日志中。

## 📊 Results Summary

### LightGBM (Top 175 Features) - Standard Evaluation

| Horizon | Brier ↓ | ECE ↓ | ROC-AUC ↑ | PR-AUC ↑ | MAE ↓ | RMSE ↓ | R² ↑ |
|---------|---------|-------|-----------|----------|-------|--------|------|
| 3h      | 0.0028  | 0.0015| 0.9965    | 0.9965   | 1.14   | 1.52    | 0.9703|
| 6h      | 0.0040  | 0.0025| 0.9926    | 0.9926   | 1.55   | 2.02    | 0.9481|
| 12h     | 0.0043  | 0.0025| 0.9892    | 0.9892   | 1.79   | 2.33    | 0.9304|
| 24h     | 0.0060  | 0.0048| 0.9843    | 0.9843   | 1.93   | 2.51    | 0.9196|

### LightGBM (Top 175 Features) - LOSO Evaluation

| Horizon | ROC-AUC ↑ | MAE ↓ | RMSE ↓ | R² ↑ |
|---------|-----------|-------|--------|------|
| 3h      | 0.9974    | 1.14   | 1.52    | 0.9703|
| 6h      | 0.9938    | 1.55   | 2.02    | 0.9481|
| 12h     | 0.9905    | 1.79   | 2.33    | 0.9304|
| 24h     | 0.9878    | 1.93   | 2.51    | 0.9196|

**Key Findings:**
- ✅ Excellent spatial generalization (LOSO ROC-AUC > 0.98 for all horizons)
- ✅ Outstanding probability calibration (Brier Score < 0.01, ECE < 0.005)
- ✅ High-precision temperature prediction (MAE < 2°C, R² > 0.91)

## 📚 Documentation

### Main Documentation

- **[USER_GUIDE.md](docs/USER_GUIDE.md)**: Complete user guide - setup, quick start, and advanced usage
- **[TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)**: Technical documentation - architecture, API reference, configuration
- **[DATA_DOCUMENTATION.md](docs/DATA_DOCUMENTATION.md)**: Data documentation - data overview, QC processing, variable usage
- **[FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md)**: Feature engineering - 298 features design and implementation
- **[FEATURE_REFERENCE.md](docs/FEATURE_REFERENCE.md)**: Complete feature reference - all 298 features with acquisition methods
- **[TRAINING_AND_EVALUATION.md](docs/TRAINING_AND_EVALUATION.md)**: Training and evaluation - configuration, LOSO evaluation, performance comparison
- **[INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md)**: Inference guide - how to use trained models for prediction

### Scripts Documentation

- **[Training Scripts](scripts/train/README.md)**: Training scripts usage guide - modular structure and usage examples
- **[Analysis Scripts](scripts/analysis/README.md)**: Feature analysis scripts - analyze features, importance, and generate reports
- **[Evaluation Scripts](scripts/evaluate/README.md)**: Evaluation scripts - model comparison and evaluation

### Reports

- **[LIGHTGBM_ANALYSIS.md](docs/report/LIGHTGBM_ANALYSIS.md)**: Detailed LightGBM performance analysis
- **[CALIBRATION_AND_RELIABILITY_REPORT.md](docs/report/CALIBRATION_AND_RELIABILITY_REPORT.md)**: Calibration and reliability evaluation
- **[FEATURE_SET_COMPARISON.md](docs/report/FEATURE_SET_COMPARISON.md)**: Feature set comparison (298 vs 175 features)
- **[MODEL_COMPARISON_GUIDE.md](docs/MODEL_COMPARISON_GUIDE.md)**: Model comparison guide (LightGBM vs XGBoost)

For complete documentation, see [docs/README_Frost_Forecast_Project.md](docs/README_Frost_Forecast_Project.md).

## 🧾 Project Structure

```
frost-risk-forecast-challenge/
├── data/                    # Data directory
│   ├── raw/                 # Raw data
│   ├── interim/             # Intermediate data
│   └── processed/           # Processed data
├── src/                     # Source code (library code)
│   ├── data/                # Data loading, cleaning, feature engineering
│   ├── models/              # Model implementations (LightGBM, XGBoost, etc.)
│   ├── evaluation/          # Evaluation metrics and validation methods
│   ├── visualization/       # Visualization utilities
│   ├── utils/               # Utility functions
│   └── training/            # Training modules (data prep, model config, trainer, LOSO)
├── scripts/                 # Scripts directory
│   ├── train/               # Training scripts
│   ├── evaluate/            # Evaluation scripts
│   ├── inference/           # Inference scripts
│   ├── analysis/            # Feature analysis scripts
│   │   ├── analyze_all_features.py          # Analyze all features
│   │   ├── analyze_feature_importance.py    # Analyze feature importance
│   │   ├── compare_feature_sets.py          # Compare feature sets
│   │   ├── compare_features.py              # Compare features
│   │   └── generate_feature_report.py       # Generate feature report
│   └── tools/               # Utility scripts (metadata, feature selection, pipeline)
├── experiments/             # Experiment results
│   ├── lightgbm/            # LightGBM models
│   │   ├── feature_importance/  # Feature importance analysis
│   │   └── top175_features/     # Top 175 features configuration
│   │       ├── full_training/    # Standard evaluation
│   │       └── loso/             # LOSO evaluation
│   └── xgboost/            # XGBoost models
│       └── top175_features/     # Top 175 features configuration
│           └── full_training/    # Standard evaluation
├── config/                  # Configuration files
├── docs/                    # Documentation
└── tests/                   # Test code
```

## 🎯 Key Features

- **298 Engineered Features**: Time-based, lagged, rolling statistics, derived, and station features
- **Top 175 Feature Selection**: 90% importance features for optimal performance
- **Multi-Horizon Forecasting**: 3h, 6h, 12h, and 24h predictions
- **Probabilistic Outputs**: Calibrated frost probabilities with temperature predictions
- **Spatial Generalization**: LOSO evaluation across 18 CIMIS stations
- **Model Comparison**: LightGBM and XGBoost implementations
- **Feature Analysis Tools**: Comprehensive feature analysis scripts for exploration and reporting

## 📦 Status

| Item | Status | Location |
|------|--------|----------|
| Data acquisition | ✅ Complete | `data/raw/`, `data/external/` |
| Feature engineering | ✅ Complete | 298 features, `docs/FEATURE_ENGINEERING.md` |
| Model training | ✅ Complete | LightGBM (Top 175), XGBoost (in progress) |
| LOSO evaluation | ✅ Complete | `experiments/lightgbm/top175_features/lightgbm/loso/` |
| Inference services | ✅ Complete | `scripts/inference/predict_frost.py` |
| Feature analysis | ✅ Complete | `scripts/analysis/`, feature analysis scripts |

## 🔗 Links

- **Data Repository**: https://github.com/CarlSaganPhD/frost-risk-forecast-challenge
- **Full Documentation**: [docs/README_Frost_Forecast_Project.md](docs/README_Frost_Forecast_Project.md)

## 📄 License

MIT License --- For research and educational use.

