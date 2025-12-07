# AgriFrost-AI Technical Documentation

<div align="center">

<img src="logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

</div>

**Last Updated**: 2025-12-06

This document provides technical architecture, API reference, and development guidelines.

## 📋 Table of Contents

1. [Project Architecture](#project-architecture)
2. [Supported Models](#supported-models)
3. [API Reference](#api-reference)
4. [Configuration Management](#configuration-management)
5. [Extension Development](#extension-development)

---

## Project Architecture

### Core Design Principles

1. **Modular Design**: Each functional module is independent, facilitating testing and replacement
2. **Standardized Interfaces**: Unified data interfaces, model interfaces, evaluation interfaces
3. **Extensibility**: Adding new models/features/evaluation metrics without modifying core code
4. **Reproducibility**: All experiment configurations, random seeds, version numbers are traceable

### Project Directory Structure

```
frost-risk-forecast-challenge/
├── config/                    # Configuration files
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   ├── processed/             # Cleaned data
│   └── external/              # External data
├── src/                       # Source code
│   ├── data/                  # Data processing module
│   ├── models/                # Model module
│   ├── evaluation/            # Evaluation module
│   └── utils/                 # Utility functions
├── scripts/                   # Executable scripts
│   ├── data_prep/             # Data preparation
│   ├── train/                 # Training scripts
│   └── evaluate/              # Evaluation scripts
├── experiments/               # Experiment output (not tracked)
├── results/                   # Result summaries (not tracked)
└── docs/                      # Documentation
```

### Core Modules

#### 1. Data Module (`src/data/`)

- **`loaders.py`**: Data loaders
- **`cleaners.py`**: QC cleaning and data processing
- **`feature_engineering.py`**: Feature engineering
- **`validators.py`**: Data validation

#### 2. Model Module (`src/models/`)

- **`base.py`**: Base model interface
- **`ml/`**: Machine learning models (LightGBM, XGBoost)
- **`traditional/`**: Traditional time series models
- **`deep/`**: Deep learning models

#### 3. Evaluation Module (`src/evaluation/`)

- **`metrics.py`**: Evaluation metrics
- **`validators.py`**: Cross-validation strategies
- **`comparators.py`**: Model comparison

---

## Supported Models

### LightGBM ⭐ (Default)

**Features**:
- Fast training and prediction
- Automatic missing value handling
- Feature importance extraction
- High memory efficiency

**Configuration Example**:
```python
{
    "model_type": "lightgbm",
    "task_type": "regression",
    "model_params": {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "random_state": 42
    }
}
```

### XGBoost

**Features**:
- Stable performance
- Strong regularization capability
- Feature importance support

### Model Comparison Table

| Model | Category | Regression | Classification | Feature Importance | Speed |
|-------|----------|------------|---------------|-------------------|-------|
| LightGBM | ML | ✅ | ✅ | ✅ | ⚡⚡⚡ |
| XGBoost | ML | ✅ | ✅ | ✅ | ⚡⚡ |

---

## API Reference

### Data Loading

```python
from src.data.loaders import DataLoader

# Load raw data
df = DataLoader.load_raw_data(Path("data/raw/frost-risk-forecast-challenge/stations"))
```

### Data Cleaning

```python
from src.data.cleaners import DataCleaner

cleaner = DataCleaner()
df_cleaned = cleaner.clean_pipeline(df)
```

### Feature Engineering

```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
feature_config = {
    "time_features": True,
    "lag_features": {"enabled": True, "columns": [...], "lags": [1, 3, 6, 12, 24]},
    "rolling_features": {"enabled": True, ...},
    "derived_features": True
}
df_features = engineer.build_feature_set(df_cleaned, feature_config)
```

### Model Usage

```python
from src.models.ml.lightgbm import LightGBMModel

# Create model
model = LightGBMModel(config)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # If supported

# Feature importance
importance = model.get_feature_importance()

# Save/load
model.save(Path("model_dir"))
loaded_model = LightGBMModel.load(Path("model_dir"))
```

### Evaluation

```python
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.validators import CrossValidator

# Calculate metrics
metrics = MetricsCalculator.calculate_all_metrics(
    y_true, y_pred, task_type="regression"
)

# Cross-validation
splits = CrossValidator.leave_one_station_out(df)
```

---

## Configuration Management

### Model Configuration File Structure

```yaml
model_name: "lightgbm_baseline"
model_type: "lightgbm"
task_type: "regression"

data:
  input_path: "data/interim/features/cimis_features.parquet"
  target_column: "Air Temp (C)"
  feature_columns: []  # Empty = auto-select

model_params:
  n_estimators: 100
  learning_rate: 0.05
  max_depth: 6

training:
  validation_strategy: "time_split"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

evaluation:
  metrics:
    regression: ["mae", "rmse", "r2", "mape"]
```

---

## Extension Development

### Adding New Models

1. Create new file under `src/models/`
2. Inherit from `BaseModel` class
3. Implement `fit()`, `predict()`, `predict_proba()` methods
4. Create configuration file
5. Add unit tests

### Adding New Features

1. Add new method in `FeatureEngineer` class
2. Enable the feature in configuration file
3. Validate feature quality (correlation, importance)

### Adding New Evaluation Metrics

1. Add method in `MetricsCalculator`
2. Add to metrics list in configuration file
3. Automatically included in comparison reports

---

## 📚 Related Documentation

- **[User Guide](../guides/USER_GUIDE.md)**: User guide
- **[Data Documentation](DATA_DOCUMENTATION.md)**: Data documentation
- **[Feature Guide](../features/FEATURE_GUIDE.md)**: Complete feature engineering guide
- **[Training Guide](../training/TRAINING_GUIDE.md)**: Training and evaluation documentation

---

**Last Updated**: 2025-12-06  
**Document Version**: 1.0
