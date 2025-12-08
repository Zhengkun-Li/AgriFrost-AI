# Example Experiment: LightGBM Matrix A Full Training

This directory contains a complete example experiment output from the AgriFrost-AI frost risk forecasting pipeline. This demonstrates the baseline configuration (Matrix A) using single-station raw meteorological features.

## Quick Links

- **[Main README](../../../../README.md)**: Project overview, setup, and usage guide
- **[Pipeline Configuration](../../../../config/pipeline/matrix_a.yaml)**: This experiment's configuration file
- **[Feature Guide](../../../../docs/features/FEATURE_GUIDE.md)**: Feature engineering documentation
- **[Training Guide](../../../../docs/training/TRAINING_GUIDE.md)**: How to run training experiments
- **[Model Guide](../../../../docs/models/MODELS_GUIDE.md)**: Model architecture details

## Experiment Overview

### Configuration
- **Model**: LightGBM (Gradient Boosting Decision Tree)
- **Feature Matrix**: **Matrix A** (Baseline)
  - **Type**: Single-station, raw features
  - **No feature engineering**: Uses only raw meteorological variables
  - **No spatial aggregation**: Single station prediction
- **Task**: Multi-task learning
  - Frost classification (binary: frost/non-frost)
  - Temperature regression (minimum temperature prediction)
- **Forecast Horizons**: 3h, 6h, 12h, 24h
- **Evaluation**: Full training + Leave-One-Station-Out (LOSO) cross-validation

### Features (12 raw meteorological variables)
- `Hour (PST)`: Hour of day (0-23)
- `Jul`: Julian day (day of year, 1-366)
- `ETo (mm)`: Reference evapotranspiration
- `Precip (mm)`: Precipitation
- `Sol Rad (W/sq.m)`: Solar radiation
- `Vap Pres (kPa)`: Vapor pressure
- `Air Temp (C)`: Air temperature
- `Rel Hum (%)`: Relative humidity
- `Dew Point (C)`: Dew point temperature
- `Wind Speed (m/s)`: Wind speed
- `Wind Dir (0-360)`: Wind direction
- `Soil Temp (C)`: Soil temperature

### Model Configuration
- **Learning Rate**: 0.05
- **N Estimators**: 200 trees
- **Max Depth**: 8
- **Num Leaves**: 63
- **Subsample**: 0.8
- **Colsample by Tree**: 0.8
- **Regularization**: L1=0.1, L2=0.1
- **Class Imbalance Handling**: `is_unbalance=True` (automatic class weight adjustment)

## Performance Summary

### Full Training Results

| Horizon | Frost Classification | Temperature Regression |
|---------|---------------------|------------------------|
| **3h**  | ROC-AUC: 0.997, PR-AUC: 0.715, Recall: 63.3% | RMSE: 1.63°C, MAE: 1.24°C, R²: 0.966 |
| **6h**  | ROC-AUC: 0.992, PR-AUC: 0.540, Recall: 56.5% | RMSE: 2.25°C, MAE: 1.73°C, R²: 0.935 |
| **12h** | ROC-AUC: 0.986, PR-AUC: 0.388, Recall: 41.2% | RMSE: 2.79°C, MAE: 2.16°C, R²: 0.901 |
| **24h** | ROC-AUC: 0.982, PR-AUC: 0.306, Recall: 38.7% | RMSE: 2.59°C, MAE: 1.99°C, R²: 0.914 |

**Key Observations:**
- Classification performance degrades with longer forecast horizons (as expected)
- Temperature prediction remains accurate even at 24h horizon (R² > 0.91)
- Class imbalance challenge: Only ~0.87% positive (frost) samples
- Optimal threshold selection uses F2-score to prioritize recall (minimize false negatives)

### LOSO Cross-Validation (Spatial Generalization)

**18 CIMIS weather stations** in California's Central Valley:
- Mean ROC-AUC across stations (3h): 0.997 ± 0.002
- Mean PR-AUC across stations (3h): 0.820 ± 0.062
- Mean RMSE across stations (3h): 2.03°C ± 0.13°C
- Mean R² across stations (3h): 0.947 ± 0.007

**Interpretation**: Models show good spatial generalization, with consistent performance across different weather stations.

## Directory Structure

```
full_training_example/
├── data_run_metadata.json          # Data processing metadata and feature config
├── experiment.log                   # Main experiment execution log
├── summary.json                     # Overall experiment summary with all metrics
│
├── horizon_3h/                      # 3-hour forecast horizon results
│   ├── frost_classifier/           # Frost classification model
│   │   ├── config.json             # Model hyperparameters
│   │   └── model.pkl               # Trained LightGBM model (binary)
│   ├── temp_regressor/             # Temperature regression model
│   │   ├── config.json
│   │   └── model.pkl               # Trained LightGBM model (regression)
│   ├── frost_metrics.json          # Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score, ECE)
│   ├── temp_metrics.json           # Regression metrics (MAE, RMSE, R², MAPE)
│   ├── frost_feature_importance.csv # Feature importance rankings for classification
│   ├── temp_feature_importance.csv  # Feature importance rankings for regression
│   ├── reliability_diagram.png     # Probability calibration visualization
│   ├── training.log                # Training process log
│   ├── training_detailed.log       # Detailed training log with iterations
│   └── run_metadata.json           # Run-specific metadata
│
├── horizon_6h/                     # 6-hour forecast horizon (same structure)
├── horizon_12h/                    # 12-hour forecast horizon (same structure)
├── horizon_24h/                    # 24-hour forecast horizon (same structure)
│
└── loso/                           # Leave-One-Station-Out cross-validation
    ├── checkpoint.json             # LOSO progress tracking
    ├── station_results.json        # Per-station detailed results
    ├── station_metrics.csv         # Per-station metrics summary (CSV format)
    ├── summary.json                # LOSO summary statistics (mean, std, min, max)
    └── station_*/                  # Per-station LOSO results (18 stations)
        └── horizon_*/              # Per-horizon training logs
            ├── training.log
            └── training_detailed.log
```

## File Descriptions

### Metrics Files (`*_metrics.json`)
- **Frost Classification Metrics**:
  - `accuracy`, `precision`, `recall`, `f1`: Standard classification metrics
  - `roc_auc`: Area under ROC curve (discrimination ability)
  - `pr_auc`: Area under Precision-Recall curve (better for imbalanced data)
  - `brier_score`: Probability calibration quality (lower is better)
  - `ece`: Expected Calibration Error (probability calibration)
  - `tp`, `tn`, `fp`, `fn`: Confusion matrix counts
  
- **Temperature Regression Metrics**:
  - `mae`: Mean Absolute Error (°C)
  - `rmse`: Root Mean Squared Error (°C)
  - `r2`: Coefficient of determination (explained variance)
  - `mape`: Mean Absolute Percentage Error (%)

### Feature Importance Files (`*_feature_importance.csv`)
- Ranked list of features by importance (gain-based)
- Useful for understanding which meteorological variables are most predictive
- Separate rankings for frost classification vs. temperature regression

### Model Files (`model.pkl`)
- Serialized LightGBM models using Python `pickle`
- Can be loaded for inference: `pickle.load(open('model.pkl', 'rb'))`
- Separate models for classification and regression tasks

### Reliability Diagrams (`reliability_diagram.png`)
- Visualizes probability calibration quality
- Shows predicted probability vs. observed frequency
- Well-calibrated models should lie on the diagonal

## What's Excluded (Large Files)

To keep the repository size manageable (~12 MB instead of ~70+ MB), the following large files have been excluded:

- **`labeled_data.parquet`** (~32 MB): Preprocessed training dataset with all features and labels
- **`predictions.json`** (~30 MB per horizon): Full prediction arrays for all samples
  - Contains: `y_true`, `y_pred`, `y_proba` for both tasks

**Note**: These files can be regenerated by running the training pipeline:
```bash
python -m src.cli.train \
    --model lightgbm \
    --config config/pipeline/matrix_a.yaml \
    --output experiments/lightgbm/raw/A/full_training
```

## How to Use This Example

1. **Understand experiment structure**: This demonstrates the standard output format for all experiments
2. **Compare configurations**: Compare with Matrix B/C/D to see feature engineering impact
3. **Analyze metrics**: Review performance across horizons and tasks
4. **Inspect feature importance**: Understand which features drive predictions
5. **Check model calibration**: Review reliability diagrams for probability quality
6. **Validate spatial generalization**: Review LOSO results for cross-station performance

## Related Experiments

- **Matrix B**: Single-station with feature engineering (lag, rolling, time features)
- **Matrix C**: Spatial aggregation (multi-station features with distance weighting)
- **Matrix D**: Full feature engineering + spatial aggregation

## Total Size

**After excluding large files: ~12 MB** (200 files)

This example demonstrates the complete structure of experiment outputs while keeping the repository size reasonable for version control and sharing.

