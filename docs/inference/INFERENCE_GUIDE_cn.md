# AgriFrost-AI Inference Guide

<div align="center">

<img src="logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

</div>

## Overview

This guide explains how to use the trained models to generate frost forecasts in the format required by the challenge:

**"There is a 30% chance of frost in the next 3 hours, predicted temperature: 4.50 °C"**

## ⚠️ Important: Historical Data Requirements

**Key Point**: The model **requires historical data** to make predictions, not just the current data point!

### Why Historical Data is Needed

The model uses **281 features**, including:
- **51 lag features**: Require 1h, 3h, 6h, 12h, 24h historical values
- **180 rolling features**: Require 3h, 6h, 12h, 24h historical windows

**Total**: 231 out of 281 features (82%) require historical data!

### Minimum Data Requirements

| Requirement | Details |
|-------------|---------|
| **Current data point** | Current time step values |
| **Historical data** | **At least 24 hours** of historical data |
| **Data continuity** | Continuous time series (no missing hours) |
| **Total** | **Current + 24 hours minimum** |

### What This Means

- ❌ **Cannot predict with just one data point**
- ✅ **Need at least 24 hours of historical data**
- ✅ **Features must be computed from historical + current data**
- ❌ **Cannot rely on model capacity alone**

## Detailed Historical Data Requirements

### Problem Analysis

The model uses **298 features** (updated from 281), of which **231 features (77.5%) require historical data**:

| Feature Type | Count | Requires History | Minimum History |
|-------------|-------|------------------|-----------------|
| **Lag Features** | 51 | ✅ Yes | 24 hours |
| **Rolling Features** | 180 | ✅ Yes | 24 hours |
| **Time Features** | 11 | ❌ No | 0 hours |
| **Derived Features** | 11 | ❌ No | 0 hours |
| **Other Features** | 45 | ❌ No | 0 hours |
| **Total** | **298** | **231 (77.5%)** | **24 hours** |

### Why Historical Data is Required

#### 1. Lag Features (51 features)

**How they work**: Look back in time at previous values

**Examples**:
- `Air Temp (C)_lag_1` → Temperature 1 hour ago
- `Air Temp (C)_lag_3` → Temperature 3 hours ago
- `Air Temp (C)_lag_6` → Temperature 6 hours ago
- `Air Temp (C)_lag_12` → Temperature 12 hours ago
- `Air Temp (C)_lag_24` → Temperature 24 hours ago

**Requirement**: Need at least **24 hours of historical data** (for lag_24)

#### 2. Rolling Features (180 features)

**How they work**: Calculate statistics over a rolling window

**Examples**:
- `Air Temp (C)_rolling_3h_mean` → Average temperature over last 3 hours
- `Air Temp (C)_rolling_6h_mean` → Average temperature over last 6 hours
- `Air Temp (C)_rolling_12h_mean` → Average temperature over last 12 hours
- `Air Temp (C)_rolling_24h_mean` → Average temperature over last 24 hours

**Requirement**: Need at least **24 hours of historical data** (for rolling_24h)

#### 3. Time and Derived Features (67 features)

**How they work**: Only need current data point

**Examples**:
- `hour`, `month`, `day_of_year` → Current time
- `wind_chill`, `heat_index` → Current values only
- `Air Temp (C)`, `Dew Point (C)` → Current values only

**Requirement**: Only need **current data point**

### What Does NOT Work

```python
# ❌ This will NOT work
current_data = {
    "timestamp": "2025-01-02 00:00",
    "Air Temp (C)": 2.5,
    "Dew Point (C)": 1.0,
    ...
}

# Cannot predict with just current data!
prediction = model.predict(current_data)  # ❌ Missing 231 features!
```

**Why it fails**:
- Missing `Air Temp (C)_lag_1` through `_lag_24` (5 features)
- Missing `Air Temp (C)_rolling_3h_mean` through `_rolling_24h_max` (20 features)
- Missing all other lag and rolling features (231 total)
- Model expects 298 features but only gets ~67 features

### What DOES Work

```python
# ✅ This works
historical_data = [
    {"timestamp": "2025-01-01 00:00", "Air Temp (C)": 5.0, ...},  # 24h ago
    {"timestamp": "2025-01-01 01:00", "Air Temp (C)": 4.8, ...},  # 23h ago
    ...
    {"timestamp": "2025-01-01 23:00", "Air Temp (C)": 3.0, ...},  # 1h ago
    {"timestamp": "2025-01-02 00:00", "Air Temp (C)": 2.5, ...},  # Current
]

# Step 1: Feature engineering on historical + current data
features = create_features(historical_data)  # Creates all 298 features

# Step 2: Extract features for current time point
current_features = features.iloc[[-1]]  # Last row (current time)

# Step 3: Predict
prediction = model.predict(current_features)  # ✅ Works!
```

### Data Requirements Summary

| Component | Requirement | Reason |
|-----------|-------------|--------|
| **Current data point** | 1 hour | Current time step |
| **Historical data** | **24 hours minimum** | For lag_24 and rolling_24h |
| **Total** | **25 hours total** | Current + 24 hours history |
| **Data continuity** | No missing hours | Needed for lag/rolling features |

## Reproducibility Metadata (`data_run_metadata.json`)

- 无论是训练还是推理，只要通过 DataPipeline 生成 `labeled_data.parquet`，都会同步产出 `data_run_metadata.json`，其中包含 `cleaning_config_hash`、特征配置、标签规则以及时间戳。  
- **发布或共享模型/特征文件时，务必连同这一 JSON 一起提供**。部署侧可以直接校验 `cleaning_config_hash`，确保推理阶段使用的清洗/特征工程配置与训练阶段完全一致。  
- 如果仅提供历史的 `labeled_data.parquet` 而缺少 metadata，推理脚本会给出警告；推荐直接传递原始站点数据并让推理脚本重新运行 DataPipeline，以自动生成同源的 metadata。

### Real-time Inference Solutions

#### Solution 1: Real-time Inference with History (Recommended)

```python
def predict_realtime(station_id, current_time):
    """Real-time prediction with historical data."""
    
    # Step 1: Fetch historical data (last 24 hours + current)
    historical_data = fetch_historical_data(
        station_id=station_id,
        start_time=current_time - timedelta(hours=24),
        end_time=current_time
    )
    
    # Step 2: Feature engineering
    from src.data.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(historical_data)
    
    # Step 3: Extract current time point features
    current_features = df_features.iloc[[-1]]  # Last row
    
    # Step 4: Predict
    frost_proba, temp_pred = predict_single(
        models_dir, horizon, current_features
    )
    
    return frost_proba, temp_pred
```

#### Solution 2: Pre-computed Features Cache

```python
def predict_with_cache(station_id, current_time):
    """Prediction using pre-computed feature cache."""
    
    # Step 1: Load pre-computed features (last 24 hours)
    features_cache = load_features_cache(
        station_id=station_id,
        end_time=current_time,
        hours=24
    )
    
    # Step 2: Extract current time point
    current_features = features_cache.iloc[[-1]]
    
    # Step 3: Predict
    return predict_single(models_dir, horizon, current_features)
```

### Common Questions

#### Q1: Can I predict with just the current data point?

**A**: ❌ **No, you need at least 24 hours of historical data** because:
- 51 lag features require historical values
- 180 rolling features require historical windows
- Total: 231 out of 298 features (77.5%) need history

#### Q2: Why does the model need historical data?

**A**: The model was trained on features that include:
- **Historical patterns** (lag features)
- **Recent trends** (rolling features)
- **Time dynamics** (temporal dependencies)

These features are **essential** for the model's predictions. The model learned patterns like:
- "If temperature was low 24h ago and is dropping now, frost is likely"
- "If the 24h rolling mean is below 5°C, frost risk increases"

#### Q3: Can the model infer historical patterns from current data?

**A**: ❌ **No, the model cannot infer historical patterns**:
- The model is a **supervised learner**, not a time series generator
- It was trained on **pre-computed features** (lag, rolling)
- It expects these features as **input**, not as something to infer
- The model's "capacity" is for **pattern recognition**, not **data reconstruction**

## Challenge Requirements

The challenge requires models that can:
1. **Predict frost probability** (T < 0°C) for horizons 3h, 6h, 12h, and 24h
2. **Predict temperature** for the same horizons
3. **Output calibrated probabilities** in the specified format
4. **Use only information available up to the forecast time** (no data leakage)

## Current Implementation Status

### ✅ **Fully Implemented**

1. **Multi-horizon Forecasting**
   - ✅ Models trained for 3h, 6h, 12h, and 24h horizons
   - ✅ Separate models for frost probability (classification) and temperature (regression)
   - ✅ All models trained and evaluated

2. **No Data Leakage**
   - ✅ Features use only historical data (lags, rolling statistics)
   - ✅ Target labels created using forward shift (future values)
   - ✅ Time-based data splitting (train/val/test)
   - ✅ Leave-One-Station-Out (LOSO) evaluation for generalization

3. **Model Output Format**
   - ✅ Frost probability: 0-1 (calibrated)
   - ✅ Temperature prediction: °C
   - ✅ Formatted output: "There is a X% chance of frost in the next Y hours, predicted temperature: Z °C"

4. **Model Performance**
   - ✅ High accuracy for frost classification (ROC-AUC > 0.98 for all horizons)
   - ✅ Low temperature prediction error (MAE < 2°C for all horizons)
   - ✅ Well-calibrated probabilities (ECE < 0.006 for all horizons)

## Usage

### Inference Script

The inference command (`python -m src.cli inference predict`) loads trained models and generates formatted predictions.

#### Basic Usage

```bash
# Predict for a single sample at index 0
python -m src.cli inference predict \
    --models experiments/full_data_training \
    --horizons 3 \
    --index 0

# Predict for all horizons
python -m src.cli inference predict \
    --models experiments/full_data_training \
    --horizons 3 6 12 24 \
    --index 0

# Batch predictions (save to CSV)
python -m src.cli inference predict \
    --models experiments/full_data_training \
    --horizons 3 6 12 24 \
    --sample-size 100 \
    --output predictions.csv
```

#### Example Output

```
============================================================
PREDICTION
============================================================
There is a 0.0% chance of frost in the next 3 hours, predicted temperature: 9.12 °C
============================================================
Frost Probability: 0.0001
Predicted Temperature: 9.12 °C

============================================================
PREDICTIONS FOR ALL HORIZONS
============================================================
There is a 0.0% chance of frost in the next 3 hours, predicted temperature: 9.12 °C
There is a 0.0% chance of frost in the next 6 hours, predicted temperature: 7.47 °C
There is a 0.1% chance of frost in the next 12 hours, predicted temperature: 3.40 °C
There is a 0.0% chance of frost in the next 24 hours, predicted temperature: 13.56 °C
```

### Programmatic Usage

#### Option 1: Using Pre-computed Features (Current Implementation)

⚠️ **Note**: Legacy script-based APIs have been removed. Use the CLI command or the new API below.

```python
# Use CLI command instead:
# python -m src.cli inference predict --model-dir experiments/full_data_training --input data.csv --output predictions.csv

# Or use the programmatic API:
from src.inference.predictor import FrostPredictor
from pathlib import Path
import pandas as pd

# Load data that already has all features computed
df = pd.read_parquet("experiments/full_data_training/labeled_data.parquet")

# Initialize predictor
predictor = FrostPredictor(model_dir=Path("experiments/full_data_training"))

# Make prediction
result = predictor.predict(df.iloc[[0]], horizon_h=3)
print(f"Frost probability: {result.frost_probability:.2%}")
print(f"Predicted temperature: {result.temperature:.2f} °C")
```

#### Option 2: Real-time Prediction with Historical Data (Recommended)

```python
from datetime import datetime, timedelta
from src.inference.predictor import FrostPredictor
from src.data.feature_engineering import FeatureEngineer
from pathlib import Path

# Step 1: Fetch historical data (last 24 hours + current)
current_time = datetime.now()
historical_data = fetch_historical_data(
    station_id=1,
    start_time=current_time - timedelta(hours=24),
    end_time=current_time
)

# Step 2: Feature engineering (creates lag and rolling features)
# Use DataPipeline for proper feature engineering
from src.data import DataPipeline
from src.data.loaders import DataLoader

loader = DataLoader()
df = loader.load_csv(historical_data_path)  # Load historical data

# Initialize pipeline (use same config as training)
pipeline = DataPipeline.from_config(config_path)  # Same config used for training
df_features = pipeline.process(df)

# Step 3: Extract current time point features
current_features = df_features.iloc[[-1]]  # Last row (current time)

# Step 4: Predict using FrostPredictor
predictor = FrostPredictor(model_dir=Path("experiments/full_data_training"))
result = predictor.predict(current_features, horizon_h=3)

print(f"Frost probability: {result.frost_probability:.2%}")
print(f"Predicted temperature: {result.temperature:.2f} °C")
```

**Important**: `current_features` must include all 281 features, including lag and rolling features computed from historical data!

## Model Architecture

### Frost Classification Model
- **Type**: LightGBM Classifier
- **Output**: Frost probability (0-1)
- **Metrics**: ROC-AUC, Brier Score, ECE (Expected Calibration Error)

### Temperature Regression Model
- **Type**: LightGBM Regressor
- **Output**: Temperature in °C
- **Metrics**: MAE, RMSE, R²

### Features
- **Time Features**: Hour, day of year, month, season, cyclical encodings
- **Lag Features**: Historical values (1h, 3h, 6h, 12h, 24h ago)
- **Rolling Statistics**: Mean, std, min, max over windows (6h, 12h, 24h)
- **Derived Features**: Temperature decline rate, wind chill, etc.
- **Station Features**: Station ID encoding, CIMIS region

## Model Performance

### Standard Evaluation (Time-based Split)

| Horizon | Frost ROC-AUC | Frost Brier | Temp MAE (°C) | Temp RMSE (°C) | Temp R² |
|---------|---------------|-------------|---------------|----------------|---------|
| 3h      | 0.9966        | 0.0028      | 1.16          | 1.55           | 0.969   |
| 6h      | 0.9924        | 0.0042      | 1.58          | 2.06           | 0.946   |
| 12h     | 0.9879        | 0.0046      | 1.86          | 2.41           | 0.926   |
| 24h     | 0.9832        | 0.0063      | 1.98          | 2.57           | 0.915   |

### LOSO Evaluation (Cross-Station Generalization)

| Horizon | Frost ROC-AUC | Frost Brier | Temp MAE (°C) | Temp RMSE (°C) |
|---------|---------------|-------------|---------------|----------------|
| 3h      | 0.95-0.99     | 0.003-0.01  | 1.2-2.5       | 1.6-3.2        |
| 6h      | 0.93-0.98     | 0.004-0.01  | 1.5-3.0       | 2.0-3.8        |
| 12h     | 0.90-0.97     | 0.005-0.01  | 1.8-3.5       | 2.3-4.5        |
| 24h     | 0.88-0.96     | 0.006-0.02  | 2.0-4.0       | 2.6-5.0        |

## Data Requirements

### Input Data Format

The inference script expects:
- **Labeled data**: Parquet file with features and labels (from training)
- **Features**: Same features used during training (281 features)
- **Feature order**: Must match the order used during training

### Feature Preparation

Features are prepared using the same pipeline as training:
1. Load raw data
2. Clean data (handle missing values, outliers)
3. Create features (time, lags, rolling statistics)
4. Create target labels (frost probability, temperature)

## Verification

### ✅ Challenge Requirements Met

1. **Probabilistic Frost Forecasting**
   - ✅ Models predict frost probability (calibrated)
   - ✅ Output format: "There is a X% chance of frost in the next Y hours"

2. **Temperature Prediction**
   - ✅ Models predict temperature in °C
   - ✅ Output format: "predicted temperature: Z °C"

3. **Multi-horizon Forecasting**
   - ✅ 3h, 6h, 12h, 24h horizons supported
   - ✅ Separate models for each horizon

4. **No Data Leakage**
   - ✅ Features use only historical data
   - ✅ Target labels use forward shift (future values)
   - ✅ Time-based data splitting

5. **Calibrated Probabilities**
   - ✅ Well-calibrated (ECE < 0.006)
   - ✅ Reliability diagrams show good calibration

## Troubleshooting

### Common Issues

1. **Missing Features Error**
   - **Problem**: Features don't match model expectations
   - **Solution**: Ensure features are prepared using the same pipeline as training

2. **Model Not Found Error**
   - **Problem**: Model files not found
   - **Solution**: Check that models are trained and saved in the expected directory

3. **Memory Issues**
   - **Problem**: Out of memory for large batch predictions
   - **Solution**: Use smaller batch sizes or predict in chunks

## Next Steps

1. **Deployment**: Deploy models as a REST API or batch service
2. **Real-time Prediction**: Integrate with real-time weather data streams
3. **Model Monitoring**: Track model performance over time
4. **A/B Testing**: Compare different model configurations

## Probability Calculation

### How Probability Percentages Are Calculated

The model outputs probabilities, which are converted to percentages for display.

#### 1. Model Output

**Model Type**: LightGBM Classifier (Binary Classification)

**Output Format**:
```python
proba = model.predict_proba(X)
# Output: [[P(No Frost), P(Frost)]]
# Example: [[0.9806, 0.0194]]
```

**Output Details**:
- Output is a 2D array with shape `(n_samples, 2)`
- First value `proba[0][0]` is "No Frost" probability (T ≥ 0°C)
- Second value `proba[0][1]` is "Frost" probability (T < 0°C)
- Two probabilities sum to 1.0

#### 2. Extract Frost Probability

```python
# Extract Class 1 (Frost) probability
frost_proba = proba[:, 1]  # For single sample: proba[0][1]

# Example: 0.0194 (1.94%)
```

#### 3. Convert to Percentage

```python
# Probability (0-1) → Percentage (0-100)
prob_percent = frost_proba * 100

# Example: 0.0194 * 100 = 1.94
```

#### 4. Format Output

```python
# Format to 1 decimal place
formatted_percent = f"{prob_percent:.1f}%"

# Example: "1.9%"
```

### Complete Code Example

```python
# 1. Load model
model_path = Path("experiments/full_data_training/horizon_3h/frost_classifier")
with open(model_path / "model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Prepare features
X = prepare_features(data)  # Feature DataFrame

# 3. Predict probabilities
proba = model.predict_proba(X)
# Output: [[0.9806, 0.0194]]

# 4. Extract frost probability
frost_proba = proba[0][1]  # 0.0194

# 5. Convert to percentage
prob_percent = frost_proba * 100  # 1.94

# 6. Format output
formatted = f"There is a {prob_percent:.1f}% chance of frost in the next 3 hours"
# Output: "There is a 1.9% chance of frost in the next 3 hours"
```

### Probability Calibration

The model's probability outputs are **well-calibrated**:
- **Brier Score**: 0.0028-0.0063 (lower is better)
- **ECE (Expected Calibration Error)**: 0.0010-0.0058 (lower is better)
- **Reliability Diagrams**: Show good calibration

**Calibration Method**:
- LightGBM Classifier uses gradient boosting tree probability outputs
- Probabilities are naturally well-calibrated
- No additional calibration needed

### Examples

**Low Frost Probability**:
- Model output: `[[0.9999, 0.0001]]`
- Frost probability: 0.0001
- Percentage: 0.01%
- Output: "There is a 0.0% chance of frost in the next 3 hours"

**Medium Frost Probability**:
- Model output: `[[0.7000, 0.3000]]`
- Frost probability: 0.3000
- Percentage: 30.0%
- Output: "There is a 30.0% chance of frost in the next 3 hours"

**High Frost Probability**:
- Model output: `[[0.1000, 0.9000]]`
- Frost probability: 0.9000
- Percentage: 90.0%
- Output: "There is a 90.0% chance of frost in the next 3 hours"

### Summary

**Calculation Formula**:
```
Percentage = P(Frost) × 100
```

**Process**:
1. Model outputs `[[P(No Frost), P(Frost)]]`
2. Extract `P(Frost)` (second value)
3. Multiply by 100 to get percentage
4. Format to 1 decimal place
5. Insert into output string

**Example**:
```
P(Frost) = 0.30 → Percentage = 30.0% → "There is a 30.0% chance of frost in the next 3 hours"
```

## References

- [Training Guide](TRAINING_AND_EVALUATION.md)
- [Feature Guide](FEATURE_GUIDE.md)
- [Technical Documentation](TECHNICAL_DOCUMENTATION.md)

