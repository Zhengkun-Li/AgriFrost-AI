# AgriFrost-AI Complete Feature Engineering Guide

<div align="center">

<img src="../logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

</div>

**Last Updated**: 2025-12-06

This document integrates all feature engineering related content including QC field processing, Jul feature explanation, and feature engineering recommendations, providing a one-stop reference for feature engineering.

## 📋 Table of Contents

1. [Feature Overview](#feature-overview)
2. [Feature Categories](#feature-categories)
3. [Feature Implementation](#feature-implementation)
4. [Feature Selection](#feature-selection)
5. [QC Field Processing](#qc-field-processing)
6. [Jul Feature Details](#jul-feature-details)
7. [Feature Engineering Recommendations](#feature-engineering-recommendations)
8. [Performance Comparison](#performance-comparison)
9. [Feature Reference](#feature-reference)

---

## Feature Overview

### Feature Statistics

| Category | Count | Percentage | Description |
|----------|-------|------------|-------------|
| **Total Features** | **298** | **100%** | All engineered features |
| Rolling Features | 180 | 60.4% | Rolling window statistics |
| Lag Features | 51 | 17.1% | Historical lag values |
| Station Features | 24 | 8.1% | Geographic features |
| Time Features | 11 | 3.7% | Temporal patterns |
| Derived Features | 11 | 3.7% | Computed features |
| Wind Features | 7 | 2.3% | Wind-related features |
| Humidity Features | 6 | 2.0% | Humidity features |
| Radiation Features | 3 | 1.0% | Solar radiation |
| Other Features | 3 | 1.0% | ETo, Precip, Vap Pres |
| Temperature Features | 2 | 0.7% | Air temp/Soil temp |

### Key Insights

1. **Rolling features dominate** (60.4%): Capture recent trends and patterns
2. **Lag features important** (17.1%): Capture historical context
3. **Station features significant** (8.1%): Capture geographic effects
4. **Time features essential** (3.7%): Capture temporal patterns
5. **Derived features valuable** (3.7%): Capture physical relationships

---

## Feature Categories

### 1. Time Features (11)

Time features capture temporal patterns and seasonal cycles.

**Implementation**: `create_time_features()` in `src/data/features/temporal.py`

**Features**:
- `hour`, `hour_cos`, `hour_sin` - Hour of day (0-23), cyclic encoding
- `month`, `month_cos`, `month_sin` - Month (1-12), cyclic encoding
- `day_of_year` - Day of year (1-366)
- `day_of_week` - Day of week (0-6)
- `season` - Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)
- `is_night` - Night indicator (18:00-06:00)

**Purpose**: Capture diurnal and seasonal patterns (coldest at night, winter)

### 2. Lag Features (51)

Values from specific historical time points.

**Implementation**: `create_lag_features()` in `src/data/features/lagging.py`

**Lag Values**: 1h, 3h, 6h, 12h, 24h

**Variables**: Air temp, dew point, ETo, precipitation, relative humidity, soil temp, solar radiation, wind direction, wind speed, vapor pressure

**Example**: `Air Temp (C)_lag_1`, `Air Temp (C)_lag_3`, ..., `Air Temp (C)_lag_24`

**Purpose**: Capture recent trends and patterns affecting current conditions

### 3. Rolling Features (180)

Rolling window statistics (mean, min, max, std) over different time windows.

**Implementation**: `create_rolling_features()` in `src/data/features/lagging.py`

**Windows**: 3h, 6h, 12h, 24h

**Statistics**: mean, min, max, std, sum

**Variables**: Same as lag features (10 variables × 4 windows × 5 functions = 200 features) - 20 derived rolling features

**Example**: `Air Temp (C)_rolling_6h_mean`, `Air Temp (C)_rolling_24h_min`

**Purpose**: Capture recent trends and patterns through statistical aggregation

### 4. Station Features (24)

Geographic and station-specific features.

**Implementation**: Merged from station metadata

**Features**:
- `latitude`, `longitude`, `elevation` - Geographic coordinates
- `station_id` - Station identifier
- Other geographic features from metadata

**Purpose**: Capture geographic effects and spatial patterns

### 5. Derived Features (11)

Features computed from original variables.

**Examples**:
- `temp_dew_diff` - Temperature minus dew point
- `wind_chill` - Wind chill index
- `heat_index` - Heat index

**Purpose**: Capture physical relationships and interactions

### 6. Other Feature Categories

- **Radiation Features (3)**: Solar radiation related
- **Wind Features (7)**: Wind direction and speed (with cyclic encoding)
- **Humidity Features (6)**: Relative humidity and related
- **Temperature Features (2)**: Air temperature and soil temperature

---

## QC Field Processing

### Role of QC Fields

QC (Quality Control) fields are used to mark data quality and filter low-quality data.

#### QC Flag Meanings

| QC Flag | Meaning | Processing |
|---------|---------|------------|
| `''` (empty) | High-quality data | ✅ **Keep original value** |
| `'Y'` | Moderate outlier but accepted | ✅ **Keep original value** |
| `'M'` | Missing (missing data) | ❌ **Mark as NaN** |
| `'R'` | Rejected (extreme outlier) | ❌ **Mark as NaN** |
| `'S'` | Severe outlier | ❌ **Mark as NaN** |
| `'Q'` | Questionable (questionable data) | ❌ **Mark as NaN** |
| `'P'` | Provisional (temporary data) | ❌ **Mark as NaN** |

### QC Fields **Are NOT** Used as Model Features

#### Reasons

1. **Data Type Mismatch**
   - QC columns contain **string-type** flags (`'Y'`, `'M'`, `'R'`, `'S'`, `'Q'`, `'P'`)
   - During model training, feature selection uses `select_dtypes(include=[np.number])` to select only **numeric** features
   - QC columns are not numeric, **will be automatically excluded**

2. **Semantic Incorrectness**
   - QC columns are **quality control flags**, not **prediction features**
   - QC columns are only used to mark data quality, should not be used to predict target variables

3. **Would Cause Data Leakage**
   - If QC columns are encoded as features, they may contain future information (e.g., `'P'` flag may indicate data is temporary and may be modified in the future)
   - This would lead to unreasonable performance improvements

### Features Actually Used in Training (Raw Track)

Training uses `select_dtypes(include=[np.number])` to **select only numeric features**, so the features actually used in training are:

1. ✅ `Hour (PST)`: Hour (numeric metadata)
2. ✅ `Jul`: Julian day (numeric metadata)
3. ✅ `ETo (mm)`: Reference evapotranspiration (numeric)
4. ✅ `Precip (mm)`: Precipitation (numeric)
5. ✅ `Sol Rad (W/sq.m)`: Solar radiation (numeric)
6. ✅ `Vap Pres (kPa)`: Vapor pressure (numeric)
7. ✅ `Air Temp (C)`: Air temperature (numeric) ✅ **Correct**: Current temperature used to predict future temperature
8. ✅ `Rel Hum (%)`: Relative humidity (numeric)
9. ✅ `Dew Point (C)`: Dew point (numeric)
10. ✅ `Wind Speed (m/s)`: Wind speed (numeric)
11. ✅ `Wind Dir (0-360)`: Wind direction (numeric)
12. ✅ `Soil Temp (C)`: Soil temperature (numeric)

**Total: 12 numeric features** ✅

**Excluded Features**:
- Non-numeric columns: `Stn Name`, `CIMIS Region` (2)
- QC columns: `qc`, `qc.1`, ..., `qc.9` (10, all non-numeric)

---

## Jul Feature Details

### What is 'Jul'?

**`Jul`** is the abbreviation for **Julian Day**, representing **the day of the year** (1-365 or 1-366).

### Feature Source

`Jul` comes from a column in the original CIMIS data, located after `Hour (PST)`, extracted directly from CIMIS data.

### Why is 'Jul' Feature Important?

#### 1. Strong Seasonality

Frost events have clear seasonal patterns:

| Season | Julian Day Range | Frost Risk |
|--------|------------------|------------|
| **Winter** | 1-59, 334-365 | ⚠️ **High** |
| **Spring** | 60-151 | ⚠️ Moderate |
| **Summer** | 152-243 | ✅ **Low** |
| **Fall** | 244-333 | ⚠️ Moderate |

#### 2. High Feature Importance

In Frost classification tasks, `Jul` is the most important feature (**20.09%**), indicating:

- ✅ **Time (season) is a key factor in predicting frost**
- ✅ **This aligns with physical intuition** (frost mainly occurs in winter)
- ✅ **Seasonal patterns are more important than individual meteorological variables**

#### 3. Difference from Other Time Features

- **`Jul` (Julian Day)**: From original data, continuous numeric, reflects gradual seasonal changes
- **`day_of_year`**: Computed from `Date` column in feature engineering, same function as `Jul` but different name
- **`month`**: Discrete value (1-12), reflects month information

---

## Feature Implementation

### Using Feature Engineering Module

```python
from src.data import DataPipeline
from pathlib import Path

# Create pipeline from config
pipeline = DataPipeline.from_config("config/pipeline/train.yaml")

# Process data
df_processed = pipeline.process(df_raw)
```

### Manual Feature Engineering

```python
from src.data.features.temporal import create_time_features
from src.data.features.lagging import create_lag_features, create_rolling_features

# Time features
df = create_time_features(df, date_col="Date")

# Lag features
df = create_lag_features(
    df,
    columns=["Air Temp (C)", "Dew Point (C)", ...],
    lags=[1, 3, 6, 12, 24],
    groupby_col="Stn Id"
)

# Rolling features
df = create_rolling_features(
    df,
    columns=["Air Temp (C)", "Dew Point (C)", ...],
    windows=[3, 6, 12, 24],
    stats=["mean", "min", "max", "std", "sum"],
    groupby_col="Stn Id"
)
```

### Important Notes

- **Station Grouping**: Lag and rolling features are calculated within each station group
- **Temporal Leakage Protection**: Use correct time ordering to calculate features
- **Missing Value Handling**: Rolling features use `min_periods=1`
- **Cyclic Encoding**: Prevent boundary discontinuities (hour 23 → 0, month 12 → 1)

---

## Feature Selection

### Cumulative Importance Analysis

Based on feature importance analysis from trained models:

| Feature Count | Cumulative Importance | Recommendation |
|---------------|----------------------|-----------------|
| Top 64 | 50.0% | Minimum viable set |
| **Top 136** | **80.0%** | **Fast training** |
| **Top 175** | **90.0%** | **⭐ Recommended** |
| **Top 206** | **95.0%** | **Best performance** |
| Top 247 | 99.0% | Near complete set |
| All 280 | 100.0% | Complete feature set |

### Recommended Options

#### Option 1: Fast Training (Top 136, 80%)

**Use Cases**: Production environment, real-time inference, resource-constrained systems

**Advantages**:
- ~50% feature count reduction
- Faster computation
- Lower memory usage
- Shorter training time

**Trade-off**: May lose 10-20% of subtle patterns

#### Option 2: Balanced (Top 175, 90%) ⭐ **Recommended**

**Use Cases**: Most scenarios, standard training

**Advantages**:
- Optimal balance of performance and efficiency
- Retains 90% importance
- Only ~37% feature count reduction
- Practical performance for most tasks

**Implementation**:
```bash
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --output-dir experiments/lightgbm_B_175f_12h
```

#### Option 3: Best Performance (Top 206, 95%)

**Use Cases**: Final model training, maximum accuracy needed

**Advantages**:
- Retains 95% importance
- Better performance than Top 175
- Still manageable feature count

#### Option 4: Complete Set (All 280)

**Use Cases**: Final evaluation, maximum accuracy, computation cost not a concern

**Advantages**:
- Maximum possible performance
- Retains all useful features

### Feature Selection Workflow

1. **Train initial model** using all features
2. **Analyze feature importance**: Check `feature_importance.csv` in model output directory
3. **Select features** based on cumulative importance:
   - Top 136 for fast training
   - Top 175 for balance (recommended)
   - Top 206 for best performance
4. **Retrain with selected features** using appropriate track (e.g., `top175_features`)

### Should Features Be Further Reduced?

**Answer**: ❌ **No, not necessary in most cases**

**Reasons**:
- **Excellent data-feature ratio**: 2.36M samples / 298 features = **7,919:1**
- **Industry standard**: > 1,000:1 is excellent (we have 7,919:1)
- **Top 175 features** already provide optimal balance

**When to Reduce**:
- Resource constraints
- Real-time inference requirements
- Need for further optimization

---

## Feature Engineering Recommendations

### Current Status Analysis

| Item | Document Description | Actual Usage | Difference |
|------|---------------------|--------------|------------|
| **Total Features** | 298 | 31 (before fix) | -267 (-89.6%) |

### Problem Diagnosis

#### 1. Configuration Issues

**Current Configuration Problems**:
- `rolling_features.functions` missing `"sum"`
- `lag_features.columns` and `rolling_features.columns` set to `null`, may only select partial columns

**Fix Solution**:
```yaml
rolling_features:
  functions: ["mean", "min", "max", "std", "sum"]  # ✅ Add "sum"
lag_features:
  columns:  # ✅ Explicitly specify
    - "Air Temp (C)"
    - "Dew Point (C)"
    # ... other variables
rolling_features:
  columns:  # ✅ Explicitly specify
    - "Air Temp (C)"
    - "Dew Point (C)"
    # ... other variables
```

#### 2. Feature Creation Order

Ensure correct feature creation order:

1. **Time features** (no dependency on other features)
2. **Station features** (no dependency on other features)
3. **Original numeric features** (no dependency on other features)
4. **Derived features** (may depend on original features)
5. **Lag features** (depend on original and derived features)
6. **Rolling features** (depend on original and derived features)
7. **Advanced features** (depend on lag and rolling features)

### Expected Feature Count After Fix

| Category | Before Fix | After Fix | Increase |
|----------|------------|-----------|----------|
| **Original Features** | 12 | 12 | 0 |
| **Time Features** | ~11 | 11 | 0 |
| **Lag Features** | ~1 | 50 | +49 |
| **Rolling Features** | ~0 | 180 | +180 |
| **Derived Features** | ~3 | 11 | +8 |
| **Other Features** | ~15 | ~84 | +69 |
| **Total** | **~31** | **~298** | **+267** |

---

## Performance Comparison

### Feature Set Comparison (Complete vs Top 175)

Based on LightGBM model evaluation:

| Horizon | ROC-AUC (Complete) | ROC-AUC (Top175) | MAE (Complete) | MAE (Top175) | R² (Complete) | R² (Top175) |
|---------|-------------------|------------------|---------------|-------------|--------------|-------------|
| 3h | 0.9965 | 0.9965 | 1.1548 | 1.1438 | 0.9698 | 0.9703 |
| 6h | 0.9928 | 0.9926 | 1.5853 | 1.5454 | 0.9458 | 0.9481 |
| 12h | 0.9892 | 0.9892 | 1.8439 | 1.7925 | 0.9270 | 0.9304 |
| 24h | 0.9827 | 0.9843 | 1.9562 | 1.9287 | 0.9171 | 0.9196 |

### Key Findings

- **ROC-AUC**: Top 175 similar or slightly better
- **MAE**: Top 175 slightly better (removes noise)
- **R²**: Top 175 slightly better
- **Conclusion**: Top 175 features provide **optimal performance** and **reduced complexity**

---

## Feature Reference

### Quick Reference Table

#### Time Features (11)
- `hour`, `hour_cos`, `hour_sin` - Hour of day
- `month`, `month_cos`, `month_sin` - Month
- `day_of_year` - Day of year
- `day_of_week` - Day of week
- `season` - Season
- `is_night` - Night indicator

#### Lag Features (51)
- Variables: Air temp, dew point, ETo, precipitation, relative humidity, soil temp, solar radiation, wind direction, wind speed, vapor pressure
- Lag values: 1h, 3h, 6h, 12h, 24h
- Format: `{variable}_lag_{lag_hours}`

#### Rolling Features (180)
- Variables: Same as lag features
- Windows: 3h, 6h, 12h, 24h
- Statistics: mean, min, max, std, sum
- Format: `{variable}_rolling_{window}h_{statistic}`

#### Station Features (24)
- `latitude`, `longitude`, `elevation`
- `station_id`
- Other geographic metadata

#### Derived Features (11)
- `temp_dew_diff` - Temperature minus dew point
- `wind_chill` - Wind chill index
- `heat_index` - Heat index
- Other computed features

### Feature Importance Insights

**Top 10 Most Important Features** (typical):
1. `Air Temp (C)_rolling_6h_mean` - Recent temperature trend
2. `Air Temp (C)_lag_1` - Immediate past temperature
3. `Dew Point (C)_rolling_6h_mean` - Recent dew point trend
4. `hour_cos` - Diurnal cycle
5. `Air Temp (C)_rolling_24h_min` - Daily minimum
6. `Dew Point (C)_lag_1` - Immediate past dew point
7. `month_cos` - Seasonal cycle
8. `Air Temp (C)_rolling_12h_mean` - Half-day trend
9. `latitude` - Geographic effect
10. `Soil Temp (C)_rolling_6h_mean` - Recent soil temperature

### Implementation Details

**Code Locations**:
- Time features: `src/data/features/temporal.py`
- Lag features: `src/data/features/lagging.py`
- Rolling features: `src/data/features/lagging.py`
- Station features: Merged from `data/external/cimis_station_metadata.csv`

**Training CLI**:
```bash
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --output-dir experiments/model_dir
```

---

## Best Practices

### 1. Feature Selection

- ✅ Use **Top 175 features** for most scenarios (recommended)
- ✅ Use **Top 136 features** for fast training/production
- ✅ Use **Top 206 features** for best performance
- ✅ Use **all features** only for final evaluation

### 2. Feature Engineering

- ✅ Always group by station for lag/rolling feature calculation
- ✅ Use cyclic encoding for time features
- ✅ Properly handle missing values
- ✅ Prevent temporal leakage

### 3. Model Training

- ✅ Train with selected feature set (track parameter)
- ✅ Use appropriate matrix cell (A/B/C/D/E)
- ✅ Validate on holdout set
- ✅ Compare different feature sets if needed

---

## Related Documentation

- **[User Guide](../guides/USER_GUIDE.md)**: User guide with training examples
- **[Feature Importance Guide](./FEATURE_IMPORTANCE.md)**: Feature importance analysis and selection guide
- **[Technical Documentation](../technical/TECHNICAL_DOCUMENTATION.md)**: Technical details and API reference
- **[Data Documentation](../technical/DATA_DOCUMENTATION.md)**: Data processing documentation

---

**Last Updated**: 2025-12-06  
**Total Features**: 298  
**Recommended Feature Set**: Top 175 (90% importance)  
**Document Version**: 3.0
