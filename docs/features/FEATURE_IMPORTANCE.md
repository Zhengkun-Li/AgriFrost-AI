# AgriFrost-AI Complete Feature Importance Guide

<div align="center">

<img src="../logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

</div>

**Last Updated**: 2025-12-06

This document integrates all feature importance related content including evaluation methods, model-specific explanations, representation method selection, and feature selection strategies, providing a one-stop reference for feature importance analysis.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Feature Importance Evaluation Methods](#feature-importance-evaluation-methods)
3. [Feature Importance Visualization](#feature-importance-visualization)
4. [Nature of Feature Importance](#nature-of-feature-importance)
5. [Feature Importance Representation Methods](#feature-importance-representation-methods)
6. [Feature Selection Strategies](#feature-selection-strategies)
7. [Analysis Methods](#analysis-methods)
8. [Important Notes](#important-notes)

---

## Overview

Feature Importance is a key tool for understanding model decision processes. This guide explains how to extract, analyze, and visualize feature importance from trained models, and guides feature selection strategies.

### ⚠️ **Important Concept**

**Feature importance is model-specific, not dataset-specific**

Feature importance reflects **how the model uses features for prediction**, not the inherent importance of features in the dataset.

---

## Feature Importance Evaluation Methods

### 1. **Automatic Saving (During Training)**

After training completes, feature importance is automatically saved to the model directory:

```
experiments/lightgbm/raw/A/full_training/full_training/horizon_12h/
  ├── frost_feature_importance.csv    # Feature importance for frost classification model
  └── temp_feature_importance.csv     # Feature importance for temperature regression model
```

**CSV Format**:
```csv
feature,importance,importance_pct,cumulative_pct
Air Temp (C),1234.56,15.23,15.23
Dew Point (C),987.65,12.18,27.41
Soil Temp (C),876.54,10.81,38.22
...
```

**Column Descriptions**:
- `feature`: Feature name
- `importance`: Raw importance score
- `importance_pct`: Importance percentage
- `cumulative_pct`: Cumulative importance percentage

### 2. **Using CLI Command for Analysis**

Use `analysis feature-importance` command to extract and analyze feature importance:

```bash
# Analyze both frost and temp models
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h

# Analyze only frost classification model
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h \
    --task frost

# Save to specified directory and generate plots
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h \
    --output-dir results/feature_importance \
    --plot \
    --top-k 20
```

**Parameter Descriptions**:
- `--model-dir`: Path to trained model directory
- `--task`: Task to analyze (`frost`, `temp`, or `both`)
- `--output-dir`: Output directory (default: `model_dir/feature_importance`)
- `--top-k`: Only show top K most important features
- `--plot`: Generate feature importance plots
- `--format`: Output format (`csv`, `json`, or `both`)

**Output**:
- CSV/JSON files: Feature importance data
- PNG plots: Feature importance visualization (both percentage and raw value formats)
- Comparison plots: Frost vs temp feature importance comparison (if both tasks analyzed)

---

## Feature Importance Visualization

### 1. **Feature Importance for Single Model**

```python
from pathlib import Path
import pandas as pd
from src.visualization.plots import Plotter

# Read feature importance data
importance_df = pd.read_csv("experiments/lightgbm/raw/A/full_training/full_training/horizon_12h/frost_feature_importance.csv")

# Create plot (percentage)
plotter = Plotter(style="matplotlib", figsize=(12, 8))
plotter.plot_feature_importance(
    importance_df,
    top_n=20,
    title="Feature Importance - Frost Classification (12h) (%)",
    save_path="feature_importance_pct.png",
    show=False,
    importance_col='importance_pct',
    xlabel='Importance (%)'
)

# Create plot (raw values)
plotter.plot_feature_importance(
    importance_df,
    top_n=20,
    title="Feature Importance - Frost Classification (12h) (Raw Values)",
    save_path="feature_importance_raw.png",
    show=False,
    importance_col='importance',
    xlabel='Importance (Raw Value)'
)
```

### 2. **Compare Frost vs Temp Feature Importance**

Use CLI command to automatically generate comparison plots:

```bash
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h \
    --task both \
    --plot
```

This will generate in the output directory:
- `frost_feature_importance_pct.png`: Feature importance for Frost classification model (percentage)
- `frost_feature_importance_raw.png`: Feature importance for Frost classification model (raw values)
- `temp_feature_importance_pct.png`: Feature importance for Temp regression model (percentage)
- `temp_feature_importance_raw.png`: Feature importance for Temp regression model (raw values)
- `frost_temp_importance_comparison_pct.png`: Comparison plot (percentage)
- `frost_temp_importance_comparison_raw.png`: Comparison plot (raw values)

---

## Nature of Feature Importance

### 1. **Model-Specific**

Feature importance depends on:
- **Model type** (LightGBM, XGBoost, Linear, etc.)
- **Model parameters** (hyperparameter settings)
- **Training process** (training data, training strategy)
- **Whether model has been trained (fitted)**

### 2. **Why is it Model-Specific?**

#### **Tree-based Models (LightGBM, XGBoost, RandomForest)**

```python
# LightGBM feature importance example
# Importance = Frequency of feature use in decision trees × Information gain brought

Feature Importance = Σ(Information gain from using this feature at each node)
```

- Different tree structures → Different feature importance
- Different hyperparameters (e.g., `max_depth`, `num_leaves`) → Different tree structures → Different feature importance
- Different training data → Different tree structures → Different feature importance

#### **Linear Models (Linear Regression, Logistic Regression)**

```python
# Linear model feature importance
# Importance = |coefficient| (coefficient magnitude)

Feature Importance = |coefficient|
```

- Different model training results → Different coefficients → Different feature importance
- Correlation between features affects coefficient magnitude

#### **Deep Learning Models (LSTM, GRU, TCN)**

- Usually do not directly provide feature importance
- If using attention mechanism, can use attention weights as importance
- Need to use alternative methods like permutation importance

### 3. **Practical Examples**

#### **Scenario: Same Dataset, Different Models**

Assume we have the same dataset, trained three different models:

| Feature | LightGBM Importance | XGBoost Importance | Linear Importance |
|---------|---------------------|-------------------|-------------------|
| Air Temp (C) | 20.09% | 18.5% | 35.2% |
| Dew Point (C) | 13.03% | 14.2% | 22.1% |
| Soil Temp (C) | 11.77% | 12.8% | 15.3% |

**Why Different?**
- LightGBM uses gradient boosting, feature importance based on information gain
- XGBoost uses different optimization algorithm, may generate different tree structures
- Linear Regression uses coefficients, affected by feature correlation and standardization

#### **Scenario: Same Model, Different Horizons**

```python
# Model 1: LightGBM (3h horizon)
# Model 2: LightGBM (12h horizon)
# Model 3: LightGBM (24h horizon)
```

**Results**:
- 3h horizon: May rely more on current moment features (e.g., Air Temp)
- 24h horizon: May rely more on trend features (e.g., Hour, Julian Day)

**Feature importance changes with horizon!**

### 4. **Dataset-Level Feature Importance (Alternative Methods)**

If you want to obtain **dataset-level feature importance** (not dependent on specific model), you can use:

#### **Permutation Importance**

```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance on test set
perm_importance = permutation_importance(
    model, 
    X_test, 
    y_test, 
    n_repeats=10, 
    random_state=42
)
```

**Characteristics**:
- Based on model performance change
- Not dependent on model internal structure
- Can compare across models
- Higher computational cost

#### **SHAP Values**

```python
import shap

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

**Characteristics**:
- Based on game theory
- Can explain individual sample predictions
- Can be visualized
- Higher computational cost

#### **Feature Correlation Analysis**

```python
# Calculate correlation between features and target
correlations = df.corr()[target_column].sort_values(ascending=False)
```

**Characteristics**:
- Based on dataset itself
- Not dependent on model
- Only reflects linear relationships
- Does not consider feature interactions

---

## Feature Importance Representation Methods

### **Percentage (Percentage) More Common and Recommended**

In machine learning and data science, **percentage** is the more common and recommended way to represent feature importance.

### **Why is Percentage More Common?**

#### **1. Easy to Understand and Interpret**

**Percentage**:
```
Feature Importance:
- Air Temp (C): 20.09%
- Soil Temp (C): 13.03%
- Wind Speed (m/s): 8.44%
```

✅ **Advantages**:
- Intuitive: 20.09% means this feature contributes about 1/5 of total importance
- Easy to compare: Can directly see which feature is more important
- Not dependent on value range: Not affected by model type or hyperparameters

**Raw Values**:
```
Feature Importance:
- Air Temp (C): 2487.0
- Soil Temp (C): 1613.0
- Wind Speed (m/s): 907.0
```

❌ **Disadvantages**:
- Value range may be large, hard to understand (what does 2487.0 mean?)
- Different model types may have very different value ranges
- Difficult to compare intuitively

### **Recommendations for Different Scenarios**

| Scenario | Recommended Use | Reason |
|----------|----------------|--------|
| **Visualization Plots** | Percentage | More intuitive, easy to understand |
| **CSV Files** | Keep both | Meet different needs |
| **Papers and Reports** | Percentage | More professional, standardized |
| **Technical Documentation** | Provide both | Detailed and complete |
| **Cross-Model Comparison** | Percentage or normalized values | Unified standard |

### **Current Implementation Recommendations**

**Current CSV Format (Recommended)**:

```csv
feature,importance,importance_pct,cumulative_pct
Air Temp (C),2487.0,20.09,20.09
Soil Temp (C),1613.0,13.03,33.12
Wind Speed (m/s),907.0,7.33,40.45
```

**Advantages**:
- ✅ Retains raw values (for deep analysis)
- ✅ Provides percentage (for understanding and visualization)
- ✅ Provides cumulative percentage (for feature selection)

**Visualization Plots**:
- Generate plots in both formats: percentage and raw values
- Save separately as `_pct.png` and `_raw.png`

---

## Feature Selection Strategies

### **Two-Stage Feature Selection Method**

#### **Stage 1: Full Feature Training (Baseline)**

1. **Create all features** (~298)
   - Use complete feature engineering configuration
   - Ensure all features are created
   - Verify feature count meets expectations

2. **Train model, obtain baseline performance**
   - Train model using all features
   - Record performance metrics (ROC-AUC, PR-AUC, MAE, RMSE, R²)
   - Use as baseline for subsequent optimization

3. **Analyze feature importance**
   - Extract feature importance
   - Calculate cumulative importance
   - Identify most important features

#### **Stage 2: Retrain Based on Importance (Optimization)**

1. **Select features with cumulative importance of 90%**
   - Based on feature importance analysis results
   - Select features reaching 90% cumulative importance
   - May only need top 50-200 features (depending on importance distribution)

2. **Retrain using these features**
   - Retrain model using selected features
   - Compare performance improvement or decline
   - Evaluate computation cost reduction

3. **Optimize performance and cost**
   - Balance feature count and performance
   - If performance decline is not significant, keep simplified feature set
   - If performance decline is significant, adjust threshold (e.g., use 95% instead of 90%)

### **Strategy Advantages**

1. ✅ **Data-driven**: Based on actual feature importance, not guessing
2. ✅ **Performance optimization**: Retain most important features, may improve model performance (remove noise features)
3. ✅ **Cost optimization**: Reduce feature count, lower computation cost, faster training and inference time
4. ✅ **Interpretability**: Understand which features are most important, understand model decision basis
5. ✅ **Flexibility**: Can adjust feature count based on threshold, support progressive optimization

### **Implementation Steps**

#### **Step 1: Train Model Using All Features**

```bash
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track feature_engineering \
    --horizon-h 12 \
    --config config/pipeline/train_with_loso.yaml
```

#### **Step 2: Analyze Feature Importance**

```bash
# Analyze feature importance for Frost classification task
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/feature_engineering/B/full_training/full_training/horizon_12h \
    --task frost \
    --plot

# Analyze feature importance for Temp regression task
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/feature_engineering/B/full_training/full_training/horizon_12h \
    --task temp \
    --plot
```

#### **Step 3: Calculate Cumulative Importance**

```python
import pandas as pd

# Read feature importance
importance_df = pd.read_csv(
    "experiments/.../horizon_12h/feature_importance/frost_feature_importance.csv"
)

# Calculate cumulative importance
importance_df = importance_df.sort_values('importance', ascending=False)
importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
importance_df['cumulative_pct'] = (
    importance_df['cumulative_importance'] / 
    importance_df['cumulative_importance'].max() * 100
)

# Find features with cumulative importance of 90%
top_90_features = importance_df[
    importance_df['cumulative_pct'] <= 90
]['feature'].tolist()

print(f"Number of features with 90% cumulative importance: {len(top_90_features)}")
print(f"Top 10 features: {top_90_features[:10]}")
```

#### **Step 4: Retrain Using Selected Features**

```bash
# Use --feature-selection-name parameter to specify feature selection name
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track feature_engineering \
    --horizon-h 12 \
    --feature-selection-name top90 \
    --config config/pipeline/train_with_loso.yaml
```

### **Expected Effects**

#### **Stage 1: Full Feature Training**

| Metric | Expected |
|--------|----------|
| **Feature Count** | ~298 |
| **Training Time** | Longer (may need 15-20 minutes) |
| **Performance** | Baseline performance |
| **Feature Importance** | Complete feature importance distribution |

#### **Stage 2: Retrain Based on Importance**

| Metric | Expected |
|--------|----------|
| **Feature Count** | ~50-200 (depending on importance distribution) |
| **Training Time** | Shorter (may reduce 50-70%) |
| **Performance** | May improve (remove noise features) or maintain |
| **Computation Cost** | Significantly reduced |

### **Trade-off Analysis**

#### **Advantages**
1. ✅ Data-driven: Based on actual importance, not guessing
2. ✅ Performance optimization: Remove noise features, may improve performance
3. ✅ Cost optimization: Reduce feature count, lower computation cost
4. ✅ Interpretability: Understand which features are most important

#### **Risks**
1. ⚠️ **Feature Interactions**: Some features may not be important alone, but important in combination
2. ⚠️ **Threshold Selection**: 90% threshold may need adjustment (e.g., 85% or 95%)
3. ⚠️ **Task Differences**: Frost and Temp tasks may need different feature sets

#### **Recommendations**
1. ✅ **Try Multiple Thresholds**: 85%, 90%, 95%
2. ✅ **Handle Separately**: Frost and Temp tasks use different feature sets
3. ✅ **Validate Performance**: Ensure simplified feature set does not significantly reduce performance

---

## Analysis Methods

### 1. **View Top-K Features**

```python
import pandas as pd

# Read feature importance
importance_df = pd.read_csv("frost_feature_importance.csv")

# View Top 10 features
top_10 = importance_df.head(10)
print(top_10[['feature', 'importance_pct', 'cumulative_pct']])
```

### 2. **Calculate Feature Coverage**

```python
# Calculate how many features needed to cover 80% importance
coverage_80 = importance_df[importance_df['cumulative_pct'] <= 80]
print(f"Need {len(coverage_80)} features to cover 80% importance")
```

### 3. **Identify Key Features**

```python
# Identify features with importance >= 5%
key_features = importance_df[importance_df['importance_pct'] >= 5]
print(f"Key features (importance >= 5%): {list(key_features['feature'])}")
```

### 4. **Cross-Horizon Comparison**

Compare feature importance across different prediction horizons:

```python
import pandas as pd
import matplotlib.pyplot as plt

horizons = [3, 6, 12, 24]
importance_by_horizon = {}

for h in horizons:
    path = f"experiments/lightgbm/raw/A/full_training/full_training/horizon_{h}h/frost_feature_importance.csv"
    if Path(path).exists():
        df = pd.read_csv(path)
        importance_by_horizon[h] = df.set_index('feature')['importance_pct']

# Merge data
combined = pd.DataFrame(importance_by_horizon)

# Visualize
combined.plot(kind='bar', figsize=(14, 8))
plt.title('Feature Importance Across Horizons')
plt.xlabel('Feature')
plt.ylabel('Importance (%)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Horizon (hours)')
plt.tight_layout()
plt.savefig('feature_importance_across_horizons.png', dpi=300)
```

### 5. **Cross-Model Comparison**

Compare feature importance across different model types:

```python
models = ['lightgbm', 'xgboost', 'catboost']
importance_by_model = {}

for model in models:
    path = f"experiments/{model}/raw/A/full_training/full_training/horizon_12h/frost_feature_importance.csv"
    if Path(path).exists():
        df = pd.read_csv(path)
        importance_by_model[model] = df.set_index('feature')['importance_pct']

# Merge and visualize
combined = pd.DataFrame(importance_by_model)
combined.plot(kind='bar', figsize=(14, 8))
plt.title('Feature Importance Across Models')
plt.xlabel('Feature')
plt.ylabel('Importance (%)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('feature_importance_across_models.png', dpi=300)
```

---

## Important Notes

### 1. **Model Type Limitations**

- **Tree-based models**: Provide native feature importance
- **Linear models**: Provide coefficients as importance
- **Deep learning models**: Do not provide direct feature importance (need to use alternative methods)

### 2. **Correlation vs Causality**

Feature importance only reflects correlation, not necessarily causality.

### 3. **Feature Interactions**

Tree-based models automatically capture feature interactions, but importance scores may not directly reflect interaction effects.

### 4. **Data Leakage Check**

If a feature has unusually high importance, check for data leakage (e.g., label column mistakenly used as feature).

### 5. **Limitations of Feature Importance**

- Feature importance is **model-specific**, different models may have different importance
- Feature importance may **change with training**, different hyperparameters or training data will lead to different importance
- If dataset-level feature importance is needed, use Permutation Importance, SHAP Values, or correlation analysis

---

## Related Documentation

- **[Feature Engineering Guide](./FEATURE_GUIDE.md)**: Complete feature engineering guide
- **[Training Guide](../training/TRAINING_GUIDE.md)**: Training and evaluation guide
- **[Models Guide](../models/MODELS_GUIDE.md)**: Detailed model descriptions
- **[Experiment Analysis Reports](./experiments/)**: Feature importance experiment analysis

---

## Related Commands

```bash
# Train model (automatically saves feature importance)
python -m src.cli train single --model-name lightgbm --matrix-cell A --track raw --horizon-h 12

# Analyze feature importance
python -m src.cli analysis feature-importance --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h

# Evaluate model performance
python -m src.cli evaluate model --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h
```

---

**Last Updated**: 2025-12-06  
**Document Version**: 3.0
