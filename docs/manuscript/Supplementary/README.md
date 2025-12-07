# Supplementary Materials

This directory contains supplementary materials for the AgriFrost-AI project.

## File List

### `supplementary_S1_feature_list.pdf`

**Supplementary Material S1: Detailed Feature List for AgriFrost-AI ABCD Feature Matrices**

This document provides a detailed list of all features used in the ABCD feature configuration matrices of the AgriFrost-AI system, including:

1. **Matrix A: Single-station + Raw Features (16 dimensions)**
   - List of 12 raw CIMIS variables (with physical meaning, units, and data sources)
   - Time harmonic encoding (4 dimensions: hour_sin/cos, month_sin/cos)

2. **Matrix B: Single-station + Engineered Features (278 dimensions)**
   - Raw variables (12 dimensions)
   - Time features (15 dimensions) detailed list
   - Lag features (50 dimensions): 10 variables × 5 lag windows
   - Rolling window statistics (180 dimensions): 9 variables × 4 windows × 5 statistics
   - Derived meteorological features (3 dimensions): including formulas and physical meaning
   - Radiation-related features (4 dimensions)
   - Wind features (6 dimensions)
   - Humidity features (4 dimensions)
   - Trend features (1 dimension)
   - Station static features (4 dimensions)
   - Feature selection strategy description (based on cumulative importance, 90% threshold corresponds to 146 features at 12-hour horizon)

3. **Matrix C: Neighbor Aggregation + Raw Features (534 dimensions)**
   - Raw variables (12 dimensions)
   - Neighbor construction method (Haversine distance formula)
   - Time alignment method
   - 8 neighbor aggregation statistics (27 variables × 8 methods = 216 dimensions, including formulas)
     - 27 variables include: 12 raw CIMIS variables + 15 time features
   - Missing mask features (293 dimensions):
     - Missing masks for neighbor aggregation features (216 dimensions)
     - Variable missing ratios (27 dimensions)
     - Missing masks for missing ratio features (27 dimensions)
     - Missing masks for other features (22 dimensions)
   - Time harmonic encoding (2 dimensions: day_of_year_sin/cos)
   - Other features (11 dimensions): time discrete features, derived meteorological features, has_neighbors indicator

4. **Matrix D: Neighbor Aggregation + Engineered Features (818 dimensions)**
   - Single-station engineered features (278 dimensions, from Matrix B)
   - Neighbor aggregation features (216 dimensions, from Matrix C)
   - Missing mask features (299 dimensions, 6 more than Matrix C, because missing masks are also generated for engineered features)
   - Other features (43 dimensions): wind features, humidity features, radiation features, trend features, station static features, geographical features, interaction features, etc.

5. **Feature Importance Analysis**
   - Matrix B (LightGBM, 12-hour horizon) Top 20 feature importance table
     - Top 10 features for frost classification task
     - Top 10 features for temperature regression task
   - Feature importance distribution and long-tail structure analysis
   - Feature importance changes across different horizons (3/6/12/24 hours)
   - Key feature category contribution analysis (time features, lag features, rolling statistics, station static features, derived meteorological features)
   - Key insights and physical meaning interpretation

6. **Feature Selection Effectiveness Validation**
   - Feature counts at different cumulative importance thresholds (80%/85%/90%/95%)
   - Feature selection performance comparison (ROC-AUC, PR-AUC, Brier Score, temperature RMSE changes)
   - Computational efficiency improvement (training time reduced by 35-40%, inference time reduced by 30-35%)

7. **Feature Generation Implementation Details**
   - Code location (`src/data/features/` directory structure)
   - Configuration examples (`config/feature_pipeline/feature_engineering.yaml`)
   - Training CLI examples

8. **Feature Usage Recommendations**
   - Matrix selection guide (choose A/B/C/D based on application scenarios)
   - Feature selection recommendations (two-stage strategy based on cumulative importance)
   - Radius selection recommendations (by horizon: 3/6h recommend 60-100 km, 12/24h recommend 160-200 km)

### `supplementary_table_S1_stations.csv`

**Supplementary Table S1: CIMIS Station Metadata**

Contains metadata for 18 CIMIS automatic weather stations:
- Station ID
- Station name
- CIMIS region
- County/City
- Longitude and latitude
- Elevation
- GroundCover
- Start/end dates
- Whether it is an ETo station

This table is used for:
- Station filtering during spatial aggregation
- Station grouping during LOSO (Leave-One-Station-Out) evaluation
- Station location for result interpretation

### `supplementary_table_S2_all_experiments.csv`

**Supplementary Table S2: Complete Performance Metrics for All Experiment Configurations**

Contains complete performance metrics for all experiment configuration combinations, where each record corresponds to a unique model-matrix-horizon-radius configuration. Main fields include:

- **Model identifier**: `model` (LightGBM, XGBoost, CatBoost, Random Forest, GRU, LSTM, TCN)
- **Feature matrix**: `matrix_cell` (A, B, C, D)
- **Forecast horizon**: `horizon_h` (3, 6, 12, 24 hours)
- **Spatial aggregation radius**: `radius_km` (0.0 indicates single-station configuration, 20-200 indicates multi-station configuration)
- **Classification performance metrics**: `roc_auc`, `pr_auc`, `brier_score`, `f1_score`, `precision`, `recall`
- **Regression performance metrics**: `temp_rmse`, `temp_mae`, `temp_r2`
- **Experiment path**: `path` (points to the specific experiment directory)

This table is used for:
- Cross-model, cross-matrix, cross-horizon performance comparison analysis
- Spatial aggregation radius sensitivity analysis
- Model selection and hyperparameter optimization
- Result reproduction and validation

### `supplementary_table_S3_best_configurations.csv`

**Supplementary Table S3: Optimal Configurations for Each Feature Matrix and Horizon**

Contains the "representative optimal configuration" selected based on ROC-AUC and Brier Score for each feature matrix and each horizon. Main fields include:

- **Feature matrix**: `matrix_cell` (A, B, C, D)
- **Forecast horizon**: `horizon_h` (3, 6, 12, 24 hours)
- **Optimal model**: `model`
- **Optimal radius**: `radius_km` (only for matrices C/D)
- **Performance metrics**: `roc_auc`, `pr_auc`, `brier_score`, `temp_rmse`, `temp_mae`

This table is used for:
- Quick lookup of optimal configuration for specific matrix and horizon
- Analysis of horizon dependency of optimal radius
- Model selection decision support

### `supplementary_table_S4_matrix_summary.csv`

**Supplementary Table S4: Statistical Summary Aggregated by Matrix and Horizon**

Contains statistical summaries aggregated by feature matrix and horizon, including mean, maximum, minimum, etc. Main fields include:

- **Feature matrix**: `matrix_cell` (A, B, C, D)
- **Forecast horizon**: `horizon_h` (3, 6, 12, 24 hours)
- **Statistical metrics**: mean, maximum, minimum for each performance metric

This table is used for:
- Overall comparison of matrix performance
- Analysis of horizon impact on performance
- Statistical feature analysis of performance distribution

### `supplementary_table_S5_feature_category_importance.csv`

**Supplementary Table S5: Cumulative Importance of Feature Categories Across Horizons**

Contains feature category contribution analysis results for LightGBM model on Matrix B (single-station + engineered features, 278 dimensions). Main fields include:

- **Forecast horizon**: `horizon_h` (3, 6, 12, 24 hours)
- **Task type**: `task` (frost_classification, temperature_regression)
- **Feature category**: `category` (Rolling Statistics, Lag Features, Time Features, Station Features, Derived Meteorological, Wind Features, Soil Features, Other Features)
- **Cumulative importance percentage**: `cumulative_importance_pct` (cumulative importance of all features in this category)
- **Feature count**: `feature_count` (number of features in this category)

This table is used for:
- Understanding the contribution of different feature categories to model performance
- Analyzing horizon dependency of feature category importance
- Guiding feature engineering direction (prioritize high-importance categories)
- Formulating feature selection strategies (hierarchical selection based on category contribution)

### `supplementary_table_S6_top_features_by_category.csv`

**Supplementary Table S6: Importance of Top Features in Each Category Across Horizons**

Contains importance analysis results for Top-2 features in each feature category for LightGBM model on Matrix B (single-station + engineered features, 278 dimensions). Main fields include:

- **Forecast horizon**: `horizon_h` (3, 6, 12, 24 hours)
- **Task type**: `task` (frost_classification, temperature_regression)
- **Feature category**: `category` (Rolling Statistics, Lag Features, Time Features, Station Features, Derived Meteorological, Wind Features, Soil Features, Other Features)
- **Feature name**: `feature_name` (specific feature name)
- **Importance percentage**: `importance_pct` (importance percentage of this feature)
- **Rank within category**: `rank_in_category` (rank of this feature within its category, 1 or 2)

This table is used for:
- Identifying the most important specific features in each feature category
- Analyzing importance changes of key features across different horizons
- Understanding representative features of categories (e.g., `rolling_24h_mean` in rolling statistics, `lag_12` in lag features)
- Providing fine-grained guidance for feature selection (prioritize Top features within categories)

## Experiment Scale

- **Total experiments**: 471 reproducible experiments
- **Feature matrices**: 4 (A, B, C, D)
- **Forecast horizons**: 4 (3, 6, 12, 24 hours)
- **Model families**: 7 (LightGBM, XGBoost, CatBoost, Random Forest, GRU, LSTM, TCN)
- **Spatial radius range**: 0-200 km (step size 20 km)

## Key Experimental Results

### Best Configuration Performance (Matrix C + LightGBM)

- **3-hour horizon (60 km radius)**: ROC-AUC 0.9972, PR-AUC 0.7242, Temperature RMSE 1.58 °C
- **6-hour horizon (160 km radius)**: ROC-AUC 0.9943, PR-AUC 0.5871, Temperature RMSE 2.05 °C
- **12-hour horizon (200 km radius)**: ROC-AUC 0.9901, PR-AUC 0.4914, Temperature RMSE 2.42 °C
- **24-hour horizon (180 km radius)**: ROC-AUC 0.9877, PR-AUC 0.4671, Temperature RMSE 2.39 °C

### Feature Selection Effectiveness (Matrix B, LightGBM, 12-hour horizon)

- **90% cumulative importance threshold**: 146 features (47.5% compression rate)
- **Performance preservation**: ROC-AUC change < 0.01%, PR-AUC improvement 2.6%
- **Computational efficiency**: Training time reduced by 35-40%, inference time reduced by 30-35%

## Usage Instructions

- **Reproduce experiments**: Refer to feature list and implementation details sections, use unified CLI interface
- **Feature extension**: Refer to feature generation methods and formulas (Section 4.3)
- **Result interpretation**: Refer to feature importance analysis and key insights sections (Section 4.7)
- **Model deployment**: Refer to feature usage recommendations and matrix selection guide (Section 6)
- **Decision support**: Refer to Precision/Recall vs Threshold curves (Section 6)

## Related Documents

- **Main document**: `../frost_risk_progress_cn.pdf` - Complete manuscript document
- **Feature engineering guide**: `../../features/FEATURE_GUIDE.md` - Detailed feature engineering guide
- **Technical documentation**: `../../technical/TECHNICAL_DOCUMENTATION.md` - Technical implementation details

## File Naming Convention

- **Supplementary Material S1**: `supplementary_S1_feature_list.pdf` - Detailed feature list for ABCD feature configuration matrices
- **Supplementary Table S1**: `supplementary_table_S1_stations.csv` - CIMIS station metadata table
- **Supplementary Table S2**: `supplementary_table_S2_all_experiments.csv` - Complete performance metrics for all experiment configuration combinations
- **Supplementary Table S3**: `supplementary_table_S3_best_configurations.csv` - Optimal configurations for each feature matrix and horizon
- **Supplementary Table S4**: `supplementary_table_S4_matrix_summary.csv` - Statistical summary aggregated by matrix and horizon
- **Supplementary Table S5**: `supplementary_table_S5_feature_category_importance.csv` - Cumulative importance of feature categories across horizons
- **Supplementary Table S6**: `supplementary_table_S6_top_features_by_category.csv` - Importance of Top features in each category across horizons

**Note**:
- Supplementary Tables S2--S4 are core data files for experimental results, containing complete performance metrics for all experiments. These files fully correspond to the results section (Section 5) in the main document and can be used for result reproduction, in-depth analysis, and secondary research.
- Supplementary Tables S5--S6 are core data files for feature importance analysis, containing feature category contribution analysis and key feature identification results. These files fully correspond to the feature selection and feature importance analysis section (Section 5.8) in the main document and can be used for in-depth understanding of feature engineering effectiveness and feature selection guidance.

## Last Updated

2025-12-06

## Version History

- **v1.0 (2025-12-06)**: Initial version, including complete ABCD matrix feature list and station metadata table
- **v1.1 (2025-12-06)**: Added Supplementary Tables S5 (feature category importance) and S6 (Top features by category), supporting in-depth understanding of feature selection and feature importance analysis
