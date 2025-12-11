#!/usr/bin/env python3
"""Collect all feature importance data from LightGBM ABC experiments and save to supplementary directory."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
SUPP_DIR = PROJECT_ROOT / "docs/manuscript/Supplementary_lighgbm_abc"
SUPP_DIR.mkdir(parents=True, exist_ok=True)

# Cumulative importance threshold (90% as used in the paper)
CUMULATIVE_THRESHOLD = 90.0

# Optimal radius for Matrix C
OPTIMAL_RADIUS = {
    3: 60,   # 3h: 60km
    6: 100,  # 6h: 100km
    12: 200, # 12h: 200km
    24: 200  # 24h: 200km
}

def categorize_feature_matrix_a(feature_name: str) -> str:
    """Categorize a feature for Matrix A (raw features only)."""
    feature_lower = feature_name.lower()
    
    # All features in Matrix A are raw CIMIS variables
    return "Raw Features"

def categorize_feature_matrix_b(feature_name: str) -> str:
    """Categorize a feature for Matrix B based on its name."""
    feature_lower = feature_name.lower()
    
    # Rolling statistics
    if any(x in feature_lower for x in ['rolling', 'rolling_', '_rolling']):
        return "Rolling Statistics"
    
    # Lag features
    if any(x in feature_lower for x in ['lag_', '_lag']):
        return "Lag Features"
    
    # Time features
    if any(x in feature_lower for x in ['hour', 'day_of_year', 'jul', 'day_of_week', 'month', 'season', 
                                         'is_night', 'day_progress', 'day_of_year_sin', 'day_of_year_cos', 
                                         'hour_sin', 'hour_cos', 'month_sin', 'month_cos']):
        return "Time Features"
    
    # Wind features
    if any(x in feature_lower for x in ['wind_dir', 'wind_speed', 'wind_chill', 'calm_wind']):
        return "Wind Features"
    
    # Soil features
    if any(x in feature_lower for x in ['soil_temp', 'soil_temp_', 'soil_air_temp']):
        return "Soil Features"
    
    # Derived meteorological features
    if any(x in feature_lower for x in ['heat_index', 'temp_dew_diff', 'temp_change_rate', 'vapor_pressure', 
                                         'temp_decline_rate', 'soil_air_temp_diff']):
        return "Derived Met."
    
    # Station features
    if any(x in feature_lower for x in ['stn_id', 'station', 'region', 'region_encoded']):
        return "Station Features"
    
    # Other features (humidity, raw CIMIS variables not in above categories)
    return "Other Features"

def categorize_feature_matrix_c(feature_name: str) -> str:
    """Categorize a feature for Matrix C based on its name."""
    feature_lower = feature_name.lower()
    
    # Mask features (must check first)
    if any(x in feature_lower for x in ['missing', 'mask', '_mask', 'has_neighbors', 'neighbor_missing']):
        return "Mask Features"
    
    # Spatial aggregation features (neighbor statistics)
    if any(x in feature_lower for x in ['neighbor_mean', 'neighbor_std', 'neighbor_min', 'neighbor_max', 
                                         'neighbor_median', 'neighbor_gradient', 'neighbor_range', 
                                         'neighbor_distance_weighted', 'neighbor_']):
        return "Spatial Features"
    
    # Time features
    if any(x in feature_lower for x in ['hour', 'day_of_year', 'jul', 'day_of_week', 'month', 'season', 
                                         'is_night', 'day_progress', 'frost_season', 'day_of_year_sin', 
                                         'day_of_year_cos', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']):
        return "Engineering Features"
    
    # Derived meteorological features
    if any(x in feature_lower for x in ['heat_index', 'temp_dew_diff', 'temp_change_rate', 'vapor_pressure', 
                                         'temp_decline_rate', 'wind_chill', 'soil_air_temp_diff']):
        return "Engineering Features"
    
    # Raw CIMIS variables (not spatial, not mask, not engineering)
    raw_vars = ['air temp', 'dew point', 'rel hum', 'wind speed', 'wind dir', 'sol rad', 
                'soil temp', 'vap pres', 'precip', 'et', 'elevation']
    if any(var in feature_lower for var in raw_vars) and 'neighbor' not in feature_lower:
        return "Raw Features"
    
    # Other features
    return "Other Features"

def load_feature_importance(importance_path: Path) -> pd.DataFrame:
    """Load feature importance from CSV file."""
    if not importance_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(importance_path)
        if 'importance' in df.columns and 'feature' in df.columns:
            # Calculate cumulative percentage if not present
            if 'cumulative_pct' not in df.columns:
                df = df.sort_values('importance', ascending=False).reset_index(drop=True)
                df['importance_pct'] = (df['importance'] / df['importance'].sum() * 100)
                df['cumulative_pct'] = df['importance_pct'].cumsum()
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {importance_path}: {e}")
        return pd.DataFrame()

def collect_individual_feature_importance():
    """Collect all individual feature importance data."""
    experiments_root = PROJECT_ROOT / "experiments/lightgbm"
    all_data = []
    
    # Matrix A - balanced training only
    matrix_a_dir = experiments_root / "raw/A/full_training_balance/full_training"
    for horizon_dir in sorted(matrix_a_dir.glob("horizon_*")):
        horizon = int(horizon_dir.name.replace("horizon_", "").replace("h", ""))
        
        for task in ["frost_classification", "temperature_regression"]:
            # File naming: 'frost' for classification, 'temp' for regression
            if task == "temperature_regression":
                task_prefix = "temp"
            else:
                task_prefix = task.split('_')[0]  # 'frost'
            importance_file = horizon_dir / f"{task_prefix}_feature_importance.csv"
            
            df = load_feature_importance(importance_file)
            if not df.empty:
                df['Matrix'] = 'A'
                df['Radius_km'] = '--'
                df['Horizon_h'] = horizon
                df['Training'] = 'Balanced'
                df['Task'] = task
                all_data.append(df)
    
    # Matrix B - balanced training only
    matrix_b_dir = experiments_root / "feature_engineering/B/full_training_balance/full_training"
    if not matrix_b_dir.exists():
        matrix_b_dir = experiments_root / "raw/B/full_training_balance/full_training"
    
    if matrix_b_dir.exists():
        for horizon_dir in sorted(matrix_b_dir.glob("horizon_*")):
            horizon = int(horizon_dir.name.replace("horizon_", "").replace("h", ""))
            
            for task in ["frost_classification", "temperature_regression"]:
                # File naming: 'frost' for classification, 'temp' for regression
                if task == "temperature_regression":
                    task_prefix = "temp"
                else:
                    task_prefix = task.split('_')[0]  # 'frost'
                importance_file = horizon_dir / f"{task_prefix}_feature_importance.csv"
                
                df = load_feature_importance(importance_file)
                if not df.empty:
                    df['Matrix'] = 'B'
                    df['Radius_km'] = '--'
                    df['Horizon_h'] = horizon
                    df['Training'] = 'Balanced'
                    df['Task'] = task
                    all_data.append(df)
    
    # Matrix C - balanced training, optimal radius for each horizon
    matrix_c_base = experiments_root / "raw/C"
    for horizon in [3, 6, 12, 24]:
        radius = OPTIMAL_RADIUS[horizon]
        radius_dir = matrix_c_base / f"full_training_balance_{radius}km/full_training"
        horizon_dir = radius_dir / f"horizon_{horizon}h"
        
        if horizon_dir.exists():
            for task in ["frost_classification", "temperature_regression"]:
                # File naming: 'frost' for classification, 'temp' for regression
                if task == "temperature_regression":
                    task_prefix = "temp"
                else:
                    task_prefix = task.split('_')[0]  # 'frost'
                importance_file = horizon_dir / f"{task_prefix}_feature_importance.csv"
                
                df = load_feature_importance(importance_file)
                if not df.empty:
                    df['Matrix'] = 'C'
                    df['Radius_km'] = str(radius)
                    df['Horizon_h'] = horizon
                    df['Training'] = 'Balanced'
                    df['Task'] = task
                    all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # Reorder columns
        cols = ['Matrix', 'Radius_km', 'Horizon_h', 'Training', 'Task', 'feature', 
                'importance', 'importance_pct', 'cumulative_pct']
        if 'importance_pct' not in combined_df.columns:
            combined_df['importance_pct'] = combined_df.groupby(['Matrix', 'Horizon_h', 'Task'])['importance'].transform(
                lambda x: x / x.sum() * 100
            )
        if 'cumulative_pct' not in combined_df.columns:
            combined_df = combined_df.sort_values(['Matrix', 'Horizon_h', 'Task', 'importance'], ascending=[True, True, True, False])
            combined_df['cumulative_pct'] = combined_df.groupby(['Matrix', 'Horizon_h', 'Task'])['importance_pct'].cumsum()
        
        return combined_df[cols]
    return pd.DataFrame()

def calculate_category_importance(df: pd.DataFrame, categorize_func, use_threshold: bool = True) -> pd.DataFrame:
    """Calculate category importance.
    
    Args:
        df: DataFrame with individual feature importance
        categorize_func: Function to categorize features
        use_threshold: If True, use 90% cumulative threshold; If False, use all features
    """
    results = []
    
    for (matrix, horizon, task), group_df in df.groupby(['Matrix', 'Horizon_h', 'Task']):
        group_df = group_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative percentage
        group_df['importance_pct'] = (group_df['importance'] / group_df['importance'].sum() * 100)
        group_df['cumulative_pct'] = group_df['importance_pct'].cumsum()
        
        # Categorize all features
        group_df['Category'] = group_df['feature'].apply(categorize_func)
        
        if use_threshold:
            # Select features up to cumulative threshold
            last_valid_pos = None
            for pos in range(len(group_df)):
                if group_df.iloc[pos]['cumulative_pct'] <= CUMULATIVE_THRESHOLD:
                    last_valid_pos = pos
                else:
                    break
            
            if last_valid_pos is not None:
                next_pos = last_valid_pos + 1
                if next_pos < len(group_df):
                    selected_df = group_df.iloc[:next_pos+1].copy()
                else:
                    selected_df = group_df.copy()
            else:
                selected_df = group_df.head(1).copy()
            
            selected_total_importance = selected_df['importance'].sum()
            category_importance = selected_df.groupby('Category')['importance'].sum().reset_index()
            category_importance['importance_pct'] = (category_importance['importance'] / selected_total_importance * 100)
            category_importance['selected_features_count'] = selected_df.groupby('Category')['feature'].count().values
            category_importance['Cumulative_threshold'] = selected_df['cumulative_pct'].iloc[-1]
            category_importance['Selected_features_total'] = len(selected_df)
        else:
            # Use all features
            total_importance = group_df['importance'].sum()
            category_importance = group_df.groupby('Category')['importance'].sum().reset_index()
            category_importance['importance_pct'] = (category_importance['importance'] / total_importance * 100)
            category_importance['selected_features_count'] = group_df.groupby('Category')['feature'].count().values
            category_importance['Cumulative_threshold'] = 100.0
            category_importance['Selected_features_total'] = len(group_df)
        
        # Total features count (always from all features)
        category_importance['total_features_count'] = group_df.groupby('Category')['feature'].count().reindex(category_importance['Category']).fillna(0).values
        
        category_importance['Matrix'] = matrix
        category_importance['Horizon_h'] = horizon
        category_importance['Task'] = task
        
        results.append(category_importance)
    
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def main():
    """Main function to collect and save all feature importance data."""
    print("Collecting individual feature importance data...")
    individual_df = collect_individual_feature_importance()
    
    if individual_df.empty:
        print("❌ No feature importance data found!")
        return
    
    # Save individual feature importance
    individual_path = SUPP_DIR / "feature_importance_individual.csv"
    individual_df.to_csv(individual_path, index=False, float_format='%.4f')
    print(f"✅ Saved individual feature importance to: {individual_path}")
    print(f"   Total features: {len(individual_df)}")
    
    # Calculate and save category importance for each matrix
    print("\nCalculating category importance (90% threshold)...")
    
    # Matrix A - with threshold
    matrix_a_df = individual_df[individual_df['Matrix'] == 'A']
    if not matrix_a_df.empty:
        category_a = calculate_category_importance(matrix_a_df, categorize_feature_matrix_a, use_threshold=True)
        if not category_a.empty:
            category_a_path = SUPP_DIR / "feature_importance_category_matrix_A_90pct.csv"
            category_a.to_csv(category_a_path, index=False, float_format='%.4f')
            print(f"✅ Saved Matrix A category importance (90%) to: {category_a_path}")
    
    # Matrix B - with threshold
    matrix_b_df = individual_df[individual_df['Matrix'] == 'B']
    if not matrix_b_df.empty:
        category_b = calculate_category_importance(matrix_b_df, categorize_feature_matrix_b, use_threshold=True)
        if not category_b.empty:
            category_b_path = SUPP_DIR / "feature_importance_category_matrix_B_90pct.csv"
            category_b.to_csv(category_b_path, index=False, float_format='%.4f')
            print(f"✅ Saved Matrix B category importance (90%) to: {category_b_path}")
    
    # Matrix C - with threshold
    matrix_c_df = individual_df[individual_df['Matrix'] == 'C']
    if not matrix_c_df.empty:
        category_c = calculate_category_importance(matrix_c_df, categorize_feature_matrix_c, use_threshold=True)
        if not category_c.empty:
            category_c_path = SUPP_DIR / "feature_importance_category_matrix_C_90pct.csv"
            category_c.to_csv(category_c_path, index=False, float_format='%.4f')
            print(f"✅ Saved Matrix C category importance (90%) to: {category_c_path}")
    
    # Combined category importance - with threshold
    all_categories_threshold = []
    if not matrix_a_df.empty:
        cat_a = calculate_category_importance(matrix_a_df, categorize_feature_matrix_a, use_threshold=True)
        if not cat_a.empty:
            all_categories_threshold.append(cat_a)
    if not matrix_b_df.empty:
        cat_b = calculate_category_importance(matrix_b_df, categorize_feature_matrix_b, use_threshold=True)
        if not cat_b.empty:
            all_categories_threshold.append(cat_b)
    if not matrix_c_df.empty:
        cat_c = calculate_category_importance(matrix_c_df, categorize_feature_matrix_c, use_threshold=True)
        if not cat_c.empty:
            all_categories_threshold.append(cat_c)
    
    if all_categories_threshold:
        combined_category = pd.concat(all_categories_threshold, ignore_index=True)
        combined_path = SUPP_DIR / "feature_importance_category_all_90pct.csv"
        combined_category.to_csv(combined_path, index=False, float_format='%.4f')
        print(f"✅ Saved combined category importance (90%) to: {combined_path}")
    
    # Calculate and save category importance for ALL features (no threshold)
    print("\nCalculating category importance (ALL features)...")
    
    # Matrix A - all features
    if not matrix_a_df.empty:
        category_a_all = calculate_category_importance(matrix_a_df, categorize_feature_matrix_a, use_threshold=False)
        if not category_a_all.empty:
            category_a_path = SUPP_DIR / "feature_importance_category_matrix_A_all.csv"
            category_a_all.to_csv(category_a_path, index=False, float_format='%.4f')
            print(f"✅ Saved Matrix A category importance (ALL) to: {category_a_path}")
    
    # Matrix B - all features
    if not matrix_b_df.empty:
        category_b_all = calculate_category_importance(matrix_b_df, categorize_feature_matrix_b, use_threshold=False)
        if not category_b_all.empty:
            category_b_path = SUPP_DIR / "feature_importance_category_matrix_B_all.csv"
            category_b_all.to_csv(category_b_path, index=False, float_format='%.4f')
            print(f"✅ Saved Matrix B category importance (ALL) to: {category_b_path}")
    
    # Matrix C - all features
    if not matrix_c_df.empty:
        category_c_all = calculate_category_importance(matrix_c_df, categorize_feature_matrix_c, use_threshold=False)
        if not category_c_all.empty:
            category_c_path = SUPP_DIR / "feature_importance_category_matrix_C_all.csv"
            category_c_all.to_csv(category_c_path, index=False, float_format='%.4f')
            print(f"✅ Saved Matrix C category importance (ALL) to: {category_c_path}")
    
    # Combined category importance - all features
    all_categories_all = []
    if not matrix_a_df.empty:
        cat_a_all = calculate_category_importance(matrix_a_df, categorize_feature_matrix_a, use_threshold=False)
        if not cat_a_all.empty:
            all_categories_all.append(cat_a_all)
    if not matrix_b_df.empty:
        cat_b_all = calculate_category_importance(matrix_b_df, categorize_feature_matrix_b, use_threshold=False)
        if not cat_b_all.empty:
            all_categories_all.append(cat_b_all)
    if not matrix_c_df.empty:
        cat_c_all = calculate_category_importance(matrix_c_df, categorize_feature_matrix_c, use_threshold=False)
        if not cat_c_all.empty:
            all_categories_all.append(cat_c_all)
    
    if all_categories_all:
        combined_category_all = pd.concat(all_categories_all, ignore_index=True)
        combined_path = SUPP_DIR / "feature_importance_category_all_all.csv"
        combined_category_all.to_csv(combined_path, index=False, float_format='%.4f')
        print(f"✅ Saved combined category importance (ALL) to: {combined_path}")
    
    print(f"\n✅ All feature importance data saved to: {SUPP_DIR}")

if __name__ == "__main__":
    main()

