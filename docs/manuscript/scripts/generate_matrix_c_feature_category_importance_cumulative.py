#!/usr/bin/env python3
"""
Generate feature category importance figure for Matrix C based on cumulative feature importance.
First calculates cumulative importance at feature level (90% threshold),
then groups selected features by category to avoid feature count bias.
Same methodology as Matrix B (Figure 9).
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SUPPLEMENTARY_FIGURES_DIR = PROJECT_ROOT / "docs/manuscript/Supplementary_lighgbm_abc/figures"
SUPPLEMENTARY_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Cumulative importance threshold (90% recommended for comprehensive analysis)
CUMULATIVE_THRESHOLD = 90.0

# Optimal radius for each horizon (from Table 2)
OPTIMAL_RADIUS = {
    3: 60,   # 3h: 60km
    6: 100,  # 6h: 100km
    12: 200, # 12h: 200km
    24: 200  # 24h: 200km
}

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
    
    # Derived meteorological features (must check before raw vars and other features)
    if any(x in feature_lower for x in ['heat_index', 'temp_dew_diff', 'temp_change_rate', 'vapor_pressure', 
                                         'saturation_vapor', 'dew_point_proximity', 'temp_decline_rate', 
                                         'cooling_acceleration', 'temp_trend', 'vapor_pressure_deficit', 
                                         'temp_humidity_interaction', 'wind_dir_temp_interaction', 
                                         'radiation_temp_interaction', 'wind_chill', 'soil_air_temp_diff']):
        return "Engineering Features"
    
    # Station features
    if any(x in feature_lower for x in ['stn_id', 'stn id', 'station', 'region_encoded', 'station_elevation', 
                                         'station_latitude', 'station_longitude']):
        return "Other Features"
    
    # Raw CIMIS variables (16 original features)
    raw_vars = ['air temp', 'dew point', 'rel hum', 'wind speed', 'wind dir', 'sol rad', 
                'soil temp', 'vapor', 'vap pres', 'eto', 'precip', 'hour', 'jul']
    if any(x in feature_lower for x in raw_vars):
        # Check if it's a spatial aggregation (has neighbor prefix)
        if 'neighbor' not in feature_lower:
            return "Raw Features"
    
    # Other features (humidity, radiation, wind related that are not raw and not derived)
    if any(x in feature_lower for x in ['rel_hum', 'humidity', 'sol_rad', 'solar_radiation', 
                                         'calm_wind']):
        if 'neighbor' not in feature_lower:
            return "Other Features"
    
    # Default: if contains neighbor, it's spatial; otherwise other
    if 'neighbor' in feature_lower:
        return "Spatial Features"
    
    return "Other Features"

def load_feature_importance(horizon: int, task: str) -> pd.DataFrame:
    """Load feature importance for a specific horizon and task from optimal radius configuration."""
    # Matrix C uses CSV format, not JSON
    task_file = 'frost_feature_importance.csv' if task == 'frost_classification' else 'temp_feature_importance.csv'
    radius = OPTIMAL_RADIUS[horizon]
    
    # Load from optimal radius configuration (best model for each horizon)
    possible_paths = [
        PROJECT_ROOT / f'experiments/lightgbm/raw/C/full_training_balance_{radius}km/full_training/horizon_{horizon}h/{task_file}',
        PROJECT_ROOT / f'experiments/lightgbm/raw/C/full_training_{radius}km/full_training/horizon_{horizon}h/{task_file}',
        PROJECT_ROOT / f'experiments/lightgbm/feature_engineering/C/full_training_balance/full_training/horizon_{horizon}h/{task_file}',
    ]
    
    path = None
    for p in possible_paths:
        if p.exists():
            path = p
            break
    
    if path is None:
        print(f"⚠️  No feature importance file found for {horizon}h {task}")
        return pd.DataFrame()
    
    # Load CSV file
    df = pd.read_csv(path)
    
    # CSV format: should have 'feature' and 'importance' columns
    if 'feature' not in df.columns or 'importance' not in df.columns:
        # Try alternative column names
        if 'feature_name' in df.columns:
            df = df.rename(columns={'feature_name': 'feature'})
        elif len(df.columns) >= 2:
            df.columns = ['feature', 'importance'] + list(df.columns[2:])
        else:
            print(f"⚠️  Unexpected CSV format in {path}")
            return pd.DataFrame()
    
    # Check if CSV already has importance_pct and cumulative_pct columns
    if 'importance_pct' in df.columns and 'cumulative_pct' in df.columns:
        # Use existing columns
        df['cumulative_importance_pct'] = df['cumulative_pct']
    else:
        # Calculate percentage
        total_importance = df['importance'].sum()
        df['importance_pct'] = (df['importance'] / total_importance * 100) if total_importance > 0 else 0
        
        # Sort by importance
        df = df.sort_values('importance', ascending=False)
        
        # Calculate cumulative importance
        df['cumulative_importance_pct'] = df['importance_pct'].cumsum()
    
    # Categorize features
    df['category'] = df['feature'].apply(categorize_feature_matrix_c)
    
    return df

# Create figure with subplots - 4 rows (horizons) x 2 columns (tasks)
fig = plt.figure(figsize=(16, 14))  # Same size as Figure 9
gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.25)  # Same spacing as Figure 9

horizons = [3, 6, 12, 24]
tasks = ['frost_classification', 'temperature_regression']
task_labels = ['Frost Classification', 'Temperature Regression']

# Define category order and labels
# Order: spatial, masks, engineering, other, raw (from top to bottom in y-axis)
category_order = [
    'Spatial Features',
    'Mask Features',
    'Engineering Features',
    'Other Features',
    'Raw Features'
]

category_labels = {
    'Spatial Features': 'Spatial',
    'Mask Features': 'Mask',
    'Engineering Features': 'Engineering',
    'Other Features': 'Other',
    'Raw Features': 'Raw'
}

category_colors = {
    'Spatial Features': '#2ca02c',  # Green
    'Mask Features': '#9467bd',      # Purple
    'Engineering Features': '#ff7f0e',      # Orange
    'Other Features': '#7f7f7f',       # Gray
    'Raw Features': '#1f77b4'       # Blue
}

for h_idx, h in enumerate(horizons):
    for task_idx, (task, task_label) in enumerate(zip(tasks, task_labels)):
        ax = fig.add_subplot(gs[h_idx, task_idx])
        
        # Load feature importance
        df = load_feature_importance(h, task)
        
        if df.empty:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'{task_label} - {h}h ↑', fontsize=12, fontweight='bold')
            continue
        
        # Select features up to cumulative threshold (same logic as Figure 9)
        threshold_mask = df['cumulative_importance_pct'] <= CUMULATIVE_THRESHOLD
        
        if not threshold_mask.any():
            # If first feature already exceeds threshold, include it
            selected_df = df.head(1).copy()
        else:
            # Find the last position where cumulative is still <= threshold
            last_valid_pos = None
            for pos in range(len(df)):
                if df.iloc[pos]['cumulative_importance_pct'] <= CUMULATIVE_THRESHOLD:
                    last_valid_pos = pos
                else:
                    break
            
            if last_valid_pos is not None:
                # Include one more feature after the last valid one to ensure we reach or exceed threshold
                next_pos = last_valid_pos + 1
                if next_pos < len(df):
                    # Include up to and including next_pos to reach or exceed 90%
                    selected_df = df.iloc[:next_pos+1].copy()
                else:
                    selected_df = df.copy()
            else:
                # Should not happen, but fallback
                selected_df = df.head(1).copy()
        
        # Calculate actual cumulative importance of selected features
        selected_total_importance = selected_df['importance'].sum()
        actual_cumsum = selected_df['cumulative_importance_pct'].iloc[-1] if len(selected_df) > 0 else 0
        n_features_selected = len(selected_df)
        
        # Group by category and sum importance within selected features
        category_importance = selected_df.groupby('category')['importance'].sum().reset_index()
        
        # Calculate percentage within the selected feature subset (90% threshold)
        category_importance['importance_pct_in_selected'] = (
            category_importance['importance'] / selected_total_importance * 100
            if selected_total_importance > 0 else 0
        )
        
        # Ensure all 4 main categories are present (Spatial, Mask, Engineering, Raw)
        # Add missing categories with 0% importance
        main_categories = ['Spatial Features', 'Mask Features', 'Engineering Features', 'Raw Features']
        for cat in main_categories:
            if cat not in category_importance['category'].values:
                category_importance = pd.concat([
                    category_importance,
                    pd.DataFrame({'category': [cat], 'importance': [0], 'importance_pct_in_selected': [0]})
                ], ignore_index=True)
        
        category_importance = category_importance.sort_values('importance_pct_in_selected', ascending=False)
        
        # Sort by category order for consistent display (only main 4 categories)
        category_importance['category_order'] = category_importance['category'].map(
            {cat: idx for idx, cat in enumerate(category_order)}
        )
        # Filter to only main 4 categories and sort
        category_importance = category_importance[category_importance['category'].isin(main_categories)]
        category_importance = category_importance.sort_values('category_order')
        
        # Prepare data for plotting
        categories = category_importance['category'].tolist()
        importance_pct = category_importance['importance_pct_in_selected'].tolist()
        colors = [category_colors.get(cat, '#7f7f7f') for cat in categories]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(categories))
        bars = ax.barh(y_pos, importance_pct, color=colors, alpha=0.7)
        
        # Add percentage labels on bars
        for i, (cat, imp) in enumerate(zip(categories, importance_pct)):
            ax.text(imp + 0.5, i, f'{imp:.1f}%',
                   va='center', ha='left', fontsize=11, fontweight='bold')
        
        # Set y-axis labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([category_labels.get(cat, cat) for cat in categories], 
                          fontsize=11, ha='right')
        
        # Set x-axis
        ax.set_xlabel('Importance (%) within Selected Features', fontsize=13)
        ax.set_xlim(0, max(importance_pct) * 1.15 if importance_pct else 100)
        
        # Add title with cumulative info (same format as Figure 9)
        arrow = '↑'
        ax.set_title(f'{task_label} - {h}h {arrow}\n({n_features_selected} features, {actual_cumsum:.1f}% of total)', 
                    fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Format x-axis and y-axis
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=11, pad=5)

# Save figure
output_path = PROJECT_ROOT / "docs/manuscript/Supplementary_lighgbm_abc/figures" / "matrix_c_feature_category_importance.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Figure saved to: {output_path}")
print(f"   Based on {CUMULATIVE_THRESHOLD}% cumulative feature importance threshold")

plt.close('all')

