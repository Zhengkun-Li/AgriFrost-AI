#!/usr/bin/env python3
"""
Generate feature category importance figure for Matrix B based on cumulative feature importance.
First calculates cumulative importance at feature level (e.g., 80% threshold),
then groups selected features by category to avoid feature count bias.
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

def categorize_feature_matrix_b(feature_name: str) -> str:
    """Categorize a feature for Matrix B based on its name."""
    feature_lower = feature_name.lower()
    
    # Rolling statistics features (must check first to avoid conflicts)
    if any(x in feature_lower for x in ['rolling', 'rolling_', '_rolling']):
        return "Rolling Statistics"
    
    # Lag features
    if any(x in feature_lower for x in ['lag_', '_lag']):
        return "Lag Features"
    
    # Time features
    if any(x in feature_lower for x in ['hour', 'day_of_year', 'jul', 'day_of_week', 'month', 'season', 'is_night', 'day_progress', 'frost_season', 'day_of_year_sin', 'day_of_year_cos', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']):
        return "Time Features"
    
    # Wind features (check before other features to catch raw wind variables)
    if any(x in feature_lower for x in ['wind_dir', 'wind speed', 'wind_chill', 'calm_wind', 'wind_dir_sin', 'wind_dir_cos', 'wind_dir_category', 'wind_speed_change_rate']):
        return "Wind Features"
    
    # Soil features
    if any(x in feature_lower for x in ['soil temp', 'soil_temp', 'soil_air_temp']):
        return "Soil Features"
    
    # Derived meteorological features
    if any(x in feature_lower for x in ['heat_index', 'temp_dew_diff', 'temp_change_rate', 'vapor_pressure', 'saturation_vapor', 'dew_point_proximity', 'temp_decline_rate', 'cooling_acceleration', 'temp_trend', 'vapor_pressure_deficit', 'temp_humidity_interaction', 'wind_dir_temp_interaction', 'radiation_temp_interaction']):
        return "Derived Meteorological"
    
    # Station features
    if any(x in feature_lower for x in ['stn_id', 'stn id', 'station', 'region_encoded']):
        return "Station Features"
    
    # Radiation features (could be in Other or Derived, but let's put in Other for now)
    if any(x in feature_lower for x in ['sol rad', 'sol_rad', 'solar_radiation', 'nighttime_cooling_rate', 'sol_rad_change_rate', 'daily_solar_radiation']):
        return "Other Features"
    
    # Humidity features (could be in Other or Derived)
    if any(x in feature_lower for x in ['rel hum', 'rel_hum', 'humidity', 'humidity_change_rate']):
        return "Other Features"
    
    # Other raw CIMIS variables and remaining features
    return "Other Features"

def load_feature_importance(horizon: int, task: str) -> pd.DataFrame:
    """Load feature importance for a specific horizon and task."""
    task_file = 'frost_feature_importance.json' if task == 'frost_classification' else 'temp_feature_importance.json'
    
    # Try different possible paths
    possible_paths = [
        PROJECT_ROOT / f'experiments/lightgbm/feature_engineering/B/full_training_balance/full_training/horizon_{horizon}h/feature_importance/{task_file}',
        PROJECT_ROOT / f'experiments/lightgbm/feature_engineering/B/full_training/full_training/horizon_{horizon}h/feature_importance/{task_file}',
        PROJECT_ROOT / f'experiments/lightgbm/feature_engineering/B/full_training/horizon_{horizon}h/feature_importance/{task_file}',
    ]
    
    path = None
    for p in possible_paths:
        if p.exists():
            path = p
            break
    
    if path is None:
        return pd.DataFrame()
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        df = pd.DataFrame([{'feature': k, 'importance': v} for k, v in data.items()])
    else:
        return pd.DataFrame()
    
    # Calculate percentage
    total_importance = df['importance'].sum()
    df['importance_pct'] = (df['importance'] / total_importance * 100) if total_importance > 0 else 0
    
    # Sort by importance
    df = df.sort_values('importance', ascending=False)
    
    # Calculate cumulative importance
    df['cumulative_importance_pct'] = df['importance_pct'].cumsum()
    
    # Categorize features
    df['category'] = df['feature'].apply(categorize_feature_matrix_b)
    
    return df

# Create figure with subplots - 4 rows (horizons) x 2 columns (tasks)
fig = plt.figure(figsize=(16, 14))  # Increased height from 12 to 14
gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.25)  # hspace set to 0.5

horizons = [3, 6, 12, 24]
tasks = ['frost_classification', 'temperature_regression']
task_labels = ['Frost Classification', 'Temperature Regression']

# Define category order and colors
category_order = [
    'Rolling Statistics',
    'Lag Features',
    'Time Features',
    'Other Features',
    'Wind Features',
    'Derived Meteorological',
    'Station Features',
    'Soil Features'
]

category_labels = {
    'Rolling Statistics': 'Rolling Statistics',
    'Lag Features': 'Lag',
    'Time Features': 'Time',
    'Other Features': 'Other',
    'Wind Features': 'Wind',
    'Derived Meteorological': 'Derived Met.',
    'Station Features': 'Station',
    'Soil Features': 'Soil'
}

category_colors = {
    'Rolling Statistics': '#2ca02c',  # Green
    'Lag Features': '#1f77b4',       # Blue
    'Time Features': '#ff7f0e',      # Orange
    'Other Features': '#d62728',     # Red
    'Wind Features': '#9467bd',      # Purple
    'Derived Meteorological': '#8c564b',  # Brown
    'Station Features': '#e377c2',   # Pink
    'Soil Features': '#7f7f7f'       # Gray
}

for h_idx, h in enumerate(horizons):
    for task_idx, (task, task_label) in enumerate(zip(tasks, task_labels)):
        ax = fig.add_subplot(gs[h_idx, task_idx])
        
        # Load feature importance
        df = load_feature_importance(h, task)
        
        if df.empty:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Select features up to cumulative threshold
        # Find the position where cumulative importance first exceeds threshold
        threshold_mask = df['cumulative_importance_pct'] <= CUMULATIVE_THRESHOLD
        
        if not threshold_mask.any():
            # If first feature already exceeds threshold, include it
            selected_df = df.head(1).copy()
        else:
            # Find the last position where cumulative is still <= threshold
            # Use positional index (iloc) instead of label index
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
                    selected_df = df.iloc[:next_pos].copy()
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
        # This is the percentage of the 90% that each category represents
        category_importance['importance_pct_in_selected'] = (
            category_importance['importance'] / selected_total_importance * 100
            if selected_total_importance > 0 else 0
        )
        
        category_importance = category_importance.sort_values('importance_pct_in_selected', ascending=False)
        
        # Sort by category order for consistent display
        category_importance['category_order'] = category_importance['category'].map(
            {cat: idx for idx, cat in enumerate(category_order)}
        )
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
                   va='center', ha='left', fontsize=11, fontweight='bold')  # Increased from 9 to 11
        
        # Set y-axis labels (remove "features/statics" etc., keep only category names)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([category_labels.get(cat, cat) for cat in categories], 
                          fontsize=11, ha='right')  # Increased from 9 to 11
        
        # Set x-axis
        ax.set_xlabel('Importance (%) within Selected Features', fontsize=13)  # Increased from 11 to 13
        ax.set_xlim(0, max(importance_pct) * 1.15 if importance_pct else 100)
        
        # Add title with cumulative info
        arrow = '↑'
        ax.set_title(f'{task_label} - {h}h {arrow}\n({n_features_selected} features, {actual_cumsum:.1f}% of total)', 
                    fontsize=12, fontweight='bold')  # Increased from 11 to 12
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Format x-axis and y-axis (increased font sizes)
        ax.tick_params(axis='x', labelsize=12)  # Increased from 10 to 12
        ax.tick_params(axis='y', labelsize=11, pad=5)  # Increased from default to 11

# Save figure
output_path = PROJECT_ROOT / "docs/manuscript/Supplementary_lighgbm_abc/figures" / "matrix_b_feature_category_importance.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Figure saved to: {output_path}")
print(f"   Based on {CUMULATIVE_THRESHOLD}% cumulative feature importance threshold")

