#!/usr/bin/env python3
"""Generate model comparison figures for manuscript."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Colors for models
MODEL_COLORS = {
    'lightgbm': '#1f77b4',
    'xgboost': '#ff7f0e',
    'catboost': '#2ca02c',
    'random_forest': '#d62728',
    'gru': '#9467bd',
    'lstm': '#8c564b',
    'tcn': '#e377c2',
}

MODEL_NAMES = {
    'lightgbm': 'LightGBM',
    'xgboost': 'XGBoost',
    'catboost': 'CatBoost',
    'random_forest': 'Random Forest',
    'gru': 'GRU',
    'lstm': 'LSTM',
    'tcn': 'TCN',
}

def load_data():
    """Load experiment results."""
    csv_path = PROJECT_ROOT / "results/model_performance_all_models.csv"
    df = pd.read_csv(csv_path)
    return df

def plot_model_comparison_by_horizon(df, matrix_cell, metric='roc_auc', output_path=None):
    """Plot multiple models comparison across horizons for a given matrix."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    horizons = [3, 6, 12, 24]
    metrics = ['roc_auc', 'pr_auc', 'brier_score', 'temp_rmse']
    metric_labels = ['ROC-AUC', 'PR-AUC', 'Brier Score', 'Temperature RMSE (°C)']
    
    # Check if this matrix has radius information (C or D)
    has_radius = matrix_cell in ['C', 'D']
    
    for idx, (metric_name, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        # Filter data
        data = df[(df['matrix_cell'] == matrix_cell) & (df['horizon_h'].isin(horizons))]
        
        # Group by model and horizon
        for model in data['model'].unique():
            model_data = data[data['model'] == model]
            if len(model_data) == 0:
                continue
            
            # Get best performance for each horizon and record radius if applicable
            values = []
            radii = []  # Store best radius for each horizon
            for h in horizons:
                h_data = model_data[model_data['horizon_h'] == h]
                if len(h_data) > 0:
                    if metric_name in ['brier_score', 'temp_rmse']:
                        # Lower is better
                        best_idx = h_data[metric_name].idxmin()
                        val = h_data.loc[best_idx, metric_name]
                    else:
                        # Higher is better
                        best_idx = h_data[metric_name].idxmax()
                        val = h_data.loc[best_idx, metric_name]
                    
                    values.append(val)
                    
                    # Record radius if available
                    if has_radius and 'radius_km' in h_data.columns:
                        best_radius = h_data.loc[best_idx, 'radius_km']
                        radii.append(int(best_radius) if not pd.isna(best_radius) else None)
                    else:
                        radii.append(None)
                else:
                    values.append(np.nan)
                    radii.append(None)
            
            if not all(np.isnan(values)):
                color = MODEL_COLORS.get(model, '#7f7f7f')
                model_name = MODEL_NAMES.get(model, model.upper())
                
                # Add radius info to label if available
                if has_radius and any(r is not None for r in radii):
                    # Create label with radius info
                    radius_strs = []
                    for h, r in zip(horizons, radii):
                        if r is not None:
                            radius_strs.append(f"{h}h:{r}km")
                    if radius_strs:
                        label = f"{model_name} ({', '.join(radius_strs)})"
                    else:
                        label = model_name
                else:
                    label = model_name
                
                ax.plot(horizons, values, marker='o', linewidth=2, markersize=8, 
                       color=color, label=label, alpha=0.8)
        
        ax.set_xlabel('Forecast Horizon (hours)', fontweight='bold')
        ax.set_ylabel(metric_label, fontweight='bold')
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', ncol=1, framealpha=0.9, fontsize=9)
        ax.set_title(metric_label, fontweight='bold')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()

def plot_model_comparison_by_matrix(df, horizon_h, metric='roc_auc', output_path=None):
    """Plot multiple models comparison across matrices for a given horizon."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    matrices = ['A', 'B', 'C', 'D']
    metrics = ['roc_auc', 'pr_auc', 'brier_score', 'temp_rmse']
    metric_labels = ['ROC-AUC', 'PR-AUC', 'Brier Score', 'Temperature RMSE (°C)']
    
    for idx, (metric_name, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        # Filter data
        data = df[(df['horizon_h'] == horizon_h) & (df['matrix_cell'].isin(matrices))]
        
        # Group by model and matrix
        for model in data['model'].unique():
            model_data = data[data['model'] == model]
            if len(model_data) == 0:
                continue
            
            # Get best performance for each matrix
            values = []
            for m in matrices:
                m_data = model_data[model_data['matrix_cell'] == m]
                if len(m_data) > 0:
                    if metric_name in ['brier_score', 'temp_rmse']:
                        val = m_data[metric_name].min()
                    else:
                        val = m_data[metric_name].max()
                    values.append(val)
                else:
                    values.append(np.nan)
            
            if not all(np.isnan(values)):
                color = MODEL_COLORS.get(model, '#7f7f7f')
                label = MODEL_NAMES.get(model, model.upper())
                x_pos = np.arange(len(matrices))
                ax.plot(x_pos, values, marker='o', linewidth=2, markersize=8, 
                       color=color, label=label, alpha=0.8)
        
        ax.set_xlabel('Feature Matrix', fontweight='bold')
        ax.set_ylabel(metric_label, fontweight='bold')
        ax.set_xticks(range(len(matrices)))
        ax.set_xticklabels(matrices)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', ncol=2, framealpha=0.9)
        ax.set_title(metric_label, fontweight='bold')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()

def plot_model_family_comparison(df, output_path=None):
    """Plot comparison of all models (one curve per model)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    horizons = [3, 6, 12, 24]
    metrics = ['roc_auc', 'pr_auc', 'brier_score', 'temp_rmse']
    metric_labels = ['ROC-AUC', 'PR-AUC', 'Brier Score', 'Temperature RMSE (°C)']
    
    # Get all models
    all_models = sorted(df['model'].unique())
    
    for idx, (metric_name, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        
        # Plot each model
        for model in all_models:
            model_data = df[df['model'] == model]
            if len(model_data) == 0:
                continue
            
            # Get best performance for each horizon and record which matrix it comes from
            values = []
            matrix_labels = []  # Store which matrix each value comes from
            for h in horizons:
                h_data = model_data[model_data['horizon_h'] == h]
                if len(h_data) > 0:
                    if metric_name in ['brier_score', 'temp_rmse']:
                        # Lower is better
                        best_idx = h_data[metric_name].idxmin()
                        val = h_data.loc[best_idx, metric_name]
                    else:
                        # Higher is better
                        best_idx = h_data[metric_name].idxmax()
                        val = h_data.loc[best_idx, metric_name]
                    
                    values.append(val)
                    # Get the matrix label
                    best_matrix = h_data.loc[best_idx, 'matrix_cell']
                    matrix_labels.append(best_matrix)
                else:
                    values.append(np.nan)
                    matrix_labels.append(None)
            
            if not all(np.isnan(values)):
                color = MODEL_COLORS.get(model, '#7f7f7f')
                label = MODEL_NAMES.get(model, model.upper())
                # Use different line styles for tree models vs neural networks
                linestyle = '-' if model in ['lightgbm', 'xgboost', 'catboost', 'random_forest'] else '--'
                ax.plot(horizons, values, marker='o', linewidth=2, markersize=7, 
                       color=color, label=label, alpha=0.8, linestyle=linestyle)
                
                # Add matrix labels near each point with smart positioning to avoid overlap
                # Store label positions to check for overlaps
                label_positions = []
                for i, (h, val, matrix_label) in enumerate(zip(horizons, values, matrix_labels)):
                    if not np.isnan(val) and matrix_label is not None:
                        # Determine label position based on data point location and neighbors
                        # Try different positions: top-right, top-left, bottom-right, bottom-left
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                        
                        # Calculate relative position in plot
                        y_ratio = (val - ax.get_ylim()[0]) / y_range
                        x_ratio = (h - ax.get_xlim()[0]) / x_range
                        
                        # Choose position to minimize overlap
                        # For points in upper half, place label below; for lower half, place above
                        # For points on left, place label to right; for right, place to left
                        if y_ratio > 0.7:  # Upper region
                            y_offset = -12  # Below point
                        elif y_ratio < 0.3:  # Lower region
                            y_offset = 12  # Above point
                        else:  # Middle region
                            y_offset = 8  # Above point
                        
                        if x_ratio < 0.3:  # Left region
                            x_offset = 8  # Right of point
                        elif x_ratio > 0.7:  # Right region
                            x_offset = -8  # Left of point
                        else:  # Middle region
                            x_offset = 6  # Right of point
                        
                        # Check for potential overlap with nearby labels
                        label_x = h
                        label_y = val
                        overlap = False
                        for prev_x, prev_y in label_positions:
                            if abs(label_x - prev_x) < 0.5 and abs(label_y - prev_y) < y_range * 0.05:
                                overlap = True
                                # Adjust offset if overlap detected
                                if y_offset > 0:
                                    y_offset += 8
                                else:
                                    y_offset -= 8
                                break
                        
                        label_positions.append((label_x, label_y))
                        
                        ax.annotate(matrix_label, 
                                   xy=(h, val), 
                                   xytext=(x_offset, y_offset), 
                                   textcoords='offset points',
                                   fontsize=7,
                                   color=color,
                                   alpha=0.9,
                                   weight='bold',
                                   ha='center',
                                   va='center',
                                   bbox=dict(boxstyle='round,pad=0.3', 
                                            facecolor='white', 
                                            edgecolor=color, 
                                            alpha=0.7,
                                            linewidth=0.5))
        
        ax.set_xlabel('Forecast Horizon (hours)', fontweight='bold')
        ax.set_ylabel(metric_label, fontweight='bold')
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', ncol=2, framealpha=0.9, fontsize=9)
        ax.set_title(metric_label, fontweight='bold')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()

def create_model_comparison_table(df, output_path=None):
    """Create a comprehensive model comparison table."""
    # Aggregate by model and horizon
    results = []
    
    for model in df['model'].unique():
        for horizon in [3, 6, 12, 24]:
            model_data = df[(df['model'] == model) & (df['horizon_h'] == horizon)]
            if len(model_data) == 0:
                continue
            
            results.append({
                'Model': MODEL_NAMES.get(model, model.upper()),
                'Horizon (h)': horizon,
                'ROC-AUC (mean)': model_data['roc_auc'].mean(),
                'ROC-AUC (max)': model_data['roc_auc'].max(),
                'PR-AUC (mean)': model_data['pr_auc'].mean(),
                'PR-AUC (max)': model_data['pr_auc'].max(),
                'Brier (mean)': model_data['brier_score'].mean(),
                'Brier (min)': model_data['brier_score'].min(),
                'Temp RMSE (mean)': model_data['temp_rmse'].mean(),
                'Temp RMSE (min)': model_data['temp_rmse'].min(),
                'N Experiments': len(model_data),
            })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.round(4)
    
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    
    return result_df

def main():
    """Generate all comparison figures."""
    print("Loading data...")
    df = load_data()
    
    output_dir = PROJECT_ROOT / "docs/figures/v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating model comparison by horizon for Matrix A...")
    plot_model_comparison_by_horizon(
        df, 'A', output_path=output_dir / 'model_comparison_matrix_a_by_horizon.png'
    )
    
    print("\nGenerating model comparison by horizon for Matrix B...")
    plot_model_comparison_by_horizon(
        df, 'B', output_path=output_dir / 'model_comparison_matrix_b_by_horizon.png'
    )
    
    print("\nGenerating model comparison by horizon for Matrix C...")
    plot_model_comparison_by_horizon(
        df, 'C', output_path=output_dir / 'model_comparison_matrix_c_by_horizon.png'
    )
    
    print("\nGenerating model comparison by horizon for Matrix D...")
    plot_model_comparison_by_horizon(
        df, 'D', output_path=output_dir / 'model_comparison_matrix_d_by_horizon.png'
    )
    
    print("\nGenerating model comparison by matrix for 12h horizon...")
    plot_model_comparison_by_matrix(
        df, 12, output_path=output_dir / 'model_comparison_12h_by_matrix.png'
    )
    
    print("\nGenerating model family comparison...")
    plot_model_family_comparison(
        df, output_path=output_dir / 'model_family_comparison.png'
    )
    
    print("\nCreating model comparison table...")
    table = create_model_comparison_table(
        df, output_path=output_dir / 'model_comparison_table.csv'
    )
    print("\nModel comparison table preview:")
    print(table.head(20))
    
    print("\nAll figures generated successfully!")

if __name__ == "__main__":
    main()

