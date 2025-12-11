#!/usr/bin/env python3
"""Generate boxplot comparison for LOSO vs. Regular evaluation.

This script generates boxplots comparing regular evaluation and LOSO evaluation
for Matrix A, B, C across all horizons and key metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPLEMENTARY_DIR = PROJECT_ROOT / "docs/manuscript/Supplementary_lighgbm_abc"
FIGURES_DIR = SUPPLEMENTARY_DIR / "figures"
RESULTS_DIR = PROJECT_ROOT / "experiments/lightgbm"
HORIZONS = [3, 6, 12, 24]

# Set matplotlib parameters for better quality
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_regular_metrics():
    """Load regular evaluation metrics."""
    metrics_file = SUPPLEMENTARY_DIR / "all_metrics_lightgbm_abc.csv"
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        df_balanced = df[df['Training'] == 'Balanced'].copy()
        return df_balanced
    return pd.DataFrame()

def load_loso_summary():
    """Load LOSO summary results."""
    results = {}
    
    # Matrix A
    loso_dir_a = RESULTS_DIR / "raw/A/loso"
    if loso_dir_a.exists():
        summary_file = loso_dir_a / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                results['A'] = json.load(f)
    
    # Matrix B
    loso_dir_b = RESULTS_DIR / "feature_engineering/B/loso"
    if loso_dir_b.exists():
        summary_file = loso_dir_b / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                results['B'] = json.load(f)
    
    # Matrix C (optimal radius for each horizon)
    optimal_radius = {3: 60, 6: 100, 12: 200, 24: 200}
    for horizon in HORIZONS:
        radius = optimal_radius[horizon]
        loso_dir_c = RESULTS_DIR / f"raw/C/loso_radius_{radius}km/loso"
        if not loso_dir_c.exists():
            loso_dir_c = RESULTS_DIR / f"raw/C/full_training_balance_{radius}km/loso"
        
        if loso_dir_c.exists():
            summary_file = loso_dir_c / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    if 'C' not in results:
                        results['C'] = {}
                    summary = json.load(f)
                    h_key = f"{horizon}h"
                    if h_key in summary:
                        results['C'][h_key] = summary[h_key]
    
    return results

def load_station_results(matrix):
    """Load per-station LOSO results."""
    station_results = {}
    
    if matrix == 'A':
        loso_dir = RESULTS_DIR / "raw/A/loso"
    elif matrix == 'B':
        loso_dir = RESULTS_DIR / "feature_engineering/B/loso"
    else:
        return station_results
    
    # Try station_results.json first
    station_results_file = loso_dir / "station_results.json"
    if station_results_file.exists():
        with open(station_results_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                for station_data in data:
                    station_id = station_data.get('station_id', 'unknown')
                    if 'horizons' in station_data:
                        station_results[station_id] = station_data['horizons']
    
    # If not found, try individual station directories
    if not station_results:
        station_dirs = [d for d in loso_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('station_')]
        for station_dir in station_dirs:
            station_id = station_dir.name.replace('station_', '')
            # Look for result files in station directory
            result_files = list(station_dir.glob("*.json"))
            if result_files:
                with open(result_files[0]) as f:
                    station_data = json.load(f)
                    if 'horizons' in station_data:
                        station_results[station_id] = station_data['horizons']
    
    return station_results

def extract_loso_values(station_results, horizon_key, metric_key, metric_type='frost'):
    """Extract individual station values from LOSO summary."""
    if horizon_key not in summary_data:
        return []
    
    h_data = summary_data[horizon_key]
    if metric_type == 'frost':
        metrics = h_data.get('frost_metrics', {})
    else:
        metrics = h_data.get('temp_metrics', {})
    
    metric_data = metrics.get(metric_key, {})
    if isinstance(metric_data, dict) and 'values' in metric_data:
        return [v for v in metric_data['values'] if not np.isnan(v)]
    return []

def load_station_results_from_csv():
    """Load per-station LOSO results from CSV file."""
    csv_file = SUPPLEMENTARY_DIR / "loso_station_results_abc.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        return df
    return pd.DataFrame()

def generate_boxplot_comparison():
    """Generate boxplot comparison figure."""
    print("=" * 70)
    print("生成LOSO箱线图对比（Matrix A和B）")
    print("=" * 70)
    print()
    
    # Load data from CSV
    print("1. 加载数据...")
    df_stations = load_station_results_from_csv()
    
    if len(df_stations) == 0:
        print("   ❌ 未找到站点结果数据")
        return
    
    print(f"   ✅ 加载了 {len(df_stations)} 条站点结果")
    print(f"   Matrix A: {len(df_stations[df_stations['Matrix'] == 'A'])} 条")
    print(f"   Matrix B: {len(df_stations[df_stations['Matrix'] == 'B'])} 条")
    print()
    
    # Prepare data for plotting
    print("2. 准备绘图数据...")
    matrices = ['A', 'B']
    metrics_config = [
        {'name': 'PR-AUC', 'csv_col': 'PR_AUC', 'higher_better': True},
        {'name': 'ROC-AUC', 'csv_col': 'ROC_AUC', 'higher_better': True},
        {'name': 'Recall', 'csv_col': 'Recall', 'higher_better': True},
        {'name': 'Precision', 'csv_col': 'Precision', 'higher_better': True},
        {'name': 'Brier Score', 'csv_col': 'Brier_Score', 'higher_better': False},
        {'name': 'ECE', 'csv_col': 'ECE', 'higher_better': False},
        {'name': 'MAE (°C)', 'csv_col': 'MAE', 'higher_better': False},
        {'name': 'RMSE (°C)', 'csv_col': 'RMSE', 'higher_better': False},
        {'name': 'R²', 'csv_col': 'R2', 'higher_better': True},
    ]
    
    # Create figure with subplots (3x3 for 9 metrics)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    colors = {'A': '#1f77b4', 'B': '#ff7f0e'}
    
    # Load regular metrics for reference
    df_regular = load_regular_metrics()
    
    for idx, metric_config in enumerate(metrics_config):  # All 9 metrics
        ax = axes[idx]
        metric_name = metric_config['name']
        csv_col = metric_config['csv_col']
        
        # Map CSV column to regular metrics column
        regular_col_map = {
            'PR_AUC': 'Frost_PR_AUC',
            'ROC_AUC': 'Frost_ROC_AUC',
            'Recall': 'Frost_Recall',
            'Precision': 'Frost_Precision',
            'Brier_Score': 'Frost_Brier_Score',
            'ECE': 'Frost_ECE',
            'MAE': 'Temp_MAE',
            'RMSE': 'Temp_RMSE',
            'R2': 'Temp_R2',
        }
        regular_col = regular_col_map.get(csv_col, None)
        
        # Collect data for each horizon (group A and B together)
        box_data = []
        box_positions = []
        box_colors_list = []
        box_matrix_list = []  # Track which matrix each box belongs to
        regular_positions = []
        regular_values = []
        tick_positions = []
        tick_labels = []
        pos_counter = 0
        
        for h_idx, horizon in enumerate(HORIZONS):
            # For each horizon, add Matrix A and B side by side
            for matrix in matrices:
                # Extract LOSO values for this matrix-horizon combination
                mask = (df_stations['Matrix'] == matrix) & (df_stations['Horizon_h'] == horizon)
                values = df_stations[mask][csv_col].dropna().tolist()
                
                if len(values) > 0:
                    box_data.append(values)
                    box_positions.append(pos_counter)
                    box_colors_list.append(colors[matrix])
                    box_matrix_list.append(matrix)  # Track matrix type
                
                # Get regular evaluation value for reference
                if regular_col and len(df_regular) > 0:
                    regular_mask = (df_regular['Matrix'] == matrix) & (df_regular['Horizon_h'] == horizon)
                    if len(regular_mask) > 0 and regular_mask.any():
                        regular_row = df_regular[regular_mask]
                        if len(regular_row) > 0:
                            reg_val = regular_row.iloc[0].get(regular_col)
                            if reg_val is not None and not np.isnan(reg_val):
                                regular_positions.append(pos_counter)
                                regular_values.append(reg_val)
                
                pos_counter += 0.3  # Smaller spacing between A and B
            
            # Add tick label only once per horizon (in the middle)
            tick_positions.append((pos_counter - 0.3) / 2)  # Middle position between A and B
            tick_labels.append(f'{horizon}h')
            
            pos_counter += 0.5  # Spacing between horizons
        
        # Create boxplot
        if box_data:
            bp = ax.boxplot(box_data, positions=box_positions, widths=0.2,
                           patch_artist=True, showmeans=True, meanline=True,
                           zorder=2)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], box_colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)
            
            # Style other elements
            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                if element in bp:
                    plt.setp(bp[element], color='black', linewidth=1.5)
            
            # Add mean values as text (left for Matrix A, right for Matrix B)
            for i, (pos, values, matrix) in enumerate(zip(box_positions, box_data, box_matrix_list)):
                mean_val = np.mean(values)
                if matrix == 'A':
                    # Matrix A: text on the left
                    ax.text(pos - 0.12, mean_val, f'{mean_val:.3f}', 
                           ha='right', va='center', fontsize=9, fontweight='bold')
                else:  # Matrix B
                    # Matrix B: text on the right
                    ax.text(pos + 0.12, mean_val, f'{mean_val:.3f}', 
                           ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Add regular evaluation values as stars
        if regular_positions and regular_values:
            ax.scatter(regular_positions, regular_values, s=150, marker='*', 
                      color='red', edgecolors='black', linewidths=1,
                      zorder=3, label='Regular (Balanced)', alpha=0.8)
        
        # Set labels and formatting (not bold)
        ax.set_xlabel('Forecast Window', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        
        # Add arrow to title based on higher_better
        arrow = '↑' if metric_config['higher_better'] else '↓'
        ax.set_title(f'{metric_name} {arrow} (LOSO)', fontsize=12, fontweight='bold')
        
        # Set x-axis ticks and labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            mpatches.Patch(facecolor=colors['A'], alpha=0.6, edgecolor='black',
                          label='Matrix A (LOSO)'),
            mpatches.Patch(facecolor=colors['B'], alpha=0.6, edgecolor='black',
                          label='Matrix B (LOSO)'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markersize=12, markeredgecolor='black', markeredgewidth=1,
                   label='Regular (Balanced)', linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_file = FIGURES_DIR / "loso_boxplot_ab.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ 图表已保存到: {output_file}")
    print()
    
    plt.close()
    
    print("✅ 完成！")

if __name__ == "__main__":
    generate_boxplot_comparison()

