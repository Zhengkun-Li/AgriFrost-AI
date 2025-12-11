#!/usr/bin/env python3
"""
Generate feature importance figure for Matrix A (16 raw CIMIS features).
Shows importance percentages for both classification and regression tasks across different horizons.
"""

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

# Read feature importance data
df = pd.read_csv(PROJECT_ROOT / 'docs/manuscript/Supplementary/supplementary_table_S7_matrix_a_feature_importance.csv')

# Create figure with subplots - 4 rows (horizons) x 2 columns (classification and regression)
# This layout makes each subplot larger and easier to read
fig = plt.figure(figsize=(14, 16))
gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.35)

horizons = [3, 6, 12, 24]
tasks = ['frost_classification', 'temperature_regression']
task_labels = ['Frost Classification', 'Temperature Regression']

# Reorganize: rows are horizons, columns are tasks
for h_idx, h in enumerate(horizons):
    for task_idx, (task, task_label) in enumerate(zip(tasks, task_labels)):
        ax = fig.add_subplot(gs[h_idx, task_idx])
        
        # Filter data for this horizon and task
        data = df[(df['horizon_h'] == h) & (df['task'] == task)].copy()
        data = data.sort_values('importance_pct', ascending=True)  # Sort for horizontal bar chart
        
        # Create horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))
        bars = ax.barh(range(len(data)), data['importance_pct'], color=colors, alpha=0.7)
        
        # Add percentage labels on bars
        for i, (idx, row) in enumerate(data.iterrows()):
            ax.text(row['importance_pct'] + 0.5, i, f'{row["importance_pct"]:.1f}%',
                   va='center', ha='left', fontsize=9, fontweight='bold')
        
        # Set y-axis labels - no rotation needed with larger subplots
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['feature'], fontsize=9, rotation=0, ha='right')
        
        # Adjust left margin
        ax.tick_params(axis='y', pad=5)
        
        # Set x-axis
        ax.set_xlabel('Importance (%)', fontsize=11)
        ax.set_xlim(0, max(data['importance_pct']) * 1.15)
        
        # Add title with arrow
        arrow = '↑'  # Higher is better for importance
        ax.set_title(f'{h}h - {task_label} {arrow}', fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Format x-axis
        ax.tick_params(axis='x', labelsize=10)

output_path = PROJECT_ROOT / "docs/manuscript/Supplementary_lighgbm_abc/figures" / "matrix_a_feature_importance.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved figure to: {output_path}")

plt.close('all')
print("✅ Figure generated successfully!")

