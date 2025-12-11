#!/usr/bin/env python3
"""
Generate comprehensive comparison figure for class-balanced vs baseline training.
Shows improvements across all key metrics: PR-AUC, ROC-AUC, Recall, Precision, 
Brier Score, ECE, MAE, RMSE, R².
"""

import json
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

def load_result(file_path):
    """Load JSON result file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None

# Collect data for all matrices and horizons
matrices = ['A', 'B', 'C']
horizons = ['3h', '6h', '12h', '24h']
radius_map = {'3h': 60, '6h': 100, '12h': 200, '24h': 200}  # For Matrix C

data = {
    'balanced': {m: {h: {} for h in horizons} for m in matrices},
    'baseline': {m: {h: {} for h in horizons} for m in matrices}
}

# Load Matrix A
for h in horizons:
    balanced_path = PROJECT_ROOT / "experiments/lightgbm/raw/A/full_training_balance/full_training/summary.json"
    baseline_path = PROJECT_ROOT / "experiments/lightgbm/raw/A/full_training/summary.json"
    
    if balanced_path.exists():
        result = load_result(balanced_path)
        if result and 'results' in result and h in result['results']:
            data['balanced']['A'][h] = result['results'][h]
    
    if baseline_path.exists():
        result = load_result(baseline_path)
        if result and 'results' in result and h in result['results']:
            data['baseline']['A'][h] = result['results'][h]

# Load Matrix B
for h in horizons:
    balanced_path = PROJECT_ROOT / "experiments/lightgbm/feature_engineering/B/full_training_balance/full_training/summary.json"
    baseline_path = PROJECT_ROOT / "experiments/lightgbm/feature_engineering/B/full_training/summary.json"
    
    if balanced_path.exists():
        result = load_result(balanced_path)
        if result and 'results' in result and h in result['results']:
            data['balanced']['B'][h] = result['results'][h]
    
    if baseline_path.exists():
        result = load_result(baseline_path)
        if result and 'results' in result and h in result['results']:
            data['baseline']['B'][h] = result['results'][h]

# Load Matrix C
for h in horizons:
    radius = radius_map[h]
    balanced_path = PROJECT_ROOT / f"experiments/lightgbm/raw/C/full_training_balance_{radius}km/full_training/summary.json"
    baseline_path = PROJECT_ROOT / f"experiments/lightgbm/raw/C/radius_{radius}km/full_training/summary.json"
    
    if balanced_path.exists():
        result = load_result(balanced_path)
        if result and 'results' in result and h in result['results']:
            data['balanced']['C'][h] = result['results'][h]
    
    if baseline_path.exists():
        result = load_result(baseline_path)
        if result and 'results' in result and h in result['results']:
            data['baseline']['C'][h] = result['results'][h]

# Extract metrics for all configurations
configs = []
for matrix in matrices:
    for h in horizons:
        config_name = f"{matrix}-{h}"
        balanced_data = data['balanced'][matrix][h]
        baseline_data = data['baseline'][matrix][h]
        
        if balanced_data and baseline_data:
            frost_bal = balanced_data.get('frost_metrics', {})
            temp_bal = balanced_data.get('temp_metrics', {})
            frost_base = baseline_data.get('frost_metrics', {})
            temp_base = baseline_data.get('temp_metrics', {})
            
            configs.append({
                'name': config_name,
                'matrix': matrix,
                'horizon': h,
                'balanced': {
                    'pr_auc': frost_bal.get('pr_auc', np.nan),
                    'roc_auc': frost_bal.get('roc_auc', np.nan),
                    'recall': frost_bal.get('recall', np.nan),
                    'precision': frost_bal.get('precision', np.nan),
                    'brier_score': frost_bal.get('brier_score', np.nan),
                    'ece': frost_bal.get('ece', np.nan),
                    'mae': temp_bal.get('mae', np.nan),
                    'rmse': temp_bal.get('rmse', np.nan),
                    'r2': temp_bal.get('r2', np.nan),
                },
                'baseline': {
                    'pr_auc': frost_base.get('pr_auc', np.nan),
                    'roc_auc': frost_base.get('roc_auc', np.nan),
                    'recall': frost_base.get('recall', np.nan),
                    'precision': frost_base.get('precision', np.nan),
                    'brier_score': frost_base.get('brier_score', np.nan),
                    'ece': frost_base.get('ece', np.nan),
                    'mae': temp_base.get('mae', np.nan),
                    'rmse': temp_base.get('rmse', np.nan),
                    'r2': temp_base.get('r2', np.nan),
                }
            })

# Calculate improvements (for metrics where higher is better, improvement = balanced - baseline)
# For metrics where lower is better (Brier, ECE, MAE, RMSE), improvement = baseline - balanced
metrics_info = {
    'pr_auc': {'label': 'PR-AUC', 'higher_better': True, 'format': '.3f'},
    'roc_auc': {'label': 'ROC-AUC', 'higher_better': True, 'format': '.3f'},
    'recall': {'label': 'Recall', 'higher_better': True, 'format': '.3f'},
    'precision': {'label': 'Precision', 'higher_better': True, 'format': '.3f'},
    'brier_score': {'label': 'Brier Score', 'higher_better': False, 'format': '.3f'},
    'ece': {'label': 'ECE', 'higher_better': False, 'format': '.3f'},
    'mae': {'label': 'MAE (°C)', 'higher_better': False, 'format': '.2f'},
    'rmse': {'label': 'RMSE (°C)', 'higher_better': False, 'format': '.2f'},
    'r2': {'label': 'R²', 'higher_better': True, 'format': '.3f'},
}

# Create figure with subplots - adjusted spacing
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)

# Prepare data for plotting
x_pos = np.arange(len(configs))
colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}

for idx, (metric_key, metric_info) in enumerate(metrics_info.items()):
    row = idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    balanced_vals = [c['balanced'][metric_key] for c in configs]
    baseline_vals = [c['baseline'][metric_key] for c in configs]
    
    # Calculate improvements
    if metric_info['higher_better']:
        improvements = [b - bl for b, bl in zip(balanced_vals, baseline_vals)]
    else:
        improvements = [bl - b for b, bl in zip(balanced_vals, baseline_vals)]
    
    # Create grouped bar chart
    width = 0.35
    x1 = x_pos - width/2
    x2 = x_pos + width/2
    
    bars1 = ax.bar(x1, baseline_vals, width, label='Baseline', alpha=0.7, color='#d62728')
    bars2 = ax.bar(x2, balanced_vals, width, label='Class-Balanced', alpha=0.7, color='#2ca02c')
    
    # Add improvement annotations
    for i, (imp, bal, base) in enumerate(zip(improvements, balanced_vals, baseline_vals)):
        if not (np.isnan(imp) or np.isnan(bal) or np.isnan(base)):
            # Calculate percentage improvement
            if metric_info['higher_better']:
                pct_imp = (imp / base * 100) if base > 0 else 0
            else:
                pct_imp = (imp / base * 100) if base > 0 else 0
            
            # Add text annotation
            if abs(pct_imp) > 1:  # Only show if improvement > 1%
                ax.text(i, max(bal, base) + 0.02 * max(bal, base), 
                       f'+{pct_imp:.0f}%' if pct_imp > 0 else f'{pct_imp:.0f}%',
                       ha='center', va='bottom', fontsize=7, rotation=90)
    
    ax.set_ylabel(metric_info['label'], fontsize=10)
    # Add arrow to title to show which direction is better
    arrow = '↑' if metric_info['higher_better'] else '↓'
    ax.set_title(f'{metric_info["label"]} {arrow}', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c['name'] for c in configs], rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    # Format y-axis to show 2 decimal places and rotate 45 degrees
    ax.tick_params(axis='y', labelsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    plt.setp(ax.get_yticklabels(), rotation=45, ha='right')
    
    # Set y-axis limits appropriately
    all_vals = [v for v in balanced_vals + baseline_vals if not np.isnan(v)]
    if all_vals:
        y_min, y_max = min(all_vals), max(all_vals)
        y_range = y_max - y_min
        ax.set_ylim(max(0, y_min - 0.1 * y_range), y_max + 0.15 * y_range)

# Remove top title as requested
# plt.suptitle('Class-Balanced Training vs Baseline: Comprehensive Metric Comparison', 
#              fontsize=14, fontweight='bold', y=0.995)

output_path = PROJECT_ROOT / "docs/manuscript/Supplementary_lighgbm_abc/figures" / "class_balance_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved figure to: {output_path}")

# Also create a summary improvement figure
fig2, ax = plt.subplots(figsize=(14, 8))

# Calculate average improvements across all configurations
avg_improvements = {}
for metric_key, metric_info in metrics_info.items():
    improvements = []
    for c in configs:
        bal = c['balanced'][metric_key]
        base = c['baseline'][metric_key]
        if not (np.isnan(bal) or np.isnan(base)):
            if metric_info['higher_better']:
                imp = (bal - base) / base * 100 if base > 0 else 0
            else:
                imp = (base - bal) / base * 100 if base > 0 else 0
            improvements.append(imp)
    
    if improvements:
        avg_improvements[metric_key] = np.mean(improvements)

# Create bar chart of average improvements
metrics_order = ['pr_auc', 'roc_auc', 'recall', 'precision', 'brier_score', 'ece', 'mae', 'rmse', 'r2']
labels = [metrics_info[k]['label'] for k in metrics_order]
values = [avg_improvements.get(k, 0) for k in metrics_order]

colors_bar = ['#2ca02c' if v > 0 else '#d62728' for v in values]
bars = ax.barh(labels, values, color=colors_bar, alpha=0.7)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(val + (1 if val > 0 else -1), i, f'{val:+.1f}%', 
           va='center', ha='left' if val > 0 else 'right', fontsize=10, fontweight='bold')

ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('Average Improvement (%)', fontsize=12)
ax.set_title('Average Improvement of Class-Balanced Training vs Baseline\n(Across All Matrices and Forecast Horizons)', 
             fontsize=13, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

output_path2 = PROJECT_ROOT / "docs/manuscript/Supplementary_lighgbm_abc/figures" / "class_balance_improvement_summary.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✅ Saved summary figure to: {output_path2}")

plt.close('all')
print("✅ All figures generated successfully!")

