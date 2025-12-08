#!/usr/bin/env python3
"""Generate classification probability distribution plot for all horizons."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Optimal configurations: (horizon, radius)
configs = [
    (3, 60),
    (6, 160),
    (12, 200),
    (24, 180)
]

print("=" * 70)
print("📊 Generating Classification Probability Distribution Plot")
print("=" * 70)

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (horizon, radius) in enumerate(configs):
    # Load predictions
    pred_path = project_root / f"experiments/lightgbm/raw/C/radius_{radius}km/full_training/horizon_{horizon}h/predictions.json"
    
    if not pred_path.exists():
        print(f"⚠️  Warning: {pred_path} not found, skipping...")
        continue
    
    print(f"\n📂 Loading predictions for {horizon}h (radius {radius}km)...")
    with open(pred_path, 'r') as f:
        predictions = json.load(f)
    
    # Extract frost classification predictions
    y_true = np.array(predictions['frost']['y_true'])
    y_proba = np.array(predictions['frost']['y_pred'])
    
    # Sample data if too large (for visualization performance)
    n_samples = len(y_true)
    if n_samples > 500000:
        sample_idx = np.random.choice(n_samples, 500000, replace=False)
        y_true = y_true[sample_idx]
        y_proba = y_proba[sample_idx]
        print(f"   Sampled {len(y_true)} points from {n_samples} total")
    
    # Split into frost and no-frost groups
    frost_proba = y_proba[y_true == 1]
    no_frost_proba = y_proba[y_true == 0]
    
    # Create histogram
    ax = axes[idx]
    
    # Create bins from 0 to 1
    bins = np.linspace(0, 1, 51)
    
    # Plot histograms
    ax.hist(no_frost_proba, bins=bins, alpha=0.6, label=f'No Frost (n={len(no_frost_proba):,})', 
            color='steelblue', density=False, edgecolor='black', linewidth=0.5)
    ax.hist(frost_proba, bins=bins, alpha=0.8, label=f'Frost (n={len(frost_proba):,})', 
            color='coral', density=False, edgecolor='black', linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('Predicted Frost Probability', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{horizon}h Forecast (Radius: {radius}km)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim([0, 1])
    
    # Add statistics text
    frost_rate = len(frost_proba) / len(y_true) * 100
    mean_frost_proba = np.mean(frost_proba) if len(frost_proba) > 0 else 0
    mean_no_frost_proba = np.mean(no_frost_proba) if len(no_frost_proba) > 0 else 0
    stats_text = f'Frost Rate: {frost_rate:.2f}%\nMean P(Frost|Frost): {mean_frost_proba:.3f}\nMean P(Frost|No Frost): {mean_no_frost_proba:.3f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    print(f"   ✅ Loaded {len(y_true)} samples")
    print(f"   📊 Frost events: {len(frost_proba):,} ({frost_rate:.2f}%)")
    print(f"   📊 Mean proba (Frost): {mean_frost_proba:.3f}, Mean proba (No Frost): {mean_no_frost_proba:.3f}")

plt.tight_layout()

# Save figure
output_dir = project_root / "docs" / "figures" / "v2"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "classification_probability_distribution.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Classification distribution plot saved: {output_path}")

