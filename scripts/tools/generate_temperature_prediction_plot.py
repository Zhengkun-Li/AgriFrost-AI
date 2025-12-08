#!/usr/bin/env python3
"""Generate temperature prediction vs ground truth scatter plot for all horizons."""

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

# R² values from the results
r2_values = {
    3: 0.9681,
    6: 0.9464,
    12: 0.9253,
    24: 0.9271
}

print("=" * 70)
print("🌡️  Generating Temperature Prediction vs Ground Truth Plot")
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
    
    # Extract temperature predictions
    y_true = np.array(predictions['temperature']['y_true'])
    y_pred = np.array(predictions['temperature']['y_pred'])
    
    # Sample data if too large (for visualization performance)
    n_samples = len(y_true)
    if n_samples > 100000:
        sample_idx = np.random.choice(n_samples, 100000, replace=False)
        y_true = y_true[sample_idx]
        y_pred = y_pred[sample_idx]
        print(f"   Sampled {len(y_true)} points from {n_samples} total")
    
    # Create scatter plot
    ax = axes[idx]
    ax.scatter(y_true, y_pred, alpha=0.3, s=1, color='steelblue')
    
    # Add diagonal line (perfect prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    
    # Set labels and title
    ax.set_xlabel('True Temperature (°C)', fontsize=11)
    ax.set_ylabel('Predicted Temperature (°C)', fontsize=11)
    ax.set_title(f'{horizon}h Forecast (Radius: {radius}km, $R^2$ = {r2_values[horizon]:.4f})', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add statistics text (lower position to avoid overlap with legend)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    stats_text = f'MAE: {mae:.2f}°C\nRMSE: {rmse:.2f}°C'
    ax.text(0.05, 0.82, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    print(f"   ✅ Loaded {len(y_true)} samples")
    print(f"   📊 MAE: {mae:.2f}°C, RMSE: {rmse:.2f}°C, R²: {r2_values[horizon]:.4f}")

plt.tight_layout()

# Save figure
output_dir = project_root / "docs" / "figures" / "v2"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "temperature_prediction_vs_ground_truth.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Temperature prediction plot saved: {output_path}")

