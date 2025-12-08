#!/usr/bin/env python3
"""Generate confusion matrix visualization for classification results across all horizons."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
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
print("📊 Generating Classification Confusion Matrix Plot")
print("=" * 70)

# Create figure with 2x2 subplots with more spacing
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
axes = axes.flatten()

for idx, (horizon, radius) in enumerate(configs):
    # Load metrics
    metrics_path = project_root / f"experiments/lightgbm/raw/C/radius_{radius}km/full_training/horizon_{horizon}h/frost_metrics.json"
    
    if not metrics_path.exists():
        print(f"⚠️  Warning: {metrics_path} not found, skipping...")
        continue
    
    print(f"\n📂 Loading metrics for {horizon}h (radius {radius}km)...")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Extract confusion matrix values
    tp = metrics.get('tp', 0)
    tn = metrics.get('tn', 0)
    fp = metrics.get('fp', 0)
    fn = metrics.get('fn', 0)
    
    # Create confusion matrix
    cm = np.array([[tn, fp],
                   [fn, tp]])
    
    # Create heatmap
    ax = axes[idx]
    
    # Normalize for better visualization (percentage)
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
    
    # Create heatmap with annotations (no colorbar/legend)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar=False,  # No colorbar
                xticklabels=['No Frost', 'Frost'],
                yticklabels=['No Frost', 'Frost'],
                linewidths=1, linecolor='gray',
                square=True)
    
    # Add percentage annotations on top (adjusted position)
    for i in range(2):
        for j in range(2):
            text = ax.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1f}%)',
                          ha="center", va="center", color="red", fontsize=9, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold', labelpad=10)
    
    # Calculate metrics for title
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    f1 = metrics.get('f1', 0)
    accuracy = metrics.get('accuracy', 0)
    
    ax.set_title(f'{horizon}h Forecast (Radius: {radius}km)\n'
                 f'Accuracy: {accuracy:.4f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}',
                 fontsize=12, fontweight='bold', pad=15)
    
    print(f"   ✅ Loaded confusion matrix")
    print(f"   📊 TP: {tp:,}, TN: {tn:,}, FP: {fp:,}, FN: {fn:,}")
    print(f"   📊 Accuracy: {accuracy:.4f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

# Adjust layout with more padding (no need for extra bottom margin since no text box)
plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.10, hspace=0.30, wspace=0.25)

# Save figure
output_dir = project_root / "docs" / "figures" / "v2"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "classification_confusion_matrix.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Classification confusion matrix plot saved: {output_path}")

