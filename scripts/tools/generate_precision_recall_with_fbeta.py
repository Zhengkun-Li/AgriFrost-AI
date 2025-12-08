#!/usr/bin/env python3
"""Generate Precision-Recall curve with F-beta scores (F1, F2, F3, F4) for decision support."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

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
print("📊 Generating Precision-Recall with F-beta Scores Plot")
print("=" * 70)

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Threshold range to test
thresholds = np.linspace(0.01, 0.99, 100)

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
    y_proba = np.array(predictions['frost']['y_proba'])
    
    # Sample data if too large (for performance)
    n_samples = len(y_true)
    if n_samples > 100000:
        sample_idx = np.random.choice(n_samples, 100000, replace=False)
        y_true = y_true[sample_idx]
        y_proba = y_proba[sample_idx]
        print(f"   Sampled {len(y_true)} points from {n_samples} total")
    
    # Calculate metrics for different thresholds
    precisions = []
    recalls = []
    f1_scores_list = []
    f2_scores = []
    f3_scores = []
    f4_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores_list.append(f1_score(y_true, y_pred, zero_division=0))
        f2_scores.append(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
        f3_scores.append(fbeta_score(y_true, y_pred, beta=3.0, zero_division=0))
        f4_scores.append(fbeta_score(y_true, y_pred, beta=4.0, zero_division=0))
    
    # Find optimal thresholds for different metrics
    optimal_f2_idx = np.argmax(f2_scores)
    optimal_f2_threshold = thresholds[optimal_f2_idx]
    
    optimal_f1_idx = np.argmax(f1_scores_list)
    optimal_f1_threshold = thresholds[optimal_f1_idx]
    
    # Plot
    ax = axes[idx]
    
    # Plot Precision and Recall (primary metrics)
    ax.plot(thresholds, precisions, label='Precision', linewidth=2.5, color='green', 
            alpha=0.9, zorder=5)
    ax.plot(thresholds, recalls, label='Recall (Correct Frost Predictions)', linewidth=3, 
            color='red', zorder=6)
    
    # Plot F-beta scores
    ax.plot(thresholds, f1_scores_list, label='F1 Score (β=1, Balanced)', 
            linewidth=1.8, color='orange', linestyle='--', alpha=0.8, zorder=4)
    ax.plot(thresholds, f2_scores, label='F2 Score (β=2, Recall 4×)', 
            linewidth=2.2, color='purple', linestyle='-', alpha=0.9, zorder=5)
    ax.plot(thresholds, f3_scores, label='F3 Score (β=3, Recall 9×)', 
            linewidth=1.8, color='magenta', linestyle='--', alpha=0.7, zorder=4)
    ax.plot(thresholds, f4_scores, label='F4 Score (β=4, Recall 16×)', 
            linewidth=1.8, color='pink', linestyle=':', alpha=0.7, zorder=4)
    
    # Mark optimal F2 threshold (primary optimization metric)
    optimal_f2_precision = precisions[optimal_f2_idx]
    optimal_f2_recall = recalls[optimal_f2_idx]
    optimal_f2_value = f2_scores[optimal_f2_idx]
    
    ax.axvline(x=optimal_f2_threshold, color='purple', linestyle=':', linewidth=2.5, 
               alpha=0.8, zorder=3, label=f'Optimal F2: {optimal_f2_threshold:.3f}')
    
    # Mark standard 0.5 threshold
    threshold_05_idx = np.argmin(np.abs(thresholds - 0.5))
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.6,
               zorder=3, label='Standard: 0.5')
    
    # Mark optimal F1 threshold for reference
    optimal_f1_precision = precisions[optimal_f1_idx]
    optimal_f1_recall = recalls[optimal_f1_idx]
    ax.axvline(x=optimal_f1_threshold, color='orange', linestyle='--', linewidth=1.5, 
               alpha=0.5, zorder=2)
    
    # Add markers at key thresholds
    ax.plot(optimal_f2_threshold, optimal_f2_recall, 'o', color='purple', markersize=12, 
            zorder=7, markeredgecolor='white', markeredgewidth=2, label='_nolegend_')
    ax.plot(optimal_f2_threshold, optimal_f2_precision, 'o', color='purple', markersize=12, 
            zorder=7, markeredgecolor='white', markeredgewidth=2, label='_nolegend_')
    ax.plot(0.5, recalls[threshold_05_idx], 's', color='gray', markersize=8, 
            zorder=7, markeredgecolor='white', markeredgewidth=1.5, alpha=0.7, label='_nolegend_')
    ax.plot(0.5, precisions[threshold_05_idx], 's', color='gray', markersize=8, 
            zorder=7, markeredgecolor='white', markeredgewidth=1.5, alpha=0.7, label='_nolegend_')
    
    # Add text annotations
    ax.text(optimal_f2_threshold, optimal_f2_recall + 0.04, 
            f'F2 Opt:\nP={optimal_f2_precision:.3f}\nR={optimal_f2_recall:.3f}\nF2={optimal_f2_value:.3f}', 
            fontsize=9, ha='center', color='purple', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9, 
                     edgecolor='purple', linewidth=2))
    
    precision_05 = precisions[threshold_05_idx]
    recall_05 = recalls[threshold_05_idx]
    f2_05 = f2_scores[threshold_05_idx]
    ax.text(0.5, recall_05 - 0.05, 
            f'0.5:\nP={precision_05:.3f}\nR={recall_05:.3f}\nF2={f2_05:.3f}', 
            fontsize=8, ha='center', color='gray', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                     edgecolor='gray', linewidth=1.5))
    
    # Set labels and title
    ax.set_xlabel('Decision Threshold (Probability Cutoff)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{horizon}h Forecast (Radius: {radius}km)\n'
                 f'Optimal F2 Threshold: {optimal_f2_threshold:.3f} | '
                 f'F2={optimal_f2_value:.3f}, Recall={optimal_f2_recall:.3f}, Precision={optimal_f2_precision:.3f}', 
                 fontsize=11, fontweight='bold', pad=10)
    ax.legend(loc='best', fontsize=9, framealpha=0.95, edgecolor='black', 
             fancybox=True, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    print(f"   ✅ Optimal F2 threshold: {optimal_f2_threshold:.3f}")
    print(f"   📊 At F2 optimal: Precision={optimal_f2_precision:.3f}, Recall={optimal_f2_recall:.3f}, F2={optimal_f2_value:.3f}")
    print(f"   📊 At 0.5 threshold: Precision={precision_05:.3f}, Recall={recall_05:.3f}, F2={f2_05:.3f}")

plt.tight_layout()

# Save figure
output_dir = project_root / "docs" / "figures" / "v2"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "precision_recall_vs_threshold.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Precision-Recall with F-beta scores plot saved: {output_path}")

