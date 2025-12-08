#!/usr/bin/env python3
"""Generate threshold analysis plot showing accuracy, precision, recall, F1 across different thresholds."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
print("📊 Generating Threshold Analysis Plot")
print("=" * 70)

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Threshold range to test
thresholds = np.linspace(0.01, 0.99, 50)

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
    accuracies = []
    precisions = []
    recalls = []
    f1_scores_list = []
    f2_scores = []
    f3_scores = []
    f4_scores = []
    
    from sklearn.metrics import fbeta_score
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores_list.append(f1_score(y_true, y_pred, zero_division=0))
        f2_scores.append(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
        f3_scores.append(fbeta_score(y_true, y_pred, beta=3.0, zero_division=0))
        f4_scores.append(fbeta_score(y_true, y_pred, beta=4.0, zero_division=0))
    
    # Find optimal threshold using F2 score (Method 3: F-beta with β=2)
    # F2 = (1 + 2²) × (precision × recall) / (2² × precision + recall)
    # Emphasizes recall 4× more than precision, suitable for frost forecasting
    optimal_idx = np.argmax(f2_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot
    ax = axes[idx]
    
    # Plot metrics with different line styles
    # Recall (red) is the most important - it measures correct frost predictions
    ax.plot(thresholds, recalls, label='Recall (Correct Frost Predictions)', linewidth=3, color='red', zorder=6)
    ax.plot(thresholds, f2_scores, label='F2 Score (β=2, Recall 4×)', linewidth=2.5, color='purple', 
            linestyle='-', alpha=0.9, zorder=5)
    ax.plot(thresholds, f3_scores, label='F3 Score (β=3, Recall 9×)', linewidth=2, color='magenta', 
            linestyle='--', alpha=0.8, zorder=4)
    ax.plot(thresholds, f4_scores, label='F4 Score (β=4, Recall 16×)', linewidth=2, color='pink', 
            linestyle=':', alpha=0.7, zorder=4)
    ax.plot(thresholds, accuracies, label='Accuracy (All Correct)', linewidth=1.5, color='blue', alpha=0.6, zorder=3)
    ax.plot(thresholds, precisions, label='Precision', linewidth=1.5, color='green', alpha=0.6, zorder=3)
    ax.plot(thresholds, f1_scores_list, label='F1 Score (β=1, Balanced)', linewidth=1.5, color='orange', 
            linestyle='--', alpha=0.6, zorder=3)
    
    # Find accuracy at optimal and 0.5 thresholds
    optimal_acc = accuracies[optimal_idx]
    threshold_05_idx = np.argmin(np.abs(thresholds - 0.5))
    acc_05 = accuracies[threshold_05_idx]
    
    # Mark optimal threshold with vertical line (F2 score)
    ax.axvline(x=optimal_threshold, color='purple', linestyle=':', linewidth=2.5, 
               alpha=0.8, zorder=3, label=f'Optimal (F2): {optimal_threshold:.3f}')
    
    # Mark standard 0.5 threshold with vertical line
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.6,
               zorder=3, label='Standard: 0.5')
    
    # Add markers at optimal and 0.5 thresholds on accuracy curve
    ax.plot(optimal_threshold, optimal_acc, 'o', color='purple', markersize=10, 
            zorder=6, markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(0.5, acc_05, 's', color='gray', markersize=8, 
            zorder=6, markeredgecolor='white', markeredgewidth=1.5, alpha=0.7)
    
    # Add text annotations - highlight Recall and F2 at optimal threshold
    optimal_recall = recalls[optimal_idx]
    optimal_f2 = f2_scores[optimal_idx]
    recall_05 = recalls[threshold_05_idx]
    f2_05 = f2_scores[threshold_05_idx]
    
    # Annotation at optimal threshold
    ax.text(optimal_threshold, optimal_recall + 0.03, f'F2: {optimal_f2:.3f}\nRecall: {optimal_recall:.3f}', 
            fontsize=9, ha='center', color='purple', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9, edgecolor='purple', linewidth=2))
    ax.text(0.5, recall_05 - 0.03, f'F2: {f2_05:.3f}\nRecall: {recall_05:.3f}', 
            fontsize=8, ha='center', color='gray', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Set labels and title
    ax.set_xlabel('Probability Threshold', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    optimal_f2 = f2_scores[optimal_idx]
    optimal_f3 = f3_scores[optimal_idx]
    optimal_f4 = f4_scores[optimal_idx]
    ax.set_title(f'{horizon}h Forecast (Radius: {radius}km)\nF2 Optimal: {optimal_threshold:.3f} (F2: {optimal_f2:.3f}, Recall: {recalls[optimal_idx]:.3f})', 
                 fontsize=10, fontweight='bold', pad=10)
    ax.legend(loc='best', fontsize=9, framealpha=0.95, edgecolor='black', fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    print(f"   ✅ Optimal threshold (F2): {optimal_threshold:.3f}")
    print(f"   📊 At optimal: F2={f2_scores[optimal_idx]:.3f}, F3={f3_scores[optimal_idx]:.3f}, F4={f4_scores[optimal_idx]:.3f}")
    print(f"   📊 Recall={recalls[optimal_idx]:.3f}, Precision={precisions[optimal_idx]:.3f}, Accuracy={accuracies[optimal_idx]:.4f}")

plt.tight_layout()

# Save figure
output_dir = project_root / "docs" / "figures" / "v2"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "threshold_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Threshold analysis plot saved: {output_path}")

