#!/usr/bin/env python3
"""Generate confusion matrix visualization using optimal threshold for classification results."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix

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

def find_optimal_threshold(y_true, y_proba, method='f1_recall'):
    """Find optimal threshold based on F1*Recall for imbalanced data.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        method: 'f1' for F1 score, 'f1_recall' for F1*Recall (better for imbalanced)
    
    Returns:
        optimal_threshold, best_f1, best_precision, best_recall
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    if method == 'f1_recall':
        # For imbalanced data, prioritize both F1 and recall
        # Use F1*Recall as the optimization metric
        scores = f1_scores * recall
    else:
        scores = f1_scores
    
    # Find threshold with best score
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    best_f1 = f1_score(y_true, y_pred_optimal, zero_division=0)
    best_precision = precision[best_idx] if best_idx < len(precision) else 0
    best_recall = recall[best_idx] if best_idx < len(recall) else 0
    
    return optimal_threshold, best_f1, best_precision, best_recall

print("=" * 70)
print("📊 Generating Classification Confusion Matrix Plot (Optimal Threshold)")
print("=" * 70)

# Create figure with 2x2 subplots with more spacing
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
axes = axes.flatten()

optimal_thresholds = {}

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
    
    # Find optimal threshold
    optimal_threshold, best_f1, best_precision, best_recall = find_optimal_threshold(
        y_true, y_proba, method='f1_recall'
    )
    optimal_thresholds[horizon] = optimal_threshold
    
    # Convert probabilities to binary predictions using optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tp = int(cm[1, 1])
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    
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
    
    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    ax.set_title(f'{horizon}h Forecast (Radius: {radius}km)\n'
                 f'Optimal Threshold: {optimal_threshold:.3f}, Accuracy: {accuracy:.4f}\n'
                 f'Precision: {best_precision:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}',
                 fontsize=11, fontweight='bold', pad=15)
    
    print(f"   ✅ Optimal threshold: {optimal_threshold:.3f}")
    print(f"   📊 TP: {tp:,}, TN: {tn:,}, FP: {fp:,}, FN: {fn:,}")
    print(f"   📊 Accuracy: {accuracy:.4f}, Precision: {best_precision:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")

# Adjust layout with more padding (no need for extra bottom margin since no text box)
plt.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.10, hspace=0.35, wspace=0.25)

# Save figure
output_dir = project_root / "docs" / "figures" / "v2"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "classification_confusion_matrix.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Classification confusion matrix plot (optimal threshold) saved: {output_path}")
print(f"\n📊 Optimal thresholds: {optimal_thresholds}")

