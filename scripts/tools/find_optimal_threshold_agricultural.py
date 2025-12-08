#!/usr/bin/env python3
"""Find optimal threshold for frost forecasting using agricultural-focused strategies."""

import json
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def find_threshold_method1_fixed_recall(y_true, y_proba, recall_target=0.90):
    """
    Method 1: Fixed High Recall, Maximize Precision
    - First ensure Recall >= recall_target (e.g., 90% or 95%)
    - Then select threshold with maximum Precision in that range
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_precision = -1
    best_threshold = 0.5
    best_recall = 0
    valid_thresholds = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        
        if recall >= recall_target:
            valid_thresholds.append({
                'threshold': threshold,
                'recall': recall,
                'precision': precision
            })
            if precision > best_precision:
                best_precision = precision
                best_threshold = threshold
                best_recall = recall
    
    return best_threshold, best_recall, best_precision, len(valid_thresholds)

def find_threshold_method2_cost_function(y_true, y_proba, cost_fp=10, cost_fn=1000):
    """
    Method 2: Minimize Agricultural Loss Function
    - cost_fp: Cost of false positive (false alarm, e.g., running fans/water)
    - cost_fn: Cost of false negative (missed frost, crop loss)
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_cost = float('inf')
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        total_cost = fp * cost_fp + fn * cost_fn
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = threshold
            recall = recall_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            best_metrics = {
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'cost': total_cost,
                'fp': fp,
                'fn': fn,
                'tp': tp,
                'tn': tn
            }
    
    return best_threshold, best_metrics

def find_threshold_method3_fbeta(y_true, y_proba, beta=2):
    """
    Method 3: Maximize F-beta score (beta > 1 emphasizes recall)
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_fbeta = -1
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'fbeta': fbeta
            }
    
    return best_threshold, best_metrics

# Optimal configurations
configs = [
    (3, 60),
    (6, 160),
    (12, 200),
    (24, 180)
]

print("=" * 90)
print("🌾 Agricultural-Focused Optimal Threshold Selection")
print("=" * 90)
print("\n📋 Strategy Comparison:")
print("   Method 1: Fixed Recall (≥90% or ≥95%), Maximize Precision")
print("   Method 2: Minimize Cost Function (C_FP × FP + C_FN × FN)")
print("   Method 3: Maximize F-beta Score (F2 or F3)")
print()

results_summary = {}

for horizon, radius in configs:
    pred_path = project_root / f"experiments/lightgbm/raw/C/radius_{radius}km/full_training/horizon_{horizon}h/predictions.json"
    
    if not pred_path.exists():
        continue
    
    print(f"\n{'='*90}")
    print(f"📊 {horizon}h Forecast (Radius: {radius}km)")
    print(f"{'='*90}")
    
    with open(pred_path, 'r') as f:
        predictions = json.load(f)
    
    y_true = np.array(predictions['frost']['y_true'])
    y_proba = np.array(predictions['frost']['y_proba'])
    
    # Sample if too large
    if len(y_true) > 100000:
        sample_idx = np.random.choice(len(y_true), 100000, replace=False)
        y_true = y_true[sample_idx]
        y_proba = y_proba[sample_idx]
    
    # Method 1: Fixed Recall 90%
    thresh_90, recall_90, precision_90, n_valid_90 = find_threshold_method1_fixed_recall(
        y_true, y_proba, recall_target=0.90
    )
    
    # Method 1: Fixed Recall 95%
    thresh_95, recall_95, precision_95, n_valid_95 = find_threshold_method1_fixed_recall(
        y_true, y_proba, recall_target=0.95
    )
    
    # Method 2: Cost Function (C_FP=10, C_FN=1000)
    thresh_cost, metrics_cost = find_threshold_method2_cost_function(
        y_true, y_proba, cost_fp=10, cost_fn=1000
    )
    
    # Method 3: F2 Score
    thresh_f2, metrics_f2 = find_threshold_method3_fbeta(y_true, y_proba, beta=2)
    
    # Method 3: F3 Score
    thresh_f3, metrics_f3 = find_threshold_method3_fbeta(y_true, y_proba, beta=3)
    
    print(f"\n{'Method':<30} {'Threshold':<12} {'Recall':<10} {'Precision':<12} {'Additional Info':<20}")
    print("-" * 90)
    print(f"{'Method 1: Recall≥90%, Max Prec':<30} {thresh_90:>11.3f}  {recall_90:>9.3f}  {precision_90:>11.3f}  ({n_valid_90} valid thresholds)")
    print(f"{'Method 1: Recall≥95%, Max Prec':<30} {thresh_95:>11.3f}  {recall_95:>9.3f}  {precision_95:>11.3f}  ({n_valid_95} valid thresholds)")
    print(f"{'Method 2: Cost (FP=10, FN=1000)':<30} {thresh_cost:>11.3f}  {metrics_cost['recall']:>9.3f}  {metrics_cost['precision']:>11.3f}  Cost: {metrics_cost['cost']:.0f}")
    print(f"{'Method 3: F2 Score':<30} {thresh_f2:>11.3f}  {metrics_f2['recall']:>9.3f}  {metrics_f2['precision']:>11.3f}  F2: {metrics_f2['fbeta']:.3f}")
    print(f"{'Method 3: F3 Score':<30} {thresh_f3:>11.3f}  {metrics_f3['recall']:>9.3f}  {metrics_f3['precision']:>11.3f}  F3: {metrics_f3['fbeta']:.3f}")
    
    # Recommendation
    print(f"\n💡 Recommendation for {horizon}h:")
    if n_valid_90 > 0:
        print(f"   ✅ Method 1 (Recall≥90%): Threshold {thresh_90:.3f}")
        print(f"      → Captures {recall_90*100:.1f}% of frost events, Precision: {precision_90:.3f}")
    else:
        print(f"   ⚠️  Cannot achieve Recall≥90% with any threshold")
    
    if n_valid_95 > 0:
        print(f"   ✅ Method 1 (Recall≥95%): Threshold {thresh_95:.3f}")
        print(f"      → Captures {recall_95*100:.1f}% of frost events, Precision: {precision_95:.3f}")
    else:
        print(f"   ⚠️  Cannot achieve Recall≥95% with any threshold")
    
    print(f"   💰 Method 2 (Cost Function): Threshold {thresh_cost:.3f}")
    print(f"      → Total cost: {metrics_cost['cost']:.0f} (FP: {metrics_cost['fp']}, FN: {metrics_cost['fn']})")
    print(f"      → Recall: {metrics_cost['recall']:.3f}, Precision: {metrics_cost['precision']:.3f}")
    
    results_summary[horizon] = {
        'method1_90': (thresh_90, recall_90, precision_90),
        'method1_95': (thresh_95, recall_95, precision_95),
        'method2': (thresh_cost, metrics_cost),
        'method3_f2': (thresh_f2, metrics_f2),
        'method3_f3': (thresh_f3, metrics_f3)
    }

print("\n" + "="*90)
print("📝 Strategy Selection Guide:")
print("="*90)
print("""
🎯 **Method 1 (RECOMMENDED for most cases)**: Fixed Recall, Maximize Precision
   - Step 1: Set minimum Recall requirement (90% or 95%)
   - Step 2: Among all thresholds meeting Recall requirement, choose max Precision
   - Why: Ensures we don't miss frost events (high Recall) while minimizing false alarms (max Precision)
   - Best for: General agricultural applications where missing frost is critical

💰 **Method 2**: Minimize Agricultural Cost Function
   - Minimize: C_FP × FP + C_FN × FN
   - Where: C_FP = cost of false alarm (e.g., 10), C_FN = cost of missed frost (e.g., 1000)
   - Why: Directly optimizes economic impact
   - Best for: When you can quantify actual costs (energy, labor, crop value)

📊 **Method 3**: F-beta Score (F2 or F3)
   - F2: Recall weighted 4× more than Precision
   - F3: Recall weighted 10× more than Precision
   - Why: Standard metric for imbalanced classification with recall emphasis
   - Best for: When you want a standard metric but need recall emphasis
""")

print("="*90)
print("✅ Final Recommendation:")
print("="*90)
print("For frost forecasting, use **Method 1 with Recall≥90%** as the default strategy.")
print("This ensures:")
print("  ✓ At least 90% of frost events are captured (minimizes crop loss)")
print("  ✓ Maximum precision among all 90%+ recall thresholds (minimizes false alarms)")
print("  ✓ Simple, interpretable, and directly addresses agricultural needs")
print("="*90)

