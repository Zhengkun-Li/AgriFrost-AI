#!/usr/bin/env python3
"""Analyze different threshold selection strategies and their trade-offs."""

import json
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Optimal configurations
configs = [
    (3, 60),
    (6, 160),
    (12, 200),
    (24, 180)
]

print("=" * 80)
print("📊 Threshold Selection Strategy Analysis")
print("=" * 80)
print("\n🎯 Goal: Correctly predict frost events (minimize FN)")
print("   - TP (True Positive): Prediction=frost, Actual=frost ✅ (Our goal)")
print("   - FN (False Negative): Prediction=non-frost, Actual=frost ❌ (Must minimize)")
print("   - FP (False Positive): Prediction=frost, Actual=non-frost ⚠️ (Acceptable cost)")
print("   - TN (True Negative): Prediction=non-frost, Actual=non-frost (Less important)")
print()

thresholds = np.linspace(0.01, 0.99, 100)

strategies = {
    'Max Recall': lambda p, r, f1: r,  # Pure recall maximization
    'Max F1': lambda p, r, f1: f1,      # Balanced precision-recall
    'F1×Recall': lambda p, r, f1: f1 * r,  # Current method
    'F2 Score': lambda p, r, f1: 5 * (p * r) / (4 * p + r + 1e-10),  # F-beta with beta=2 (emphasizes recall)
    'F1.5 Score': lambda p, r, f1: 3.25 * (p * r) / (2.25 * p + r + 1e-10),  # F-beta with beta=1.5
}

for horizon, radius in configs:
    pred_path = project_root / f"experiments/lightgbm/raw/C/radius_{radius}km/full_training/horizon_{horizon}h/predictions.json"
    
    if not pred_path.exists():
        continue
    
    print(f"\n{'='*80}")
    print(f"📊 {horizon}h Forecast (Radius: {radius}km)")
    print(f"{'='*80}")
    
    with open(pred_path, 'r') as f:
        predictions = json.load(f)
    
    y_true = np.array(predictions['frost']['y_true'])
    y_proba = np.array(predictions['frost']['y_proba'])
    
    # Sample if too large
    if len(y_true) > 100000:
        sample_idx = np.random.choice(len(y_true), 100000, replace=False)
        y_true = y_true[sample_idx]
        y_proba = y_proba[sample_idx]
    
    results = {}
    
    for strategy_name, score_func in strategies.items():
        best_score = -1
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            score = score_func(precision, recall, f1)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'score': score
                }
        
        results[strategy_name] = best_metrics
    
    # Print comparison table
    print(f"\n{'Strategy':<15} {'Threshold':<12} {'Recall':<8} {'Precision':<10} {'F1':<8} {'Score':<10}")
    print("-" * 80)
    
    for strategy_name, metrics in results.items():
        print(f"{strategy_name:<15} {metrics['threshold']:>11.3f}  "
              f"{metrics['recall']:>7.3f}  {metrics['precision']:>9.3f}  "
              f"{metrics['f1']:>7.3f}  {metrics['score']:>9.4f}")
    
    # Highlight best strategies for frost prediction
    print(f"\n💡 Recommendation for {horizon}h:")
    print(f"   🎯 Best Recall: {results['Max Recall']['recall']:.3f} at threshold {results['Max Recall']['threshold']:.3f}")
    print(f"      → Captures {results['Max Recall']['recall']*100:.1f}% of frost events, but may have many false alarms")
    print(f"   ⚖️  Balanced (F1×Recall): {results['F1×Recall']['recall']:.3f} recall, {results['F1×Recall']['precision']:.3f} precision")
    print(f"      → Current method: balances capturing frost events while controlling false alarms")
    print(f"   📊 F2 Score: {results['F2 Score']['recall']:.3f} recall, {results['F2 Score']['precision']:.3f} precision")
    print(f"      → Alternative: emphasizes recall more than F1, but less than pure recall")

print("\n" + "="*80)
print("📝 Strategy Selection Rationale:")
print("="*80)
print("""
1. **Max Recall**: 
   - Maximizes correct frost predictions (minimizes FN)
   - But may cause too many false alarms (high FP)
   - Best when cost of missing frost >> cost of false alarms

2. **Max F1**: 
   - Balances precision and recall equally
   - May not emphasize recall enough for imbalanced data
   - Best when precision and recall are equally important

3. **F1×Recall (Current Method)**: 
   - Combines overall performance (F1) with recall emphasis
   - Better for imbalanced data than pure F1
   - Good default: balances capturing events while controlling false alarms

4. **F2 Score**: 
   - F-beta with beta=2, emphasizes recall 2× more than precision
   - More recall-focused than F1, less extreme than pure recall
   - Alternative if F1×Recall doesn't prioritize recall enough

5. **F1.5 Score**: 
   - Middle ground between F1 and F2
   - Moderate emphasis on recall
""")

print("="*80)
print("✅ For frost forecasting, F1×Recall is a good choice because:")
print("   - It emphasizes recall (critical: minimize missed frost events)")
print("   - But still considers precision (avoid too many false alarms)")
print("   - Better for imbalanced data than pure F1")
print("="*80)

