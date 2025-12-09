"""Verify that metrics calculations are correct.

This script checks:
1. ROC-AUC, PR-AUC, Brier Score, ECE calculations
2. MAE, RMSE calculations
3. Whether values are reasonable for the task
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import MetricsCalculator
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
)

def test_metrics_calculations():
    """Test that our metric calculations match sklearn implementations."""
    print("=" * 70)
    print("Testing Metrics Calculations")
    print("=" * 70)
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 10000
    n_positive = 100  # 1% positive class (imbalanced, similar to frost data)
    
    # Generate realistic predictions
    y_true = np.zeros(n_samples)
    y_true[:n_positive] = 1
    
    # Good model: high probabilities for positive class
    y_proba = np.random.beta(1, 10, n_samples)  # Most predictions are low
    y_proba[:n_positive] = np.random.beta(8, 2, n_positive)  # Positive class has high probabilities
    
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Temperature predictions (regression)
    y_temp_true = np.random.normal(5, 10, n_samples)  # Mean 5°C, std 10°C
    y_temp_pred = y_temp_true + np.random.normal(0, 1.5, n_samples)  # Good predictions (MAE ~1.5)
    
    print(f"\nTest Data:")
    print(f"  Samples: {n_samples}")
    print(f"  Positive class: {n_positive} ({n_positive/n_samples*100:.2f}%)")
    print(f"  Temperature range: {y_temp_true.min():.2f}°C to {y_temp_true.max():.2f}°C")
    
    # Test classification metrics
    print("\n" + "-" * 70)
    print("Classification Metrics (Frost)")
    print("-" * 70)
    
    # Our implementation
    our_metrics = MetricsCalculator.calculate_classification_metrics(
        y_true, y_pred, y_proba
    )
    our_prob_metrics = MetricsCalculator.calculate_probability_metrics(
        y_true, y_proba
    )
    our_metrics.update(our_prob_metrics)
    
    # Sklearn direct calculation
    sklearn_roc_auc = roc_auc_score(y_true, y_proba)
    sklearn_pr_auc = average_precision_score(y_true, y_proba)
    sklearn_brier = brier_score_loss(y_true, y_proba)
    
    print(f"\nROC-AUC:")
    print(f"  Our implementation: {our_metrics['roc_auc']:.6f}")
    print(f"  Sklearn direct:     {sklearn_roc_auc:.6f}")
    print(f"  Match: {'✓' if np.isclose(our_metrics['roc_auc'], sklearn_roc_auc, rtol=1e-6) else '✗'}")
    
    print(f"\nPR-AUC:")
    print(f"  Our implementation: {our_metrics['pr_auc']:.6f}")
    print(f"  Sklearn direct:     {sklearn_pr_auc:.6f}")
    print(f"  Match: {'✓' if np.isclose(our_metrics['pr_auc'], sklearn_pr_auc, rtol=1e-6) else '✗'}")
    
    print(f"\nBrier Score:")
    print(f"  Our implementation: {our_metrics['brier_score']:.6f}")
    print(f"  Sklearn direct:     {sklearn_brier:.6f}")
    print(f"  Match: {'✓' if np.isclose(our_metrics['brier_score'], sklearn_brier, rtol=1e-6) else '✗'}")
    
    print(f"\nECE:")
    print(f"  Our implementation: {our_metrics['ece']:.6f}")
    print(f"  (ECE is custom implementation, no sklearn equivalent)")
    
    # Test regression metrics
    print("\n" + "-" * 70)
    print("Regression Metrics (Temperature)")
    print("-" * 70)
    
    our_reg_metrics = MetricsCalculator.calculate_regression_metrics(
        y_temp_true, y_temp_pred
    )
    
    sklearn_mae = mean_absolute_error(y_temp_true, y_temp_pred)
    sklearn_rmse = np.sqrt(mean_squared_error(y_temp_true, y_temp_pred))
    
    print(f"\nMAE:")
    print(f"  Our implementation: {our_reg_metrics['mae']:.6f}")
    print(f"  Sklearn direct:     {sklearn_mae:.6f}")
    print(f"  Match: {'✓' if np.isclose(our_reg_metrics['mae'], sklearn_mae, rtol=1e-6) else '✗'}")
    
    print(f"\nRMSE:")
    print(f"  Our implementation: {our_reg_metrics['rmse']:.6f}")
    print(f"  Sklearn direct:     {sklearn_rmse:.6f}")
    print(f"  Match: {'✓' if np.isclose(our_reg_metrics['rmse'], sklearn_rmse, rtol=1e-6) else '✗'}")
    
    # Check if values are reasonable
    print("\n" + "=" * 70)
    print("Reasonableness Check")
    print("=" * 70)
    
    print(f"\nFor imbalanced classification (1% positive class):")
    print(f"  ROC-AUC: {our_metrics['roc_auc']:.4f} - {'✓ Reasonable' if 0.5 <= our_metrics['roc_auc'] <= 1.0 else '✗ Unreasonable'}")
    print(f"  PR-AUC:  {our_metrics['pr_auc']:.4f} - {'✓ Reasonable' if 0.0 <= our_metrics['pr_auc'] <= 1.0 else '✗ Unreasonable'}")
    print(f"  Brier:   {our_metrics['brier_score']:.4f} - {'✓ Reasonable' if 0.0 <= our_metrics['brier_score'] <= 1.0 else '✗ Unreasonable'}")
    print(f"  ECE:     {our_metrics['ece']:.4f} - {'✓ Reasonable' if 0.0 <= our_metrics['ece'] <= 1.0 else '✗ Unreasonable'}")
    
    print(f"\nFor temperature prediction:")
    print(f"  MAE:  {our_reg_metrics['mae']:.4f}°C - {'✓ Reasonable' if 0.0 <= our_reg_metrics['mae'] <= 20.0 else '✗ Unreasonable'}")
    print(f"  RMSE: {our_reg_metrics['rmse']:.4f}°C - {'✓ Reasonable' if 0.0 <= our_reg_metrics['rmse'] <= 20.0 else '✗ Unreasonable'}")
    
    return our_metrics, our_reg_metrics


def check_actual_results():
    """Check actual experiment results for reasonableness."""
    print("\n" + "=" * 70)
    print("Checking Actual Experiment Results")
    print("=" * 70)
    
    results_file = project_root / "results" / "matrix_horizon_metrics_summary.csv"
    if not results_file.exists():
        print(f"\n⚠️  Results file not found: {results_file}")
        return
    
    # Read CSV with header on row 2 (0-indexed: row 2 = index 2)
    df = pd.read_csv(results_file, skiprows=1)
    # First row is now the actual header
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    
    print(f"\nLoaded results from: {results_file}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check Matrix C results (best performing)
    matrix_c = df[df.iloc[:, 0] == 'C'].copy()  # First column is matrix_cell
    if len(matrix_c) == 0:
        print("\n⚠️  No Matrix C results found")
        return
    
    print("\n" + "-" * 70)
    print("Matrix C Results (Best Configuration)")
    print("-" * 70)
    
    for _, row in matrix_c.iterrows():
        horizon = row.iloc[1]  # Second column is horizon_h
        # Columns: matrix_cell, horizon_h, roc_auc_mean, roc_auc_max, pr_auc_mean, pr_auc_max, brier_mean, brier_min, temp_rmse_mean, temp_rmse_min
        roc_auc_max = float(row.iloc[3])  # roc_auc_max
        pr_auc_max = float(row.iloc[5])  # pr_auc_max
        brier_min = float(row.iloc[7])  # brier_score_min
        temp_rmse_min = float(row.iloc[9])  # temp_rmse_min
        
        print(f"\nHorizon {horizon}h:")
        print(f"  ROC-AUC (max): {roc_auc_max:.4f}")
        print(f"  PR-AUC (max):  {pr_auc_max:.4f}")
        print(f"  Brier (min):   {brier_min:.4f}")
        print(f"  Temp RMSE:     {temp_rmse_min:.4f}°C")
        
        # Reasonableness checks
        issues = []
        if roc_auc_max > 0.999:
            issues.append(f"⚠️  ROC-AUC very high ({roc_auc_max:.4f}) - check for data leakage")
        if pr_auc_max > 0.8:
            issues.append(f"⚠️  PR-AUC very high ({pr_auc_max:.4f}) - unusual for 0.87% positive class")
        if brier_min < 0.001:
            issues.append(f"⚠️  Brier Score very low ({brier_min:.4f}) - check predictions")
        if temp_rmse_min < 1.0:
            issues.append(f"⚠️  RMSE very low ({temp_rmse_min:.4f}°C) - check if using future data")
        
        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print(f"  ✓ Values appear reasonable")


def check_data_leakage():
    """Check for potential data leakage issues."""
    print("\n" + "=" * 70)
    print("Data Leakage Check")
    print("=" * 70)
    
    print("\nChecking data preparation code...")
    
    # Check if future features are included
    data_prep_file = project_root / "src" / "training" / "data_preparation.py"
    if data_prep_file.exists():
        with open(data_prep_file, 'r') as f:
            content = f.read()
            
        # Check for exclusion of future horizon labels
        if "not col.startswith('frost_')" in content and "not col.startswith('temp_')" in content:
            print("  ✓ Future horizon labels are excluded from features")
        else:
            print("  ⚠️  May not be excluding future horizon labels")
        
        # Check for date-based splitting
        if "time_split" in content or "time-based" in content.lower():
            print("  ✓ Time-based data splitting is used")
        else:
            print("  ⚠️  Time-based splitting may not be used")
    
    print("\nRecommendations:")
    print("  1. Verify that test set is truly time-held-out (15% most recent data)")
    print("  2. Check that no future information is included in features")
    print("  3. Verify that lag features only use past data")
    print("  4. Check that neighbor features only use same-time data (not future)")


if __name__ == "__main__":
    # Test metric calculations
    metrics, reg_metrics = test_metrics_calculations()
    
    # Check actual results
    check_actual_results()
    
    # Check for data leakage
    check_data_leakage()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nIf ROC-AUC > 0.99 and PR-AUC > 0.7 for imbalanced data (0.87% positive),")
    print("this could indicate:")
    print("  1. Data leakage (future information in features)")
    print("  2. Target leakage (using target-related features)")
    print("  3. Overfitting to test set")
    print("  4. Very easy task (but unlikely for weather forecasting)")
    print("\nRecommendation: Review feature engineering and data splitting logic.")

