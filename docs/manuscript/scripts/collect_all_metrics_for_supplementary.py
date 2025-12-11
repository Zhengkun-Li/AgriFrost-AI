#!/usr/bin/env python3
"""Collect all metrics from LightGBM ABC experiments and save to supplementary directory."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
SUPP_DIR = PROJECT_ROOT / "docs/manuscript/Supplementary_lighgbm_abc"
SUPP_DIR.mkdir(parents=True, exist_ok=True)

def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    if not metrics_path.exists():
        return {}
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {metrics_path}: {e}")
        return {}

def collect_metrics_from_dir(base_dir: Path, matrix: str, radius: str, training_type: str):
    """Collect metrics from a specific directory structure.
    
    Handles two directory structures:
    1. base_dir/full_training/horizon_* (balanced training)
    2. base_dir/horizon_* (unbalanced training, direct structure)
    """
    results = []
    
    # Try structure 1: base_dir/full_training/horizon_*
    training_dir = base_dir / "full_training"
    if training_dir.exists() and any(training_dir.glob("horizon_*")):
        horizon_base = training_dir
    # Try structure 2: base_dir/horizon_* (direct)
    elif any(base_dir.glob("horizon_*")):
        horizon_base = base_dir
    else:
        return results
    
    for horizon_dir in sorted(horizon_base.glob("horizon_*")):
        horizon = horizon_dir.name.replace("horizon_", "").replace("h", "")
        
        frost_metrics = load_metrics(horizon_dir / "frost_metrics.json")
        temp_metrics = load_metrics(horizon_dir / "temp_metrics.json")
        
        if frost_metrics or temp_metrics:
            result = {
                "Matrix": matrix,
                "Radius_km": radius,
                "Horizon_h": horizon,
                "Training": training_type,
            }
            
            # Frost classification metrics
            if frost_metrics:
                result.update({
                    "Frost_ROC_AUC": frost_metrics.get("roc_auc", None),
                    "Frost_PR_AUC": frost_metrics.get("pr_auc", None),
                    "Frost_Recall": frost_metrics.get("recall", None),
                    "Frost_Precision": frost_metrics.get("precision", None),
                    "Frost_F2": frost_metrics.get("f2_score", None),
                    "Frost_Brier_Score": frost_metrics.get("brier_score", None),
                    "Frost_ECE": frost_metrics.get("ece", None),
                })
            
            # Temperature regression metrics
            if temp_metrics:
                result.update({
                    "Temp_MAE": temp_metrics.get("mae", None),
                    "Temp_RMSE": temp_metrics.get("rmse", None),
                    "Temp_R2": temp_metrics.get("r2", None),
                })
            
            results.append(result)
    
    return results

def collect_all_metrics():
    """Collect all metrics from all experiments."""
    experiments_root = PROJECT_ROOT / "experiments/lightgbm"
    
    all_results = []
    
    # Matrix A experiments (both balanced and unbalanced)
    matrix_a_base = experiments_root / "raw/A"
    
    # Balanced training
    balanced_dir = matrix_a_base / "full_training_balance"
    if balanced_dir.exists():
        all_results.extend(collect_metrics_from_dir(balanced_dir, "A", "--", "Balanced"))
    
    # Unbalanced training (if exists and different from balanced)
    unbalanced_dir = matrix_a_base / "full_training"
    if unbalanced_dir.exists() and unbalanced_dir != balanced_dir:
        all_results.extend(collect_metrics_from_dir(unbalanced_dir, "A", "--", "Unbalanced"))
    
    # Matrix B experiments (both balanced and unbalanced)
    # Check both raw/B and feature_engineering/B
    for matrix_b_base in [experiments_root / "raw/B", experiments_root / "feature_engineering/B"]:
        if not matrix_b_base.exists():
            continue
        
        # Balanced training (full_training_balance)
        balanced_dir = matrix_b_base / "full_training_balance"
        if balanced_dir.exists():
            all_results.extend(collect_metrics_from_dir(balanced_dir, "B", "--", "Balanced"))
        
        # Unbalanced training (full_training, but not threshold or other variants)
        unbalanced_dir = matrix_b_base / "full_training"
        if unbalanced_dir.exists() and "threshold" not in str(unbalanced_dir) and unbalanced_dir != balanced_dir:
            all_results.extend(collect_metrics_from_dir(unbalanced_dir, "B", "--", "Unbalanced"))
    
    # Matrix C experiments (with different radii, both balanced and unbalanced)
    matrix_c_base = experiments_root / "raw/C"
    
    # Balanced training (full_training_balance_*km)
    for radius_dir in sorted(matrix_c_base.glob("full_training_balance_*km")):
        radius = radius_dir.name.replace("full_training_balance_", "").replace("km", "")
        all_results.extend(collect_metrics_from_dir(radius_dir, "C", radius, "Balanced"))
    
    # Unbalanced training (full_training_*km or radius_*km/full_training)
    for radius_dir in sorted(matrix_c_base.glob("radius_*km")):
        radius = radius_dir.name.replace("radius_", "").replace("km", "")
        all_results.extend(collect_metrics_from_dir(radius_dir, "C", radius, "Unbalanced"))
    
    # Also check for full_training directories without balance prefix
    for radius_dir in sorted(matrix_c_base.glob("full_training_*km")):
        if "balance" not in radius_dir.name:
            radius = radius_dir.name.replace("full_training_", "").replace("km", "")
            all_results.extend(collect_metrics_from_dir(radius_dir, "C", radius, "Unbalanced"))
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by Matrix, Horizon, Radius
    df = df.sort_values(["Matrix", "Horizon_h", "Radius_km"])
    
    # Save to CSV
    output_path = SUPP_DIR / "all_metrics_lightgbm_abc.csv"
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"✅ Saved all metrics to: {output_path}")
    print(f"   Total experiments: {len(df)}")
    
    # Also create separate tables for each matrix
    for matrix in ["A", "B", "C"]:
        matrix_df = df[df["Matrix"] == matrix].copy()
        if len(matrix_df) > 0:
            matrix_path = SUPP_DIR / f"metrics_matrix_{matrix}.csv"
            matrix_df.to_csv(matrix_path, index=False, float_format='%.4f')
            print(f"✅ Saved Matrix {matrix} metrics to: {matrix_path}")
    
    # Create summary table (best configuration per matrix-horizon)
    summary_results = []
    for matrix in ["A", "B", "C"]:
        matrix_df = df[df["Matrix"] == matrix].copy()
        for horizon in ["3", "6", "12", "24"]:
            horizon_df = matrix_df[matrix_df["Horizon_h"] == horizon]
            if len(horizon_df) > 0:
                # For Matrix C, select best by PR-AUC; for A and B, there's only one
                if matrix == "C":
                    best = horizon_df.loc[horizon_df["Frost_PR_AUC"].idxmax()]
                else:
                    best = horizon_df.iloc[0]
                
                summary_results.append(best.to_dict())
    
    summary_df = pd.DataFrame(summary_results)
    summary_path = SUPP_DIR / "summary_best_configurations.csv"
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"✅ Saved summary (best configurations) to: {summary_path}")
    
    return df

if __name__ == "__main__":
    df = collect_all_metrics()
    print(f"\n✅ All metrics collected and saved to: {SUPP_DIR}")

