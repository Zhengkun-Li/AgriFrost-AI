#!/usr/bin/env python3
"""Analyze distance sensitivity results for Matrix Cell C.

This script:
1. Loads all metrics from different distance configurations
2. Identifies optimal radius for each horizon
3. Generates summary report
4. Creates visualizations
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_all_metrics(base_dir: Path, horizons: List[int], radii: List[int]) -> Dict:
    """Load all metrics from different configurations."""
    results = {}
    
    for horizon in horizons:
        results[horizon] = {}
        for radius in radii:
            if radius == 0:
                horizon_dir = base_dir / "full_training" / f"horizon_{horizon}h"
            else:
                horizon_dir = base_dir / f"radius_{radius}km" / "full_training" / f"horizon_{horizon}h"
            
            frost_metrics_file = horizon_dir / "frost_metrics.json"
            temp_metrics_file = horizon_dir / "temp_metrics.json"
            
            if frost_metrics_file.exists() and temp_metrics_file.exists():
                with open(frost_metrics_file) as f:
                    frost_metrics = json.load(f)
                with open(temp_metrics_file) as f:
                    temp_metrics = json.load(f)
                
                results[horizon][radius] = {
                    "frost": frost_metrics,
                    "temp": temp_metrics,
                    "radius_km": radius
                }
    
    return results

def create_summary_dataframe(results: Dict) -> pd.DataFrame:
    """Create a summary DataFrame from results."""
    rows = []
    
    for horizon, radius_data in results.items():
        for radius, metrics in radius_data.items():
            rows.append({
                "horizon": f"{horizon}h",
                "radius_km": radius,
                "roc_auc": metrics["frost"].get("roc_auc", np.nan),
                "pr_auc": metrics["frost"].get("pr_auc", np.nan),
                "brier_score": metrics["frost"].get("brier_score", np.nan),
                "f1": metrics["frost"].get("f1", np.nan),
                "precision": metrics["frost"].get("precision", np.nan),
                "recall": metrics["frost"].get("recall", np.nan),
                "temp_rmse": metrics["temp"].get("rmse", np.nan),
                "temp_mae": metrics["temp"].get("mae", np.nan),
                "temp_r2": metrics["temp"].get("r2", np.nan),
            })
    
    df = pd.DataFrame(rows)
    return df

def find_optimal_radius(df: pd.DataFrame, horizon: str) -> Dict:
    """Find optimal radius for a given horizon based on multiple metrics."""
    horizon_df = df[df["horizon"] == horizon].copy()
    
    # Normalize metrics (higher is better for most, lower is better for some)
    # For frost prediction, prioritize ROC-AUC and PR-AUC
    # For temperature, prioritize lower RMSE
    
    # Weighted score: 40% ROC-AUC, 30% PR-AUC, 30% (1 - normalized RMSE)
    if len(horizon_df) > 0:
        max_roc = horizon_df["roc_auc"].max()
        max_pr = horizon_df["pr_auc"].max()
        max_rmse = horizon_df["temp_rmse"].max()
        
        horizon_df["normalized_roc"] = horizon_df["roc_auc"] / max_roc if max_roc > 0 else 0
        horizon_df["normalized_pr"] = horizon_df["pr_auc"] / max_pr if max_pr > 0 else 0
        horizon_df["normalized_rmse"] = 1 - (horizon_df["temp_rmse"] / max_rmse) if max_rmse > 0 else 0
        
        horizon_df["composite_score"] = (
            0.4 * horizon_df["normalized_roc"] +
            0.3 * horizon_df["normalized_pr"] +
            0.3 * horizon_df["normalized_rmse"]
        )
        
        optimal_idx = horizon_df["composite_score"].idxmax()
        optimal = horizon_df.loc[optimal_idx]
        
        return {
            "optimal_radius": int(optimal["radius_km"]),
            "roc_auc": optimal["roc_auc"],
            "pr_auc": optimal["pr_auc"],
            "temp_rmse": optimal["temp_rmse"],
            "temp_r2": optimal["temp_r2"],
            "composite_score": optimal["composite_score"]
        }
    
    return None

def plot_distance_sensitivity(df: pd.DataFrame, output_dir: Path):
    """Create distance sensitivity plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distance Sensitivity Analysis - Matrix Cell C', fontsize=16, fontweight='bold')
    
    horizons = sorted(df["horizon"].unique())
    
    # Plot 1: ROC-AUC vs Distance
    ax1 = axes[0, 0]
    for horizon in horizons:
        horizon_df = df[df["horizon"] == horizon].sort_values("radius_km")
        ax1.plot(horizon_df["radius_km"], horizon_df["roc_auc"], 
                marker='o', label=f'Horizon {horizon}', linewidth=2, markersize=6)
    ax1.set_xlabel('Radius (km)', fontsize=12)
    ax1.set_ylabel('ROC-AUC', fontsize=12)
    ax1.set_title('Frost Classification: ROC-AUC vs Distance', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='No spatial aggregation')
    
    # Plot 2: PR-AUC vs Distance
    ax2 = axes[0, 1]
    for horizon in horizons:
        horizon_df = df[df["horizon"] == horizon].sort_values("radius_km")
        ax2.plot(horizon_df["radius_km"], horizon_df["pr_auc"], 
                marker='s', label=f'Horizon {horizon}', linewidth=2, markersize=6)
    ax2.set_xlabel('Radius (km)', fontsize=12)
    ax2.set_ylabel('PR-AUC', fontsize=12)
    ax2.set_title('Frost Classification: PR-AUC vs Distance', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: Temperature RMSE vs Distance
    ax3 = axes[1, 0]
    for horizon in horizons:
        horizon_df = df[df["horizon"] == horizon].sort_values("radius_km")
        ax3.plot(horizon_df["radius_km"], horizon_df["temp_rmse"], 
                marker='^', label=f'Horizon {horizon}', linewidth=2, markersize=6)
    ax3.set_xlabel('Radius (km)', fontsize=12)
    ax3.set_ylabel('Temperature RMSE (°C)', fontsize=12)
    ax3.set_title('Temperature Regression: RMSE vs Distance', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 4: Temperature R² vs Distance
    ax4 = axes[1, 1]
    for horizon in horizons:
        horizon_df = df[df["horizon"] == horizon].sort_values("radius_km")
        ax4.plot(horizon_df["radius_km"], horizon_df["temp_r2"], 
                marker='d', label=f'Horizon {horizon}', linewidth=2, markersize=6)
    ax4.set_xlabel('Radius (km)', fontsize=12)
    ax4.set_ylabel('Temperature R²', fontsize=12)
    ax4.set_title('Temperature Regression: R² vs Distance', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "distance_sensitivity_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot: {plot_path}")
    
    plt.close()

def plot_optimal_radius_comparison(optimal_results: Dict, output_dir: Path):
    """Create a bar plot comparing optimal radii across horizons."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optimal Radius Analysis by Horizon', fontsize=16, fontweight='bold')
    
    horizons = sorted(optimal_results.keys())
    optimal_radii = [optimal_results[h]["optimal_radius"] for h in horizons]
    roc_aucs = [optimal_results[h]["roc_auc"] for h in horizons]
    pr_aucs = [optimal_results[h]["pr_auc"] for h in horizons]
    temp_rmses = [optimal_results[h]["temp_rmse"] for h in horizons]
    
    # Plot 1: Optimal Radius
    ax1 = axes[0, 0]
    bars = ax1.bar(horizons, optimal_radii, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    ax1.set_xlabel('Horizon', fontsize=12)
    ax1.set_ylabel('Optimal Radius (km)', fontsize=12)
    ax1.set_title('Optimal Radius by Horizon', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (h, r) in enumerate(zip(horizons, optimal_radii)):
        if r == 0:
            ax1.text(i, r + 5, f'{r}km\n(itself)', ha='center', va='bottom', fontsize=10)
        else:
            ax1.text(i, r + 5, f'{r}km', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: ROC-AUC at Optimal Radius
    ax2 = axes[0, 1]
    bars = ax2.bar(horizons, roc_aucs, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    ax2.set_xlabel('Horizon', fontsize=12)
    ax2.set_ylabel('ROC-AUC', fontsize=12)
    ax2.set_title('ROC-AUC at Optimal Radius', fontsize=13, fontweight='bold')
    ax2.set_ylim([0.98, 1.0])
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (h, auc) in enumerate(zip(horizons, roc_aucs)):
        ax2.text(i, auc + 0.001, f'{auc:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: PR-AUC at Optimal Radius
    ax3 = axes[1, 0]
    bars = ax3.bar(horizons, pr_aucs, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    ax3.set_xlabel('Horizon', fontsize=12)
    ax3.set_ylabel('PR-AUC', fontsize=12)
    ax3.set_title('PR-AUC at Optimal Radius', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, (h, auc) in enumerate(zip(horizons, pr_aucs)):
        ax3.text(i, auc + 0.01, f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Temperature RMSE at Optimal Radius
    ax4 = axes[1, 1]
    bars = ax4.bar(horizons, temp_rmses, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
    ax4.set_xlabel('Horizon', fontsize=12)
    ax4.set_ylabel('Temperature RMSE (°C)', fontsize=12)
    ax4.set_title('Temperature RMSE at Optimal Radius', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (h, rmse) in enumerate(zip(horizons, temp_rmses)):
        ax4.text(i, rmse + 0.05, f'{rmse:.2f}°C', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "optimal_radius_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot: {plot_path}")
    
    plt.close()

def generate_report(df: pd.DataFrame, optimal_results: Dict, output_dir: Path):
    """Generate a text report summarizing the analysis."""
    report_path = output_dir / "distance_sensitivity_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Distance Sensitivity Analysis Report - Matrix Cell C\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total configurations tested: {len(df)}\n")
        f.write(f"Horizons: {', '.join(sorted(df['horizon'].unique()))}\n")
        f.write(f"Radius configurations: {', '.join([str(r) for r in sorted(df['radius_km'].unique())])} km\n\n")
        
        f.write("OPTIMAL RADIUS BY HORIZON\n")
        f.write("-" * 80 + "\n")
        for horizon in sorted(optimal_results.keys()):
            opt = optimal_results[horizon]
            f.write(f"\nHorizon {horizon}:\n")
            f.write(f"  Optimal Radius: {opt['optimal_radius']} km")
            if opt['optimal_radius'] == 0:
                f.write(" (itself, no spatial aggregation)\n")
            else:
                f.write("\n")
            f.write(f"  ROC-AUC: {opt['roc_auc']:.4f}\n")
            f.write(f"  PR-AUC: {opt['pr_auc']:.4f}\n")
            f.write(f"  Temperature RMSE: {opt['temp_rmse']:.2f}°C\n")
            f.write(f"  Temperature R²: {opt['temp_r2']:.4f}\n")
            f.write(f"  Composite Score: {opt['composite_score']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED RESULTS BY HORIZON\n")
        f.write("=" * 80 + "\n\n")
        
        for horizon in sorted(df["horizon"].unique()):
            horizon_df = df[df["horizon"] == horizon].sort_values("radius_km")
            f.write(f"Horizon {horizon}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Radius':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'Temp RMSE':<12} {'Temp R²':<10}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in horizon_df.iterrows():
                radius_str = f"{int(row['radius_km'])}km"
                if row['radius_km'] == 0:
                    radius_str = "0km (itself)"
                f.write(f"{radius_str:<10} {row['roc_auc']:<10.4f} {row['pr_auc']:<10.4f} "
                       f"{row['temp_rmse']:<12.2f} {row['temp_r2']:<10.4f}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("INSIGHTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Calculate improvements
        for horizon in sorted(optimal_results.keys()):
            horizon_df = df[df["horizon"] == horizon]
            baseline = horizon_df[horizon_df["radius_km"] == 0].iloc[0]
            optimal = optimal_results[horizon]
            
            roc_improvement = optimal["roc_auc"] - baseline["roc_auc"]
            pr_improvement = optimal["pr_auc"] - baseline["pr_auc"]
            rmse_improvement = baseline["temp_rmse"] - optimal["temp_rmse"]
            
            f.write(f"Horizon {horizon}:\n")
            if optimal["optimal_radius"] > 0:
                f.write(f"  Spatial aggregation improves ROC-AUC by {roc_improvement:.4f} "
                       f"({roc_improvement/baseline['roc_auc']*100:.2f}%)\n")
                f.write(f"  Spatial aggregation improves PR-AUC by {pr_improvement:.4f} "
                       f"({pr_improvement/baseline['pr_auc']*100:.2f}%)\n")
                f.write(f"  Spatial aggregation reduces RMSE by {rmse_improvement:.2f}°C "
                       f"({rmse_improvement/baseline['temp_rmse']*100:.2f}%)\n")
            else:
                f.write("  No spatial aggregation (radius=0km) is optimal\n")
            f.write("\n")
    
    print(f"✅ Saved report: {report_path}")

def main():
    """Main analysis function."""
    base_dir = Path("experiments/lightgbm/raw/C")
    output_dir = Path("results/distance_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    horizons = [3, 6, 12, 24]
    radii = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    
    print("=" * 80)
    print("Distance Sensitivity Analysis - Matrix Cell C")
    print("=" * 80)
    print()
    
    # Load all metrics
    print("Loading metrics...")
    results = load_all_metrics(base_dir, horizons, radii)
    print(f"✅ Loaded metrics for {sum(len(r) for r in results.values())} configurations")
    
    # Create summary DataFrame
    print("\nCreating summary DataFrame...")
    df = create_summary_dataframe(results)
    print(f"✅ Created DataFrame with {len(df)} rows")
    
    # Save DataFrame
    df_path = output_dir / "distance_sensitivity_results.csv"
    df.to_csv(df_path, index=False)
    print(f"✅ Saved DataFrame: {df_path}")
    
    # Find optimal radius for each horizon
    print("\nFinding optimal radius for each horizon...")
    optimal_results = {}
    for horizon in horizons:
        optimal = find_optimal_radius(df, f"{horizon}h")
        if optimal:
            optimal_results[f"{horizon}h"] = optimal
            print(f"  Horizon {horizon}h: Optimal radius = {optimal['optimal_radius']}km "
                  f"(ROC-AUC={optimal['roc_auc']:.4f}, PR-AUC={optimal['pr_auc']:.4f})")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_distance_sensitivity(df, output_dir)
    plot_optimal_radius_comparison(optimal_results, output_dir)
    
    # Generate report
    print("\nGenerating report...")
    generate_report(df, optimal_results, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("  - distance_sensitivity_results.csv")
    print("  - distance_sensitivity_analysis.png")
    print("  - optimal_radius_analysis.png")
    print("  - distance_sensitivity_report.txt")

if __name__ == "__main__":
    main()

