#!/usr/bin/env python3
"""Generate Matrix C feature category importance analysis and figures."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.manuscript_style import (
    apply_matplotlib_style,
    format_axes,
)

# Best configurations for Matrix C (from supplementary_table_S3_best_configurations.csv)
BEST_CONFIGS = {
    3: {"radius": 60, "path": "experiments/lightgbm/raw/C/radius_60km/full_training/horizon_3h"},
    6: {"radius": 160, "path": "experiments/lightgbm/raw/C/radius_160km/full_training/horizon_6h"},
    12: {"radius": 200, "path": "experiments/lightgbm/raw/C/radius_200km/full_training/horizon_12h"},
    24: {"radius": 180, "path": "experiments/lightgbm/raw/C/radius_180km/full_training/horizon_24h"},
}

HORIZONS = [3, 6, 12, 24]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "manuscript" / "figures" / "v2"
SUPPLEMENTARY_DIR = PROJECT_ROOT / "docs" / "manuscript" / "Supplementary"


def categorize_feature(feature_name: str) -> str:
    """Categorize a feature based on its name."""
    feature_lower = feature_name.lower()
    
    # Time features
    if any(x in feature_lower for x in ['hour', 'day_of_year', 'jul', 'day_of_week', 'month', 'season', 'is_night']):
        if '_neighbor_' in feature_lower:
            return "Spatial Aggregation Features"  # Time features aggregated spatially
        return "Time Features"
    
    # Spatial aggregation features (neighbor statistics)
    if '_neighbor_' in feature_lower:
        return "Spatial Aggregation Features"
    
    # Derived meteorological features
    if any(x in feature_lower for x in ['wind_chill', 'heat_index', 'soil_air_temp_diff', 'dew_point']):
        # But exclude if it's a raw feature name
        if feature_lower in ['dew point (c)', 'dew point']:
            return "Raw Features"
        return "Derived Meteorological Features"
    
    # Raw features (original CIMIS variables)
    raw_vars = [
        'air temp', 'soil temp', 'wind dir', 'wind speed', 'rel hum', 
        'sol rad', 'vap pres', 'eto', 'precip'
    ]
    if any(x in feature_lower for x in raw_vars):
        return "Raw Features"
    
    # Default to Other if not categorized
    return "Other Features"


def load_and_categorize_features(horizon: int, task: str) -> pd.DataFrame:
    """Load feature importance and categorize features."""
    config = BEST_CONFIGS[horizon]
    base_path = PROJECT_ROOT / config["path"]
    
    task_file = "frost_feature_importance.csv" if task == "frost_classification" else "temp_feature_importance.csv"
    file_path = base_path / task_file
    
    if not file_path.exists():
        raise FileNotFoundError(f"Feature importance file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df["category"] = df["feature"].apply(categorize_feature)
    df["horizon_h"] = horizon
    df["task"] = task
    df["radius_km"] = config["radius"]
    
    return df


def aggregate_category_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature importance by category."""
    category_df = (
        df.groupby("category")["importance_pct"]
        .sum()
        .reset_index()
        .rename(columns={"importance_pct": "cumulative_importance_pct"})
    )
    category_df["horizon_h"] = df["horizon_h"].iloc[0]
    category_df["task"] = df["task"].iloc[0]
    category_df["radius_km"] = df["radius_km"].iloc[0]
    
    return category_df


def generate_category_importance_table() -> pd.DataFrame:
    """Generate feature category importance table for all horizons and tasks."""
    all_data = []
    
    for horizon in HORIZONS:
        for task in ["frost_classification", "temperature_regression"]:
            df = load_and_categorize_features(horizon, task)
            category_df = aggregate_category_importance(df)
            all_data.append(category_df)
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def plot_matrix_c_feature_category_importance(output_path: Path, category_df: pd.DataFrame) -> None:
    """Generate Matrix C feature category importance figure in bar chart format."""
    tasks = ["frost_classification", "temperature_regression"]
    task_labels = {
        "frost_classification": "Frost Classification",
        "temperature_regression": "Temperature Regression",
    }
    
    apply_matplotlib_style()
    
    # Create figure with two subplots (stacked vertically like Figure 13)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for ax, task in zip(axes, tasks):
        task_df = category_df[category_df["task"] == task].copy()
        categories = sorted(task_df["category"].unique())
        
        # Prepare data for bar chart (x-axis = categories)
        x = np.arange(len(categories))
        width = 0.2  # Width of bars for each horizon
        horizon_labels = [f"{h}h" for h in HORIZONS]
        
        # Colors for different horizons (same as Figure 13)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red
        
        # Plot bars for each horizon
        for i, horizon in enumerate(HORIZONS):
            horizon_data = (
                task_df[task_df["horizon_h"] == horizon]
                .set_index("category")
                .reindex(categories)["cumulative_importance_pct"]
                .fillna(0)
            )
            
            values = [horizon_data.loc[cat] if cat in horizon_data.index else 0 for cat in categories]
            
            offset = (i - 1.5) * width
            ax.bar(x + offset, values, width, label=horizon_labels[i], 
                  color=colors[i], alpha=0.8)
        
        ax.set_xlabel("Feature Category", fontsize=11)
        ax.set_ylabel("Importance (%)", fontsize=11)
        ax.set_title(task_labels[task], fontsize=12, fontweight="bold")
        ax.set_xticks(x + width * 1.5 - width / 2)
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, max(task_df["cumulative_importance_pct"].max() * 1.1, 50))
        ax.legend(title="Forecast Horizon", fontsize=9, title_fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        format_axes(ax)
    
    plt.tight_layout(pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved Matrix C feature category importance figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Matrix C feature category importance analysis"
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=OUTPUT_DIR / "matrix_c_feature_category_importance.png",
        help="Output path for the figure",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=SUPPLEMENTARY_DIR / "supplementary_table_S8_matrix_c_feature_category_importance.csv",
        help="Output path for the supplementary table",
    )
    args = parser.parse_args()
    
    # Generate category importance table
    print("Generating Matrix C feature category importance analysis...")
    category_df = generate_category_importance_table()
    
    # Save supplementary table
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    category_df.to_csv(args.output_table, index=False)
    print(f"✅ Saved Matrix C feature category importance table to {args.output_table}")
    print(f"   Total rows: {len(category_df)}")
    print(f"   Categories: {category_df['category'].nunique()}")
    print(f"   Horizons: {category_df['horizon_h'].nunique()}")
    print(f"   Tasks: {category_df['task'].nunique()}")
    
    # Generate and save figure
    plot_matrix_c_feature_category_importance(args.output_figure, category_df)


if __name__ == "__main__":
    main()

