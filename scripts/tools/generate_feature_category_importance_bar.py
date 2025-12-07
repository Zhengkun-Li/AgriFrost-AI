#!/usr/bin/env python3
"""Generate feature category importance figure in bar chart format (like Figure 12)."""

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

HORIZONS = [3, 6, 12, 24]
CSV_PATH = PROJECT_ROOT / "docs" / "manuscript" / "Supplementary" / "supplementary_table_S5_feature_category_importance.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "manuscript" / "figures" / "v2"


def plot_feature_category_importance_bar(output_path: Path) -> None:
    """Generate feature category importance figure in bar chart format (like Figure 12)."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing feature category data: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    tasks = ["frost_classification", "temperature_regression"]
    task_labels = {
        "frost_classification": "Frost Classification",
        "temperature_regression": "Temperature Regression",
    }
    
    apply_matplotlib_style()
    
    # Create figure with two subplots (stacked vertically like Figure 12)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for ax, task in zip(axes, tasks):
        task_df = df[df["task"] == task].copy()
        categories = sorted(task_df["category"].unique())
        
        # Prepare data for bar chart (x-axis = categories, like Figure 12)
        x = np.arange(len(categories))
        width = 0.2  # Width of bars for each horizon
        horizon_labels = [f"{h}h" for h in HORIZONS]
        
        # Colors for different horizons (same as Figure 12)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red
        
        # Plot bars for each horizon
        for i, horizon in enumerate(HORIZONS):
            horizon_data = (
                task_df[task_df["horizon_h"] == horizon]
                .set_index("category")
                .reindex(categories)["cumulative_importance_pct"]
                .fillna(0)
            )
            
            # Use cumulative_importance_pct directly (it represents category importance, not truly cumulative)
            values = [horizon_data.loc[cat] if cat in horizon_data.index else 0 for cat in categories]
            
            offset = (i - 1.5) * width
            ax.bar(x + offset, values, width, label=horizon_labels[i], 
                  color=colors[i], alpha=0.8)
        
        ax.set_xlabel("Feature Category", fontsize=11)
        ax.set_ylabel("Importance (%)", fontsize=11)
        ax.set_title(task_labels[task], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
        ax.legend(title="Forecast Horizon", fontsize=9, title_fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        format_axes(ax)
    
    plt.tight_layout(pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved feature category importance bar chart to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate feature category importance figure in bar chart format"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "feature_category_cross_horizon_bar.png",
        help="Output path for the figure",
    )
    args = parser.parse_args()
    
    plot_feature_category_importance_bar(args.output)


if __name__ == "__main__":
    main()

