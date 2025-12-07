#!/usr/bin/env python3
"""Generate Matrix A feature importance figure across horizons."""

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
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "lightgbm" / "raw" / "A" / "full_training"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "manuscript" / "figures" / "v2"


def load_feature_importance(horizon: int, task: str = "frost") -> pd.DataFrame:
    """Load feature importance for a specific horizon."""
    file_path = EXPERIMENT_DIR / f"horizon_{horizon}h" / f"{task}_feature_importance.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Feature importance file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df["horizon"] = horizon
    df["task"] = task
    return df


def plot_matrix_a_feature_importance(output_path: Path) -> None:
    """Generate Matrix A feature importance figure across horizons."""
    # Load data for all horizons
    frost_data = []
    temp_data = []
    
    for horizon in HORIZONS:
        try:
            frost_df = load_feature_importance(horizon, task="frost")
            temp_df = load_feature_importance(horizon, task="temp")
            frost_data.append(frost_df)
            temp_data.append(temp_df)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
    
    if not frost_data:
        raise FileNotFoundError("No feature importance data found for Matrix A")
    
    # Combine data
    frost_combined = pd.concat(frost_data, ignore_index=True)
    temp_combined = pd.concat(temp_data, ignore_index=True)
    
    # Apply manuscript style
    apply_matplotlib_style()
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot frost classification
    plot_single_task(frost_combined, axes[0], "Frost Classification")
    
    # Plot temperature regression
    plot_single_task(temp_combined, axes[1], "Temperature Regression")
    
    plt.tight_layout(pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved Matrix A feature importance figure to {output_path}")


def plot_single_task(df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
    """Plot feature importance for a single task."""
    # Get unique features
    features = df["feature"].unique()
    
    # Prepare data for plotting
    horizon_labels = [f"{h}h" for h in HORIZONS]
    x = np.arange(len(features))
    width = 0.2  # Width of bars
    
    # Colors for different horizons
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red
    
    # Plot bars for each horizon
    for i, horizon in enumerate(HORIZONS):
        horizon_data = df[df["horizon"] == horizon].set_index("feature")
        importance_pct = [horizon_data.loc[f, "importance_pct"] if f in horizon_data.index else 0 
                         for f in features]
        
        offset = (i - 1.5) * width
        ax.bar(x + offset, importance_pct, width, label=horizon_labels[i], 
               color=colors[i], alpha=0.8)
    
    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("Importance (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=9)
    ax.legend(title="Forecast Horizon", fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    format_axes(ax)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Matrix A feature importance figure"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "matrix_a_feature_importance_by_horizon.png",
        help="Output path for the figure",
    )
    args = parser.parse_args()
    
    plot_matrix_a_feature_importance(args.output)


if __name__ == "__main__":
    main()

