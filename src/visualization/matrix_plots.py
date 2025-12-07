"""Matrix visualization plots."""

import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.visualization.plots import Plotter

_logger = logging.getLogger(__name__)


def plot_matrix_summary(
    summary_df: pd.DataFrame,
    output_path: Path,
    metrics: List[str] = ["mae", "rmse", "roc_auc"],
    style: str = "matplotlib"
) -> None:
    """Plot matrix summary.
    
    Args:
        summary_df: Summary DataFrame from build_matrix_summary
        output_path: Path to save plot
        metrics: List of metrics to plot (default: mae, rmse, roc_auc)
        style: Plotting style ("matplotlib" or "plotly")
    """
    plotter = Plotter(style=style)
    
    # Prepare metrics dict for plot_metrics_comparison
    # Group by matrix_cell and track, extract metrics
    models = []
    metrics_dict = {}
    
    for _, row in summary_df.iterrows():
        model_name = f"{row.get('matrix_cell', '?')}_{row.get('track', '?')}"
        models.append(model_name)
        
        if model_name not in metrics_dict:
            metrics_dict[model_name] = {}
        
        # Extract metrics (handle nested or flat structure)
        for metric in metrics:
            # Try flat column first
            if metric in row.index:
                metrics_dict[model_name][metric] = row[metric]
            # Try nested (frost_metrics_mae, temp_metrics_mae)
            else:
                for prefix in ["frost_metrics_", "temp_metrics_"]:
                    nested_col = f"{prefix}{metric}"
                    if nested_col in row.index and pd.notna(row[nested_col]):
                        # Use first non-null value
                        if metric not in metrics_dict[model_name]:
                            metrics_dict[model_name][metric] = row[nested_col]
    
    if not metrics_dict:
        _logger.warning("No metrics found for plotting")
        return
    
    # Plot comparison
    plotter.plot_metrics_comparison(
        metrics=metrics_dict,
        title="Matrix Summary",
        save_path=output_path,
        show=False
    )
    
    _logger.info(f"Matrix summary plot saved to {output_path}")


def plot_spatial_sensitivity(
    sensitivity_df: pd.DataFrame,
    output_path: Path,
    param_name: str = "radius_km",
    style: str = "matplotlib"
) -> None:
    """Plot spatial sensitivity.
    
    Args:
        sensitivity_df: Sensitivity DataFrame from build_spatial_sensitivity
        output_path: Path to save plot
        param_name: Parameter name ("radius_km" or "knn_k")
        style: Plotting style ("matplotlib" or "plotly")
    """
    plotter = Plotter(style=style)
    
    # Extract metrics for plotting
    metrics = ["mae", "rmse", "roc_auc"]
    metrics_dict = {}
    
    for _, row in sensitivity_df.iterrows():
        param_value = row[param_name]
        model_name = f"{param_name}={param_value}"
        metrics_dict[model_name] = {}
        
        for metric in metrics:
            if metric in row.index:
                metrics_dict[model_name][metric] = row[metric]
            else:
                for prefix in ["frost_metrics_", "temp_metrics_"]:
                    nested_col = f"{prefix}{metric}"
                    if nested_col in row.index and pd.notna(row[nested_col]):
                        if metric not in metrics_dict[model_name]:
                            metrics_dict[model_name][metric] = row[nested_col]
    
    if not metrics_dict:
        _logger.warning("No metrics found for plotting")
        return
    
    plotter.plot_metrics_comparison(
        metrics=metrics_dict,
        title=f"Spatial Sensitivity ({param_name})",
        save_path=output_path,
        show=False
    )
    
    _logger.info(f"Spatial sensitivity plot saved to {output_path}")

