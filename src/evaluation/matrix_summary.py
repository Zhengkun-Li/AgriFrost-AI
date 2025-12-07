"""
Matrix summary and experiment result loading.

⚠️ Hard constraint: Only work through run_metadata.json + metrics.json, no path parsing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from src.utils.metadata import ExperimentMetadata

_logger = logging.getLogger(__name__)


def load_experiment_results(
    model_dirs: List[Path],
    filter_config: Optional[Dict] = None,
    include_legacy: bool = True
) -> pd.DataFrame:
    """Load experiment results (enforced to use metadata).
    
    ⚠️ Hard constraint: Only through run_metadata.json + metrics.json, no path parsing.
    
    Args:
        model_dirs: List of model directories
        filter_config: Filter configuration (matrix_cell, track, horizon, etc.)
        include_legacy: Whether to include legacy runs (matrix_cell=None experiments)
    
    Returns:
        Merged metrics DataFrame (contains metadata columns)
    
    Raises:
        FileNotFoundError: If a model_dir is missing run_metadata.json (invalid run)
        ValueError: If metadata format is invalid
    """
    results = []
    
    for model_dir in model_dirs:
        model_dir = Path(model_dir)
        
        # 1. Read run_metadata.json (must exist)
        metadata_path = model_dir / "run_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Invalid run: {model_dir} missing run_metadata.json. "
                "All experiments must write ExperimentMetadata."
            )
        
        # Use ExperimentMetadata.load() (unidirectional import, avoid circular dependency)
        try:
            metadata = ExperimentMetadata.load(metadata_path)
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid metadata format in {metadata_path}: {e}")
        
        # 2. Skip legacy runs (if not including)
        if not include_legacy and metadata.is_legacy_run():
            _logger.debug(f"Skipping legacy run: {model_dir}")
            continue
        
        # 3. Read metrics.json (must exist)
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Invalid run: {model_dir} missing metrics.json"
            )
        
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_dict = json.load(f)
        
        # 4. Apply filter (based on metadata, not path parsing)
        if filter_config:
            if not _matches_filter(metadata, filter_config):
                continue
        
        # 5. Merge metadata + metrics
        row = {
            **metadata.to_dict(),
            **metrics_dict,
            'model_dir': str(model_dir)
        }
        results.append(row)
    
    if not results:
        _logger.warning("No experiment results found after filtering")
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def _matches_filter(metadata: ExperimentMetadata, filter_config: Dict) -> bool:
    """Check if metadata matches filter conditions (no path parsing)."""
    for key, value in filter_config.items():
        if hasattr(metadata, key):
            if getattr(metadata, key) != value:
                return False
    return True


def build_matrix_summary(
    results_df: pd.DataFrame,
    groupby_cols: List[str] = ["matrix_cell", "track", "horizon_h"]
) -> pd.DataFrame:
    """Build 2×2+1 matrix summary (using metadata columns).
    
    ⚠️ Automatically filters legacy runs (experiments with matrix_cell=None do not participate in matrix summary).
    
    Args:
        results_df: Results DataFrame from load_experiment_results
        groupby_cols: Columns to group by (default: matrix_cell, track, horizon_h)
    
    Returns:
        Summary DataFrame grouped by specified columns
    
    Raises:
        ValueError: If no standard experiments found
    """
    if results_df.empty:
        raise ValueError("Results DataFrame is empty")
    
    # 1. Only use standard experiments (filter legacy runs)
    standard_df = results_df[
        results_df["matrix_cell"].notna() & 
        results_df["track"].notna()
    ]
    
    if standard_df.empty:
        raise ValueError(
            "No standard experiments found for matrix summary. "
            "All runs are legacy runs (matrix_cell is None)."
        )
    
    # 2. Extract metrics (handle nested structure)
    # metrics.json may have frost_metrics/temp_metrics structure
    metric_cols = []
    for col in standard_df.columns:
        if col in ["mae", "rmse", "roc_auc"]:
            metric_cols.append(col)
        elif col in ["frost_metrics", "temp_metrics"]:
            # Handle nested metrics
            if len(standard_df) > 0 and isinstance(standard_df[col].iloc[0], dict):
                for metric_name in ["mae", "rmse", "roc_auc"]:
                    nested_col = f"{col}_{metric_name}"
                    def extract_metric(x):
                        """Extract metric from dict, return None if invalid."""
                        if isinstance(x, dict):
                            return x.get(metric_name)
                        return None
                    standard_df[nested_col] = standard_df[col].apply(extract_metric)
                    metric_cols.append(nested_col)
    
    # 3. Group by specified columns and aggregate
    agg_dict = {}
    for col in metric_cols:
        if col in standard_df.columns:
            agg_dict[col] = "mean"
    
    if not agg_dict:
        # Fallback: try to find any numeric columns
        numeric_cols = standard_df.select_dtypes(include=['number']).columns.tolist()
        for col in numeric_cols:
            if col not in groupby_cols and col != "horizon_h":
                agg_dict[col] = "mean"
    
    if not agg_dict:
        raise ValueError("No metric columns found for aggregation")
    
    summary = standard_df.groupby(groupby_cols).agg(agg_dict).reset_index()
    
    return summary


def build_spatial_sensitivity(
    results_df: pd.DataFrame,
    spatial_param: str = "radius_km"  # or "knn_k"
) -> pd.DataFrame:
    """Build spatial parameter sensitivity analysis (using metadata columns).
    
    ⚠️ Uses metadata.radius_km or metadata.knn_k, no path parsing.
    
    Args:
        results_df: Results DataFrame from load_experiment_results
        spatial_param: Spatial parameter name ("radius_km" or "knn_k")
    
    Returns:
        Sensitivity DataFrame grouped by spatial parameter
    
    Raises:
        ValueError: If no experiments with spatial parameter found
    """
    # Filter experiments with spatial parameter
    spatial_df = results_df[results_df[spatial_param].notna()]
    
    if spatial_df.empty:
        raise ValueError(f"No experiments with {spatial_param} found.")
    
    # Extract metrics (handle nested structure)
    metric_cols = []
    for col in spatial_df.columns:
        if col in ["mae", "rmse", "roc_auc"]:
            metric_cols.append(col)
        elif col in ["frost_metrics", "temp_metrics"]:
            # Check if column exists and has valid data
            if col in spatial_df.columns and len(spatial_df) > 0:
                first_val = spatial_df[col].iloc[0]
                if isinstance(first_val, dict):
                    for metric_name in ["mae", "rmse", "roc_auc"]:
                        nested_col = f"{col}_{metric_name}"
                        def extract_metric(x):
                            """Extract metric from dict, return None if invalid."""
                            if isinstance(x, dict):
                                return x.get(metric_name)
                            return None
                        spatial_df[nested_col] = spatial_df[col].apply(extract_metric)
                        metric_cols.append(nested_col)
    
    # Group by spatial parameter
    agg_dict = {}
    for col in metric_cols:
        if col in spatial_df.columns:
            agg_dict[col] = "mean"
    
    if not agg_dict:
        numeric_cols = spatial_df.select_dtypes(include=['number']).columns.tolist()
        for col in numeric_cols:
            if col not in [spatial_param, "horizon_h"]:
                agg_dict[col] = "mean"
    
    if not agg_dict:
        raise ValueError("No metric columns found for aggregation")
    
    sensitivity = spatial_df.groupby(spatial_param).agg(agg_dict).reset_index()
    
    return sensitivity

