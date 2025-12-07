"""Data preparation module for frost forecasting training.

This module handles:
- Data loading and cleaning
- Feature engineering
- Frost label creation
- Feature and target preparation
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
import pandas as pd
import numpy as np

_logger = logging.getLogger(__name__)

from src.data import DataPipeline
from src.data.frost_labels import FrostLabelGenerator
from src.data.preprocessors import preprocess_with_loso

DEFAULT_FEATURE_ENGINEERING_PARAMS = {
    "create_time_features": True,
    "create_lag_features": True,
    "create_rolling_features": True,
    "create_interaction_features": False,
    "lag_periods": [1, 3, 6, 12, 24],
    "rolling_windows": [3, 6, 12, 24],
}

DATA_CLEANING_CONFIG_MAP = {
    "raw": Path(project_root / "config" / "data_cleaning_raw.yaml"),
    "fe": Path(project_root / "config" / "data_cleaning_fe.yaml"),
    "graph": Path(project_root / "config" / "data_cleaning_graph.yaml"),
}


def _optimize_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _select_cleaning_config_path(
    use_feature_engineering: bool,
    matrix_cell: Optional[str],
    cleaning_config_path: Optional[Path],
) -> Path:
    if cleaning_config_path:
        return cleaning_config_path
    key = "fe" if use_feature_engineering else "raw"
    if matrix_cell:
        cell = matrix_cell.upper()
        if cell in {"A", "C"}:
            key = "raw"
        elif cell in {"B", "D"}:
            key = "fe"
        elif cell == "E":
            key = "graph"
    return DATA_CLEANING_CONFIG_MAP.get(key, DATA_CLEANING_CONFIG_MAP["fe"])


def _build_pipeline_config(
    use_feature_engineering: bool,
    feature_engineering_config: Optional[Dict] = None,
    matrix_cell: Optional[str] = None,
    cleaning_config_path: Optional[Path] = None,
) -> Dict:
    if feature_engineering_config:
        feature_cfg = dict(feature_engineering_config)
    else:
        feature_cfg = dict(DEFAULT_FEATURE_ENGINEERING_PARAMS)
    cleaning_path = _select_cleaning_config_path(
        use_feature_engineering=use_feature_engineering,
        matrix_cell=matrix_cell,
        cleaning_config_path=cleaning_config_path,
    )
    return {
        "cleaning": {
            "config_path": str(cleaning_path),
        },
        "feature_engineering": {
            "enabled": use_feature_engineering,
            "config": feature_cfg,
        }
    }


def load_and_prepare_data(
    data_path: Path,
    sample_size: int = None,
    use_feature_engineering: bool = True,
    feature_engineering_config: Optional[Dict] = None,
    return_metadata: bool = False,
    matrix_cell: Optional[str] = None,
    cleaning_config_path: Optional[Path] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Load, clean, and optionally engineer features via DataPipeline.
    
    Args:
        data_path: Path to raw data file.
        sample_size: Optional sample size for quick testing.
        use_feature_engineering: If False, skip feature engineering (for Raw-only track).
        feature_engineering_config: Optional override for feature engineering config.
        return_metadata: If True, also return pipeline run metadata.
    
    Returns:
        DataFrame with processed features, optionally accompanied by run metadata.
    """
    pipeline_config = _build_pipeline_config(
        use_feature_engineering=use_feature_engineering,
        feature_engineering_config=feature_engineering_config,
        matrix_cell=matrix_cell,
        cleaning_config_path=cleaning_config_path,
    )
    
    _logger.info("=" * 60)
    _logger.info("Step 1: Data Pipeline (load → clean → features)")
    _logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting pipeline run...")
    _logger.info("=" * 60)
    pipeline = DataPipeline(config=pipeline_config)
    bundle = pipeline.run(
        data_path=data_path,
        horizons=[],
        use_feature_engineering=use_feature_engineering,
        feature_config=pipeline_config["feature_engineering"],
        generate_labels=False,
    )
    df = bundle.data
    _logger.info(f"Pipeline output: {len(df)} rows, {len(df.columns)} columns")
    
    # Optional sampling for quick experiments
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        _logger.info(f"Sampled {len(df)} rows for quick testing")
    
    # Optimize dtypes to reduce memory footprint
    df = _optimize_numeric_dtypes(df)
    memory_gb = df.memory_usage(deep=True).sum() / 1024 ** 3
    _logger.info(f"Memory usage after optimization: {memory_gb:.2f} GB")
    
    if return_metadata:
        return df, bundle.run_metadata
    return df


def create_frost_labels(
    df: pd.DataFrame,
    horizons: list,
    frost_threshold: float = 0.0
) -> pd.DataFrame:
    """Create frost labels for all horizons.
    
    Args:
        df: DataFrame with temperature data.
        horizons: List of forecast horizons in hours.
        frost_threshold: Temperature threshold for frost.
    
    Returns:
        DataFrame with frost labels added.
    """
    _logger.info("=" * 60)
    _logger.info("Step 4: Creating Frost Labels")
    _logger.info("=" * 60)
    
    label_generator = FrostLabelGenerator(frost_threshold=frost_threshold)
    df_labeled = label_generator.create_frost_labels(df, horizons=horizons)
    
    _logger.info(f"Created frost labels for horizons: {horizons}")
    _logger.info(f"Final dataset: {len(df_labeled)} rows, {len(df_labeled.columns)} columns")
    
    return df_labeled


def prepare_features_and_targets(
    df: pd.DataFrame,
    horizon: int,
    feature_selection: Optional[Dict] = None,
    indices: Optional[pd.Index] = None,
    track: str = "top175_features",
    model_type: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare features and targets for a specific horizon.
    
    Args:
        df: DataFrame with features and labels.
        horizon: Forecast horizon in hours.
        feature_selection: Optional feature selection config.
        indices: Optional indices to filter data.
        track: Track type ("raw" or "top175_features").
        model_type: Model type (e.g., "prophet" needs Date column).
    
    Returns:
        Tuple of (X, y_frost, y_temp) DataFrames/Series.
    
    Raises:
        ValueError: If DataFrame is empty or missing required columns.
    """
    # Input validation: Check if DataFrame is empty
    if df.empty:
        raise ValueError(f"Input DataFrame is empty. Cannot prepare features for {horizon}h horizon.")
    
    # Filter by indices if provided
    if indices is not None:
        df_subset = df.loc[indices]
        if df_subset.empty:
            raise ValueError(f"Filtered DataFrame (after applying indices) is empty for {horizon}h horizon.")
        df = df_subset
    
    # Strict validation: Check for required target columns
    frost_col = f"frost_{horizon}h"
    temp_col = f"temp_{horizon}h"
    missing_targets = []
    if frost_col not in df.columns:
        missing_targets.append(frost_col)
    if temp_col not in df.columns:
        missing_targets.append(temp_col)
    if missing_targets:
        raise KeyError(
            f"Missing required target columns for {horizon}h horizon: {missing_targets}. "
            f"Available columns: {list(df.columns)[:20]}..."
        )
    
    # Get features (exclude target columns and metadata)
    # Prophet model needs Date column, so don't exclude it for Prophet
    # Graph models (dcrnn, st_gcn, gat_lstm, graphwavenet) need Stn Id for node mapping
    graph_models = ["dcrnn", "st_gcn", "gat_lstm", "graphwavenet"]
    exclude_cols = [
        "Station Name", "County",
        f"frost_{horizon}h", f"temp_{horizon}h"
    ]
    # Keep Stn Id for graph models
    if model_type not in graph_models:
        exclude_cols.append("Stn Id")
    if model_type != "prophet":
        exclude_cols.append("Date")
    # Exclude ALL horizon label columns to prevent data leakage
    # Only keep the current horizon labels as targets
    feature_cols = [
        col for col in df.columns 
        if col not in exclude_cols 
        and not col.startswith('frost_')  # Exclude all frost_*h columns
        and not col.startswith('temp_')   # Exclude all temp_*h columns
    ]
    
    # Track-aware feature selection
    if track == "raw":
        # DEBUG: Log input df columns
        neighbor_cols_in_df = [c for c in df.columns if 'neighbor' in c.lower()]
        _logger.info(
            f"🔍 DEBUG [prepare_features_and_targets] track='raw': "
            f"Input df has {len(df.columns)} columns, {len(neighbor_cols_in_df)} neighbor columns"
        )
        if len(neighbor_cols_in_df) > 0:
            _logger.info(f"🔍 DEBUG: Neighbor feature examples in input df: {neighbor_cols_in_df[:5]}")
        else:
            _logger.warning(
                f"⚠️  DEBUG WARNING [prepare_features_and_targets]: "
                f"No neighbor features found in input DataFrame! "
                f"This may indicate DataPipeline did not generate neighbor features."
            )
        
        # Select numeric, non-engineered raw columns
        # Heuristics: drop common engineered patterns (lag/rolling/interaction/cyclical/station meta)
        # BUT keep neighbor_* features for Matrix Cell C (raw + spatial aggregation)
        engineered_patterns = (
            "_lag_", "rolling", "interaction", "daily_", "_decline_rate",
            "station_id_encoded", "is_eto_station",
            "latitude", "longitude", "latitude_", "longitude_", "distance_to_",
            "station_density", "county_encoded", "city_encoded", "groundcover_encoded",
            "hour_sin", "hour_cos", "month_sin", "month_cos", "season", "is_night"
        )
        def is_raw_feature(col: str) -> bool:
            # Keep neighbor features for Matrix Cell C (raw + spatial aggregation)
            if "neighbor" in col.lower():
                return True
            # Filter out engineered patterns (but allow _gradient and _range if they're from neighbors)
            # The engineered_patterns for non-neighbor features will exclude _gradient and _range
            return not any(pat in col for pat in engineered_patterns)
        raw_feature_cols = [c for c in feature_cols if is_raw_feature(c)]
        
        # DEBUG: Log raw_feature_cols
        neighbor_cols_in_raw = [c for c in raw_feature_cols if 'neighbor' in c.lower()]
        _logger.info(
            f"🔍 DEBUG [prepare_features_and_targets] track='raw': "
            f"After is_raw_feature filtering: {len(raw_feature_cols)} columns, {len(neighbor_cols_in_raw)} neighbor columns"
        )
        
        # For Prophet, include Date column even if it's not numeric
        if model_type == "prophet" and "Date" in df.columns:
            # Include Date column (avoid copy if not needed)
            X_numeric = df[raw_feature_cols].select_dtypes(include=[np.number])
            X_date = df[["Date"]]
            X = pd.concat([X_date, X_numeric], axis=1)
        else:
            X = df[raw_feature_cols].select_dtypes(include=[np.number])
        
        # DEBUG: Log final X columns
        neighbor_cols_in_X = [c for c in X.columns if 'neighbor' in c.lower()]
        _logger.info(
            f"🔍 DEBUG [prepare_features_and_targets] track='raw': "
            f"Final X (after select_dtypes) has {len(X.columns)} columns, {len(neighbor_cols_in_X)} neighbor columns"
        )
        if len(neighbor_cols_in_X) > 0:
            _logger.info(f"🔍 DEBUG: Neighbor features in final X: {neighbor_cols_in_X[:5]}... (showing first 5)")
        if len(neighbor_cols_in_X) == 0:
            _logger.warning(
                f"⚠️  DEBUG WARNING [prepare_features_and_targets]: No neighbor features in final X! "
                f"This may indicate a bug. df had {len(neighbor_cols_in_df)} neighbor columns, "
                f"raw_feature_cols had {len(neighbor_cols_in_raw)} neighbor columns"
            )
    else:
        # Default: use engineered features (top175_features track)
        # For Prophet, include Date column even if it's not numeric
        if model_type == "prophet" and "Date" in df.columns:
            X_numeric = df[feature_cols].select_dtypes(include=[np.number])
            X_date = df[["Date"]]
            X = pd.concat([X_date, X_numeric], axis=1)
        else:
            X = df[feature_cols].select_dtypes(include=[np.number])
    
    # Strict validation: Check that we have feature columns
    if len(X.columns) == 0:
        raise ValueError(
            f"No feature columns found after preparation for {horizon}h horizon. "
            f"This may indicate an issue with feature engineering or data filtering. "
            f"Available columns in df: {list(df.columns)[:20]}..."
        )
    
    # Warn if some columns were excluded
    excluded_non_numeric = [col for col in feature_cols if col not in X.columns]
    if excluded_non_numeric:
        _logger.warning(f"Excluded {len(excluded_non_numeric)} non-numeric columns: {excluded_non_numeric[:5]}...")
    
    # Apply feature selection if provided
    if feature_selection and track != "raw":
        if "top_n" in feature_selection:
            # Use top N features (should be pre-computed)
            top_features = feature_selection.get("features", [])
            if top_features:
                X = X[top_features]
        elif "importance_threshold" in feature_selection:
            # Use features above importance threshold
            top_features = feature_selection.get("features", [])
            if top_features:
                X = X[top_features]
    
    # Get targets
    y_frost = df[f"frost_{horizon}h"]
    y_temp = df[f"temp_{horizon}h"]
    
    # Remove rows with missing targets (targets must be valid - cannot predict without target)
    valid_mask = ~(y_frost.isna() | y_temp.isna())
    X = X[valid_mask]
    y_frost = y_frost[valid_mask]
    y_temp = y_temp[valid_mask]
    
    # Handle remaining NaN in features (preserve time continuity)
    # Note: DataCleaner already used forward_fill in preprocessing, but some NaN may remain
    # (e.g., at sequence start). We use forward_fill again here to preserve time continuity
    # instead of deleting rows, which would break the hourly time series structure.
    # Tree models (LightGBM, XGBoost, CatBoost) can handle NaN natively, but we fill here
    # for consistency across all models and to preserve time continuity.
    # For Prophet, preserve Date column separately (it shouldn't have NaN, but if it does, don't fill it)
    date_col = None
    if model_type == "prophet" and "Date" in X.columns:
        date_col = X["Date"]
        X_numeric = X.drop(columns=["Date"])
    else:
        X_numeric = X
    
    if X_numeric.isna().any().any():
        n_nan_before = X_numeric.isna().sum().sum()
        # Forward fill within each station to preserve time continuity
        if "Stn Id" in df.columns:
            # Get station IDs for the valid rows
            station_ids = df.loc[valid_mask, "Stn Id"]
            # Group by station and forward fill
            X_numeric = X_numeric.groupby(station_ids).ffill()
        else:
            # If no station ID, just forward fill globally
            X_numeric = X_numeric.ffill()
        
        # If still NaN (e.g., at sequence start), use backward fill
        if X_numeric.isna().any().any():
            if "Stn Id" in df.columns:
                X_numeric = X_numeric.groupby(station_ids).bfill()
            else:
                X_numeric = X_numeric.bfill()
        
        n_nan_after = X_numeric.isna().sum().sum()
        if n_nan_before > 0:
            _logger.warning(f"Found {n_nan_before} NaN values in features, filled using forward/backward fill")
            if n_nan_after > 0:
                _logger.debug(f"{n_nan_after} NaN values remain (likely at sequence boundaries)")
    
    # Recombine Date column if it was separated
    if date_col is not None:
        X = pd.concat([date_col, X_numeric], axis=1)
    else:
        X = X_numeric
    
    _logger.info(f"Features: {len(X.columns)}, Samples: {len(X)}")
    _logger.info(f"Frost labels: {y_frost.sum()} positive ({y_frost.mean()*100:.2f}%)")
    _logger.info(f"Temperature range: {y_temp.min():.2f}°C to {y_temp.max():.2f}°C")
    
    return X, y_frost, y_temp

