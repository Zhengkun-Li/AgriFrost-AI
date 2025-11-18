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
from typing import Dict, Optional, Tuple
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.data.loaders import DataLoader
from src.data.cleaners import DataCleaner
from src.data.feature_engineering import FeatureEngineer
from src.data.frost_labels import FrostLabelGenerator
from src.data.preprocessors import preprocess_with_loso
from src.evaluation.validators import CrossValidator


def load_and_prepare_data(
    data_path: Path,
    sample_size: int = None,
    use_feature_engineering: bool = True
) -> pd.DataFrame:
    """Load, clean, and optionally engineer features.
    
    Args:
        data_path: Path to raw data file.
        sample_size: Optional sample size for quick testing.
        use_feature_engineering: If False, skip feature engineering (for Raw-only track).
    
    Returns:
        DataFrame with cleaned and optionally engineered features.
    """
    step_start_time = time.time()
    step_start_datetime = datetime.now()
    print("\n" + "="*60)
    print("Step 1: Loading Data")
    print(f"[{step_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}] Starting data loading...")
    print("="*60)
    
    # Load raw data
    df = DataLoader.load_raw_data(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Optimize data types to reduce memory usage
    print("Optimizing data types to reduce memory...")
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    print(f"Memory usage after optimization: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {len(df)} rows")
    
    # Clean data
    step_elapsed = time.time() - step_start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 1 completed in {step_elapsed:.2f} seconds")
    
    step_start_time = time.time()
    step_start_datetime = datetime.now()
    print("\n" + "="*60)
    print("Step 2: Cleaning Data")
    print(f"[{step_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}] Starting data cleaning...")
    print("="*60)
    
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean_pipeline(df)
    print(f"After cleaning: {len(df_cleaned)} rows")
    
    # Delete original df to free memory
    del df
    import gc
    gc.collect()
    
    # Feature engineering
    step_elapsed = time.time() - step_start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 2 completed in {step_elapsed:.2f} seconds")
    
    if use_feature_engineering:
        step_start_time = time.time()
        step_start_datetime = datetime.now()
        print("\n" + "="*60)
        print("Step 3: Feature Engineering")
        print(f"[{step_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}] Starting feature engineering...")
        print("="*60)
        
        engineer = FeatureEngineer()
        feature_config = {
            "create_time_features": True,
            "create_lag_features": True,
            "create_rolling_features": True,
            "create_interaction_features": False,  # Disable to reduce memory
            "lag_periods": [1, 3, 6, 12, 24],
            "rolling_windows": [3, 6, 12, 24],
        }
        df_engineered = engineer.engineer_features(df_cleaned, feature_config)
        print(f"After feature engineering: {len(df_engineered)} rows, {len(df_engineered.columns)} columns")
        
        # Delete cleaned df to free memory
        del df_cleaned
        gc.collect()
        
        # Optimize data types again after feature engineering
        print("Optimizing data types after feature engineering...")
        for col in df_engineered.select_dtypes(include=['float64']).columns:
            df_engineered[col] = pd.to_numeric(df_engineered[col], downcast='float')
        for col in df_engineered.select_dtypes(include=['int64']).columns:
            df_engineered[col] = pd.to_numeric(df_engineered[col], downcast='integer')
        print(f"Memory usage after optimization: {df_engineered.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
        
        step_elapsed = time.time() - step_start_time
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 3 completed in {step_elapsed:.2f} seconds")
        
        return df_engineered
    else:
        # Skip feature engineering for Raw-only track
        print("\n" + "="*60)
        print("Step 3: Skipping Feature Engineering (Raw-only track)")
        print("="*60)
        print(f"Using raw features only: {len(df_cleaned)} rows, {len(df_cleaned.columns)} columns")
        
        # Optimize data types
        print("Optimizing data types...")
        for col in df_cleaned.select_dtypes(include=['float64']).columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], downcast='float')
        for col in df_cleaned.select_dtypes(include=['int64']).columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], downcast='integer')
        print(f"Memory usage after optimization: {df_cleaned.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
        
        step_elapsed = time.time() - step_start_time
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 3 (skipped) completed in {step_elapsed:.2f} seconds")
        
        return df_cleaned


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
    print("\n" + "="*60)
    print("Step 4: Creating Frost Labels")
    print("="*60)
    
    label_generator = FrostLabelGenerator(frost_threshold=frost_threshold)
    df_labeled = label_generator.create_frost_labels(df, horizons=horizons)
    
    print(f"Created frost labels for horizons: {horizons}")
    print(f"Final dataset: {len(df_labeled)} rows, {len(df_labeled.columns)} columns")
    
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
    """
    # Filter by indices if provided
    if indices is not None:
        df = df.loc[indices]
    
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
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Track-aware feature selection
    if track == "raw":
        # Select numeric, non-engineered raw columns
        # Heuristics: drop common engineered patterns (lag/rolling/interaction/cyclical/station meta)
        engineered_patterns = (
            "_lag_", "rolling", "interaction", "daily_", "_decline_rate",
            "_gradient", "_range", "station_id_encoded", "is_eto_station",
            "latitude", "longitude", "latitude_", "longitude_", "distance_to_",
            "station_density", "county_encoded", "city_encoded", "groundcover_encoded",
            "hour_sin", "hour_cos", "month_sin", "month_cos", "season", "is_night"
        )
        def is_raw_feature(col: str) -> bool:
            return not any(pat in col for pat in engineered_patterns)
        raw_feature_cols = [c for c in feature_cols if is_raw_feature(c)]
        # For Prophet, include Date column even if it's not numeric
        if model_type == "prophet" and "Date" in df.columns:
            # Include Date column
            X_numeric = df[raw_feature_cols].select_dtypes(include=[np.number]).copy()
            X_date = df[["Date"]].copy()
            X = pd.concat([X_date, X_numeric], axis=1)
        else:
            X = df[raw_feature_cols].select_dtypes(include=[np.number]).copy()
    else:
        # Default: use engineered features (top175_features track)
        # For Prophet, include Date column even if it's not numeric
        if model_type == "prophet" and "Date" in df.columns:
            X_numeric = df[feature_cols].select_dtypes(include=[np.number]).copy()
            X_date = df[["Date"]].copy()
            X = pd.concat([X_date, X_numeric], axis=1)
        else:
            X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    
    # Warn if some columns were excluded
    excluded_non_numeric = [col for col in feature_cols if col not in X.columns]
    if excluded_non_numeric:
        print(f"  ⚠️  Warning: Excluded {len(excluded_non_numeric)} non-numeric columns: {excluded_non_numeric[:5]}...")
    
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
        date_col = X["Date"].copy()
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
            print(f"  ⚠️  Warning: Found {n_nan_before} NaN values in features, filled using forward/backward fill")
            if n_nan_after > 0:
                print(f"     {n_nan_after} NaN values remain (likely at sequence boundaries)")
    
    # Recombine Date column if it was separated
    if date_col is not None:
        X = pd.concat([date_col, X_numeric], axis=1)
    else:
        X = X_numeric
    
    print(f"Features: {len(X.columns)}, Samples: {len(X)}")
    print(f"Frost labels: {y_frost.sum()} positive ({y_frost.mean()*100:.2f}%)")
    print(f"Temperature range: {y_temp.min():.2f}°C to {y_temp.max():.2f}°C")
    
    return X, y_frost, y_temp

