"""Model training module for frost forecasting.

This module handles:
- Model training for a specific horizon
- Model evaluation
- Model saving
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import time
import json
import os
import gc

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.evaluation.metrics import MetricsCalculator
from src.evaluation.validators import CrossValidator
from src.visualization.plots import Plotter
from src.utils.path_utils import ensure_dir
from src.training.data_preparation import prepare_features_and_targets
from src.training.model_config import get_model_config, get_model_class


def check_models_exist(
    horizon_dir: Path,
    model_type: str
) -> bool:
    """Check if models already exist for a horizon.
    
    Args:
        horizon_dir: Directory for the horizon.
        model_type: Model type.
    
    Returns:
        True if models exist, False otherwise.
    """
    if model_type == "lstm_multitask":
        # Multi-task model saves to multiple locations
        multitask_model_path = horizon_dir / "multitask_model" / "model.pth"
        frost_model_path = horizon_dir / "frost_classifier" / "model.pth"
        return multitask_model_path.exists() or frost_model_path.exists()
    else:
        # Regular models have separate frost and temp models
        frost_model_dir = horizon_dir / "frost_classifier"
        temp_model_dir = horizon_dir / "temp_regressor"
        
        frost_model_exists = False
        temp_model_exists = False
        
        if frost_model_dir.exists():
            frost_model_files = list(frost_model_dir.glob("model.*"))
            frost_model_exists = len(frost_model_files) > 0
        
        if temp_model_dir.exists():
            temp_model_files = list(temp_model_dir.glob("model.*"))
            temp_model_exists = len(temp_model_files) > 0
        
        return frost_model_exists and temp_model_exists


def load_existing_results(horizon_dir: Path) -> Optional[Dict]:
    """Load existing results if models exist.
    
    Args:
        horizon_dir: Directory for the horizon.
    
    Returns:
        Dictionary with metrics if found, None otherwise.
    """
    frost_metrics_path = horizon_dir / "frost_metrics.json"
    temp_metrics_path = horizon_dir / "temp_metrics.json"
    
    if frost_metrics_path.exists() and temp_metrics_path.exists():
        with open(frost_metrics_path, "r") as f:
            frost_metrics = json.load(f)
        with open(temp_metrics_path, "r") as f:
            temp_metrics = json.load(f)
        
        print(f"  Frost - Brier: {frost_metrics.get('brier_score', 'N/A'):.4f}, "
              f"ECE: {frost_metrics.get('ece', 'N/A'):.4f}, "
              f"ROC-AUC: {frost_metrics.get('roc_auc', 'N/A'):.4f}")
        print(f"  Temp  - MAE: {temp_metrics.get('mae', 'N/A'):.4f}, "
              f"RMSE: {temp_metrics.get('rmse', 'N/A'):.4f}, "
              f"R²: {temp_metrics.get('r2', 'N/A'):.4f}")
        
        return {
            "frost_metrics": frost_metrics,
            "temp_metrics": temp_metrics
        }
    return None


def train_frost_model(
    model_type: str,
    model_class,
    frost_config: Dict,
    X_train: pd.DataFrame,
    y_frost_train: pd.Series,
    X_val: pd.DataFrame,
    y_frost_val: pd.Series,
    station_ids_train: Optional[np.ndarray] = None,
    **fit_kwargs
):
    """Train frost classification model.
    
    Args:
        model_type: Model type.
        model_class: Model class.
        frost_config: Model configuration.
        X_train: Training features.
        y_frost_train: Training frost labels.
        X_val: Validation features.
        y_frost_val: Validation frost labels.
        station_ids_train: Optional station IDs for LSTM models.
    
    Returns:
        Trained model.
    """
    model_frost = model_class(frost_config)
    
    if model_type in ["lightgbm", "xgboost", "catboost"]:
        model_frost.fit(X_train, y_frost_train, eval_set=[(X_val, y_frost_val)])
    elif model_type == "prophet":
        if "Date" not in X_train.columns:
            print("  ⚠️  Warning: Date column not found. Prophet may not work correctly.")
            print("     Consider ensuring Date column is available in feature engineering.")
        model_frost.fit(X_train, y_frost_train)
    elif model_type == "lstm":
        # Get checkpoint directory from config or kwargs
        checkpoint_dir = fit_kwargs.get('checkpoint_dir', frost_config.get("checkpoint_dir", None))
        fit_kwargs_lstm = {'station_ids': station_ids_train}
        # CRITICAL FIX: Pass external validation set to avoid double-split
        # Get validation station IDs if available
        station_ids_val = fit_kwargs.get('station_ids_val', None)
        if station_ids_val is not None:
            fit_kwargs_lstm['station_ids_val'] = station_ids_val
        # Pass eval_set (external validation from time split)
        eval_set = [(X_val, y_frost_val)] if 'X_val' in locals() and 'y_frost_val' in locals() else None
        if checkpoint_dir:
            fit_kwargs_lstm['checkpoint_dir'] = checkpoint_dir
        if 'resume_from_checkpoint' in fit_kwargs:
            fit_kwargs_lstm['resume_from_checkpoint'] = fit_kwargs['resume_from_checkpoint']
        # Pass kwargs directly to model.fit(), not to train_frost_model()
        model_frost.fit(X_train, y_frost_train, eval_set=eval_set, **fit_kwargs_lstm)
    elif model_type == "lstm_multitask":
        # Multi-task LSTM needs both y_temp and y_frost
        # This will be handled in train_models_for_horizon
        raise ValueError("lstm_multitask should use train_multitask_model instead")
    else:
        # Random Forest and Ensemble don't use eval_set
        model_frost.fit(X_train, y_frost_train)
    
    return model_frost


def train_temp_model(
    model_type: str,
    model_class,
    temp_config: Dict,
    X_train: pd.DataFrame,
    y_temp_train: pd.Series,
    X_val: pd.DataFrame,
    y_temp_val: pd.Series,
    station_ids_train: Optional[np.ndarray] = None,
    **fit_kwargs
):
    """Train temperature regression model.
    
    Args:
        model_type: Model type.
        model_class: Model class.
        temp_config: Model configuration.
        X_train: Training features.
        y_temp_train: Training temperature values.
        X_val: Validation features.
        y_temp_val: Validation temperature values.
        station_ids_train: Optional station IDs for LSTM models.
    
    Returns:
        Trained model.
    """
    # Initialize temp model (reuse model_class for same model type)
    if model_type == "lstm":
        from src.models.deep.lstm_model import LSTMForecastModel
        temp_model_class = LSTMForecastModel
    elif model_type == "prophet":
        from src.models.traditional.prophet_model import ProphetModel
        temp_model_class = ProphetModel
    else:
        temp_model_class = model_class
    
    model_temp = temp_model_class(temp_config)
    
    if model_type in ["lightgbm", "xgboost", "catboost"]:
        model_temp.fit(X_train, y_temp_train, eval_set=[(X_val, y_temp_val)])
    elif model_type == "prophet":
        if "Date" not in X_train.columns:
            print("  ⚠️  Warning: Date column not found. Prophet may not work correctly.")
        model_temp.fit(X_train, y_temp_train)
    elif model_type == "lstm":
        # Get checkpoint directory from config or kwargs
        checkpoint_dir = fit_kwargs.get('checkpoint_dir', temp_config.get("checkpoint_dir", None))
        fit_kwargs_lstm = {'station_ids': station_ids_train}
        if checkpoint_dir:
            fit_kwargs_lstm['checkpoint_dir'] = checkpoint_dir
        if 'resume_from_checkpoint' in fit_kwargs:
            fit_kwargs_lstm['resume_from_checkpoint'] = fit_kwargs['resume_from_checkpoint']
        # Pass kwargs directly to model.fit(), not to train_temp_model()
        model_temp.fit(X_train, y_temp_train, **fit_kwargs_lstm)
    else:
        # Random Forest and Ensemble don't use eval_set
        model_temp.fit(X_train, y_temp_train)
    
    return model_temp


def train_multitask_model(
    model_type: str,
    model_class,
    frost_config: Dict,
    X_train: pd.DataFrame,
    y_temp_train: pd.Series,
    y_frost_train: pd.Series,
    station_ids_train: Optional[np.ndarray] = None,
    **fit_kwargs
):
    """Train multi-task model (for lstm_multitask).
    
    Args:
        model_type: Model type (should be "lstm_multitask").
        model_class: Model class.
        frost_config: Model configuration.
        X_train: Training features.
        y_temp_train: Training temperature values.
        y_frost_train: Training frost labels.
        station_ids_train: Optional station IDs.
    
    Returns:
        Trained model.
    """
    if model_type != "lstm_multitask":
        raise ValueError(f"train_multitask_model only supports lstm_multitask, got {model_type}")
    
    model_frost = model_class(frost_config)
    # Get checkpoint directory from config or kwargs
    checkpoint_dir = fit_kwargs.get('checkpoint_dir', frost_config.get("checkpoint_dir", None))
    fit_kwargs_mt = {'station_ids': station_ids_train}
    if checkpoint_dir:
        fit_kwargs_mt['checkpoint_dir'] = checkpoint_dir
    if 'resume_from_checkpoint' in fit_kwargs:
        fit_kwargs_mt['resume_from_checkpoint'] = fit_kwargs['resume_from_checkpoint']
    model_frost.fit(X_train, y_temp_train, y_frost_train, **fit_kwargs_mt)
    return model_frost


def evaluate_models(
    model_type: str,
    model_frost,
    model_temp,
    X_test: pd.DataFrame,
    y_frost_test: pd.Series,
    y_temp_test: pd.Series,
    station_ids_test: Optional[np.ndarray] = None
) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate models on test set.
    
    Args:
        model_type: Model type.
        model_frost: Frost classification model.
        model_temp: Temperature regression model.
        X_test: Test features.
        y_frost_test: Test frost labels.
        y_temp_test: Test temperature values.
    
    Returns:
        Tuple of (frost_metrics, temp_metrics, y_frost_pred, y_frost_proba, y_temp_pred).
    """
    # For multi-task models, use specific prediction methods
    if model_type == "lstm_multitask":
        y_temp_pred = model_frost.predict_temp(X_test, station_ids=station_ids_test)
        y_frost_proba = model_frost.predict_frost_proba(X_test, station_ids=station_ids_test)
        y_frost_pred = (y_frost_proba >= 0.5).astype(int)
    else:
        # For LSTM single-task, pass station ids for boundary-safe windowing when available
        if model_type == "lstm":
            y_frost_pred = model_frost.predict(X_test, station_ids=station_ids_test)
        else:
            y_frost_pred = model_frost.predict(X_test)
        y_frost_proba = model_frost.predict_proba(X_test)
        
        # Handle models that don't support predict_proba (LSTM, Prophet)
        if y_frost_proba is None:
            print("  ⚠️  Model doesn't support predict_proba. Using temperature regression for frost probability.")
            if model_type == "lstm":
                y_temp_pred = model_temp.predict(X_test, station_ids=station_ids_test)
            elif model_type == "lstm_multitask":
                y_temp_pred = model_frost.predict_temp(X_test, station_ids=station_ids_test)
            else:
                y_temp_pred = model_temp.predict(X_test)
            frost_threshold = 0.0
            scale = 2.0
            y_frost_proba = 1.0 / (1.0 + np.exp((y_temp_pred - frost_threshold) / scale))
            y_frost_pred = (y_temp_pred < frost_threshold).astype(int)
        else:
            if model_type == "lstm":
                y_temp_pred = model_temp.predict(X_test, station_ids=station_ids_test)
            else:
                y_temp_pred = model_temp.predict(X_test)
    
    # Calculate metrics
    frost_metrics = MetricsCalculator.calculate_classification_metrics(
        y_frost_test.values, y_frost_pred, y_frost_proba
    )
    frost_prob_metrics = MetricsCalculator.calculate_probability_metrics(
        y_frost_test.values, y_frost_proba
    )
    frost_metrics.update(frost_prob_metrics)
    
    temp_metrics = MetricsCalculator.calculate_regression_metrics(
        y_temp_test.values, y_temp_pred
    )
    
    return frost_metrics, temp_metrics, y_frost_pred, y_frost_proba, y_temp_pred


def save_models_and_results(
    model_type: str,
    model_frost,
    model_temp,
    horizon_dir: Path,
    frost_metrics: Dict,
    temp_metrics: Dict,
    y_frost_test: pd.Series,
    y_frost_pred: np.ndarray,
    y_frost_proba: np.ndarray,
    y_temp_test: pd.Series,
    y_temp_pred: np.ndarray,
    horizon: int
):
    """Save models, metrics, and results.
    
    Args:
        model_type: Model type.
        model_frost: Frost classification model.
        model_temp: Temperature regression model.
        horizon_dir: Directory to save results.
        frost_metrics: Frost classification metrics.
        temp_metrics: Temperature regression metrics.
        y_frost_test: Test frost labels.
        y_frost_pred: Predicted frost labels.
        y_frost_proba: Predicted frost probabilities.
        y_temp_test: Test temperature values.
        y_temp_pred: Predicted temperature values.
        horizon: Forecast horizon.
    """
    ensure_dir(horizon_dir)
    
    # Save models
    if model_type == "lstm_multitask":
        model_frost.save(horizon_dir / "multitask_model")
        model_frost.save(horizon_dir / "frost_classifier")
        model_frost.save(horizon_dir / "temp_regressor")
    else:
        model_frost.save(horizon_dir / "frost_classifier")
        model_temp.save(horizon_dir / "temp_regressor")
    
    # Save metrics
    with open(horizon_dir / "frost_metrics.json", "w") as f:
        json.dump(frost_metrics, f, indent=2, default=str)
    with open(horizon_dir / "temp_metrics.json", "w") as f:
        json.dump(temp_metrics, f, indent=2, default=str)
    
    # Generate reliability diagram
    print(f"\nGenerating reliability diagram...")
    plotter = Plotter(style="matplotlib", figsize=(10, 8))
    plotter.plot_reliability_diagram(
        y_frost_test.values,
        y_frost_proba,
        n_bins=10,
        title=f"Reliability Diagram - {horizon}h Horizon",
        save_path=horizon_dir / "reliability_diagram.png",
        show=False
    )
    
    # Save predictions
    predictions = {
        "frost": {
            "y_true": y_frost_test.values.tolist(),
            "y_pred": y_frost_pred.tolist(),
            "y_proba": y_frost_proba.tolist()
        },
        "temperature": {
            "y_true": y_temp_test.values.tolist(),
            "y_pred": y_temp_pred.tolist()
        }
    }
    with open(horizon_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=2, default=str)


def train_models_for_horizon(
    df: pd.DataFrame,
    horizon: int,
    output_dir: Path,
    model_type: str = "lightgbm",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    skip_if_exists: bool = True,
    feature_selection: Optional[Dict] = None,
    track: Optional[str] = None,
    matrix_cell: Optional[str] = None
) -> Dict:
    """Train classification and regression models for a specific horizon.
    
    Args:
        df: DataFrame with features and labels.
        horizon: Forecast horizon in hours.
        output_dir: Output directory for models and results.
        model_type: Model type.
        train_ratio: Training data ratio.
        val_ratio: Validation data ratio.
        skip_if_exists: If True, skip training if models already exist.
        feature_selection: Optional feature selection config.
    
    Returns:
        Dictionary with model results and metrics.
    """
    horizon_start_time = time.time()
    horizon_start_datetime = datetime.now()
    print(f"\n{'='*60}")
    print(f"Training models for {horizon}h horizon", flush=True)
    print(f"[{horizon_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}] Starting {horizon}h horizon training...", flush=True)
    import sys
    sys.stdout.flush()
    print("="*60)
    
    # Infer track/matrix_cell from output_dir if not provided
    parts = [p for p in output_dir.parts]
    if track is None:
        track = "raw" if "raw" in parts else ("top175_features" if "top175_features" in parts else "top175_features")
    if matrix_cell is None:
        # Look for A-E in path parts
        candidates = {"A","B","C","D","E"}
        found = [p for p in parts if p in candidates]
        matrix_cell = found[0] if found else ("B" if track == "top175_features" else "A")
    
    # Resolve base dir: add matrix_cell only if not already present in output_dir
    out_parts = set(output_dir.parts)
    base_dir = output_dir if matrix_cell in out_parts else (output_dir / matrix_cell)
    
    # Check if models already exist
    horizon_dir = base_dir / "full_training" / f"horizon_{horizon}h"
    
    if skip_if_exists and check_models_exist(horizon_dir, model_type):
        print(f"✅ Models for {horizon}h already exist, loading results...")
        results = load_existing_results(horizon_dir)
        if results:
            return results
        else:
            print(f"⚠️  Models exist but metrics not found, retraining...")
    
    # Ensure horizon directory exists (for logs and outputs)
    ensure_dir(horizon_dir)
    # Prepare per-horizon brief log file (detailed log will be derived automatically)
    horizon_log_file = str(horizon_dir / "training.log")
    
    # Prepare data
    X, y_frost, y_temp = prepare_features_and_targets(
        df, horizon, feature_selection=feature_selection, track=track, model_type=model_type
    )
    print(f"Features: {len(X.columns)}")
    print(f"Samples: {len(X)}")
    print(f"Frost events: {y_frost.sum()} ({y_frost.mean()*100:.2f}%)")
    
    # Time-based split
    df_split = df.loc[X.index].copy()
    train_df, val_df, test_df = CrossValidator.time_split(
        df_split, train_ratio=train_ratio, val_ratio=val_ratio
    )
    
    # Note: time_split resets index, so we need to map back to original indices
    # train_df, val_df, test_df now have sequential indices (0, 1, 2, ...)
    # but we need to use the original indices from df_split to index into X, y_frost, y_temp
    
    # Get original indices before time_split resets them
    train_orig_idx = df_split.index[train_df.index]
    val_orig_idx = df_split.index[val_df.index]
    test_orig_idx = df_split.index[test_df.index]
    
    # CRITICAL FIX for LSTM: Group by station first, then sort by time within each station
    # This ensures temporal continuity within each station for sequence building
    # Without this, interleaved stations cause huge gaps in station indices, preventing sequence generation
    if model_type in ["lstm", "lstm_multitask"]:
        # Strategy: Group by station, then sort by time within each station
        # This maximizes sequence generation by ensuring station-level temporal continuity
        
        def reorganize_by_station(df_split_subset, orig_idx_subset, X_subset, y_frost_subset, y_temp_subset):
            """Reorganize data: group by station, then sort by time within each station."""
            if len(df_split_subset) == 0:
                return None, None, None, None, None
            
            # Create a DataFrame with original indices and station IDs
            reorg_data = {'orig_idx': orig_idx_subset}
            
            if "Stn Id" in df_split_subset.columns:
                reorg_data['stn_id'] = df_split_subset["Stn Id"].values
            else:
                return None, None, None, None, None
            
            # Get Date from df_split (original dataframe) using orig_idx
            if "Date" in df_split.columns:
                date_map = dict(zip(df_split.index, df_split["Date"].values))
                reorg_data['date'] = [date_map.get(idx) for idx in orig_idx_subset]
            
            reorg_df = pd.DataFrame(reorg_data)
            
            if "Date" in df_split.columns:
                # Group by station, then sort by Date within each station
                # This ensures temporal continuity within each station
                reorg_df = reorg_df.sort_values(['stn_id', 'date'], kind='stable')
            else:
                # If no Date, just group by station
                reorg_df = reorg_df.sort_values('stn_id', kind='stable')
            
            # Get reordered indices
            reordered_idx = reorg_df['orig_idx'].values
            
            # Filter to only indices that exist in X
            reordered_idx = [idx for idx in reordered_idx if idx in X_subset.index]
            
            if len(reordered_idx) == 0:
                return None, None, None, None, None
            
            # Extract data in new order
            X_reordered = X_subset.loc[reordered_idx].reset_index(drop=True)
            y_frost_reordered = y_frost_subset.loc[reordered_idx].reset_index(drop=True)
            y_temp_reordered = y_temp_subset.loc[reordered_idx].reset_index(drop=True)
            
            # Get station IDs in the same order
            if "Stn Id" in df_split_subset.columns:
                station_map = dict(zip(orig_idx_subset, df_split_subset["Stn Id"].values))
                station_ids_reordered = np.array([station_map.get(idx) for idx in reordered_idx])
            else:
                station_ids_reordered = None
            
            return X_reordered, y_frost_reordered, y_temp_reordered, station_ids_reordered, reordered_idx
        
        # Reorganize train/val/test data by station, then time
        train_result = reorganize_by_station(train_df, train_orig_idx, X, y_frost, y_temp)
        if train_result[0] is not None:
            X_train, y_frost_train, y_temp_train, station_ids_train, _ = train_result
        else:
            # Fallback to original order
            train_idx = train_orig_idx.intersection(X.index)
            train_idx_ordered = [idx for idx in train_orig_idx if idx in train_idx]
            X_train = X.loc[train_idx_ordered].reset_index(drop=True)
            y_frost_train = y_frost.loc[train_idx_ordered].reset_index(drop=True)
            y_temp_train = y_temp.loc[train_idx_ordered].reset_index(drop=True)
            if len(train_df) > 0 and "Stn Id" in train_df.columns:
                train_station_map = dict(zip(train_orig_idx, train_df["Stn Id"].values))
                station_ids_train = np.array([train_station_map.get(idx) for idx in train_idx_ordered])
            else:
                station_ids_train = None
        
        val_result = reorganize_by_station(val_df, val_orig_idx, X, y_frost, y_temp)
        if val_result[0] is not None:
            X_val, y_frost_val, y_temp_val, station_ids_val, _ = val_result
        else:
            # Fallback to original order
            val_idx = val_orig_idx.intersection(X.index)
            val_idx_ordered = [idx for idx in val_orig_idx if idx in val_idx]
            X_val = X.loc[val_idx_ordered].reset_index(drop=True)
            y_frost_val = y_frost.loc[val_idx_ordered].reset_index(drop=True)
            y_temp_val = y_temp.loc[val_idx_ordered].reset_index(drop=True)
            if len(val_df) > 0 and "Stn Id" in val_df.columns:
                val_station_map = dict(zip(val_orig_idx, val_df["Stn Id"].values))
                station_ids_val = np.array([val_station_map.get(idx) for idx in val_idx_ordered])
            else:
                station_ids_val = None
        
        test_result = reorganize_by_station(test_df, test_orig_idx, X, y_frost, y_temp)
        if test_result[0] is not None:
            X_test, y_frost_test, y_temp_test, station_ids_test, _ = test_result
        else:
            # Fallback to original order
            test_idx = test_orig_idx.intersection(X.index)
            test_idx_ordered = [idx for idx in test_orig_idx if idx in test_idx]
            X_test = X.loc[test_idx_ordered].reset_index(drop=True)
            y_frost_test = y_frost.loc[test_idx_ordered].reset_index(drop=True)
            y_temp_test = y_temp.loc[test_idx_ordered].reset_index(drop=True)
            if len(test_df) > 0 and "Stn Id" in test_df.columns:
                test_station_map = dict(zip(test_orig_idx, test_df["Stn Id"].values))
                station_ids_test = np.array([test_station_map.get(idx) for idx in test_idx_ordered])
            else:
                station_ids_test = None
    else:
        # For non-LSTM models, use simple intersection (order doesn't matter)
        train_idx = train_orig_idx.intersection(X.index)
        val_idx = val_orig_idx.intersection(X.index)
        test_idx = test_orig_idx.intersection(X.index)
        
        X_train = X.loc[train_idx]
        X_val = X.loc[val_idx]
        X_test = X.loc[test_idx]
        y_frost_train = y_frost.loc[train_idx]
        y_frost_val = y_frost.loc[val_idx]
        y_frost_test = y_frost.loc[test_idx]
        y_temp_train = y_temp.loc[train_idx]
        y_temp_val = y_temp.loc[val_idx]
        y_temp_test = y_temp.loc[test_idx]
        
        # Initialize station_ids for non-LSTM models (set to None)
        station_ids_train = None
        station_ids_val = None
        station_ids_test = None
        # Try to extract station IDs if available (for compatibility)
        if "Stn Id" in df.columns:
            try:
                station_ids_train = df.loc[train_idx, "Stn Id"].values if train_idx.intersection(df.index).size > 0 else None
                station_ids_val = df.loc[val_idx, "Stn Id"].values if val_idx.intersection(df.index).size > 0 else None
                station_ids_test = df.loc[test_idx, "Stn Id"].values if test_idx.intersection(df.index).size > 0 else None
            except Exception:
                station_ids_train = None
                station_ids_val = None
                station_ids_test = None
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Get model configuration
    max_workers = min(8, max(1, os.cpu_count() // 4))
    model_class = get_model_class(model_type)
    
    # Train classification model (frost probability)
    task_start_time = time.time()
    task_start_datetime = datetime.now()
    print(f"\n[{task_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}] Training classification model for frost probability...", flush=True)
    import sys
    sys.stdout.flush()
    
    frost_config = get_model_config(model_type, horizon, "classification", max_workers, for_loso=False)
    
    if model_type == "lstm_multitask":
        # Multi-task model needs both y_temp and y_frost
        model_frost = train_multitask_model(
            model_type, model_class, frost_config,
            X_train, y_temp_train, y_frost_train, station_ids_train,
            log_file=horizon_log_file
        )
        model_temp = model_frost  # Same instance for consistency
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Multi-task model already trained both temperature and frost prediction tasks together.")
    else:
        # Setup checkpoint directory for LSTM models
        if model_type in ["lstm", "lstm_multitask"]:
            frost_checkpoint_dir = horizon_dir / "checkpoints" / "frost_classifier"
            frost_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            frost_config["checkpoint_dir"] = str(frost_checkpoint_dir)
        
        # Pass checkpoint_dir and resume_from_checkpoint to fit method
        fit_kwargs = {"log_file": horizon_log_file}
        if model_type in ["lstm", "lstm_multitask"]:
            fit_kwargs['checkpoint_dir'] = str(frost_checkpoint_dir)
            # resume_from_checkpoint is optional, only pass if provided
            # (not used in standard training, only for resume scenarios)
        
        # Pass fit_kwargs to train_frost_model (for LSTM checkpoint_dir, etc.)
        # CRITICAL FIX: Pass station_ids_val for LSTM to use external validation
        if model_type == "lstm" and station_ids_val is not None:
            fit_kwargs['station_ids_val'] = station_ids_val
        model_frost = train_frost_model(
            model_type, model_class, frost_config,
            X_train, y_frost_train, X_val, y_frost_val, station_ids_train,
            **fit_kwargs
        )
        
        task_elapsed = time.time() - task_start_time
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Frost classification model training completed in {task_elapsed:.2f} seconds ({task_elapsed/60:.2f} minutes)", flush=True)
        import sys
        sys.stdout.flush()
        
        # Train regression model (temperature)
        task_start_time = time.time()
        task_start_datetime = datetime.now()
        print(f"\n[{task_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}] Training regression model for temperature...", flush=True)
        import sys
        sys.stdout.flush()
        
        temp_config = get_model_config(model_type, horizon, "regression", max_workers, for_loso=False)
        model_temp = train_temp_model(
            model_type, model_class, temp_config,
            X_train, y_temp_train, X_val, y_temp_val, station_ids_train,
            log_file=horizon_log_file
        )
        
        task_elapsed = time.time() - task_start_time
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Temperature regression model training completed in {task_elapsed:.2f} seconds ({task_elapsed/60:.2f} minutes)", flush=True)
        import sys
        sys.stdout.flush()
    
    # Evaluate on test set
    eval_start_time = time.time()
    eval_start_datetime = datetime.now()
    print(f"\n[{eval_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}] Evaluating on test set...")
    
    frost_metrics, temp_metrics, y_frost_pred, y_frost_proba, y_temp_pred = evaluate_models(
        model_type, model_frost, model_temp, X_test, y_frost_test, y_temp_test, station_ids_test
    )
    
    print(f"\nClassification Metrics (Frost Probability):")
    print(MetricsCalculator.format_metrics(frost_metrics))
    print(f"\nRegression Metrics (Temperature):")
    print(MetricsCalculator.format_metrics(temp_metrics))
    
    # Save models and results
    save_models_and_results(
        model_type, model_frost, model_temp, horizon_dir,
        frost_metrics, temp_metrics,
        y_frost_test, y_frost_pred, y_frost_proba,
        y_temp_test, y_temp_pred, horizon
    )
    
    eval_elapsed = time.time() - eval_start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluation completed in {eval_elapsed:.2f} seconds")
    
    # Free memory aggressively
    del X_train, X_val, X_test
    del y_frost_train, y_frost_val, y_frost_test
    del y_temp_train, y_temp_val, y_temp_test
    del y_frost_pred, y_frost_proba, y_temp_pred
    del model_frost, model_temp
    if 'station_ids_train' in locals():
        del station_ids_train
    if 'station_ids_val' in locals():
        del station_ids_val
    if 'station_ids_test' in locals():
        del station_ids_test
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU cache if using CUDA
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    horizon_elapsed = time.time() - horizon_start_time
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {horizon}h horizon training completed in {horizon_elapsed:.2f} seconds ({horizon_elapsed/60:.2f} minutes)", flush=True)
    import sys
    sys.stdout.flush()
    
    return {
        "horizon": horizon,
        "frost_metrics": frost_metrics,
        "temp_metrics": temp_metrics
    }

