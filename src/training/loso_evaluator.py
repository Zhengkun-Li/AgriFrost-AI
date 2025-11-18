"""LOSO (Leave-One-Station-Out) evaluation module for frost forecasting.

This module handles:
- LOSO cross-validation evaluation
- Per-station model training and evaluation
- LOSO summary statistics calculation
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import time
import json
import os
import gc

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.data.preprocessors import preprocess_with_loso
from src.evaluation.metrics import MetricsCalculator
from src.utils.path_utils import ensure_dir
from src.training.data_preparation import prepare_features_and_targets
from src.training.model_config import get_model_config, get_model_class
from src.training.model_trainer import (
    train_frost_model, train_temp_model, train_multitask_model,
    evaluate_models
)


def calculate_loso_summary(
    station_metrics: List[Dict],
    horizons: List[int]
) -> Dict:
    """Calculate summary statistics across stations.
    
    Args:
        station_metrics: List of station result dictionaries.
        horizons: List of forecast horizons in hours.
    
    Returns:
        Dictionary with mean ± std for each metric and horizon.
    """
    summary = {}
    
    for horizon in horizons:
        horizon_key = f"{horizon}h"
        
        # Collect metrics for this horizon
        brier_scores = []
        ece_scores = []
        roc_aucs = []
        pr_aucs = []
        mae_scores = []
        rmse_scores = []
        r2_scores = []
        
        for station_result in station_metrics:
            if horizon_key in station_result.get("horizons", {}):
                h_metrics = station_result["horizons"][horizon_key]
                frost_metrics = h_metrics.get("frost_metrics", {})
                temp_metrics = h_metrics.get("temp_metrics", {})
                
                if "brier_score" in frost_metrics and not np.isnan(frost_metrics["brier_score"]):
                    brier_scores.append(frost_metrics["brier_score"])
                if "ece" in frost_metrics and not np.isnan(frost_metrics["ece"]):
                    ece_scores.append(frost_metrics["ece"])
                if "roc_auc" in frost_metrics and not np.isnan(frost_metrics["roc_auc"]):
                    roc_aucs.append(frost_metrics["roc_auc"])
                if "pr_auc" in frost_metrics and not np.isnan(frost_metrics["pr_auc"]):
                    pr_aucs.append(frost_metrics["pr_auc"])
                if "mae" in temp_metrics and not np.isnan(temp_metrics["mae"]):
                    mae_scores.append(temp_metrics["mae"])
                if "rmse" in temp_metrics and not np.isnan(temp_metrics["rmse"]):
                    rmse_scores.append(temp_metrics["rmse"])
                if "r2" in temp_metrics and not np.isnan(temp_metrics["r2"]):
                    r2_scores.append(temp_metrics["r2"])
        
        # Calculate statistics
        summary[horizon_key] = {
            "n_stations": len([s for s in station_metrics if horizon_key in s.get("horizons", {})]),
            "frost_metrics": {
                "brier_score": {
                    "mean": float(np.mean(brier_scores)) if brier_scores else np.nan,
                    "std": float(np.std(brier_scores)) if brier_scores else np.nan,
                    "min": float(np.min(brier_scores)) if brier_scores else np.nan,
                    "max": float(np.max(brier_scores)) if brier_scores else np.nan
                },
                "ece": {
                    "mean": float(np.mean(ece_scores)) if ece_scores else np.nan,
                    "std": float(np.std(ece_scores)) if ece_scores else np.nan,
                    "min": float(np.min(ece_scores)) if ece_scores else np.nan,
                    "max": float(np.max(ece_scores)) if ece_scores else np.nan
                },
                "roc_auc": {
                    "mean": float(np.mean(roc_aucs)) if roc_aucs else np.nan,
                    "std": float(np.std(roc_aucs)) if roc_aucs else np.nan,
                    "min": float(np.min(roc_aucs)) if roc_aucs else np.nan,
                    "max": float(np.max(roc_aucs)) if roc_aucs else np.nan
                },
                "pr_auc": {
                    "mean": float(np.mean(pr_aucs)) if pr_aucs else np.nan,
                    "std": float(np.std(pr_aucs)) if pr_aucs else np.nan,
                    "min": float(np.min(pr_aucs)) if pr_aucs else np.nan,
                    "max": float(np.max(pr_aucs)) if pr_aucs else np.nan
                }
            },
            "temp_metrics": {
                "mae": {
                    "mean": float(np.mean(mae_scores)) if mae_scores else np.nan,
                    "std": float(np.std(mae_scores)) if mae_scores else np.nan,
                    "min": float(np.min(mae_scores)) if mae_scores else np.nan,
                    "max": float(np.max(mae_scores)) if mae_scores else np.nan
                },
                "rmse": {
                    "mean": float(np.mean(rmse_scores)) if rmse_scores else np.nan,
                    "std": float(np.std(rmse_scores)) if rmse_scores else np.nan,
                    "min": float(np.min(rmse_scores)) if rmse_scores else np.nan,
                    "max": float(np.max(rmse_scores)) if rmse_scores else np.nan
                },
                "r2": {
                    "mean": float(np.mean(r2_scores)) if r2_scores else np.nan,
                    "std": float(np.std(r2_scores)) if r2_scores else np.nan,
                    "min": float(np.min(r2_scores)) if r2_scores else np.nan,
                    "max": float(np.max(r2_scores)) if r2_scores else np.nan
                }
            }
        }
    
    return summary


def train_loso_models_for_horizon(
    model_type: str,
    horizon: int,
    test_station: int,
    X_train: pd.DataFrame,
    y_frost_train: pd.Series,
    y_temp_train: pd.Series,
    X_test: pd.DataFrame,
    y_frost_test: pd.Series,
    y_temp_test: pd.Series,
    station_ids_train: Optional[np.ndarray] = None,
    log_file: Optional[str] = None
) -> Tuple[object, object, Dict, Dict]:
    """Train models for a specific station and horizon in LOSO evaluation.
    
    Args:
        model_type: Model type.
        horizon: Forecast horizon in hours.
        test_station: Test station ID.
        X_train: Training features.
        y_frost_train: Training frost labels.
        y_temp_train: Training temperature values.
        X_test: Test features.
        y_frost_test: Test frost labels.
        y_temp_test: Test temperature values.
        station_ids_train: Optional station IDs for LSTM models.
    
    Returns:
        Tuple of (model_frost, model_temp, frost_metrics, temp_metrics).
    """
    max_workers_loso = min(8, max(1, os.cpu_count() // 4))
    model_class = get_model_class(model_type)
    
    # Train classification model (frost probability)
    frost_config = get_model_config(model_type, horizon, "classification", max_workers_loso, for_loso=True, station_id=test_station)
    
    if model_type == "lstm_multitask":
        model_frost = train_multitask_model(
            model_type, model_class, frost_config,
            X_train, y_temp_train, y_frost_train, station_ids_train,
            **({"log_file": log_file} if log_file else {})
        )
        model_temp = model_frost
        print(f"      ℹ️  Multi-task model already trained both temperature and frost prediction tasks together.")
    else:
        model_frost = train_frost_model(
            model_type, model_class, frost_config,
            X_train, y_frost_train, X_test, y_frost_test, station_ids_train,
            **({"log_file": log_file} if log_file else {})
        )
        
        # Train regression model (temperature)
        temp_config = get_model_config(model_type, horizon, "regression", max_workers_loso, for_loso=True, station_id=test_station)
        model_temp = train_temp_model(
            model_type, model_class, temp_config,
            X_train, y_temp_train, X_test, y_temp_test, station_ids_train,
            **({"log_file": log_file} if log_file else {})
        )
    
    # Evaluate models
    frost_metrics, temp_metrics, y_frost_pred, y_frost_proba, y_temp_pred = evaluate_models(
        model_type, model_frost, model_temp, X_test, y_frost_test, y_temp_test
    )
    
    return model_frost, model_temp, frost_metrics, temp_metrics


def perform_loso_evaluation(
    data_source,  # Can be DataFrame or Path to parquet file
    horizons: List[int],
    output_dir: Path,
    model_type: str = "lightgbm",
    frost_threshold: float = 0.0,
    resume: bool = False,
    feature_selection: Optional[Dict] = None,
    save_models: bool = False,
    save_worst_n: Optional[int] = None,
    save_horizons: Optional[List[int]] = None,
    track: Optional[str] = None,
    matrix_cell: Optional[str] = None
) -> Dict:
    """Perform LOSO evaluation with no data leakage and optimized memory usage.
    
    Process one station at a time:
    1. Load data on-demand (from disk if path provided)
    2. For each station, process all horizons
    3. Train on all other stations (17 stations)
    4. Test on this station (1 station)
    5. Save results immediately after each station completes
    6. Free memory after each station
    7. Support resume to skip completed stations
    8. Optionally save models based on criteria
    
    Args:
        data_source: Labeled DataFrame OR Path to parquet file with features and targets
        horizons: List of forecast horizons in hours
        output_dir: Output directory for results
        model_type: Model type
        frost_threshold: Temperature threshold for frost
        resume: If True, skip already completed stations
        feature_selection: Optional feature selection config
        save_models: If True, save all LOSO models
        save_worst_n: If specified, save only the worst N stations' models
        save_horizons: If specified, save models only for these horizons
    
    Returns:
        Dictionary with summary statistics and per-station metrics.
    """
    # Load data if path provided, otherwise use DataFrame
    if isinstance(data_source, (str, Path)):
        data_path = Path(data_source)
        print(f"Loading data from disk: {data_path}")
        df_full = pd.read_parquet(data_path)
        print(f"Loaded {len(df_full)} rows, {len(df_full.columns)} columns")
        print(f"Memory usage: {df_full.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
        # Optimize data types
        for col in df_full.select_dtypes(include=['float64']).columns:
            df_full[col] = pd.to_numeric(df_full[col], downcast='float')
        for col in df_full.select_dtypes(include=['int64']).columns:
            df_full[col] = pd.to_numeric(df_full[col], downcast='integer')
        print(f"Memory usage after optimization: {df_full.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    else:
        df_full = data_source
    
    # Get station IDs first (before creating splits to save memory)
    station_ids = sorted(df_full["Stn Id"].unique())
    print(f"Found {len(station_ids)} stations: {station_ids}")
    
    # Store data source for later use (if path provided, we'll reload; if DataFrame, we'll reuse)
    data_source_for_loop = data_source
    is_path_source = isinstance(data_source, (str, Path))
    
    # Create LOSO splits (store station IDs and masks)
    loso_splits = []
    for test_station_id in station_ids:
        train_mask = df_full["Stn Id"] != test_station_id
        test_mask = df_full["Stn Id"] == test_station_id
        loso_splits.append((test_station_id, train_mask, test_mask))
    
    # If we loaded from path, free the DataFrame now (we'll reload per station)
    if is_path_source:
        del df_full
        gc.collect()
        print("Freed full dataset from memory - will reload per station on-demand")
    else:
        print("Using provided DataFrame - will create subsets using masks (memory efficient)")
    
    # Infer track/matrix_cell from output_dir if not provided
    parts = [p for p in output_dir.parts]
    if track is None:
        track = "raw" if "raw" in parts else ("top175_features" if "top175_features" in parts else "top175_features")
    if matrix_cell is None:
        candidates = {"A","B","C","D","E"}
        found = [p for p in parts if p in candidates]
        matrix_cell = found[0] if found else ("B" if track == "top175_features" else "A")
    
    # Resolve base dir: add matrix_cell only if not present
    out_parts = set(output_dir.parts)
    base_dir = output_dir if matrix_cell in out_parts else (output_dir / matrix_cell)
    loso_dir = base_dir / "loso"
    ensure_dir(loso_dir)
    
    # Determine which horizons to save models for
    horizons_to_save = set(horizons)
    if save_horizons is not None:
        horizons_to_save = set(save_horizons)
        print(f"📊 Will save models only for horizons: {sorted(horizons_to_save)}h")
    
    # Checkpoint file for tracking completed stations
    checkpoint_file = loso_dir / "checkpoint.json"
    station_results_file = loso_dir / "station_results.json"
    
    # Load existing results if resuming
    completed_stations = set()
    station_metrics = []
    if resume and checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
            completed_stations = set(checkpoint.get("completed_stations", []))
            print(f"📋 Resuming: {len(completed_stations)} stations already completed")
            print(f"   Completed stations: {sorted(completed_stations)}")
        
        if station_results_file.exists():
            with open(station_results_file, "r") as f:
                station_metrics = json.load(f)
                print(f"   Loaded {len(station_metrics)} station results")
    
    loso_eval_start_time = time.time()
    loso_eval_start_datetime = datetime.now()
    print(f"\n{'='*60}")
    print(f"LOSO Evaluation: Processing {len(loso_splits)} stations")
    print(f"   Memory optimization: Load data on-demand, free after each station")
    print(f"   Horizons: {horizons}")
    print(f"[{loso_eval_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}] Starting LOSO evaluation...")
    print(f"{'='*60}")
    
    # Process by station (one station at a time)
    for i, (test_station, train_mask, test_mask) in enumerate(loso_splits, 1):
        test_station = int(test_station)
        
        # Skip if already completed
        if resume and test_station in completed_stations:
            print(f"\n[{i}/{len(loso_splits)}] Station {test_station}: ✅ Already completed, skipping...")
            continue
        
        station_start_time = time.time()
        station_start_datetime = datetime.now()
        print(f"\n{'='*60}")
        print(f"[{i}/{len(loso_splits)}] Processing Station {test_station}")
        print(f"   Train stations: {len(station_ids)-1}, Test station: {test_station}")
        print(f"[{station_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}] Starting station {test_station}...")
        print(f"{'='*60}")
        
        # Initialize station result
        station_result = {
            "station_id": test_station,
            "horizons": {}
        }
        
        # Load or use data for this station
        df_station = None
        if is_path_source:
            data_path = Path(data_source_for_loop)
            print(f"  Loading data for station {test_station}...")
            df_station = pd.read_parquet(data_path)
            # Optimize data types immediately
            for col in df_station.select_dtypes(include=['float64']).columns:
                df_station[col] = pd.to_numeric(df_station[col], downcast='float')
            for col in df_station.select_dtypes(include=['int64']).columns:
                df_station[col] = pd.to_numeric(df_station[col], downcast='integer')
            print(f"  Memory usage: {df_station.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
            
            train_df = df_station[train_mask].copy()
            test_df = df_station[test_mask].copy()
        else:
            print(f"  Creating train/test splits for station {test_station}...")
            train_df = data_source_for_loop[train_mask].copy()
            test_df = data_source_for_loop[test_mask].copy()
        
        print(f"  Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        # Process all horizons for this station
        for horizon in horizons:
            try:
                horizon_start_time = time.time()
                horizon_start_datetime = datetime.now()
                print(f"\n  [{horizon_start_datetime.strftime('%H:%M:%S')}] Processing {horizon}h horizon...")
                
                # Get indices first
                train_idx = train_df.index
                test_idx = test_df.index
                combined_idx = train_idx.union(test_idx)
                
                # Prepare features and targets
                if is_path_source and df_station is not None:
                    data_for_features = df_station
                else:
                    data_for_features = pd.concat([train_df, test_df])
                
                X, y_frost, y_temp = prepare_features_and_targets(
                    data_for_features, 
                    horizon, 
                    indices=combined_idx,
                    feature_selection=feature_selection,
                    track=track
                )
                
                # Get indices for train and test sets (after filtering)
                train_idx_filtered = train_idx.intersection(X.index)
                test_idx_filtered = test_idx.intersection(X.index)
                
                if len(train_idx_filtered) == 0 or len(test_idx_filtered) == 0:
                    print(f"    ⚠️  Skipping (no data)")
                    continue
                
                print(f"    Train samples: {len(train_idx_filtered)}, Test samples: {len(test_idx_filtered)}")
                
                X_train_raw = X.loc[train_idx_filtered]
                X_test_raw = X.loc[test_idx_filtered]
                y_frost_train = y_frost.loc[train_idx_filtered]
                y_frost_test = y_frost.loc[test_idx_filtered]
                y_temp_train = y_temp.loc[train_idx_filtered]
                y_temp_test = y_temp.loc[test_idx_filtered]
                
                # Get station IDs for LSTM models
                station_ids_train_loso = None
                if model_type in ["lstm", "lstm_multitask"] and "Stn Id" in data_for_features.columns:
                    station_ids_train_loso = data_for_features.loc[train_idx_filtered, "Stn Id"].values if len(train_idx_filtered) > 0 else None
                
                # Preprocess with no data leakage
                X_train, X_test = preprocess_with_loso(
                    train_df.loc[train_idx_filtered],
                    test_df.loc[test_idx_filtered],
                    feature_cols=list(X.columns),
                    scaling_method=None  # No scaling for tree-based models
                )
                
                # Train and evaluate models
                # Prepare per-station/horizon log file
                model_dir = loso_dir / f"station_{test_station}" / f"horizon_{horizon}h"
                ensure_dir(model_dir)
                loso_log_file = str(model_dir / "training.log")
                
                model_frost, model_temp, frost_metrics, temp_metrics = train_loso_models_for_horizon(
                    model_type, horizon, test_station,
                    X_train, y_frost_train, y_temp_train,
                    X_test, y_frost_test, y_temp_test,
                    station_ids_train_loso,
                    log_file=loso_log_file
                )
                
                # Store results
                station_result["horizons"][f"{horizon}h"] = {
                    "frost_metrics": frost_metrics,
                    "temp_metrics": temp_metrics
                }
                
                horizon_elapsed = time.time() - horizon_start_time
                print(f"    ✅ Brier={frost_metrics.get('brier_score', 'N/A'):.4f}, "
                      f"ECE={frost_metrics.get('ece', 'N/A'):.4f}, "
                      f"ROC-AUC={frost_metrics.get('roc_auc', 'N/A'):.4f}, "
                      f"MAE={temp_metrics.get('mae', 'N/A'):.4f}°C")
                print(f"    [{datetime.now().strftime('%H:%M:%S')}] {horizon}h horizon completed in {horizon_elapsed:.2f} seconds ({horizon_elapsed/60:.2f} minutes)")
                
                # Save models if criteria are met
                should_save = False
                if save_models:
                    should_save = True
                elif save_worst_n is not None:
                    should_save = True  # Save all temporarily, filter later
                elif save_horizons is not None:
                    should_save = horizon in horizons_to_save
                
                if should_save:
                    model_dir = loso_dir / f"station_{test_station}" / f"horizon_{horizon}h"
                    ensure_dir(model_dir)
                    if model_type == "lstm_multitask":
                        model_frost.save(model_dir / "multitask_model")
                        model_frost.save(model_dir / "frost_classifier")
                        model_frost.save(model_dir / "temp_regressor")
                    else:
                        model_frost.save(model_dir / "frost_classifier")
                        model_temp.save(model_dir / "temp_regressor")
                    if save_worst_n is not None:
                        print(f"    💾 Saved models (temporary, will filter to worst {save_worst_n} stations)")
                    else:
                        print(f"    💾 Saved models to {model_dir}")
                
                # Free memory after each horizon
                del X_train_raw, X_test_raw
                del X_train, X_test
                del y_frost_train, y_frost_test, y_temp_train, y_temp_test
                del model_frost, model_temp
                gc.collect()
                
            except Exception as e:
                print(f"    ❌ Error processing {horizon}h horizon: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Store station result
        station_metrics.append(station_result)
        
        # Save checkpoint and results after each station
        completed_stations.add(test_station)
        checkpoint = {
            "completed_stations": sorted(completed_stations),
            "last_updated": datetime.now().isoformat()
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        
        with open(station_results_file, "w") as f:
            json.dump(station_metrics, f, indent=2, default=str)
        
        station_elapsed = time.time() - station_start_time
        station_end_datetime = datetime.now()
        print(f"\n[{station_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}] Station {test_station} completed in {station_elapsed:.2f} seconds ({station_elapsed/60:.2f} minutes)")
        
        # Free memory after each station
        del train_df, test_df
        if df_station is not None:
            del df_station
        gc.collect()
    
    # Calculate summary statistics
    print(f"\n{'='*60}")
    print("Calculating LOSO summary statistics...")
    print(f"{'='*60}")
    
    summary = calculate_loso_summary(station_metrics, horizons)
    
    # Save summary
    with open(loso_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save per-station metrics as CSV
    station_rows = []
    for station_result in station_metrics:
        station_id = station_result["station_id"]
        for horizon_key, h_metrics in station_result.get("horizons", {}).items():
            row = {"station_id": station_id, "horizon": horizon_key}
            frost_metrics = h_metrics.get("frost_metrics", {})
            for key, value in frost_metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    row[f"frost_{key}"] = value
            temp_metrics = h_metrics.get("temp_metrics", {})
            for key, value in temp_metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    row[f"temp_{key}"] = value
            station_rows.append(row)
    
    if station_rows:
        station_metrics_df = pd.DataFrame(station_rows)
        station_metrics_df.to_csv(loso_dir / "station_metrics.csv", index=False)
        print(f"✅ Saved station metrics to {loso_dir / 'station_metrics.csv'}")
    
    loso_eval_end_time = time.time()
    loso_eval_end_datetime = datetime.now()
    loso_eval_duration = loso_eval_end_time - loso_eval_start_time
    
    print(f"\n{'='*60}")
    print(f"LOSO Evaluation Complete")
    print(f"   Started: {loso_eval_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Ended: {loso_eval_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Duration: {loso_eval_duration:.2f} seconds ({loso_eval_duration/60:.2f} minutes, {loso_eval_duration/3600:.2f} hours)")
    print(f"{'='*60}")
    
    return {
        "summary": summary,
        "station_metrics": station_metrics,
        "n_stations": len(station_ids),
        "horizons": horizons
    }

