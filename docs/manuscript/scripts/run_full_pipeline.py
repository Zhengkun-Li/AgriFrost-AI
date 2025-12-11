#!/usr/bin/env python3
"""Run full pipeline on real CIMIS data across matrix cells."""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import argparse
import os
import json
import copy

# Setup logging
_logger = logging.getLogger(__name__)

# Check if running in virtual environment
venv_path = Path(__file__).parent.parent / ".venv"
if venv_path.exists():
    venv_python = venv_path / "bin" / "python3"
    if venv_python.exists() and sys.executable != str(venv_python):
        _logger.warning("Not running in virtual environment!")
        print("⚠️  Warning: Not running in virtual environment!")
        print(f"   Current Python: {sys.executable}")
        print(f"   Expected Python: {venv_python}")
        print("   Consider using: ./scripts/run_with_venv.sh scripts/run_full_pipeline.py [args]")
        print()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.training.data_preparation import load_and_prepare_data
from src.models.ml.lightgbm_model import LightGBMModel
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.validators import CrossValidator
from src.visualization.plots import Plotter
from src.utils.path_utils import ensure_dir


DEFAULT_FULL_PIPELINE_FEATURE_CONFIG = {
    "time_features": True,
    "lag_features": {
        "enabled": True,
        "columns": [
            "Air Temp (C)",
            "Dew Point (C)",
            "Rel Hum (%)",
            "Sol Rad (W/sq.m)",
            "Wind Speed (m/s)",
            "Wind Dir (0-360)",
            "Soil Temp (C)",
            "ETo (mm)",
            "Precip (mm)",
            "Vap Pres (kPa)",
        ],
        "lags": [1, 3, 6, 12, 24],
    },
    "rolling_features": {
        "enabled": True,
        "columns": [
            "Air Temp (C)",
            "Dew Point (C)",
            "Rel Hum (%)",
            "Sol Rad (W/sq.m)",
            "Wind Speed (m/s)",
            "Soil Temp (C)",
            "ETo (mm)",
            "Precip (mm)",
            "Vap Pres (kPa)",
        ],
        "windows": [3, 6, 12, 24],
        "functions": ["mean", "min", "max", "std", "sum"],
    },
    "derived_features": True,
    "radiation_features": True,
    "wind_features": True,
    "humidity_features": True,
    "trend_features": True,
    "station_features": True,
    "station_metadata_path": "data/external/cimis_station_metadata.csv",
}


def infer_matrix_cell_from_output(output_dir: Path) -> str:
    candidates = {"A", "B", "C", "D", "E"}
    for part in output_dir.parts:
        if part in candidates:
            return part
    return "B"


def resolve_data_path(cli_path: Optional[Path]) -> Path:
    if cli_path:
        return Path(cli_path)
    raw_dir = project_root / "data" / "raw" / "frost-risk-forecast-challenge"
    stations_dir = raw_dir / "stations"
    if stations_dir.exists() and stations_dir.is_dir():
        return stations_dir
    candidate = raw_dir / "cimis_all_stations.csv.gz"
    if candidate.exists():
        return candidate
    return raw_dir / "cimis_all_stations.csv"


def run_pipeline_for_cell(matrix_cell: str, output_dir: Path, data_path: Path, args) -> None:
    """Run pipeline for a single matrix cell.
    
    Args:
        matrix_cell: Matrix cell identifier (A-E).
        output_dir: Output directory for results.
        data_path: Path to raw data.
        args: Command line arguments.
    
    Raises:
        ValueError: If matrix_cell is invalid.
        OSError: If file operations fail.
    """
    # Input validation
    if matrix_cell not in {"A", "B", "C", "D", "E"}:
        raise ValueError(f"Invalid matrix_cell: {matrix_cell}. Must be A, B, C, D, or E")
    
    try:
        ensure_dir(output_dir)
    except OSError as e:
        _logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise
    
    _logger.info(f"Running pipeline for matrix cell {matrix_cell}")
    print(f"\n=== Matrix Cell {matrix_cell} | Output: {output_dir} ===")
    
    feature_config = copy.deepcopy(DEFAULT_FULL_PIPELINE_FEATURE_CONFIG)
    if args.skip_cleaning:
        _logger.warning("DataPipeline always runs cleaning; --skip-cleaning is ignored")
        print("⚠️  DataPipeline 始终会执行清洗步骤，--skip-cleaning 将被忽略。")
    use_feature_engineering = matrix_cell in {"B", "D"}
    if args.skip_features:
        _logger.info("--skip-features specified: skipping feature engineering")
        print("ℹ️  --skip-features 指定：将跳过特征工程，仅使用原始变量。")
        use_feature_engineering = False
    elif use_feature_engineering:
        _logger.info("Will execute feature engineering (top175 track)")
        print("ℹ️  将执行特征工程（top175 轨）。")
    else:
        _logger.info("Raw/graph track: using raw variables")
        print("ℹ️  Raw/图轨：使用原始变量。")
    
    try:
        df_features, data_run_metadata = load_and_prepare_data(
            data_path,
            sample_size=args.sample_size,
            use_feature_engineering=use_feature_engineering,
            feature_engineering_config=feature_config if use_feature_engineering else None,
            matrix_cell=matrix_cell,
            return_metadata=True,
        )
    except (ValueError, FileNotFoundError, OSError) as e:
        _logger.error(f"Failed to load and prepare data: {e}")
        raise
    
    metadata_path = output_dir / "data_run_metadata.json"
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(data_run_metadata, f, indent=2)
        _logger.info(f"Recorded data pipeline metadata at {metadata_path}")
    except (IOError, OSError) as e:
        _logger.error(f"Failed to save metadata to {metadata_path}: {e}")
        raise OSError(f"Failed to save metadata: {e}") from e
    
    print(f"Recorded data pipeline metadata at {metadata_path}")
    
    features_path = output_dir / "features.parquet"
    try:
        df_features.to_parquet(features_path)
        _logger.info(f"Saved processed features to {features_path}")
    except (IOError, OSError) as e:
        _logger.error(f"Failed to save features to {features_path}: {e}")
        raise OSError(f"Failed to save features: {e}") from e
    
    print(f"Saved processed features to {features_path}")
    
    print("\n" + "="*60)
    print("Step 4: Preparing Training Data")
    print("="*60)
    
    if "Date" in df_features.columns:
        df_features["Date"] = pd.to_datetime(df_features["Date"])
    
    exclude_cols = {
        "Stn Id", "Stn Name", "CIMIS Region", "Date", "Hour (PST)", "Jul",
        "Air Temp (C)", "qc", "qc.1", "qc.2", "qc.3", "qc.4", "qc.5",
        "qc.6", "qc.7", "qc.8", "qc.9"
    }
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = df_features[feature_cols].copy()
    y = df_features["Air Temp (C)"].copy()
    mask = ~y.isna()
    X = X[mask].copy()
    y = y[mask].copy()
    df_features = df_features.loc[X.index]
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    train_df, val_df, test_df = CrossValidator.time_split(
        df_features, train_ratio=0.7, val_ratio=0.15
    )
    train_idx = train_df.index.intersection(X.index)
    val_idx = val_df.index.intersection(X.index)
    test_idx = test_df.index.intersection(X.index)
    
    X_train = X.loc[train_idx]; y_train = y.loc[train_idx]
    X_val = X.loc[val_idx]; y_val = y.loc[val_idx]
    X_test = X.loc[test_idx]; y_test = y.loc[test_idx]
    
    if X_train.empty or X_val.empty or X_test.empty:
        _logger.error(f"Insufficient data for splits. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        raise ValueError(f"Insufficient data for splits. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    _logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    print("\n" + "="*60)
    print(f"Step 5: Training {args.model.upper()} Model")
    print("="*60)
    
    if args.model == "lightgbm":
        model_class = LightGBMModel
        model_config = {
            "model_name": "lightgbm",
            "model_type": "lightgbm",
            "task_type": "regression",
            "model_params": {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 31,
                "random_state": 42,
                "verbose": -1
            }
        }
    else:
        from src.models.ml.xgboost_model import XGBoostModel
        model_class = XGBoostModel
        model_config = {
            "model_name": "xgboost",
            "model_type": "xgboost",
            "task_type": "regression",
            "model_params": {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "random_state": 42,
                "verbosity": 0
            }
        }
    
    try:
        model = model_class(model_config)
        model.fit(X_train, y_train, eval_set=[(X_val.values, y_val.values)])
    except (ValueError, AttributeError) as e:
        _logger.error(f"Failed to train model: {e}")
        raise ValueError(f"Model training failed: {e}") from e
    
    model_dir = output_dir / "model"
    try:
        model.save(model_dir)
        _logger.info(f"Model saved to {model_dir}")
    except OSError as e:
        _logger.error(f"Failed to save model to {model_dir}: {e}")
        raise
    
    print(f"Model saved to {model_dir}")
    
    print("\n" + "="*60)
    print("Step 6: Evaluating Model")
    print("="*60)
    
    all_results = {}
    all_predictions = {}
    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test)
    ]:
        try:
            y_pred = model.predict(X_split)
            metrics = MetricsCalculator.calculate_all_metrics(
                y_split.values, y_pred, task_type="regression"
            )
            all_results[split_name] = metrics
            all_predictions[split_name] = {
                "y_true": y_split.values.tolist(),
                "y_pred": y_pred.tolist()
            }
            _logger.info(f"{split_name} metrics calculated")
            print(f"\n{split_name.upper()} Metrics:")
            print(MetricsCalculator.format_metrics(metrics))
            
            metrics_path = output_dir / f"{split_name}_metrics.json"
            try:
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, default=str)
            except (IOError, OSError) as e:
                _logger.error(f"Failed to save {split_name} metrics to {metrics_path}: {e}")
                raise
        except (ValueError, TypeError) as e:
            _logger.error(f"Failed to evaluate on {split_name} set: {e}")
            raise
    
    predictions_path = output_dir / "predictions.json"
    try:
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, indent=2, default=str)
    except (IOError, OSError) as e:
        _logger.error(f"Failed to save predictions to {predictions_path}: {e}")
        raise OSError(f"Failed to save predictions: {e}") from e
    
    summary = {
        "model_name": model_config.get("model_name", "unknown"),
        "model_type": model_config.get("model_type", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "test_metrics": all_results.get("test", {})
    }
    summary_path = output_dir / "summary.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
    except (IOError, OSError) as e:
        _logger.error(f"Failed to save summary to {summary_path}: {e}")
        raise OSError(f"Failed to save summary: {e}") from e
    
    print("\n" + "="*60)
    print("Step 7: Generating Visualizations")
    print("="*60)
    
    plots_dir = output_dir / "plots"
    ensure_dir(plots_dir)
    plotter = Plotter(style="matplotlib", figsize=(14, 8))
    
    y_test_pred = model.predict(X_test)
    test_dates = df_features.loc[X_test.index, "Date"] if "Date" in df_features.columns else None
    plotter.plot_predictions(
        y_test.values,
        y_test_pred,
        dates=test_dates,
        title="Test Set Predictions",
        save_path=plots_dir / "predictions.png",
        show=False
    )
    importance = model.get_feature_importance()
    if importance is not None:
        plotter.plot_feature_importance(
            importance,
            top_n=20,
            title="Top 20 Feature Importance",
            save_path=plots_dir / "feature_importance.png",
            show=False
        )
        importance_path = output_dir / "feature_importance.csv"
        importance.to_csv(importance_path, index=False)
    
    if args.loso:
        print("\n" + "="*60)
        print("Step 8: LOSO (Leave-One-Station-Out) Evaluation")
        print("="*60)
        try:
            loso_splits = CrossValidator.leave_one_station_out(df_features)
        except (ValueError, KeyError) as e:
            _logger.error(f"Failed to create LOSO splits: {e}")
            raise
        
        loso_results = []
        for i, (train_df, test_df) in enumerate(loso_splits, 1):
            test_station = test_df["Stn Id"].iloc[0]
            _logger.info(f"Testing on station {test_station} ({i}/{len(loso_splits)})")
            print(f"\n[{i}/{len(loso_splits)}] Testing on station {test_station}...")
            train_idx = train_df.index.intersection(X.index)
            test_idx = test_df.index.intersection(X.index)
            if len(train_idx) == 0 or len(test_idx) == 0:
                _logger.warning(f"Insufficient data for station {test_station}")
                print("  Skipping (insufficient data)")
                continue
            X_train_loso = X.loc[train_idx]; y_train_loso = y.loc[train_idx]
            X_test_loso = X.loc[test_idx]; y_test_loso = y.loc[test_idx]
            model_loso = model_class(model_config)
            try:
                model_loso.fit(X_train_loso, y_train_loso)
                y_pred_loso = model_loso.predict(X_test_loso)
                metrics_loso = MetricsCalculator.calculate_all_metrics(
                    y_test_loso.values, y_pred_loso, task_type="regression"
                )
                metrics_loso["station_id"] = int(test_station)
                loso_results.append(metrics_loso)
                _logger.debug(f"Station {test_station}: MAE={metrics_loso['mae']:.4f}, RMSE={metrics_loso['rmse']:.4f}, R²={metrics_loso['r2']:.4f}")
                print(f"  MAE: {metrics_loso['mae']:.4f}, RMSE: {metrics_loso['rmse']:.4f}, R²: {metrics_loso['r2']:.4f}")
            except (ValueError, AttributeError) as e:
                _logger.error(f"Error training/testing for station {test_station}: {e}")
                print(f"  Error: {e}")
            except Exception as err:
                _logger.error(f"Unexpected error for station {test_station}: {err}", exc_info=True)
                print(f"  Error: {err}")
        if loso_results:
            loso_df = pd.DataFrame(loso_results)
            loso_dir = output_dir / "loso"
            try:
                ensure_dir(loso_dir)
                loso_df.to_csv(loso_dir / "station_metrics.csv", index=False)
                
                loso_summary = {
                    "mean_mae": float(loso_df["mae"].mean()),
                    "std_mae": float(loso_df["mae"].std()),
                    "mean_rmse": float(loso_df["rmse"].mean()),
                    "std_rmse": float(loso_df["rmse"].std()),
                    "mean_r2": float(loso_df["r2"].mean()),
                    "std_r2": float(loso_df["r2"].std()),
                    "n_stations": len(loso_df)
                }
                with open(loso_dir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump(loso_summary, f, indent=2)
                _logger.info(f"LOSO evaluation completed: {loso_summary}")
                print(f"\nLOSO Summary: {loso_summary}")
            except (IOError, OSError) as e:
                _logger.error(f"Failed to save LOSO results: {e}")
                raise
        else:
            _logger.warning("No LOSO results generated")
    
    _logger.info(f"Pipeline for matrix cell {matrix_cell} completed. Results saved to: {output_dir}")
    print("\nPipeline for matrix cell", matrix_cell, "completed. Results saved to:", output_dir)


def main():
    """Main pipeline function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description="Run full pipeline on real CIMIS data")
    parser.add_argument("--data", type=Path, default=None, help="Path to raw CIMIS data")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: experiments/full_pipeline_YYYYMMDD_HHMMSS)")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm", "xgboost"],
                        help="Model type to use")
    parser.add_argument("--skip-cleaning", action="store_true", help="(Deprecated) cleaning always runs in pipeline")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature engineering")
    parser.add_argument("--sample-size", type=int, default=None, help="Use only a sample of data (for quick testing)")
    parser.add_argument("--loso", action="store_true", help="Perform LOSO evaluation (simplified)")
    parser.add_argument("--matrix-cell", type=str, choices=list("ABCDE"),
                        help="Single matrix cell hint (default inferred from output path)")
    parser.add_argument("--matrix-cells", type=str, nargs="+", choices=list("ABCDE"),
                        help="Run pipeline for multiple matrix cells, e.g., --matrix-cells A B D")
    
    args = parser.parse_args()
    
    if args.output:
        base_output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = project_root / "experiments" / f"full_pipeline_{timestamp}"
    ensure_dir(base_output_dir)
    print(f"Base output directory: {base_output_dir}")
    
    data_path = resolve_data_path(args.data)
    if not data_path.exists():
        _logger.error(f"Data path not found: {data_path}")
        print(f"❌ Data path not found: {data_path}")
        print("Expected stations directory or cimis_all_stations.csv(.gz)")
        return 1
    
    if args.matrix_cells:
        matrix_cells = args.matrix_cells
    else:
        inferred_cell = args.matrix_cell or infer_matrix_cell_from_output(base_output_dir)
        matrix_cells = [inferred_cell]
    
    # Validate matrix cells
    for cell in matrix_cells:
        if cell not in {"A", "B", "C", "D", "E"}:
            _logger.error(f"Invalid matrix cell: {cell}")
            print(f"❌ Invalid matrix cell: {cell}. Must be A, B, C, D, or E")
            return 1
    
    for idx, cell in enumerate(matrix_cells, 1):
        cell_output_dir = base_output_dir if len(matrix_cells) == 1 else base_output_dir / cell
        _logger.info(f"Running pipeline for matrix cell {cell} ({idx}/{len(matrix_cells)})")
        print(f"\n==== [{idx}/{len(matrix_cells)}] Running pipeline for matrix cell {cell} ====")
        try:
            run_pipeline_for_cell(cell, cell_output_dir, data_path, args)
        except (ValueError, OSError) as e:
            _logger.error(f"Failed to run pipeline for matrix cell {cell}: {e}")
            print(f"❌ Failed to run pipeline for matrix cell {cell}: {e}")
            return 1
        except Exception as e:
            _logger.error(f"Unexpected error running pipeline for matrix cell {cell}: {e}", exc_info=True)
            print(f"❌ Unexpected error: {e}")
            return 1
    
    if len(matrix_cells) > 1:
        _logger.info("Multi-cell pipeline completed")
        print("\nMulti-cell pipeline completed. Individual results stored in:")
        for cell in matrix_cells:
            cell_dir = base_output_dir if len(matrix_cells) == 1 else base_output_dir / cell
            print(f"  - {cell}: {cell_dir}")
    else:
        _logger.info(f"Pipeline completed successfully. Results saved to: {base_output_dir}")
        print(f"\nPipeline completed successfully. Results saved to: {base_output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())