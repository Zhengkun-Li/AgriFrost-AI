#!/usr/bin/env python3
"""Train multi-horizon frost forecasting models.

This script trains models to predict:
1. Frost probability (classification) for horizons 3h, 6h, 12h, 24h
2. Temperature (regression) for the same horizons

Output format: "There is a 30% chance of frost in the next 3 hours, predicted temperature: 4.50 °C"
"""

import sys
import os
# Force unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import argparse
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
try:
    import torch
except Exception:
    torch = None

from src.utils.path_utils import ensure_dir
from src.training.data_preparation import load_and_prepare_data, create_frost_labels
from src.training.model_trainer import train_models_for_horizon
from src.training.loso_evaluator import perform_loso_evaluation


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train multi-horizon frost forecasting models"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to raw CIMIS data (default: auto-detect)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: experiments/frost_forecast_YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[3, 6, 12, 24],
        help="Forecast horizons in hours (default: 3 6 12 24)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=[
            "lightgbm", "xgboost", "catboost", "random_forest", "ensemble",
            "lstm", "lstm_multitask", "prophet",
            "extratrees", "linear_regression", "ridge", "elasticnet", "logreg", "persistence", "gru", "tcn",
            "dcrnn", "st_gcn", "gat_lstm", "graphwavenet"  # Graph neural network models
        ],
        help="Model type (default: lightgbm)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for quick testing (default: use all data)"
    )
    parser.add_argument(
        "--frost-threshold",
        type=float,
        default=0.0,
        help="Temperature threshold for frost in °C (default: 0.0)"
    )
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Perform Leave-One-Station-Out evaluation"
    )
    parser.add_argument(
        "--resume-loso",
        action="store_true",
        help="Resume LOSO evaluation from checkpoint (if supported)"
    )
    parser.add_argument(
        "--feature-selection",
        type=Path,
        default=None,
        help="Path to feature selection config JSON file"
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=None,
        help="Use top K features based on importance (overrides feature selection config)"
    )
    parser.add_argument(
        "--save-loso-models",
        action="store_true",
        help="Save all LOSO models (default: False, saves ~180MB for 144 models)"
    )
    parser.add_argument(
        "--save-loso-worst-n",
        type=int,
        default=None,
        help="Save only the worst N stations' models (e.g., --save-loso-worst-n 3)"
    )
    parser.add_argument(
        "--save-loso-horizons",
        type=int,
        nargs="+",
        default=None,
        help="Save models only for specified horizons (e.g., --save-loso-horizons 24)"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "experiments" / f"frost_forecast_{timestamp}"
    
    ensure_dir(output_dir)
    
    # Setup root training log file
    root_log_path = output_dir / "training_root.log"
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    
    # Open log file for appending (will create if doesn't exist)
    log_file = open(root_log_path, 'a', buffering=1)  # Line buffered
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create a class that writes to both file and stdout
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
        def isatty(self):
            # Check if any of the files is a TTY (for tqdm compatibility)
            return any(hasattr(f, 'isatty') and f.isatty() for f in self.files if hasattr(f, 'isatty'))
        def fileno(self):
            # Return the fileno of the first file (usually stdout)
            if self.files and hasattr(self.files[0], 'fileno'):
                return self.files[0].fileno()
            return None
    
    # Redirect stdout and stderr to both file and console
    sys.stdout = TeeOutput(original_stdout, log_file)
    sys.stderr = TeeOutput(original_stderr, log_file)
    
    try:
        print(f"Output directory: {output_dir}")
        print(f"Training log: {root_log_path}")
        
        # Find data path
        if args.data:
            data_path = Path(args.data)
        else:
            raw_dir = project_root / "data" / "raw" / "frost-risk-forecast-challenge"
            stations_dir = raw_dir / "stations"
            if stations_dir.exists() and stations_dir.is_dir():
                data_path = stations_dir
            else:
                data_path = raw_dir / "cimis_all_stations.csv.gz"
                if not data_path.exists():
                    data_path = raw_dir / "cimis_all_stations.csv"
        
        if not data_path.exists():
            print(f"❌ Data path not found: {data_path}")
            return 1
        
        # Optional CUDA matmul precision tuning and runtime info
        if torch is not None and torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
            try:
                device_name = torch.cuda.get_device_name(0)
                cudnn_version = getattr(torch.backends.cudnn, "version", lambda: None)()
                print(f"\nCUDA Runtime Info: device={device_name}, cuDNN={cudnn_version}")
            except Exception:
                pass
        # Record training start time
        training_start_time = time.time()
        training_start_datetime = datetime.now()
        print("="*80)
        print(f"🚀 Training Started")
        print(f"   Start Time: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Model Type: {args.model}")
        print(f"   Horizons: {args.horizons}")
        print(f"   Output Directory: {output_dir}")
        print("="*80)
        
        # Determine if we should use feature engineering based on output path
        # A轨 (Raw-only, 单站): experiments/A/... -> skip feature engineering
        # E轨 (Raw-only, 图模型): experiments/E/... -> skip feature engineering
        # B轨 (FE 175): experiments/B/... -> use feature engineering
        output_str = str(output_dir)
        use_fe = not ("/A/" in output_str or output_str.endswith("/A") or 
                      "/E/" in output_str or output_str.endswith("/E") or 
                      "raw" in output_str.lower())
        if not use_fe:
            print(f"  ℹ️  Raw-only track detected: skipping feature engineering")
        
        # Load and prepare data
        df_features = load_and_prepare_data(data_path, args.sample_size, use_feature_engineering=use_fe)
        
        # Create frost labels
        df_labeled = create_frost_labels(
            df_features,
            horizons=args.horizons,
            frost_threshold=args.frost_threshold
        )
        
        # Save labeled data
        labeled_path_top = output_dir / "labeled_data.parquet"
        labeled_path_full = output_dir / "full_training" / "labeled_data.parquet"
        df_labeled.to_parquet(labeled_path_top)
        ensure_dir(output_dir / "full_training")
        df_labeled.to_parquet(labeled_path_full)
        print(f"\nSaved labeled data to {labeled_path_top} and {labeled_path_full}")
        
        # Load feature selection config if provided
        feature_selection = None
        if args.feature_selection:
            feature_selection_path = Path(args.feature_selection)
            if feature_selection_path.exists():
                with open(feature_selection_path, 'r') as f:
                    feature_selection = json.load(f)
                print(f"\nLoaded feature selection config from {feature_selection_path}")
                print(f"  Enabled: {feature_selection.get('enabled', False)}")
                print(f"  Method: {feature_selection.get('method', 'N/A')}")
                if feature_selection.get('top_k'):
                    print(f"  Top K: {feature_selection.get('top_k')}")
        elif args.top_k_features:
            # Use top K features based on importance
            importance_path = project_root / "experiments" / "lightgbm" / "feature_importance" / "feature_importance_3h_all.csv"
            if importance_path.exists():
                feature_selection = {
                    "enabled": True,
                    "method": "importance",
                    "top_k": args.top_k_features,
                    "importance_path": str(importance_path),
                    "save_report": True
                }
                print(f"\nUsing top {args.top_k_features} features based on importance")
                print(f"  Importance file: {importance_path}")
            else:
                print(f"⚠️  Warning: Importance file not found: {importance_path}")
                print("  Continuing without feature selection...")
        
        # Train models for each horizon
        horizons_start_time = time.time()
        horizons_start_datetime = datetime.now()
        print(f"\n[{horizons_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}] Starting model training for all horizons...")
        results = {}
        for horizon in args.horizons:
            try:
                result = train_models_for_horizon(
                    df_labeled,
                    horizon,
                    output_dir,
                    model_type=args.model,
                    skip_if_exists=True,
                    feature_selection=feature_selection
                )
                results[horizon] = result
                
                # Free memory aggressively after each horizon
                import gc
                del result  # Free result dictionary
                gc.collect()
                
                # Clear GPU cache if using CUDA
                try:
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception as e:
                print(f"❌ Error training {horizon}h model: {e}")
                import traceback
                traceback.print_exc()
        
        horizons_elapsed = time.time() - horizons_start_time
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All horizons training completed in {horizons_elapsed:.2f} seconds ({horizons_elapsed/60:.2f} minutes)")
        
        # Generate summary report
        print("\n" + "="*60)
        print("Summary Report")
        print("="*60)
        
        training_end_time = time.time()
        training_end_datetime = datetime.now()
        training_duration = training_end_time - training_start_time
        
        summary = {
            "model_type": args.model,
            "horizons": args.horizons,
            "frost_threshold": args.frost_threshold,
            "start_time": training_start_datetime.isoformat(),
            "end_time": training_end_datetime.isoformat(),
            "duration_seconds": training_duration,
            "duration_minutes": training_duration / 60,
            "duration_hours": training_duration / 3600,
            "timestamp": training_end_datetime.isoformat(),
            "results": {}
        }
        
        for horizon, result in results.items():
            summary["results"][f"{horizon}h"] = {
                "frost_metrics": result["frost_metrics"],
                "temp_metrics": result["temp_metrics"]
            }
            
            print(f"\n{horizon}h Horizon:")
            print(f"  Frost - Brier: {result['frost_metrics'].get('brier_score', 'N/A'):.4f}, "
                  f"ECE: {result['frost_metrics'].get('ece', 'N/A'):.4f}, "
                  f"ROC-AUC: {result['frost_metrics'].get('roc_auc', 'N/A'):.4f}")
            print(f"  Temp  - MAE: {result['temp_metrics'].get('mae', 'N/A'):.4f}, "
                  f"RMSE: {result['temp_metrics'].get('rmse', 'N/A'):.4f}, "
                  f"R²: {result['temp_metrics'].get('r2', 'N/A'):.4f}")
        
        # Save summary
        summary_path = output_dir / "full_training" / "summary.json"
        ensure_dir(output_dir / "full_training")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✅ Training completed!")
        print(f"   Start Time: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   End Time: {training_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Duration: {training_duration/60:.2f} minutes ({training_duration/3600:.2f} hours)")
        print(f"Results saved to: {output_dir}")
        
        # LOSO Evaluation
        if args.loso:
            loso_start_time = time.time()
            loso_start_datetime = datetime.now()
            print("\n" + "="*60)
            print("LOSO (Leave-One-Station-Out) Evaluation")
            print(f"[{loso_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}] Starting LOSO evaluation...")
            print("="*60)
            
            # Free memory before LOSO evaluation
            labeled_path = output_dir / "labeled_data.parquet"
            if not labeled_path.exists():
                df_labeled.to_parquet(labeled_path)
                print(f"Saved labeled data to {labeled_path} for LOSO evaluation")
            
            # Free the large DataFrame from memory
            del df_labeled
            import gc
            gc.collect()
            print("Freed memory: released full dataset from RAM")
            
            # Perform LOSO evaluation
            loso_results = perform_loso_evaluation(
                labeled_path,
                args.horizons,
                output_dir,
                model_type=args.model,
                frost_threshold=args.frost_threshold,
                resume=args.resume_loso,
                feature_selection=feature_selection,
                save_models=args.save_loso_models,
                save_worst_n=args.save_loso_worst_n,
                save_horizons=args.save_loso_horizons
            )
            
            # Save LOSO results
            loso_dir = output_dir / "loso"
            ensure_dir(loso_dir)
            
            loso_end_time = time.time()
            loso_end_datetime = datetime.now()
            loso_duration = loso_end_time - loso_start_time
            
            loso_summary = loso_results["summary"]
            loso_summary["start_time"] = loso_start_datetime.isoformat()
            loso_summary["end_time"] = loso_end_datetime.isoformat()
            loso_summary["duration_seconds"] = loso_duration
            loso_summary["duration_minutes"] = loso_duration / 60
            loso_summary["duration_hours"] = loso_duration / 3600
            
            with open(loso_dir / "summary.json", "w") as f:
                json.dump(loso_summary, f, indent=2, default=str)
            
            # Print summary
            print("\n" + "="*60)
            print("LOSO Summary Statistics")
            print("="*60)
            for horizon in args.horizons:
                horizon_key = f"{horizon}h"
                if horizon_key in loso_results["summary"]:
                    h_summary = loso_results["summary"][horizon_key]
                    print(f"\n{horizon}h Horizon:")
                    print(f"  Stations evaluated: {h_summary.get('n_stations', 0)}")
                    frost_summary = h_summary.get("frost_metrics", {})
                    temp_summary = h_summary.get("temp_metrics", {})
                    
                    if "brier_score" in frost_summary:
                        brier = frost_summary["brier_score"]
                        print(f"  Brier Score: {brier.get('mean', 'N/A'):.4f} ± {brier.get('std', 'N/A'):.4f}")
                    if "ece" in frost_summary:
                        ece = frost_summary["ece"]
                        print(f"  ECE: {ece.get('mean', 'N/A'):.4f} ± {ece.get('std', 'N/A'):.4f}")
                    if "roc_auc" in frost_summary:
                        roc = frost_summary["roc_auc"]
                        print(f"  ROC-AUC: {roc.get('mean', 'N/A'):.4f} ± {roc.get('std', 'N/A'):.4f}")
                    if "pr_auc" in frost_summary:
                        pr = frost_summary["pr_auc"]
                        print(f"  PR-AUC: {pr.get('mean', 'N/A'):.4f} ± {pr.get('std', 'N/A'):.4f}")
                    if "mae" in temp_summary:
                        mae = temp_summary["mae"]
                        print(f"  Temp MAE: {mae.get('mean', 'N/A'):.4f} ± {mae.get('std', 'N/A'):.4f}°C")
                    if "rmse" in temp_summary:
                        rmse = temp_summary["rmse"]
                        print(f"  Temp RMSE: {rmse.get('mean', 'N/A'):.4f} ± {rmse.get('std', 'N/A'):.4f}°C")
                    if "r2" in temp_summary:
                        r2 = temp_summary["r2"]
                        print(f"  Temp R²: {r2.get('mean', 'N/A'):.4f} ± {r2.get('std', 'N/A'):.4f}")
            
            print(f"\n✅ LOSO evaluation completed!")
            print(f"   Start Time: {loso_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   End Time: {loso_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Duration: {loso_duration/60:.2f} minutes ({loso_duration/3600:.2f} hours)")
            print(f"Results saved to: {loso_dir}")
            
            # Update main summary with total time
            total_duration = training_end_time - training_start_time + loso_duration
            summary["total_duration_seconds"] = total_duration
            summary["total_duration_minutes"] = total_duration / 60
            summary["total_duration_hours"] = total_duration / 3600
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
    
    finally:
        # Restore original stdout/stderr and close log file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
