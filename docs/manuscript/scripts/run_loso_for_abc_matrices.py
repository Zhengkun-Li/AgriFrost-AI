#!/usr/bin/env python3
"""Run LOSO evaluation for best ABC matrix configurations.

This script:
1. Loads best configurations for matrices A, B, C from supplementary data
2. Runs LOSO evaluation for each matrix using best configurations
3. Collects and compares LOSO results across matrices
4. Saves results for manuscript table generation
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import json
import logging
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.loso_evaluator import perform_loso_evaluation
from src.training.data_preparation import load_and_prepare_data, create_frost_labels
from src.utils.path_utils import ensure_dir

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SUPPLEMENTARY_DIR = PROJECT_ROOT / "docs/manuscript/Supplementary_lighgbm_abc"
RESULTS_DIR = PROJECT_ROOT / "experiments/lightgbm"
HORIZONS = [3, 6, 12, 24]


def load_best_configurations() -> pd.DataFrame:
    """Load best configurations from supplementary data."""
    best_config_path = SUPPLEMENTARY_DIR / "summary_best_configurations.csv"
    
    if not best_config_path.exists():
        logger.warning(f"Best configurations file not found: {best_config_path}")
        logger.info("Will use default configurations based on PR-AUC from metrics files")
        return None
    
    df = pd.read_csv(best_config_path)
    logger.info(f"Loaded best configurations: {len(df)} rows")
    return df


def get_best_config_for_matrix(matrix: str, horizon: int, df_best: Optional[pd.DataFrame]) -> Dict:
    """Get best configuration for a matrix and horizon."""
    if df_best is not None:
        config = df_best[(df_best['Matrix'] == matrix) & (df_best['Horizon_h'] == horizon)]
        if len(config) > 0:
            row = config.iloc[0]
            return {
                'matrix': matrix,
                'horizon': horizon,
                'radius_km': row.get('Radius_km', '--'),
                'training': row.get('Training', 'Balanced'),
                'pr_auc': row.get('Frost_PR_AUC', None)
            }
    
    # Fallback: use default best configurations
    defaults = {
        'A': {'radius': None, 'training': 'Balanced'},
        'B': {'radius': None, 'training': 'Balanced'},
        'C': {
            3: {'radius': 60, 'training': 'Balanced'},
            6: {'radius': 100, 'training': 'Balanced'},
            12: {'radius': 200, 'training': 'Balanced'},
            24: {'radius': 200, 'training': 'Balanced'}
        }
    }
    
    if matrix == 'C':
        config = defaults['C'].get(horizon, {'radius': 200, 'training': 'Balanced'})
        return {
            'matrix': matrix,
            'horizon': horizon,
            'radius_km': config['radius'],
            'training': config['training'],
            'pr_auc': None
        }
    else:
        config = defaults[matrix]
        return {
            'matrix': matrix,
            'horizon': horizon,
            'radius_km': None,
            'training': config['training'],
            'pr_auc': None
        }


def find_data_path(matrix: str, radius_km: Optional[int] = None) -> Path:
    """Find the data path for a matrix configuration."""
    if matrix == 'A':
        # Matrix A: raw features, no spatial aggregation
        data_dir = RESULTS_DIR / "raw" / "A"
        parquet_path = data_dir / "labeled_data.parquet"
        if parquet_path.exists():
            return parquet_path
        
        # Try to find in full_training_balance
        full_training_dir = data_dir / "full_training_balance"
        if full_training_dir.exists():
            parquet_path = full_training_dir / "labeled_data.parquet"
            if parquet_path.exists():
                return parquet_path
        
        # Fallback: use raw data source
        return PROJECT_ROOT / "data/raw/frost-risk-forecast-challenge/stations"
    
    elif matrix == 'B':
        # Matrix B: feature engineering, no spatial aggregation
        data_dir = RESULTS_DIR / "feature_engineering" / "B"
        parquet_path = data_dir / "labeled_data.parquet"
        if parquet_path.exists():
            return parquet_path
        
        # Try to find in full_training_balance
        full_training_dir = data_dir / "full_training_balance"
        if full_training_dir.exists():
            parquet_path = full_training_dir / "labeled_data.parquet"
            if parquet_path.exists():
                return parquet_path
        
        # Fallback: use raw data source (will be processed with feature engineering)
        return PROJECT_ROOT / "data/raw/frost-risk-forecast-challenge/stations"
    
    elif matrix == 'C':
        # Matrix C: spatial aggregation with specific radius
        if radius_km is None:
            radius_km = 100  # Default
        
        radius_dir = f"full_training_balance_{radius_km}km"
        data_dir = RESULTS_DIR / "raw" / "C" / radius_dir
        parquet_path = data_dir / "labeled_data.parquet"
        if parquet_path.exists():
            return parquet_path
        
        # Try alternative naming
        radius_dir_alt = f"radius_{radius_km}km"
        data_dir_alt = RESULTS_DIR / "raw" / "C" / radius_dir_alt / "full_training_balance"
        parquet_path_alt = data_dir_alt / "labeled_data.parquet"
        if parquet_path_alt.exists():
            return parquet_path_alt
        
        # Fallback: use raw data source (will be processed with spatial aggregation)
        return PROJECT_ROOT / "data/raw/frost-risk-forecast-challenge/stations"
    
    else:
        raise ValueError(f"Unknown matrix: {matrix}")


def run_loso_for_matrix(matrix: str, horizons: List[int], radius_km: Optional[int] = None, output_base: Path = None) -> Dict:
    """Run LOSO evaluation for a matrix with all horizons.
    
    Args:
        matrix: Matrix identifier (A, B, or C)
        horizons: List of horizons to evaluate (e.g., [3, 6, 12, 24])
        radius_km: Radius for Matrix C (None for A and B)
        output_base: Base output directory (default: RESULTS_DIR)
    """
    if output_base is None:
        output_base = RESULTS_DIR
    
    logger.info(f"Running LOSO for Matrix {matrix}, Horizons: {horizons}h")
    if radius_km:
        logger.info(f"  Radius: {radius_km}km")
    
    # Determine output directory (base_dir for loso_evaluator)
    # loso_evaluator logic: base_dir = output_dir if matrix_cell in out_parts else (output_dir / matrix_cell)
    # loso_dir = base_dir / "loso"
    # For Matrix C with different radii, we need separate output directories to avoid overwriting
    if matrix == 'C' and radius_km:
        # For Matrix C, create radius-specific output directory
        # This ensures different radius LOSO results don't overwrite each other
        base_dir = output_base / "raw" / "C" / f"loso_radius_{radius_km}km"
        # Ensure directory exists (loso_evaluator will create loso/ subdirectory)
        ensure_dir(base_dir)
    elif matrix == 'B':
        base_dir = output_base / "feature_engineering" / "B"
    else:  # Matrix A
        base_dir = output_base / "raw" / "A"
    
    # Find data path
    data_path = find_data_path(matrix, radius_km)
    logger.info(f"  Data path: {data_path}")
    logger.info(f"  Output base_dir: {base_dir}")
    
    # Determine track and matrix_cell
    track = "raw" if matrix in ['A', 'C'] else "feature_engineering"
    matrix_cell = matrix
    
    # Run LOSO evaluation for all horizons at once
    try:
        result = perform_loso_evaluation(
            data_source=data_path,
            horizons=horizons,  # All horizons at once
            output_dir=base_dir,  # Pass base_dir, evaluator will create loso/ subdirectory
            model_type="lightgbm",
            frost_threshold=0.0,
            resume=True,  # Resume if already partially completed
            track=track,
            matrix_cell=matrix_cell
        )
        
        logger.info(f"  ✅ LOSO completed for Matrix {matrix}, Horizons: {horizons}h")
        return result
    
    except Exception as e:
        logger.error(f"  ❌ LOSO failed for Matrix {matrix}, Horizons: {horizons}h: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def collect_loso_results() -> pd.DataFrame:
    """Collect all LOSO results from experiment directories."""
    results = []
    
    # Matrix A
    loso_dir_a = RESULTS_DIR / "raw" / "A" / "full_training_balance" / "loso"
    if loso_dir_a.exists():
        summary_file = loso_dir_a / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                for horizon in HORIZONS:
                    h_key = f"{horizon}h"
                    if h_key in summary:
                        h_data = summary[h_key]
                        frost_metrics = h_data.get('frost_metrics', {})
                        temp_metrics = h_data.get('temp_metrics', {})
                        results.append({
                            'Matrix': 'A',
                            'Horizon_h': horizon,
                            'ROC_AUC_LOSO': frost_metrics.get('roc_auc', {}).get('mean'),
                            'PR_AUC_LOSO': frost_metrics.get('pr_auc', {}).get('mean'),
                            'Recall_LOSO': None,  # Not in summary
                            'Precision_LOSO': None,
                            'MAE_LOSO': temp_metrics.get('mae', {}).get('mean'),
                            'RMSE_LOSO': temp_metrics.get('rmse', {}).get('mean'),
                            'R2_LOSO': temp_metrics.get('r2', {}).get('mean')
                        })
    
    # Matrix B
    loso_dir_b = RESULTS_DIR / "feature_engineering" / "B" / "full_training_balance" / "loso"
    if loso_dir_b.exists():
        summary_file = loso_dir_b / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                for horizon in HORIZONS:
                    h_key = f"{horizon}h"
                    if h_key in summary:
                        h_data = summary[h_key]
                        frost_metrics = h_data.get('frost_metrics', {})
                        temp_metrics = h_data.get('temp_metrics', {})
                        results.append({
                            'Matrix': 'B',
                            'Horizon_h': horizon,
                            'ROC_AUC_LOSO': frost_metrics.get('roc_auc', {}).get('mean'),
                            'PR_AUC_LOSO': frost_metrics.get('pr_auc', {}).get('mean'),
                            'Recall_LOSO': None,
                            'Precision_LOSO': None,
                            'MAE_LOSO': temp_metrics.get('mae', {}).get('mean'),
                            'RMSE_LOSO': temp_metrics.get('rmse', {}).get('mean'),
                            'R2_LOSO': temp_metrics.get('r2', {}).get('mean')
                        })
    
    # Matrix C - check radius-specific LOSO directories
    # Each radius has its own LOSO directory to avoid overwriting
    optimal_radius = {3: 60, 6: 100, 12: 200, 24: 200}
    for radius in [60, 100, 200]:
        # Check radius-specific directory
        loso_dir_c = RESULTS_DIR / "raw" / "C" / f"loso_radius_{radius}km" / "loso"
        if not loso_dir_c.exists():
            # Try standard location (for backward compatibility)
            loso_dir_c = RESULTS_DIR / "raw" / "C" / "loso"
        
        if loso_dir_c.exists():
            summary_file = loso_dir_c / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                    # Get horizons that use this radius
                    horizons_for_radius = [h for h, r in optimal_radius.items() if r == radius]
                    for horizon in horizons_for_radius:
                        h_key = f"{horizon}h"
                        if h_key in summary:
                            h_data = summary[h_key]
                            frost_metrics = h_data.get('frost_metrics', {})
                            temp_metrics = h_data.get('temp_metrics', {})
                            results.append({
                                'Matrix': 'C',
                                'Radius_km': radius,
                                'Horizon_h': horizon,
                                'ROC_AUC_LOSO': frost_metrics.get('roc_auc', {}).get('mean'),
                                'PR_AUC_LOSO': frost_metrics.get('pr_auc', {}).get('mean'),
                                'Recall_LOSO': None,
                                'Precision_LOSO': None,
                                'MAE_LOSO': temp_metrics.get('mae', {}).get('mean'),
                                'RMSE_LOSO': temp_metrics.get('rmse', {}).get('mean'),
                                'R2_LOSO': temp_metrics.get('r2', {}).get('mean')
                            })
                break  # Found results for this radius, move to next
    
    if results:
        df = pd.DataFrame(results)
        return df
    else:
        return pd.DataFrame()


def main():
    """Main function to run LOSO evaluation for ABC matrices."""
    logger.info("=" * 70)
    logger.info("LOSO Evaluation for ABC Matrices")
    logger.info("=" * 70)
    
    # Load best configurations
    df_best = load_best_configurations()
    
    # Collect existing LOSO results
    logger.info("\n1. Collecting existing LOSO results...")
    df_existing = collect_loso_results()
    
    if len(df_existing) > 0:
        logger.info(f"   Found {len(df_existing)} existing LOSO results")
        logger.info(f"   Matrices: {df_existing['Matrix'].unique()}")
    else:
        logger.info("   No existing LOSO results found")
    
    # Determine which matrices need LOSO evaluation
    # For each matrix, run LOSO once with all horizons
    matrices_to_run = []
    
    for matrix in ['A', 'B', 'C']:
        # Check if all horizons exist for this matrix
        if matrix == 'C':
            # For Matrix C, check each radius-horizon combination
            # We need to run LOSO for each optimal radius
            optimal_radius = {3: 60, 6: 100, 12: 200, 24: 200}
            radius_horizon_map = {}  # Map radius to list of horizons using that radius
            
            if len(df_existing) == 0:
                # No existing results, need to run all radius-horizon combinations
                for horizon in HORIZONS:
                    radius = optimal_radius[horizon]
                    if radius not in radius_horizon_map:
                        radius_horizon_map[radius] = []
                    radius_horizon_map[radius].append(horizon)
            else:
                for horizon in HORIZONS:
                    radius = optimal_radius[horizon]
                exists = len(df_existing[
                    (df_existing['Matrix'] == matrix) &
                    (df_existing.get('Radius_km', None) == radius) &
                    (df_existing['Horizon_h'] == horizon)
                ]) > 0
                    if not exists:
                        if radius not in radius_horizon_map:
                            radius_horizon_map[radius] = []
                        radius_horizon_map[radius].append(horizon)
            
            if radius_horizon_map:
                # Run LOSO for each radius with only the horizons that need it
                for radius, horizons_for_radius in radius_horizon_map.items():
                    matrices_to_run.append({
                        'matrix': matrix,
                        'horizons': horizons_for_radius,
                        'radius_km': radius
                    })
                    logger.info(f"   Will run LOSO: Matrix {matrix}, Radius {radius}km, Horizons {horizons_for_radius}h")
            else:
                logger.info(f"   Skipping (exists): Matrix {matrix} (all radii-horizons)")
        else:
            # For Matrix A and B, check if all horizons exist
            missing_horizons = []
            if len(df_existing) == 0:
                # No existing results, need to run all horizons
                missing_horizons = HORIZONS
            else:
                for horizon in HORIZONS:
                exists = len(df_existing[
                    (df_existing['Matrix'] == matrix) &
                    (df_existing['Horizon_h'] == horizon)
                ]) > 0
            if not exists:
                        missing_horizons.append(horizon)
            
            if missing_horizons:
                matrices_to_run.append({
                    'matrix': matrix,
                    'horizons': HORIZONS,  # Run all horizons
                    'radius_km': None
                })
                logger.info(f"   Will run LOSO: Matrix {matrix}, Horizons {HORIZONS}h")
            else:
                logger.info(f"   Skipping (exists): Matrix {matrix} (all horizons)")
    
    # Run LOSO for missing matrices (SEQUENTIALLY - one at a time to save memory)
    if matrices_to_run:
        logger.info(f"\n2. Running LOSO for {len(matrices_to_run)} matrix configurations...")
        logger.info("   ⚠️  Running SEQUENTIALLY (one at a time) to avoid memory overflow")
        logger.info("   ⚠️  Each LOSO run processes stations one by one with memory cleanup")
        
        for i, config in enumerate(matrices_to_run, 1):
            logger.info(f"\n   [{i}/{len(matrices_to_run)}] Starting LOSO for Matrix {config['matrix']}...")
            try:
                result = run_loso_for_matrix(
                    matrix=config['matrix'],
                    horizons=config['horizons'],
                    radius_km=config.get('radius_km'),
                    output_base=RESULTS_DIR
                )
                if result:
                    logger.info(f"   ✅ Completed Matrix {config['matrix']} LOSO")
                else:
                    logger.warning(f"   ⚠️  Matrix {config['matrix']} LOSO returned None (may have failed)")
            except Exception as e:
                logger.error(f"   ❌ Matrix {config['matrix']} LOSO failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Continue with next matrix even if one fails
                continue
            
            # Force garbage collection between matrices to free memory
            import gc
            gc.collect()
            logger.info(f"   Memory cleaned up after Matrix {config['matrix']}")
    else:
        logger.info("\n2. All LOSO evaluations already completed!")
    
    # Collect final results
    logger.info("\n3. Collecting final LOSO results...")
    df_final = collect_loso_results()
    
    # Save results
    output_file = SUPPLEMENTARY_DIR / "loso_results_abc.csv"
    df_final.to_csv(output_file, index=False)
    logger.info(f"   Saved LOSO results to: {output_file}")
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("LOSO Results Summary")
    logger.info("=" * 70)
    print(df_final.to_string())
    
    logger.info("\n✅ LOSO evaluation complete!")


if __name__ == "__main__":
    main()

