#!/usr/bin/env python3
"""Feature selection script powered by DataPipeline and config-driven rules."""

import sys
import logging
from pathlib import Path
from typing import Optional
import argparse
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

# Setup logging
_logger = logging.getLogger(__name__)

from src.training.data_preparation import (
    load_and_prepare_data,
    create_frost_labels,
    prepare_features_and_targets,
)
from src.data.feature_selection import (
    FeatureSelector,
    load_feature_selection_config,
    select_features_with_config,
)

DEFAULT_OUTPUT_DIR = project_root / "feature_selection"


def resolve_data_path(raw_data: Optional[Path]) -> Path:
    if raw_data:
        return Path(raw_data)
    raw_dir = project_root / "data" / "raw" / "frost-risk-forecast-challenge"
    stations_dir = raw_dir / "stations"
    if stations_dir.exists() and stations_dir.is_dir():
        return stations_dir
    gz_path = raw_dir / "cimis_all_stations.csv.gz"
    if gz_path.exists():
        return gz_path
    return raw_dir / "cimis_all_stations.csv"


def infer_matrix_cell_from_path(path: Path, default: str = "B") -> str:
    candidates = {"A", "B", "C", "D", "E"}
    for part in path.parts:
        if part in candidates:
            return part
    return default


def load_labeled_dataframe(
    labeled_path: Optional[Path],
    raw_data_path: Optional[Path],
    matrix_cell: str,
    frost_threshold: float,
    sample_size: Optional[int],
) -> tuple[pd.DataFrame, Optional[dict]]:
    """Load labeled dataframe from either labeled file or raw data.
    
    Args:
        labeled_path: Path to pre-labeled parquet file.
        raw_data_path: Path to raw data (if labeled_path not provided).
        matrix_cell: Matrix cell identifier (A-E).
        frost_threshold: Temperature threshold for frost labeling.
        sample_size: Optional sample size limit.
    
    Returns:
        Tuple of (DataFrame, metadata_dict).
    
    Raises:
        FileNotFoundError: If data path not found.
        ValueError: If matrix_cell is invalid or inputs are invalid.
    """
    # Input validation
    if matrix_cell not in {"A", "B", "C", "D", "E"}:
        raise ValueError(f"Invalid matrix_cell: {matrix_cell}. Must be A, B, C, D, or E")
    
    if labeled_path:
        labeled_path = Path(labeled_path)
        if not labeled_path.exists():
            raise FileNotFoundError(f"Labeled data path not found: {labeled_path}")
        
        _logger.info(f"Loading labeled data from {labeled_path}...")
        print(f"Loading labeled data from {labeled_path}...")
        try:
            df = pd.read_parquet(labeled_path)
            return df, None
        except (IOError, OSError, pd.errors.EmptyDataError) as e:
            raise OSError(f"Failed to load labeled data from {labeled_path}: {e}") from e
    
    data_path = resolve_data_path(raw_data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    _logger.info(f"Running DataPipeline on {data_path} (matrix_cell={matrix_cell})...")
    print(f"Running DataPipeline on {data_path} (matrix_cell={matrix_cell})...")
    try:
        df_features, metadata = load_and_prepare_data(
            data_path,
            sample_size=sample_size,
            use_feature_engineering=matrix_cell in {"B", "D"},
            matrix_cell=matrix_cell,
            return_metadata=True,
        )
        df_labeled = create_frost_labels(
            df_features,
            horizons=[3, 6, 12, 24],
            frost_threshold=frost_threshold,
        )
        return df_labeled, metadata
    except (ValueError, FileNotFoundError, OSError) as e:
        _logger.error(f"Failed to prepare data: {e}")
        raise


def build_cli_config(args) -> dict:
    method = args.method
    return {
        "min_importance": args.min_importance,
        "max_correlation": args.max_correlation,
        "max_missing_rate": args.max_missing,
        "min_variance": args.min_variance,
        "remove_correlated": method in {"correlation", "all"},
        "remove_high_missing": method in {"missing", "all"},
        "remove_low_variance": method in {"variance", "all"},
        "importance_threshold": args.min_importance if method in {"importance", "all"} else None,
        "top_k": args.top_k,
        "importance_path": str(args.importance) if args.importance else None,
        "save_report": True,
    }


def main():
    """Main feature selection function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description="Config-driven feature selection")
    parser.add_argument("--data", type=Path, help="Path to labeled parquet (legacy)")
    parser.add_argument("--raw-data", type=Path, help="Path to raw stations directory or CSV")
    parser.add_argument("--matrix-cell", type=str, choices=list("ABCDE"),
                        help="Matrix cell hint (A/B/C/D/E)")
    parser.add_argument("--horizon", type=int, default=3, choices=[3, 6, 12, 24],
                        help="Forecast horizon to extract features for")
    parser.add_argument("--frost-threshold", type=float, default=0.0,
                        help="Threshold for frost labeling when using raw data")
    parser.add_argument("--config", type=Path,
                        help="Feature selection config JSON (overrides CLI thresholds)")
    parser.add_argument("--importance", type=Path,
                        help="Feature importance CSV (used when config references importance)")
    parser.add_argument("--method", type=str, default="all",
                        choices=["importance", "correlation", "missing", "variance", "all"],
                        help="Quick selection switches (ignored if --config provided)")
    parser.add_argument("--top-k", type=int, help="Keep top K features (importance mode)")
    parser.add_argument("--min-importance", type=float, default=0.01,
                        help="Minimum importance ratio (legacy CLI)")
    parser.add_argument("--max-correlation", type=float, default=0.95,
                        help="Maximum allowable correlation")
    parser.add_argument("--max-missing", type=float, default=0.5,
                        help="Maximum missing rate")
    parser.add_argument("--min-variance", type=float, default=0.0,
                        help="Minimum variance threshold")
    parser.add_argument("--sample-size", type=int, help="Sample size for quick experiments")
    parser.add_argument("--output", type=Path,
                        help="Path to selected features JSON (default: feature_selection/selected_features.json)")
    parser.add_argument("--report", type=Path,
                        help="Path to selection report JSON (default: alongside output)")
    
    args = parser.parse_args()
    
    # Input validation
    if args.horizon not in [3, 6, 12, 24]:
        _logger.error(f"Invalid horizon: {args.horizon}")
        print(f"❌ Invalid horizon: {args.horizon}. Must be 3, 6, 12, or 24")
        return 1
    
    output_path = args.output or (DEFAULT_OUTPUT_DIR / f"selected_features_h{args.horizon}.json")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _logger.error(f"Failed to create output directory {output_path.parent}: {e}")
        return 1
    
    report_path = args.report or (output_path.parent / f"{output_path.stem}_report.json")
    
    matrix_cell = args.matrix_cell or infer_matrix_cell_from_path(output_path, default="B")
    try:
        df, metadata = load_labeled_dataframe(
            labeled_path=args.data,
            raw_data_path=args.raw_data,
            matrix_cell=matrix_cell,
            frost_threshold=args.frost_threshold,
            sample_size=args.sample_size,
        )
    except (FileNotFoundError, ValueError, OSError) as e:
        _logger.error(f"Failed to load labeled dataframe: {e}")
        print(f"❌ Failed to load data: {e}")
        return 1
    
    if df.empty:
        _logger.error("Loaded dataset is empty")
        print("❌ Loaded dataset is empty")
        return 1
    
    _logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Loaded dataset: {len(df)} rows, columns={len(df.columns)}")
    
    track = "top175_features" if matrix_cell in {"B", "D"} else "raw"
    _logger.info(f"Preparing features for horizon {args.horizon}h (track={track})...")
    print(f"\nPreparing features for horizon {args.horizon}h (track={track})...")
    
    try:
        X, _, _ = prepare_features_and_targets(df, args.horizon, track=track)
        _logger.info(f"Feature matrix: {len(X)} samples × {len(X.columns)} columns")
    except (ValueError, KeyError) as e:
        _logger.error(f"Failed to prepare features: {e}")
        print(f"❌ Failed to prepare features: {e}")
        return 1
    
    print(f"Feature matrix: {len(X)} samples × {len(X.columns)} columns")
    
    try:
        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                _logger.error(f"Config file not found: {config_path}")
                print(f"❌ Config file not found: {config_path}")
                return 1
            fs_config = load_feature_selection_config(config_path)
            if "report_path" not in fs_config:
                fs_config["report_path"] = str(report_path)
        else:
            fs_config = build_cli_config(args)
            fs_config["report_path"] = str(report_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        _logger.error(f"Failed to load feature selection config: {e}")
        print(f"❌ Failed to load config: {e}")
        return 1
    
    feature_importance = None
    importance_path = fs_config.get("importance_path")
    if args.importance and not importance_path:
        fs_config["importance_path"] = str(args.importance)
        importance_path = str(args.importance)
    if importance_path and Path(importance_path).exists():
        try:
            feature_importance = pd.read_csv(importance_path)
            _logger.info(f"Loaded feature importance from {importance_path}")
            print(f"Loaded feature importance from {importance_path}")
        except (IOError, OSError, pd.errors.EmptyDataError) as e:
            _logger.warning(f"Failed to load feature importance from {importance_path}: {e}")
    
    try:
        X_selected, selector = select_features_with_config(
            X,
            fs_config,
            feature_importance=feature_importance
        )
    except (ValueError, KeyError) as e:
        _logger.error(f"Feature selection failed: {e}")
        print(f"❌ Feature selection failed: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("Feature Selection Results")
    print("=" * 60)
    print(f"Original features: {len(X.columns)}")
    print(f"Selected features: {len(X_selected.columns)}")
    print(f"Reduction: {(1 - len(X_selected.columns) / len(X.columns)) * 100:.1f}%")
    
    report = selector.get_selection_report()
    print("\nRemoved features by category:")
    for category, count in report["removed_features"].items():
        print(f"  {category}: {count}")
    
    output_payload = {
        "selected_features": list(X_selected.columns),
        "n_selected": len(X_selected.columns),
        "n_original": len(X.columns),
        "reduction_rate": 1 - len(X_selected.columns) / len(X.columns),
        "removed_features": report["removed_features_detail"],
        "config": fs_config,
        "metadata_path": str(report_path),
    }
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        _logger.info(f"Selected features saved to {output_path}")
        print(f"\n✅ Selected features saved to {output_path}")
        print(f"✅ Selection report saved to {report_path}")
    except (IOError, OSError) as e:
        _logger.error(f"Failed to save selected features to {output_path}: {e}")
        print(f"❌ Failed to save selected features: {e}")
        return 1
    
    if metadata:
        metadata_path = output_path.parent / "data_run_metadata.json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            _logger.info(f"Data pipeline metadata saved to {metadata_path}")
            print(f"ℹ️  Data pipeline metadata saved to {metadata_path}")
        except (IOError, OSError) as e:
            _logger.warning(f"Failed to save metadata to {metadata_path}: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

