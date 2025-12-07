#!/usr/bin/env python3
"""Generate pipeline_config.yaml for existing trained models.

This script creates pipeline_config.yaml for models trained before the fix
that saves this file automatically. It extracts configuration from run_metadata.json
and reconstructs the pipeline config.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_pipeline_config(horizon_dir: Path) -> bool:
    """Generate pipeline_config.yaml for an existing model.
    
    Args:
        horizon_dir: Path to horizon directory (e.g., horizon_12h/)
    
    Returns:
        True if config was generated, False otherwise
    """
    # Load run_metadata.json
    metadata_file = horizon_dir / "run_metadata.json"
    if not metadata_file.exists():
        logger.warning(f"No run_metadata.json found in {horizon_dir}")
        return False
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    matrix_cell = metadata.get("matrix_cell", "A")
    track = metadata.get("track", "raw")
    radius_km = metadata.get("radius_km")
    
    # Build pipeline config based on metadata
    pipeline_config = {
        "cleaning": {
            "config_path": "config/data_cleaning_raw.yaml" if track == "raw" else "config/data_cleaning_fe.yaml"
        },
        "feature_engineering": {
            "enabled": track != "raw"  # Enabled for feature_engineering track, disabled for raw
        }
    }
    
    # Add spatial aggregation config for Matrix Cell C (raw + spatial aggregation)
    if matrix_cell in ["C", "D"] and radius_km is not None:
        pipeline_config["feature_engineering"]["spatial_aggregation"] = {
            "enabled": True,
            "distance_threshold_km": float(radius_km),
            "weight_method": "distance",  # Default
            "aggregation_methods": ["mean", "max", "min", "std", "median", "weighted_mean", "gradient", "range", "missing_ratio"]
        }
    
    # Add kNN config for Matrix Cell E
    if matrix_cell == "E":
        knn_k = metadata.get("knn_k")
        if knn_k is not None:
            pipeline_config["feature_engineering"]["spatial_aggregation"] = {
                "enabled": True,
                "k_neighbors": int(knn_k),
                "weight_method": "distance",
                "aggregation_methods": ["mean", "max", "min", "std", "median", "weighted_mean", "gradient", "range", "missing_ratio"]
            }
    
    # Save pipeline_config.yaml
    pipeline_config_path = horizon_dir / "pipeline_config.yaml"
    with pipeline_config_path.open("w", encoding="utf-8") as f:
        yaml.dump(pipeline_config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"✅ Generated pipeline_config.yaml for {horizon_dir}")
    return True


def main():
    """Main function."""
    print("=" * 80)
    print("🔧 Generate pipeline_config.yaml for Existing Models")
    print("=" * 80)
    print()
    
    # Find all horizon directories
    base_dir = project_root / "experiments" / "lightgbm" / "raw" / "C"
    
    if not base_dir.exists():
        print(f"⚠️  Base directory not found: {base_dir}")
        return
    
    generated_count = 0
    existing_count = 0
    
    # Process all radius directories
    for radius_dir in sorted(base_dir.glob("radius_*km")):
        if not radius_dir.is_dir():
            continue
        
        horizon_dir = radius_dir / "full_training" / "horizon_12h"
        if not horizon_dir.exists():
            continue
        
        print(f"\nProcessing: {horizon_dir}")
        
        # Check if pipeline_config.yaml already exists
        pipeline_config_path = horizon_dir / "pipeline_config.yaml"
        if pipeline_config_path.exists():
            print(f"  ⏭️  pipeline_config.yaml already exists")
            existing_count += 1
            continue
        
        # Generate config
        if generate_pipeline_config(horizon_dir):
            generated_count += 1
    
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"✅ Generated: {generated_count} pipeline_config.yaml files")
    print(f"⏭️  Skipped (already exists): {existing_count} files")
    print()


if __name__ == "__main__":
    main()

