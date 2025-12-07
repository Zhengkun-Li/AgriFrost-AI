#!/usr/bin/env python3
"""Create missing pipeline_config.yaml files for existing models.

This script generates pipeline_config.yaml files for models that were
trained before the config saving logic was implemented.
"""

import sys
from pathlib import Path
import yaml
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.pipeline_runner import _build_pipeline_config, DataSection, LabelSection

def create_pipeline_config(horizon_dir: Path, radius_km: float = None):
    """Create pipeline_config.yaml for a horizon directory.
    
    Args:
        horizon_dir: Directory containing the model (e.g., experiments/lightgbm/raw/C/radius_30km/full_training/horizon_3h)
        radius_km: Radius in km for spatial aggregation (None if not applicable)
    """
    config_path = horizon_dir / "pipeline_config.yaml"
    
    if config_path.exists():
        print(f"  ⏭️  {config_path.name} already exists, skipping...")
        return
    
    # Load run_metadata.json to get configuration info
    metadata_path = horizon_dir / "run_metadata.json"
    if not metadata_path.exists():
        print(f"  ⚠️  {metadata_path.name} not found, cannot infer config")
        return
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    matrix_cell = metadata.get("matrix_cell", "C")
    track = metadata.get("track", "raw")
    
    # Build feature_engineering config
    fe_config = {"enabled": False}  # Raw track has feature_engineering disabled
    
    # Add spatial aggregation if radius_km is provided (for Matrix Cell C)
    if radius_km is not None and matrix_cell in ["C", "D"]:
        fe_config["spatial"] = {"radius_km": float(radius_km)}
        # Map to spatial_aggregation (as done in pipeline_runner.py)
        fe_config["spatial_aggregation"] = {
            "enabled": True,
            "distance_threshold_km": float(radius_km)
        }
        fe_config["_spatial_mapped_to_spatial_aggregation"] = True
    
    # Build pipeline config
    pipeline_config = {
        "cleaning": {},
        "feature_engineering": fe_config,
        "labels": {"threshold": -2.0}  # Default threshold
    }
    
    # Save config
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(pipeline_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"  ✅ Created {config_path.name}")
    print(f"     Spatial aggregation: {fe_config.get('spatial_aggregation', {}).get('enabled', False)}")
    if radius_km is not None:
        print(f"     Distance threshold: {radius_km}km")


def main():
    """Create pipeline_config.yaml for all existing models."""
    base_dir = project_root / "experiments" / "lightgbm" / "raw" / "C"
    
    print("=" * 90)
    print("🔧 创建缺失的 pipeline_config.yaml 文件")
    print("=" * 90)
    print()
    
    # Check all radius directories
    for radius_dir in sorted(base_dir.glob("radius_*km")):
        radius_str = radius_dir.name.replace("radius_", "").replace("km", "")
        try:
            radius_km = float(radius_str)
        except ValueError:
            print(f"⚠️  无法解析半径: {radius_dir.name}")
            continue
        
        print(f"\n检查 {radius_dir.name}:")
        
        full_training_dir = radius_dir / "full_training"
        if not full_training_dir.exists():
            print(f"  ⚠️  full_training 目录不存在")
            continue
        
        # Check all horizon directories
        for horizon_dir in sorted(full_training_dir.glob("horizon_*h")):
            print(f"  Horizon {horizon_dir.name}:")
            create_pipeline_config(horizon_dir, radius_km)
    
    print()
    print("=" * 90)
    print("✅ 完成")
    print("=" * 90)


if __name__ == "__main__":
    main()


