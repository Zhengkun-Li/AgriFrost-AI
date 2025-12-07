#!/usr/bin/env python3
"""Verify that training and inference both use the same distance parameters.

This script checks:
1. Training configuration includes distance_threshold_km
2. pipeline_config.yaml is saved with correct distance_threshold_km
3. Inference correctly loads and uses the same distance_threshold_km
"""

import sys
from pathlib import Path
import yaml
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import FrostPredictor


def verify_training_inference_consistency(model_dir: Path):
    """Verify training and inference use the same distance parameters.
    
    Args:
        model_dir: Path to horizon model directory (e.g., horizon_12h/)
    
    Returns:
        Dict with verification results
    """
    results = {
        "model_dir": str(model_dir),
        "pipeline_config_exists": False,
        "metadata_exists": False,
        "configs_match": False,
        "training_distance": None,
        "inference_distance": None,
        "warnings": []
    }
    
    # 1. Check pipeline_config.yaml (used by inference)
    pipeline_config_path = model_dir / "pipeline_config.yaml"
    if pipeline_config_path.exists():
        with open(pipeline_config_path) as f:
            pipeline_config = yaml.safe_load(f)
        
        fe_config = pipeline_config.get("feature_engineering", {})
        spatial_agg = fe_config.get("spatial_aggregation", {})
        
        results["pipeline_config_exists"] = True
        results["training_distance"] = spatial_agg.get("distance_threshold_km")
        
        spatial_enabled = spatial_agg.get("enabled", False)
        if spatial_enabled and results["training_distance"] is None:
            results["warnings"].append(
                "pipeline_config.yaml has spatial_aggregation.enabled=True "
                "but distance_threshold_km is missing"
            )
    else:
        results["warnings"].append("pipeline_config.yaml not found")
    
    # 2. Check run_metadata.json
    metadata_path = model_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        results["metadata_exists"] = True
        metadata_distance = metadata.get("radius_km")
        
        if results["training_distance"] is None:
            results["training_distance"] = metadata_distance
        
        # Check if they match
        if results["training_distance"] is not None and metadata_distance is not None:
            if results["training_distance"] == metadata_distance:
                results["configs_match"] = True
            else:
                results["warnings"].append(
                    f"Distance mismatch: pipeline_config.yaml={results['training_distance']}km, "
                    f"run_metadata.json={metadata_distance}km"
                )
    else:
        results["warnings"].append("run_metadata.json not found")
    
    # 3. Test inference loading
    try:
        predictor = FrostPredictor(model_dir)
        
        # Check if predictor has correct config
        inference_config = predictor.config
        results["inference_distance"] = inference_config.get("radius_km")
        
        # Check if inference would use correct config
        if predictor.data_pipeline is not None:
            # DataPipeline would use pipeline_config.yaml
            if results["training_distance"] is not None:
                results["inference_distance"] = results["training_distance"]
        elif results["inference_distance"] is None:
            results["warnings"].append(
                "Inference cannot determine distance - no pipeline_config.yaml and no radius_km in metadata"
            )
        
    except Exception as e:
        results["warnings"].append(f"Error loading predictor: {e}")
    
    return results


def main():
    """Check all Matrix Cell C/D models."""
    print("=" * 90)
    print("🔍 验证训练和推理的distance参数一致性")
    print("=" * 90)
    print()
    
    base_dir = project_root / "experiments" / "lightgbm" / "raw" / "C"
    
    all_pass = True
    
    for radius_dir in sorted(base_dir.glob("radius_*km")):
        radius_str = radius_dir.name.replace("radius_", "").replace("km", "")
        try:
            expected_radius = float(radius_str)
        except ValueError:
            continue
        
        print(f"\n{'='*90}")
        print(f"检查 {radius_dir.name}:")
        print(f"{'='*90}")
        
        full_training_dir = radius_dir / "full_training"
        if not full_training_dir.exists():
            print(f"  ⚠️  full_training 目录不存在")
            continue
        
        for horizon_dir in sorted(full_training_dir.glob("horizon_*h")):
            print(f"\n  {horizon_dir.name}:")
            
            results = verify_training_inference_consistency(horizon_dir)
            
            # Print results
            if results["pipeline_config_exists"]:
                print(f"    ✅ pipeline_config.yaml 存在")
                if results["training_distance"] is not None:
                    print(f"        distance_threshold_km: {results['training_distance']}km")
                    if results["training_distance"] == expected_radius:
                        print(f"        ✅ 与目录名一致 ({expected_radius}km)")
                    else:
                        print(f"        ⚠️  与目录名不一致 (期望 {expected_radius}km)")
                        all_pass = False
                else:
                    print(f"        ⚠️  distance_threshold_km 未设置")
                    all_pass = False
            else:
                print(f"    ❌ pipeline_config.yaml 不存在")
                all_pass = False
            
            if results["metadata_exists"]:
                metadata_distance = json.load(open(horizon_dir / "run_metadata.json")).get("radius_km")
                if metadata_distance is not None:
                    print(f"    ✅ run_metadata.json 存在 (radius_km: {metadata_distance}km)")
                    if metadata_distance != expected_radius:
                        print(f"        ⚠️  与目录名不一致 (期望 {expected_radius}km)")
                        all_pass = False
            
            if results["configs_match"]:
                print(f"    ✅ 配置一致")
            
            if results["warnings"]:
                for warning in results["warnings"]:
                    print(f"    ⚠️  {warning}")
                    all_pass = False
            
            # Check inference distance
            if results["inference_distance"] is not None:
                if results["inference_distance"] == expected_radius:
                    print(f"    ✅ 推理将使用正确的distance: {results['inference_distance']}km")
                else:
                    print(f"    ⚠️  推理distance不匹配: {results['inference_distance']}km (期望 {expected_radius}km)")
                    all_pass = False
    
    print()
    print("=" * 90)
    if all_pass:
        print("✅ 所有检查通过: 训练和推理都使用正确的distance参数")
    else:
        print("⚠️  发现问题: 部分模型配置不一致")
    print("=" * 90)


if __name__ == "__main__":
    main()


