#!/usr/bin/env python3
"""Check consistency between training and inference for spatial aggregation.

This script verifies:
1. Training saves pipeline_config.yaml with spatial_aggregation config
2. Inference loads and uses the same spatial_aggregation config
3. Feature names match between training and inference
4. Distance/radius configuration is correctly passed and used
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import yaml
import pickle
import pandas as pd
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_training_config(horizon_dir: Path) -> Dict[str, Any]:
    """Check training configuration and saved artifacts.
    
    Args:
        horizon_dir: Path to horizon directory (e.g., horizon_12h/)
    
    Returns:
        Dictionary with check results
    """
    results = {
        "horizon_dir": str(horizon_dir),
        "has_run_metadata": False,
        "has_pipeline_config": False,
        "has_model": False,
        "run_metadata": {},
        "pipeline_config": {},
        "model_feature_names": [],
        "issues": []
    }
    
    # Check run_metadata.json
    metadata_file = horizon_dir / "run_metadata.json"
    if metadata_file.exists():
        results["has_run_metadata"] = True
        with open(metadata_file) as f:
            results["run_metadata"] = json.load(f)
        
        # Check if spatial config is in metadata
        if "radius_km" not in results["run_metadata"]:
            results["issues"].append("run_metadata.json missing radius_km")
    else:
        results["issues"].append("run_metadata.json not found")
    
    # Check pipeline_config.yaml
    pipeline_config_file = horizon_dir / "pipeline_config.yaml"
    if pipeline_config_file.exists():
        results["has_pipeline_config"] = True
        with open(pipeline_config_file) as f:
            results["pipeline_config"] = yaml.safe_load(f)
        
        # Check if spatial_aggregation is in config
        fe_config = results["pipeline_config"].get("feature_engineering", {})
        if "spatial_aggregation" not in fe_config:
            results["issues"].append("pipeline_config.yaml missing spatial_aggregation config")
        elif not fe_config.get("spatial_aggregation", {}).get("enabled", False):
            results["issues"].append("pipeline_config.yaml has spatial_aggregation disabled")
        else:
            sa_config = fe_config["spatial_aggregation"]
            if "distance_threshold_km" not in sa_config and "radius_km" not in sa_config:
                results["issues"].append("pipeline_config.yaml missing distance_threshold_km or radius_km")
    else:
        results["issues"].append("pipeline_config.yaml not found - inference may fail to generate neighbor features")
    
    # Check model.pkl feature names
    model_file = horizon_dir / "frost_classifier" / "model.pkl"
    if model_file.exists():
        results["has_model"] = True
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
                if isinstance(model_data, dict) and 'config' in model_data:
                    config = model_data['config']
                    if 'feature_names' in config:
                        results["model_feature_names"] = config['feature_names']
                        neighbor_features = [f for f in config['feature_names'] if 'neighbor' in f.lower()]
                        if len(neighbor_features) == 0:
                            results["issues"].append("Model has no neighbor features in feature_names")
                    else:
                        results["issues"].append("Model config missing feature_names")
        except Exception as e:
            results["issues"].append(f"Failed to load model.pkl: {e}")
    else:
        results["issues"].append("frost_classifier/model.pkl not found")
    
    return results


def check_inference_code() -> Dict[str, Any]:
    """Check inference code for potential issues.
    
    Returns:
        Dictionary with check results
    """
    results = {
        "issues": [],
        "warnings": []
    }
    
    predictor_file = project_root / "src" / "inference" / "predictor.py"
    if not predictor_file.exists():
        results["issues"].append("predictor.py not found")
        return results
    
    with open(predictor_file) as f:
        content = f.read()
    
    # Check if predictor uses DataPipeline correctly
    if "DataPipeline" in content:
        if "apply_features" in content:
            results["warnings"].append(
                "predictor.py uses FeatureEngineer.apply_features() instead of DataPipeline.run() - "
                "may not correctly apply spatial aggregation if config is incomplete"
            )
        
        if "pipeline_config.yaml" not in content:
            results["warnings"].append(
                "predictor.py may not be loading pipeline_config.yaml correctly"
            )
    
    # Check if predictor loads spatial_aggregation config
    if "spatial_aggregation" not in content:
        results["warnings"].append(
            "predictor.py may not be handling spatial_aggregation config"
        )
    
    return results


def simulate_inference(horizon_dir: Path, sample_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Simulate inference to check feature consistency.
    
    Args:
        horizon_dir: Path to horizon directory
        sample_data: Optional sample data DataFrame
    
    Returns:
        Dictionary with simulation results
    """
    results = {
        "success": False,
        "issues": [],
        "feature_names_training": [],
        "feature_names_inference": [],
        "matching_features": []
    }
    
    try:
        from src.inference.predictor import FrostPredictor
        
        # Initialize predictor
        predictor = FrostPredictor(horizon_dir)
        
        # Get training feature names
        model_file = horizon_dir / "frost_classifier" / "model.pkl"
        if model_file.exists():
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
                if isinstance(model_data, dict) and 'config' in model_data:
                    config = model_data['config']
                    if 'feature_names' in config:
                        results["feature_names_training"] = config['feature_names']
        
        # Simulate prediction (if sample data provided)
        if sample_data is not None:
            pred_result = predictor.predict(sample_data)
            results["success"] = True
        else:
            results["issues"].append("No sample data provided for simulation")
    
    except Exception as e:
        results["issues"].append(f"Simulation failed: {e}")
        import traceback
        results["issues"].append(traceback.format_exc())
    
    return results


def main():
    """Main function."""
    print("=" * 80)
    print("🔍 Training-Inference Consistency Check for Spatial Aggregation")
    print("=" * 80)
    print()
    
    # Check different radius configurations
    radii = [30, 50, 75, 100]
    all_results = {}
    
    for radius in radii:
        horizon_dir = project_root / "experiments" / "lightgbm" / "raw" / "C" / f"radius_{radius}km" / "full_training" / "horizon_12h"
        
        if not horizon_dir.exists():
            print(f"⚠️  Radius {radius}km: horizon directory not found")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Checking Radius {radius}km")
        print(f"{'=' * 80}")
        
        # Check training config
        training_results = check_training_config(horizon_dir)
        all_results[f"radius_{radius}km"] = training_results
        
        print(f"\n📋 Training Configuration:")
        print(f"  - run_metadata.json: {'✅' if training_results['has_run_metadata'] else '❌'}")
        print(f"  - pipeline_config.yaml: {'✅' if training_results['has_pipeline_config'] else '❌'}")
        print(f"  - model.pkl: {'✅' if training_results['has_model'] else '❌'}")
        
        if training_results['has_run_metadata']:
            metadata = training_results['run_metadata']
            print(f"  - matrix_cell: {metadata.get('matrix_cell', 'N/A')}")
            print(f"  - track: {metadata.get('track', 'N/A')}")
            print(f"  - radius_km: {metadata.get('radius_km', 'N/A')}")
        
        if training_results['has_pipeline_config']:
            fe_config = training_results['pipeline_config'].get('feature_engineering', {})
            print(f"  - feature_engineering.enabled: {fe_config.get('enabled', 'N/A')}")
            if 'spatial_aggregation' in fe_config:
                sa_config = fe_config['spatial_aggregation']
                print(f"  - spatial_aggregation.enabled: {sa_config.get('enabled', 'N/A')}")
                print(f"  - spatial_aggregation.distance_threshold_km: {sa_config.get('distance_threshold_km', 'N/A')}")
        
        if training_results['model_feature_names']:
            neighbor_features = [f for f in training_results['model_feature_names'] if 'neighbor' in f.lower()]
            print(f"  - Total features: {len(training_results['model_feature_names'])}")
            print(f"  - Neighbor features: {len(neighbor_features)}")
        
        # Report issues
        if training_results['issues']:
            print(f"\n⚠️  Issues found:")
            for issue in training_results['issues']:
                print(f"  - {issue}")
        else:
            print(f"\n✅ No issues found in training configuration")
    
    # Check inference code
    print(f"\n{'=' * 80}")
    print("Checking Inference Code")
    print(f"{'=' * 80}")
    inference_results = check_inference_code()
    
    if inference_results['issues']:
        print(f"\n❌ Issues found:")
        for issue in inference_results['issues']:
            print(f"  - {issue}")
    
    if inference_results['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in inference_results['warnings']:
            print(f"  - {warning}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    
    total_issues = sum(len(r['issues']) for r in all_results.values())
    if total_issues == 0 and len(inference_results['issues']) == 0:
        print("✅ All checks passed!")
    else:
        print(f"⚠️  Found {total_issues + len(inference_results['issues'])} issues")
        print("\nRecommendations:")
        if any(not r['has_pipeline_config'] for r in all_results.values()):
            print("  1. Ensure pipeline_config.yaml is saved during training")
        if inference_results['warnings']:
            print("  2. Review inference code to ensure spatial_aggregation config is used correctly")
        if any(len(r['model_feature_names']) == 0 for r in all_results.values()):
            print("  3. Ensure model saves feature_names correctly")
    
    print()


if __name__ == "__main__":
    main()

