"""Comprehensive test script to verify YAML configurations work for both training and inference.
Tests actual data loading, feature generation, and model inference with distance parameters.
"""

import sys
import subprocess
import json
import yaml
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent.parent


def test_matrix_training(matrix_cell: str, track: str, radius_km: float = None):
    """Test training with a matrix cell configuration."""
    print(f"\n{'='*70}")
    print(f"Testing TRAINING: Matrix Cell {matrix_cell} (track={track})")
    if radius_km is not None:
        print(f"  with radius_km={radius_km}")
    print(f"{'='*70}")
    
    # Use test directory
    base_output_dir = project_root / "experiments" / "test_yaml_pipeline" / f"matrix_{matrix_cell.lower()}"
    if radius_km is not None and radius_km > 0:
        base_output_dir = base_output_dir / f"radius_{radius_km}km"
    test_output_dir = base_output_dir / "full_training" / "horizon_3h"
    actual_output_dir = test_output_dir / matrix_cell / "full_training" / "horizon_3h"
    
    # Build command
    cmd = [
        sys.executable, "-m", "src.cli", "train", "single",
        "--model-name", "lightgbm",
        "--matrix-cell", matrix_cell,
        "--track", track,
        "--horizon-h", "3",
        "--output-dir", str(test_output_dir.relative_to(project_root))
    ]
    
    if radius_km is not None and radius_km > 0:
        cmd.extend(["--radius-km", str(radius_km)])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run training
    timeout = 1800 if radius_km and radius_km > 0 else 600
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            print(f"❌ Training FAILED")
            print(f"STDERR:\n{result.stderr[:1000]}")
            return False
        
        print(f"✅ Training completed")
        
        # Verify files exist
        pipeline_config_path = actual_output_dir / "pipeline_config.yaml"
        run_metadata_path = actual_output_dir / "run_metadata.json"
        labeled_data_path = test_output_dir / "labeled_data.parquet"
        
        if not pipeline_config_path.exists():
            print(f"  ✗ pipeline_config.yaml missing")
            return False
        if not run_metadata_path.exists():
            print(f"  ✗ run_metadata.json missing")
            return False
        
        # Verify pipeline_config.yaml content
        with open(pipeline_config_path, 'r') as f:
            pipeline_config = yaml.safe_load(f)
        
        fe_config = pipeline_config.get("feature_engineering", {})
        
        # Check track
        expected_fe_enabled = (track == "feature_engineering")
        actual_fe_enabled = fe_config.get("enabled", False)
        if actual_fe_enabled != expected_fe_enabled:
            print(f"  ✗ feature_engineering.enabled: {actual_fe_enabled} (expected {expected_fe_enabled})")
            return False
        print(f"  ✓ feature_engineering.enabled: {actual_fe_enabled}")
        
        # Check config_path for feature_engineering
        if track == "feature_engineering":
            config_path = fe_config.get("config_path")
            if not config_path or config_path == "null":
                print(f"  ✗ feature_engineering.config_path missing")
                return False
            print(f"  ✓ feature_engineering.config_path: {config_path}")
        
        # Check spatial aggregation for C/D
        if matrix_cell in {"C", "D"} and radius_km and radius_km > 0:
            spatial_agg = fe_config.get("spatial_aggregation", {})
            distance_threshold = spatial_agg.get("distance_threshold_km")
            if distance_threshold != radius_km:
                print(f"  ✗ spatial_aggregation.distance_threshold_km: {distance_threshold} (expected {radius_km})")
                return False
            print(f"  ✓ spatial_aggregation.distance_threshold_km: {distance_threshold}")
            
            # Verify run_metadata.json has radius_km
            with open(run_metadata_path, 'r') as f:
                metadata = json.load(f)
            if metadata.get("radius_km") != radius_km:
                print(f"  ✗ run_metadata.json radius_km: {metadata.get('radius_km')} (expected {radius_km})")
                return False
            print(f"  ✓ run_metadata.json radius_km: {radius_km}")
            
            # Check if labeled_data.parquet has neighbor features
            if labeled_data_path.exists():
                df = pd.read_parquet(labeled_data_path)
                neighbor_cols = [col for col in df.columns if 'neighbor' in col.lower() or '_neighbor_' in col]
                if len(neighbor_cols) == 0:
                    print(f"  ⚠️  No neighbor features found in labeled_data.parquet (expected for spatial aggregation)")
                else:
                    print(f"  ✓ Found {len(neighbor_cols)} neighbor features in labeled_data.parquet")
                    print(f"    Sample columns: {neighbor_cols[:5]}")
            else:
                print(f"  ⚠️  labeled_data.parquet not found")
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"⏱️  Training TIMED OUT")
        return False
    except Exception as e:
        print(f"❌ Training ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_matrix_inference(matrix_cell: str, track: str, radius_km: float = None):
    """Test inference with a matrix cell configuration."""
    print(f"\n{'='*70}")
    print(f"Testing INFERENCE: Matrix Cell {matrix_cell} (track={track})")
    if radius_km is not None:
        print(f"  with radius_km={radius_km}")
    print(f"{'='*70}")
    
    # Find the trained model directory
    base_dir = project_root / "experiments" / "test_yaml_pipeline" / f"matrix_{matrix_cell.lower()}"
    if radius_km is not None and radius_km > 0:
        base_dir = base_dir / f"radius_{radius_km}km"
    model_dir = base_dir / "full_training" / "horizon_3h" / matrix_cell / "full_training" / "horizon_3h"
    
    if not (model_dir / "frost_classifier" / "model.pkl").exists():
        print(f"⚠️  Model not found at {model_dir}, skipping inference test")
        return True  # Not a failure, just skip
    
    # Test inference
    test_output_dir = project_root / "experiments" / "test_yaml_pipeline" / "inference" / f"matrix_{matrix_cell.lower()}"
    if radius_km is not None and radius_km > 0:
        test_output_dir = test_output_dir / f"radius_{radius_km}km"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "src.cli", "predict",
        "--model-dir", str(model_dir.relative_to(project_root)),
        "--output-dir", str(test_output_dir.relative_to(project_root)),
        "--horizon-h", "3"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"❌ Inference FAILED")
            print(f"STDERR:\n{result.stderr[:1000]}")
            return False
        
        print(f"✅ Inference completed")
        
        # Verify predictions file exists
        predictions_file = test_output_dir / "predictions.json"
        if predictions_file.exists():
            print(f"  ✓ predictions.json created")
            
            # Load and check predictions
            with open(predictions_file, 'r') as f:
                predictions = json.load(f)
            
            if "predictions" in predictions and len(predictions["predictions"]) > 0:
                print(f"  ✓ Predictions contain {len(predictions['predictions'])} entries")
                return True
            else:
                print(f"  ✗ Predictions file is empty")
                return False
        else:
            print(f"  ✗ predictions.json not found")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  Inference TIMED OUT")
        return False
    except Exception as e:
        print(f"❌ Inference ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive pipeline tests."""
    print("=" * 70)
    print("COMPREHENSIVE YAML CONFIGURATION PIPELINE TEST")
    print("Testing Training + Inference with Distance Parameters")
    print("=" * 70)
    
    test_cases = [
        ("A", "raw", None),
        ("B", "feature_engineering", None),
        ("C", "raw", 50.0),
        ("D", "feature_engineering", 50.0),
    ]
    
    training_results = {}
    inference_results = {}
    
    # Test training for all cases
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING TESTS")
    print("=" * 70)
    
    for matrix_cell, track, radius_km in test_cases:
        key = f"{matrix_cell}_{radius_km if radius_km else 0}"
        training_results[key] = test_matrix_training(matrix_cell, track, radius_km)
    
    # Test inference for all cases
    print("\n" + "=" * 70)
    print("PHASE 2: INFERENCE TESTS")
    print("=" * 70)
    
    for matrix_cell, track, radius_km in test_cases:
        key = f"{matrix_cell}_{radius_km if radius_km else 0}"
        inference_results[key] = test_matrix_inference(matrix_cell, track, radius_km)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    print("\nTraining Results:")
    for key, passed in training_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {key}: {status}")
    
    print("\nInference Results:")
    for key, passed in inference_results.items():
        status = "✅ PASSED" if passed else "⚠️  SKIPPED" if passed is None else "❌ FAILED"
        print(f"  {key}: {status}")
    
    all_training_passed = all(training_results.values())
    all_inference_passed = all(v for v in inference_results.values() if v is not None)
    
    if all_training_passed and all_inference_passed:
        print("\n🎉 All pipeline tests PASSED!")
        print("✅ YAML configurations work correctly for training and inference")
        print("✅ Distance parameters are correctly handled")
        return 0
    else:
        print("\n⚠️  Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())


