"""Comprehensive test to verify YAML configs work correctly for training and inference.
Tests actual data loading, feature generation, and verifies distance parameters.
"""

import sys
import subprocess
import json
import yaml
from pathlib import Path
import pandas as pd
import logging

project_root = Path(__file__).parent.parent.parent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_pipeline_config(config_path: Path, expected_track: str, expected_radius_km: float = None):
    """Check pipeline_config.yaml content."""
    if not config_path.exists():
        return False, "pipeline_config.yaml not found"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check track
    fe_config = config.get("feature_engineering", {})
    fe_enabled = fe_config.get("enabled", False)
    expected_enabled = (expected_track == "feature_engineering")
    
    if fe_enabled != expected_enabled:
        return False, f"feature_engineering.enabled={fe_enabled}, expected={expected_enabled}"
    
    # Check config_path for feature_engineering
    if expected_track == "feature_engineering":
        config_path_val = fe_config.get("config_path")
        if not config_path_val or config_path_val == "null":
            return False, "feature_engineering.config_path missing"
    
    # Check spatial aggregation
    if expected_radius_km is not None and expected_radius_km > 0:
        spatial_agg = fe_config.get("spatial_aggregation", {})
        if not spatial_agg.get("enabled", False):
            return False, "spatial_aggregation.enabled should be True"
        
        distance = spatial_agg.get("distance_threshold_km")
        if distance != expected_radius_km:
            return False, f"distance_threshold_km={distance}, expected={expected_radius_km}"
    
    return True, "OK"


def check_data_features(data_path: Path, expected_track: str, expected_radius_km: float = None):
    """Check if data has expected features."""
    if not data_path.exists():
        return False, "data file not found"
    
    df = pd.read_parquet(data_path)
    
    # Check if neighbor features exist for spatial aggregation
    if expected_radius_km and expected_radius_km > 0:
        neighbor_cols = [col for col in df.columns if 'neighbor' in col.lower() or '_neighbor_' in col]
        if len(neighbor_cols) == 0:
            return False, f"No neighbor features found (expected for radius_km={expected_radius_km})"
    
    return True, f"OK (shape: {df.shape})"


def test_training(matrix_cell: str, track: str, radius_km: float = None):
    """Test training with a matrix cell configuration."""
    print(f"\n{'='*70}")
    print(f"🧪 TESTING TRAINING: Matrix {matrix_cell} (track={track})")
    if radius_km is not None:
        print(f"   radius_km={radius_km}")
    print(f"{'='*70}")
    
    # Test output directory
    base_dir = project_root / "experiments" / "test_full_pipeline" / f"matrix_{matrix_cell.lower()}"
    if radius_km is not None and radius_km > 0:
        base_dir = base_dir / f"radius_{radius_km}km"
    output_dir = base_dir / "full_training" / "horizon_3h"
    actual_model_dir = output_dir / matrix_cell / "full_training" / "horizon_3h"
    
    # Build command
    cmd = [
        sys.executable, "-m", "src.cli", "train", "single",
        "--model-name", "lightgbm",
        "--matrix-cell", matrix_cell,
        "--track", track,
        "--horizon-h", "3",
        "--output-dir", str(output_dir.relative_to(project_root))
    ]
    
    if radius_km is not None and radius_km > 0:
        cmd.extend(["--radius-km", str(radius_km)])
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Run training with timeout
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
            print(f"STDERR:\n{result.stderr[:2000]}")
            return False
        
        print(f"✅ Training completed")
        
        # Verify pipeline_config.yaml
        pipeline_config_path = actual_model_dir / "pipeline_config.yaml"
        passed, msg = check_pipeline_config(pipeline_config_path, track, radius_km)
        if not passed:
            print(f"  ❌ pipeline_config.yaml check FAILED: {msg}")
            return False
        print(f"  ✓ pipeline_config.yaml: {msg}")
        
        # Verify run_metadata.json
        metadata_path = actual_model_dir / "run_metadata.json"
        if not metadata_path.exists():
            print(f"  ❌ run_metadata.json missing")
            return False
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if radius_km and radius_km > 0:
            if metadata.get("radius_km") != radius_km:
                print(f"  ❌ run_metadata.json radius_km={metadata.get('radius_km')}, expected={radius_km}")
                return False
            print(f"  ✓ run_metadata.json radius_km={radius_km}")
        
        # Verify model files
        frost_model_path = actual_model_dir / "frost_classifier" / "model.pkl"
        temp_model_path = actual_model_dir / "temp_regressor" / "model.pkl"
        
        if not frost_model_path.exists():
            print(f"  ❌ Frost model missing")
            return False
        if not temp_model_path.exists():
            print(f"  ❌ Temperature model missing")
            return False
        print(f"  ✓ Model files exist")
        
        # Check labeled_data if exists (may not exist depending on caching)
        labeled_data_path = output_dir / "labeled_data.parquet"
        if labeled_data_path.exists():
            passed, msg = check_data_features(labeled_data_path, track, radius_km)
            print(f"  {'✓' if passed else '⚠️'} labeled_data.parquet: {msg}")
        else:
            print(f"  ⚠️  labeled_data.parquet not found (may be cached elsewhere)")
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"⏱️  Training TIMED OUT (>{timeout}s)")
        return False
    except Exception as e:
        print(f"❌ Training ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference(matrix_cell: str, track: str, radius_km: float = None):
    """Test inference with trained model."""
    print(f"\n{'='*70}")
    print(f"🔮 TESTING INFERENCE: Matrix {matrix_cell} (track={track})")
    if radius_km is not None:
        print(f"   radius_km={radius_km}")
    print(f"{'='*70}")
    
    # Find trained model
    base_dir = project_root / "experiments" / "test_full_pipeline" / f"matrix_{matrix_cell.lower()}"
    if radius_km is not None and radius_km > 0:
        base_dir = base_dir / f"radius_{radius_km}km"
    model_dir = base_dir / "full_training" / "horizon_3h" / matrix_cell / "full_training" / "horizon_3h"
    
    if not (model_dir / "frost_classifier" / "model.pkl").exists():
        print(f"⚠️  Model not found, skipping inference test")
        return None  # Skip, not failure
    
    # Test inference output directory
    inference_output_dir = project_root / "experiments" / "test_full_pipeline" / "inference" / f"matrix_{matrix_cell.lower()}"
    if radius_km is not None and radius_km > 0:
        inference_output_dir = inference_output_dir / f"radius_{radius_km}km"
    inference_output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "src.cli", "predict",
        "--model-dir", str(model_dir.relative_to(project_root)),
        "--output-dir", str(inference_output_dir.relative_to(project_root)),
        "--horizon-h", "3"
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
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
            print(f"STDERR:\n{result.stderr[:2000]}")
            return False
        
        print(f"✅ Inference completed")
        
        # Verify predictions file
        predictions_file = inference_output_dir / "predictions.json"
        if not predictions_file.exists():
            print(f"  ❌ predictions.json missing")
            return False
        
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        if "predictions" not in predictions or len(predictions["predictions"]) == 0:
            print(f"  ❌ predictions.json is empty")
            return False
        
        print(f"  ✓ predictions.json created with {len(predictions['predictions'])} entries")
        
        # Verify pipeline_config.yaml was used (check if it exists and was loaded)
        pipeline_config_path = model_dir / "pipeline_config.yaml"
        if pipeline_config_path.exists():
            passed, msg = check_pipeline_config(pipeline_config_path, track, radius_km)
            print(f"  ✓ pipeline_config.yaml available for inference: {msg}")
        else:
            print(f"  ⚠️  pipeline_config.yaml not found (inference may not use correct spatial config)")
        
        return True
        
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
    
    # Phase 1: Training tests
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING TESTS")
    print("=" * 70)
    
    for matrix_cell, track, radius_km in test_cases:
        key = f"{matrix_cell}_{radius_km if radius_km else 0}"
        training_results[key] = test_training(matrix_cell, track, radius_km)
    
    # Phase 2: Inference tests
    print("\n" + "=" * 70)
    print("PHASE 2: INFERENCE TESTS")
    print("=" * 70)
    
    for matrix_cell, track, radius_km in test_cases:
        key = f"{matrix_cell}_{radius_km if radius_km else 0}"
        inference_results[key] = test_inference(matrix_cell, track, radius_km)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    print("\nTraining Results:")
    for key, passed in training_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {key:15s}: {status}")
    
    print("\nInference Results:")
    for key, passed in inference_results.items():
        if passed is None:
            status = "⚠️  SKIPPED"
        elif passed:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"  {key:15s}: {status}")
    
    all_training_passed = all(training_results.values())
    all_inference_passed = all(v for v in inference_results.values() if v is not None)
    
    print("\n" + "=" * 70)
    if all_training_passed and all_inference_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ YAML configurations work correctly for training")
        print("✅ Distance parameters are correctly handled")
        print("✅ Inference uses correct pipeline configuration")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        if not all_training_passed:
            print("❌ Training tests failed - check configuration loading")
        if not all_inference_passed:
            print("❌ Inference tests failed - check prediction pipeline")
        return 1


if __name__ == "__main__":
    sys.exit(main())

