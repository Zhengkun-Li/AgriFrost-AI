"""Test script to verify Matrix Cell A/B/C/D YAML configurations.
Runs small experiments in a test directory to avoid polluting real results.
"""

import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent.parent


def test_matrix_config(matrix_cell: str, track: str, radius_km: float = None):
    """Test a matrix cell configuration with a small experiment."""
    print(f"\n{'='*60}")
    print(f"Testing Matrix Cell {matrix_cell} (track={track})")
    if radius_km is not None:
        print(f"  with radius_km={radius_km}")
    print(f"{'='*60}")
    
    # Use test directory
    # Note: The actual output will be {output_dir}/{matrix_cell}/full_training/horizon_3h/
    base_output_dir = project_root / "experiments" / "test_yaml_configs" / f"matrix_{matrix_cell.lower()}"
    if radius_km is not None and radius_km > 0:
        base_output_dir = base_output_dir / f"radius_{radius_km}km"
    test_output_dir = base_output_dir / "full_training" / "horizon_3h"
    
    # Actual files will be in {test_output_dir}/{matrix_cell}/full_training/horizon_3h/
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
    
    # Add radius_km if specified
    if radius_km is not None and radius_km > 0:
        cmd.extend(["--radius-km", str(radius_km)])
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Output dir: {test_output_dir}")
    
    # Determine timeout: longer for spatial aggregation (C/D)
    timeout = 1800 if radius_km and radius_km > 0 else 600  # 30 min for C/D, 10 min for A/B
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"✅ Matrix Cell {matrix_cell} test PASSED")
            
            # Verify output files (actual location)
            required_files = [
                actual_output_dir / "pipeline_config.yaml",
                actual_output_dir / "run_metadata.json",
                actual_output_dir / "frost_classifier" / "config.json",
            ]
            all_exist = True
            for f in required_files:
                if f.exists():
                    print(f"  ✓ {f.name} exists")
                else:
                    print(f"  ✗ {f.name} MISSING")
                    all_exist = False
            
            if all_exist:
                # Check pipeline_config.yaml content
                import yaml
                with open(required_files[0], 'r') as f:
                    pipeline_config = yaml.safe_load(f)
                
                # Verify track
                fe_config = pipeline_config.get("feature_engineering", {})
                fe_enabled = fe_config.get("enabled", False)
                expected_enabled = (track == "feature_engineering")
                
                if fe_enabled == expected_enabled:
                    print(f"  ✓ feature_engineering.enabled = {fe_enabled} (correct)")
                else:
                    print(f"  ✗ feature_engineering.enabled = {fe_enabled} (expected {expected_enabled})")
                
                # Check config_path
                if track == "feature_engineering":
                    config_path = fe_config.get("config_path")
                    if config_path:
                        print(f"  ✓ feature_engineering.config_path = {config_path}")
                    else:
                        print(f"  ✗ feature_engineering.config_path MISSING")
                
                # Check spatial aggregation for C/D
                if matrix_cell in {"C", "D"} and radius_km and radius_km > 0:
                    spatial_agg = fe_config.get("spatial_aggregation", {})
                    distance_threshold = spatial_agg.get("distance_threshold_km")
                    if distance_threshold == radius_km:
                        print(f"  ✓ spatial_aggregation.distance_threshold_km = {distance_threshold} (correct)")
                    else:
                        print(f"  ✗ spatial_aggregation.distance_threshold_km = {distance_threshold} (expected {radius_km})")
            
            return True
        else:
            print(f"❌ Matrix Cell {matrix_cell} test FAILED")
            print(f"Return code: {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  Matrix Cell {matrix_cell} test TIMED OUT")
        return False
    except Exception as e:
        print(f"❌ Matrix Cell {matrix_cell} test ERROR: {e}")
        return False


def main():
    """Run all matrix cell configuration tests."""
    print("Starting Matrix Cell Configuration Tests")
    print("=" * 60)
    
    results = {}
    
    # Test Matrix Cell A: raw features, no spatial aggregation
    results["A"] = test_matrix_config("A", "raw")
    
    # Test Matrix Cell B: feature engineering, no spatial aggregation
    results["B"] = test_matrix_config("B", "feature_engineering")
    
    # Test Matrix Cell C: raw features, with spatial aggregation
    results["C"] = test_matrix_config("C", "raw", radius_km=50.0)
    
    # Test Matrix Cell D: feature engineering, with spatial aggregation
    results["D"] = test_matrix_config("D", "feature_engineering", radius_km=50.0)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for cell, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"Matrix Cell {cell}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print("\n⚠️  Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

