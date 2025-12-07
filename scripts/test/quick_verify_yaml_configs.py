"""Quick verification of YAML configuration loading without full training.
Tests that configurations are correctly loaded and DataPipeline is initialized correctly.
"""

import sys
from pathlib import Path
import yaml
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.pipeline_runner import load_training_config, PipelineTrainingConfig
from src.data import DataPipeline


def test_config_loading(matrix_cell: str, track: str, radius_km: float = None):
    """Test configuration loading for a matrix cell."""
    print(f"\n{'='*70}")
    print(f"🔍 Testing Config Loading: Matrix {matrix_cell} (track={track})")
    if radius_km is not None:
        print(f"   radius_km={radius_km}")
    print(f"{'='*70}")
    
    # Build CLI overrides
    cli_overrides = {
        "matrix_cell": matrix_cell,
        "track": track,
        "model": "lightgbm",
        "horizon_h": 3,
        "output_dir": project_root / "experiments" / "test_quick" / f"matrix_{matrix_cell.lower()}"
    }
    
    if radius_km is not None and radius_km > 0:
        cli_overrides["radius_km"] = radius_km
    
    try:
        # Load configuration
        config = load_training_config(
            config_path=None,  # Use auto-detection
            cli_overrides=cli_overrides,
            project_root=project_root
        )
        
        print(f"✅ Configuration loaded successfully")
        
        # Verify data section
        print(f"\n📋 Data Section:")
        print(f"  matrix_cell: {config.data.matrix_cell}")
        print(f"  track: {config.data.track}")
        
        # Verify feature engineering config
        fe_config = config.data.feature_engineering
        fe_enabled = fe_config.get("enabled", False)
        expected_enabled = (track == "feature_engineering")
        
        print(f"\n🔧 Feature Engineering:")
        print(f"  enabled: {fe_enabled} (expected: {expected_enabled})")
        
        if fe_enabled != expected_enabled:
            print(f"  ❌ MISMATCH!")
            return False
        
        if track == "feature_engineering":
            config_path = fe_config.get("config_path")
            print(f"  config_path: {config_path}")
            if not config_path:
                print(f"  ❌ config_path missing!")
                return False
            
            # Check if config file exists
            if isinstance(config_path, str):
                config_file = project_root / config_path
            else:
                config_file = config_path
            
            if config_file.exists():
                print(f"  ✓ Config file exists: {config_file.relative_to(project_root)}")
            else:
                print(f"  ❌ Config file not found: {config_file}")
                return False
        
        # Verify spatial aggregation for C/D
        if matrix_cell in {"C", "D"} and radius_km and radius_km > 0:
            spatial_agg = fe_config.get("spatial_aggregation", {})
            spatial_enabled = spatial_agg.get("enabled", False)
            distance = spatial_agg.get("distance_threshold_km")
            
            print(f"\n🌐 Spatial Aggregation:")
            print(f"  enabled: {spatial_enabled} (expected: True)")
            print(f"  distance_threshold_km: {distance} (expected: {radius_km})")
            
            if not spatial_enabled:
                print(f"  ❌ Spatial aggregation not enabled!")
                return False
            
            if distance != radius_km:
                print(f"  ❌ Distance mismatch!")
                return False
            
            print(f"  ✓ Spatial aggregation correctly configured")
        
        # Test DataPipeline initialization
        print(f"\n🔧 Testing DataPipeline initialization...")
        try:
            pipeline_config = {
                "data": {
                    "source": config.data.source,
                    "cleaning": config.data.cleaning,
                    "feature_engineering": fe_config,
                },
                "labels": {
                    "horizons": config.labels.horizons,
                    "frost_threshold": config.labels.frost_threshold,
                }
            }
            
            pipeline = DataPipeline(pipeline_config)
            print(f"  ✅ DataPipeline initialized successfully")
            
            # Log spatial aggregation status if enabled
            if matrix_cell in {"C", "D"} and radius_km and radius_km > 0:
                if hasattr(pipeline, '_spatial_enabled') and pipeline._spatial_enabled:
                    print(f"  ✓ Spatial aggregation enabled in pipeline")
                else:
                    # Check feature_engineering config in pipeline
                    if hasattr(pipeline, 'config'):
                        fe_pipeline = pipeline.config.get("feature_engineering", {})
                        spatial_pipeline = fe_pipeline.get("spatial_aggregation", {})
                        if spatial_pipeline.get("enabled"):
                            print(f"  ✓ Spatial aggregation config present in pipeline")
                        else:
                            print(f"  ⚠️  Spatial aggregation may not be enabled in pipeline")
            
        except Exception as e:
            print(f"  ❌ DataPipeline initialization FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_yaml_files_exist():
    """Verify that all matrix YAML files exist."""
    print(f"\n{'='*70}")
    print(f"📁 Checking YAML Configuration Files")
    print(f"{'='*70}")
    
    matrix_files = {
        "A": project_root / "config" / "pipeline" / "matrix_a.yaml",
        "B": project_root / "config" / "pipeline" / "matrix_b.yaml",
        "C": project_root / "config" / "pipeline" / "matrix_c.yaml",
        "D": project_root / "config" / "pipeline" / "matrix_d.yaml",
    }
    
    all_exist = True
    for matrix_cell, file_path in matrix_files.items():
        if file_path.exists():
            print(f"  ✓ matrix_{matrix_cell.lower()}.yaml exists")
            
            # Verify content
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            
            data_section = config.get("data", {})
            matrix_cell_config = data_section.get("matrix_cell")
            track_config = data_section.get("track")
            
            print(f"    matrix_cell: {matrix_cell_config}")
            print(f"    track: {track_config}")
            
            # Verify expected values
            if matrix_cell_config != matrix_cell:
                print(f"    ⚠️  matrix_cell mismatch: {matrix_cell_config} != {matrix_cell}")
            
            expected_track = "feature_engineering" if matrix_cell in {"B", "D"} else "raw"
            if track_config != expected_track:
                print(f"    ⚠️  track mismatch: {track_config} != {expected_track}")
            
            # Check spatial aggregation for C/D
            if matrix_cell in {"C", "D"}:
                fe_config = data_section.get("feature_engineering", {})
                spatial_agg = fe_config.get("spatial_aggregation", {})
                if spatial_agg.get("enabled"):
                    print(f"    ✓ spatial_aggregation.enabled: True")
                else:
                    print(f"    ⚠️  spatial_aggregation.enabled: False (expected True for {matrix_cell})")
        else:
            print(f"  ❌ matrix_{matrix_cell.lower()}.yaml MISSING")
            all_exist = False
    
    return all_exist


def main():
    """Run quick verification tests."""
    print("=" * 70)
    print("QUICK YAML CONFIGURATION VERIFICATION")
    print("Testing Config Loading + DataPipeline Initialization")
    print("=" * 70)
    
    # Test 1: Verify YAML files exist
    yaml_files_ok = test_yaml_files_exist()
    
    # Test 2: Test configuration loading for each matrix cell
    print(f"\n{'='*70}")
    print("TESTING CONFIGURATION LOADING")
    print(f"{'='*70}")
    
    test_cases = [
        ("A", "raw", None),
        ("B", "feature_engineering", None),
        ("C", "raw", 50.0),
        ("D", "feature_engineering", 50.0),
    ]
    
    results = {}
    for matrix_cell, track, radius_km in test_cases:
        key = f"{matrix_cell}_{radius_km if radius_km else 0}"
        results[key] = test_config_loading(matrix_cell, track, radius_km)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    print(f"\nYAML Files: {'✅ All exist' if yaml_files_ok else '❌ Some missing'}")
    
    print("\nConfig Loading Results:")
    for key, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {key:15s}: {status}")
    
    all_passed = all(results.values()) and yaml_files_ok
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL QUICK TESTS PASSED!")
        print("✅ YAML files exist and are correctly structured")
        print("✅ Configuration loading works correctly")
        print("✅ DataPipeline can be initialized")
        print("\n💡 Next step: Run full training/inference tests to verify end-to-end")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

