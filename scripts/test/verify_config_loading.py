"""Quick script to verify YAML configuration loading without running full training."""

import sys
from pathlib import Path
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.pipeline_runner import load_training_config


def verify_config(matrix_cell: str, track: str, radius_km: float = None):
    """Verify configuration loading for a matrix cell."""
    print(f"\n{'='*60}")
    print(f"Verifying Matrix Cell {matrix_cell} (track={track})")
    if radius_km is not None:
        print(f"  with radius_km={radius_km}")
    print(f"{'='*60}")
    
    # Build CLI overrides
    cli_overrides = {
        "matrix_cell": matrix_cell,
        "track": track,
        "model": "lightgbm",  # Required for config loading
    }
    
    if radius_km is not None and radius_km > 0:
        cli_overrides["feature_engineering"] = {
            "spatial": {"radius_km": radius_km}
        }
    
    try:
        # Load config (this should auto-select matrix_{cell}.yaml)
        config = load_training_config(
            config_path=None,  # Auto-select based on matrix_cell
            project_root=project_root,
            cli_overrides=cli_overrides
        )
        
        print(f"✅ Configuration loaded successfully")
        print(f"  Matrix Cell: {config.data.matrix_cell}")
        print(f"  Track: {config.data.track}")
        print(f"  Cleaning config: {config.data.cleaning.get('config_path', 'N/A')}")
        print(f"  Feature engineering enabled: {config.data.feature_engineering.get('enabled', False)}")
        print(f"  Feature engineering config_path: {config.data.feature_engineering.get('config_path', 'N/A')}")
        
        # Verify track matches
        if config.data.track == track:
            print(f"  ✓ Track matches: {track}")
        else:
            print(f"  ✗ Track mismatch: expected {track}, got {config.data.track}")
            return False
        
        # Verify feature engineering
        expected_fe_enabled = (track == "feature_engineering")
        actual_fe_enabled = config.data.feature_engineering.get("enabled", False)
        if actual_fe_enabled == expected_fe_enabled:
            print(f"  ✓ Feature engineering enabled: {actual_fe_enabled} (correct)")
        else:
            print(f"  ✗ Feature engineering enabled: {actual_fe_enabled} (expected {expected_fe_enabled})")
            return False
        
        # Verify config_path for feature_engineering track
        if track == "feature_engineering":
            config_path = config.data.feature_engineering.get("config_path")
            if config_path:
                print(f"  ✓ Feature engineering config_path: {config_path}")
            else:
                print(f"  ✗ Feature engineering config_path MISSING")
                return False
        
        # Verify spatial aggregation for C/D
        if matrix_cell in {"C", "D"} and radius_km and radius_km > 0:
            spatial = config.data.feature_engineering.get("spatial", {})
            spatial_radius = spatial.get("radius_km")
            if spatial_radius == radius_km:
                print(f"  ✓ Spatial aggregation radius_km: {spatial_radius} (correct)")
            else:
                print(f"  ✗ Spatial aggregation radius_km: {spatial_radius} (expected {radius_km})")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Verify all matrix cell configurations."""
    print("Matrix Cell Configuration Verification")
    print("=" * 60)
    
    results = {}
    
    # Verify Matrix Cell A
    results["A"] = verify_config("A", "raw")
    
    # Verify Matrix Cell B
    results["B"] = verify_config("B", "feature_engineering")
    
    # Verify Matrix Cell C
    results["C"] = verify_config("C", "raw", radius_km=50.0)
    
    # Verify Matrix Cell D
    results["D"] = verify_config("D", "feature_engineering", radius_km=50.0)
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    for cell, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"Matrix Cell {cell}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All configurations verified successfully!")
        return 0
    else:
        print("\n⚠️  Some configurations failed verification")
        return 1


if __name__ == "__main__":
    sys.exit(main())

