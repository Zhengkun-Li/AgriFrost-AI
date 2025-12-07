#!/usr/bin/env python3
"""Generate Matrix A feature importance supplementary table."""

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "lightgbm" / "raw" / "A" / "full_training"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "manuscript" / "Supplementary"

HORIZONS = [3, 6, 12, 24]


def generate_matrix_a_supplementary_table(output_path: Path) -> None:
    """Generate Matrix A feature importance supplementary table."""
    all_data = []
    
    for horizon in HORIZONS:
        # Load frost classification importance
        frost_file = EXPERIMENT_DIR / f"horizon_{horizon}h" / "frost_feature_importance.csv"
        temp_file = EXPERIMENT_DIR / f"horizon_{horizon}h" / "temp_feature_importance.csv"
        
        if not frost_file.exists() or not temp_file.exists():
            print(f"Warning: Missing files for horizon {horizon}h")
            continue
        
        frost_df = pd.read_csv(frost_file)
        temp_df = pd.read_csv(temp_file)
        
        # Add horizon and task columns
        frost_df["horizon_h"] = horizon
        frost_df["task"] = "frost_classification"
        temp_df["horizon_h"] = horizon
        temp_df["task"] = "temperature_regression"
        
        # Combine
        combined = pd.concat([frost_df, temp_df], ignore_index=True)
        all_data.append(combined)
    
    if not all_data:
        raise FileNotFoundError("No feature importance data found for Matrix A")
    
    # Combine all horizons
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns
    final_df = final_df[["feature", "horizon_h", "task", "importance", "importance_pct", "cumulative_pct"]]
    
    # Sort by task, horizon, then importance
    final_df = final_df.sort_values(["task", "horizon_h", "importance"], ascending=[True, True, False])
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"✅ Saved Matrix A feature importance table to {output_path}")
    print(f"   Total rows: {len(final_df)}")
    print(f"   Features: {final_df['feature'].nunique()}")
    print(f"   Horizons: {final_df['horizon_h'].nunique()}")
    print(f"   Tasks: {final_df['task'].nunique()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Matrix A feature importance supplementary table"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "supplementary_table_S7_matrix_a_feature_importance.csv",
        help="Output path for the table",
    )
    args = parser.parse_args()
    
    generate_matrix_a_supplementary_table(args.output)


if __name__ == "__main__":
    main()

