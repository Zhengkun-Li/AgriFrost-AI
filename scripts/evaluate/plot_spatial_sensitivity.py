#!/usr/bin/env python3
"""Plot spatial radius/kNN sensitivity for C/D experiments.

Inputs:
- A base directory under which multiple runs exist with run_metadata.json or evaluation_metrics.json
  containing matrix_cell in {C, D} and radius_km or knn_k.

Outputs:
- sensitivity_{metric}_by_radius.png
-, sensitivity_{metric}_by_knn.png
"""

import sys
import argparse
from pathlib import Path
import json
from typing import Dict, Any, List

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.utils.path_utils import ensure_dir


def collect_results(base_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in base_dir.rglob("evaluation_metrics.json"):
        row = {"dir": str(p.parent)}
        # Load metrics
        with open(p, "r") as f:
            metrics = json.load(f)
        # Framework fields
        for key in ["matrix_cell", "track", "radius_km", "knn_k"]:
            fw = metrics.get(f"framework_{key}")
            if fw is not None:
                row[key] = fw
        # Basic metrics
        for k, v in metrics.items():
            if k.startswith("test_"):
                row[k.replace("test_", "")] = v
            elif k in ["mae", "rmse", "r2", "roc_auc", "mape"]:
                row[k] = v
        # Fallback to metadata
        meta_path = p.parent / "run_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            for key in ["matrix_cell", "track", "radius_km", "knn_k"]:
                if row.get(key) is None and meta.get(key) is not None:
                    row[key] = meta[key]
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def plot_sensitivity(df: pd.DataFrame, x_field: str, metric: str, out_path: Path, hue: str = None):
    if not MATPLOTLIB_AVAILABLE or df.empty:
        print("matplotlib not available or empty data; skipping plot.")
        return
    ensure_dir(out_path.parent)
    plt.figure(figsize=(7, 4))
    if hue:
        sns.lineplot(data=df, x=x_field, y=metric, hue=hue, marker="o")
    else:
        sns.lineplot(data=df, x=x_field, y=metric, marker="o")
    plt.grid(True, alpha=0.3)
    plt.title(f"Sensitivity of {metric} vs {x_field}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Spatial Sensitivity Plotter (C/D)")
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Base directory to scan for evaluation outputs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for sensitivity plots"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mae", "r2", "roc_auc"],
        help="Metrics to plot"
    )
    parser.add_argument(
        "--hue",
        type=str,
        default="horizon",
        help="Optional hue column (e.g., horizon) if derivable from dir name"
    )
    args = parser.parse_args()

    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = project_root / "experiments" / "summaries" / "sensitivity"
    ensure_dir(out_dir)

    df = collect_results(Path(args.base_dir))
    if df.empty:
        print("No evaluation results found.")
        return 1

    # Derive simple horizon if present in path
    if "horizon" not in df.columns:
        df["horizon"] = df["dir"].str.extract(r"horizon_(\d+)h").iloc[:, 0]

    # Filter to C/D only
    df_cd = df[df["matrix_cell"].isin(["C", "D"])].copy()
    if df_cd.empty:
        print("No C/D results found.")
        return 1

    for m in args.metrics:
        if "radius_km" in df_cd.columns and df_cd["radius_km"].notna().any():
            sub = df_cd[df_cd["radius_km"].notna()].copy()
            if m in sub.columns:
                plot_sensitivity(sub.sort_values("radius_km"), "radius_km", m, out_dir / f"sensitivity_{m}_by_radius.png", hue=args.hue)
        if "knn_k" in df_cd.columns and df_cd["knn_k"].notna().any():
            sub = df_cd[df_cd["knn_k"].notna()].copy()
            if m in sub.columns:
                plot_sensitivity(sub.sort_values("knn_k"), "knn_k", m, out_dir / f"sensitivity_{m}_by_knn.png", hue=args.hue)

    # Save the flattened data for inspection
    df_cd.to_csv(out_dir / "sensitivity_flat.csv", index=False)
    print(f"Saved: {out_dir / 'sensitivity_flat.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


