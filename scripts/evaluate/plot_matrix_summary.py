#!/usr/bin/env python3
"""Plot 2x2 matrix summary across model result directories.

Inputs:
- One or more result directories (each containing evaluation_metrics.json and/or run_metadata.json)

Outputs:
- matrix_summary.csv: flattened table with matrix_cell/track and key metrics
- plots/matrix_heatmap_{metric}.png: 2x2 heatmaps for selected metrics
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


def load_one_result(dir_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "dir": str(dir_path),
        "matrix_cell": None,
        "track": None,
        "radius_km": None,
        "knn_k": None,
    }
    metrics_path = dir_path / "evaluation_metrics.json"
    meta_path = dir_path / "run_metadata.json"

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        result.update({f"metric_{k}": v for k, v in metrics.items() if not str(k).startswith("framework_")})
        # Prefer embedded framework fields if present
        for key in ["matrix_cell", "track", "radius_km", "knn_k"]:
            fw_key = f"framework_{key}"
            if fw_key in metrics and metrics[fw_key] is not None:
                result[key] = metrics[fw_key]

    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        for key in ["matrix_cell", "track", "radius_km", "knn_k"]:
            if meta.get(key) is not None and result.get(key) is None:
                result[key] = meta[key]

    return result


def build_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # Expect lower-is-better for mae/rmse, higher-is-better for r2/roc_auc
    # For summary, we aggregate by mean per cell
    pivot = (
        df.groupby(["track", "matrix_cell"], dropna=False)[metric]
        .mean()
        .reset_index()
        .pivot(index="track", columns="matrix_cell", values=metric)
        .reindex(index=["raw", "top175_features"])
        .reindex(columns=["A", "B", "C", "D"], axis=1)
    )
    return pivot


def plot_heatmap(pivot: pd.DataFrame, metric: str, out_path: Path):
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available; skipping heatmap.")
        return
    ensure_dir(out_path.parent)
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar=True)
    plt.title(f"2x2 Matrix Summary - {metric}")
    plt.ylabel("track")
    plt.xlabel("matrix_cell")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="2x2 Matrix Summary Plotter")
    parser.add_argument(
        "result_dirs",
        type=Path,
        nargs="+",
        help="Paths to evaluation output dirs (contain evaluation_metrics.json)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for summary and plots"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mae", "r2", "roc_auc"],
        help="Metrics to summarize"
    )
    args = parser.parse_args()

    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = project_root / "experiments" / "summaries" / "matrix"
    ensure_dir(out_dir)

    rows: List[Dict[str, Any]] = []
    for d in args.result_dirs:
        d = Path(d)
        if not d.exists():
            print(f"Warning: missing dir {d}")
            continue
        rows.append(load_one_result(d))

    if not rows:
        print("No inputs loaded.")
        return 1

    df = pd.DataFrame(rows)
    # Extract metrics columns
    for m in args.metrics:
        # Try get from metric_test_{m} or metric_{m}
        if f"metric_test_{m}" in df.columns:
            df[m] = df[f"metric_test_{m}"]
        elif f"metric_{m}" in df.columns:
            df[m] = df[f"metric_{m}"]

    csv_path = out_dir / "matrix_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot per metric
    for m in args.metrics:
        if m in df.columns:
            pivot = build_matrix(df, m)
            plot_heatmap(pivot, m, out_dir / f"matrix_heatmap_{m}.png")
        else:
            print(f"Metric '{m}' not found in inputs; skip.")

    return 0


if __name__ == "__main__":
    sys.exit(main())


