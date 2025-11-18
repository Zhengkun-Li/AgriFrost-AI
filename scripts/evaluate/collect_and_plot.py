#!/usr/bin/env python3
"""Collect experiment result directories and invoke 2x2 matrix and spatial sensitivity plotting.

This utility scans an experiments tree for evaluation outputs (directories containing
`evaluation_metrics.json`), optionally filters them, then:
  1) Calls plot_matrix_summary.py with the collected directories
  2) Calls plot_spatial_sensitivity.py pointing to the base experiments directory
"""

import sys
import argparse
from pathlib import Path
import re
import subprocess
from typing import List

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.path_utils import ensure_dir


def find_result_dirs(base_dir: Path) -> List[Path]:
    return [p.parent for p in base_dir.rglob("evaluation_metrics.json")]


def apply_filters(
    dirs: List[Path],
    include_cells: List[str] = None,
    include_tracks: List[str] = None,
    horizon_regex: str = None,
) -> List[Path]:
    filtered = []
    pattern = re.compile(horizon_regex) if horizon_regex else None
    for d in dirs:
        s = str(d)
        if include_cells:
            # expect /{A|B|C|D}/ segment present
            if not any(f"/{cell}/" in s for cell in include_cells):
                continue
        if include_tracks:
            if not any(f"/{trk}/" in s for trk in include_tracks):
                continue
        if pattern:
            if not pattern.search(s):
                continue
        filtered.append(d)
    return filtered


def run_plot_matrix_summary(result_dirs: List[Path], out_dir: Path, metrics: List[str]):
    if not result_dirs:
        print("No result directories to summarize for 2x2 matrix.")
        return
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "evaluate" / "plot_matrix_summary.py"),
        *[str(p) for p in result_dirs],
        "--output",
        str(out_dir),
        "--metrics",
        *metrics,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_plot_spatial_sensitivity(base_dir: Path, out_dir: Path, metrics: List[str]):
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "evaluate" / "plot_spatial_sensitivity.py"),
        str(base_dir),
        "--output",
        str(out_dir),
        "--metrics",
        *metrics,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Collect experiments and plot 2x2 matrix + spatial sensitivity")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=project_root / "experiments",
        help="Root directory to scan"
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=project_root / "experiments" / "summaries" / "matrix",
        help="Output directory for 2x2 matrix summary"
    )
    parser.add_argument(
        "--sensitivity-output",
        type=Path,
        default=project_root / "experiments" / "summaries" / "sensitivity",
        help="Output directory for spatial sensitivity plots"
    )
    parser.add_argument(
        "--cells",
        type=str,
        nargs="*",
        default=None,
        help="Filter by matrix cells, e.g., A B C D"
    )
    parser.add_argument(
        "--tracks",
        type=str,
        nargs="*",
        default=None,
        help="Filter by tracks, e.g., raw top175_features"
    )
    parser.add_argument(
        "--horizon-regex",
        type=str,
        default=None,
        help="Regex to filter by horizon path (e.g., 'horizon_(3|6|12|24)h')"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mae", "r2", "roc_auc"],
        help="Metrics to use for plots"
    )
    parser.add_argument(
        "--skip-matrix",
        action="store_true",
        help="Skip 2x2 matrix summary"
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip spatial sensitivity plots"
    )

    args = parser.parse_args()

    base_dir = Path(args.experiments_root)
    if not base_dir.exists():
        raise FileNotFoundError(f"Experiments root not found: {base_dir}")

    all_dirs = find_result_dirs(base_dir)
    selected = apply_filters(all_dirs, args.cells, args.tracks, args.horizon_regex)
    print(f"Found {len(all_dirs)} result dirs; selected {len(selected)} after filters.")

    if not args.skip_matrix:
        run_plot_matrix_summary(selected, Path(args.matrix_output), args.metrics)

    if not args.skip_sensitivity:
        run_plot_spatial_sensitivity(base_dir, Path(args.sensitivity_output), args.metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())


