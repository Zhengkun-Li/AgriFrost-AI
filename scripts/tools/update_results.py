#!/usr/bin/env python3
"""
Aggregate experiment metrics into the canonical results CSV files.

Usage:
    python scripts/tools/update_results.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"

EXCLUDED_DIR_NAMES = {"graph_cache"}
EXCLUDED_PREFIXES = ("test",)


def _contains_excluded_part(path: Path) -> bool:
    """Return True if the path includes folders we should ignore."""
    try:
        parts = path.relative_to(PROJECT_ROOT).parts
    except ValueError:
        parts = path.parts

    for part in parts:
        lowered = part.lower()
        if part in EXCLUDED_DIR_NAMES:
            return True
        if any(lowered.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            return True
    return False


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_run_records() -> List[Dict]:
    """Walk through the experiments directory and collect metrics."""
    records: List[Dict] = []
    for metadata_path in EXPERIMENTS_DIR.rglob("run_metadata.json"):
        if _contains_excluded_part(metadata_path):
            continue

        run_dir = metadata_path.parent
        frost_metrics_path = run_dir / "frost_metrics.json"
        temp_metrics_path = run_dir / "temp_metrics.json"

        if not frost_metrics_path.exists():
            continue

        metadata = _load_json(metadata_path)
        frost_metrics = _load_json(frost_metrics_path)
        temp_metrics = _load_json(temp_metrics_path) if temp_metrics_path.exists() else {}

        radius = metadata.get("radius_km")
        record = {
            "model": metadata.get("model_name"),
            "matrix_cell": metadata.get("matrix_cell"),
            "track": metadata.get("track"),
            "horizon_h": int(metadata.get("horizon_h")),
            "radius_km": _to_float(radius) if radius is not None else None,
            "roc_auc": _to_float(frost_metrics.get("roc_auc")),
            "pr_auc": _to_float(frost_metrics.get("pr_auc")),
            "brier_score": _to_float(frost_metrics.get("brier_score")),
            "f1_score": _to_float(frost_metrics.get("f1")),
            "precision": _to_float(frost_metrics.get("precision")),
            "recall": _to_float(frost_metrics.get("recall")),
            "accuracy": _to_float(frost_metrics.get("accuracy")),
            "ece": _to_float(frost_metrics.get("ece")),
            "temp_rmse": _to_float(temp_metrics.get("rmse")),
            "temp_mae": _to_float(temp_metrics.get("mae")),
            "temp_r2": _to_float(temp_metrics.get("r2")),
            "path": str(run_dir.relative_to(PROJECT_ROOT)),
        }

        if record["model"] is None or record["matrix_cell"] is None:
            continue

        if record["radius_km"] is None:
            record["radius_km"] = 0.0

        records.append(record)
    return records


def write_model_performance_csv(records: List[Dict]) -> None:
    """Write the flat table with every run."""
    fieldnames = [
        "model",
        "matrix_cell",
        "horizon_h",
        "radius_km",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "f1_score",
        "precision",
        "recall",
        "temp_rmse",
        "temp_mae",
        "temp_r2",
        "path",
    ]

    rows = sorted(
        records,
        key=lambda item: (
            item["matrix_cell"],
            item["horizon_h"],
            item["model"],
            item["radius_km"],
            item["path"],
        ),
    )

    output_path = RESULTS_DIR / "model_performance_all_models.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def select_best_runs(records: Iterable[Dict]) -> List[Dict]:
    """Pick the best run per matrix cell and forecast horizon."""
    grouped: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for record in records:
        grouped[(record["matrix_cell"], record["horizon_h"])].append(record)

    best_rows: List[Dict] = []
    for key, run_list in grouped.items():
        best = max(
            run_list,
            key=lambda item: (
                item.get("roc_auc") or 0.0,
                item.get("pr_auc") or 0.0,
            ),
        )
        best_rows.append(
            {
                "model": best["model"],
                "matrix_cell": best["matrix_cell"],
                "horizon_h": best["horizon_h"],
                "radius_km": best["radius_km"],
                "roc_auc": best.get("roc_auc"),
                "pr_auc": best.get("pr_auc"),
                "brier_score": best.get("brier_score"),
                "temp_rmse": best.get("temp_rmse"),
                "path": best.get("path"),
            }
        )
    return sorted(
        best_rows,
        key=lambda item: (item["matrix_cell"], item["horizon_h"]),
    )


def write_best_runs_csv(best_rows: List[Dict]) -> None:
    fieldnames = [
        "model",
        "matrix_cell",
        "horizon_h",
        "radius_km",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "temp_rmse",
        "path",
    ]
    output_path = RESULTS_DIR / "best_per_matrix_horizon.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in best_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def write_matrix_horizon_summary(records: Iterable[Dict]) -> None:
    grouped: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for record in records:
        grouped[(record["matrix_cell"], record["horizon_h"])].append(record)

    matrix_order = sorted({key[0] for key in grouped.keys()})
    horizon_order = sorted({key[1] for key in grouped.keys()})

    def metric_values(run_list: List[Dict], key: str) -> List[float]:
        return [value for value in (run.get(key) for run in run_list) if value is not None]

    output_path = RESULTS_DIR / "matrix_horizon_metrics_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "", "roc_auc", "roc_auc", "pr_auc", "pr_auc", "brier_score", "brier_score", "temp_rmse", "temp_rmse"])
        writer.writerow(["", "", "mean", "max", "mean", "max", "mean", "min", "mean", "min"])
        writer.writerow(["matrix_cell", "horizon_h", "", "", "", "", "", "", "", ""])

        for matrix_cell in matrix_order:
            for horizon in horizon_order:
                run_list = grouped.get((matrix_cell, horizon))
                if not run_list:
                    continue

                roc_values = metric_values(run_list, "roc_auc")
                pr_values = metric_values(run_list, "pr_auc")
                brier_values = metric_values(run_list, "brier_score")
                temp_values = metric_values(run_list, "temp_rmse")

                row = [
                    matrix_cell,
                    horizon,
                    _format_metric(sum(roc_values) / len(roc_values)) if roc_values else "",
                    _format_metric(max(roc_values)) if roc_values else "",
                    _format_metric(sum(pr_values) / len(pr_values)) if pr_values else "",
                    _format_metric(max(pr_values)) if pr_values else "",
                    _format_metric(sum(brier_values) / len(brier_values)) if brier_values else "",
                    _format_metric(min(brier_values)) if brier_values else "",
                    _format_metric(sum(temp_values) / len(temp_values)) if temp_values else "",
                    _format_metric(min(temp_values)) if temp_values else "",
                ]
                writer.writerow(row)


def main() -> None:
    records = collect_run_records()
    if not records:
        print("No experiment runs found. Abort.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    write_model_performance_csv(records)
    best_rows = select_best_runs(records)
    write_best_runs_csv(best_rows)
    write_matrix_horizon_summary(records)

    print(f"Wrote {len(records)} runs into model_performance_all_models.csv")


if __name__ == "__main__":
    main()

