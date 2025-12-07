#!/usr/bin/env python3
"""Aggregate key metrics from every experiment.log under experiments/."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATTERN = re.compile(r"^(Model|Matrix Cell|Track|Horizons):\s*(.+)$")
METRIC_PATTERN = re.compile(
    r"^\s+ROC-AUC:\s*([\d.]+)\s*$|^\s+PR-AUC:\s*([\d.]+)\s*$|^\s+Brier Score:\s*([\d.]+)\s*$|^\s+MAE:\s*([\d.]+)°C\s*$|^\s+RMSE:\s*([\d.]+)°C\s*$"
)
HORIZON_PATTERN = re.compile(r"^\s+Training horizon:\s*(\d+)h")


def parse_log(log_path: Path) -> List[Dict[str, str]]:
    result_rows: List[Dict[str, str]] = []
    header: Dict[str, str] = {}
    current_horizon: Optional[str] = None
    current_row: Dict[str, str] = {}

    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                # Header metadata
                header_match = LOG_PATTERN.match(line)
                if header_match:
                    key, value = header_match.groups()
                    header[key.lower()] = value.strip()
                    continue

                # Horizon marker
                horizon_match = HORIZON_PATTERN.match(line)
                if horizon_match:
                    # flush previous row
                    if current_row:
                        current_row.update(header)
                        result_rows.append(current_row)
                        current_row = {}
                    current_horizon = horizon_match.group(1)
                    current_row = {
                        "horizon_h": current_horizon,
                        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
                    }
                    continue

                # Metrics
                metric_match = METRIC_PATTERN.match(line)
                if metric_match and current_row:
                    roc_auc, pr_auc, brier, mae, rmse = metric_match.groups()
                    if roc_auc:
                        current_row["roc_auc"] = roc_auc
                    if pr_auc:
                        current_row["pr_auc"] = pr_auc
                    if brier:
                        current_row["brier_score"] = brier
                    if mae:
                        current_row["mae_c"] = mae
                    if rmse:
                        current_row["rmse_c"] = rmse
    except Exception as exc:  # pragma: no cover
        print(f"⚠️  Failed to parse {log_path}: {exc}", file=sys.stderr)

    if current_row:
        current_row.update(header)
        result_rows.append(current_row)
    return result_rows


def main():
    logs = sorted(PROJECT_ROOT.glob("experiments/**/experiment.log"))
    all_rows: List[Dict[str, str]] = []
    for log_path in logs:
        rows = parse_log(log_path)
        all_rows.extend(rows)

    if not all_rows:
        print("No experiment.log entries found.")
        return

    fieldnames = [
        "model",
        "matrix cell",
        "track",
        "horizons",
        "horizon_h",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "mae_c",
        "rmse_c",
        "log_path",
    ]

    output_path = PROJECT_ROOT / "docs" / "experiments" / "experiment_log_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in all_rows:
            f.write(
                ",".join(row.get(field, "") for field in fieldnames)
                .replace("\n", " ")
                + "\n"
            )

    print(f"✅ Parsed {len(logs)} logs, wrote summary to {output_path}")


if __name__ == "__main__":
    main()

