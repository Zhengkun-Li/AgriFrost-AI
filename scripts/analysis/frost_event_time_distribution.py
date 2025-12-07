#!/usr/bin/env python3
"""Generate frost event time distributions by month and day-of-month."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.manuscript_style import (
    MANUSCRIPT_COLORS,
    apply_matplotlib_style,
    format_axes,
)

DATA_PATH = Path("data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz")
OUTPUT_DIR = Path("docs/manuscript/figures")
MONTH_FIG_PATH = OUTPUT_DIR / "frost_events_by_month.png"
DAY_FIG_PATH = OUTPUT_DIR / "frost_events_by_day_of_month.png"

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def aggregate_frost_events(
    data_path: Path,
    chunksize: int = 250_000,
) -> Tuple[pd.Series, pd.Series, int]:
    """Count frost events per month and day-of-month using chunked loading."""
    month_counter: Counter[int] = Counter()
    day_counter: Counter[int] = Counter()
    total_rows = 0

    usecols = ["Date", "Air Temp (C)"]

    for chunk in pd.read_csv(
        data_path,
        compression="gzip",
        usecols=usecols,
        chunksize=chunksize,
    ):
        total_rows += len(chunk)
        chunk["Air Temp (C)"] = pd.to_numeric(chunk["Air Temp (C)"], errors="coerce")
        frost_chunk = chunk[chunk["Air Temp (C)"] < 0].copy()

        if frost_chunk.empty:
            continue

        frost_chunk["Date"] = pd.to_datetime(
            frost_chunk["Date"],
            errors="coerce",
        )
        frost_chunk = frost_chunk.dropna(subset=["Date"])

        month_counter.update(frost_chunk["Date"].dt.month.value_counts().to_dict())
        day_counter.update(frost_chunk["Date"].dt.day.value_counts().to_dict())

    month_index = range(1, 13)
    day_index = range(1, 32)

    month_series = pd.Series(
        [month_counter.get(idx, 0) for idx in month_index],
        index=month_index,
        dtype="int64",
    )
    day_series = pd.Series(
        [day_counter.get(idx, 0) for idx in day_index],
        index=day_index,
        dtype="int64",
    )

    total_frost_events = int(month_series.sum())

    print(f"Scanned {total_rows:,} rows and found {total_frost_events:,} frost events.")

    return month_series, day_series, total_frost_events


def plot_distribution(
    counts: pd.Series,
    labels,
    title: str,
    xlabel: str,
    output_path: Path,
    total_count: int,
):
    """Plot a simple bar chart for the provided counts."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    color = MANUSCRIPT_COLORS[0]
    bars = ax.bar(range(len(counts)), counts.values, color=color)

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of frost events")
    format_axes(ax)

    # Annotate bars with counts and percentages for clarity
    for bar, count in zip(bars, counts.values):
        if count > 0:
            share = (count / total_count * 100) if total_count else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts.values) * 0.01,
                f"{count:,}\n({share:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout(pad=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_path}")


def main():
    """Run the frost event distribution analysis."""
    apply_matplotlib_style()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    month_counts, day_counts, total_events = aggregate_frost_events(DATA_PATH)

    divisor = total_events if total_events > 0 else 1

    print("\nTop months for frost events:")
    print(
        pd.DataFrame(
            {
                "month": MONTH_NAMES,
                "events": month_counts.values,
                "share_pct": (month_counts.values / divisor) * 100,
            }
        )
    )

    print("\nTop days of month for frost events:")
    day_summary = pd.DataFrame(
        {
            "day": day_counts.index,
            "events": day_counts.values,
            "share_pct": (day_counts.values / divisor) * 100,
        }
    )
    print(day_summary.sort_values("events", ascending=False).head(10))

    plot_distribution(
        month_counts,
        MONTH_NAMES,
        "Frost events by calendar month",
        "Month",
        MONTH_FIG_PATH,
        total_events,
    )

    plot_distribution(
        day_counts,
        [str(day) for day in day_counts.index],
        "Frost events by day-of-month",
        "Day of month",
        DAY_FIG_PATH,
        total_events,
    )

    print("\nAnalysis complete.")
    print(f"Total frost events detected: {total_events:,}")
    print(f"Monthly distribution figure: {MONTH_FIG_PATH}")
    print(f"Day-of-month distribution figure: {DAY_FIG_PATH}")


if __name__ == "__main__":
    main()

