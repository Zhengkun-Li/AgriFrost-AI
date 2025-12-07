#!/usr/bin/env python3
"""Generate performance-focused figures directly from aggregated results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_CSV = PROJECT_ROOT / "results" / "model_performance_all_models.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures" / "v2"

from src.visualization.manuscript_style import (  # type: ignore  # noqa: E402
    MANUSCRIPT_COLORS,
    apply_matplotlib_style,
    format_axes,
)

HORIZON_ORDER = [3, 6, 12, 24]
MATRIX_ORDER = ["A", "B", "C", "D"]
HORIZON_LABELS = {h: f"{h}h" for h in HORIZON_ORDER}


def load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing results CSV: {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)
    df["model"] = df["model"].str.lower()
    df["matrix_cell"] = df["matrix_cell"].str.upper()
    return df


def _select_lightgbm_runs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["model"] == "lightgbm"].copy()


def _choose_best_per_horizon(
    df: pd.DataFrame,
    *,
    matrix: str,
    prefer_raw: bool = False,
) -> pd.DataFrame:
    subset = df[df["matrix_cell"] == matrix].copy()
    if prefer_raw:
        subset = subset[subset["path"].str.contains("/raw/", case=False, na=False)]
    subset = subset.sort_values(
        ["horizon_h", "roc_auc", "pr_auc"],
        ascending=[True, False, False],
    )
    return subset.groupby("horizon_h", as_index=False).first()


def plot_radius_horizon_performance(df: pd.DataFrame, output_path: Path) -> None:
    subset = df[
        (df["matrix_cell"] == "C")
        & (df["radius_km"].notna())
        & (df["radius_km"] > 0)
        & df["path"].str.contains("/raw/", case=False, na=False)
    ].copy()
    if subset.empty:
        raise ValueError("No Matrix C raw runs with radius metadata found.")

    metrics = [
        ("roc_auc", "ROC-AUC (↑)"),
        ("pr_auc", "PR-AUC (↑)"),
        ("brier_score", "Brier Score (↓)"),
        ("temp_rmse", "Temp RMSE (°C, ↓)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    color_map = {
        3: MANUSCRIPT_COLORS[0],
        6: MANUSCRIPT_COLORS[1],
        12: MANUSCRIPT_COLORS[2],
        24: MANUSCRIPT_COLORS[3],
    }

    for ax, (metric, title) in zip(axes.flat, metrics):
        for horizon in HORIZON_ORDER:
            horizon_df = (
                subset[subset["horizon_h"] == horizon]
                .sort_values("radius_km")
                .dropna(subset=[metric])
            )
            if horizon_df.empty:
                continue
            ax.plot(
                horizon_df["radius_km"],
                horizon_df[metric],
                marker="o",
                linewidth=2,
                color=color_map[horizon],
                label=HORIZON_LABELS[horizon],
            )
            if metric in {"roc_auc", "pr_auc"}:
                best_idx = horizon_df[metric].idxmax()
            else:  # lower is better
                best_idx = horizon_df[metric].idxmin()
            best_row = horizon_df.loc[best_idx]
            ax.scatter(
                best_row["radius_km"],
                best_row[metric],
                color=color_map[horizon],
                s=45,
                zorder=5,
            )
            ax.annotate(
                f"{int(best_row['radius_km'])} km",
                (best_row["radius_km"], best_row[metric]),
                textcoords="offset points",
                xytext=(0, -12 if metric == "temp_rmse" else 8),
                ha="center",
                fontsize=8,
                color=color_map[horizon],
            )
        ax.set_title(title)
        ax.set_xlabel("Radius (km)")
        format_axes(ax)
        # Add legend to each subplot, similar to Figure 11 style
        ax.legend(loc='best', ncol=2, framealpha=0.9, fontsize=9)

    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    plt.tight_layout(pad=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"💾 Saved radius × horizon figure to {output_path}")


def plot_single_vs_spatial(df: pd.DataFrame, output_path: Path) -> None:
    single = _choose_best_per_horizon(
        df, matrix="A", prefer_raw=True
    ).assign(matrix_label="Single (Matrix A, raw)")
    spatial = (
        _choose_best_per_horizon(df, matrix="C", prefer_raw=True)
        .assign(matrix_label="Spatial (Matrix C, raw + neighbors)")
    )
    comparison = pd.concat([single, spatial], ignore_index=True)
    comparison = comparison[comparison["horizon_h"].isin(HORIZON_ORDER)]
    comparison.sort_values(["matrix_label", "horizon_h"], inplace=True)

    matrices = comparison["matrix_label"].unique().tolist()
    x = np.arange(len(HORIZON_ORDER))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    metrics = [("pr_auc", "PR-AUC (↑)"), ("temp_rmse", "Temp RMSE (°C, ↓)")]

    for ax, (metric, title) in zip(axes, metrics):
        for i, matrix in enumerate(matrices):
            values = (
                comparison[comparison["matrix_label"] == matrix]
                .set_index("horizon_h")
                .reindex(HORIZON_ORDER)[metric]
            )
            offsets = x + (i - 0.5) * width
            color = MANUSCRIPT_COLORS[i]
            bars = ax.bar(offsets, values, width=width, color=color, label=matrix)
            for bar, value, horizon in zip(bars, values, HORIZON_ORDER):
                if pd.isna(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.003 if metric == "pr_auc" else 0.05),
                    f"{value:.3f}" if metric == "pr_auc" else f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([HORIZON_LABELS[h] for h in HORIZON_ORDER])
        ax.set_title(title)
        ax.set_xlabel("Forecast horizon")
        format_axes(ax)

    axes[0].set_ylabel("Score")
    axes[1].set_ylabel("Error (°C)")
    axes[0].legend(loc="upper left", frameon=False)
    plt.tight_layout(pad=1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"💾 Saved single-vs-spatial comparison to {output_path}")


def plot_matrix_performance(df: pd.DataFrame, output_path: Path) -> None:
    summaries: Dict[str, pd.DataFrame] = {}
    for matrix in MATRIX_ORDER:
        prefer_raw = matrix in {"A", "C"}
        summaries[matrix] = _choose_best_per_horizon(
            df, matrix=matrix, prefer_raw=prefer_raw
        )

    metrics = [
        ("roc_auc", "ROC-AUC (↑)"),
        ("pr_auc", "PR-AUC (↑)"),
        ("brier_score", "Brier Score (↓)"),
        ("temp_rmse", "Temp RMSE (°C, ↓)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        for idx, matrix in enumerate(MATRIX_ORDER):
            values = (
                summaries[matrix]
                .set_index("horizon_h")
                .reindex(HORIZON_ORDER)[metric]
            )
            ax.plot(
                HORIZON_ORDER,
                values,
                marker="o",
                linewidth=2,
                color=MANUSCRIPT_COLORS[idx],
                label=f"Matrix {matrix}",
            )
        ax.set_title(title)
        ax.set_xticks(HORIZON_ORDER)
        ax.set_xticklabels([HORIZON_LABELS[h] for h in HORIZON_ORDER])
        ax.set_xlabel("Forecast horizon (hours)")
        format_axes(ax)

    axes[0, 0].legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=4,
        frameon=False,
    )
    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    plt.tight_layout(pad=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"💾 Saved matrix performance overview to {output_path}")


def plot_feature_category_importance(output_path: Path) -> None:
    """Generate Figure 13: Feature category importance across horizons."""
    csv_path = (
        PROJECT_ROOT
        / "docs"
        / "manuscript"
        / "Supplementary"
        / "supplementary_table_S5_feature_category_importance.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing feature category data: {csv_path}")

    df = pd.read_csv(csv_path)
    tasks = ["frost_classification", "temperature_regression"]
    task_labels = {
        "frost_classification": "Frost Classification",
        "temperature_regression": "Temperature Regression",
    }

    # Define darker colors for feature categories
    # Using darker shades of the manuscript colors
    category_colors = {
        "Rolling Statistics": "#1f77b4",  # Darker blue
        "Lag Features": "#ff7f0e",  # Darker orange
        "Time Features": "#2ca02c",  # Darker green
        "Other Features": "#d62728",  # Darker red
        "Wind Features": "#9467bd",  # Darker purple
        "Derived Meteorological": "#8c564b",  # Darker brown
        "Station Features": "#e377c2",  # Darker pink
        "Soil Features": "#7f7f7f",  # Darker gray
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, task in zip(axes, tasks):
        task_df = df[df["task"] == task].copy()
        categories = task_df["category"].unique()

        # Stacked area plot
        x = HORIZON_ORDER
        bottom = np.zeros(len(x))

        for category in categories:
            category_data = (
                task_df[task_df["category"] == category]
                .set_index("horizon_h")
                .reindex(HORIZON_ORDER)["cumulative_importance_pct"]
                .fillna(0)
            )
            color = category_colors.get(category, "#000000")
            ax.fill_between(
                x,
                bottom,
                bottom + category_data,
                label=category,
                color=color,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += category_data

        ax.set_title(task_labels[task])
        ax.set_xlabel("Forecast Horizon (hours)")
        ax.set_ylabel("Cumulative Importance (%)")
        ax.set_xticks(HORIZON_ORDER)
        ax.set_xticklabels([HORIZON_LABELS[h] for h in HORIZON_ORDER])
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, linestyle="--")
        format_axes(ax)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    plt.tight_layout(pad=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved feature category importance figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manuscript-ready performance figures."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store generated figures.",
    )
    args = parser.parse_args()

    apply_matplotlib_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_df = _select_lightgbm_runs(load_results())

    plot_radius_horizon_performance(
        results_df, args.output_dir / "matrix_c_radius_vs_horizon.png"
    )
    plot_single_vs_spatial(
        results_df, args.output_dir / "single_vs_spatial_performance.png"
    )
    plot_matrix_performance(
        results_df, args.output_dir / "matrix_performance_over_horizons.png"
    )
    plot_feature_category_importance(
        args.output_dir / "feature_category_cross_horizon.png"
    )

    print("\n✅ Performance figures refreshed.")


if __name__ == "__main__":
    main()

