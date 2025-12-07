#!/usr/bin/env python3
"""Generate manuscript figures with consistent academic styling."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - plotly is optional
    px = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.manuscript_style import (  # type: ignore  # noqa: E402
    MANUSCRIPT_COLORS,
    STATUS_COLOR_MAP,
    apply_matplotlib_style,
    format_axes,
    get_status_colors,
)
from src.evaluation.metrics import MetricsCalculator  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_PATH = PROJECT_ROOT / "data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz"
METADATA_PATH = PROJECT_ROOT / "data/external/cimis_station_metadata.json"
BEST_MODEL_PATH = PROJECT_ROOT / "results/best_per_matrix_horizon.csv"
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "experiments/lightgbm/raw/C/radius_100km/full_training/horizon_3h/predictions.json"
)
OUTPUT_DIR = PROJECT_ROOT / "docs/manuscript/figures"

QC_COLUMNS = {
    "qc": "ETo (mm)",
    "qc.1": "Precip (mm)",
    "qc.2": "Sol Rad (W/sq.m)",
    "qc.3": "Vap Pres (kPa)",
    "qc.4": "Air Temp (C)",
    "qc.5": "Rel Hum (%)",
    "qc.6": "Dew Point (C)",
    "qc.7": "Wind Speed (m/s)",
    "qc.8": "Wind Dir (0-360)",
    "qc.9": "Soil Temp (C)",
}

STATUS_ORDER = ["PASS", "Y", "Q", "P", "M", "R", "S"]
BAD_FLAGS = {"Q", "P", "M", "R", "S"}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def normalize_flags(series: pd.Series) -> pd.Series:
    """Normalize QC flags (blank/NaN -> PASS, uppercase labels)."""
    flags = series.astype("string")
    flags = flags.str.strip().str.upper()
    flags = flags.fillna("PASS")
    flags = flags.replace({"": "PASS", "NAN": "PASS"})
    return flags


def aggregate_qc_statistics(
    data_path: Path,
    chunksize: int = 250_000,
) -> Tuple[
    Counter,
    Dict[int, Counter],
    Dict[str, Counter],
    Counter,
    Counter,
    Counter,
    Counter,
]:
    """Scan the CIMIS dataset and collect QC distributions."""
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    overall_counts: Counter = Counter()
    station_flag_counts: Dict[int, Counter] = defaultdict(Counter)
    variable_flag_counts: Dict[str, Counter] = {var: Counter() for var in QC_COLUMNS.values()}
    station_totals: Counter = Counter()
    station_bad: Counter = Counter()
    variable_totals: Counter = Counter()
    variable_bad: Counter = Counter()

    usecols = ["Stn Id"] + list(QC_COLUMNS.keys())

    print(f"🔍 Aggregating QC statistics from {data_path} ...")
    row_counter = 0

    for chunk in pd.read_csv(
        data_path,
        usecols=usecols,
        chunksize=chunksize,
        compression="gzip",
    ):
        chunk["Stn Id"] = chunk["Stn Id"].astype(int)
        stn_ids = chunk["Stn Id"]
        row_counter += len(chunk)

        for qc_col, variable_name in QC_COLUMNS.items():
            flags = normalize_flags(chunk[qc_col])
            counts = flags.value_counts()

            overall_counts.update(counts.to_dict())
            variable_flag_counts[variable_name].update(counts.to_dict())

            variable_totals[variable_name] += len(flags)
            variable_bad[variable_name] += int(flags.isin(BAD_FLAGS).sum())

            crosstab = pd.crosstab(stn_ids, flags)
            for stn_id, row in crosstab.iterrows():
                station_flag_counts[stn_id].update(row.to_dict())
                total = int(row.sum())
                station_totals[stn_id] += total
                station_bad[stn_id] += int(row.reindex(BAD_FLAGS, fill_value=0).sum())

    print(f"✅ Processed {row_counter:,} hourly records for QC analysis.")
    return (
        overall_counts,
        station_flag_counts,
        variable_flag_counts,
        station_totals,
        station_bad,
        variable_totals,
        variable_bad,
    )


def load_station_metadata(metadata_path: Path) -> Dict[int, str]:
    """Return mapping from station id to display name."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    mapping = {}
    for entry in metadata:
        try:
            stn_id = int(entry["Stn Id"])
        except (KeyError, ValueError, TypeError):
            continue
        name = entry.get("Stn Name") or f"Station {stn_id}"
        mapping[stn_id] = name
    return mapping


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_overall_qc_distribution(overall_counts: Counter, output_path: Path) -> None:
    """Plot aggregated QC distribution."""
    total = sum(overall_counts.values())
    ordered_counts = [overall_counts.get(status, 0) for status in STATUS_ORDER]
    other = total - sum(ordered_counts)

    statuses = STATUS_ORDER + (["OTHER"] if other > 0 else [])
    counts = ordered_counts + ([other] if other > 0 else [])
    shares = [count / total * 100 if total else 0 for count in counts]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = get_status_colors(statuses)

    bars = ax.bar(range(len(statuses)), counts, color=colors)
    ax.set_xticks(range(len(statuses)))
    ax.set_xticklabels(statuses)
    ax.set_ylabel("Number of QC flags")
    ax.set_title("QC flag distribution (all variables)")
    format_axes(ax)

    for bar, count, share in zip(bars, counts, shares):
        if count == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.001,
            f"{count:,}\n({share:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved {output_path}")


def build_station_qc_dataframe(
    station_flag_counts: Dict[int, Counter],
    station_totals: Counter,
    station_names: Dict[int, str],
) -> pd.DataFrame:
    """Convert station QC counters into percentage dataframe."""
    records = []
    statuses = STATUS_ORDER + ["OTHER"]

    for stn_id in sorted(station_flag_counts.keys()):
        total = station_totals.get(stn_id, 0)
        if total == 0:
            continue
        counts = station_flag_counts[stn_id]
        row = {
            "station": f"{station_names.get(stn_id, f'Station {stn_id}')} ({stn_id})",
        }
        accounted = 0
        for status in STATUS_ORDER:
            value = counts.get(status, 0)
            row[status] = value / total * 100
            accounted += value
        row["OTHER"] = max(total - accounted, 0) / total * 100
        records.append(row)

    df = pd.DataFrame(records)
    return df.sort_values("station", key=lambda col: col.str.lower())


def plot_station_qc_stacks(
    station_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot stacked percentages of QC flags per station."""
    statuses = [status for status in STATUS_ORDER + ["OTHER"] if status in station_df.columns]
    colors = get_status_colors(statuses)

    fig, ax = plt.subplots(figsize=(10, 7))
    bottom = np.zeros(len(station_df))

    for status, color in zip(statuses, colors):
        values = station_df[status].values
        ax.barh(station_df["station"], values, left=bottom, color=color, label=status)
        bottom += values

    ax.set_xlabel("Share of QC flags (%)")
    ax.set_title("QC composition by station")
    ax.legend(loc="lower right", ncol=2, frameon=False)
    format_axes(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved {output_path}")


def plot_bad_by_variable(
    variable_totals: Counter,
    variable_bad: Counter,
    output_path: Path,
) -> None:
    """Plot percentage of bad QC flags per variable."""
    rows = []
    for variable, total in variable_totals.items():
        bad = variable_bad.get(variable, 0)
        share = bad / total * 100 if total else 0
        rows.append({"variable": variable, "bad_pct": share})
    df = pd.DataFrame(rows).sort_values("bad_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["variable"], df["bad_pct"], color=MANUSCRIPT_COLORS[1])
    ax.set_xlabel("Share of QC = {Q,P,M,R,S} (%)")
    ax.set_title("Problematic QC share by variable")
    format_axes(ax)

    for y, value in enumerate(df["bad_pct"]):
        ax.text(value + 0.1, y, f"{value:.2f}%", va="center", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved {output_path}")


def plot_bad_by_station(
    station_totals: Counter,
    station_bad: Counter,
    station_names: Dict[int, str],
    output_path: Path,
) -> None:
    """Plot percentage of bad QC flags per station."""
    rows = []
    for stn_id, total in station_totals.items():
        bad = station_bad.get(stn_id, 0)
        share = bad / total * 100 if total else 0
        rows.append(
            {
                "station": f"{station_names.get(stn_id, f'Station {stn_id}')} ({stn_id})",
                "bad_pct": share,
            }
        )
    df = pd.DataFrame(rows).sort_values("bad_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(df["station"], df["bad_pct"], color=MANUSCRIPT_COLORS[2])
    ax.invert_yaxis()
    ax.set_xlabel("Share of QC = {Q,P,M,R,S} (%)")
    ax.set_title("Stations with most QC issues (all variables)")
    format_axes(ax)

    for y, value in enumerate(df["bad_pct"]):
        ax.text(value + 0.1, y, f"{value:.2f}%", va="center", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved {output_path}")


def plot_matrix_model_comparison(csv_path: Path, output_path: Path) -> None:
    """Plot best ROC-AUC per matrix cell and model family."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary file: {csv_path}")

    df = pd.read_csv(csv_path)
    df["model"] = df["model"].str.lower()

    summaries = []
    for (matrix, model), group in df.groupby(["matrix_cell", "model"]):
        idx = group["roc_auc"].idxmax()
        row = group.loc[idx]
        summaries.append(
            {
                "matrix": matrix,
                "model": model,
                "roc_auc": row["roc_auc"],
                "horizon": int(row["horizon_h"]),
            }
        )

    summary_df = pd.DataFrame(summaries)
    matrix_order = ["A", "B", "C", "D"]
    model_order = ["lightgbm", "catboost", "xgboost"]

    x = np.arange(len(matrix_order))
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(model_order):
        subset = (
            summary_df[summary_df["model"] == model]
            .set_index("matrix")
            .reindex(matrix_order)
        )
        values = subset["roc_auc"].values
        offsets = x + (i - 1) * bar_width
        color = MANUSCRIPT_COLORS[i % len(MANUSCRIPT_COLORS)]
        bars = ax.bar(offsets, values, width=bar_width, label=model.title(), color=color)

        for bar, value, (_, row) in zip(bars, values, subset.iterrows()):
            if np.isnan(value):
                continue
            horizon = row["horizon"]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{value:.3f}\n{horizon}h",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(matrix_order)
    ax.set_ylabel("Best ROC-AUC")
    ax.set_title("Best ROC-AUC per matrix cell and model family")
    ax.set_ylim(0.9, 1.01)
    ax.legend(frameon=False, ncol=3)
    format_axes(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved {output_path}")


def plot_reliability_diagram(predictions_path: Path, output_path: Path, n_bins: int = 12) -> None:
    """Re-generate the reliability diagram with consistent styling."""
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions: {predictions_path}")

    with open(predictions_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    frost_preds = preds.get("frost")
    if frost_preds is None:
        raise KeyError("Predictions JSON missing 'frost' key.")

    y_true = np.array(frost_preds.get("y_true", []), dtype=float)
    y_proba = np.array(frost_preds.get("y_proba", []), dtype=float)

    if len(y_true) == 0 or len(y_true) != len(y_proba):
        raise ValueError("Invalid frost predictions arrays for reliability diagram.")

    reliability = MetricsCalculator.calculate_reliability_data(y_true, y_proba, n_bins)
    ece = MetricsCalculator.calculate_ece(y_true, y_proba, n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    perfect = np.linspace(0, 1, 50)
    ax.plot(perfect, perfect, linestyle="--", color="#6c757d", label="Perfect calibration")

    valid = ~(
        np.isnan(reliability["predicted_probs"]) | np.isnan(reliability["actual_freqs"])
    )
    ax.plot(
        reliability["predicted_probs"][valid],
        reliability["actual_freqs"][valid],
        marker="o",
        linewidth=2,
        color=MANUSCRIPT_COLORS[0],
        label="Model",
    )

    for pred, actual, count in zip(
        reliability["predicted_probs"][valid],
        reliability["actual_freqs"][valid],
        reliability["counts"][valid],
    ):
        ax.annotate(
            f"n={int(count)}",
            (pred, actual),
            textcoords="offset points",
            xytext=(4, -10),
            fontsize=8,
        )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram – Matrix C, 3h horizon")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)
    ax.text(
        0.98,
        0.05,
        f"ECE = {ece:.4f}",
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        transform=ax.transAxes,
    )
    format_axes(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved {output_path}")


def plot_reliability_diagram_2x2(
    predictions_paths: Dict[str, Path], output_path: Path, n_bins: int = 12
) -> None:
    """Generate 2x2 reliability diagram for four horizons."""
    # Optimal radii for each horizon (from Table 1 in manuscript)
    horizon_configs = {
        "3h": {"radius": "60km", "label": "3小时 (60 km)"},
        "6h": {"radius": "160km", "label": "6小时 (160 km)"},
        "12h": {"radius": "200km", "label": "12小时 (200 km)"},
        "24h": {"radius": "180km", "label": "24小时 (180 km)"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx, (horizon, config) in enumerate(horizon_configs.items()):
        ax = axes[idx]
        pred_path = predictions_paths.get(horizon)

        if pred_path is None or not pred_path.exists():
            print(f"⚠️  Missing predictions for {horizon}: {pred_path}")
            ax.text(0.5, 0.5, f"Missing data\nfor {horizon}", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            format_axes(ax)
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)

        frost_preds = preds.get("frost")
        if frost_preds is None:
            print(f"⚠️  Missing 'frost' key in {pred_path}")
            continue

        y_true = np.array(frost_preds.get("y_true", []), dtype=float)
        y_proba = np.array(frost_preds.get("y_proba", []), dtype=float)

        if len(y_true) == 0 or len(y_true) != len(y_proba):
            print(f"⚠️  Invalid predictions for {horizon}")
            continue

        reliability = MetricsCalculator.calculate_reliability_data(y_true, y_proba, n_bins)
        ece = MetricsCalculator.calculate_ece(y_true, y_proba, n_bins)
        # Calculate Brier score manually
        brier = np.mean((y_proba - y_true) ** 2)

        # Perfect calibration line
        perfect = np.linspace(0, 1, 50)
        ax.plot(perfect, perfect, linestyle="--", color="#6c757d", linewidth=1.5, label="Perfect calibration")

        # Model reliability curve
        valid = ~(
            np.isnan(reliability["predicted_probs"]) | np.isnan(reliability["actual_freqs"])
        )
        ax.plot(
            reliability["predicted_probs"][valid],
            reliability["actual_freqs"][valid],
            marker="o",
            linewidth=2,
            markersize=6,
            color=MANUSCRIPT_COLORS[0],
            label="Model",
        )

        # Add sample counts (only for bins with sufficient samples)
        for pred, actual, count in zip(
            reliability["predicted_probs"][valid],
            reliability["actual_freqs"][valid],
            reliability["counts"][valid],
        ):
            if count > 100:  # Only annotate bins with >100 samples
                ax.annotate(
                    f"n={int(count)}",
                    (pred, actual),
                    textcoords="offset points",
                    xytext=(4, -10),
                    fontsize=7,
                )

        ax.set_xlabel("Mean predicted probability", fontsize=10)
        ax.set_ylabel("Observed frequency", fontsize=10)
        # Simplified title to avoid text cutoff
        title_text = config['label'].replace("小时", "h").replace("km", " km")
        ax.set_title(f"{title_text}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        
        # Add metrics text
        metrics_text = f"ECE = {ece:.4f}\nBrier = {brier:.4f}"
        ax.text(
            0.98,
            0.05,
            metrics_text,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            transform=ax.transAxes,
        )
        format_axes(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # No overall title (removed as requested)
    plt.tight_layout(rect=[0, 0, 1, 1], pad=2.5, h_pad=3.0, w_pad=3.0)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"💾 Saved {output_path}")


def plot_station_distribution(metadata_path: Path, output_path: Path) -> None:
    """Create a static map of station locations."""
    if px is None:
        print("⚠️  plotly is not installed; skipping station map.")
        return
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata for map: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    df = pd.DataFrame(metadata)
    if not {"Latitude", "Longitude", "Stn Id", "Stn Name"}.issubset(df.columns):
        raise ValueError("Metadata file missing required columns for map plotting.")

    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        hover_name="Stn Name",
        hover_data={"Stn Id": True},
        color_discrete_sequence=[MANUSCRIPT_COLORS[0]],
        zoom=5,
        height=600,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=10, r=10, t=40, b=10),
        title="CIMIS station distribution",
        font=dict(family="DejaVu Sans", size=16),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), width=900, height=600, scale=2)
    print(f"💾 Saved {output_path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    apply_matplotlib_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    station_names = load_station_metadata(METADATA_PATH)
    (
        overall_counts,
        station_flag_counts,
        variable_flag_counts,
        station_totals,
        station_bad,
        variable_totals,
        variable_bad,
    ) = aggregate_qc_statistics(DATA_PATH)

    plot_overall_qc_distribution(
        overall_counts,
        OUTPUT_DIR / "qc_flag_distribution.png",
    )

    station_df = build_station_qc_dataframe(station_flag_counts, station_totals, station_names)
    plot_station_qc_stacks(
        station_df,
        OUTPUT_DIR / "qc_flag_distribution_per_station.png",
    )

    plot_bad_by_variable(
        variable_totals,
        variable_bad,
        OUTPUT_DIR / "qc_bad_by_variable.png",
    )

    plot_bad_by_station(
        station_totals,
        station_bad,
        station_names,
        OUTPUT_DIR / "qc_bad_by_station.png",
    )

    plot_matrix_model_comparison(
        BEST_MODEL_PATH,
        OUTPUT_DIR / "matrix_model_comparison.png",
    )

    # Generate 2x2 reliability diagram for all horizons
    # Optimal configurations: 3h@60km, 6h@160km, 12h@200km, 24h@180km
    reliability_paths = {
        "3h": PROJECT_ROOT / "experiments/lightgbm/raw/C/radius_60km/full_training/horizon_3h/predictions.json",
        "6h": PROJECT_ROOT / "experiments/lightgbm/raw/C/radius_160km/full_training/horizon_6h/predictions.json",
        "12h": PROJECT_ROOT / "experiments/lightgbm/raw/C/radius_200km/full_training/horizon_12h/predictions.json",
        "24h": PROJECT_ROOT / "experiments/lightgbm/raw/C/radius_180km/full_training/horizon_24h/predictions.json",
    }
    plot_reliability_diagram_2x2(
        reliability_paths,
        OUTPUT_DIR / "reliability_diagram_all_horizons.png",
    )
    
    # Also keep the single 3h diagram for backward compatibility
    plot_reliability_diagram(
        reliability_paths["3h"],
        OUTPUT_DIR / "reliability_diagram_3h.png",
    )

    # Skip station distribution if plotly/kaleido is not available
    try:
        plot_station_distribution(
            METADATA_PATH,
            OUTPUT_DIR / "station_distribution_map.png",
        )
    except Exception as e:
        print(f"⚠️  Skipping station distribution plot: {e}")

    print("\n✅ All manuscript figures refreshed with unified style.")


if __name__ == "__main__":
    main()

