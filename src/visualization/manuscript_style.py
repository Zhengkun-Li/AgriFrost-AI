"""Shared styling utilities for manuscript-quality figures."""

from __future__ import annotations

from typing import Iterable, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Core palette used across manuscript figures (dark-to-bright ordering)
MANUSCRIPT_COLORS: List[str] = [
    "#1b4965",  # deep blue
    "#ca3c25",  # muted red
    "#2a9d8f",  # teal
    "#ffba08",  # amber
    "#6a4c93",  # purple
    "#386641",  # green
]

# Dedicated palette for QC status breakdowns
STATUS_COLOR_MAP = {
    "PASS": "#1b4965",
    "Y": "#2a9d8f",
    "Q": "#ca3c25",
    "P": "#ffba08",
    "M": "#6a4c93",
    "R": "#bc4749",
    "S": "#4c4a59",
    "OTHER": "#adb5bd",
}


def apply_matplotlib_style() -> None:
    """Apply a consistent, publication-friendly matplotlib/seaborn style."""
    sns.set_theme(
        context="notebook",
        style="ticks",
        palette=MANUSCRIPT_COLORS,
        font="DejaVu Sans",
        rc={
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 16,
            "figure.facecolor": "white",
        },
    )

    mpl.rcParams.update(
        {
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "axes.prop_cycle": mpl.cycler(color=MANUSCRIPT_COLORS),
            "font.size": 11,
        }
    )


def format_axes(ax: plt.Axes) -> None:
    """Apply shared tick and spine formatting to the provided axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_facecolor("white")


def get_status_colors(statuses: Iterable[str]) -> List[str]:
    """Return colors for the requested QC statuses."""
    return [STATUS_COLOR_MAP.get(status, STATUS_COLOR_MAP["OTHER"]) for status in statuses]

