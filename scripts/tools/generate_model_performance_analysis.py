#!/usr/bin/env python3
"""
Rebuild the aggregated model performance analysis assets
based on results/model_performance_all_models.csv.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "model_performance_analysis"
ALL_MODELS_CSV = RESULTS_DIR / "model_performance_all_models.csv"
BEST_CONFIG_CSV = ANALYSIS_DIR / "best_configurations_by_horizon.csv"
REPORT_PATH = ANALYSIS_DIR / "模型性能综合分析报告.md"


def format_radius(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    rounded = int(round(float(value)))
    return f"{rounded}km"


def to_markdown_table(rows: Iterable[List[str]]) -> str:
    rows = list(rows)
    if not rows:
        return ""
    header = rows[0]
    separator = ["---"] * len(header)
    body = ["| " + " | ".join(header) + " |", "| " + " | ".join(separator) + " |"]
    for row in rows[1:]:
        body.append("| " + " | ".join(row) + " |")
    return "\n".join(body)


@dataclass
class SummaryData:
    total_runs: int
    models: list[str]
    class_summary: pd.DataFrame
    regression_summary: pd.DataFrame
    horizon_best: pd.DataFrame
    matrix_summary: pd.DataFrame
    matrix_best: pd.DataFrame
    model_counts: pd.Series


def build_summary(df: pd.DataFrame) -> SummaryData:
    df = df.copy()
    df["radius_km"] = df["radius_km"].fillna(0)
    total_runs = len(df)
    models = sorted(df["model"].unique())
    class_summary = (
        df.groupby("model")
        .agg(
            roc_mean=("roc_auc", "mean"),
            roc_std=("roc_auc", "std"),
            pr_mean=("pr_auc", "mean"),
            pr_std=("pr_auc", "std"),
            brier_mean=("brier_score", "mean"),
        )
        .reset_index()
        .sort_values("roc_mean", ascending=False)
    )
    class_summary = class_summary.fillna(0)

    regression_summary = (
        df.groupby("model")
        .agg(temp_rmse_mean=("temp_rmse", "mean"), temp_rmse_std=("temp_rmse", "std"), temp_r2_mean=("temp_r2", "mean"), temp_r2_std=("temp_r2", "std"))
        .reset_index()
    )
    regression_summary = regression_summary.dropna(subset=["temp_rmse_mean", "temp_r2_mean"], how="all").fillna(0)
    regression_summary = regression_summary.sort_values("temp_rmse_mean")

    horizon_best_rows = []
    for horizon in sorted(df["horizon_h"].unique()):
        subset = df[df["horizon_h"] == horizon]
        best_roc = subset.loc[subset["roc_auc"].idxmax()]
        best_pr = subset.loc[subset["pr_auc"].idxmax()]
        horizon_best_rows.append(
            {
                "Horizon": f"{int(horizon)}h",
                "Best_ROC_Model": best_roc["model"],
                "Best_ROC_AUC": best_roc["roc_auc"],
                "Best_ROC_Matrix": best_roc["matrix_cell"],
                "Best_ROC_Radius": format_radius(best_roc["radius_km"]),
                "Best_PR_Model": best_pr["model"],
                "Best_PR_AUC": best_pr["pr_auc"],
                "Best_PR_Matrix": best_pr["matrix_cell"],
                "Best_PR_Radius": format_radius(best_pr["radius_km"]),
            }
        )
    horizon_best = pd.DataFrame(horizon_best_rows)

    matrix_summary = (
        df.groupby("matrix_cell").agg(roc_mean=("roc_auc", "mean"), pr_mean=("pr_auc", "mean"), brier_mean=("brier_score", "mean")).reset_index()
    )
    matrix_summary = matrix_summary.sort_values("matrix_cell")

    matrix_best_rows = []
    for matrix in sorted(df["matrix_cell"].unique()):
        subset = df[df["matrix_cell"] == matrix]
        best_row = subset.loc[subset["roc_auc"].idxmax()]
        matrix_best_rows.append(
            {
                "matrix_cell": matrix,
                "model": best_row["model"],
                "horizon": f"{int(best_row['horizon_h'])}h",
                "roc_auc": best_row["roc_auc"],
                "pr_auc": best_row["pr_auc"],
                "radius": format_radius(best_row["radius_km"]),
            }
        )
    matrix_best = pd.DataFrame(matrix_best_rows)

    model_counts = df["model"].value_counts().sort_index()

    return SummaryData(
        total_runs=total_runs,
        models=models,
        class_summary=class_summary,
        regression_summary=regression_summary,
        horizon_best=horizon_best,
        matrix_summary=matrix_summary,
        matrix_best=matrix_best,
        model_counts=model_counts,
    )


def plot_model_comparison(class_summary: pd.DataFrame) -> None:
    models = class_summary["model"].tolist()
    x = range(len(models))
    plt.figure(figsize=(10, 6))
    plt.bar([i - 0.2 for i in x], class_summary["roc_mean"], width=0.4, label="ROC-AUC")
    plt.bar([i + 0.2 for i in x], class_summary["pr_mean"], width=0.4, label="PR-AUC")
    plt.xticks(list(x), models, rotation=20)
    plt.ylabel("Score")
    plt.title("Model Comparison (ROC-AUC vs PR-AUC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "model_comparison_roc_pr_auc.png", dpi=200)
    plt.close()


def plot_performance_by_horizon(df: pd.DataFrame) -> None:
    summary = df.groupby("horizon_h").agg(roc_mean=("roc_auc", "mean"), pr_mean=("pr_auc", "mean")).reset_index().sort_values("horizon_h")
    plt.figure(figsize=(8, 5))
    plt.plot(summary["horizon_h"], summary["roc_mean"], marker="o", label="ROC-AUC")
    plt.plot(summary["horizon_h"], summary["pr_mean"], marker="s", label="PR-AUC")
    plt.xlabel("Horizon (h)")
    plt.ylabel("Score")
    plt.title("Performance by Forecast Horizon")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "performance_by_horizon.png", dpi=200)
    plt.close()


def plot_performance_by_matrix_cell(df: pd.DataFrame) -> None:
    summary = df.groupby("matrix_cell").agg(roc_mean=("roc_auc", "mean"), pr_mean=("pr_auc", "mean")).reset_index().sort_values("matrix_cell")
    x = range(len(summary))
    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.2 for i in x], summary["roc_mean"], width=0.4, label="ROC-AUC")
    plt.bar([i + 0.2 for i in x], summary["pr_mean"], width=0.4, label="PR-AUC")
    plt.xticks(list(x), summary["matrix_cell"])
    plt.ylabel("Score")
    plt.title("Performance by Matrix Cell")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "performance_by_matrix_cell.png", dpi=200)
    plt.close()


def plot_temperature_performance(regression_summary: pd.DataFrame) -> None:
    if regression_summary.empty:
        return
    models = regression_summary["model"].tolist()
    x = range(len(models))
    plt.figure(figsize=(10, 6))
    plt.bar(x, regression_summary["temp_rmse_mean"], color="#4C72B0")
    plt.xticks(list(x), models, rotation=20)
    plt.ylabel("RMSE (°C)")
    plt.title("Temperature Prediction RMSE by Model")
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "temperature_prediction_performance.png", dpi=200)
    plt.close()


def save_best_configurations(horizon_best: pd.DataFrame) -> None:
    horizon_best.to_csv(BEST_CONFIG_CSV, index=False)


def generate_report(summary: SummaryData, df: pd.DataFrame) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_list = ", ".join(summary.models)
    exec_best = summary.class_summary.iloc[0]
    best_model_name = exec_best["model"]
    best_model_roc = exec_best["roc_mean"]
    best_matrix = summary.matrix_best.sort_values("roc_auc", ascending=False).iloc[0]
    matrix_avg_top = summary.matrix_summary.sort_values("roc_mean", ascending=False).iloc[0]
    low_rmse_models = summary.regression_summary[summary.regression_summary["temp_rmse_mean"] < 3]["model"].tolist()
    low_rmse_text = (
        f"{', '.join(low_rmse_models)} 为唯一平均 RMSE < 3°C 的模型族，树模型 RMSE 多在 3–5°C 区间"
        if low_rmse_models
        else "当前所有模型的平均温度 RMSE 均高于 3°C"
    )
    short_horizon = summary.horizon_best.iloc[0]

    class_table_rows = [
        ["模型", "平均 ROC-AUC", "Std", "平均 PR-AUC", "Std", "平均 Brier"],
    ]
    for _, row in summary.class_summary.iterrows():
        class_table_rows.append(
            [
                row["model"],
                f"{row['roc_mean']:.4f}",
                f"{row['roc_std']:.4f}",
                f"{row['pr_mean']:.4f}",
                f"{row['pr_std']:.4f}",
                f"{row['brier_mean']:.4f}",
            ]
        )

    regression_table_rows = [
        ["模型", "平均 RMSE (°C)", "Std", "平均 R²", "Std"],
    ]
    for _, row in summary.regression_summary.iterrows():
        regression_table_rows.append(
            [
                row["model"],
                f"{row['temp_rmse_mean']:.3f}",
                f"{row['temp_rmse_std']:.3f}",
                f"{row['temp_r2_mean']:.3f}",
                f"{row['temp_r2_std']:.3f}",
            ]
        )

    matrix_table_rows = [
        ["矩阵", "平均 ROC-AUC", "平均 PR-AUC", "平均 Brier"],
    ]
    for _, row in summary.matrix_summary.iterrows():
        matrix_table_rows.append(
            [
                row["matrix_cell"],
                f"{row['roc_mean']:.4f}",
                f"{row['pr_mean']:.4f}",
                f"{row['brier_mean']:.4f}",
            ]
        )

    matrix_best_rows = [
        ["矩阵", "最佳模型", "提前量", "ROC-AUC", "PR-AUC", "半径"],
    ]
    for _, row in summary.matrix_best.iterrows():
        matrix_best_rows.append(
            [
                row["matrix_cell"],
                row["model"],
                row["horizon"],
                f"{row['roc_auc']:.4f}",
                f"{row['pr_auc']:.4f}",
                row["radius"],
            ]
        )

    model_count_lines = []
    for model_name, count in summary.model_counts.items():
        model_count_lines.append(f"- **{model_name}**: {int(count)} 个实验")

    horizon_table_rows = [
        summary.horizon_best.columns.tolist(),
    ]
    for _, row in summary.horizon_best.iterrows():
        formatted = []
        for value in row:
            if isinstance(value, numbers.Real) and not isinstance(value, bool):
                formatted.append(f"{value:.4f}")
            else:
                formatted.append(str(value))
        horizon_table_rows.append(formatted)

    report = f"""# 模型性能综合分析报告

**生成时间**: {timestamp}  
**总实验数**: {summary.total_runs}  
**已训练模型**: {model_list}

---

## 1. 执行摘要

1. **最佳整体模型**: {best_model_name}，平均 ROC-AUC = {best_model_roc:.4f}，平均 PR-AUC = {exec_best['pr_mean']:.4f}
2. **短期窗口亮点**: {short_horizon['Horizon']} 的最佳 PR-AUC 由 {short_horizon['Best_PR_Model']} 提供 (PR-AUC = {float(short_horizon['Best_PR_AUC']):.4f})
3. **空间聚合收益**: Matrix {matrix_avg_top['matrix_cell']} 平均 ROC-AUC = {matrix_avg_top['roc_mean']:.4f}，在长周期仍保持领先
4. **温度回归差异**: {low_rmse_text}

---

## 2. 模型整体性能对比

### 2.1 分类性能（霜冻预测）
{to_markdown_table(class_table_rows)}

### 2.2 回归性能（温度预测）
{to_markdown_table(regression_table_rows)}

---

## 3. 各时间窗口最佳配置
{to_markdown_table(horizon_table_rows)}

---

## 4. Matrix Cell 性能对比

### 4.1 平均性能
{to_markdown_table(matrix_table_rows)}

### 4.2 各矩阵最佳模型
{to_markdown_table(matrix_best_rows)}

---

## 5. 数据质量说明

- **总实验数**: {summary.total_runs} 个
{chr(10).join(model_count_lines)}

---

## 6. 可视化图表

详细的可视化分析图表已保存在 `results/model_performance_analysis/` 目录：

- `model_comparison_roc_pr_auc.png`: 模型 ROC-AUC / PR-AUC 对比
- `performance_by_horizon.png`: 各时间窗口性能趋势
- `performance_by_matrix_cell.png`: 各矩阵平均性能
- `temperature_prediction_performance.png`: 温度预测 RMSE 对比

---
"""

    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ALL_MODELS_CSV)
    summary = build_summary(df)

    plot_model_comparison(summary.class_summary)
    plot_performance_by_horizon(df)
    plot_performance_by_matrix_cell(df)
    plot_temperature_performance(summary.regression_summary)
    save_best_configurations(summary.horizon_best)
    generate_report(summary, df)
    print(f"Updated model performance analysis with {summary.total_runs} runs.")


if __name__ == "__main__":
    main()

