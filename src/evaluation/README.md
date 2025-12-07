# Evaluation Module

评估模块 (`src/evaluation`) 提供模型评估指标和交叉验证策略。

## 📁 模块结构

```
src/evaluation/
├── __init__.py                        # 模块导出
├── metrics.py                         # 评估指标计算
├── validators.py                      # 交叉验证策略
├── registry.py                        # 评估策略注册表
├── multi_horizon_evaluator.py         # 多时间窗口评估
├── matrix_evaluator.py                # 2×2+1 矩阵评估
└── spatial_sensitivity_evaluator.py   # 空间参数敏感性分析
```

## 🔧 核心组件

### 1. MetricsCalculator (`metrics.py`)

评估指标计算器，支持回归和分类任务：

- **回归指标**: MAE, RMSE, R², MAPE
- **分类指标**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score
- **概率指标**: Brier Score, ROC-AUC, PR-AUC, ECE (Expected Calibration Error)
- **可靠性图**: 用于可视化概率校准

**关键特性**:
- ✅ 输入验证（空值检查、形状检查、范围检查）
- ✅ 数值稳定性（概率裁剪、非有限值检测）
- ✅ 优雅降级（sklearn 不可用时使用 fallback）
- ✅ 错误处理（区分可恢复错误和意外错误）

**使用示例**:
```python
from src.evaluation import MetricsCalculator
import numpy as np

# 回归指标
regression_metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)

# 分类指标
classification_metrics = MetricsCalculator.calculate_classification_metrics(
    y_true, y_pred, y_proba=y_proba
)

# 概率指标（包含 ECE）
prob_metrics = MetricsCalculator.calculate_probability_metrics(y_true, y_proba)

# 可靠性图数据
reliability_data = MetricsCalculator.calculate_reliability_data(y_true, y_proba, n_bins=10)
```

### 2. CrossValidator (`validators.py`)

交叉验证策略，支持时间序列和分组数据：

- **time_split**: 时间序列划分（train/val/test），支持按时间断点分割
- **leave_one_station_out**: Leave-One-Station-Out (LOSO)，带时间排序和泄漏防护
- **station_time_split**: 站点内时间分割，适用于 C/D 轨道
- **group_kfold**: Group K-Fold（不适用于空间泛化任务）
- **time_series_split**: Time Series Split (sklearn)

**关键特性**:
- ✅ 输入验证（DataFrame 空值检查、列检查、参数验证）
- ✅ **严格时间排序**（LOSO 和 time_split 都包含 date + hour 排序）
- ✅ **时间泄漏防护**（验证 train_max_date < test_min_date）
- ✅ **站点隔离验证**（LOSO 确保 train 和 test 无相同站点）
- ✅ 边界条件处理（确保每个 split 都有数据）
- ✅ 错误处理（清晰的错误消息）
- ✅ 调试日志（记录每个 fold 的样本数）

**使用示例**:
```python
from src.evaluation import CrossValidator

# 时间序列划分
train_df, val_df, test_df = CrossValidator.time_split(
    df, train_ratio=0.7, val_ratio=0.15, date_col="Date"
)

# LOSO 交叉验证
loso_splits = CrossValidator.leave_one_station_out(df, station_col="Stn Id")
for train_df, test_df in loso_splits:
    # Train and evaluate
    pass

# Group K-Fold
group_splits = CrossValidator.group_kfold(df, n_splits=5, group_col="Stn Id")

# Time Series Split
ts_splits = CrossValidator.time_series_split(df, n_splits=5, date_col="Date")
```

### 3. Evaluation Registry (`registry.py`)

评估策略注册表，用于动态注册和检索评估策略：

- **register_evaluation_strategy**: 注册评估策略
- **get_evaluation_handler**: 获取评估处理器

**关键特性**:
- ✅ 输入验证（名称和处理器验证）
- ✅ **支持参数化策略**（handler 支持 `*args, **kwargs`，如 `radius_km`）
- ✅ 重复注册警告
- ✅ 清晰的错误消息（列出可用策略）

### 4. MultiHorizonEvaluator (`multi_horizon_evaluator.py`)

多时间窗口评估器，用于跨多个预报窗口（3h, 6h, 12h, 24h）聚合和分析结果：

- **evaluate**: 评估所有时间窗口的结果
- **聚合指标**: 自动计算 mean, std, min, max 跨时间窗口
- **最佳时间窗口**: 自动找出最佳预报窗口

**关键特性**:
- ✅ 支持新格式（classification/regression）和旧格式（frost_metrics/temp_metrics）
- ✅ 自动聚合指标（classification 和 regression 分别聚合）
- ✅ 最佳时间窗口查找（基于综合评分）
- ✅ 自动保存 JSON 结果

**使用示例**:
```python
from src.evaluation import MultiHorizonEvaluator

evaluator = MultiHorizonEvaluator(horizons=[3, 6, 12, 24])
results = evaluator.evaluate(results_dict, model_name="lightgbm")
# Returns: {
#   "horizons": {"3h": {...}, "6h": {...}, ...},
#   "summary": {"classification": {...}, "regression": {...}},
#   "best_horizon": {"horizon": 3, "metrics": {...}}
# }
```

### 5. MatrixEvaluator (`matrix_evaluator.py`)

2×2+1 矩阵评估器，用于跨所有矩阵单元（A, B, C, D, E）的比较和汇总：

- **evaluate**: 评估所有矩阵单元的结果
- **矩阵汇总**: 自动对比所有单元，找出最佳单元
- **Insights 生成**: 自动生成 insights（raw vs FE, single vs multi-station）

**关键特性**:
- ✅ 支持完整的 2×2+1 矩阵框架（A/B/C/D/E）
- ✅ 自动单元对比（classification 和 regression 指标）
- ✅ 最佳单元查找（基于综合评分）
- ✅ 每个时间窗口的最佳单元分析
- ✅ 自动生成 Markdown 汇总报告

**使用示例**:
```python
from src.evaluation import MatrixEvaluator

matrix_eval = MatrixEvaluator(
    matrix_cells=["A", "B", "C", "D", "E"],
    horizons=[3, 6, 12, 24]
)
results = matrix_eval.evaluate(matrix_results_dict, model_type="lightgbm")
# Returns: {
#   "cells": {"A": {...}, "B": {...}, ...},
#   "matrix_summary": {"best_cell": {...}, "comparison": {...}, "insights": [...]},
#   "horizon_analysis": {"3h": {...}, ...}
# }
```

### 6. SpatialSensitivityEvaluator (`spatial_sensitivity_evaluator.py`)

空间参数敏感性评估器，用于分析空间聚合参数（radius_km, k_neighbors）的影响：

- **evaluate**: 评估不同参数值的结果
- **趋势分析**: 自动分析指标趋势（increasing, decreasing, stable, mixed）
- **最优参数查找**: 自动找出最优空间参数

**关键特性**:
- ✅ 支持多种参数类型（radius_km, k_neighbors）
- ✅ 默认参数值（radius: [25, 50, 75, 100], k: [1, 3, 5, 7, 10]）
- ✅ 自动趋势分析（参数值对指标的影响）
- ✅ 每个时间窗口的最佳参数分析
- ✅ 自动生成 Markdown 汇总报告

**使用示例**:
```python
from src.evaluation import SpatialSensitivityEvaluator

sensitivity_eval = SpatialSensitivityEvaluator(
    param_name="radius_km",
    param_values=[25, 50, 75, 100],
    horizons=[3, 6, 12, 24]
)
results = sensitivity_eval.evaluate(radius_results_dict, model_name="lightgbm")
# Returns: {
#   "parameters": {"25": {...}, "50": {...}, ...},
#   "sensitivity_analysis": {"optimal_parameter": {...}, "insights": [...]},
#   "horizon_analysis": {"3h": {...}, ...}
# }
```

**使用示例**:
```python
from src.evaluation.registry import register_evaluation_strategy, get_evaluation_handler

# 注册自定义策略
def my_evaluation_handler(runner, dataset, params):
    # Custom evaluation logic
    pass

register_evaluation_strategy("my_strategy", my_evaluation_handler)

# 获取处理器
handler = get_evaluation_handler("my_strategy")
handler(runner, dataset, params)
```

## ✅ 代码质量改进

### 已完成

1. **日志标准化** ✅
   - 所有模块添加了 `_logger`
   - 关键操作记录日志（调试、警告、错误）
   - 文件: metrics.py, validators.py, registry.py

2. **错误处理改进** ✅
   - 使用具体异常类型（`ValueError`, `ImportError`）
   - 区分可恢复错误和意外错误
   - 清晰的错误消息（包含可用选项）

3. **输入验证** ✅
   - **metrics.py**: 空值检查、形状检查、范围检查、非有限值检测
   - **validators.py**: DataFrame 空值检查、列检查、参数验证（ratios, n_splits）
   - **registry.py**: 名称和处理器验证

4. **数值稳定性** ✅
   - 概率裁剪（clip to [0, 1]）
   - 非有限值检测和警告
   - 边界条件处理（空 split 检测）

## 📝 使用示例

### 评估指标

```python
from src.evaluation import MetricsCalculator
import numpy as np

# 回归指标
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
print(metrics)  # {'mae': 0.08, 'rmse': 0.09, 'r2': 0.99}

# 分类指标
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 0, 1])
y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.95])
metrics = MetricsCalculator.calculate_classification_metrics(y_true, y_pred, y_proba)
print(metrics)  # {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, ...}

# 概率校准指标（ECE）
prob_metrics = MetricsCalculator.calculate_probability_metrics(y_true, y_proba)
print(prob_metrics)  # {'brier_score': 0.05, 'roc_auc': 1.0, 'ece': 0.02}
```

### 交叉验证

```python
from src.evaluation import CrossValidator
import pandas as pd

# 时间序列划分
df = pd.DataFrame({
    "Date": pd.date_range("2020-01-01", periods=1000, freq="H"),
    "value": np.random.randn(1000)
})

train_df, val_df, test_df = CrossValidator.time_split(
    df, train_ratio=0.7, val_ratio=0.15, date_col="Date"
)

# LOSO 交叉验证
df_stations = pd.DataFrame({
    "Stn Id": [1, 1, 2, 2, 3, 3],
    "value": [1, 2, 3, 4, 5, 6]
})

loso_splits = CrossValidator.leave_one_station_out(df_stations, station_col="Stn Id")
for train_df, test_df in loso_splits:
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
```

## ⚠️ 注意事项

1. **依赖要求**:
   - `metrics.py` 需要 `sklearn`（可选，有 fallback）
   - `validators.py` 需要 `sklearn`（必需，用于 GroupKFold 和 TimeSeriesSplit）

2. **输入验证**: 所有公共函数都包含输入验证，确保参数有效性

3. **概率范围**: 概率值会自动裁剪到 [0, 1]，并记录警告

4. **空值处理**: 如果无法计算某个指标（例如，只有一类标签），会返回 `np.nan` 并记录调试日志

## 📊 状态

**模块状态**: ✅ **生产就绪**

**最后更新**: 2025-11-19

所有关键改进已完成：
- ✅ 日志标准化
- ✅ 错误处理
- ✅ 输入验证
- ✅ 数值稳定性
- ✅ 多任务模型支持
- ✅ 时间泄漏防护
- ✅ 高级评估功能（multi-horizon, matrix, spatial sensitivity）

