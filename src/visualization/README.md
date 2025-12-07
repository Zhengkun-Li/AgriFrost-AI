# Visualization Module

可视化模块 (`src/visualization`) 提供模型预测和分析的可视化工具。

## 📁 模块结构

```
src/visualization/
├── __init__.py    # 模块导出
└── plots.py       # 绘图工具
```

## 🔧 核心组件

### Plotter (`plots.py`)

绘图工具类，支持 matplotlib 和 plotly 两种后端：

- **plot_predictions**: 预测值 vs 真实值对比图（包含残差图）
- **plot_feature_importance**: 特征重要性图
- **plot_metrics_comparison**: 模型指标对比图
- **plot_reliability_diagram**: 可靠性图（概率校准）

**关键特性**:
- ✅ 输入验证（空值检查、形状检查、范围检查）
- ✅ 错误处理（文件系统错误、绘图库错误）
- ✅ 支持两种后端（matplotlib 和 plotly）
- ✅ 优雅降级（依赖库不可用时抛出清晰错误）

**使用示例**:
```python
from src.visualization import Plotter
import numpy as np
import pandas as pd

# 创建绘图器
plotter = Plotter(style="matplotlib", figsize=(12, 6))

# 绘制预测对比图
plotter.plot_predictions(
    y_true=y_test,
    y_pred=predictions,
    dates=test_dates,
    title="Temperature Predictions",
    save_path="plots/predictions.png",
    show=False
)

# 绘制特征重要性
importance_df = pd.DataFrame({
    'feature': ['temp_lag_1h', 'humidity', 'wind_speed'],
    'importance': [0.3, 0.2, 0.1]
})
plotter.plot_feature_importance(
    importance=importance_df,
    top_n=10,
    save_path="plots/importance.png"
)

# 绘制可靠性图
plotter.plot_reliability_diagram(
    y_true=y_test_binary,
    y_proba=probabilities,
    n_bins=10,
    save_path="plots/reliability.png"
)
```

## ✅ 代码质量改进

### 已完成

1. **日志标准化** ✅
   - 添加了 `_logger`
   - 关键操作记录日志（调试、信息、错误）

2. **错误处理改进** ✅
   - 文件系统错误处理（IOError, OSError）
   - 绘图库错误处理
   - 清晰的错误消息

3. **输入验证** ✅
   - 数组空值检查和形状检查
   - DataFrame 列检查
   - 参数范围验证（top_n, n_bins, figsize）
   - 概率范围验证和裁剪

4. **数值稳定性** ✅
   - 概率裁剪（clip to [0, 1]）
   - 边界条件处理

## 📝 使用示例

### 预测对比图

```python
from src.visualization import Plotter
import numpy as np
import pandas as pd

plotter = Plotter(style="matplotlib")

# 时间序列预测图
dates = pd.date_range("2024-01-01", periods=100, freq="H")
y_true = np.random.randn(100) + 10
y_pred = y_true + np.random.randn(100) * 0.5

plotter.plot_predictions(
    y_true=y_true,
    y_pred=y_pred,
    dates=dates,
    title="Temperature Forecast",
    save_path="plots/forecast.png"
)
```

### 特征重要性

```python
importance_df = pd.DataFrame({
    'feature': ['feature_1', 'feature_2', 'feature_3'],
    'importance': [0.5, 0.3, 0.2]
})

plotter.plot_feature_importance(
    importance=importance_df,
    top_n=10,
    title="Top 10 Features",
    save_path="plots/importance.png"
)
```

### 可靠性图

```python
# 概率校准可视化
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.95, 0.15, 0.85, 0.75, 0.25, 0.9])

plotter.plot_reliability_diagram(
    y_true=y_true,
    y_proba=y_proba,
    n_bins=10,
    title="Frost Probability Calibration",
    save_path="plots/reliability.png"
)
```

## ⚠️ 注意事项

1. **依赖要求**:
   - `matplotlib` 后端需要 `matplotlib`
   - `plotly` 后端需要 `plotly` (建议版本 ≥ 4.12，但提供向后兼容)
   - 可靠性图需要 `src.evaluation.metrics.MetricsCalculator`

2. **输入验证**: 所有绘图方法都包含输入验证，确保参数有效性

3. **文件保存**: 保存路径的父目录会自动创建，失败时会记录错误并抛出异常

4. **概率范围**: 概率值会自动裁剪到 [0, 1]，并记录警告

## 📊 状态

**模块状态**: ✅ **生产就绪**

**最后更新**: 2025-11-19

所有关键改进已完成：
- ✅ 日志标准化
- ✅ 错误处理
- ✅ 输入验证
- ✅ 数值稳定性

### 最新改进（2025-11-19）

**修复的问题**:
- ✅ **移除 seaborn 依赖**（matplotlib 不再需要 seaborn）
- ✅ **修复资源泄漏**（使用 `plt.close(fig)` 而不是 `plt.close()`）
- ✅ **Plotly 向后兼容**（`add_hline()` 添加 fallback 支持旧版本）
- ✅ **性能优化**（残差图使用小点标记处理大数据集）
- ✅ **自动布局**（metrics comparison 支持多行布局避免挤压）
- ✅ **NaN 处理**（feature importance 和 reliability diagram 增强 NaN 检查）
- ✅ **标签优化**（长模型名 X 轴标签旋转和截断）
- ✅ **特征名截断**（避免长特征名导致图表变形）

