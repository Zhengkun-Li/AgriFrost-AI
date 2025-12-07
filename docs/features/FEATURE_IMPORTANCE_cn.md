# AgriFrost-AI 特征重要性完整指南

<div align="center">

<img src="../logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

</div>

**最后更新**: 2025-11-20

本文档整合了特征重要性评估、模型特定性说明、表示方法选择和特征选择策略等所有相关内容，为特征重要性分析提供一站式参考。

## 📋 目录

1. [概述](#概述)
2. [特征重要性评估方法](#特征重要性评估方法)
3. [特征重要性可视化](#特征重要性可视化)
4. [特征重要性的本质](#特征重要性的本质)
5. [特征重要性表示方法](#特征重要性表示方法)
6. [特征选择策略](#特征选择策略)
7. [分析方法](#分析方法)
8. [注意事项](#注意事项)

---

## 概述

特征重要性（Feature Importance）是理解模型决策过程的关键工具。本指南介绍如何从训练好的模型中提取、分析和可视化特征重要性，并指导特征选择策略。

### ⚠️ **重要概念**

**特征重要性是针对特定模型的，不是针对数据集的**

特征重要性反映的是**模型如何使用特征进行预测**，而不是特征在数据集中的固有重要性。

---

## 特征重要性评估方法

### 1. **自动保存（训练时）**

训练完成后，特征重要性会自动保存到模型目录：

```
experiments/lightgbm/raw/A/full_training/full_training/horizon_12h/
  ├── frost_feature_importance.csv    # 霜冻分类模型的特征重要性
  └── temp_feature_importance.csv     # 温度回归模型的特征重要性
```

**CSV格式**：
```csv
feature,importance,importance_pct,cumulative_pct
Air Temp (C),1234.56,15.23,15.23
Dew Point (C),987.65,12.18,27.41
Soil Temp (C),876.54,10.81,38.22
...
```

**列说明**：
- `feature`: 特征名称
- `importance`: 原始重要性分数
- `importance_pct`: 重要性百分比
- `cumulative_pct`: 累积重要性百分比

### 2. **使用CLI命令分析**

使用 `analysis feature-importance` 命令提取和分析特征重要性：

```bash
# 分析frost和temp模型
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h

# 只分析frost分类模型
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h \
    --task frost

# 保存到指定目录并生成图表
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h \
    --output-dir results/feature_importance \
    --plot \
    --top-k 20
```

**参数说明**：
- `--model-dir`: 训练好的模型目录路径
- `--task`: 要分析的任务（`frost`, `temp`, 或 `both`）
- `--output-dir`: 输出目录（默认：`model_dir/feature_importance`）
- `--top-k`: 只显示前K个最重要的特征
- `--plot`: 生成特征重要性图表
- `--format`: 输出格式（`csv`, `json`, 或 `both`）

**输出**：
- CSV/JSON文件：特征重要性数据
- PNG图表：特征重要性可视化（百分比和原始值两种格式）
- 比较图表：frost vs temp 特征重要性对比（如果两个任务都分析）

---

## 特征重要性可视化

### 1. **单个模型的特征重要性**

```python
from pathlib import Path
import pandas as pd
from src.visualization.plots import Plotter

# 读取特征重要性数据
importance_df = pd.read_csv("experiments/lightgbm/raw/A/full_training/full_training/horizon_12h/frost_feature_importance.csv")

# 创建图表（百分比）
plotter = Plotter(style="matplotlib", figsize=(12, 8))
plotter.plot_feature_importance(
    importance_df,
    top_n=20,
    title="Feature Importance - Frost Classification (12h) (%)",
    save_path="feature_importance_pct.png",
    show=False,
    importance_col='importance_pct',
    xlabel='Importance (%)'
)

# 创建图表（原始值）
plotter.plot_feature_importance(
    importance_df,
    top_n=20,
    title="Feature Importance - Frost Classification (12h) (Raw Values)",
    save_path="feature_importance_raw.png",
    show=False,
    importance_col='importance',
    xlabel='Importance (Raw Value)'
)
```

### 2. **比较Frost vs Temp特征重要性**

使用CLI命令自动生成对比图表：

```bash
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h \
    --task both \
    --plot
```

这会在输出目录中生成：
- `frost_feature_importance_pct.png`: Frost分类模型的特征重要性（百分比）
- `frost_feature_importance_raw.png`: Frost分类模型的特征重要性（原始值）
- `temp_feature_importance_pct.png`: Temp回归模型的特征重要性（百分比）
- `temp_feature_importance_raw.png`: Temp回归模型的特征重要性（原始值）
- `frost_temp_importance_comparison_pct.png`: 对比图表（百分比）
- `frost_temp_importance_comparison_raw.png`: 对比图表（原始值）

---

## 特征重要性的本质

### 1. **模型特定（Model-Specific）**

特征重要性取决于：
- **模型类型**（LightGBM, XGBoost, Linear等）
- **模型参数**（超参数设置）
- **训练过程**（训练数据、训练策略）
- **模型是否已被训练（fitted）**

### 2. **为什么是模型特定的？**

#### **Tree-based模型（LightGBM, XGBoost, RandomForest）**

```python
# LightGBM特征重要性示例
# 重要性 = 特征在决策树中使用的频率 × 带来的信息增益

特征重要性 = Σ(每个节点使用该特征带来的信息增益)
```

- 不同的树结构 → 不同的特征重要性
- 不同的超参数（如`max_depth`, `num_leaves`）→ 不同的树结构 → 不同的特征重要性
- 不同的训练数据 → 不同的树结构 → 不同的特征重要性

#### **Linear模型（Linear Regression, Logistic Regression）**

```python
# Linear模型特征重要性
# 重要性 = |系数| (coefficient magnitude)

特征重要性 = |coefficient|
```

- 不同的模型训练结果 → 不同的系数 → 不同的特征重要性
- 特征之间的相关性会影响系数大小

#### **Deep Learning模型（LSTM, GRU, TCN）**

- 通常不直接提供特征重要性
- 如果使用attention机制，可以通过attention权重作为重要性
- 需要使用permutation importance等替代方法

### 3. **实际例子**

#### **场景：同一个数据集，不同的模型**

假设我们有同一个数据集，训练了三个不同的模型：

| 特征 | LightGBM重要性 | XGBoost重要性 | Linear重要性 |
|------|---------------|---------------|--------------|
| Air Temp (C) | 20.09% | 18.5% | 35.2% |
| Dew Point (C) | 13.03% | 14.2% | 22.1% |
| Soil Temp (C) | 11.77% | 12.8% | 15.3% |

**为什么不同？**
- LightGBM使用梯度提升，特征重要性基于信息增益
- XGBoost使用不同的优化算法，可能生成不同的树结构
- Linear Regression使用系数，受特征相关性和标准化影响

#### **场景：同一个模型，不同的horizon**

```python
# 模型1：LightGBM (3h horizon)
# 模型2：LightGBM (12h horizon)
# 模型3：LightGBM (24h horizon)
```

**结果**：
- 3h horizon：可能更依赖当前时刻的特征（如Air Temp）
- 24h horizon：可能更依赖趋势性特征（如Hour, Julian Day）

**特征重要性会随horizon变化！**

### 4. **数据集级别的特征重要性（替代方法）**

如果你想获得**数据集级别的特征重要性**（不依赖特定模型），可以使用：

#### **Permutation Importance（置换重要性）**

```python
from sklearn.inspection import permutation_importance

# 对测试集计算permutation importance
perm_importance = permutation_importance(
    model, 
    X_test, 
    y_test, 
    n_repeats=10, 
    random_state=42
)
```

**特点**：
- 基于模型性能变化
- 不依赖模型内部结构
- 可以跨模型比较
- 计算成本较高

#### **SHAP Values**

```python
import shap

# 计算SHAP值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

**特点**：
- 基于博弈论
- 可以解释单个样本的预测
- 可以可视化
- 计算成本较高

#### **特征相关性分析**

```python
# 计算特征与目标的相关性
correlations = df.corr()[target_column].sort_values(ascending=False)
```

**特点**：
- 基于数据集本身
- 不依赖模型
- 只反映线性关系
- 不考虑特征交互

---

## 特征重要性表示方法

### **百分比（Percentage）更常见和推荐**

在机器学习和数据科学领域，**百分比**是表示特征重要性的更常见和推荐的方式。

### **为什么百分比更常见？**

#### **1. 易于理解和解释**

**百分比**：
```
特征重要性：
- Air Temp (C): 20.09%
- Soil Temp (C): 13.03%
- Wind Speed (m/s): 8.44%
```

✅ **优点**：
- 直观易懂：20.09% 意味着该特征贡献了总重要性的约1/5
- 容易比较：可以直接看出哪个特征更重要
- 不依赖数值范围：不受模型类型或超参数影响

**原始数值**：
```
特征重要性：
- Air Temp (C): 2487.0
- Soil Temp (C): 1613.0
- Wind Speed (m/s): 907.0
```

❌ **缺点**：
- 数值范围可能很大，难以理解（如2487.0意味着什么？）
- 不同模型类型的数值范围可能差异很大
- 难以直观比较

### **不同场景的推荐**

| 场景 | 推荐使用 | 原因 |
|------|---------|------|
| **可视化图表** | 百分比 | 更直观，易于理解 |
| **CSV文件** | 两者都保留 | 满足不同需求 |
| **论文和报告** | 百分比 | 更专业，标准化 |
| **技术文档** | 两者都提供 | 详细和完整 |
| **跨模型比较** | 百分比或归一化值 | 统一标准 |

### **当前实现的建议**

**当前CSV格式（推荐）**：

```csv
feature,importance,importance_pct,cumulative_pct
Air Temp (C),2487.0,20.09,20.09
Soil Temp (C),1613.0,13.03,33.12
Wind Speed (m/s),907.0,7.33,40.45
```

**优点**：
- ✅ 保留了原始数值（用于深度分析）
- ✅ 提供了百分比（用于理解和可视化）
- ✅ 提供了累积百分比（用于特征选择）

**可视化图表**：
- 生成两种格式的图表：百分比和原始值
- 分别保存为 `_pct.png` 和 `_raw.png`

---

## 特征选择策略

### **两阶段特征选择方法**

#### **阶段 1: 全特征训练（基准）**

1. **创建所有特征**（~298 个）
   - 使用完整的特征工程配置
   - 确保所有特征都被创建
   - 验证特征数量达到预期

2. **训练模型，获得基准性能**
   - 使用所有特征训练模型
   - 记录性能指标（ROC-AUC, PR-AUC, MAE, RMSE, R²）
   - 作为后续优化的基准

3. **分析特征重要性**
   - 提取特征重要性
   - 计算累积重要性
   - 识别最重要的特征

#### **阶段 2: 基于重要性重新训练（优化）**

1. **选择累积重要性占 90% 的特征**
   - 根据特征重要性分析结果
   - 选择累积重要性达到 90% 的特征
   - 可能只需要前 50-200 个特征（取决于重要性分布）

2. **使用这些特征重新训练**
   - 使用选定的特征重新训练模型
   - 对比性能提升或下降
   - 评估计算成本降低

3. **优化性能和成本**
   - 平衡特征数量和性能
   - 如果性能下降不明显，保留简化特征集
   - 如果性能下降明显，调整阈值（例如，使用 95% 而非 90%）

### **策略优势**

1. ✅ **数据驱动**: 基于实际特征重要性，而非猜测
2. ✅ **性能优化**: 保留最重要的特征，可能提升模型性能（去除噪声特征）
3. ✅ **成本优化**: 减少特征数量，降低计算成本，更快的训练和推理时间
4. ✅ **可解释性**: 了解哪些特征最重要，理解模型决策依据
5. ✅ **灵活性**: 可以根据阈值调整特征数量，支持渐进式优化

### **实施步骤**

#### **步骤 1: 使用所有特征训练模型**

```bash
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track feature_engineering \
    --horizon-h 12 \
    --config config/pipeline/train_with_loso.yaml
```

#### **步骤 2: 分析特征重要性**

```bash
# 分析 Frost 分类任务的特征重要性
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/feature_engineering/B/full_training/full_training/horizon_12h \
    --task frost \
    --plot

# 分析 Temp 回归任务的特征重要性
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/feature_engineering/B/full_training/full_training/horizon_12h \
    --task temp \
    --plot
```

#### **步骤 3: 计算累积重要性**

```python
import pandas as pd

# 读取特征重要性
importance_df = pd.read_csv(
    "experiments/.../horizon_12h/feature_importance/frost_feature_importance.csv"
)

# 计算累积重要性
importance_df = importance_df.sort_values('importance', ascending=False)
importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
importance_df['cumulative_pct'] = (
    importance_df['cumulative_importance'] / 
    importance_df['cumulative_importance'].max() * 100
)

# 找到累积重要性占 90% 的特征
top_90_features = importance_df[
    importance_df['cumulative_pct'] <= 90
]['feature'].tolist()

print(f"累积重要性占 90% 的特征数: {len(top_90_features)}")
print(f"前10个特征: {top_90_features[:10]}")
```

#### **步骤 4: 使用选定的特征重新训练**

```bash
# 使用 --feature-selection-name 参数指定特征选择名称
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track feature_engineering \
    --horizon-h 12 \
    --feature-selection-name top90 \
    --config config/pipeline/train_with_loso.yaml
```

### **预期效果**

#### **阶段 1: 全特征训练**

| 指标 | 预期 |
|------|------|
| **特征数量** | ~298 个 |
| **训练时间** | 更长（可能需要 15-20 分钟） |
| **性能** | 基准性能 |
| **特征重要性** | 完整的特征重要性分布 |

#### **阶段 2: 基于重要性重新训练**

| 指标 | 预期 |
|------|------|
| **特征数量** | ~50-200 个（取决于重要性分布） |
| **训练时间** | 更短（可能减少 50-70%） |
| **性能** | 可能提升（去除噪声特征）或保持 |
| **计算成本** | 显著降低 |

### **权衡分析**

#### **优势**
1. ✅ 数据驱动：基于实际重要性，而非猜测
2. ✅ 性能优化：去除噪声特征，可能提升性能
3. ✅ 成本优化：减少特征数量，降低计算成本
4. ✅ 可解释性：了解哪些特征最重要

#### **风险**
1. ⚠️ **特征交互**: 某些特征可能单独不重要，但组合起来重要
2. ⚠️ **阈值选择**: 90% 阈值可能需要调整（例如，85% 或 95%）
3. ⚠️ **任务差异**: Frost 和 Temp 任务可能需要不同的特征集

#### **建议**
1. ✅ **尝试多个阈值**: 85%, 90%, 95%
2. ✅ **分别处理**: Frost 和 Temp 任务使用不同的特征集
3. ✅ **验证性能**: 确保简化特征集不会显著降低性能

---

## 分析方法

### 1. **查看Top-K特征**

```python
import pandas as pd

# 读取特征重要性
importance_df = pd.read_csv("frost_feature_importance.csv")

# 查看Top 10特征
top_10 = importance_df.head(10)
print(top_10[['feature', 'importance_pct', 'cumulative_pct']])
```

### 2. **计算特征覆盖率**

```python
# 计算需要多少特征才能覆盖80%的重要性
coverage_80 = importance_df[importance_df['cumulative_pct'] <= 80]
print(f"需要 {len(coverage_80)} 个特征来覆盖80%的重要性")
```

### 3. **识别关键特征**

```python
# 识别重要性超过5%的特征
key_features = importance_df[importance_df['importance_pct'] >= 5]
print(f"关键特征（重要性 >= 5%）：{list(key_features['feature'])}")
```

### 4. **跨Horizon比较**

比较不同预测horizon的特征重要性：

```python
import pandas as pd
import matplotlib.pyplot as plt

horizons = [3, 6, 12, 24]
importance_by_horizon = {}

for h in horizons:
    path = f"experiments/lightgbm/raw/A/full_training/full_training/horizon_{h}h/frost_feature_importance.csv"
    if Path(path).exists():
        df = pd.read_csv(path)
        importance_by_horizon[h] = df.set_index('feature')['importance_pct']

# 合并数据
combined = pd.DataFrame(importance_by_horizon)

# 可视化
combined.plot(kind='bar', figsize=(14, 8))
plt.title('Feature Importance Across Horizons')
plt.xlabel('Feature')
plt.ylabel('Importance (%)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Horizon (hours)')
plt.tight_layout()
plt.savefig('feature_importance_across_horizons.png', dpi=300)
```

### 5. **跨模型比较**

比较不同模型类型的特征重要性：

```python
models = ['lightgbm', 'xgboost', 'catboost']
importance_by_model = {}

for model in models:
    path = f"experiments/{model}/raw/A/full_training/full_training/horizon_12h/frost_feature_importance.csv"
    if Path(path).exists():
        df = pd.read_csv(path)
        importance_by_model[model] = df.set_index('feature')['importance_pct']

# 合并和可视化
combined = pd.DataFrame(importance_by_model)
combined.plot(kind='bar', figsize=(14, 8))
plt.title('Feature Importance Across Models')
plt.xlabel('Feature')
plt.ylabel('Importance (%)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('feature_importance_across_models.png', dpi=300)
```

---

## 注意事项

### 1. **模型类型限制**

- **Tree-based模型**: 提供原生特征重要性
- **Linear模型**: 提供系数作为重要性
- **Deep learning模型**: 不提供直接特征重要性（需要使用替代方法）

### 2. **相关性 vs 因果性**

特征重要性只反映相关性，不一定是因果关系。

### 3. **特征交互**

Tree-based模型会自动捕获特征交互，但重要性分数可能无法直接反映交互效应。

### 4. **数据泄漏检查**

如果某个特征重要性异常高，检查是否存在数据泄漏（例如：标签列被误用为特征）。

### 5. **特征重要性的局限性**

- 特征重要性是**模型特定的**，不同模型可能有不同的重要性
- 特征重要性可能**随训练变化**，不同的超参数或训练数据会导致不同的重要性
- 如果需要数据集级别的特征重要性，使用Permutation Importance、SHAP Values或相关性分析

---

## 相关文档

- **[特征工程指南](./FEATURE_GUIDE.md)**: 特征工程完整指南
- **[训练指南](../training/TRAINING_GUIDE.md)**: 训练和评估指南
- **[模型指南](../models/MODELS_GUIDE.md)**: 模型详细说明
- **[实验分析报告](./experiments/)**: 特征重要性实验分析

---

## 相关命令

```bash
# 训练模型（自动保存特征重要性）
python -m src.cli train single --model-name lightgbm --matrix-cell A --track raw --horizon-h 12

# 分析特征重要性
python -m src.cli analysis feature-importance --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h

# 评估模型性能
python -m src.cli evaluate model --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h
```

---

**最后更新**: 2025-11-20  
**文档版本**: 3.0

