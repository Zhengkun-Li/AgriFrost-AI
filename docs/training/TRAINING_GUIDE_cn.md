# AgriFrost-AI 训练和评估完整指南

<div align="center">

<img src="../logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

</div>

**最后更新**: 2025-11-20

本文档整合了训练配置、LOSO评估、训练监控、命令详解等所有训练相关内容，为模型训练提供一站式参考。

## 📋 目录

1. [训练命令详解](#训练命令详解)
2. [训练配置](#训练配置)
3. [LOSO 评估](#loso-评估)
4. [训练监控](#训练监控)
5. [性能对比](#性能对比)
6. [命令行详解](#命令行详解)
7. [常见问题](#常见问题)

---

## 环境准备

### ⚠️ 重要：使用虚拟环境

在开始训练之前，请确保已创建并激活虚拟环境：

```bash
# 创建虚拟环境（如果还没有）
python3 -m venv .venv

# 激活虚拟环境
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate.bat

# 确保已安装所有依赖
pip install -r requirements.txt
```

**验证环境：**
```bash
# 检查 CLI 是否可用
python -m src.cli --help

# 检查关键依赖
python -c "import lightgbm, xgboost, torch; print('✅ Environment ready!')"
```

更多环境设置说明，请参考 [快速开始指南](../guides/QUICK_START.md#1-环境准备)。

---

## 训练命令详解

### 基本命令格式

```bash
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell A \
    --track raw \
    --horizon-h 12 \
    --config config/pipeline/train_with_loso.yaml \
    --data-path data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz
```

### ⚠️ **关键问题：`--horizon-h 12` 的实际作用**

#### 现象

虽然命令中指定了 `--horizon-h 12`，但实际训练**包含了所有horizon**（3h, 6h, 12h, 24h）。

从实验目录可以看到：
```
experiments/lightgbm/raw/A/full_training/full_training/
  ├── horizon_3h/
  ├── horizon_6h/
  ├── horizon_12h/
  └── horizon_24h/
```

#### 原因分析

##### 1. **标签生成阶段**

在 `src/cli/commands/train.py` 的 `single()` 函数中：

```python
# CRITICAL: Generate labels for all horizons [3, 6, 12, 24] even when training single horizon
# This ensures labeled_data.parquet contains all horizon labels
cli_overrides: Dict[str, Any] = {
    "model": model_name,
    "matrix_cell": matrix_cell,
    "horizons": [3, 6, 12, 24],  # Generate labels for all horizons
}
```

**关键点**: 即使命令行指定了 `--horizon-h 12`，代码中**强制设置**了 `horizons: [3, 6, 12, 24]` 来生成所有horizon的标签。

**原因**: 这修复了之前的一个bug，确保 `labeled_data.parquet` 包含所有horizon的标签（`frost_3h`, `frost_6h`, `frost_12h`, `frost_24h`）。

##### 2. **配置文件覆盖**

在 `config/pipeline/train_with_loso.yaml` 中：

```yaml
labels:
  horizons: [3, 6, 12, 24]
```

配置文件中也指定了所有horizon，这会与CLI参数合并。

##### 3. **实际训练阶段**

在 `src/training/pipeline_runner.py` 中：

```python
# Train only horizons that have labels
training_horizons = [h for h in self.config.labels.horizons if h in available_horizons]
```

`TrainingRunner` 会训练 `config.labels.horizons` 中**所有有标签的horizon**。

由于标签生成时生成了所有horizon的标签（`[3, 6, 12, 24]`），因此**所有horizon都会被训练**。

##### 4. **`--horizon-h 12` 的实际用途**

**`--horizon-h 12` 参数主要用于**：
- 生成输出路径中的提示信息
- 在成功消息中显示horizon信息
- **不实际限制训练的horizon**

### 完整训练流程

```
1. 命令行解析
   ├── --horizon-h 12 (用于提示信息)
   └── 其他参数

2. 配置合并
   ├── CLI overrides: horizons = [3, 6, 12, 24] (强制设置)
   ├── 配置文件: horizons: [3, 6, 12, 24]
   └── 最终配置: horizons = [3, 6, 12, 24]

3. 数据加载和标签生成
   ├── DataPipeline.run() 生成所有horizon的标签
   └── labeled_data.parquet 包含: frost_3h, frost_6h, frost_12h, frost_24h

4. 训练阶段
   ├── TrainingRunner 遍历 config.labels.horizons
   ├── 训练 3h horizon → horizon_3h/
   ├── 训练 6h horizon → horizon_6h/
   ├── 训练 12h horizon → horizon_12h/
   └── 训练 24h horizon → horizon_24h/

5. LOSO评估（如果启用）
   └── 对所有horizon进行评估
```

### 如何只训练单个Horizon？

#### 方法1：修改配置文件

创建或修改配置文件，只指定一个horizon：

```yaml
labels:
  horizons: [12]  # 只训练12h
```

**注意**: 由于代码中的强制设置，这种方法**可能不生效**。

#### 方法2：修改代码

如果需要只训练单个horizon，需要修改 `src/cli/commands/train.py`：

```python
# 修改前：
"horizons": [3, 6, 12, 24],  # Generate labels for all horizons

# 修改后：
"horizons": [horizon_h],  # Only generate labels for specified horizon
```

**注意**: 这会带来其他问题（LOSO评估可能失败）。

### ✅ **当前行为总结**

| 项目 | 说明 |
|------|------|
| **命令行参数** | `--horizon-h 12` |
| **标签生成** | 生成所有horizon的标签 `[3, 6, 12, 24]` |
| **实际训练** | 训练所有horizon `[3, 6, 12, 24]` |
| **输出目录** | 包含所有horizon的子目录 |
| **`--horizon-h` 作用** | 主要用于提示信息，不限制训练的horizon |

---

## 训练配置

### 硬件配置

- **GPU**: NVIDIA RTX 5090 (32GB)
- **CPU**: AMD 9950 (32核)
- **内存**: 60GB

### 数据规模

- **总数据量**: 2,367,360 行
- **站点数**: 18
- **时间范围**: 2010-09-28 到 2025-09-28 (15年数据)

### 模型配置优化

```python
{
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 63,
    "n_jobs": 8,  # 限制CPU核心使用（避免内存溢出）
    "force_col_wise": True,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}
```

### 启动训练

使用新的 CLI 接口：

```bash
# 激活虚拟环境
source .venv/bin/activate

# 训练单个模型
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track feature_engineering \
    --horizon-h 12 \
    --output-dir experiments/lightgbm_B_12h

# 批量训练（Matrix Experiments）
python -m src.cli train matrix \
    --config config/pipeline/matrix_experiments.yaml
```

### 预计训练时间

- **数据加载**: ~2-5分钟
- **数据清洗**: ~5-10分钟
- **特征工程**: ~30-60分钟
- **标准评估训练** (4个时间窗口): ~40-80分钟
- **LOSO评估训练** (18个站点 × 4个时间窗口): ~180-360分钟 (3-6小时)

**总预计时间** (包含LOSO评估): **4-7小时**

**注意**：
- 如果不运行LOSO评估（不使用`--loso`参数），总时间约为**1.5-2.5小时**
- 如果运行LOSO评估，总时间约为**4-7小时**（LOSO评估需要额外3-6小时）

---

## LOSO 评估

### 什么是LOSO？

**LOSO (Leave-One-Station-Out)** 是一种交叉验证方法，用于评估模型的空间泛化能力。

### LOSO评估流程

1. **选择一个站点作为测试集**
   - 例如：选择站点 "Davis" 作为测试集
   - 其他17个站点作为训练集

2. **使用训练集训练模型**
   - 使用除 "Davis" 外的所有站点数据训练模型

3. **使用测试集评估模型**
   - 使用 "Davis" 站点数据评估模型性能

4. **重复上述过程**
   - 对每个站点重复上述过程
   - 最终得到18个站点的评估结果

5. **汇总结果**
   - 计算所有站点的平均性能
   - 计算标准差，评估性能的稳定性

### LOSO评估的优势

1. ✅ **空间泛化能力**: 评估模型在未见过的站点上的性能
2. ✅ **稳健性评估**: 评估模型对不同微气候的适应性
3. ✅ **实际应用价值**: 更接近实际部署场景

### 启用LOSO评估

在配置文件中启用LOSO评估：

```yaml
# config/pipeline/train_with_loso.yaml
training:
  loso:
    enabled: true
    params:
      stations: null  # null means use all stations
      horizons: [3, 6, 12, 24]
```

或使用CLI参数：

```bash
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell A \
    --track raw \
    --horizon-h 12 \
    --config config/pipeline/train_with_loso.yaml
```

### LOSO评估结果

LOSO评估结果保存在：

```
experiments/lightgbm/raw/A/full_training/loso/
  ├── summary.json          # 汇总统计（均值 ± 标准差）
  └── station_metrics.json  # 每个站点的详细指标
```

**汇总统计示例**：
```json
{
  "summary": {
    "3h": {
      "frost_metrics": {
        "brier_score": {"mean": 0.1234, "std": 0.0123},
        "ece": {"mean": 0.0567, "std": 0.0045},
        "roc_auc": {"mean": 0.9876, "std": 0.0089},
        "pr_auc": {"mean": 0.8765, "std": 0.0234}
      },
      "temp_metrics": {
        "mae": {"mean": 1.23, "std": 0.45},
        "rmse": {"mean": 1.56, "std": 0.67},
        "r2": {"mean": 0.91, "std": 0.08}
      }
    },
    ...
  }
}
```

---

## 训练监控

### 日志文件

训练过程中会生成多个日志文件：

#### 1. **实验级别日志** (`experiment.log`)

位置：`experiments/<model>/<track>/<cell>/<scope>/experiment.log`

**内容**：
- 数据加载信息（样本数、特征数、站点数、日期范围、标签统计）
- 训练/验证/测试集划分信息
- 每个horizon的训练结果摘要
- LOSO评估结果摘要（均值 ± 标准差，前10个站点的详细指标）
- 总实验时长

**示例**：
```
[Data Loading]
  ✅ Data loaded successfully
  📊 Total samples: 2,367,360
  📊 Total features: 12
  📊 Stations: 18
  📊 Date range: 2010-09-28 to 2025-09-28

[Label Statistics]
  3h: 45,234 frost events (1.91%)
  6h: 89,567 frost events (3.78%)
  12h: 156,789 frost events (6.62%)
  24h: 234,567 frost events (9.90%)

[Training]
  Training horizon: 12h
    ✅ Training completed in 123.45 seconds (2.06 minutes)
    📊 Frost Metrics:
       ROC-AUC: 0.9892
       PR-AUC: 0.8765
       Brier Score: 0.1234
    📊 Temp Metrics:
       MAE: 1.84°C
       RMSE: 2.45°C
       R²: 0.9270
    📁 Model saved to: horizon_12h/

[LOSO Evaluation]
  ✅ LOSO evaluation completed in 1800.00 seconds (30.00 minutes)
  📊 LOSO Results Summary (across all stations):
    Horizon 12h:
      Frost Metrics:
        Brier Score: 0.1345 ± 0.0123
        Expected Calibration Error (ECE): 0.0567 ± 0.0045
        ROC-AUC (discrimination): 0.9876 ± 0.0089
        PR-AUC (discrimination): 0.8765 ± 0.0234
      Temp Metrics:
        MAE: 1.96°C ± 0.45°C
        RMSE: 2.56°C ± 0.67°C
        R²: 0.9167 ± 0.0800
```

#### 2. **Horizon级别日志** (`training.log`)

位置：`experiments/<model>/<track>/<cell>/<scope>/horizon_<h>/training.log`

**内容**：
- 数据准备信息（特征数、样本数、霜冻事件数、**特征列表**）
- 数据划分信息（训练/验证/测试集大小和百分比）
- 训练过程详情（每个epoch的指标）
- 评估结果详情（校准指标、判别技能指标）
- 模型保存位置

**示例**：
```
📊 Data preparation:
   Features: 12
   Samples: 2,367,360
   Frost events: 156,789 (6.62%)
   Feature list: Hour (PST), Jul, ETo (mm), Precip (mm), ...

📊 Data split:
   Train: 1,657,152 (70.0%)
   Val: 355,104 (15.0%)
   Test: 355,104 (15.0%)

📊 Evaluation Results:
   Calibration & Reliability:
     Brier Score: 0.1234
     Expected Calibration Error (ECE): 0.0567
     Reliability Diagram: horizon_12h/reliability_diagram.png
   Discrimination Skill:
     ROC-AUC: 0.9892
     PR-AUC: 0.8765
   Temp Metrics:
     MAE: 1.84°C
     RMSE: 2.45°C
     R²: 0.9270
   Evaluation time: 12.34 seconds
   Model saved to: horizon_12h/
```

### 特征重要性文件

训练完成后，特征重要性会自动保存：

```
experiments/lightgbm/raw/A/full_training/full_training/horizon_12h/
  ├── frost_feature_importance.csv    # 霜冻分类模型的特征重要性
  └── temp_feature_importance.csv     # 温度回归模型的特征重要性
```

---

## 性能对比

### 标准评估 vs LOSO评估

| 指标 | 标准评估 | LOSO评估 | 差异 |
|------|---------|---------|------|
| **训练数据** | 70% 数据 | 94.4% 站点（17/18） | 更多训练数据 |
| **测试数据** | 15% 数据 | 5.6% 站点（1/18） | 更少测试数据 |
| **性能** | 通常更好 | 通常稍差 | 更真实 |
| **泛化能力** | 有限 | 更强 | 空间泛化 |

### 不同Horizon的性能对比

| Horizon | ROC-AUC | PR-AUC | MAE (°C) | RMSE (°C) | R² |
|---------|---------|--------|----------|-----------|-----|
| 3h | 0.9965 | 0.9543 | 1.15 | 1.45 | 0.9698 |
| 6h | 0.9928 | 0.9234 | 1.59 | 2.01 | 0.9458 |
| 12h | 0.9892 | 0.8765 | 1.84 | 2.45 | 0.9270 |
| 24h | 0.9827 | 0.8123 | 1.96 | 2.67 | 0.9171 |

**趋势**：
- 随着horizon增加，性能逐渐下降
- 这是预期的，因为长期预测更困难

---

## 命令行详解

### 复合命令示例

```bash
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h \
    --top-k 12 2>&1 | grep -E "(Saved plot|Top.*Features)" | head -5
```

#### **命令结构**

这是一个**复合命令**（Pipeline），使用管道符 `|` 连接多个命令：

```
命令1 | 命令2 | 命令3
```

#### **逐部分详解**

##### **1️⃣ Python CLI 命令（主要部分）**

```bash
python -m src.cli analysis feature-importance \
    --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h \
    --top-k 12 2>&1
```

- `python -m src.cli`: 以模块方式运行CLI
- `analysis feature-importance`: 分析特征重要性子命令
- `--model-dir`: 指定模型目录路径
- `--top-k 12`: 只显示前12个最重要的特征
- `2>&1`: 将标准错误重定向到标准输出（让`grep`能够搜索所有输出）

##### **2️⃣ grep 过滤（中间部分）**

```bash
grep -E "(Saved plot|Top.*Features)"
```

- `grep`: 文本搜索工具
- `-E`: 启用扩展正则表达式
- `"(Saved plot|Top.*Features)"`: 搜索包含 "Saved plot" 或 "Top.*Features" 的行

##### **3️⃣ head 限制输出（最后部分）**

```bash
head -5
```

- `head`: 显示前N行
- `-5`: 只显示前5行匹配结果

#### **完整执行流程**

```
1. Python CLI 命令执行
   ↓ 输出所有日志（标准输出 + 标准错误）
   
2. grep 过滤
   ↓ 只保留包含 "Saved plot" 或 "Top.*Features" 的行
   
3. head 限制
   ↓ 只显示前5行匹配结果
   
4. 终端显示
   ✅ 最终输出
```

---

## 常见问题

### Q1: 为什么训练包含所有horizon，即使只指定了`--horizon-h 12`？

**A**: 这是设计决定，用于：
- 确保所有标签都被生成
- 支持LOSO评估
- 避免标签生成的bug

详见[训练命令详解](#训练命令详解)部分。

### Q2: 如何只训练单个horizon？

**A**: 修改配置文件或代码，详见[如何只训练单个Horizon？](#如何只训练单个horizon)部分。

### Q3: LOSO评估需要多长时间？

**A**: 通常需要3-6小时，取决于数据规模和模型复杂度。

### Q4: 训练日志保存在哪里？

**A**: 
- 实验级别：`experiments/<model>/<track>/<cell>/<scope>/experiment.log`
- Horizon级别：`experiments/<model>/<track>/<cell>/<scope>/horizon_<h>/training.log`

### Q5: 如何查看特征重要性？

**A**: 使用 `analysis feature-importance` 命令，详见[命令行详解](#命令行详解)部分。

---

## 相关文档

- **[特征工程指南](../features/FEATURE_GUIDE.md)**: 特征工程完整指南
- **[特征重要性指南](../features/FEATURE_IMPORTANCE.md)**: 特征重要性分析指南
- **[模型指南](../models/MODELS_GUIDE.md)**: 模型详细说明
- **[推理指南](../inference/INFERENCE_GUIDE.md)**: 模型推理指南

---

**最后更新**: 2025-11-20  
**文档版本**: 3.0

