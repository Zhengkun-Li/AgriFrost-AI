# AgriFrost-AI 用户指南

<div align="center">

<img src="logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

</div>

**最后更新**: 2025-11-19

本指南涵盖从环境设置、快速开始到高级使用的所有内容。

## 📋 目录

1. [环境设置](#环境设置)
2. [快速开始](#快速开始)
3. [数据准备与加载](#数据准备与加载)
4. [完整流程指南](#完整流程指南)
5. [模型训练](#模型训练)
6. [模型评估](#模型评估)
7. [模型推理](#模型推理)
8. [结果解读](#结果解读)
9. [常见问题](#常见问题)

---

## 环境设置

### ⚠️ 重要：使用虚拟环境

**强烈建议使用虚拟环境**来安装项目依赖，原因：
- ✅ **隔离依赖**：避免与系统 Python 或其他项目的依赖冲突
- ✅ **版本一致性**：确保团队成员使用相同的依赖版本
- ✅ **易于管理**：可以轻松删除和重建环境
- ✅ **避免污染**：不会影响系统 Python 环境

### 快速设置（推荐）

#### 步骤 1: 创建虚拟环境

```bash
# 创建虚拟环境（推荐使用 .venv）
python3 -m venv .venv

# 或者使用其他名称
# python3 -m venv venv
# python3 -m venv env
```

#### 步骤 2: 激活虚拟环境

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
# PowerShell
.venv\Scripts\Activate.ps1

# Command Prompt
.venv\Scripts\activate.bat
```

**验证激活成功：**
- 命令提示符前应显示 `(.venv)` 或 `(venv)`
- 运行 `which python` (Linux/macOS) 或 `where python` (Windows) 应显示虚拟环境路径

#### 步骤 3: 安装依赖

```bash
# 升级 pip（重要：确保最新版本以支持最新包）
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

#### 步骤 4: 验证安装

```bash
# 验证关键依赖
python3 -c "import pandas, numpy, lightgbm, xgboost; print('✅ All packages installed')"

# 验证 CLI 可用
python -m src.cli --help
```

#### 退出虚拟环境

完成工作后，可以退出虚拟环境：
```bash
deactivate
```

### 常见问题

**Q: 如何知道虚拟环境已激活？**
- A: 命令提示符前会显示 `(.venv)` 或 `(venv)`

**Q: 每次使用都需要激活吗？**
- A: 是的，每次打开新的终端窗口都需要重新激活虚拟环境

**Q: 可以删除虚拟环境吗？**
- A: 可以，直接删除 `.venv` 目录即可，然后重新创建

**Q: 虚拟环境占用空间大吗？**
- A: 大约 1-2GB，包含所有 Python 包和依赖

---

## 快速开始

### 最简单的使用方式

使用统一的 CLI 接口进行训练：

```bash
# 激活虚拟环境
source .venv/bin/activate

# 训练单个模型（LightGBM, Matrix Cell B, Top 175 Features, 12小时预测）
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --output-dir experiments/lightgbm_B_12h
```

这将自动完成：
1. 从数据目录加载数据
2. 清洗数据并构建特征
3. 训练模型（分类 + 回归）
4. 在 train/val/test 上评估
5. 保存所有结果到输出目录

### 批量训练（Matrix Experiments）

```bash
# 使用配置文件批量训练多个矩阵单元
python -m src.cli train matrix \
    --config config/pipeline/matrix_experiments.yaml
```

### 使用不同模型

```bash
# XGBoost
python -m src.cli train single \
    --model-name xgboost \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --output-dir experiments/xgboost_B_12h

# LSTM (需要 GPU)
python -m src.cli train single \
    --model-name lstm \
    --matrix-cell A \
    --track raw \
    --horizon-h 12 \
    --output-dir experiments/lstm_A_12h \
    --config config/pipeline/lstm_config.yaml
```

---

## 数据准备与加载

### 数据位置

数据位于：`data/raw/frost-risk-forecast-challenge/stations/`

包含 18 个站点的 CSV 文件（每个约 14-15MB），系统会自动加载并合并。

### 自动加载（推荐）

系统按以下顺序自动检测数据：
1. `stations/` 目录（首选，18 个站点文件）
2. `cimis_all_stations.csv.gz`（备选）
3. `cimis_all_stations.csv`（最后）

**当前使用**: `stations/` 目录自动加载方法

### 数据加载过程

```
Loading 18 station files from stations/...
  Loaded 18/18 files...
Combining 18 station DataFrames...
Combined data: 2367360 rows, 26 columns
Stations: 18
```

### 性能说明

- **加载时间**: 18 个文件约需 10-30 秒
- **内存使用**: 合并后约 236 万行，内存占用约 500MB-1GB
- **文件大小**: 每个站点文件约 14-15MB，总计约 254MB

---

## 完整流程指南

### 数据流程图

```
原始数据 (CSV)
    ↓
[数据加载] → DataFrame (236万行, 26列)
    ↓
[QC过滤] → 低质量数据标记为 NaN
    ↓
[哨兵值处理] → -6999, -9999 → NaN
    ↓
[缺失值插补] → 前向填充
    ↓
[特征工程] → DataFrame (236万行, 300+列)
    ├─ 时间特征 (hour, month, season, ...)
    ├─ 滞后特征 (lag_1, lag_3, lag_6, ...)
    ├─ 滚动特征 (rolling_6h_mean, ...)
    ├─ 辐射特征 (Sol Rad相关)
    ├─ 风向特征 (Wind Dir周期性编码)
    └─ 派生特征 (temp_dew_diff, ...)
    ↓
[标签生成] → DataFrame (236万行, 300+特征列 + 8标签列)
    ├─ frost_3h, frost_6h, frost_12h, frost_24h
    └─ temp_3h, temp_6h, temp_12h, temp_24h
    ↓
[数据划分] → Train (70%) / Val (15%) / Test (15%)
    ↓
[模型训练] → 对每个时间窗口训练2个模型
    ├─ 分类模型 (frost probability)
    └─ 回归模型 (temperature)
    ↓
[模型评估] → 计算所有指标
    ↓
[模型保存] → 模型文件和元数据
```

### 关键步骤说明

#### 1. 数据清洗

系统自动执行以下清洗步骤：
- **QC 过滤**: 根据 QC 标记过滤低质量数据（保留空白和 `Y`，标记 `M/R/S/Q/P` 为 NaN）
- **哨兵值处理**: 将 `-6999`, `-9999` 等哨兵值替换为 `NaN`
- **缺失值插补**: 使用前向填充（按站点分组）

#### 2. 特征工程

系统自动创建以下特征：
- **时间特征**: hour, day_of_year, month, season, 周期性编码
- **滞后特征**: 1h, 3h, 6h, 12h, 24h 前的值
- **滚动统计**: 6h, 12h, 24h 窗口的 mean, min, max, std
- **辐射特征**: 日累积辐射、辐射变化率、夜间冷却率
- **风向特征**: 周期性编码、类别编码
- **派生特征**: 温度差、风寒指数、热指数等

详细说明请参考 [特征工程指南](FEATURE_GUIDE.md)。

#### 3. 标签生成

对每个预测时间窗口（3h, 6h, 12h, 24h），创建：
- **霜冻标签** (`frost_{h}h`): 未来温度是否 < 0°C（二分类）
- **温度标签** (`temp_{h}h`): 未来温度值（回归）

---

## 模型训练

### 使用 CLI 训练（推荐）

#### 单个模型训练

```bash
# 基本训练
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --output-dir experiments/lightgbm_B_12h

# 使用配置文件
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --config config/pipeline/train.yaml
```

#### 批量训练（Matrix Experiments）

```bash
# 使用配置文件批量训练多个矩阵单元
python -m src.cli train matrix \
    --config config/pipeline/matrix_experiments.yaml
```

### 训练参数说明

#### CLI 参数

- `--model-name`: 模型类型（lightgbm, xgboost, lstm, gru, tcn 等）
- `--matrix-cell`: 矩阵单元（A/B/C/D/E）
- `--track`: 特征轨道（raw, top175_features 等）
- `--horizon-h`: 预测时间窗口（3, 6, 12, 24 小时）
- `--radius-km`: 空间半径（C/D 轨道需要）
- `--knn-k`: KNN k 参数（E 轨道需要）
- `--config`: 配置文件路径（YAML）
- `--output-dir`: 输出目录
- `--data-path`: 输入数据路径（可选）

#### 配置文件

可以通过 YAML 配置文件设置详细的训练参数：

```yaml
data:
  source: "data/raw/frost-risk-forecast-challenge/stations/"
  matrix_cell: "B"
  
training:
  model: "lightgbm"
  horizons: [3, 6, 12, 24]
  
model_params:
  lightgbm:
    n_estimators: 200
    learning_rate: 0.05
    max_depth: 8
    num_leaves: 63
```

### 训练输出

每个时间窗口会训练两个模型：
1. **分类模型** (`frost_classifier`): 预测霜冻概率
2. **回归模型** (`temp_regressor`): 预测未来温度

训练结果保存在输出目录：

```
experiments/lightgbm_B_12h/
├── horizon_12h/
│   ├── frost_classifier/
│   │   ├── model.pkl
│   │   ├── train_metrics.json
│   │   ├── val_metrics.json
│   │   ├── test_metrics.json
│   │   └── feature_importance.csv
│   ├── temp_regressor/
│   │   ├── model.pkl
│   │   ├── train_metrics.json
│   │   ├── val_metrics.json
│   │   └── test_metrics.json
│   └── run_metadata.json
```

---

## 模型评估

### 评估单个模型

```bash
# 评估已训练的模型
python -m src.cli evaluate model \
    --model-dir experiments/lightgbm_B_12h/horizon_12h \
    --config config/evaluation.yaml \
    --output-dir evaluation_results/
```

### 比较多个模型

```bash
# 比较两个或多个模型
python -m src.cli evaluate compare \
    --model-dirs experiments/model1 experiments/model2 \
    --output-dir comparison/
```

### 生成矩阵摘要

```bash
# 生成所有实验的矩阵摘要
python -m src.cli evaluate matrix \
    --experiments-dir experiments/ \
    --output-dir matrix_summary/
```

### 评估指标

每个模型自动在 train/val/test 三个数据集上评估：

**回归指标**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

**分类指标**:
- Brier Score
- ROC-AUC
- PR-AUC (Precision-Recall AUC)
- ECE (Expected Calibration Error)

详细说明请参考 [训练和评估文档](TRAINING_AND_EVALUATION.md)。

---

## 模型推理

### 生成预测

```bash
# 使用训练好的模型生成预测
python -m src.cli inference predict \
    --model-dir experiments/lightgbm_B_12h/horizon_12h \
    --input data/test.csv \
    --output predictions.csv \
    --horizon-h 12
```

### 多时间窗口预测

```bash
# 生成多个时间窗口的预测
python -m src.cli inference predict \
    --model-dir experiments/lightgbm_B_12h \
    --input data/test.csv \
    --output predictions.csv \
    --horizon-h 3 --horizon-h 6 --horizon-h 12 --horizon-h 24
```

详细说明请参考 [推理指南](INFERENCE_GUIDE.md)。

---

## 结果解读

### 结果组织结构

```
experiments/lightgbm_B_12h/
├── horizon_12h/
│   ├── frost_classifier/
│   │   ├── model.pkl
│   │   ├── train_metrics.json
│   │   ├── val_metrics.json
│   │   ├── test_metrics.json
│   │   └── feature_importance.csv
│   ├── temp_regressor/
│   │   ├── model.pkl
│   │   ├── train_metrics.json
│   │   ├── val_metrics.json
│   │   └── test_metrics.json
│   └── run_metadata.json
```

### 关键指标位置

**单个模型的关键指标**:
- **测试集 MAE**: `test_metrics.json` → `mae`（越小越好）
- **测试集 R²**: `test_metrics.json` → `r2`（越接近 1 越好）
- **测试集 ROC-AUC**: `test_metrics.json` → `roc_auc`（分类模型，越接近 1 越好）

### 结果质量判断

- **MAE < 1°C**: 预测精度高
- **R² > 0.9**: 模型拟合良好
- **Train vs Test 差异小**: 无过拟合
- **ROC-AUC > 0.95**: 分类性能优秀

---

## 常见问题

### Q: 如何查看模型结果？

```bash
# 查看测试指标
cat experiments/lightgbm_B_12h/horizon_12h/frost_classifier/test_metrics.json

# 查看实验元数据
cat experiments/lightgbm_B_12h/horizon_12h/run_metadata.json
```

### Q: 如何获取帮助？

```bash
# 查看所有命令
python -m src.cli --help

# 查看训练命令帮助
python -m src.cli train --help

# 查看单个训练命令帮助
python -m src.cli train single --help
```

### Q: 如何自定义模型参数？

编辑配置文件（YAML）或在 CLI 中使用配置选项。示例配置文件见 `config/pipeline/` 目录。

### Q: 结果会自动保存吗？

**是的！** 所有结果都会自动保存：
- 模型文件（.pkl）
- 评估指标（JSON）
- 实验元数据（run_metadata.json）
- 特征重要性（CSV，如果支持）

### Q: 如何在不同矩阵单元之间切换？

使用 `--matrix-cell` 参数：
- `A`: Raw features, Single-station
- `B`: Feature-engineered, Single-station
- `C`: Raw features, Multi-station
- `D`: Feature-engineered, Multi-station
- `E`: Graph neural networks

---

## 📚 相关文档

- **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)**: 技术文档和 API 参考
- **[DATA_DOCUMENTATION.md](DATA_DOCUMENTATION.md)**: 数据说明和 QC 处理
- **[FEATURE_GUIDE.md](FEATURE_GUIDE.md)**: 特征工程完整指南
- **[TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md)**: 训练和评估详细指南
- **[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)**: 推理使用指南
