# AgriFrost-AI: 快速开始指南

<div align="center">

<img src="../logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

## 🌡️ AgriFrost-AI 快速开始

**AI-Powered Frost Risk Prediction System for California Agriculture**

*在 15 分钟内从零开始运行您的第一个霜冻预测模型*

</div>

---

## 📋 目录

1. [环境准备](#1-环境准备)
2. [数据下载](#2-数据下载)
3. [第一个模型训练](#3-第一个模型训练)
4. [模型评估](#4-模型评估)
5. [生成预测](#5-生成预测)
6. [下一步](#6-下一步)

---

## 1. 环境准备

### 1.1 系统要求

- **Python**: 3.10 - 3.14（推荐 3.12）
- **操作系统**: Linux, macOS, Windows
- **GPU**（可选）: NVIDIA GPU with CUDA 13.0+（用于深度学习模型）
- **内存**: 建议 16GB+ RAM
- **存储**: 至少 10GB 可用空间

### 1.2 安装步骤

#### 步骤 1: 克隆项目

```bash
# 克隆仓库（如果没有数据仓库，可以稍后下载数据）
git clone <your-repo-url>
cd frost-risk-forecast-challenge
```

#### 步骤 2: 创建和激活虚拟环境

**⚠️ 重要：强烈建议使用虚拟环境！**

虚拟环境可以：
- ✅ 隔离项目依赖，避免与系统 Python 冲突
- ✅ 保持项目依赖版本一致性
- ✅ 方便管理和清理依赖

**创建虚拟环境：**

```bash
# 创建虚拟环境（推荐使用 .venv）
python3 -m venv .venv

# 或者使用其他名称
# python3 -m venv venv
# python3 -m venv env
```

**激活虚拟环境：**

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

**退出虚拟环境：**
```bash
deactivate
```

#### 步骤 3: 安装依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

**注意**：
- 如果使用 **CPU 版本**（无 GPU），需要修改 `requirements.txt`，将 PyTorch 安装改为 CPU 版本：
  ```bash
  # 注释掉 CUDA 版本的 PyTorch，安装 CPU 版本
  pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu
  ```

#### 步骤 4: 验证安装

```bash
# 检查 CLI 是否可用
python -m src.cli --help

# 检查关键依赖
python -c "import lightgbm, xgboost, torch, pandas; print('✅ All dependencies installed!')"
```

---

## 2. 数据下载

### 2.1 数据来源

数据来自 **F3 Innovate Frost Risk Forecasting Challenge** 官方仓库：
- **仓库**: https://github.com/CarlSaganPhD/frost-risk-forecast-challenge
- **数据格式**: CSV 文件（gzipped）
- **大小**: ~38 MB (压缩后), ~200 MB (解压后)
- **时间范围**: 2010-09-28 至 2025-09-28
- **站点数量**: 18 个 CIMIS 气象站

### 2.2 下载方法

#### 方法 1: Git 克隆（推荐）

```bash
# 创建数据目录
mkdir -p data/raw/frost-risk-forecast-challenge

# 克隆数据仓库
git clone https://github.com/CarlSaganPhD/frost-risk-forecast-challenge.git data_repo_temp

# 复制数据文件
cp -r data_repo_temp/stations data/raw/frost-risk-forecast-challenge/
cp data_repo_temp/cimis_all_stations.csv.gz data/raw/frost-risk-forecast-challenge/

# 清理临时目录
rm -rf data_repo_temp

# 验证数据
ls -lh data/raw/frost-risk-forecast-challenge/
# 应该看到:
# - stations/ (包含 18 个 CSV 文件)
# - cimis_all_stations.csv.gz
```

#### 方法 2: 手动下载

1. 访问：https://github.com/CarlSaganPhD/frost-risk-forecast-challenge
2. 下载 `cimis_all_stations.csv.gz` 文件
3. 下载 `stations/` 目录（或其中的所有 CSV 文件）
4. 放置到 `data/raw/frost-risk-forecast-challenge/` 目录

#### 方法 3: 使用 Python 脚本（如果有 API）

```bash
# 如果有下载脚本（需要根据实际情况调整）
python scripts/tools/download_data.py
```

### 2.3 验证数据

```bash
# 检查数据文件
python -c "
from pathlib import Path
data_dir = Path('data/raw/frost-risk-forecast-challenge')
print(f'📁 数据目录: {data_dir}')
print(f'📊 合并文件: {data_dir / \"cimis_all_stations.csv.gz\"} exists: {(data_dir / \"cimis_all_stations.csv.gz\").exists()}')
print(f'📁 站点目录: {data_dir / \"stations\"} exists: {(data_dir / \"stations\").exists()}')
if (data_dir / 'stations').exists():
    station_files = list((data_dir / 'stations').glob('*.csv'))
    print(f'📈 站点文件数量: {len(station_files)}')
"
```

**预期输出**：
```
📁 数据目录: data/raw/frost-risk-forecast-challenge
📊 合并文件: exists: True
📁 站点目录: exists: True
📈 站点文件数量: 18
```

---

## 3. 第一个模型训练

### 3.1 最简单的训练命令

让我们训练一个 **LightGBM** 模型，使用 **Top 175 特征**，预测 **12 小时**后的霜冻风险：

```bash
# 激活虚拟环境（如果还没激活）
source .venv/bin/activate

# 训练单个模型
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --output-dir experiments/my_first_model_12h
```

**参数说明**：
- `--model-name lightgbm`: 使用 LightGBM 模型（快速、准确）
- `--matrix-cell B`: 使用特征工程 + 单站点（Matrix Cell B）
- `--track top175_features`: 使用 Top 175 精选特征（最佳性能）
- `--horizon-h 12`: 预测 12 小时后的霜冻
- `--output-dir`: 模型保存目录

**预计时间**：
- **数据加载和预处理**: ~2-5 分钟
- **特征工程**: ~10-30 分钟
- **模型训练**: ~5-10 分钟
- **总计**: ~20-45 分钟（取决于硬件）

### 3.2 训练过程说明

训练过程会自动执行以下步骤：

1. **数据加载**: 从 `data/raw/` 加载原始数据
2. **数据清洗**: QC 过滤、异常值处理、缺失值填补
3. **特征工程**: 生成 175 个精选特征
4. **标签生成**: 为 12h 时间范围生成霜冻标签
5. **数据分割**: 70% 训练，15% 验证，15% 测试
6. **模型训练**: 
   - 分类模型（霜冻概率）
   - 回归模型（温度预测）
7. **模型保存**: 保存到 `experiments/my_first_model_12h/horizon_12h/`

### 3.3 查看训练结果

训练完成后，检查输出目录：

```bash
# 查看模型文件
ls -lh experiments/my_first_model_12h/horizon_12h/

# 应该看到：
# - frost_model.pkl (分类模型)
# - temp_model.pkl (回归模型)
# - run_metadata.json (实验元数据)
# - train_metrics.json (训练指标)
# - validation_metrics.json (验证指标)
# - test_metrics.json (测试指标)
```

**查看训练指标**：

```bash
# 查看测试集性能
cat experiments/my_first_model_12h/horizon_12h/test_metrics.json

# 或使用 Python
python -c "
import json
from pathlib import Path
metrics = json.load(open('experiments/my_first_model_12h/horizon_12h/test_metrics.json'))
print('📊 测试集性能:')
print(f'  ROC-AUC (分类): {metrics[\"classification\"][\"roc_auc\"]:.4f}')
print(f'  Brier Score (校准): {metrics[\"classification\"][\"brier_score\"]:.4f}')
print(f'  MAE (回归): {metrics[\"regression\"][\"mae\"]:.4f}°C')
print(f'  R² (回归): {metrics[\"regression\"][\"r2\"]:.4f}')
"
```

**预期性能**（LightGBM + Top 175 特征，12h）：
- ROC-AUC: > 0.98
- Brier Score: < 0.01
- MAE: < 2°C
- R²: > 0.91

---

## 4. 模型评估

### 4.1 标准评估

评估刚才训练的模型：

```bash
# 评估单个模型
python -m src.cli evaluate model \
    --model-dir experiments/my_first_model_12h \
    --config config/evaluation.yaml
```

这会生成详细的评估报告，包括：
- 分类指标（ROC-AUC, PR-AUC, Brier Score, ECE）
- 回归指标（MAE, RMSE, R²）
- 校准曲线和可靠性图

### 4.2 LOSO 评估（空间泛化）

为了测试模型在不同站点的泛化能力，运行 LOSO（留一站交叉验证）评估：

```bash
# LOSO 评估（需要较长时间）
python -m src.cli train single \
    --model-name lightgbm \
    --matrix-cell B \
    --track top175_features \
    --horizon-h 12 \
    --loso \
    --output-dir experiments/my_first_model_12h_loso
```

**注意**：
- LOSO 评估需要训练 18 个模型（每个站点一个）
- 预计时间：**3-6 小时**（取决于硬件）
- 使用简化的模型配置（更快但性能略低）

### 4.3 多时间范围评估

训练所有时间范围（3h, 6h, 12h, 24h）：

```bash
# 训练矩阵实验（所有时间范围）
python -m src.cli train matrix \
    --config config/pipeline/matrix_experiments.yaml
```

或逐个训练：

```bash
for horizon in 3 6 12 24; do
    python -m src.cli train single \
        --model-name lightgbm \
        --matrix-cell B \
        --track top175_features \
        --horizon-h $horizon \
        --output-dir experiments/lightgbm_B_${horizon}h
done
```

---

## 5. 生成预测

### 5.1 准备预测数据

预测数据应该与训练数据格式相同。示例：

```bash
# 创建测试数据目录（如果还没有）
mkdir -p data/test

# 使用历史数据的一部分作为测试数据
python -c "
import pandas as pd
from pathlib import Path

# 加载数据
data_path = Path('data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz')
df = pd.read_csv(data_path)

# 取最后 1000 行作为测试数据
test_df = df.tail(1000)

# 保存测试数据
test_df.to_csv('data/test/prediction_input.csv', index=False)
print(f'✅ 测试数据已保存: {len(test_df)} 行')
"
```

### 5.2 生成预测

```bash
# 使用训练好的模型生成预测
python -m src.cli inference predict \
    --model-dir experiments/my_first_model_12h \
    --input data/test/prediction_input.csv \
    --output predictions.csv
```

**输出格式**：
```csv
Date,Stn Id,Frost Probability,Temperature Prediction
2025-09-28 12:00:00,2,0.0234,8.5
2025-09-28 12:00:00,7,0.0156,9.2
...
```

### 5.3 查看预测结果

```bash
# 查看前几行预测
head -20 predictions.csv

# 使用 Python 分析预测
python -c "
import pandas as pd
df = pd.read_csv('predictions.csv')
print('📊 预测结果统计:')
print(f'  总预测数: {len(df)}')
print(f'  平均霜冻概率: {df[\"Frost Probability\"].mean():.4f}')
print(f'  高风险预测 (>0.5): {(df[\"Frost Probability\"] > 0.5).sum()}')
print(f'  平均温度预测: {df[\"Temperature Prediction\"].mean():.2f}°C')
"
```

---

## 6. 下一步

### 6.1 探索更多功能

1. **尝试不同模型**：
   ```bash
   # XGBoost
   python -m src.cli train single --model-name xgboost --matrix-cell B --track top175_features --horizon-h 12 --output-dir experiments/xgboost_B_12h
   
   # LSTM (需要 GPU)
   python -m src.cli train single --model-name lstm --matrix-cell B --track top175_features --horizon-h 12 --output-dir experiments/lstm_B_12h
   ```

2. **尝试不同矩阵单元**：
   ```bash
   # Matrix Cell C (多站点，原始特征)
   python -m src.cli train single --model-name lightgbm --matrix-cell C --track raw_features --horizon-h 12 --output-dir experiments/lightgbm_C_12h
   
   # Matrix Cell D (多站点，工程特征)
   python -m src.cli train single --model-name lightgbm --matrix-cell D --track top175_features --horizon-h 12 --output-dir experiments/lightgbm_D_12h
   ```

3. **特征分析**：
   ```bash
   # 完整特征分析
   python -m src.cli analysis full \
       --data-path data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz \
       --model-dir experiments/my_first_model_12h \
       --output-dir analysis/features
   ```

### 6.2 深入学习

- 📖 **用户指南**: `docs/USER_GUIDE.md` - 完整使用说明
- 🏗️ **实现指南**: `docs/IMPLEMENTATION_GUIDE.md` - 系统架构和方法论
- 🔬 **技术文档**: `docs/TECHNICAL_DOCUMENTATION.md` - 技术细节
- 🤖 **模型指南**: `docs/MODELS_GUIDE.md` - 所有模型的详细说明
- 📊 **特征指南**: `docs/FEATURE_GUIDE.md` - 特征工程详解

### 6.3 常用命令速查

```bash
# ===== 训练 =====
# 单模型训练
python -m src.cli train single --model-name lightgbm --matrix-cell B --track top175_features --horizon-h 12 --output-dir experiments/model

# 矩阵批量训练
python -m src.cli train matrix --config config/pipeline/matrix_experiments.yaml

# LOSO 训练
python -m src.cli train single --loso --output-dir experiments/loso_model

# ===== 评估 =====
# 单模型评估
python -m src.cli evaluate model --model-dir experiments/model

# 模型比较
python -m src.cli evaluate compare --model-dirs experiments/model1 experiments/model2 --output-dir comparison/

# 矩阵总结
python -m src.cli evaluate matrix --experiments-dir experiments/ --output-dir matrix_summary/

# ===== 推理 =====
# 生成预测
python -m src.cli inference predict --model-dir experiments/model --input data/test.csv --output predictions.csv

# ===== 分析 =====
# 特征分析
python -m src.cli analysis full --data-path data/train.csv --model-dir experiments/model --output-dir analysis/

# 工具 =====
# 生成站点分布地图
python scripts/tools/generate_station_map.py

# 获取站点元数据
python scripts/tools/fetch_station_metadata.py
```

### 6.4 故障排除

#### 问题 1: 数据未找到

```
FileNotFoundError: Data file not found: data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz
```

**解决方案**：
1. 确认数据已下载（见 [数据下载](#2-数据下载)）
2. 检查路径是否正确：`ls -lh data/raw/frost-risk-forecast-challenge/`

#### 问题 2: 内存不足

```
MemoryError: Unable to allocate array
```

**解决方案**：
1. 减少数据量：使用 `--sample-size` 参数
   ```bash
   python -m src.cli train single --model-name lightgbm --matrix-cell B --track top175_features --horizon-h 12 --output-dir experiments/model --sample-size 100000
   ```
2. 使用 Top 175 特征（而不是完整 298 特征）
3. 增加系统内存或使用更大内存的机器

#### 问题 3: GPU 不可用（深度学习模型）

```
RuntimeError: CUDA error: no kernel image is available for execution
```

**解决方案**：
1. 检查 CUDA 版本：`nvidia-smi`
2. 确认 PyTorch 版本匹配：`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
3. 使用 CPU 版本或重新安装正确的 PyTorch 版本
4. 对于深度学习模型，可以考虑使用 LightGBM/XGBoost（不需要 GPU）

#### 问题 4: 训练时间过长

**解决方案**：
1. 使用 LightGBM（最快）
2. 减少 `n_estimators`（树的数量）
3. 使用 Top 175 特征（而不是完整特征集）
4. 减少数据量（用于快速测试）

---

## 📞 获取帮助

- 📖 **完整文档**: 查看 `docs/` 目录下的详细文档
- 🐛 **问题报告**: 在 GitHub Issues 中报告问题
- 💬 **讨论**: 在 GitHub Discussions 中提问

---

**恭喜！** 🎉 您已经完成了 AgriFrost-AI 的快速开始！现在可以：
- 训练更多模型进行实验
- 探索不同的配置和参数
- 阅读详细文档深入学习
- 开始您的霜冻预测研究！

---

**文档版本**: 1.0  
**最后更新**: 2025-11-19  
**作者**: Zhengkun LI (TRIC Robotics / UF ABE)

