# AgriFrost-AI 技术文档

<div align="center">

<img src="logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="150"/>

</div>

**最后更新**: 2025-11-12

本文档提供技术架构、API 参考和开发指南。

## 📋 目录

1. [项目架构](#项目架构)
2. [支持的模型](#支持的模型)
3. [API 参考](#api-参考)
4. [配置管理](#配置管理)
5. [扩展开发](#扩展开发)

---

## 项目架构

### 核心设计原则

1. **模块化设计**：每个功能模块独立，便于测试与替换
2. **接口标准化**：统一的数据接口、模型接口、评估接口
3. **可扩展性**：新增模型/特征/评估指标无需修改核心代码
4. **可复现性**：所有实验配置、随机种子、版本号可追溯

### 项目目录结构

```
frost-risk-forecast-challenge/
├── config/                    # 配置文件
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 清洗后数据
│   └── external/              # 外部数据
├── src/                       # 源代码
│   ├── data/                  # 数据处理模块
│   ├── models/                # 模型模块
│   ├── evaluation/            # 评估模块
│   └── utils/                 # 工具函数
├── scripts/                   # 可执行脚本
│   ├── data_prep/             # 数据准备
│   ├── train/                 # 训练脚本
│   └── evaluate/              # 评估脚本
├── experiments/               # 实验输出（不跟踪）
├── results/                   # 结果汇总（不跟踪）
└── docs/                      # 文档
```

### 核心模块

#### 1. 数据模块 (`src/data/`)

- **`loaders.py`**: 数据加载器
- **`cleaners.py`**: QC 清洗和数据处理
- **`feature_engineering.py`**: 特征工程
- **`validators.py`**: 数据验证

#### 2. 模型模块 (`src/models/`)

- **`base.py`**: 基础模型接口
- **`ml/`**: 机器学习模型（LightGBM, XGBoost）
- **`traditional/`**: 传统时间序列模型
- **`deep/`**: 深度学习模型

#### 3. 评估模块 (`src/evaluation/`)

- **`metrics.py`**: 评估指标
- **`validators.py`**: 交叉验证策略
- **`comparators.py`**: 模型对比

---

## 支持的模型

### LightGBM ⭐ (默认)

**特点**:
- 快速训练和预测
- 自动处理缺失值
- 特征重要性提取
- 内存效率高

**配置示例**:
```python
{
    "model_type": "lightgbm",
    "task_type": "regression",
    "model_params": {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "random_state": 42
    }
}
```

### XGBoost

**特点**:
- 性能稳定
- 正则化能力强
- 特征重要性支持

### 模型对比表

| 模型 | 类别 | 回归 | 分类 | 特征重要性 | 速度 |
|------|------|------|------|------------|------|
| LightGBM | ML | ✅ | ✅ | ✅ | ⚡⚡⚡ |
| XGBoost | ML | ✅ | ✅ | ✅ | ⚡⚡ |

---

## API 参考

### 数据加载

```python
from src.data.loaders import DataLoader

# 加载原始数据
df = DataLoader.load_raw_data(Path("data/raw/frost-risk-forecast-challenge/stations"))
```

### 数据清洗

```python
from src.data.cleaners import DataCleaner

cleaner = DataCleaner()
df_cleaned = cleaner.clean_pipeline(df)
```

### 特征工程

```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
feature_config = {
    "time_features": True,
    "lag_features": {"enabled": True, "columns": [...], "lags": [1, 3, 6, 12, 24]},
    "rolling_features": {"enabled": True, ...},
    "derived_features": True
}
df_features = engineer.build_feature_set(df_cleaned, feature_config)
```

### 模型使用

```python
from src.models.ml.lightgbm import LightGBMModel

# 创建模型
model = LightGBMModel(config)

# 训练
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # 如果支持

# 特征重要性
importance = model.get_feature_importance()

# 保存/加载
model.save(Path("model_dir"))
loaded_model = LightGBMModel.load(Path("model_dir"))
```

### 评估

```python
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.validators import CrossValidator

# 计算指标
metrics = MetricsCalculator.calculate_all_metrics(
    y_true, y_pred, task_type="regression"
)

# 交叉验证
splits = CrossValidator.leave_one_station_out(df)
```

---

## 配置管理

### 模型配置文件结构

```yaml
model_name: "lightgbm_baseline"
model_type: "lightgbm"
task_type: "regression"

data:
  input_path: "data/interim/features/cimis_features.parquet"
  target_column: "Air Temp (C)"
  feature_columns: []  # 空=自动选择

model_params:
  n_estimators: 100
  learning_rate: 0.05
  max_depth: 6

training:
  validation_strategy: "time_split"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

evaluation:
  metrics:
    regression: ["mae", "rmse", "r2", "mape"]
```

---

## 扩展开发

### 添加新模型

1. 在 `src/models/` 下创建新文件
2. 继承 `BaseModel` 类
3. 实现 `fit()`, `predict()`, `predict_proba()` 方法
4. 创建配置文件
5. 添加单元测试

### 添加新特征

1. 在 `FeatureEngineer` 类中添加新方法
2. 在配置文件中启用该特征
3. 验证特征质量（相关性、重要性）

### 添加新评估指标

1. 在 `MetricsCalculator` 中添加方法
2. 在配置文件中添加到指标列表
3. 自动包含在对比报告中

---

## 📚 相关文档

- **[USER_GUIDE.md](USER_GUIDE.md)**: 用户指南
- **[DATA_DOCUMENTATION.md](DATA_DOCUMENTATION.md)**: 数据文档
- **[FEATURE_GUIDE.md](FEATURE_GUIDE.md)**: 特征工程完整指南
- **[TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md)**: 训练和评估文档

