# Models Module

模型模块 (`src/models`) 提供所有用于霜冻风险预测的模型实现。

## 📁 模块结构

```
src/models/
├── __init__.py           # 模块导出
├── base.py               # 基础模型接口
├── registry.py           # 模型注册表
├── deep/                 # 深度学习模型
│   ├── lstm.py
│   ├── lstm_multitask.py
│   ├── gru.py
│   └── tcn.py
├── graph/                # 图神经网络模型
│   ├── base_graph.py
│   ├── dcrnn.py
│   ├── st_gcn.py
│   ├── gat_lstm.py
│   └── graphwavenet.py
├── ml/                   # 机器学习模型
│   ├── lightgbm.py
│   ├── xgboost.py
│   ├── catboost.py
│   ├── random_forest.py
│   ├── extratrees.py
│   ├── linear.py
│   ├── ensemble_model.py
│   └── persistence.py
├── traditional/          # 传统时间序列模型
│   └── prophet.py
└── utils/                # 模型训练工具
    ├── checkpoint_manager.py    # 检查点管理（GPU/CPU兼容、best-k保存、resume训练）
    ├── config_validator.py      # 配置验证（2×2+1框架规则、strict/fallback模式）
    ├── curve_plotter.py         # 训练曲线绘制（TrainingHistory集成、路径规范化）
    ├── graph_builder.py         # 图构建工具（2×2+1兼容、缓存校验、metadata导出）
    ├── progress_logger.py       # 进度日志（rotate/truncate、flush优化、字段统一）
    └── training_history.py      # 训练历史记录（metrics统一、duration精度、字段统一）
```

## 🔧 核心组件

### 1. BaseModel (`base.py`)

所有模型的基础抽象类，定义了统一的接口：

- **fit**: 训练模型
- **predict**: 点预测
- **predict_proba**: 概率预测（分类任务）
- **save/load**: 模型序列化
- **get_feature_importance**: 特征重要性（可选）

**关键特性**:
- ✅ 输入验证（path 验证、is_fitted 检查）
- ✅ 文件操作错误处理（IOError, OSError, pickle.UnpicklingError）
- ✅ 日志标准化（替换 print() 为 logging）
- ✅ 训练工具支持（history, checkpoint, progress logger）

**使用示例**:
```python
from src.models import BaseModel

class MyModel(BaseModel):
    def fit(self, X, y, **kwargs):
        # Implementation
        self.is_fitted = True
        return self
    
    def predict(self, X):
        # Implementation
        return predictions
    
    def predict_proba(self, X):
        # Implementation
        return probabilities

# Save and load
model = MyModel(config)
model.fit(X_train, y_train)
model.save("models/my_model")

loaded_model = MyModel.load("models/my_model")
```

### 2. Model Registry (`registry.py`)

模型注册表，用于动态注册和检索模型类：

- **register_model**: 注册模型类
- **get_model_class**: 获取模型类（支持 legacy 映射）

**关键特性**:
- ✅ 输入验证（名称和类验证）
- ✅ 清晰的错误消息（列出可用模型）
- ✅ 日志记录（注册和覆盖警告）

**使用示例**:
```python
from src.models.registry import register_model, get_model_class

# Register custom model
register_model("my_model", MyModelClass)

# Get model class
ModelClass = get_model_class("my_model")
model = ModelClass(config)
```

### 3. Deep Learning Models (`deep/`)

深度学习模型实现：

- **LSTM**: Long Short-Term Memory
- **LSTM Multitask**: 多任务 LSTM（同时预测温度和霜冻）
- **GRU**: Gated Recurrent Unit
- **TCN**: Temporal Convolutional Network

**关键特性**:
- ✅ 日志标准化（训练信息使用 logging）
- ✅ 错误处理改进（ImportError 处理）
- ✅ PyTorch 集成
- ✅ 序列数据处理

### 4. Graph Neural Network Models (`graph/`)

图神经网络模型实现：

- **DCRNN**: Diffusion Convolutional Recurrent Neural Network
- **ST-GCN**: Spatial-Temporal Graph Convolutional Network
- **GAT-LSTM**: Graph Attention Network + LSTM
- **GraphWaveNet**: Graph Wavelet Neural Network

**关键特性**:
- ✅ 错误处理改进（文件系统错误 vs 意外错误）
- ✅ 图缓存机制
- ✅ 多站空间建模
- ✅ PyTorch Geometric 集成

### 5. Machine Learning Models (`ml/`)

传统机器学习模型实现：

- **Tree-based**: LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees
- **Linear**: Linear Regression, Ridge, Lasso
- **Ensemble**: Voting, Stacking
- **Persistence**: 基准模型

**关键特性**:
- scikit-learn 集成
- 特征重要性支持
- 快速训练和推理

### 6. Traditional Models (`traditional/`)

传统时间序列模型：

- **Prophet**: Facebook Prophet（需要 Date 列）

**关键特性**:
- 时间序列特定处理
- 季节性建模

### 7. Model Utils (`utils/`)

模型训练和监控工具，提供统一的训练支持：

#### 7.1 ProgressLogger (`progress_logger.py`)

统一的训练进度日志接口：

**关键特性**:
- ✅ 双日志模式（brief log + detailed log）
- ✅ 日志自动轮转（默认 100MB，防止日志过大）
- ✅ 优化的 flush 机制（累计行数达阈值再 flush，默认 10）
- ✅ tqdm 配置化（mininterval 可配置，默认 1.0s）
- ✅ 字段统一（与 TrainingHistory 对齐：train_loss, val_loss, learning_rate, epoch_time）

**使用示例**:
```python
from src.models.utils import ProgressLogger
from pathlib import Path

logger = ProgressLogger(
    flush_interval=10,
    max_log_size_mb=100.0,
    use_metric_schema=True
)
logger.bind_files(
    brief_path=Path("training.log"),
    detailed_path=Path("training_detailed.log")
)

logger.on_training_start("LightGBM", device="cpu")
logger.on_epoch(
    epoch=1, total_epochs=100,
    train_loss=0.5, val_loss=0.45,
    learning_rate=0.01, epoch_time=12.5
)
```

#### 7.2 TrainingHistory (`training_history.py`)

训练历史记录和指标追踪：

**关键特性**:
- ✅ 字段统一（与 ProgressLogger 对齐）
- ✅ metrics 列表定义预期指标（防止自动增长新字段）
- ✅ epoch_time 标准字段（不在 kwargs 中）
- ✅ 精确的 duration 计算（使用 sum(epoch_times)）
- ✅ load() 时 metrics 过滤（只加载 expected_metrics）

**使用示例**:
```python
from src.models.utils import TrainingHistory
from pathlib import Path

history = TrainingHistory(metrics=['train_loss', 'val_loss', 'learning_rate', 'epoch_time'])
history.start_training()

for epoch in range(100):
    # Train...
    history.record_epoch(
        epoch=epoch + 1,
        train_loss=0.5,
        val_loss=0.45,
        learning_rate=0.01,
        epoch_time=12.5
    )

history.save(Path("training_history.json"))

# Load
loaded_history = TrainingHistory.load(Path("training_history.json"))
```

#### 7.3 CheckpointManager (`checkpoint_manager.py`)

模型检查点管理：

**关键特性**:
- ✅ GPU/CPU 兼容（保存时转 CPU，加载时确保 CPU）
- ✅ best-k 保存（keep_top_k，默认 3，自动管理）
- ✅ resume 训练支持（完整的恢复功能）
- ✅ checkpoint metadata 暴露（轻量级检查点信息）

**使用示例**:
```python
from src.models.utils import CheckpointManager
from pathlib import Path

checkpoint_mgr = CheckpointManager(
    checkpoint_dir=Path("checkpoints"),
    checkpoint_frequency=10,
    save_best=True,
    best_metric="val_loss",
    keep_top_k=3
)

# Save checkpoint
checkpoint_mgr.save_checkpoint(
    epoch=epoch,
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict(),
    metrics={"val_loss": 0.45}
)

# Save best
checkpoint_mgr.save_best_checkpoint(
    epoch=epoch,
    model_state=model.state_dict(),
    metric_value=0.45
)

# Resume training
resume_info = checkpoint_mgr.resume_training(epoch=50)
if resume_info:
    model.load_state_dict(resume_info['model_state'])
    optimizer.load_state_dict(resume_info['optimizer_state'])
```

#### 7.4 GraphBuilder (`graph_builder.py`)

图结构构建工具：

**关键特性**:
- ✅ 2×2+1 框架兼容（graph metadata 导出到 run_metadata.json）
- ✅ 增强的缓存校验（station_ids + coords hash）
- ✅ 自动 metadata 导出（graph_type/graph_param → radius_km/knn_k）

**使用示例**:
```python
from src.models.utils import GraphBuilder
from pathlib import Path

builder = GraphBuilder(metadata_path=Path("station_metadata.json"))

# Build radius graph
graph = builder.build_radius_graph(radius_km=50.0, edge_weight="gaussian")

# Save with metadata export
GraphBuilder.save_graph(
    graph, 
    path=Path("models/dcrnn/graph.pkl"),
    metadata_path=Path("models/dcrnn/run_metadata.json")
)
```

#### 7.5 ConfigValidator (`config_validator.py`)

配置验证工具：

**关键特性**:
- ✅ 2×2+1 框架规则验证（A/B/E 禁止 radius, C/D 必须 radius, E 必须 knn_k）
- ✅ ExperimentMetadata 强校验
- ✅ strict/fallback 模式（strict_mode=True 拒绝未知 key）

**使用示例**:
```python
from src.config.schema.validator import ConfigValidator

# Validate experiment metadata
valid, msg = ConfigValidator.validate_experiment_metadata(
    matrix_cell='C',
    track='raw',
    horizon_h=12,
    model_name='dcrnn',
    radius_km=50.0  # Required for C/D cells
)

if not valid:
    raise ValueError(f"Invalid config: {msg}")

# Validate training args (strict mode)
valid, msg = ConfigValidator.validate_training_args(
    model_type='lstm',
    checkpoint_dir=Path("checkpoints"),
    strict_mode=True,
    **kwargs
)
```

#### 7.6 TrainingCurvePlotter (`curve_plotter.py`)

训练曲线可视化：

**关键特性**:
- ✅ TrainingHistory 集成（直接接受 TrainingHistory 实例）
- ✅ 路径规范化（model_dir/curves/loss.png）
- ✅ 与 visualization 模块统一

**使用示例**:
```python
from src.visualization.plots import plot_training_curves
from pathlib import Path

# Stateless function (not a class)
plot_training_curves(
    history=training_history,  # TrainingHistory instance
    save_path=Path("models/lstm/curves/training_curves.png"),
    title="Training Curves"
)
```

## ✅ 代码质量改进

### 已完成（核心文件）

1. **日志标准化** ✅
   - `base.py`: 替换 print() 为 logging
   - `deep/tcn.py`: 替换 print() 为 logging
   - `deep/gru.py`: 替换 print() 为 logging
   - `deep/lstm.py`: 替换 print() 为 logging
   - `registry.py`: 添加 logging

2. **错误处理改进** ✅
   - `registry.py`: 使用具体异常类型，清晰的错误消息
   - `base.py`: 文件操作错误处理（IOError, OSError, pickle.UnpicklingError）
   - `graph/dcrnn.py`: 替换 bare `except:` 为具体异常（ValueError, TypeError）
   - `graph/base_graph.py`: 区分文件系统错误和意外错误
   - `deep/tcn.py`: 使用 ImportError

3. **输入验证** ✅
   - `registry.py`: 名称和类验证
   - `base.py`: path 验证，is_fitted 检查

### 最新改进（2025-11-19）

#### Model Utils 全面改进 ✅

所有 `utils/` 工具类已完成全面审查和改进：

1. **ProgressLogger + TrainingHistory 字段统一** ✅
   - 统一字段命名（train_loss, val_loss, learning_rate, epoch_time）
   - ProgressLogger.on_epoch() 与 TrainingHistory.record_epoch() 对齐

2. **GraphBuilder + ConfigValidator 2×2+1 框架兼容** ✅
   - GraphBuilder.save_graph(): 导出 graph metadata 到 run_metadata.json
   - ConfigValidator.validate_experiment_metadata(): 强制矩阵规则验证
   - get_graph_cache_path(): 增强缓存校验（station_ids + coords hash）

3. **ProgressLogger 优化** ✅
   - 日志自动轮转（默认 100MB）
   - flush 机制优化（累计行数达阈值再 flush）
   - tqdm 配置化（mininterval 可配置）

4. **TrainingHistory 改进** ✅
   - metrics 列表定义预期指标（防止自动增长）
   - epoch_time 标准字段（不在 kwargs 中）
   - duration 精度提升（使用 sum(epoch_times)）
   - load() 时 metrics 过滤（只加载 expected_metrics）

5. **CheckpointManager 增强** ✅
   - GPU/CPU 兼容（保存时转 CPU，加载时确保 CPU）
   - best-k 保存（keep_top_k，默认 3）
   - resume 训练支持（完整的恢复功能）
   - checkpoint metadata 暴露

6. **TrainingCurvePlotter 集成** ✅
   - TrainingHistory 集成（直接接受实例）
   - 路径规范化（model_dir/curves/loss.png）
   - 与 visualization 模块统一

### 待改进（其他文件）

以下文件可能仍有 `print()` 和 `except Exception`，但由于模块较大（13000+ 行），建议按需改进：

- `deep/lstm_multitask.py`: 可能有 print()
- `ml/*.py`: 可能有 except Exception
- `traditional/*.py`: 可能需要进一步审查

## 📝 使用示例

### 创建自定义模型

```python
from src.models import BaseModel
import pandas as pd
import numpy as np

class MyCustomModel(BaseModel):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        # Training logic
        self.model = trained_model
        self.is_fitted = True
        self.feature_names = list(X.columns)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        # Prediction logic
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        # Probability prediction logic
        return probabilities
```

### 使用模型注册表

```python
from src.models.registry import register_model, get_model_class

# Register model
register_model("my_model", MyCustomModel)

# Get and use
ModelClass = get_model_class("my_model")
model = ModelClass(config={"model_params": {...}})
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 使用训练工具

```python
from src.models import BaseModel

model = MyModel(config)

# Setup training utilities
model.setup_training_tools(
    checkpoint_dir=Path("checkpoints"),
    log_file=Path("training.log"),
    checkpoint_frequency=10,
    save_best=True,
    best_metric="val_loss"
)

# Train (model will use these tools internally)
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

# Save training artifacts
model.save_training_artifacts(Path("output"))
```

## ⚠️ 注意事项

1. **模型接口**: 所有模型必须继承 `BaseModel` 并实现 `fit`, `predict`, `predict_proba`

2. **特征名称**: 模型应该保存 `self.feature_names` 以便在推理时验证特征一致性

3. **训练工具**: 使用 `setup_training_tools()` 可以获得一致的训练监控和检查点功能

4. **模型保存**: `save()` 方法会保存模型和配置，`load()` 用于恢复

5. **依赖要求**:
   - 深度学习模型需要 `torch`
   - 图神经网络模型需要 `torch` 和 `torch_geometric`
   - 机器学习模型需要 `scikit-learn`, `lightgbm`, `xgboost`, `catboost` 等

## 📊 状态

**核心文件状态**: ✅ **生产就绪**

**最后更新**: 2025-11-19

已完成核心文件的改进：
- ✅ `base.py`
- ✅ `registry.py`
- ✅ `deep/tcn.py`
- ✅ `deep/gru.py`
- ✅ `deep/lstm.py`
- ✅ `graph/dcrnn.py`
- ✅ `graph/base_graph.py`
- ✅ `utils/progress_logger.py` (全面改进)
- ✅ `utils/training_history.py` (全面改进)
- ✅ `utils/graph_builder.py` (全面改进)
- ✅ `utils/config_validator.py` (全面改进)
- ✅ `utils/checkpoint_manager.py` (全面改进)
- ✅ `utils/curve_plotter.py` (全面改进)

**其他文件**: ⚠️ **部分审查**

由于模块较大（13000+ 行），其他文件（`ml/`, `traditional/`）可能需要按需进一步改进。**Model Utils 已全面审查并改进完成** ✅

