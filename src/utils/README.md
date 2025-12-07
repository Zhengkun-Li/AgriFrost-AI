# Utils Module

工具模块 (`src/utils`) 提供通用的工具函数和类，供其他模块使用。

**注意**: 模型训练工具（ProgressLogger, TrainingHistory, CheckpointManager 等）位于 `src/models/utils/`，详见 [`src/models/README.md`](../models/README.md) 和 [`docs/MODEL_TRAINING_UTILITIES.md`](../../docs/MODEL_TRAINING_UTILITIES.md)。

## 📁 模块结构

```
src/utils/
├── __init__.py          # 模块导出
├── calibration.py       # 概率校准工具
├── hyperopt.py          # 超参数优化
├── losses.py            # 损失函数（PyTorch）
└── path_utils.py        # 路径工具函数
```

## 🔧 核心组件

### 1. ProbabilityCalibrator (`calibration.py`)

概率校准工具，用于改进 Brier Score 和 ECE：

- **Platt Scaling**: 使用逻辑回归校准概率
- **Isotonic Regression**: 使用非参数等渗回归校准概率

**关键特性**:
- ✅ 输入验证（空值检查、形状检查、范围检查）
- ✅ 数值稳定性（clip 防止 log(0)）
- ✅ 优雅降级（sklearn 不可用时禁用）
- ✅ 错误处理（区分可恢复错误和意外错误）

**使用示例**:
```python
from src.utils import ProbabilityCalibrator
import numpy as np

calibrator = ProbabilityCalibrator(method="platt")
calibrator.fit(y_prob_val, y_true_val)
y_prob_calibrated = calibrator.transform(y_prob_test)
```

### 2. HyperparameterOptimizer (`hyperopt.py`)

使用 Hyperopt 进行超参数优化：

- **TPE (Tree-structured Parzen Estimator)**: 贝叶斯优化算法
- **Cross-validation**: 默认使用 CV 作为目标函数
- **自定义目标函数**: 支持自定义优化目标

**关键特性**:
- ✅ 输入验证（DataFrame/Series 空值检查、形状检查）
- ✅ 错误处理（区分参数错误和意外错误）
- ✅ 试验跟踪和汇总
- ✅ 优雅降级（依赖库不可用时抛出清晰错误）

**使用示例**:
```python
from src.utils import HyperparameterOptimizer
from hyperopt import hp

optimizer = HyperparameterOptimizer(
    model_class=LightGBMModel,
    config_template={"model_type": "lightgbm"},
    max_evals=50
)

space = {
    "model_learning_rate": hp.loguniform("learning_rate", -5, -1),
    "model_max_depth": hp.choice("max_depth", [5, 7, 9]),
}

best_params = optimizer.optimize(X, y, space, cv=3)
```

### 3. Loss Functions (`losses.py`)

用于不平衡分类的损失函数：

- **FocalLoss**: 专注于难样本的损失函数
- **WeightedBCEWithLogitsLoss**: 加权二元交叉熵

**关键特性**:
- ✅ 输入验证（形状检查、参数范围检查）
- ✅ 数值稳定性
- ✅ 清晰的文档和公式说明

**使用示例**:
```python
from src.utils import FocalLoss

criterion = FocalLoss(alpha=0.25, gamma=2.0)
loss = criterion(logits, targets)
```

### 4. Path Utilities (`path_utils.py`)

路径工具函数：

- **ensure_dir**: 确保目录存在
- **get_project_root**: 获取项目根目录
- **get_data_dir**: 获取数据目录路径

**关键特性**:
- ✅ 输入验证（None 检查、类型检查）
- ✅ 错误处理（OSError 处理）
- ✅ 清晰的错误消息

**使用示例**:
```python
from src.utils.path_utils import ensure_dir, get_data_dir

output_dir = ensure_dir(Path("experiments/my_run"))
raw_data_dir = get_data_dir("raw")
```

## ✅ 代码质量改进

### 已完成

1. **日志标准化** ✅
   - 所有模块添加了 `_logger`
   - 关键操作记录日志（调试、警告、错误）

2. **错误处理改进** ✅
   - 使用具体异常类型（`ValueError`, `ImportError`, `OSError`）
   - 区分可恢复错误和意外错误
   - 清晰的错误消息

3. **输入验证** ✅
   - 参数范围验证（alpha, gamma, max_evals, cv）
   - 空值检查（DataFrame, Series, arrays）
   - 形状兼容性检查（inputs/targets）
   - 类型验证（data_type, reduction）

4. **数值稳定性** ✅
   - 概率裁剪（clip to [0, 1]）
   - 防止除零和对数域错误

## 📝 使用示例

### 概率校准

```python
from src.utils import ProbabilityCalibrator

# 在验证集上拟合校准器
calibrator = ProbabilityCalibrator(method="platt")
calibrator.fit(y_prob_val, y_true_val)

# 在校验集上应用校准
y_prob_calibrated = calibrator.transform(y_prob_test)
```

### 损失函数

```python
from src.utils import FocalLoss

# 创建 Focal Loss（适合极不平衡数据）
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# 在训练循环中使用
loss = criterion(logits, targets)
loss.backward()
```

### 路径工具

```python
from src.utils.path_utils import ensure_dir, get_data_dir

# 确保输出目录存在
output_dir = ensure_dir(Path("experiments/my_run"))

# 获取数据目录
raw_data_dir = get_data_dir("raw")
```

## ⚠️ 注意事项

1. **依赖要求**:
   - `calibration.py` 需要 `sklearn`（可选）
   - `hyperopt.py` 需要 `hyperopt` 和 `sklearn`（必需）
   - `losses.py` 需要 `torch`（必需）

2. **输入验证**: 所有公共函数都包含输入验证，确保参数有效性

3. **错误处理**: 模块设计为优雅降级（可选依赖）或清晰报错（必需依赖）

## ✅ 代码质量改进

### 最新改进（2025-11-19）

1. **WeightedBCEWithLogitsLoss 修复** ✅
   - 实现了真正的自动 `pos_weight` 计算
   - 在 `forward()` 中根据 batch targets 动态计算
   - 使用 `torch.no_grad()` 避免不必要的梯度计算

2. **FocalLoss 性能优化** ✅
   - 使用 `torch.no_grad()` 包装概率和权重计算
   - 避免重复的 sigmoid 计算带来的梯度开销

3. **Loss 函数安全性** ✅
   - 将 `squeeze()` 改为 `squeeze(-1)` 避免意外删除 batch 维度
   - 更新了文档说明支持的 shape

4. **Path Utilities 改进** ✅
   - `ensure_dir` 参数类型从 `Path` 改为 `Union[str, Path]`
   - 更准确地反映函数实际接受的参数类型

5. **Hyperopt 参数解码** ✅
   - 在 `optimize()` 中保存 space 参数
   - 在 `get_trials_summary()` 中使用 `space_eval` 解码参数
   - 返回真实的超参数值而不是内部索引

## 📊 状态

**模块状态**: ✅ **生产就绪**

**最后更新**: 2025-11-19

所有关键改进已完成：
- ✅ 日志标准化
- ✅ 错误处理
- ✅ 输入验证
- ✅ 数值稳定性
- ✅ Loss 函数 bug 修复
- ✅ 性能优化

