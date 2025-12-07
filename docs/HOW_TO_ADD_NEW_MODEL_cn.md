# 如何添加新模型到训练系统

本指南说明如何在 `/home/zhengkun-li/frost-risk-forecast-challenge/src/models` 下添加新的模型。

## 系统架构概览

当前系统已支持以下类型的模型：
- **ML 模型** (`src/models/ml/`): LightGBM, XGBoost, CatBoost, Random Forest, ExtraTrees, Linear Models
- **深度学习模型** (`src/models/deep/`): LSTM, GRU, TCN, LSTM Multi-task
- **图神经网络模型** (`src/models/graph/`): DCRNN, ST-GCN, GAT-LSTM, GraphWaveNet
- **传统模型** (`src/models/traditional/`): Prophet

## 添加新模型的步骤

### 步骤 1: 创建模型类文件

根据模型类型，在相应目录下创建模型文件：

- **ML 模型**: `src/models/ml/your_model.py`
- **深度学习模型**: `src/models/deep/your_model.py`
- **图神经网络**: `src/models/graph/your_model.py`
- **传统模型**: `src/models/traditional/your_model.py`

### 步骤 2: 实现模型类

模型类必须继承 `BaseModel` 并实现以下抽象方法：

```python
from src.models.base import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np

class YourModel(BaseModel):
    """Your model description."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model.
        
        Args:
            config: Configuration dictionary containing:
                - model_params: Model-specific parameters
                - task_type: "classification" or "regression"
        """
        super().__init__(config)
        self.task_type = config.get("task_type", "regression")
        model_params = config.get("model_params", {})
        
        # Initialize your model here
        # Example:
        # if self.task_type == "classification":
        #     self.model = YourClassifier(**model_params)
        # else:
        #     self.model = YourRegressor(**model_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "YourModel":
        """Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            **kwargs: Additional training arguments
        
        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_names = list(X.columns)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make point predictions.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (for classification).
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of probabilities [n_samples] for binary classification
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task_type == "classification":
            # For binary classification, return probabilities for positive class
            proba = self.model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]  # Return probability of positive class
            return proba
        else:
            # For regression, return predictions as probabilities
            return self.predict(X)
    
    def save(self, path: Path) -> None:
        """Save model (optional override for custom save logic).
        
        Default implementation saves as pickle, but you can override
        for models that need special handling (e.g., .cbm for CatBoost).
        """
        # Custom save logic if needed
        # Otherwise, use parent class save method
        super().save(path)
    
    def load(self, path: Path) -> "YourModel":
        """Load model (optional override for custom load logic).
        
        Default implementation loads from pickle, but you can override
        for models that need special handling.
        """
        # Custom load logic if needed
        # Otherwise, use parent class load method
        super().load(path)
        return self
```

### 步骤 3: 在 `model_config.py` 中注册模型

需要修改两个地方：

#### 3.1 添加模型到支持的模型列表

在 `src/training/model_config.py` 的 `get_model_params()` 函数中，找到 `supported_models` 列表并添加你的模型名称：

```python
supported_models = [
    "lightgbm", "xgboost", "catboost", "random_forest",
    "ensemble", "lstm", "lstm_multitask", "prophet",
    "extratrees", "linear_regression", "ridge", "elasticnet", "logreg",
    "gru", "tcn", "persistence",
    "dcrnn", "st_gcn", "gat_lstm", "graphwavenet",
    "your_model_name"  # 添加这里
]
```

#### 3.2 添加模型参数配置

在 `get_model_params()` 函数中添加你的模型参数配置：

```python
elif model_type == "your_model_name":
    if task_type == "classification":
        return {
            "param1": value1,
            "param2": value2,
            "random_state": 42,
            "n_jobs": max_workers,
            # ... 其他分类任务参数
        }
    else:  # regression
        return {
            "param1": value1,
            "param2": value2,
            "random_state": 42,
            "n_jobs": max_workers,
            # ... 其他回归任务参数
        }
```

#### 3.3 添加模型类映射

在 `get_model_class()` 函数中添加模型类映射：

```python
elif model_type == "your_model_name":
    from src.models.ml.your_model import YourModel  # 或 deep/ graph/ traditional
    return YourModel
```

### 步骤 4: 测试新模型

创建测试脚本来验证新模型：

```python
# scripts/test/test_your_model.py
from src.models.ml.your_model import YourModel
from src.training.model_config import get_model_config, get_model_class

# Test model initialization
config = get_model_config("your_model_name", horizon=3, task_type="classification")
model_class = get_model_class("your_model_name")
model = model_class(config)

# Test training
import pandas as pd
import numpy as np
X_train = pd.DataFrame(np.random.randn(100, 10))
y_train = pd.Series(np.random.randint(0, 2, 100))
model.fit(X_train, y_train)

# Test prediction
X_test = pd.DataFrame(np.random.randn(10, 10))
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("Model test passed!")
```

## 完整示例：添加 AdaBoost 模型

以下是一个完整的示例，展示如何添加 AdaBoost 模型：

### 1. 创建模型文件 `src/models/ml/adaboost.py`

```python
"""AdaBoost model implementation for frost forecasting."""

from typing import Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.models.base import BaseModel


class AdaBoostModel(BaseModel):
    """AdaBoost implementation for frost forecasting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize AdaBoost model.
        
        Args:
            config: Configuration dictionary with:
                - model_params: AdaBoost parameters
                - task_type: "classification" or "regression"
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not installed. Install it with: pip install scikit-learn"
            )
        
        super().__init__(config)
        
        self.task_type = config.get("task_type", "regression")
        model_params = config.get("model_params", {})
        
        # Ensure consistent random_state
        if "random_state" not in model_params:
            model_params["random_state"] = 42
        
        # Initialize model
        if self.task_type == "classification":
            self.model = AdaBoostClassifier(**model_params)
        else:
            self.model = AdaBoostRegressor(**model_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "AdaBoostModel":
        """Train the model."""
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_names = list(X.columns)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make point predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task_type == "classification":
            proba = self.model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        else:
            return self.predict(X)
```

### 2. 在 `model_config.py` 中注册

```python
# 在 supported_models 列表中添加
supported_models = [
    # ... 其他模型
    "adaboost"  # 添加这里
]

# 在 get_model_params() 中添加参数配置
elif model_type == "adaboost":
    return {
        "n_estimators": n_estimators,
        "learning_rate": 0.1,
        "random_state": 42,
    }

# 在 get_model_class() 中添加类映射
elif model_type == "adaboost":
    from src.models.ml.adaboost import AdaBoostModel
    return AdaBoostModel
```

### 3. 使用新模型训练

添加后，就可以通过 CLI 使用新模型：

```bash
python -m src.cli train single \
    --model-name adaboost \
    --matrix-cell C \
    --track raw \
    --horizon-h 3 \
    --radius-km 100 \
    --output-dir experiments/adaboost/raw/C/radius_100km
```

## 注意事项

1. **模型保存格式**: 
   - 大多数模型使用 `.pkl` 格式（pickle）
   - CatBoost 使用 `.cbm` 格式（需要覆盖 `save()` 和 `load()` 方法）

2. **分类 vs 回归**:
   - `task_type == "classification"` 用于霜冻预测
   - `task_type == "regression"` 用于温度预测
   - 确保模型支持两种任务类型

3. **参数一致性**:
   - 所有模型应该使用 `random_state=42` 保证可复现性
   - 使用 `n_jobs` 或 `thread_count` 参数支持并行训练

4. **内存管理**:
   - 对于内存密集型模型，考虑在 `get_model_params()` 中根据可用内存调整参数
   - 可以使用 `for_loso` 参数提供较小的配置用于内存受限场景

5. **模型文件扩展名**:
   - 如果使用特殊扩展名（如 `.cbm`），需要：
     - 在 `check_models_exist()` 函数中处理（`model_trainer.py`）
     - 在批量训练脚本中正确处理（`batch_train_all_models.py`）

## 常见问题

**Q: 是否需要修改训练脚本？**  
A: 不需要。只要正确注册了模型，现有的训练脚本（`model_trainer.py`, `batch_train_all_models.py`）会自动支持新模型。

**Q: 如何确保模型与现有系统兼容？**  
A: 确保：
1. 继承 `BaseModel` 并实现所有抽象方法
2. 正确设置 `task_type` 和 `is_fitted` 标志
3. `predict_proba()` 对于二进制分类返回一维数组（正类概率）

**Q: 深度学习模型需要特殊处理吗？**  
A: 是的。深度学习模型通常需要：
- 序列长度配置
- 批量大小和训练轮数
- GPU 支持
- 梯度裁剪和学习率调度

参考 `src/models/deep/lstm.py` 作为示例。

## 总结

添加新模型只需要：
1. ✅ 创建模型类（继承 `BaseModel`）
2. ✅ 在 `model_config.py` 中注册（3个地方）
3. ✅ 测试新模型

完成后，新模型就可以通过现有训练管道使用，无需修改其他代码！

