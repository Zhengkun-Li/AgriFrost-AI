# How to Add a New Model to the Training System

This guide explains how to add a new model under `/home/zhengkun-li/frost-risk-forecast-challenge/src/models`.

## System Architecture Overview

The current system supports the following model types:
- **ML Models** (`src/models/ml/`): LightGBM, XGBoost, CatBoost, Random Forest, ExtraTrees, Linear Models
- **Deep Learning Models** (`src/models/deep/`): LSTM, GRU, TCN, LSTM Multi-task
- **Graph Neural Network Models** (`src/models/graph/`): DCRNN, ST-GCN, GAT-LSTM, GraphWaveNet
- **Traditional Models** (`src/models/traditional/`): Prophet

## Steps to Add a New Model

### Step 1: Create Model Class File

Create a model file in the appropriate directory based on model type:

- **ML Model**: `src/models/ml/your_model.py`
- **Deep Learning Model**: `src/models/deep/your_model.py`
- **Graph Neural Network**: `src/models/graph/your_model.py`
- **Traditional Model**: `src/models/traditional/your_model.py`

### Step 2: Implement Model Class

The model class must inherit from `BaseModel` and implement the following abstract methods:

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

### Step 3: Register Model in `model_config.py`

You need to modify two places:

#### 3.1 Add Model to Supported Models List

In the `get_model_params()` function of `src/training/model_config.py`, find the `supported_models` list and add your model name:

```python
supported_models = [
    "lightgbm", "xgboost", "catboost", "random_forest",
    "ensemble", "lstm", "lstm_multitask", "prophet",
    "extratrees", "linear_regression", "ridge", "elasticnet", "logreg",
    "gru", "tcn", "persistence",
    "dcrnn", "st_gcn", "gat_lstm", "graphwavenet",
    "your_model_name"  # Add here
]
```

#### 3.2 Add Model Parameter Configuration

Add your model parameter configuration in the `get_model_params()` function:

```python
elif model_type == "your_model_name":
    if task_type == "classification":
        return {
            "param1": value1,
            "param2": value2,
            "random_state": 42,
            "n_jobs": max_workers,
            # ... other classification task parameters
        }
    else:  # regression
        return {
            "param1": value1,
            "param2": value2,
            "random_state": 42,
            "n_jobs": max_workers,
            # ... other regression task parameters
        }
```

#### 3.3 Add Model Class Mapping

Add model class mapping in the `get_model_class()` function:

```python
elif model_type == "your_model_name":
    from src.models.ml.your_model import YourModel  # or deep/ graph/ traditional
    return YourModel
```

### Step 4: Test New Model

Create a test script to verify the new model:

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

## Complete Example: Adding AdaBoost Model

The following is a complete example showing how to add an AdaBoost model:

### 1. Create Model File `src/models/ml/adaboost.py`

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

### 2. Register in `model_config.py`

```python
# Add to supported_models list
supported_models = [
    # ... other models
    "adaboost"  # Add here
]

# Add parameter configuration in get_model_params()
elif model_type == "adaboost":
    return {
        "n_estimators": n_estimators,
        "learning_rate": 0.1,
        "random_state": 42,
    }

# Add class mapping in get_model_class()
elif model_type == "adaboost":
    from src.models.ml.adaboost import AdaBoostModel
    return AdaBoostModel
```

### 3. Train with New Model

After adding, you can use the new model via CLI:

```bash
python -m src.cli train single \
    --model-name adaboost \
    --matrix-cell C \
    --track raw \
    --horizon-h 3 \
    --radius-km 100 \
    --output-dir experiments/adaboost/raw/C/radius_100km
```

## Important Notes

1. **Model Save Format**: 
   - Most models use `.pkl` format (pickle)
   - CatBoost uses `.cbm` format (need to override `save()` and `load()` methods)

2. **Classification vs Regression**:
   - `task_type == "classification"` for frost prediction
   - `task_type == "regression"` for temperature prediction
   - Ensure model supports both task types

3. **Parameter Consistency**:
   - All models should use `random_state=42` for reproducibility
   - Use `n_jobs` or `thread_count` parameters to support parallel training

4. **Memory Management**:
   - For memory-intensive models, consider adjusting parameters in `get_model_params()` based on available memory
   - Can use `for_loso` parameter to provide smaller configuration for memory-constrained scenarios

5. **Model File Extensions**:
   - If using special extensions (e.g., `.cbm`), need to:
     - Handle in `check_models_exist()` function (`model_trainer.py`)
     - Properly handle in batch training scripts (`batch_train_all_models.py`)

## Frequently Asked Questions

**Q: Do I need to modify training scripts?**  
A: No. As long as the model is correctly registered, existing training scripts (`model_trainer.py`, `batch_train_all_models.py`) will automatically support the new model.

**Q: How to ensure model compatibility with existing system?**  
A: Ensure:
1. Inherit from `BaseModel` and implement all abstract methods
2. Correctly set `task_type` and `is_fitted` flags
3. `predict_proba()` returns one-dimensional array (positive class probability) for binary classification

**Q: Do deep learning models need special handling?**  
A: Yes. Deep learning models typically need:
- Sequence length configuration
- Batch size and training epochs
- GPU support
- Gradient clipping and learning rate scheduling

Refer to `src/models/deep/lstm.py` as an example.

## Summary

Adding a new model only requires:
1. ✅ Create model class (inherit from `BaseModel`)
2. ✅ Register in `model_config.py` (3 places)
3. ✅ Test new model

After completion, the new model can be used through existing training pipelines without modifying other code!

---

**Last Updated**: 2025-12-06  
**Document Version**: 1.0
