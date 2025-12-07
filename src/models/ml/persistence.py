"""Persistence model for time series: ŷ_{t+h} = y_t, and frost prob from current temp."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

from ..base import BaseModel


class PersistenceModel(BaseModel):
    """Persistence baseline model for time series forecasting.
    
    This is a simple baseline model that predicts future values based on current values.
    It serves as a "naive" baseline for comparison with more sophisticated models.
    
    **Decision Rules:**
    
    1. **Regression (task_type='regression')**:
       - Prediction: ŷ(t+h) = y(t) where h is the forecast horizon
       - In practice: Returns the current temperature value from the specified temperature column
       - This assumes temperature changes slowly (good baseline for short horizons)
    
    2. **Classification (task_type='classification')**:
       - Prediction: Binary decision based on current temperature vs frost threshold
       - Decision rule: frost = 1 if current_temp < frost_threshold, else 0
       - Probability: Uses sigmoid function: P(frost) = sigmoid((threshold - temp) / scale)
       - The `scale` parameter controls the steepness of the sigmoid transition
    
    **Horizon Dependency:**
    - The model does NOT explicitly use `horizon_h` in its prediction logic
    - It assumes y(t+h) ≈ y(t) for all horizons h
    - Performance typically degrades as horizon increases
    - For horizon-specific behavior, consider passing `horizon_h` in `model_params`:
      ```python
      config = {
          "task_type": "regression",
          "model_params": {
              "horizon_h": 12,  # Forecast horizon in hours
              "temp_column": "Air Temp (C)",
              ...
          }
      }
      ```
    
    **Configuration Parameters:**
    - `temp_column` (str): Name of temperature column to use (default: "Air Temp (C)")
    - `frost_threshold` (float): Temperature threshold for frost classification (default: 0.0°C)
    - `scale` (float): Scale factor for sigmoid probability calculation (default: 2.0)
    - `horizon_h` (int, optional): Forecast horizon in hours (for metadata/logging)
    
    **Use Cases:**
    - Baseline comparison: Always compare sophisticated models against persistence
    - Short-horizon forecasting: Often competitive for very short horizons (1-3 hours)
    - Debugging: Simple model to verify pipeline correctness
    - Reference in papers/reports: Standard baseline for time series forecasting
    """
    def __init__(self, config: Dict[str, Any]):
        """Initialize Persistence model.
        
        Args:
            config: Model configuration dictionary with:
                - task_type: "regression" or "classification"
                - model_params: Dictionary with:
                    - temp_column: Temperature column name (default: "Air Temp (C)")
                    - frost_threshold: Frost threshold in °C (default: 0.0)
                    - scale: Sigmoid scale factor (default: 2.0)
                    - horizon_h: Forecast horizon in hours (optional, for metadata)
        """
        super().__init__(config)
        self.task_type = config.get("task_type", "regression")
        params: Dict[str, Any] = config.get("model_params", {}) or {}
        self.temp_column: str = params.get("temp_column", "Air Temp (C)")
        self.frost_threshold: float = float(params.get("frost_threshold", 0.0))
        self.scale: float = float(params.get("scale", 2.0))  # temperature scale for sigmoid
        self.horizon_h: Optional[int] = params.get("horizon_h", None)  # Optional horizon for metadata
        self.is_fitted = True  # no training required
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "PersistenceModel":
        self.feature_names = list(X.columns)
        # No training required
        if self.progress_logger:
            self.progress_logger.log("  ✅ Training (no-op) completed", flush=True)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """For regression: return current temp; for classification: binary decision vs threshold."""
        if self.task_type == "classification":
            temp = self._current_temp(X)
            return (temp < self.frost_threshold).astype(int)
        # regression: return current temp
        return self._current_temp(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict frost probability for classification tasks.
        
        Uses sigmoid function: P(frost) = sigmoid((threshold - temp) / scale)
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Array of frost probabilities (shape: n_samples).
        
        Raises:
            ValueError: If task_type is not 'classification'.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        
        temp = self._current_temp(X)
        # Probability of frost as sigmoid of (threshold - temp)
        # Lower temperature = higher frost probability
        scale = self.scale if self.scale > 0 else 1.0  # Ensure positive scale
        logits = (self.frost_threshold - temp) / max(scale, 1e-6)
        prob_pos = 1.0 / (1.0 + np.exp(-logits))
        return prob_proba_safe(prob_pos)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance for persistence model.
        
        Returns:
            DataFrame with temperature column as the sole important feature.
            Returns None if temperature column is not in feature names.
        """
        if not self.feature_names or self.temp_column not in self.feature_names:
            return None
        return pd.DataFrame({
            "feature": [self.temp_column],
            "importance": [1.0]
        })

    def save(self, path: Path):
        path = Path(path)
        if path.suffix:
            model_dir = path.parent
            model_filename = path.name
        else:
            model_dir = path
            model_filename = "model.pkl"
        model_dir.mkdir(parents=True, exist_ok=True)
        import pickle, json
        with open(model_dir / model_filename, "wb") as f:
            pickle.write(b"persistence")  # marker
        with open(model_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        _logger.info(f"Model saved to {model_dir / model_filename}")

    @classmethod
    def load(cls, path: Path) -> "PersistenceModel":
        if isinstance(path, str):
            path = Path(path)
        if path.is_dir():
            config_path = path / "config.json"
        else:
            config_path = path.with_suffix(".json")
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
        inst = cls(config)
        return inst

    def _current_temp(self, X: pd.DataFrame) -> np.ndarray:
        if self.temp_column in X.columns:
            arr = X[self.temp_column].to_numpy()
        else:
            # fallback to first column
            arr = X.iloc[:, 0].to_numpy()
        return arr.astype(float)

def prob_proba_safe(x: np.ndarray) -> np.ndarray:
    """Ensure 1D positive-class probabilities in [0,1]."""
    x = np.asarray(x).astype(float)
    x = np.clip(x, 0.0, 1.0)
    return x


