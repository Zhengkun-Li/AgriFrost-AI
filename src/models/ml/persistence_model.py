"""Persistence model for time series: ŷ_{t+h} = y_t, and frost prob from current temp."""
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from ..base import BaseModel


class PersistenceModel(BaseModel):
    """Persistence baseline for both regression and classification.
    
    - For regression (predict temp_{t+h}): returns current temperature feature value (e.g., 'Air Temp (C)').
    - For classification (frost probability): uses current temperature vs threshold with a sigmoid or step.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task_type = config.get("task_type", "regression")
        params: Dict[str, Any] = config.get("model_params", {}) or {}
        self.temp_column: str = params.get("temp_column", "Air Temp (C)")
        self.frost_threshold: float = float(params.get("frost_threshold", 0.0))
        self.scale: float = float(params.get("scale", 2.0))  # temperature scale for sigmoid
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

    def predict_proba(self, X: pd.DataFrame):
        """For classification, return positive-class probability via sigmoid on (T_threshold - T_now)/scale."""
        if self.task_type != "classification":
            return None
        temp = self._current_temp(X)
        # probability of frost as sigmoid of (threshold - temp)
        logits = (self.frost_threshold - temp) / max(self.recall_scale(), 1e-6)
        prob_pos = 1.0 / (1.0 + np.exp(-logits))
        return prob_proba_safe(prob_pos)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        # Not applicable; but we can indicate that current temp is sole feature
        if not self.feature_names or self.temp_column not in self.feature_list():
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
        print(f"Model saved to {model_dir / model_filename}")

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

    def recall_scale(self) -> float:
        # ensure positive scale
        return self.scale if self.scale > 0 else 1.0

def prob_proba_safe(x: np.ndarray) -> np.ndarray:
    """Ensure 1D positive-class probabilities in [0,1]."""
    x = np.asarray(x).astype(float)
    x = np.clip(x, 0.0, 1.0)
    return x


