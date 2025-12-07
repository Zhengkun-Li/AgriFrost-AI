"""ExtraTrees model implementation for frost forecasting (classification/regression)."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    SKLEARN_AVAILABLE = True
except Exception as e:
    SKLEARN_AVAILABLE = False

from ..base import BaseModel


class ExtraTreesModel(BaseModel):
    """ExtraTrees model for classification/regression."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: {
                "task_type": "classification" | "regression",
                "model_params": { ... }  # passed to sklearn ExtraTrees* ctor
            }
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

        super().__init__(config)
        model_params = config.get("model_params", {})
        self.task_type = config.get("task_type", "regression")
        
        # Ensure consistent random_state (default: 42 for reproducibility)
        if "random_state" not in model_params:
            model_params["random_state"] = 42
        
        # Ensure consistent n_jobs (default: -1 for all CPUs)
        if "n_jobs" not in model_params:
            model_params["n_jobs"] = -1

        if self.task_type == "classification":
            self.model = ExtraTuresClassifierWrapper(**model_params)
        else:
            self.model = ExtraTreesRegressor(**model_params)

        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ExtraTreesModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        if self.progress_logger:
            self.progress_logger.log("  ✅ Training completed", flush=True)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        preds = self.model.predict(X)
        if self.task_type == "classification":
            # ensure integer 0/1
            return (preds > 0.5).astype(int) if preds.ndim == 1 else np.argmax(preds, axis=1)
        return preds

    def predict_proba(self, X: pd.DataFrame):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.task_type != "classification":
            return None
        proba = self.model.predict_proba(X)
        # Return positive class probability for binary classification
        if isinstance(proba, np.ndarray):
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
        return None

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if not hasattr(self.model, "feature_importances_"):
            return None
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        })
        return df.sort_values("importance", ascending=False)

    def save(self, path: Path):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        path = Path(path)
        if path.suffix:
            model_dir = path.parent
            model_filename = path.name
        else:
            model_dir = path
            model_filename = "model.pkl"
        model_dir.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(model_dir / model_filename, "wb") as f:
            pickle.dump(self.model, f)
        # Save config
        import json
        with open(model_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        _logger.info(f"Model saved to {model_dir / model_filename}")

    @classmethod
    def load(cls, path: Path) -> "ExtraTreesModel":
        if isinstance(path, str):
            path = Path(path)
        if path.is_dir():
            model_path = path / "model.pkl"
            config_path = path / "config.json"
        else:
            model_path = path
            config_path = path.with_suffix(".json")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        import pickle, json
        with open(config_path, "r") as f:
            config = json.load(f)
        inst = cls(config)
        with open(model_path, "rb") as f:
            inst.model = pickle.load(f)
        inst.is_fitted = True
        return inst


class ExtraTuresClassifierWrapper(ExtraTreesClassifier):
    """Wrapper subclass to ensure predict() returns class labels correctly for binary tasks."""
    def predict(self, X):
        proba = super().predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] == 2:
            return (proba[:, 1] >= 0.5).astype(int)
        # Multiclass - defer to base implementation
        return super().predict(X)


