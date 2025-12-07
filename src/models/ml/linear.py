"""Linear/Logistic/Ridge/ElasticNet models for regression/classification on raw features."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, LogisticRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

from ..base import BaseModel


class LinearModel(BaseModel):
    """Generic linear-family model for regression/classification.
    
    **Important Notes:**
    - **Feature Scaling**: Linear models (especially Ridge and ElasticNet) assume features are 
      standardized/normalized. Features should be preprocessed with StandardScaler or MinMaxScaler 
      before training for optimal performance. The model itself does NOT perform feature scaling.
    
    - **Regression vs Classification**: The model type is determined by `task_type`:
      - `task_type='regression'`: Uses LinearRegression, Ridge, or ElasticNet
      - `task_type='classification'`: Uses LogisticRegression with different penalties
    
    **Supported Model Types:**
    
    For regression (task_type='regression'):
      - `model_type='linear_regression'` => LinearRegression (no regularization)
      - `model_type='ridge'` => Ridge (L2 regularization)
      - `model_type='elasticnet'` => ElasticNet (L1+L2 regularization)
    
    For classification (task_type='classification'):
      - `model_type='logreg'` => LogisticRegression (default solver='lbfgs')
      - `model_type='ridge'` => LogisticRegression(penalty='l2')
      - `model_type='elasticnet'` => LogisticRegression(penalty='elasticnet', solver='saga')
      - `model_type='linear_regression'` => LogisticRegression(penalty='l2') (as baseline)
    
    **Feature Importance:**
    - Returns absolute values of coefficients (`|coef_|`) as importance scores
    - For multi-class, coefficients are flattened and absolute values taken
    """

    def __init__(self, config: Dict[str, Any]):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        super().__init__(config)
        self.task_type = config.get("task_type", "regression")
        params = dict(config.get("model_params", {}))
        mtype = config.get("model_type", "linear_regression")

        # choose estimator based on task + model type
        if self.task_type == "classification":
            if mtype == "ridge":
                C = 1.0 / max(1e-8, float(params.get("alpha", 1.0)))
                self.model = LogisticRegression(penalty="l2", C=C, max_iter=int(params.get("max_iter", 200)), n_jobs=params.get("n_jobs", None))
            elif mtype == "elasticnet":
                self.model = LogisticRegression(
                    penalty="elastic", solver="saga",
                    l1_ratio=float(params.get("l1_ratio", 0.5)),
                    C=float(params.get("C", 1.0)),
                    max_iter=int(params.get("max_iter", 500)),
                    n_jobs=int(params.get("n_jobs", -1)),
                )
            else:
                # default to standard logistic regression
                self.model = LogisticRegression(max_iter=int(params.get("max_iter", 200)), n_jobs=params.get("n_jobs", None))
        else:
            # regression
            if mtype == "ridge":
                self.model = Ridge(alpha=float(params.get("alpha", 1.0)), random_state=42)
            elif mtype == "elasticnet":
                self.model = ElasticNet(alpha=float(params.get("alpha", 0.5)), l1_ratio=float(params.get("l1_ratio", 0.5)))
            else:
                self.model = LinearRegression()

        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "LinearModel":
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
            # some solvers return class labels via predict, but we ensure 0/1
            if preds.ndim == 1:
                return (preds > 0.5).astype(int)
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (classification only).
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Probability array. For binary classification, returns positive class probability (1D array).
        
        Raises:
            ValueError: If task_type is not 'classification' or model doesn't support predict_proba.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Model does not support predict_proba")
        
        proba = self.model.predict_proba(X)
        if isinstance(proba, np.ndarray):
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]  # Return positive class probability for binary
            elif proba.ndim == 2:
                return proba  # Return full probability matrix for multi-class
            else:
                return proba.flatten()
        else:
            raise ValueError(f"Unexpected predict_proba output type: {type(proba)}")

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        coef = None
        if hasattr(self.model, "coef_"):
            coef_arr = getattr(self.model, "coef_")
            if coef_arr is not None:
                coef = np.ravel(coef_arr) if hasattr(coef_arr, "ravel") else coef_arr
        if coef is None:
            return None
        return pd.DataFrame({"feature": self.feature_names, "importance": np.abs(coef)})

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
        import pickle, json
        with open(model_dir / model_filename, "wb") as f:
            pickle.dump(self.model, f)
        with open(model_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        _logger.info(f"Model saved to {model_dir / model_filename}")

    @classmethod
    def load(cls, path: Path) -> "LinearModel":
        if isinstance(path, str):
            path = Path(path)
        if path.is_dir():
            model_path = path / "model.pkl"
            config_path = path / "config.json"
        else:
            model_path = path
            config_path = path.with_suffix(".json")
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")
        import pickle, json
        with open(config_path, "r") as f:
            config = json.load(f)
        inst = cls(config)
        with open(model_path, "rb") as f:
            inst.model = pickle.load(f)
        inst.is_fitted = True
        return inst


