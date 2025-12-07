"""XGBoost model implementation for frost forecasting."""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from ..base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model for frost forecasting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize XGBoost model.
        
        Args:
            config: Model configuration dictionary.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
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
            self.model = xgb.XGBClassifier(**model_params)
        else:
            self.model = xgb.XGBRegressor(**model_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "XGBoostModel":
        """Train the XGBoost model.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            **kwargs: Additional arguments:
                - eval_set: List of (X_val, y_val) tuples for validation
                - eval_names: List of names for eval sets
                - eval_metric: Metric for evaluation
                - callbacks: List of callbacks (e.g., early_stopping)
                - checkpoint_dir: Optional directory for saving checkpoints
                - log_file: Optional path for training log file
        
        Returns:
            Self for method chaining.
        """
        # Setup training tools if requested
        checkpoint_dir = kwargs.pop('checkpoint_dir', None)
        log_file = kwargs.pop('log_file', None)
        if checkpoint_dir or log_file:
            model_params = self.config.get("model_params", {})
            checkpoint_frequency = model_params.get("checkpoint_frequency", 0)
            self.setup_training_tools(
                checkpoint_dir=checkpoint_dir,
                log_file=log_file,
                checkpoint_frequency=checkpoint_frequency,
                save_best=True,
                best_metric="val_loss" if self.task_type == "regression" else "val_auc",
                best_mode="min" if self.task_type == "regression" else "max"
            )
            if self.progress_logger:
                self.progress_logger.on_training_start(
                    model_name="XGBoost",
                    config={
                        "task_type": self.task_type,
                        "n_estimators": model_params.get("n_estimators", "default"),
                        "learning_rate": model_params.get("learning_rate", "default")
                    }
                )
        
        self.feature_names = list(X.columns)
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Handle eval_set conversion
        if "eval_set" in kwargs:
            eval_set = kwargs["eval_set"]
            eval_set = [
                (x_val if isinstance(x_val, np.ndarray) else np.array(x_val),
                 y_val if isinstance(y_val, np.ndarray) else np.array(y_val))
                for x_val, y_val in eval_set
            ]
            kwargs["eval_set"] = eval_set
        
        self.model.fit(X_array, y_array, **kwargs)
        
        # Extract training history from XGBoost (if available)
        if self.training_history and hasattr(self.model, 'evals_result'):
            evals_result = self.model.evals_result()
            if evals_result:
                # XGBoost stores results per eval set
                for eval_name, metrics in evals_result.items():
                    for metric_name, values in metrics.items():
                        for i, value in enumerate(values):
                            self.training_history.record_epoch(
                                epoch=i + 1,
                                **{f"{eval_name}_{metric_name}": value}
                            )
        
        # Save training artifacts if tools were set up
        if checkpoint_dir and self.training_history and len(self.training_history) > 0:
            self.save_training_artifacts(Path(checkpoint_dir))
        
        if self.progress_logger:
            self.progress_logger.log("  ✅ Training completed", flush=True)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make point predictions.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Predictions array.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_array)
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Make probability predictions (for classification).
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Probability predictions array, or None for regression.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task_type == "classification":
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            return self.model.predict_proba(X_array)[:, 1]  # Return positive class probability
        else:
            # For regression, raise error (consistent with other models)
            raise ValueError("predict_proba is only available for classification tasks")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance.
        
        Returns:
            DataFrame with feature names and importance scores.
        """
        if not self.is_fitted:
            return None
        
        try:
            importance = self.model.feature_importances_
            if self.feature_names:
                return pd.DataFrame({
                    "feature": self.feature_names,
                    "importance": importance
                }).sort_values("importance", ascending=False)
            else:
                return pd.DataFrame({
                    "feature": [f"feature_{i}" for i in range(len(importance))],
                    "importance": importance
                }).sort_values("importance", ascending=False)
        except Exception:
            return None
    
    def save(self, path: Path) -> None:
        """Save model to disk.
        
        Args:
            path: Directory path to save model.
        """
        import pickle
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        model_path = path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            "model_type": "xgboost",
            "task_type": self.task_type,
            "feature_names": self.feature_names,
            "config": self.config
        }
        
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, path: Path) -> "XGBoostModel":
        """Load model from disk.
        
        Args:
            path: Directory path containing saved model.
        
        Returns:
            Loaded model instance.
        """
        import pickle
        path = Path(path)
        
        model_path = path / "model.pkl"
        metadata_path = path / "metadata.pkl"
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        instance = cls(metadata["config"])
        instance.model = model
        instance.feature_names = metadata.get("feature_names", [])
        instance.is_fitted = True
        
        return instance

