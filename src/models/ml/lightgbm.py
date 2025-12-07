"""LightGBM model implementation for frost forecasting."""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from src.models.base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM implementation for frost forecasting.
    
    Supports both regression (temperature prediction) and classification (frost probability).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LightGBM model.
        
        Args:
            config: Configuration dictionary with:
                - model_params: LightGBM parameters (n_estimators, learning_rate, etc.)
                - task_type: "regression" or "classification" (default: "regression")
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is not installed. Install it with: pip install lightgbm"
            )
        
        super().__init__(config)
        
        # Determine task type
        self.task_type = config.get("task_type", "regression")
        
        # Get model parameters with unified defaults
        model_params = config.get("model_params", {})
        
        # Ensure consistent random_state (default: 42 for reproducibility)
        if "random_state" not in model_params:
            model_params["random_state"] = 42
        
        # Ensure consistent n_jobs (default: -1 for all CPUs)
        if "n_jobs" not in model_params:
            model_params["n_jobs"] = -1
        
        # Create model
        if self.task_type == "classification":
            self.model = lgb.LGBMClassifier(**model_params)
        else:
            self.model = lgb.LGBMRegressor(**model_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "LightGBMModel":
        """Train the LightGBM model.
        
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
                    model_name="LightGBM",
                    config={
                        "task_type": self.task_type,
                        "n_estimators": model_params.get("n_estimators", "default"),
                        "learning_rate": model_params.get("learning_rate", "default")
                    }
                )
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Convert to numpy if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Handle eval_set conversion
        if "eval_set" in kwargs:
            eval_set = kwargs["eval_set"]
            # Convert eval_set tuples to numpy arrays
            eval_set = [
                (x_val if isinstance(x_val, np.ndarray) else np.array(x_val),
                 y_val if isinstance(y_val, np.ndarray) else np.array(y_val))
                for x_val, y_val in eval_set
            ]
            kwargs["eval_set"] = eval_set
        
        # Train model
        self.model.fit(X_array, y_array, **kwargs)
        
        # Extract training history from LightGBM (if available)
        if self.training_history and hasattr(self.model, 'evals_result_'):
            evals_result = self.model.evals_result_
            if evals_result:
                # LightGBM stores results per eval set
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
            Array of predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_array)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities.
        
        For regression tasks, converts predictions to probabilities using a threshold.
        For classification tasks, returns class probabilities.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Array of probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task_type == "classification":
            # Return probabilities for classification
            return self.model.predict_proba(X)[:, 1] if len(self.model.classes_) == 2 else self.model.predict_proba(X)
        else:
            # For regression, raise error (consistent with other models)
            raise ValueError("predict_proba is only available for classification tasks")
    
    def _temperature_to_probability(self, temperatures: np.ndarray) -> np.ndarray:
        """Convert temperature predictions to frost probability.
        
        This is a simple heuristic. In practice, should use proper calibration.
        
        Args:
            temperatures: Temperature predictions.
        
        Returns:
            Frost probabilities.
        """
        # Simple sigmoid-like function: lower temp = higher probability
        # Frost threshold around 0°C
        threshold = 0.0
        # Scale factor (adjust based on domain knowledge)
        scale = 2.0
        
        probabilities = 1 / (1 + np.exp(scale * (temperatures - threshold)))
        return probabilities
    
    def get_feature_importance(self, importance_type: str = "gain") -> Optional[pd.DataFrame]:
        """Return feature importance.
        
        Args:
            importance_type: Type of importance ("gain", "split", "gain_by_feature").
        
        Returns:
            DataFrame with feature importance, or None if not available.
        """
        if not self.is_fitted:
            return None
        
        importances = self.model.feature_importances_
        
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()
    
    def set_params(self, **params) -> "LightGBMModel":
        """Set model parameters."""
        self.model.set_params(**params)
        if "model_params" not in self.config:
            self.config["model_params"] = {}
        self.config["model_params"].update(params)
        return self

