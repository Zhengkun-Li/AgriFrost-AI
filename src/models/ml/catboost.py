"""CatBoost model implementation for frost forecasting."""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from ..base import BaseModel


class CatBoostModel(BaseModel):
    """CatBoost model for frost forecasting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize CatBoost model.
        
        Args:
            config: Model configuration dictionary.
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is required. Install with: pip install catboost")
        
        super().__init__(config)
        
        model_params = config.get("model_params", {})
        self.task_type = config.get("task_type", "regression")
        
        # CatBoost parameters
        # Set verbose=False by default to reduce output
        if "verbose" not in model_params:
            model_params["verbose"] = False
        
        # Ensure consistent random_state (default: 42 for reproducibility)
        # Note: CatBoost uses 'random_seed' instead of 'random_state'
        if "random_seed" not in model_params and "random_state" not in model_params:
            model_params["random_seed"] = 42
        elif "random_state" in model_params and "random_seed" not in model_params:
            # Map random_state to random_seed for CatBoost
            model_params["random_seed"] = model_params.pop("random_state")
        
        # Ensure consistent thread_count (CatBoost equivalent of n_jobs, default: -1)
        if "thread_count" not in model_params:
            model_params["thread_count"] = -1
        
        if self.task_type == "classification":
            self.model = CatBoostClassifier(**model_params)
        else:
            self.model = CatBoostRegressor(**model_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "CatBoostModel":
        """Train the CatBoost model.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            **kwargs: Additional arguments:
                - eval_set: List of (X_val, y_val) tuples for validation
                - eval_metric: Metric for evaluation
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
                    model_name="CatBoost",
                    config={
                        "task_type": self.task_type,
                        "iterations": model_params.get("iterations", "default"),
                        "learning_rate": model_params.get("learning_rate", "default")
                    }
                )
        
        self.feature_names = list(X.columns)
        
        # Handle eval_set if provided
        eval_set = kwargs.get("eval_set", None)
        if eval_set is not None:
            # CatBoost expects eval_set as a list of tuples
            if isinstance(eval_set, list) and len(eval_set) > 0:
                # If it's already a list, check the first element
                first_elem = eval_set[0]
                if isinstance(first_elem, tuple) and len(first_elem) == 2:
                    # Already in correct format: [(X_val, y_val)]
                    pass
                else:
                    # Single tuple: (X_val, y_val) -> convert to list
                    eval_set = [first_elem] if isinstance(first_elem, tuple) else [(first_elem, None)]
            elif isinstance(eval_set, tuple) and len(eval_set) == 2:
                # Single tuple: (X_val, y_val) -> convert to list
                eval_set = [eval_set]
            else:
                # Unexpected format, try to handle gracefully
                eval_set = None
            kwargs["eval_set"] = eval_set
        
        self.model.fit(X, y, **kwargs)
        
        # Extract training history from CatBoost (if available)
        if self.training_history and hasattr(self.model, 'get_best_iteration'):
            # CatBoost stores metrics in evals_result_ if available
            # Note: CatBoost doesn't expose evals_result_ directly, but we can track iterations
            best_iteration = self.model.get_best_iteration()
            if best_iteration is not None and best_iteration > 0:
                # Record that training completed
                self.training_history.record_epoch(
                    epoch=best_iteration,
                    train_loss=None,  # CatBoost doesn't expose this easily
                    val_loss=None,
                    learning_rate=self.config.get("model_params", {}).get("learning_rate", None)
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
            Predicted values.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (classification only).
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Probability array of shape (n_samples, n_classes).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        
        proba = self.model.predict_proba(X)
        # CatBoost returns (n_samples, n_classes), extract positive class for binary
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]  # Return positive class probability (consistent with LightGBM/XGBoost)
        return proba
    
    def get_feature_importance(self, importance_type: str = "FeatureImportance") -> pd.DataFrame:
        """Get feature importance.
        
        Args:
            importance_type: Type of importance ("FeatureImportance", "PredictionValuesChange", etc.)
        
        Returns:
            DataFrame with feature names and importance scores.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importances = self.model.get_feature_importance(importance_type=importance_type)
        
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
    
    def save(self, path: Path):
        """Save the model to disk.
        
        Args:
            path: Directory path to save the model (will create model.cbm inside).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        path = Path(path)
        # If path is a file path, use parent directory. Otherwise use as directory.
        if path.suffix:
            # It's a file path, extract directory and filename
            model_dir = path.parent
            model_filename = path.name
        else:
            # It's a directory path, create model.cbm inside
            model_dir = path
            model_filename = "model.cbm"
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model (CatBoost uses .cbm extension)
        model_path = model_dir / model_filename
        self.model.save_model(str(model_path))
        
        # Save metadata (consistent with LightGBM/XGBoost structure)
        metadata_path = model_dir / f"{Path(model_filename).stem}_metadata.json"
        metadata = {
            "feature_names": self.feature_names,
            "task_type": self.task_type,
            "model_type": "catboost"
        }
        
        import json
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "CatBoostModel":
        """Load a saved model from disk.
        
        Args:
            path: Directory path containing model.cbm, or direct path to model file.
        
        Returns:
            Loaded CatBoostModel instance.
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is required. Install with: pip install catboost")
        
        import json
        
        path = Path(path)
        
        # Determine model file path (try .cbm first, then any file)
        if path.is_dir():
            # Directory path, look for model.cbm
            model_path = path / "model.cbm"
            metadata_path = path / "model_metadata.json"
            # If model.cbm doesn't exist, try to find any .cbm file
            if not model_path.exists():
                cbm_files = list(path.glob("*.cbm"))
                if cbm_files:
                    model_path = cbm_files[0]
        else:
            # File path (backward compatibility)
            model_path = path
            metadata_path = path.parent / f"{path.stem}_metadata.json"
        
        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        task_type = metadata.get("task_type", "regression")
        
        # Create model instance
        if task_type == "classification":
            model = CatBoostClassifier()
        else:
            model = CatBoostRegressor()
        
        # Load model
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model.load_model(str(model_path))
        
        # Create config
        config = {
            "model_params": {},
            "task_type": task_type
        }
        
        instance = cls(config)
        instance.model = model
        instance.is_fitted = True
        instance.feature_names = metadata.get("feature_names", [])
        
        return instance

