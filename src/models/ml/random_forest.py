"""Random Forest model implementation for frost forecasting."""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest model for frost forecasting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Random Forest model.
        
        Args:
            config: Model configuration dictionary.
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
            self.model = RandomForestClassifier(**model_params)
        else:
            self.model = RandomForestRegressor(**model_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "RandomForestModel":
        """Train the Random Forest model.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            **kwargs: Additional arguments:
                - checkpoint_dir: Optional directory for checkpoints
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
                save_best=False,  # Random Forest doesn't support incremental training
                best_metric="val_loss" if self.task_type == "regression" else "val_auc",
                best_mode="min" if self.task_type == "regression" else "max"
            )
            if self.progress_logger:
                self.progress_logger.on_training_start(
                    model_name="Random Forest",
                    config={
                        "task_type": self.task_type,
                        "n_estimators": model_params.get("n_estimators", "default"),
                        "max_depth": model_params.get("max_depth", "default")
                    }
                )
        
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        
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
            Probability array. For binary classification, returns positive class probability (1D array).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        
        proba = self.model.predict_proba(X)
        # Random Forest returns (n_samples, n_classes), extract positive class for binary
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]  # Return positive class probability (consistent with LightGBM/XGBoost)
        return proba
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance.
        
        Returns:
            DataFrame with feature names and importance scores.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importances = self.model.feature_importances_
        
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
    
    def save(self, path: Path):
        """Save the model to disk.
        
        Args:
            path: Directory path to save the model (will create model.pkl inside).
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
            # It's a directory path, create model.pkl inside
            model_dir = path
            model_filename = "model.pkl"
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using pickle
        import pickle
        model_path = model_dir / model_filename
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        
        # Save metadata (consistent with LightGBM/XGBoost structure)
        metadata_path = model_dir / f"{Path(model_filename).stem}_metadata.json"
        metadata = {
            "feature_names": self.feature_names,
            "task_type": self.task_type,
            "model_type": "random_forest"
        }
        
        import json
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "RandomForestModel":
        """Load a saved model from disk.
        
        Args:
            path: Directory path containing model.pkl, or direct path to model file.
        
        Returns:
            Loaded RandomForestModel instance.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        import pickle
        import json
        
        path = Path(path)
        
        # Determine model file path
        if path.is_dir():
            # Directory path, look for model.pkl
            model_path = path / "model.pkl"
            metadata_path = path / "model_metadata.json"
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
        
        # Load model
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
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

