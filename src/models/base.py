"""Base model interface for all forecasting models."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json

_logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all forecasting models.
    
    All models must inherit from this class and implement the abstract methods.
    This ensures a consistent interface across all model types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model with configuration.
        
        Args:
            config: Model configuration dictionary containing:
                - model_params: Parameters specific to the model
                - training: Training configuration
                - evaluation: Evaluation configuration
        """
        self.config = config
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        
        # Optional training utilities (can be set up via setup_training_tools)
        self.training_history = None
        self.checkpoint_manager = None
        self.curve_plotter = None
        self.progress_logger = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseModel":
        """Train the model.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            **kwargs: Additional training arguments (validation sets, callbacks, etc.).
        
        Returns:
            Self for method chaining.
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make point predictions.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Array of predictions.
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (for classification tasks).
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Array of probabilities (shape: [n_samples] for binary, [n_samples, n_classes] for multi-class).
        """
        pass
    
    def save(self, path: Path) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save the model (directory or file).
        """
        if isinstance(path, str):
            path = Path(path)
        
        # If path is a directory, create model.pkl inside it
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            model_path = path / "model.pkl"
            config_path = path / "config.json"
        else:
            model_path = path
            config_path = path.parent / "config.json"
            model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        
        # Save config (include feature_names if available)
        config_to_save = self.config.copy() if self.config else {}
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            config_to_save['feature_names'] = self.feature_names
        
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_to_save, f, indent=2, default=str)
        except (IOError, OSError, TypeError) as e:
            _logger.warning(f"Failed to save config to {config_path}: {e}")
        
        _logger.info(f"Model saved to {model_path}")
        _logger.info(f"Config saved to {config_path}")
    
    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load model from disk.
        
        Args:
            path: Path to model directory or file.
        
        Returns:
            Loaded model instance.
        
        Raises:
            FileNotFoundError: If model file does not exist.
            ValueError: If path is None.
            (IOError, OSError, pickle.UnpicklingError): If file operations fail.
        
        Note:
            This is a base implementation. Subclasses may override for custom loading.
        """
        if path is None:
            raise ValueError("load path cannot be None")
        
        if isinstance(path, str):
            path = Path(path)
        
        # Determine paths
        if path.is_dir():
            model_path = path / "model.pkl"
            config_path = path / "config.json"
        else:
            model_path = path
            config_path = path.parent / "config.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load config
        config = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (IOError, OSError, json.JSONDecodeError) as e:
                _logger.warning(f"Failed to load config from {config_path}: {e}. Using empty config.")
        
        # Extract feature_names from config if present
        feature_names = config.pop('feature_names', None)
        
        # Create instance
        instance = cls(config)
        
        # Load model
        try:
            with open(model_path, "rb") as f:
                instance.model = pickle.load(f)
        except (IOError, OSError, pickle.UnpicklingError) as e:
            raise OSError(f"Failed to load model from {model_path}: {e}") from e
        
        # Restore feature_names if available
        if feature_names is not None:
            instance.feature_names = feature_names
        
        instance.is_fitted = True
        _logger.debug(f"Loaded model from {model_path}")
        return instance
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Return feature importance if available.
        
        Returns:
            DataFrame with columns ['feature', 'importance'], or None if not available.
        """
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for reporting and analysis.
        
        Returns:
            Dictionary containing:
                - model_type: Model type name (e.g., "lightgbm", "lstm")
                - task_type: Task type ("classification" or "regression")
                - n_params: Number of trainable parameters (if available)
                - library: Underlying library name (e.g., "lightgbm", "torch")
                - is_fitted: Whether model has been fitted
                - feature_names: List of feature names (if available)
        """
        # Get model type from class name
        model_type = self.__class__.__name__.lower().replace("model", "").replace("forecast", "")
        
        # Determine library
        library = "unknown"
        if hasattr(self, "model") and self.model is not None:
            module_name = self.model.__class__.__module__
            if "lightgbm" in module_name:
                library = "lightgbm"
            elif "xgboost" in module_name:
                library = "xgboost"
            elif "catboost" in module_name:
                library = "catboost"
            elif "sklearn" in module_name:
                library = "scikit-learn"
            elif "torch" in module_name:
                library = "torch"
            elif "prophet" in module_name.lower():
                library = "prophet"
        
        # Try to get number of parameters
        n_params = None
        if hasattr(self, "model") and self.model is not None:
            try:
                # For sklearn models
                if hasattr(self.model, "coef_"):
                    n_params = np.prod(self.model.coef_.shape) if hasattr(self.model.coef_, "shape") else None
                # For tree models
                elif hasattr(self.model, "n_features_in_"):
                    n_params = self.model.n_features_in_
                # For PyTorch models
                elif hasattr(self.model, "parameters"):
                    n_params = sum(p.numel() for p in self.model.parameters())
            except Exception:
                pass
        
        # Get task type
        task_type = getattr(self, "task_type", None)
        if task_type is None:
            # Try to infer from model type
            if hasattr(self, "model") and self.model is not None:
                if "Classifier" in self.model.__class__.__name__:
                    task_type = "classification"
                elif "Regressor" in self.model.__class__.__name__:
                    task_type = "regression"
        
        info = {
            "model_type": model_type,
            "task_type": task_type,
            "n_params": n_params,
            "library": library,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names if hasattr(self, "feature_names") else None
        }
        
        return info
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters.
        """
        return self.config.get("model_params", {})
    
    def set_params(self, **params) -> "BaseModel":
        """Set model parameters.
        
        Args:
            **params: Parameters to set.
        
        Returns:
            Self for method chaining.
        """
        if "model_params" not in self.config:
            self.config["model_params"] = {}
        self.config["model_params"].update(params)
        return self
    
    def setup_training_tools(
        self,
        checkpoint_dir: Optional[Path] = None,
        log_file: Optional[Path] = None,
        detailed_log_file: Optional[Path] = None,
        checkpoint_frequency: int = 10,
        save_best: bool = True,
        best_metric: str = "val_loss",
        best_mode: str = "min"
    ) -> "BaseModel":
        """Setup optional training utilities.
        
        This method initializes training history, checkpoint manager, and progress logger.
        These tools use event-based APIs with Runner path management.
        
        Args:
            checkpoint_dir: Directory to save checkpoints (None = disabled).
            log_file: Path to brief log file (training.log, suitable for GitHub).
            detailed_log_file: Path to detailed log file (training_detailed.log, excluded from GitHub).
            checkpoint_frequency: Save checkpoint every N epochs (0 = disabled).
            save_best: Whether to save the best model based on metric.
            best_metric: Metric name to use for determining best model.
            best_mode: "min" or "max" - whether lower or higher is better.
        
        Returns:
            Self for method chaining.
        """
        # Use MetricSchema for default metrics if available
        try:
            from src.training.metrics import MetricSchema
            default_metrics = MetricSchema.default_history_metrics()
        except ImportError:
            default_metrics = ['train_loss', 'val_loss', 'learning_rate', 'epoch_time']
        
        from src.models.utils import TrainingHistory, CheckpointManager, ProgressLogger
        
        # Training history (with schema injection)
        self.training_history = TrainingHistory(metrics=default_metrics, use_metric_schema=True)
        
        # Checkpoint manager (path management by Runner)
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_frequency=checkpoint_frequency,
                save_best=save_best,
                best_metric=best_metric,
                best_mode=best_mode,
                keep_top_k=3
            )
            self.checkpoint_manager.bind_dir(Path(checkpoint_dir))
        else:
            self.checkpoint_manager = None
        
        # Progress logger (event-based, path management by Runner)
        self.progress_logger = ProgressLogger(use_metric_schema=True)
        if log_file or detailed_log_file:
            # Convert to Path if needed (log_file may be str or Path)
            if log_file:
                log_path = Path(log_file)
                if not detailed_log_file:
                    detailed_log_file = log_path.parent / f"{log_path.stem}_detailed{log_path.suffix}"
            else:
                log_path = None
            if detailed_log_file:
                detailed_log_path = Path(detailed_log_file)
            else:
                detailed_log_path = None
            self.progress_logger.bind_files(
                brief_path=log_path,
                detailed_path=detailed_log_path
            )
        
        # Curve plotter is now stateless (no instance needed)
        self.curve_plotter = None
        
        return self
    
    def save_training_artifacts(self, output_dir: Path) -> None:
        """Save training artifacts (history, curves) if available.
        
        Uses stateless plotting functions from `src.visualization.plots`.
        
        Args:
            output_dir: Directory to save artifacts.
        
        Raises:
            ValueError: If output_dir is None.
            OSError: If directory creation fails.
        """
        if output_dir is None:
            raise ValueError("output_dir cannot be None")
        
        output_dir = Path(output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create output directory {output_dir}: {e}") from e
        
        # Save training history
        if self.training_history and len(self.training_history) > 0:
            history_path = output_dir / "training_history.json"
            self.training_history.save(history_path)
            
            # Plot training curves (stateless function from visualization.plots)
            from src.visualization.plots import plot_training_curves, plot_multitask_curves
            
            curve_path = output_dir / "training_curves.png"
            
            history_dict = self.training_history.get_history()
            
            # Check if it's a multi-task model
            if isinstance(history_dict, dict) and any('_temp' in k or '_frost' in k or 'train_loss_total' in k for k in history_dict.keys()):
                plot_multitask_curves(self.training_history, curve_path)
            else:
                plot_training_curves(self.training_history, curve_path)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        class_name = self.__class__.__name__
        fitted = "fitted" if self.is_fitted else "not fitted"
        return f"{class_name}({fitted})"

