"""Base model interface for all forecasting models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json


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
        
        # Save config
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        
        print(f"Model saved to {model_path}")
        print(f"Config saved to {config_path}")
    
    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load model from disk.
        
        Args:
            path: Path to model directory or file.
        
        Returns:
            Loaded model instance.
        
        Note:
            This is a base implementation. Subclasses may override for custom loading.
        """
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
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}
        
        # Create instance
        instance = cls(config)
        
        # Load model
        with open(model_path, "rb") as f:
            instance.model = pickle.load(f)
        
        instance.is_fitted = True
        return instance
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Return feature importance if available.
        
        Returns:
            DataFrame with columns ['feature', 'importance'], or None if not available.
        """
        return None
    
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
        
        This method initializes training history, checkpoint manager, curve plotter,
        and progress logger. These tools can be used by any model type for consistent
        training monitoring and checkpointing.
        
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
        from src.models.utils import (
            TrainingHistory, CheckpointManager, TrainingCurvePlotter, ProgressLogger
        )
        
        # Training history (always available)
        self.training_history = TrainingHistory()
        
        # Checkpoint manager (if checkpoint_dir provided)
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=Path(checkpoint_dir),
                checkpoint_frequency=checkpoint_frequency,
                save_best=save_best,
                best_metric=best_metric,
                best_mode=best_mode
            )
        
        # Curve plotter (always available)
        self.curve_plotter = TrainingCurvePlotter()
        
        # Progress logger (always available)
        # If detailed_log_file not provided but log_file is, derive it
        if log_file and not detailed_log_file:
            log_path = Path(log_file)
            detailed_log_file = log_path.parent / f"{log_path.stem}_detailed{log_path.suffix}"
        
        self.progress_logger = ProgressLogger(
            log_file=log_file,
            detailed_log_file=detailed_log_file
        )
        
        return self
    
    def save_training_artifacts(self, output_dir: Path) -> None:
        """Save training artifacts (history, curves) if available.
        
        Args:
            output_dir: Directory to save artifacts.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training history
        if self.training_history and len(self.training_history) > 0:
            history_path = output_dir / "training_history.json"
            self.training_history.save(history_path)
            
            # Plot training curves
            if self.curve_plotter:
                curve_path = output_dir / "training_curves.png"
                history_dict = self.training_history.get_history()
                
                # Check if it's a multi-task model
                if any('_temp' in k or '_frost' in k for k in history_dict.keys()):
                    self.curve_plotter.plot_multitask(history_dict, curve_path)
                else:
                    self.curve_plotter.plot(history_dict, curve_path)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        class_name = self.__class__.__name__
        fitted = "fitted" if self.is_fitted else "not fitted"
        return f"{class_name}({fitted})"

