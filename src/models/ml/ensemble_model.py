"""Ensemble model implementation for frost forecasting.

Combines multiple models (LightGBM, XGBoost, CatBoost) for improved performance.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from ..base import BaseModel
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel
from .catboost_model import CatBoostModel


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple gradient boosting models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ensemble model.
        
        Args:
            config: Model configuration dictionary with:
                - base_models: List of model types to include (default: ["lightgbm", "xgboost", "catboost"])
                - ensemble_method: "mean" or "weighted" (default: "mean")
                - weights: Optional weights for weighted ensemble (default: None, equal weights)
                - model_params: Parameters for each base model
        """
        super().__init__(config)
        
        self.base_models_config = config.get("base_models", ["lightgbm", "xgboost", "catboost"])
        self.ensemble_method = config.get("ensemble_method", "mean")
        self.weights = config.get("weights", None)
        self.task_type = config.get("task_type", "regression")
        
        # Initialize base models
        self.models: List[BaseModel] = []
        model_params = config.get("model_params", {})
        
        for model_type in self.base_models_config:
            model_config = {
                "model_params": model_params.get(model_type, {}),
                "task_type": self.task_type
            }
            
            if model_type == "lightgbm":
                self.models.append(LightGBMModel(model_config))
            elif model_type == "xgboost":
                self.models.append(XGBoostModel(model_config))
            elif model_type == "catboost":
                self.models.append(CatBoostModel(model_config))
            else:
                raise ValueError(f"Unsupported base model type: {model_type}")
        
        # Set equal weights if not provided
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        if len(self.weights) != len(self.models):
            raise ValueError(f"Number of weights ({len(self.weights)}) must match number of models ({len(self.models)})")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "EnsembleModel":
        """Train all base models.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            **kwargs: Additional arguments passed to base models:
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
                save_best=False,  # Ensemble doesn't support incremental training
                best_metric="val_loss" if self.task_type == "regression" else "val_auc",
                best_mode="min" if self.task_type == "regression" else "max"
            )
            if self.progress_logger:
                self.progress_logger.log_training_start(
                    model_name="Ensemble",
                    config={
                        "task_type": self.task_type,
                        "base_models": self.base_models_config,
                        "ensemble_method": self.ensemble_method
                    }
                )
        
        self.feature_names = list(X.columns)
        
        # Train each base model
        for i, model in enumerate(self.models):
            if self.progress_logger:
                self.progress_logger.log(f"Training {self.base_models_config[i]} model ({i+1}/{len(self.models)})...", flush=True)
            model.fit(X, y, **kwargs)
        
        if self.progress_logger:
            self.progress_logger.log("  ✅ Training completed", flush=True)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Ensemble predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from all base models
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            # Ensure 1D array
            pred = np.asarray(pred).flatten()
            predictions.append(pred)
        
        predictions = np.array(predictions)  # Shape: (n_models, n_samples)
        
        # Combine predictions
        if self.task_type == "classification":
            # For classification, use majority voting or average probabilities
            if self.ensemble_method == "mean":
                # Average probabilities, then threshold at 0.5
                avg_proba = np.mean(predictions, axis=0)
                return (avg_proba >= 0.5).astype(int)
            elif self.ensemble_method == "weighted":
                avg_proba = np.average(predictions, axis=0, weights=self.weights)
                return (avg_proba >= 0.5).astype(int)
            else:
                raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
        else:
            # For regression, use mean or weighted average
            if self.ensemble_method == "mean":
                return np.mean(predictions, axis=0)
            elif self.ensemble_method == "weighted":
                return np.average(predictions, axis=0, weights=self.weights)
            else:
                raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
    
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
        
        # Get probability predictions from all base models
        probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            # Ensure 1D array (positive class probability)
            if proba.ndim == 2:
                if proba.shape[1] == 2:
                    proba = proba[:, 1]  # Take positive class probability
                else:
                    proba = proba.flatten()
            elif proba.ndim > 2:
                proba = proba.flatten()
            # Ensure it's a 1D numpy array
            proba = np.asarray(proba).flatten()
            probas.append(proba)
        
        probas = np.array(probas)  # Shape: (n_models, n_samples)
        
        # Combine probabilities
        if self.ensemble_method == "mean":
            ensemble_proba = np.mean(probas, axis=0)
        elif self.ensemble_method == "weighted":
            ensemble_proba = np.average(probas, axis=0, weights=self.weights)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
        
        # Return 1D array (consistent with LightGBM/XGBoost/CatBoost)
        return ensemble_proba
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get average feature importance across all base models.
        
        Returns:
            DataFrame with feature names and average importance scores.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Get importance from all models
        all_importances = []
        for model in self.models:
            try:
                importance_df = model.get_feature_importance()
                if importance_df is not None:
                    all_importances.append(importance_df)
            except Exception:
                continue
        
        if not all_importances:
            raise ValueError("Could not get feature importance from any base model")
        
        # Merge and average
        merged = all_importances[0].copy()
        for imp_df in all_importances[1:]:
            merged = merged.merge(imp_df, on="feature", suffixes=("", "_other"))
            merged["importance"] = (merged["importance"] + merged["importance_other"]) / 2
            merged = merged.drop(columns=["importance_other"])
        
        return merged.sort_values("importance", ascending=False)
    
    def save(self, path: Path):
        """Save all base models to disk.
        
        Args:
            path: Base directory path to save models.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save each base model
        for i, model in enumerate(self.models):
            model_path = path / f"{self.base_models_config[i]}_model"
            model.save(model_path)
        
        # Save ensemble metadata
        metadata = {
            "base_models": self.base_models_config,
            "ensemble_method": self.ensemble_method,
            "weights": self.weights,
            "task_type": self.task_type,
            "feature_names": self.feature_names,
            "model_type": "ensemble"
        }
        
        import json
        metadata_path = path / "ensemble_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        """Load ensemble model from disk.
        
        Args:
            path: Base directory path containing saved models.
        
        Returns:
            Loaded EnsembleModel instance.
        """
        path = Path(path)
        
        # Load metadata
        metadata_path = path / "ensemble_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        import json
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Load base models
        models = []
        for model_type in metadata["base_models"]:
            model_path = path / f"{model_type}_model"
            
            if model_type == "lightgbm":
                models.append(LightGBMModel.load(model_path))
            elif model_type == "xgboost":
                models.append(XGBoostModel.load(model_path))
            elif model_type == "catboost":
                models.append(CatBoostModel.load(model_path))
            else:
                raise ValueError(f"Unsupported base model type: {model_type}")
        
        # Create config
        config = {
            "base_models": metadata["base_models"],
            "ensemble_method": metadata["ensemble_method"],
            "weights": metadata["weights"],
            "task_type": metadata["task_type"]
        }
        
        instance = cls(config)
        instance.models = models
        instance.is_fitted = True
        instance.feature_names = metadata.get("feature_names", [])
        
        return instance

