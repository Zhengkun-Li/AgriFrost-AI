"""Ensemble model implementation for frost forecasting.

Combines multiple models (LightGBM, XGBoost, CatBoost) for improved performance.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import logging

from ..base import BaseModel

_logger = logging.getLogger(__name__)
from .lightgbm import LightGBMModel
from .xgboost import XGBoostModel
from .catboost import CatBoostModel


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
                self.progress_logger.on_training_start(
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
        
        # Combine predictions - strict separation by task_type
        if self.task_type == "classification":
            # For classification, use probabilities (from predict_proba) not class labels
            # This should only be called after getting probabilities, but for backward compatibility
            # we handle both cases: if predictions are in [0, 1] range, treat as probabilities
            # Otherwise, use hard voting
            
            # Check if predictions look like probabilities (all in [0, 1])
            if np.all((predictions >= 0) & (predictions <= 1)):
                # Treat as probabilities, then threshold at 0.5
                if self.ensemble_method == "mean":
                    avg_proba = np.mean(predictions, axis=0)
                    return (avg_proba >= 0.5).astype(int)
                elif self.ensemble_method == "weighted":
                    avg_proba = np.average(predictions, axis=0, weights=self.weights)
                    return (avg_proba >= 0.5).astype(int)
            else:
                # Hard voting: majority class
                if self.ensemble_method == "mean":
                    return (np.mean(predictions, axis=0) >= 0.5).astype(int)
                elif self.ensemble_method == "weighted":
                    return (np.average(predictions, axis=0, weights=self.weights) >= 0.5).astype(int)
            
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
        else:
            # For regression, use mean or weighted average (strictly regression path)
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
        # Ensure consistent shape: (n_samples,) for binary classification
        probas = []
        for i, model in enumerate(self.models):
            try:
                proba = model.predict_proba(X)
                
                # Handle different output shapes consistently
                proba = np.asarray(proba)
                
                # For binary classification, ensure we get positive class probability
                if proba.ndim == 2:
                    if proba.shape[1] == 2:
                        # Binary classification: take positive class (column 1)
                        proba = proba[:, 1]
                    elif proba.shape[1] > 2:
                        # Multi-class: keep full probability matrix
                        # For now, we'll flatten to 1D, but this should be documented
                        # TODO: Consider multi-class support properly
                        _logger.warning(f"Multi-class probabilities detected for model {i}, flattening")
                        proba = proba.flatten()
                    else:
                        # Single class (edge case)
                        proba = proba.flatten()
                elif proba.ndim == 1:
                    # Already 1D (positive class probability for binary)
                    pass
                else:
                    # Higher dimensions: flatten
                    proba = proba.flatten()
                
                # Ensure it's a 1D numpy array with consistent length
                proba = np.asarray(proba).flatten()
                
                # Verify length matches sample count
                if len(proba) != len(X):
                    raise ValueError(
                        f"Probability shape mismatch for model {i}: "
                        f"expected {len(X)}, got {len(proba)}"
                    )
                
                probas.append(proba)
            except ValueError as e:
                # Re-raise with context
                raise ValueError(f"Error getting probabilities from base model {i} ({self.base_models_config[i]}): {e}") from e
        
        # Stack into array: shape (n_models, n_samples)
        probas = np.array(probas)
        
        # Verify all models have same output length
        if not all(len(p) == len(X) for p in probas):
            raise ValueError(
                f"Inconsistent probability shapes: expected all length {len(X)}, "
                f"got {[len(p) for p in probas]}"
            )
        
        # Combine probabilities with strict shape checking
        if self.ensemble_method == "mean":
            ensemble_proba = np.mean(probas, axis=0)
        elif self.ensemble_method == "weighted":
            # Ensure weights sum to 1 for proper averaging
            weights = np.array(self.weights)
            if not np.isclose(weights.sum(), 1.0):
                weights = weights / weights.sum()
            ensemble_proba = np.average(probas, axis=0, weights=weights)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
        
        # Return 1D array (positive class probability for binary classification)
        # Shape: (n_samples,)
        return ensemble_proba
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get average feature importance across all base models.
        
        Returns:
            DataFrame with feature names and average importance scores.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Special case: single model ensemble
        if len(self.models) == 1:
            importance_df = self.models[0].get_feature_importance()
            if importance_df is not None:
                return importance_df
            else:
                raise ValueError("Base model does not support feature importance")
        
        # Get importance from all models
        all_importances = []
        for i, model in enumerate(self.models):
            try:
                importance_df = model.get_feature_importance()
                if importance_df is not None and not importance_df.empty:
                    all_importances.append((i, importance_df))
            except Exception as e:
                _logger.warning(f"Could not get feature importance from model {i} ({self.base_models_config[i]}): {e}")
                continue
        
        if not all_importances:
            raise ValueError("Could not get feature importance from any base model")
        
        # Merge and average (weighted average if weights provided)
        merged = all_importances[0][1].copy()
        
        for i, imp_df in all_importances[1:]:
            merged = merged.merge(imp_df, on="feature", suffixes=("", "_other"), how="outer")
            # Fill NaN with 0 for features not present in all models
            merged["importance"] = merged["importance"].fillna(0)
            merged["importance_other"] = merged["importance_other"].fillna(0)
            
            # Weighted average if using weighted ensemble
            if self.ensemble_method == "weighted" and len(self.weights) == len(all_importances):
                # Use weights for averaging
                weight_prev = self.weights[all_importances[0][0]] if all_importances[0][0] < len(self.weights) else 1.0
                weight_curr = self.weights[i] if i < len(self.weights) else 1.0
                total_weight = weight_prev + weight_curr
                merged["importance"] = (merged["importance"] * weight_prev + merged["importance_other"] * weight_curr) / total_weight
            else:
                # Simple average
                merged["importance"] = (merged["importance"] + merged["importance_other"]) / 2
            
            merged = merged.drop(columns=["importance_other"])
        
        # If more than 2 models, continue averaging
        if len(all_importances) > 2:
            for i, imp_df in all_importances[2:]:
                merged = merged.merge(imp_df, on="feature", suffixes=("", "_other"), how="outer")
                merged["importance"] = merged["importance"].fillna(0)
                merged["importance_other"] = merged["importance_other"].fillna(0)
                
                if self.ensemble_method == "weighted" and len(self.weights) == len(all_importances):
                    weight_curr = self.weights[i] if i < len(self.weights) else 1.0
                    total_weight = merged["importance"].sum() * (len(all_importances) - 1) + merged["importance_other"].sum() * weight_curr
                    merged["importance"] = (merged["importance"] * (len(all_importances) - 1) + merged["importance_other"] * weight_curr) / (len(all_importances) - 1 + weight_curr)
                else:
                    merged["importance"] = (merged["importance"] * (len(all_importances) - 1) + merged["importance_other"]) / len(all_importances)
                
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
        
        # Save ensemble metadata with detailed base model information
        metadata = {
            "ensemble_type": "weighted_soft" if self.ensemble_method == "weighted" else "mean",
            "base_models": [
                {
                    "name": model_type,
                    "path": str(path / f"{model_type}_model"),
                    "weight": self.weights[i] if i < len(self.weights) else 1.0 / len(self.base_models_config)
                }
                for i, model_type in enumerate(self.base_models_config)
            ],
            "ensemble_method": self.ensemble_method,
            "weights": self.weights,
            "task_type": self.task_type,
            "feature_names": self.feature_names,
            "model_type": "ensemble",
            "n_base_models": len(self.base_models_config)
        }
        
        import json
        metadata_path = path / "ensemble_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
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

