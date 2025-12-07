"""Prophet model implementation for frost forecasting."""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from ..base import BaseModel


class ProphetModel(BaseModel):
    """Prophet model for time series forecasting.
    
    Note: Prophet requires a specific data format with 'ds' (datetime) and 'y' (target) columns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Prophet model.
        
        Args:
            config: Model configuration dictionary.
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required. Install with: pip install prophet")
        
        super().__init__(config)
        
        model_params = config.get("model_params", {})
        self.model = Prophet(**model_params)
        self.date_column = config.get("date_column", "Date")
        self.target_column = config.get("target_column", "Air Temp (C)")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ProphetModel":
        """Train the Prophet model.
        
        Args:
            X: Feature DataFrame (must include date column).
            y: Target Series.
            **kwargs: Additional arguments (ignored for Prophet).
        
        Returns:
            Self for method chaining.
        """
        # Prophet requires 'ds' and 'y' columns
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in X")
        
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(X[self.date_column]),
            "y": y.values
        })
        
        # Add additional regressors if specified
        regressor_cols = self.config.get("regressor_columns", [])
        for col in regressor_cols:
            if col in X.columns:
                self.model.add_regressor(col)
                prophet_df[col] = X[col].values
        
        self.model.fit(prophet_df)
        self.is_fitted = True
        self.feature_names = list(X.columns)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make point predictions.
        
        Args:
            X: Feature DataFrame (must include date column).
        
        Returns:
            Predictions array.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.date_column not in X.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in X")
        
        # Prepare future dataframe
        future_df = pd.DataFrame({
            "ds": pd.to_datetime(X[self.date_column])
        })
        
        # Add regressors
        regressor_cols = self.config.get("regressor_columns", [])
        for col in regressor_cols:
            if col in X.columns:
                future_df[col] = X[col].values
        
        forecast = self.model.predict(future_df)
        return forecast["yhat"].values
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Prophet doesn't support probability predictions.
        
        Returns:
            None (Prophet is regression-only).
        """
        return None
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Prophet doesn't provide feature importance in the same way.
        
        Returns:
            None (Prophet uses different importance metrics).
        """
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
        
        metadata = {
            "model_type": "prophet",
            "date_column": self.date_column,
            "target_column": self.target_column,
            "feature_names": self.feature_names,
            "config": self.config
        }
        
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, path: Path) -> "ProphetModel":
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
        instance.date_column = metadata.get("date_column", "Date")
        instance.target_column = metadata.get("target_column", "Air Temp (C)")
        instance.feature_names = metadata.get("feature_names", [])
        instance.is_fitted = True
        
        return instance

