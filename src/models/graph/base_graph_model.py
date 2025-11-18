"""Base class for graph neural network models.

This module provides a base class for all graph neural network models,
with common functionality for graph structure handling and node feature preparation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..base import BaseModel
from src.models.utils import GraphBuilder, get_graph_cache_path


if not TORCH_AVAILABLE:
    raise ImportError("PyTorch is required for graph models. Please install torch.")


class BaseGraphModel(BaseModel, ABC):
    """Base class for graph neural network models.
    
    Provides common functionality for:
    - Graph structure loading/saving
    - Node feature preparation (Raw variables + time encoding)
    - Graph data handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize graph model.
        
        Args:
            config: Model configuration dictionary containing:
                - model_params: Parameters specific to the model
                - training: Training configuration
                - graph_type: 'radius' or 'knn'
                - graph_param: Radius in km (for 'radius') or k (for 'knn')
                - edge_weight: 'gaussian', 'distance', 'binary', or 'learnable'
        """
        super().__init__(config)
        
        # Graph-specific configuration
        self.graph_type = config.get('graph_type', 'radius')  # 'radius' or 'knn'
        self.graph_param = config.get('graph_param', 50.0)  # Radius (km) or k
        self.edge_weight = config.get('edge_weight', 'gaussian')
        self.graph_cache_dir = config.get('graph_cache_dir', None)
        
        # Graph structure (loaded in fit or load)
        self.graph = None
        self.graph_builder = None
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Feature names (set in fit)
        self.feature_names = None
        self.node_feature_size = None
    
    def _init_graph_builder(self):
        """Initialize graph builder if not already initialized."""
        if self.graph_builder is None:
            metadata_path = self.config.get('metadata_path', None)
            self.graph_builder = GraphBuilder(metadata_path=metadata_path)
    
    def _load_or_build_graph(self, use_cache: bool = True) -> Dict:
        """Load graph from cache or build new graph.
        
        Args:
            use_cache: Whether to use cached graph if available.
        
        Returns:
            Graph dictionary.
        """
        self._init_graph_builder()
        
        # Try to load from cache
        if use_cache:
            cache_path = get_graph_cache_path(
                self.graph_type,
                self.graph_param,
                self.graph_cache_dir
            )
            if cache_path.exists():
                try:
                    graph = GraphBuilder.load_graph(cache_path)
                    if self.progress_logger:
                        self.progress_logger.log(
                            f"  ✅ Loaded graph from cache: {cache_path}",
                            flush=True,
                            detailed=True
                        )
                    return graph
                except Exception as e:
                    if self.progress_logger:
                        self.progress_logger.log(
                            f"  ⚠️  Failed to load cached graph: {e}, building new graph",
                            flush=True,
                            detailed=True
                        )
        
        # Build new graph
        if self.graph_type == 'radius':
            graph = self.graph_builder.build_radius_graph(
                radius_km=self.graph_param,
                edge_weight=self.edge_weight
            )
        elif self.graph_type == 'knn':
            graph = self.graph_builder.build_knn_graph(
                k=int(self.graph_param),
                edge_weight=self.edge_weight
            )
        else:
            raise ValueError(f"Unknown graph_type: {self.graph_type}")
        
        # Save to cache
        if use_cache:
            cache_path = get_graph_cache_path(
                self.graph_type,
                self.graph_param,
                self.graph_cache_dir
            )
            try:
                GraphBuilder.save_graph(graph, cache_path)
                if self.progress_logger:
                    self.progress_logger.log(
                        f"  ✅ Saved graph to cache: {cache_path}",
                        flush=True,
                        detailed=True
                    )
            except Exception as e:
                if self.progress_logger:
                    self.progress_logger.log(
                        f"  ⚠️  Failed to save graph to cache: {e}",
                        flush=True,
                        detailed=True
                    )
        
        return graph
    
    def _prepare_node_features(
        self,
        X: pd.DataFrame,
        station_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare node features from input DataFrame.
        
        For E category (Raw-only + Multi-station), node features should be:
        - Raw variables (Air Temp, Humidity, Wind Speed, etc.)
        - Time encoding (hour_sin/cos, day_sin/cos, month_sin/cos)
        - NO feature engineering (no FE pipeline)
        
        Args:
            X: Feature DataFrame.
            station_ids: Optional array of station IDs.
        
        Returns:
            Tuple of (node_features, station_ids_array).
            node_features: (n_samples, n_features) array.
            station_ids_array: (n_samples,) array.
        """
        # Extract raw features (exclude time encoding if already present)
        # Time encoding will be added if not present
        raw_features = []
        time_features = []
        
        # Check for time encoding columns
        time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                     'month_sin', 'month_cos', 'season_sin', 'season_cos']
        
        # Extract raw meteorological variables
        raw_vars = ['Air Temp (C)', 'Relative Humidity (%)', 'Wind Speed (m/s)',
                   'Wind Direction (deg)', 'Solar Radiation (W/m²)', 'Dew Point (C)',
                   'Vapor Pressure (kPa)', 'ET0 (mm)']
        
        for col in X.columns:
            if col in raw_vars:
                raw_features.append(col)
            elif col in time_cols:
                time_features.append(col)
            elif col not in ['Date', 'Stn Id']:
                # Other features (might be derived, but we include them for now)
                raw_features.append(col)
        
        # Prepare feature array
        feature_cols = raw_features + time_features
        if len(feature_cols) == 0:
            raise ValueError("No valid features found in input DataFrame")
        
        node_features = X[feature_cols].values.astype(np.float32)
        
        # Get station IDs
        if station_ids is None:
            if 'Stn Id' in X.columns:
                station_ids_array = X['Stn Id'].values
            else:
                raise ValueError("Station IDs not provided and 'Stn Id' column not found")
        else:
            station_ids_array = np.asarray(station_ids)
        
        # Store feature names
        self.feature_names = feature_cols
        self.node_feature_size = len(feature_cols)
        
        return node_features, station_ids_array
    
    def _get_station_indices(
        self,
        station_ids: np.ndarray,
        graph_station_ids: np.ndarray
    ) -> np.ndarray:
        """Map station IDs to graph node indices.
        
        Args:
            station_ids: Station IDs from data.
            graph_station_ids: Station IDs in graph (from graph['station_ids']).
        
        Returns:
            Array of node indices (same length as station_ids).
        """
        # Create mapping from station ID to node index
        station_to_node = {sid: idx for idx, sid in enumerate(graph_station_ids)}
        
        # Map station IDs to node indices
        node_indices = np.array([station_to_node.get(sid, -1) for sid in station_ids])
        
        # Check for unmapped stations
        unmapped = (node_indices == -1)
        if unmapped.any():
            unmapped_stations = np.unique(station_ids[unmapped])
            raise ValueError(
                f"Stations not found in graph: {unmapped_stations}. "
                f"Graph contains stations: {graph_station_ids}"
            )
        
        return node_indices
    
    def save(self, path: Path) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save the model (directory).
        """
        if isinstance(path, str):
            path = Path(path)
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        if self.model is not None:
            model_path = path / "model.pth"
            torch.save(self.model.state_dict(), model_path)
        
        # Save metadata
        metadata = {
            'config': self.config,
            'feature_names': self.feature_names,
            'node_feature_size': self.node_feature_size,
            'graph_type': self.graph_type,
            'graph_param': self.graph_param,
            'edge_weight': self.edge_weight,
            'is_fitted': self.is_fitted,
        }
        
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save graph structure
        if self.graph is not None:
            graph_path = path / "graph.pkl"
            GraphBuilder.save_graph(self.graph, graph_path)
        
        # Save config JSON
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseGraphModel":
        """Load model from disk.
        
        Args:
            path: Directory path containing saved model.
        
        Returns:
            Loaded model instance.
        """
        pass
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseGraphModel":
        """Train the graph model.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            **kwargs: Additional training arguments.
        
        Returns:
            Self for method chaining.
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, station_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Make point predictions.
        
        Args:
            X: Feature DataFrame.
            station_ids: Optional array of station IDs.
        
        Returns:
            Array of predictions.
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame, station_ids: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Predict probabilities (for classification tasks).
        
        Args:
            X: Feature DataFrame.
            station_ids: Optional array of station IDs.
        
        Returns:
            Array of probabilities.
        """
        pass

