"""Graph dataset utilities for unified graph tensor construction.

This module provides utilities for converting DataFrame inputs into
graph temporal tensors (T, N, F) used by graph neural network models.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging

_logger = logging.getLogger(__name__)


class GraphDatasetBuilder:
    """Utility class for building graph temporal tensors from DataFrames.
    
    This class provides a unified interface for converting DataFrame inputs
    into graph temporal tensors (T, N, F) used by graph neural network models.
    It handles time grouping, missing data, and feature scaling.
    """
    
    @staticmethod
    def build_graph_tensor(
        X: pd.DataFrame,
        node_features: np.ndarray,
        node_indices: np.ndarray,
        station_ids_array: np.ndarray,
        num_nodes: int,
        node_feature_size: int,
        valid_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """Build graph temporal tensor from node features.
        
        This method groups node features by time, constructs a full-graph
        tensor (T, N, F), and handles missing data via forward/backward fill.
        
        Args:
            X: Original feature DataFrame (for time extraction).
            node_features: Node feature array (n_samples, n_features).
            node_indices: Node indices array (n_samples,).
            station_ids_array: Station IDs array (n_samples,).
            num_nodes: Number of nodes in graph.
            node_feature_size: Number of features per node.
            valid_mask: Optional boolean mask for valid samples.
        
        Returns:
            Tuple of (X_graph, valid_mask_graph, time_steps):
            - X_graph: (T_all, N, F) graph temporal tensor
            - valid_mask_graph: (T_all, N) boolean mask for valid nodes
            - time_steps: List of time step identifiers
        """
        # Apply valid mask if provided
        if valid_mask is not None:
            node_features = node_features[valid_mask]
            node_indices = node_indices[valid_mask]
            station_ids_array = station_ids_array[valid_mask]
        
        # Group by time
        time_groups = GraphDatasetBuilder._group_by_time(
            X, node_features, node_indices, valid_mask
        )
        
        # Build graph tensor: (T, N, F)
        all_time_steps = sorted(time_groups.keys())
        T_all = len(all_time_steps)
        N = num_nodes
        F = node_feature_size
        
        X_graph = np.full((T_all, N, F), np.nan, dtype=np.float32)
        valid_mask_graph = np.zeros((T_all, N), dtype=bool)
        
        for t_idx, time_step in enumerate(all_time_steps):
            for node_idx in range(N):
                if node_idx in time_groups[time_step]:
                    feat = time_groups[time_step][node_idx]
                    X_graph[t_idx, node_idx, :] = feat
                    valid_mask_graph[t_idx, node_idx] = True
        
        # Handle NaN: forward fill then backward fill
        X_graph = GraphDatasetBuilder._fill_missing_values(X_graph)
        
        return X_graph, valid_mask_graph, all_time_steps
    
    @staticmethod
    def _group_by_time(
        X: pd.DataFrame,
        node_features: np.ndarray,
        node_indices: np.ndarray,
        valid_mask: Optional[np.ndarray] = None
    ) -> Dict:
        """Group node features by time.
        
        Args:
            X: Original feature DataFrame.
            node_features: Node feature array.
            node_indices: Node indices array.
            valid_mask: Optional boolean mask for valid samples.
        
        Returns:
            Dictionary mapping time identifiers to node features.
        """
        # Apply valid mask to X if provided
        X_filtered = X.copy()
        if valid_mask is not None:
            X_filtered = X_filtered[valid_mask]
        
        # Extract time identifiers
        has_date = 'Date' in X_filtered.columns
        has_hour = 'Hour (PST)' in X_filtered.columns or 'Hour' in X_filtered.columns
        
        if has_date:
            time_identifiers = GraphDatasetBuilder._extract_time_identifiers(
                X_filtered, has_hour
            )
        else:
            # Fallback: assume time-ordered
            time_identifiers = GraphDatasetBuilder._generate_sequential_time(
                node_indices
            )
        
        # Group by time
        time_groups = {}
        for feat, node_idx, time_id in zip(node_features, node_indices, time_identifiers):
            if time_id not in time_groups:
                time_groups[time_id] = {}
            time_groups[time_id][node_idx] = feat
        
        return time_groups
    
    @staticmethod
    def _extract_time_identifiers(
        X: pd.DataFrame,
        has_hour: bool
    ) -> np.ndarray:
        """Extract time identifiers from DataFrame.
        
        Args:
            X: Feature DataFrame.
            has_hour: Whether hour column exists.
        
        Returns:
            Array of time identifiers.
        """
        if has_hour:
            hour_col = 'Hour (PST)' if 'Hour (PST)' in X.columns else 'Hour'
            try:
                hour_numeric = pd.to_numeric(X[hour_col], errors='coerce')
                hour_str = hour_numeric.astype('Int64').astype(str).str.zfill(4)
            except (ValueError, TypeError):
                hour_str = X[hour_col].astype(str).str.zfill(4)
            
            ts_str = X['Date'].astype(str) + " " + hour_str
            timestamps = pd.to_datetime(
                ts_str, format="%m/%d/%Y %H%M", errors='coerce', infer_datetime_format=True
            )
            
            if timestamps.isna().all():
                time_identifiers = ts_str.values
            elif timestamps.isna().any():
                time_identifiers = []
                for ts, ts_str_val in zip(timestamps, ts_str):
                    time_identifiers.append(ts if not pd.isna(ts) else ts_str_val)
                time_identifiers = np.array(time_identifiers)
            else:
                time_identifiers = timestamps.values
        else:
            timestamps = pd.to_datetime(
                X['Date'], errors='coerce', infer_datetime_format=True
            )
            
            if timestamps.isna().all():
                time_identifiers = X['Date'].values
            elif timestamps.isna().any():
                time_identifiers = []
                for ts, date_val in zip(timestamps, X['Date'].values):
                    time_identifiers.append(ts if not pd.isna(ts) else date_val)
                time_identifiers = np.array(time_identifiers)
            else:
                time_identifiers = timestamps.values
        
        return time_identifiers
    
    @staticmethod
    def _generate_sequential_time(node_indices: np.ndarray) -> np.ndarray:
        """Generate sequential time identifiers when date column is missing.
        
        Args:
            node_indices: Node indices array.
        
        Returns:
            Array of sequential time identifiers.
        """
        time_groups = {}
        current_time = 0
        prev_node_set = set()
        time_identifiers = []
        
        for node_idx in node_indices:
            if node_idx in prev_node_set and len(prev_node_set) > 1:
                current_time += 1
                prev_node_set = set()
            
            time_identifiers.append(current_time)
            prev_node_set.add(node_idx)
        
        return np.array(time_identifiers)
    
    @staticmethod
    def _fill_missing_values(X_graph: np.ndarray) -> np.ndarray:
        """Fill missing values in graph tensor via forward/backward fill.
        
        Args:
            X_graph: Graph tensor (T, N, F).
        
        Returns:
            Graph tensor with missing values filled.
        """
        T, N, F = X_graph.shape
        
        for node_idx in range(N):
            for feat_idx in range(F):
                col = X_graph[:, node_idx, feat_idx]
                if ~np.isnan(col).all():
                    col_series = pd.Series(col)
                    col = col_series.ffill().bfill().values
                    X_graph[:, node_idx, feat_idx] = col
        
        return X_graph
    
    @staticmethod
    def flatten_predictions_to_dataframe(
        predictions: np.ndarray,
        X: pd.DataFrame,
        station_ids: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Convert flattened predictions to DataFrame with time and station_id.
        
        Args:
            predictions: Flattened predictions array (n_valid_samples,).
            X: Original feature DataFrame.
            station_ids: Optional array of station IDs.
            valid_mask: Optional boolean mask for valid samples.
        
        Returns:
            DataFrame with columns: time, station_id, y_hat
        """
        # Get time identifiers
        has_date = 'Date' in X.columns
        has_hour = 'Hour (PST)' in X.columns or 'Hour' in X.columns
        
        X_filtered = X.copy()
        if valid_mask is not None:
            X_filtered = X_filtered[valid_mask]
        
        if has_date:
            time_identifiers = GraphDatasetBuilder._extract_time_identifiers(
                X_filtered, has_hour
            )
        else:
            time_identifiers = np.arange(len(X_filtered))
        
        # Get station IDs
        if station_ids is None:
            if 'Stn Id' in X_filtered.columns:
                station_ids_array = X_filtered['Stn Id'].values
            else:
                raise ValueError("Station IDs not found in DataFrame and not provided")
        else:
            station_ids_array = np.asarray(station_ids)
            if valid_mask is not None:
                station_ids_array = station_ids_array[valid_mask]
        
        # Create DataFrame
        result_df = pd.DataFrame({
            'time': time_identifiers,
            'station_id': station_ids_array,
            'y_hat': predictions
        })
        
        return result_df

