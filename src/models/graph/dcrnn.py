"""DCRNN (Diffusion Convolutional Recurrent Neural Network) model for spatial-temporal forecasting.

DCRNN combines diffusion convolution for spatial modeling with RNN for temporal modeling.
It's particularly suitable for temperature diffusion patterns in frost forecasting.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import os
import pickle
import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_graph import BaseGraphModel
from .dataset import GraphDatasetBuilder
from src.models.utils.graph_builder import GraphBuilder
# FocalLoss is imported but not currently used - kept for potential future use
# from src.utils.losses import FocalLoss
from src.utils.calibration import ProbabilityCalibrator

if not TORCH_AVAILABLE:
    raise ImportError("PyTorch is required for DCRNN models. Please install torch.")


class DiffusionConvolution(nn.Module):
    """Diffusion convolution layer for spatial modeling.
    
    Models information diffusion on graph using random walk with restart.
    """
    
    def __init__(self, in_features: int, out_features: int, num_diffusion_steps: int = 2):
        """Initialize diffusion convolution layer.
        
        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            num_diffusion_steps: Number of diffusion steps (default: 2).
        """
        super(DiffusionConvolution, self).__init__()
        self.num_diffusion_steps = num_diffusion_steps
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable weight matrices for each diffusion step
        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_features, out_features))
            for _ in range(num_diffusion_steps)
        ])
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
    
    def forward(self, x: torch.Tensor, adj_normalized: torch.Tensor) -> torch.Tensor:
        """Apply diffusion convolution.
        
        Args:
            x: Input node features (batch_size, num_nodes, in_features).
            adj_normalized: Pre-normalized adjacency matrix (num_nodes, num_nodes).
                           Should be normalized with self-loops already added.
        
        Returns:
            Output features (batch_size, num_nodes, out_features).
        """
        batch_size, num_nodes, in_features = x.shape
        
        # Pre-compute adj_normalized expansion once (reused for all diffusion steps)
        adj_expanded = adj_normalized.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply diffusion steps
        outputs = []
        x_current = x
        for k in range(self.num_diffusion_steps):
            # Compute (A^k) * X more efficiently
            if k > 0:
                x_current = torch.bmm(adj_expanded, x_current)
            else:
                x_current = x  # k=0: no diffusion, use original x
            
            # Apply weight matrix
            x_k = torch.matmul(x_current, self.weights[k])
            outputs.append(x_k)
        
        # Sum over diffusion steps
        output = sum(outputs)
        return output


class DCRNNCell(nn.Module):
    """DCRNN cell combining diffusion convolution with GRU.
    
    This is the core building block of DCRNN.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_nodes: int,
        num_diffusion_steps: int = 2
    ):
        """Initialize DCRNN cell.
        
        Args:
            input_size: Input feature dimension.
            hidden_size: Hidden state dimension.
            num_nodes: Number of nodes in graph.
            num_diffusion_steps: Number of diffusion steps.
        """
        super(DCRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        
        # Diffusion convolution for input and hidden state
        self.diff_conv_input = DiffusionConvolution(
            input_size + hidden_size,
            hidden_size * 2,  # For reset and update gates
            num_diffusion_steps
        )
        self.diff_conv_hidden = DiffusionConvolution(
            input_size + hidden_size,
            hidden_size,  # For candidate activation
            num_diffusion_steps
        )
    
    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        adj_normalized: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of DCRNN cell.
        
        Args:
            x: Input features (batch_size, num_nodes, input_size).
            h: Hidden state (batch_size, num_nodes, hidden_size).
            adj_normalized: Pre-normalized adjacency matrix (num_nodes, num_nodes).
        
        Returns:
            New hidden state (batch_size, num_nodes, hidden_size).
        """
        # Concatenate input and hidden state
        xh = torch.cat([x, h], dim=-1)  # (batch_size, num_nodes, input_size + hidden_size)
        
        # Compute reset and update gates using diffusion convolution
        gates = self.diff_conv_input(xh, adj_normalized)  # (batch_size, num_nodes, hidden_size * 2)
        reset_gate, update_gate = torch.split(gates, self.hidden_size, dim=-1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        
        # Compute candidate activation
        xh_reset = torch.cat([x, reset_gate * h], dim=-1)
        candidate = torch.tanh(self.diff_conv_hidden(xh_reset, adj_normalized))
        
        # Update hidden state
        h_new = (1 - update_gate) * h + update_gate * candidate
        
        return h_new


class DCRNNModel(nn.Module):
    """DCRNN model for spatial-temporal forecasting.
    
    Combines multiple DCRNN cells for temporal modeling with diffusion convolution for spatial modeling.
    """
    
    def __init__(
        self,
        num_nodes: int,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_diffusion_steps: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """Initialize DCRNN model.
        
        Args:
            num_nodes: Number of nodes in graph.
            input_size: Input feature dimension.
            hidden_size: Hidden state dimension.
            num_layers: Number of DCRNN layers.
            num_diffusion_steps: Number of diffusion steps.
            dropout: Dropout rate.
            output_size: Output dimension (default: 1 for regression/classification).
        """
        super(DCRNNModel, self).__init__()
        self.num_nodes = num_nodes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_diffusion_steps = num_diffusion_steps
        
        # Pre-computed normalized adjacency matrix (registered as buffer for device handling)
        self.register_buffer("adj_normalized", None)
        
        # DCRNN cells
        self.dcrnn_cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.dcrnn_cells.append(
                DCRNNCell(input_dim, hidden_size, num_nodes, num_diffusion_steps)
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def set_adj(self, adj_matrix: torch.Tensor):
        """Pre-compute normalized adjacency matrix once.
        
        This avoids recomputing normalization in every forward pass.
        
        Args:
            adj_matrix: Raw adjacency matrix (num_nodes, num_nodes).
        """
        num_nodes = adj_matrix.size(0)
        device = adj_matrix.device
        
        # Normalize adjacency matrix (row normalization for random walk)
        # Add self-loops and normalize
        adj_normalized = adj_matrix + torch.eye(num_nodes, device=device)
        row_sum = adj_normalized.sum(dim=1, keepdim=True)
        adj_normalized = adj_normalized / (row_sum + 1e-6)
        
        # Store as buffer (automatically moves to correct device)
        self.adj_normalized = adj_normalized
    
    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        h0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input sequence (batch_size, seq_len, num_nodes, input_size).
            adj_matrix: Adjacency matrix (num_nodes, num_nodes). 
                       If None, uses pre-computed self.adj_normalized.
            h0: Initial hidden state (optional).
        
        Returns:
            Output (batch_size, num_nodes, output_size).
        """
        batch_size, seq_len, num_nodes, input_size = x.shape
        
        # Use pre-computed normalized adjacency if available, otherwise normalize on-the-fly
        if self.adj_normalized is not None:
            adj = self.adj_normalized
        elif adj_matrix is not None:
            # Fallback: normalize on-the-fly (for backward compatibility)
            adj_normalized = adj_matrix + torch.eye(num_nodes, device=adj_matrix.device)
            row_sum = adj_normalized.sum(dim=1, keepdim=True)
            adj = adj_normalized / (row_sum + 1e-6)
        else:
            raise ValueError("Either set_adj() must be called or adj_matrix must be provided")
        
        # Initialize hidden states for each layer
        if h0 is None:
            h = [torch.zeros(batch_size, num_nodes, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
        else:
            # CRITICAL FIX: Clone h0 for each layer to avoid shared tensor reference
            # Each layer should have its own independent hidden state
            h = [h0.clone() for _ in range(self.num_layers)]
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (batch_size, num_nodes, input_size)
            
            # Forward through layers
            for layer_idx, dcrnn_cell in enumerate(self.dcrnn_cells):
                h[layer_idx] = dcrnn_cell(x_t, h[layer_idx], adj)
                x_t = h[layer_idx]  # Use hidden state as input for next layer
            
            # Apply dropout (except for last time step)
            if t < seq_len - 1:
                for layer_idx in range(self.num_layers):
                    h[layer_idx] = self.dropout(h[layer_idx])
        
        # Use final hidden state from last layer
        h_final = h[-1]  # (batch_size, num_nodes, hidden_size)
        h_final = self.dropout(h_final)
        
        # Output layer
        output = self.fc(h_final)  # (batch_size, num_nodes, output_size)
        
        return output


# Legacy: used by older per-node seq training, not used in current full-graph pipeline
class GraphTimeSeriesDataset(Dataset):
    """Dataset for graph-based time series data.
    
    Similar to TimeSeriesDataset but organized for graph models.
    
    NOTE: This class is legacy and not used in the current full-graph temporal pipeline.
    The current implementation uses FullGraphDataset instead.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        station_ids: np.ndarray,
        node_indices: np.ndarray,
        sequence_length: int = 24,
        graph_station_ids: np.ndarray = None
    ):
        """Initialize dataset.
        
        Args:
            X: Feature array (n_samples, n_features).
            y: Target array (n_samples,).
            station_ids: Station IDs for each sample.
            node_indices: Node indices in graph for each sample.
            sequence_length: Length of input sequences.
            graph_station_ids: Station IDs in graph (for validation).
        """
        self.sequence_length = sequence_length
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.station_ids = station_ids
        self.node_indices = node_indices
        self.graph_station_ids = graph_station_ids
        
        # Build sequences grouped by station and node
        self.sequence_indices = []
        unique_nodes = np.unique(node_indices)
        
        for node_idx in unique_nodes:
            node_mask = node_indices == node_idx
            node_data_indices = np.where(node_mask)[0]
            
            if len(node_data_indices) < sequence_length:
                continue
            
            # Sort by time (assuming data is time-ordered)
            node_data_indices = np.sort(node_data_indices)
            
            # Generate sequences with sliding window
            for i in range(len(node_data_indices) - sequence_length + 1):
                seq_indices = node_data_indices[i:i + sequence_length]
                self.sequence_indices.append(seq_indices)
    
    def __len__(self):
        return len(self.sequence_indices)
    
    def __getitem__(self, idx):
        seq_indices = self.sequence_indices[idx]
        
        # Get sequence data
        X_seq = self.X[seq_indices]  # (seq_len, n_features)
        y_target = self.y[seq_indices[-1]]  # Target at last time step
        
        # Get node index for this sequence
        node_idx = self.node_indices[seq_indices[0]]
        
        return X_seq, y_target, node_idx


class DCRNNForecastModel(BaseGraphModel):
    """DCRNN model for frost forecasting.
    
    Wraps DCRNNModel with BaseGraphModel interface and full training logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DCRNN forecast model.
        
        Args:
            config: Model configuration dictionary.
        """
        super().__init__(config)
        
        # Model parameters
        model_params = config.get("model_params", {})
        self.hidden_size = model_params.get("hidden_size", 64)
        self.num_layers = model_params.get("num_layers", 2)
        self.num_diffusion_steps = model_params.get("num_diffusion_steps", 2)
        self.dropout = model_params.get("dropout", 0.2)
        self.sequence_length = model_params.get("sequence_length", 24)
        self.batch_size = model_params.get("batch_size", 32)
        self.epochs = model_params.get("epochs", 100)
        self.learning_rate = model_params.get("learning_rate", 0.0003)
        
        # Training parameters
        self.early_stopping = model_params.get("early_stopping", True)
        self.patience = model_params.get("patience", 20)
        self.min_delta = model_params.get("min_delta", 1e-6)
        self.use_amp = model_params.get("use_amp", True)
        self.gradient_clip = model_params.get("gradient_clip", 1.0)
        
        # Task type
        self.task_type = config.get("task_type", "classification")
        
        # Model will be initialized in fit()
        self.model = None
        self.num_nodes = None
        
        # Scalers and calibrator
        self._x_scaler = None
        self._y_scaler = None
        self._calibrator = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set=None,
        **kwargs
    ) -> "DCRNNForecastModel":
        """Train the DCRNN model.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            eval_set: Optional list of (X_val, y_val) tuples.
            **kwargs: Additional arguments:
                - station_ids: Station IDs for training data
                - station_ids_val: Station IDs for validation data
                - checkpoint_dir: Directory for checkpoints
                - log_file: Log file path
        
        Returns:
            Self for method chaining.
        """
        import numpy as np
        import gc
        
        # Setup logging (use v2 API if available)
        log_file = kwargs.get('log_file', None)
        if log_file and not hasattr(self, 'progress_logger'):
            from src.models.utils import ProgressLogger
            self.progress_logger = ProgressLogger(use_metric_schema=True)
            self.progress_logger.bind_files(brief_path=log_file, detailed_path=None)
        
        # Load or build graph
        if self.progress_logger:
            self.progress_logger.log("Loading/building graph structure...", flush=True)
        self.graph = self._load_or_build_graph(use_cache=True)
        self.num_nodes = len(self.graph['station_ids'])
        
        if self.progress_logger:
            self.progress_logger.log(
                f"  ✅ Graph loaded: {self.graph['graph_type']}={self.graph['graph_param']}, "
                f"nodes={self.num_nodes}, edges={self.graph['adj_matrix'].sum() / 2:.0f}",
                flush=True
            )
        
        # OPTIMIZATION 3: Full graph temporal design
        # Reorganize data as (T_all, N, F) instead of single-node sequences
        # This eliminates the need for expand operations and truly utilizes graph structure
        
        # Prepare node features
        node_features, station_ids = self._prepare_node_features(X)
        self.node_feature_size = node_features.shape[1]
        
        # Get node indices
        node_indices = self._get_station_indices(station_ids, self.graph['station_ids'])
        
        # Convert to numpy arrays
        y_array = y.values.astype(np.float32)
        
        # Handle NaN
        valid_mask = ~(np.isnan(y_array) | np.isnan(node_features).any(axis=1))
        node_features = node_features[valid_mask]
        y_array = y_array[valid_mask]
        node_indices = node_indices[valid_mask]
        station_ids = station_ids[valid_mask]
        
        if self.progress_logger:
            self.progress_logger.log(
                f"  ✅ Prepared {len(node_features)} samples, "
                f"features={self.node_feature_size}, nodes={self.num_nodes}",
                flush=True
            )
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        self._x_scaler = StandardScaler()
        node_features = self._x_scaler.fit_transform(node_features).astype(np.float32)
        
        if self.task_type != "classification":
            self._y_scaler = StandardScaler()
            y_array = self._y_scaler.fit_transform(y_array.reshape(-1, 1)).astype(np.float32).ravel()
        
        # Reorganize data as (T_all, N, F) - full graph temporal format
        # Group by time step, then by node
        # Use Date column if available, otherwise assume data is time-ordered
        
        # Check if we have Date and Hour columns for proper time grouping
        # CIMIS data is hourly, so we need Date + Hour to create unique time steps
        has_date = 'Date' in X.columns
        has_hour = 'Hour (PST)' in X.columns or 'Hour' in X.columns
        
        if has_date:
            # Group by Date + Hour to create unique hourly time steps
            # Since we filtered valid_mask, we need to map original indices
            X_with_time = X.copy()
            X_with_time = X_with_time[valid_mask]  # Apply same mask
            
            # Create time step identifier: Date + Hour (properly formatted for sorting)
            if has_hour:
                hour_col = 'Hour (PST)' if 'Hour (PST)' in X_with_time.columns else 'Hour'
                # CIMIS hour like 100, 200 -> pad to "0100", "0200" for correct string sorting
                # More robust: handle both int and string hour formats
                try:
                    hour_numeric = pd.to_numeric(X_with_time[hour_col], errors='coerce')
                    hour_str = hour_numeric.astype('Int64').astype(str).str.zfill(4)
                except (ValueError, TypeError) as e:
                    _logger.debug(f"Could not convert hour column to numeric: {e}. Using string format.")
                    hour_str = X_with_time[hour_col].astype(str).str.zfill(4)
                
                ts_str = X_with_time['Date'].astype(str) + " " + hour_str
                timestamps = pd.to_datetime(ts_str, format="%m/%d/%Y %H%M", errors='coerce', infer_datetime_format=True)
                if timestamps.isna().all():
                    time_identifiers = ts_str.values
                elif timestamps.isna().any():
                    time_identifiers = []
                    for ts, ts_str_val in zip(timestamps, ts_str):
                        if pd.isna(ts):
                            time_identifiers.append(ts_str_val)
                        else:
                            time_identifiers.append(ts)
                    time_identifiers = np.array(time_identifiers)
                else:
                    time_identifiers = timestamps.values
            else:
                # Fallback: use Date only (less precise, but better than nothing)
                timestamps = pd.to_datetime(X_with_time['Date'], errors='coerce', infer_datetime_format=True)
                if timestamps.isna().all():
                    time_identifiers = X_with_time['Date'].values
                elif timestamps.isna().any():
                    time_identifiers = []
                    for ts, date_val in zip(timestamps, X_with_time['Date'].values):
                        if pd.isna(ts):
                            time_identifiers.append(date_val)
                        else:
                            time_identifiers.append(ts)
                    time_identifiers = np.array(time_identifiers)
                else:
                    time_identifiers = timestamps.values
            
            # Group by time identifier
            time_groups = {}
            for i, (feat, target, node_idx, time_id) in enumerate(
                zip(node_features, y_array, node_indices, time_identifiers)
            ):
                if time_id not in time_groups:
                    time_groups[time_id] = {}
                time_groups[time_id][node_idx] = (feat, target, i)
        else:
            # Fallback: assume data is time-ordered and group by consecutive samples
            # This assumes samples come in batches where each batch = one time step with all nodes
            # Group samples that appear together (likely same time step)
            time_groups = {}
            current_time = 0
            prev_node_set = set()
            
            for i, (feat, target, node_idx) in enumerate(zip(node_features, y_array, node_indices)):
                # If we see a node we've seen before in current_time, it's likely a new time step
                if node_idx in prev_node_set and len(prev_node_set) > 1:
                    current_time += 1
                    prev_node_set = set()
                
                if current_time not in time_groups:
                    time_groups[current_time] = {}
                time_groups[current_time][node_idx] = (feat, target, i)
                prev_node_set.add(node_idx)
        
        # Convert to (T, N, F) format
        # Find all unique time steps and nodes
        all_time_steps = sorted(time_groups.keys())
        all_nodes = list(range(self.num_nodes))
        
        # Build full graph temporal tensor: (T, N, F)
        T_all = len(all_time_steps)
        N = self.num_nodes
        F = self.node_feature_size
        
        X_graph = np.full((T_all, N, F), np.nan, dtype=np.float32)
        y_graph = np.full((T_all, N), np.nan, dtype=np.float32)
        valid_mask_graph = np.zeros((T_all, N), dtype=bool)
        
        for t_idx, time_step in enumerate(all_time_steps):
            for node_idx in all_nodes:
                if node_idx in time_groups[time_step]:
                    feat, target, _ = time_groups[time_step][node_idx]
                    X_graph[t_idx, node_idx, :] = feat
                    y_graph[t_idx, node_idx] = target
                    valid_mask_graph[t_idx, node_idx] = True
        
        # Handle NaN values: forward fill then backward fill
        # This ensures we have valid values for graph convolution
        for node_idx in range(N):
            for feat_idx in range(F):
                col = X_graph[:, node_idx, feat_idx]
                # Forward fill then backward fill (using modern pandas API)
                mask = ~np.isnan(col)
                if mask.any():
                    col_series = pd.Series(col)
                    # Use ffill() and bfill() instead of deprecated method parameter
                    col = col_series.ffill().bfill().values
                    X_graph[:, node_idx, feat_idx] = col
        
        # Create sliding window sequences: (T, N, F) -> sequences of (seq_len, N, F)
        sequences = []
        for t in range(self.sequence_length - 1, T_all):
            # Extract sequence: X_graph[t-seq_len+1:t+1] -> (seq_len, N, F)
            X_seq = X_graph[t - self.sequence_length + 1:t + 1, :, :]  # (seq_len, N, F)
            y_target = y_graph[t, :]  # (N,) - targets for all nodes at time t
            
            # CRITICAL FIX: valid_mask should be for the target time step, not the window
            # Only nodes with valid labels at time t should be used for loss calculation
            valid_target = ~np.isnan(y_target)  # (N,) - nodes with valid labels at time t
            
            # Optional: check if sequence has any valid data in the window (for reference)
            seq_valid = valid_mask_graph[t - self.sequence_length + 1:t + 1, :].any(axis=0)  # (N,)
            
            if valid_target.any():
                sequences.append({
                    'X': X_seq,  # (seq_len, N, F)
                    'y': y_target,  # (N,)
                    'valid_mask': valid_target,  # (N,) - which nodes have valid targets at time t
                    'time_step': t
                })
        
        if len(sequences) == 0:
            raise ValueError("No valid sequences created. Check sequence_length and data continuity.")
        
        if self.progress_logger:
            self.progress_logger.log(
                f"  ✅ Created {len(sequences)} full-graph sequences (T={T_all}, N={N}, F={F})",
                flush=True
            )
        
        # Split train/val
        if eval_set is not None and len(eval_set) > 0:
            # Use external validation set
            X_val, y_val = eval_set[0]
            node_features_val, station_ids_val = self._prepare_node_features(X_val)
            node_indices_val = self._get_station_indices(station_ids_val, self.graph['station_ids'])
            y_val_array = y_val.values.astype(np.float32)
            
            # Handle NaN
            valid_mask_val = ~(np.isnan(y_val_array) | np.isnan(node_features_val).any(axis=1))
            node_features_val = node_features_val[valid_mask_val]
            y_val_array = y_val_array[valid_mask_val]
            node_indices_val = node_indices_val[valid_mask_val]
            
            # Apply scaling
            node_features_val = self._x_scaler.transform(node_features_val).astype(np.float32)
            if self._y_scaler is not None:
                y_val_array = self._y_scaler.transform(y_val_array.reshape(-1, 1)).astype(np.float32).ravel()
            
            # Reorganize validation data as (T_all, N, F) - same as training
            has_date_val = 'Date' in X_val.columns
            has_hour_val = 'Hour (PST)' in X_val.columns or 'Hour' in X_val.columns
            
            if has_date_val:
                X_val_with_time = X_val.copy()
                X_val_with_time = X_val_with_time[valid_mask_val]
                
                # Create time step identifier: Date + Hour (same as training)
                if has_hour_val:
                    hour_col = 'Hour (PST)' if 'Hour (PST)' in X_val_with_time.columns else 'Hour'
                    # CIMIS hour like 100, 200 -> pad to "0100", "0200" for correct string sorting
                    # More robust: handle both int and string hour formats
                    try:
                        hour_numeric = pd.to_numeric(X_val_with_time[hour_col], errors='coerce')
                        hour_str = hour_numeric.astype('Int64').astype(str).str.zfill(4)
                    except (ValueError, TypeError) as e:
                        _logger.debug(f"Could not convert hour column to numeric: {e}. Using string format.")
                        hour_str = X_val_with_time[hour_col].astype(str).str.zfill(4)
                    
                    ts_str = X_val_with_time['Date'].astype(str) + " " + hour_str
                    timestamps = pd.to_datetime(ts_str, format="%m/%d/%Y %H%M", errors='coerce', infer_datetime_format=True)
                    if timestamps.isna().all():
                        val_time_identifiers = ts_str.values
                    elif timestamps.isna().any():
                        val_time_identifiers = []
                        for ts, ts_str_val in zip(timestamps, ts_str):
                            if pd.isna(ts):
                                val_time_identifiers.append(ts_str_val)
                            else:
                                val_time_identifiers.append(ts)
                        val_time_identifiers = np.array(val_time_identifiers)
                    else:
                        val_time_identifiers = timestamps.values
                else:
                    timestamps = pd.to_datetime(X_val_with_time['Date'], errors='coerce')
                    if timestamps.isna().any():
                        val_time_identifiers = X_val_with_time['Date'].values
                    else:
                        val_time_identifiers = timestamps.values
                
                val_time_groups = {}
                for i, (feat, target, node_idx, time_id) in enumerate(
                    zip(node_features_val, y_val_array, node_indices_val, val_time_identifiers)
                ):
                    if time_id not in val_time_groups:
                        val_time_groups[time_id] = {}
                    val_time_groups[time_id][node_idx] = (feat, target, i)
            else:
                val_time_groups = {}
                current_time = 0
                prev_node_set = set()
                
                for i, (feat, target, node_idx) in enumerate(zip(node_features_val, y_val_array, node_indices_val)):
                    if node_idx in prev_node_set and len(prev_node_set) > 1:
                        current_time += 1
                        prev_node_set = set()
                    
                    if current_time not in val_time_groups:
                        val_time_groups[current_time] = {}
                    val_time_groups[current_time][node_idx] = (feat, target, i)
                    prev_node_set.add(node_idx)
            
            # Build validation graph tensor: (T, N, F)
            val_time_steps = sorted(val_time_groups.keys())
            T_val = len(val_time_steps)
            N = self.num_nodes
            F = self.node_feature_size
            
            X_val_graph = np.full((T_val, N, F), np.nan, dtype=np.float32)
            y_val_graph = np.full((T_val, N), np.nan, dtype=np.float32)
            valid_mask_val_graph = np.zeros((T_val, N), dtype=bool)
            
            for t_idx, time_step in enumerate(val_time_steps):
                for node_idx in range(N):
                    if node_idx in val_time_groups[time_step]:
                        feat, target, _ = val_time_groups[time_step][node_idx]
                        X_val_graph[t_idx, node_idx, :] = feat
                        y_val_graph[t_idx, node_idx] = target
                        valid_mask_val_graph[t_idx, node_idx] = True
            
            # Create validation sequences
            val_sequences = []
            for t in range(self.sequence_length - 1, T_val):
                X_seq = X_val_graph[t - self.sequence_length + 1:t + 1, :, :]
                y_target = y_val_graph[t, :]
                
                # CRITICAL FIX: valid_mask should be for the target time step
                valid_target = ~np.isnan(y_target)  # (N,) - nodes with valid labels at time t
                
                if valid_target.any():
                    val_sequences.append({
                        'X': X_seq,
                        'y': y_target,
                        'valid_mask': valid_target,  # Use target time step mask
                        'time_step': t
                    })
            
            train_sequences = sequences
        else:
            # Internal split
            train_size = int(0.8 * len(sequences))
            train_sequences = sequences[:train_size]
            val_sequences = sequences[train_size:]
        
        # Create dataset wrapper for full-graph sequences
        class FullGraphDataset(Dataset):
            def __init__(self, sequences):
                self.sequences = sequences
            
            def __len__(self):
                return len(self.sequences)
            
            def __getitem__(self, idx):
                seq = self.sequences[idx]
                # X: (seq_len, N, F) - already in full graph format!
                # y: (N,) - targets for all nodes
                # valid_mask: (N,) - which nodes have valid targets
                X_seq = torch.FloatTensor(seq['X'])  # (seq_len, N, F)
                y_target = torch.FloatTensor(seq['y'])  # (N,)
                valid_mask = torch.BoolTensor(seq['valid_mask'])  # (N,)
                
                return X_seq, y_target, valid_mask
        
        train_dataset = FullGraphDataset(train_sequences)
        val_dataset = FullGraphDataset(val_sequences)
        
        # Create data loaders with optimized settings
        # Increase num_workers for better data loading parallelism
        num_workers = min(8, max(2, os.cpu_count() // 2))
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else 2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else 2
        )
        
        # Initialize model
        self.model = DCRNNModel(
            num_nodes=self.num_nodes,
            input_size=self.node_feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_diffusion_steps=self.num_diffusion_steps,
            dropout=self.dropout,
            output_size=1
        ).to(self.device)
        
        # Prepare and pre-compute normalized adjacency matrix
        adj_matrix = torch.FloatTensor(self.graph['adj_matrix']).to(self.device)
        self.model.set_adj(adj_matrix)  # Pre-compute normalization once
        
        # Loss function
        if self.task_type == "classification":
            # Compute pos_weight for imbalanced data
            pos_count = (y_array > 0.5).sum()
            neg_count = len(y_array) - pos_count
            pos_weight = torch.tensor([neg_count / (pos_count + 1e-6)], device=self.device)
            self._pos_weight = pos_weight
            
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.MSELoss()
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # LR scheduler
        # Remove verbose parameter for compatibility with older PyTorch versions
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # AMP scaler
        scaler = None
        if self.use_amp and torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        import time
        self._train_start_time = time.time()
        
        if self.progress_logger:
            self.progress_logger.log(
                f"Starting training: {len(train_loader)} train batches, {len(val_loader)} val batches, "
                f"{self.epochs} epochs",
                flush=True
            )
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss_sum = 0.0
            train_loss_count = 0
            for batch_X, batch_y, batch_valid_mask in train_loader:
                # batch_X: (batch_size, seq_len, N, F) - already in full graph format!
                # batch_y: (batch_size, N) - targets for all nodes
                # batch_valid_mask: (batch_size, N) - which nodes have valid targets
                
                batch_size, seq_len, N, F = batch_X.shape
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                batch_valid_mask = batch_valid_mask.to(self.device, non_blocking=True)
                
                # No expand needed! batch_X is already (B, T, N, F)
                optimizer.zero_grad()
                
                # Use a flag to track if we have valid nodes (avoid continue inside with-block)
                has_valid = True
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        output = self.model(batch_X)  # (batch_size, N, 1), uses pre-computed adj
                        output = output.squeeze(-1)  # (batch_size, N)
                        
                        # Compute loss only for valid nodes
                        output_flat = output[batch_valid_mask]
                        y_flat = batch_y[batch_valid_mask]
                        
                        if len(output_flat) > 0:
                            loss = criterion(output_flat, y_flat)
                        else:
                            has_valid = False
                    
                    if not has_valid:
                        continue  # Skip if no valid nodes
                    
                    scaler.scale(loss).backward()
                    if self.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = self.model(batch_X)  # Uses pre-computed adj
                    output = output.squeeze(-1)  # (batch_size, N)
                    
                    # Compute loss only for valid nodes
                    output_flat = output[batch_valid_mask]
                    y_flat = batch_y[batch_valid_mask]
                    
                    if len(output_flat) > 0:
                        loss = criterion(output_flat, y_flat)
                    else:
                        continue  # Skip if no valid nodes
                    
                    loss.backward()
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    optimizer.step()
                
                # Uniformly accumulate loss outside branches (as long as not continued)
                train_loss_sum += loss.item()
                train_loss_count += 1
            
            # Validation
            self.model.eval()
            val_loss_sum = 0.0
            val_loss_count = 0
            with torch.no_grad():
                for batch_X, batch_y, batch_valid_mask in val_loader:
                    batch_size, seq_len, N, F = batch_X.shape
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    batch_valid_mask = batch_valid_mask.to(self.device, non_blocking=True)
                    
                    # No expand needed! batch_X is already (B, T, N, F)
                    output = self.model(batch_X)  # Uses pre-computed adj
                    output = output.squeeze(-1)  # (batch_size, N)
                    
                    # Compute loss only for valid nodes
                    output_flat = output[batch_valid_mask]
                    y_flat = batch_y[batch_valid_mask]
                    
                    if len(output_flat) > 0:
                        loss = criterion(output_flat, y_flat)
                        val_loss_sum += loss.item()
                        val_loss_count += 1
            
            train_loss = train_loss_sum / max(1, train_loss_count)
            # Handle edge case: if no valid validation batches, set val_loss to inf
            if val_loss_count == 0:
                val_loss = float('inf')
            else:
                val_loss = val_loss_sum / val_loss_count
            
            scheduler.step(val_loss)
            
            # Log progress more frequently (every epoch for first 10, then every 5)
            if self.progress_logger:
                log_frequency = 1 if epoch < 10 else 5
                if epoch % log_frequency == 0 or epoch == self.epochs - 1:
                    elapsed = time.time() - self._train_start_time
                    eta = (elapsed / (epoch + 1)) * (self.epochs - epoch - 1) if epoch > 0 else 0
                    current_lr = optimizer.param_groups[0]['lr']
                    self.progress_logger.log(
                        f"Epoch {epoch+1}/{self.epochs} - train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                        f"lr={current_lr:.6f}, ETA: {eta/60:.1f}m",
                        flush=True
                    )
            
            # Early stopping
            if self.early_stopping:
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.progress_logger:
                            self.progress_logger.log(
                                f"Early stopping at epoch {epoch}",
                                flush=True
                            )
                        break
        
        self.is_fitted = True
        
        # Fit probability calibrator if classification
        if self.task_type == "classification" and self.config.get("model_params", {}).get("use_probability_calibration", True):
            # Get validation predictions for calibration
            self.model.eval()
            val_probas = []
            val_targets = []
            with torch.no_grad():
                for batch_X, batch_y, batch_valid_mask in val_loader:
                    # batch_X: (B, T, N, F) - already in full graph format
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_valid_mask = batch_valid_mask.to(self.device)
                    
                    # Forward: (B, N, 1) -> (B, N)
                    output = self.model(batch_X).squeeze(-1)
                    
                    # Flatten valid nodes
                    output_flat = output[batch_valid_mask]  # logits
                    y_flat = batch_y[batch_valid_mask]      # targets
                    
                    if output_flat.numel() == 0:
                        continue
                    
                    proba = torch.sigmoid(output_flat).cpu().numpy()
                    val_probas.extend(proba)
                    val_targets.extend(y_flat.cpu().numpy())
            
            if len(val_probas) > 0:
                self._calibrator = ProbabilityCalibrator(
                    method=self.config.get("model_params", {}).get("calibration_method", "platt")
                )
                self._calibrator.fit(np.array(val_probas), np.array(val_targets))
        
        return self
    
    def _build_graph_tensor_for_inference(
        self,
        X: pd.DataFrame,
        station_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Build graph temporal tensor for inference (same format as training).
        
        Args:
            X: Feature DataFrame.
            station_ids: Optional array of station IDs.
        
        Returns:
            Tuple of (X_graph, valid_mask_graph, time_steps):
            - X_graph: (T_all, N, F) graph temporal tensor
            - valid_mask_graph: (T_all, N) boolean mask for valid nodes
            - time_steps: List of time step identifiers
        """
        # Prepare node features
        node_features, station_ids_array = self._prepare_node_features(X, station_ids)
        node_indices = self._get_station_indices(station_ids_array, self.graph['station_ids'])
        
        # Apply scaling
        node_features = self._x_scaler.transform(node_features).astype(np.float32)
        
        # Handle NaN
        valid_mask = ~np.isnan(node_features).any(axis=1)
        node_features = node_features[valid_mask]
        node_indices = node_indices[valid_mask]
        station_ids_array = station_ids_array[valid_mask]
        
        # Group by time (same logic as fit)
        has_date = 'Date' in X.columns
        has_hour = 'Hour (PST)' in X.columns or 'Hour' in X.columns
        
        if has_date:
            X_with_time = X.copy()
            X_with_time = X_with_time[valid_mask]
            
            if has_hour:
                hour_col = 'Hour (PST)' if 'Hour (PST)' in X_with_time.columns else 'Hour'
                # CIMIS hour like 100, 200 -> pad to "0100", "0200" for correct string sorting
                # More robust: handle both int and string hour formats
                try:
                    hour_numeric = pd.to_numeric(X_with_time[hour_col], errors='coerce')
                    hour_str = hour_numeric.astype('Int64').astype(str).str.zfill(4)
                except (ValueError, TypeError) as e:
                    _logger.debug(f"Could not convert hour column to numeric: {e}. Using string format.")
                    hour_str = X_with_time[hour_col].astype(str).str.zfill(4)
                
                ts_str = X_with_time['Date'].astype(str) + " " + hour_str
                timestamps = pd.to_datetime(ts_str, format="%m/%d/%Y %H%M", errors='coerce', infer_datetime_format=True)
                if timestamps.isna().all():
                    time_identifiers = ts_str.values
                elif timestamps.isna().any():
                    time_identifiers = []
                    for ts, ts_str_val in zip(timestamps, ts_str):
                        if pd.isna(ts):
                            time_identifiers.append(ts_str_val)
                        else:
                            time_identifiers.append(ts)
                    time_identifiers = np.array(time_identifiers)
                else:
                    time_identifiers = timestamps.values
            else:
                timestamps = pd.to_datetime(X_with_time['Date'], errors='coerce', infer_datetime_format=True)
                if timestamps.isna().all():
                    time_identifiers = X_with_time['Date'].values
                elif timestamps.isna().any():
                    time_identifiers = []
                    for ts, date_val in zip(timestamps, X_with_time['Date'].values):
                        if pd.isna(ts):
                            time_identifiers.append(date_val)
                        else:
                            time_identifiers.append(ts)
                    time_identifiers = np.array(time_identifiers)
                else:
                    time_identifiers = timestamps.values
            
            time_groups = {}
            for i, (feat, node_idx, time_id) in enumerate(
                zip(node_features, node_indices, time_identifiers)
            ):
                if time_id not in time_groups:
                    time_groups[time_id] = {}
                time_groups[time_id][node_idx] = feat
        else:
            # Fallback: assume time-ordered
            time_groups = {}
            current_time = 0
            prev_node_set = set()
            
            for i, (feat, node_idx) in enumerate(zip(node_features, node_indices)):
                if node_idx in prev_node_set and len(prev_node_set) > 1:
                    current_time += 1
                    prev_node_set = set()
                
                if current_time not in time_groups:
                    time_groups[current_time] = {}
                time_groups[current_time][node_idx] = feat
                prev_node_set.add(node_idx)
        
        # Build graph tensor: (T, N, F)
        all_time_steps = sorted(time_groups.keys())
        T_all = len(all_time_steps)
        N = self.num_nodes
        F = self.node_feature_size
        
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
    
    def predict(
        self,
        X: pd.DataFrame,
        station_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Make point predictions using full-graph format (consistent with training).
        
        Args:
            X: Feature DataFrame.
            station_ids: Optional array of station IDs.
        
        Returns:
            Array of predictions (flattened, only for valid nodes).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        
        # Build graph temporal tensor
        X_graph, valid_mask_graph, time_steps = self._build_graph_tensor_for_inference(X, station_ids)
        
        if len(time_steps) == 0:
            return np.array([])
        
        # Create sliding window sequences
        seqs = []
        seq_masks = []
        
        for t in range(self.sequence_length - 1, len(time_steps)):
            X_seq = X_graph[t - self.sequence_length + 1:t + 1, :, :]  # (seq_len, N, F)
            mask_seq = valid_mask_graph[t]  # (N,) - valid nodes at current time
            seqs.append(X_seq)
            seq_masks.append(mask_seq)
        
        if not seqs:
            return np.array([])
        
        # Batch process
        batch_X = torch.FloatTensor(np.stack(seqs)).to(self.device)  # (B, seq_len, N, F)
        batch_valid_mask = torch.BoolTensor(np.stack(seq_masks)).to(self.device)  # (B, N)
        
        predictions = []
        with torch.no_grad():
            output = self.model(batch_X).squeeze(-1)  # (B, N)
            
            # Extract predictions for valid nodes only
            for b in range(batch_X.shape[0]):
                valid_nodes = batch_valid_mask[b]
                if valid_nodes.any():
                    pred_b = output[b, valid_nodes].cpu().numpy()
                    predictions.extend(pred_b)
        
        predictions = np.array(predictions)
        
        # Inverse transform if regression
        if self.task_type != "classification" and self._y_scaler is not None:
            predictions = self._y_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
        
        # Convert to binary predictions if classification
        if self.task_type == "classification":
            predictions = (predictions > 0).astype(int)
        
        return predictions
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        station_ids: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Predict probabilities using full-graph format (consistent with training).
        
        Args:
            X: Feature DataFrame.
            station_ids: Optional array of station IDs.
        
        Returns:
            Array of probabilities (flattened, only for valid nodes).
        """
        if self.task_type != "classification":
            return None
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        
        # Build graph temporal tensor (reuse same logic as predict)
        X_graph, valid_mask_graph, time_steps = self._build_graph_tensor_for_inference(X, station_ids)
        
        if len(time_steps) == 0:
            return np.array([])
        
        # Create sliding window sequences
        seqs = []
        seq_masks = []
        
        for t in range(self.sequence_length - 1, len(time_steps)):
            X_seq = X_graph[t - self.sequence_length + 1:t + 1, :, :]  # (seq_len, N, F)
            mask_seq = valid_mask_graph[t]  # (N,) - valid nodes at current time
            seqs.append(X_seq)
            seq_masks.append(mask_seq)
        
        if not seqs:
            return np.array([])
        
        # Batch process
        batch_X = torch.FloatTensor(np.stack(seqs)).to(self.device)  # (B, seq_len, N, F)
        batch_valid_mask = torch.BoolTensor(np.stack(seq_masks)).to(self.device)  # (B, N)
        
        probabilities = []
        with torch.no_grad():
            output = self.model(batch_X).squeeze(-1)  # (B, N)
            prob = torch.sigmoid(output)  # (B, N)
            
            # Extract probabilities for valid nodes only
            for b in range(batch_X.shape[0]):
                valid_nodes = batch_valid_mask[b]
                if valid_nodes.any():
                    prob_b = prob[b, valid_nodes].cpu().numpy()
                    probabilities.extend(prob_b)
        
        probabilities = np.array(probabilities)
        
        # Apply calibration if available
        if self._calibrator is not None:
            probabilities = self._calibrator.transform(probabilities)
        
        return probabilities
    
    @classmethod
    def load(cls, path: Path) -> "DCRNNForecastModel":
        """Load model from disk.
        
        Args:
            path: Directory path containing saved model.
        
        Returns:
            Loaded model instance.
        """
        path = Path(path)
        
        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        instance = cls(metadata['config'])
        instance.graph = GraphBuilder.load_graph(path / "graph.pkl")
        instance.num_nodes = len(instance.graph['station_ids'])
        instance.node_feature_size = metadata['node_feature_size']
        instance.is_fitted = metadata['is_fitted']
        
        # Initialize model
        instance.model = DCRNNModel(
            num_nodes=instance.num_nodes,
            input_size=instance.node_feature_size,
            hidden_size=instance.hidden_size,
            num_layers=instance.num_layers,
            num_diffusion_steps=instance.num_diffusion_steps,
            dropout=instance.dropout,
            output_size=1
        ).to(instance.device)
        
        # Load model weights
        model_path = path / "model.pth"
        instance.model.load_state_dict(torch.load(model_path, map_location=instance.device))
        instance.model.eval()
        
        # Set adjacency matrix (required for forward pass)
        adj_matrix = torch.FloatTensor(instance.graph['adj_matrix']).to(instance.device)
        instance.model.set_adj(adj_matrix)
        
        # Load scalers
        if (path / "x_scaler.pkl").exists():
            with open(path / "x_scaler.pkl", 'rb') as f:
                instance._x_scaler = pickle.load(f)
        if (path / "y_scaler.pkl").exists():
            with open(path / "y_scaler.pkl", 'rb') as f:
                instance._y_scaler = pickle.load(f)
        
        # Load calibrator
        if (path / "calibrator.pkl").exists():
            with open(path / "calibrator.pkl", 'rb') as f:
                instance._calibrator = pickle.load(f)
        
        return instance

