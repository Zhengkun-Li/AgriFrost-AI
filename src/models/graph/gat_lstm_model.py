"""GAT-LSTM (Graph Attention Network + LSTM) model for spatial-temporal forecasting.

GAT-LSTM combines graph attention mechanism for dynamic spatial relationships
with LSTM for temporal modeling. It's particularly suitable for complex spatial
relationships (e.g., wind direction effects).
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import os
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_graph_model import BaseGraphModel
from src.utils.losses import FocalLoss
from src.utils.calibration import ProbabilityCalibrator

if not TORCH_AVAILABLE:
    raise ImportError("PyTorch is required for GAT-LSTM models. Please install torch.")


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer.
    
    Implements multi-head graph attention mechanism.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        alpha: float = 0.2
    ):
        """Initialize graph attention layer.
        
        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            alpha: LeakyReLU negative slope.
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        
        # Linear transformations for each head
        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, self.head_dim))
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * self.head_dim, 1))
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Apply graph attention.
        
        Args:
            x: Input features (batch_size, num_nodes, in_features).
            adj_matrix: Adjacency matrix (num_nodes, num_nodes).
        
        Returns:
            Output features (batch_size, num_nodes, out_features).
        """
        batch_size, num_nodes, in_features = x.shape
        
        # Apply linear transformation for each head
        h = torch.matmul(x.unsqueeze(1), self.W)  # (batch_size, num_heads, num_nodes, head_dim)
        h = h.transpose(1, 2)  # (batch_size, num_nodes, num_heads, head_dim)
        
        # Compute attention scores
        # For each node pair (i, j), compute attention: a^T [Wh_i || Wh_j]
        h_i = h.unsqueeze(2)  # (batch_size, num_nodes, 1, num_heads, head_dim)
        h_j = h.unsqueeze(1)  # (batch_size, 1, num_nodes, num_heads, head_dim)
        
        # Concatenate
        h_concat = torch.cat([h_i.expand(-1, -1, num_nodes, -1, -1),
                              h_j.expand(-1, num_nodes, -1, -1, -1)], dim=-1)
        # (batch_size, num_nodes, num_nodes, num_heads, 2*head_dim)
        
        # Compute attention scores
        e = torch.matmul(h_concat, self.a.unsqueeze(0).unsqueeze(0).unsqueeze(0))
        # (batch_size, num_nodes, num_nodes, num_heads, 1)
        e = e.squeeze(-1)  # (batch_size, num_nodes, num_nodes, num_heads)
        e = self.leaky_relu(e)
        
        # Apply adjacency mask
        adj_mask = adj_matrix.unsqueeze(0).unsqueeze(-1)  # (1, num_nodes, num_nodes, 1)
        e = e.masked_fill(adj_mask == 0, float('-inf'))
        
        # Softmax
        attention = F.softmax(e, dim=2)  # (batch_size, num_nodes, num_nodes, num_heads)
        attention = self.dropout(attention)
        
        # Apply attention to features
        h_out = torch.matmul(attention, h)  # (batch_size, num_nodes, num_nodes, num_heads, head_dim)
        # Sum over neighbor dimension
        h_out = h_out.sum(dim=2)  # (batch_size, num_nodes, num_heads, head_dim)
        
        # Concatenate heads
        h_out = h_out.reshape(batch_size, num_nodes, self.out_features)
        
        return h_out


class GATLSTMModel(nn.Module):
    """GAT-LSTM model for spatial-temporal forecasting.
    
    Combines graph attention for spatial modeling with LSTM for temporal modeling.
    """
    
    def __init__(
        self,
        num_nodes: int,
        input_size: int,
        hidden_size: int = 64,
        num_gat_layers: int = 2,
        num_lstm_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """Initialize GAT-LSTM model.
        
        Args:
            num_nodes: Number of nodes in graph.
            input_size: Input feature dimension.
            hidden_size: Hidden state dimension.
            num_gat_layers: Number of GAT layers.
            num_lstm_layers: Number of LSTM layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            output_size: Output dimension.
        """
        super(GATLSTMModel, self).__init__()
        self.num_nodes = num_nodes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_gat_layers = num_gat_layers
        self.num_lstm_layers = num_lstm_layers
        
        # GAT layers for spatial modeling
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.gat_layers.append(
                GraphAttentionLayer(in_dim, hidden_size, num_heads, dropout)
            )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor,
        h0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input sequence (batch_size, seq_len, num_nodes, input_size).
            adj_matrix: Adjacency matrix (num_nodes, num_nodes).
            h0: Initial LSTM hidden state (optional).
        
        Returns:
            Output (batch_size, num_nodes, output_size).
        """
        batch_size, seq_len, num_nodes, input_size = x.shape
        
        # Apply GAT at each time step
        gat_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (batch_size, num_nodes, input_size)
            
            # Apply GAT layers
            for gat_layer in self.gat_layers:
                x_t = gat_layer(x_t, adj_matrix)
                x_t = self.dropout(x_t)
            
            gat_outputs.append(x_t)
        
        # Stack: (batch_size, seq_len, num_nodes, hidden_size)
        gat_sequence = torch.stack(gat_outputs, dim=1)
        
        # Apply LSTM for each node
        # Reshape: (batch_size * num_nodes, seq_len, hidden_size)
        gat_sequence = gat_sequence.view(batch_size * num_nodes, seq_len, self.hidden_size)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(gat_sequence, h0)
        # lstm_out: (batch_size * num_nodes, seq_len, hidden_size)
        # Use final hidden state
        lstm_final = lstm_out[:, -1, :]  # (batch_size * num_nodes, hidden_size)
        
        # Reshape back: (batch_size, num_nodes, hidden_size)
        lstm_final = lstm_final.view(batch_size, num_nodes, self.hidden_size)
        lstm_final = self.dropout(lstm_final)
        
        # Output layer
        output = self.fc(lstm_final)  # (batch_size, num_nodes, output_size)
        
        return output


class GATLSTMForecastModel(BaseGraphModel):
    """GAT-LSTM model for frost forecasting.
    
    Wraps GATLSTMModel with BaseGraphModel interface and full training logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize GAT-LSTM forecast model."""
        super().__init__(config)
        
        model_params = config.get("model_params", {})
        self.hidden_size = model_params.get("hidden_size", 64)
        self.num_gat_layers = model_params.get("num_gat_layers", 2)
        self.num_lstm_layers = model_params.get("num_lstm_layers", 2)
        self.num_heads = model_params.get("num_heads", 4)
        self.dropout = model_params.get("dropout", 0.2)
        self.sequence_length = model_params.get("sequence_length", 24)
        self.batch_size = model_params.get("batch_size", 32)
        self.epochs = model_params.get("epochs", 100)
        self.learning_rate = model_params.get("learning_rate", 0.0003)
        
        self.early_stopping = model_params.get("early_stopping", True)
        self.patience = model_params.get("patience", 20)
        self.min_delta = model_params.get("min_delta", 1e-6)
        self.use_amp = model_params.get("use_amp", True)
        self.gradient_clip = model_params.get("gradient_clip", 1.0)
        
        self.task_type = config.get("task_type", "classification")
        
        self.model = None
        self.num_nodes = None
        self._x_scaler = None
        self._y_scaler = None
        self._calibrator = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set=None,
        **kwargs
    ) -> "GATLSTMForecastModel":
        """Train the GAT-LSTM model.
        
        Reuses training logic from DCRNN (similar structure).
        """
        import numpy as np
        import gc
        
        log_file = kwargs.get('log_file', None)
        if log_file and not hasattr(self, 'progress_logger'):
            from src.models.utils import ProgressLogger
            self.progress_logger = ProgressLogger(log_file)
        
        # Load graph
        if self.progress_logger:
            self.progress_logger.log("Loading/building graph structure...", flush=True)
        self.graph = self._load_or_build_graph(use_cache=True)
        self.num_nodes = len(self.graph['station_ids'])
        
        # Prepare data (reuse DCRNN logic)
        node_features, station_ids = self._prepare_node_features(X)
        self.node_feature_size = node_features.shape[1]
        node_indices = self._get_station_indices(station_ids, self.graph['station_ids'])
        y_array = y.values.astype(np.float32)
        
        valid_mask = ~(np.isnan(y_array) | np.isnan(node_features).any(axis=1))
        node_features = node_features[valid_mask]
        y_array = y_array[valid_mask]
        node_indices = node_indices[valid_mask]
        station_ids = station_ids[valid_mask]
        
        from sklearn.preprocessing import StandardScaler
        self._x_scaler = StandardScaler()
        node_features = self._x_scaler.fit_transform(node_features).astype(np.float32)
        
        if self.task_type != "classification":
            self._y_scaler = StandardScaler()
            y_array = self._y_scaler.fit_transform(y_array.reshape(-1, 1)).astype(np.float32).ravel()
        
        # Organize by node and create sequences
        node_data_dict = {}
        for i, node_idx in enumerate(node_indices):
            if node_idx not in node_data_dict:
                node_data_dict[node_idx] = {'X': [], 'y': []}
            node_data_dict[node_idx]['X'].append(node_features[i])
            node_data_dict[node_idx]['y'].append(y_array[i])
        
        sequences = []
        for node_idx, data in node_data_dict.items():
            X_node = np.array(data['X'])
            y_node = np.array(data['y'])
            
            if len(X_node) < self.sequence_length:
                continue
            
            for i in range(len(X_node) - self.sequence_length + 1):
                sequences.append({
                    'X': X_node[i:i + self.sequence_length],
                    'y': y_node[i + self.sequence_length - 1],
                    'node_idx': node_idx
                })
        
        if len(sequences) == 0:
            raise ValueError("No valid sequences created.")
        
        # Split train/val (reuse DCRNN logic)
        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            node_features_val, station_ids_val = self._prepare_node_features(X_val)
            node_indices_val = self._get_station_indices(station_ids_val, self.graph['station_ids'])
            y_val_array = y_val.values.astype(np.float32)
            
            valid_mask_val = ~(np.isnan(y_val_array) | np.isnan(node_features_val).any(axis=1))
            node_features_val = node_features_val[valid_mask_val]
            y_val_array = y_val_array[valid_mask_val]
            node_indices_val = node_indices_val[valid_mask_val]
            
            node_features_val = self._x_scaler.transform(node_features_val).astype(np.float32)
            if self._y_scaler is not None:
                y_val_array = self._y_scaler.transform(y_val_array.reshape(-1, 1)).astype(np.float32).ravel()
            
            val_node_data_dict = {}
            for i, node_idx in enumerate(node_indices_val):
                if node_idx not in val_node_data_dict:
                    val_node_data_dict[node_idx] = {'X': [], 'y': []}
                val_node_data_dict[node_idx]['X'].append(node_features_val[i])
                val_node_data_dict[node_idx]['y'].append(y_val_array[i])
            
            val_sequences = []
            for node_idx, data in val_node_data_dict.items():
                X_node = np.array(data['X'])
                y_node = np.array(data['y'])
                
                if len(X_node) < self.sequence_length:
                    continue
                
                for i in range(len(X_node) - self.sequence_length + 1):
                    val_sequences.append({
                        'X': X_node[i:i + self.sequence_length],
                        'y': y_node[i + self.sequence_length - 1],
                        'node_idx': node_idx
                    })
            
            train_sequences = sequences
        else:
            train_size = int(0.8 * len(sequences))
            train_sequences = sequences[:train_size]
            val_sequences = sequences[train_size:]
        
        # Create dataset
        class SimpleGraphDataset(Dataset):
            def __init__(self, sequences):
                self.sequences = sequences
            
            def __len__(self):
                return len(self.sequences)
            
            def __getitem__(self, idx):
                seq = self.sequences[idx]
                return (
                    torch.FloatTensor(seq['X']),
                    torch.FloatTensor([seq['y']]),
                    seq['node_idx']
                )
        
        train_dataset = SimpleGraphDataset(train_sequences)
        val_dataset = SimpleGraphDataset(val_sequences)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() // 4),
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count() // 4),
            pin_memory=torch.cuda.is_available()
        )
        
        # Initialize model
        self.model = GATLSTMModel(
            num_nodes=self.num_nodes,
            input_size=self.node_feature_size,
            hidden_size=self.hidden_size,
            num_gat_layers=self.num_gat_layers,
            num_lstm_layers=self.num_lstm_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            output_size=1
        ).to(self.device)
        
        adj_matrix = torch.FloatTensor(self.graph['adj_matrix']).to(self.device)
        
        # Loss and optimizer
        if self.task_type == "classification":
            pos_count = (y_array > 0.5).sum()
            neg_count = len(y_array) - pos_count
            pos_weight = torch.tensor([neg_count / (pos_count + 1e-6)], device=self.device)
            self._pos_weight = pos_weight
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        scaler = None
        if self.use_amp and torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y, batch_nodes in train_loader:
                batch_size, seq_len, n_features = batch_X.shape
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                batch_X_graph = batch_X.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        output = self.model(batch_X_graph, adj_matrix)
                        node_outputs = output[torch.arange(batch_size), batch_nodes.long(), 0]
                        loss = criterion(node_outputs, batch_y.squeeze())
                    
                    scaler.scale(loss).backward()
                    if self.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = self.model(batch_X_graph, adj_matrix)
                    node_outputs = output[torch.arange(batch_size), batch_nodes.long(), 0]
                    loss = criterion(node_outputs, batch_y.squeeze())
                    
                    loss.backward()
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    optimizer.step()
                
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y, batch_nodes in val_loader:
                    batch_size, seq_len, n_features = batch_X.shape
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    batch_X_graph = batch_X.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
                    
                    output = self.model(batch_X_graph, adj_matrix)
                    node_outputs = output[torch.arange(batch_size), batch_nodes.long(), 0]
                    loss = criterion(node_outputs, batch_y.squeeze())
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            if self.progress_logger and epoch % 10 == 0:
                self.progress_logger.log(
                    f"Epoch {epoch}/{self.epochs} - train_loss={train_loss:.6f}, val_loss={val_loss:.6f}",
                    flush=True
                )
            
            if self.early_stopping:
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.progress_logger:
                            self.progress_logger.log(f"Early stopping at epoch {epoch}", flush=True)
                        break
        
        self.is_fitted = True
        
        # Fit calibrator
        if self.task_type == "classification" and self.config.get("model_params", {}).get("use_probability_calibration", True):
            self.model.eval()
            val_probas = []
            val_targets = []
            with torch.no_grad():
                for batch_X, batch_y, batch_nodes in val_loader:
                    batch_size, seq_len, n_features = batch_X.shape
                    batch_X = batch_X.to(self.device)
                    batch_X_graph = batch_X.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
                    
                    output = self.model(batch_X_graph, adj_matrix)
                    node_outputs = output[torch.arange(batch_size), batch_nodes.long(), 0]
                    proba = torch.sigmoid(node_outputs).cpu().numpy()
                    val_probas.extend(proba)
                    val_targets.extend(batch_y.numpy())
            
            if len(val_probas) > 0:
                self._calibrator = ProbabilityCalibrator(
                    method=self.config.get("model_params", {}).get("calibration_method", "platt")
                )
                self._calibrator.fit(np.array(val_probas), np.array(val_targets))
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        station_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Make point predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        
        node_features, station_ids_array = self._prepare_node_features(X, station_ids)
        node_indices = self._get_station_indices(station_ids_array, self.graph['station_ids'])
        node_features = self._x_scaler.transform(node_features).astype(np.float32)
        
        predictions = []
        node_data_dict = {}
        for i, node_idx in enumerate(node_indices):
            if node_idx not in node_data_dict:
                node_data_dict[node_idx] = []
            node_data_dict[node_idx].append((i, node_features[i]))
        
        adj_matrix = torch.FloatTensor(self.graph['adj_matrix']).to(self.device)
        
        with torch.no_grad():
            for node_idx, data_list in node_data_dict.items():
                data_list.sort(key=lambda x: x[0])
                node_data = np.array([d[1] for d in data_list])
                original_indices = [d[0] for d in data_list]
                
                for i, orig_idx in enumerate(original_indices):
                    start_idx = max(0, i - self.sequence_length + 1)
                    seq_data = node_data[start_idx:i+1]
                    
                    if len(seq_data) < self.sequence_length:
                        padding = np.tile(seq_data[0:1], (self.sequence_length - len(seq_data), 1))
                        seq_data = np.vstack([padding, seq_data])
                    
                    x_tensor = torch.FloatTensor(seq_data).unsqueeze(0).unsqueeze(2).expand(
                        1, self.sequence_length, self.num_nodes, -1
                    ).to(self.device)
                    
                    output = self.model(x_tensor, adj_matrix)
                    pred = output[0, node_idx, 0].cpu().item()
                    predictions.append((orig_idx, pred))
        
        predictions.sort(key=lambda x: x[0])
        predictions = np.array([p[1] for p in predictions])
        
        if self.task_type != "classification" and self._y_scaler is not None:
            predictions = self._y_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
        
        if self.task_type == "classification":
            predictions = (predictions > 0).astype(int)
        
        return predictions
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        station_ids: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Predict probabilities."""
        if self.task_type != "classification":
            return None
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        
        node_features, station_ids_array = self._prepare_node_features(X, station_ids)
        node_indices = self._get_station_indices(station_ids_array, self.graph['station_ids'])
        node_features = self._x_scaler.transform(node_features).astype(np.float32)
        
        probabilities = []
        node_data_dict = {}
        for i, node_idx in enumerate(node_indices):
            if node_idx not in node_data_dict:
                node_data_dict[node_idx] = []
            node_data_dict[node_idx].append((i, node_features[i]))
        
        adj_matrix = torch.FloatTensor(self.graph['adj_matrix']).to(self.device)
        
        with torch.no_grad():
            for node_idx, data_list in node_data_dict.items():
                data_list.sort(key=lambda x: x[0])
                node_data = np.array([d[1] for d in data_list])
                original_indices = [d[0] for d in data_list]
                
                for i, orig_idx in enumerate(original_indices):
                    start_idx = max(0, i - self.sequence_length + 1)
                    seq_data = node_data[start_idx:i+1]
                    
                    if len(seq_data) < self.sequence_length:
                        padding = np.tile(seq_data[0:1], (self.sequence_length - len(seq_data), 1))
                        seq_data = np.vstack([padding, seq_data])
                    
                    x_tensor = torch.FloatTensor(seq_data).unsqueeze(0).unsqueeze(2).expand(
                        1, self.sequence_length, self.num_nodes, -1
                    ).to(self.device)
                    
                    output = self.model(x_tensor, adj_matrix)
                    logit = output[0, node_idx, 0]
                    proba = torch.sigmoid(logit).cpu().item()
                    probabilities.append((orig_idx, proba))
        
        probabilities.sort(key=lambda x: x[0])
        probabilities = np.array([p[1] for p in probabilities])
        
        if self._calibrator is not None:
            probabilities = self._calibrator.transform(probabilities)
        
        return probabilities
    
    @classmethod
    def load(cls, path: Path) -> "GATLSTMForecastModel":
        """Load model from disk."""
        import pickle
        from src.models.utils import GraphBuilder
        
        path = Path(path)
        
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        instance = cls(metadata['config'])
        instance.graph = GraphBuilder.load_graph(path / "graph.pkl")
        instance.num_nodes = len(instance.graph['station_ids'])
        instance.node_feature_size = metadata['node_feature_size']
        instance.is_fitted = metadata['is_fitted']
        
        instance.model = GATLSTMModel(
            num_nodes=instance.num_nodes,
            input_size=instance.node_feature_size,
            hidden_size=instance.hidden_size,
            num_gat_layers=instance.num_gat_layers,
            num_lstm_layers=instance.num_lstm_layers,
            num_heads=instance.num_heads,
            dropout=instance.dropout,
            output_size=1
        ).to(instance.device)
        
        model_path = path / "model.pth"
        instance.model.load_state_dict(torch.load(model_path, map_location=instance.device))
        instance.model.eval()
        
        if (path / "x_scaler.pkl").exists():
            with open(path / "x_scaler.pkl", 'rb') as f:
                instance._x_scaler = pickle.load(f)
        if (path / "y_scaler.pkl").exists():
            with open(path / "y_scaler.pkl", 'rb') as f:
                instance._y_scaler = pickle.load(f)
        if (path / "calibrator.pkl").exists():
            with open(path / "calibrator.pkl", 'rb') as f:
                instance._calibrator = pickle.load(f)
        
        return instance

