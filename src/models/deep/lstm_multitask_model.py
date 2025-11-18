"""Multi-task LSTM model for simultaneous temperature and frost probability prediction."""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import os
import pandas as pd
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..base import BaseModel
from src.utils.losses import FocalLoss


class TimeSeriesDataset(Dataset):
    """Dataset for time series data with station grouping support."""
    
    def __init__(self, X: np.ndarray, y_temp: np.ndarray, y_frost: np.ndarray, 
                 sequence_length: int = 24, station_ids: np.ndarray = None):
        """Initialize dataset.
        
        Args:
            X: Feature array.
            y_temp: Temperature target array.
            y_frost: Frost target array (0/1).
            sequence_length: Length of input sequences.
            station_ids: Optional array of station IDs. If provided, sequences will not cross station boundaries.
        """
        self.sequence_length = sequence_length
        self.X = torch.FloatTensor(X)
        self.y_temp = torch.FloatTensor(y_temp)
        self.y_frost = torch.FloatTensor(y_frost)
        self.station_ids = station_ids
        
        # Build boundary-safe sequence indices with diagnostics
        # Build sequence index lists that never cross station boundaries
        # OPTIMIZED: Maximize data utilization by generating all valid sequences
        # Key insight: If data is grouped by station and time-ordered within each station,
        # we can generate sequences using sliding windows with gap tolerance
        # Note: max_gap is in terms of index difference (sample steps), not time difference
        # After reorganize_by_station(), indices are continuous (0,1,2,...) within each station,
        # so this mainly handles cases where there are missing samples within a station's data
        # For hourly data, max_gap=24 means allowing up to 24 missing samples between consecutive indices
        max_gap = 24  # Maximum allowed gap in indices (sample steps, not time units)
        self.sequence_indices = []
        total_samples = len(self.X)
        skipped_stations = 0
        total_gaps = 0
        total_contiguous_runs = 0
        
        if station_ids is not None:
            unique_stations = np.unique(station_ids)
            for station_id in unique_stations:
                station_idx = np.where(station_ids == station_id)[0]
                if len(station_idx) < self.sequence_length:
                    skipped_stations += 1
                    continue
                
                # OPTIMIZED: Generate sequences with gap tolerance
                # Split only on large gaps (> max_gap), allowing small gaps within sequences
                diffs = np.diff(station_idx)
                gaps = (diffs > 1).sum()
                total_gaps += gaps
                
                # Find runs separated by large gaps only
                run_start = 0
                station_runs = 0
                
                for j, d in enumerate(diffs, start=1):
                    if d > max_gap:  # Large gap: end current run
                        run = station_idx[run_start:j]
                        if len(run) >= self.sequence_length:
                            station_runs += 1
                            # Generate all possible sliding windows (step=1 for maximum coverage)
                            for k in range(0, len(run) - self.sequence_length + 1):
                                self.sequence_indices.append(run[k:k + self.sequence_length])
                        run_start = j
                
                # Process tail run
                run = station_idx[run_start:]
                if len(run) >= self.sequence_length:
                    station_runs += 1
                    for k in range(0, len(run) - self.sequence_length + 1):
                        self.sequence_indices.append(run[k:k + self.sequence_length])
                
                total_contiguous_runs += station_runs
        else:
            # No station grouping: generate all possible sequences with step=1
            # This maximizes data utilization when no station boundaries exist
            if total_samples >= self.sequence_length:
                for i in range(0, total_samples - self.sequence_length + 1):
                    self.sequence_indices.append(np.arange(i, i + self.sequence_length))
        
        # Store diagnostics for potential logging
        self._seq_diagnostics = {
            "total_samples": total_samples,
            "total_sequences": len(self.sequence_indices),
            "skipped_stations": skipped_stations if station_ids is not None else 0,
            "total_gaps": total_gaps if station_ids is not None else 0,
            "total_contiguous_runs": total_contiguous_runs if station_ids is not None else 1
        }
    
    def __len__(self):
        return len(self.sequence_indices)
    
    def __getitem__(self, idx):
        seq_idx = self.sequence_indices[idx]
        return (self.X[seq_idx], self.y_temp[seq_idx[-1]], self.y_frost[seq_idx[-1]])


class LSTMMultiTaskModel(nn.Module):
    """Multi-task LSTM neural network model.
    
    This model has two output heads:
    1. Temperature prediction (regression)
    2. Frost probability prediction (classification)
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """Initialize multi-task LSTM model.
        
        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super(LSTMMultiTaskModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Shared LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
        # Two output heads
        self.fc_temp = nn.Linear(hidden_size, 1)      # Temperature (regression)
        self.fc_frost = nn.Linear(hidden_size, 1)     # Frost classification (logit)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence, features).
        
        Returns:
            Tuple of (temperature_pred, frost_logit):
            - temperature_pred: Tensor of shape (batch,)
            - frost_logit: Tensor of shape (batch,) (raw logits; apply sigmoid at inference)
        """
        lstm_out, _ = self.lstm(x)
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        
        temp_pred = self.fc_temp(lstm_out).flatten()
        frost_logit = self.fc_frost(lstm_out).flatten()
        return temp_pred, frost_logit


class LSTMMultiTaskForecastModel(BaseModel):
    """Multi-task LSTM forecast model wrapper.
    
    This model predicts both temperature and frost probability simultaneously.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-task LSTM forecast model.
        
        Args:
            config: Configuration dictionary containing:
                - model_params: Model-specific parameters
                    - sequence_length: Length of input sequences (default: 24)
                    - hidden_size: Number of hidden units (default: 128)
                    - num_layers: Number of LSTM layers (default: 2)
                    - dropout: Dropout rate (default: 0.2)
                    - learning_rate: Learning rate (default: 0.001)
                    - batch_size: Batch size (default: 32)
                    - epochs: Number of training epochs (default: 100)
                    - loss_weight_temp: Weight for temperature loss (default: 0.5)
                    - loss_weight_frost: Weight for frost loss (default: 0.5)
                    - early_stopping: Use early stopping (default: True)
                    - patience: Early stopping patience (default: 10)
                    - lr_scheduler: Use learning rate scheduler (default: True)
                    - gradient_clip: Gradient clipping value (default: 1.0)
                    - save_best_model: Save best model during training (default: True)
        """
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM models. Install with: pip install torch")
        
        model_params = config.get("model_params", {})
        
        # Model architecture parameters
        self.sequence_length = model_params.get("sequence_length", 24)
        self.hidden_size = model_params.get("hidden_size", 128)
        self.num_layers = model_params.get("num_layers", 2)
        self.dropout = model_params.get("dropout", 0.2)
        
        # Training parameters
        self.learning_rate = model_params.get("learning_rate", 0.001)
        self.batch_size = model_params.get("batch_size", 32)
        self.epochs = model_params.get("epochs", 100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # AMP (mixed precision)
        self.use_amp = model_params.get("use_amp", False)
        # Scalers (fit() will initialize when available)
        self._x_scaler = None
        self._y_temp_scaler = None
        
        # Loss weights for multi-task learning
        self.loss_weight_temp = model_params.get("loss_weight_temp", 0.5)
        self.loss_weight_frost = model_params.get("loss_weight_frost", 0.5)
        
        # Optimization parameters
        self.use_early_stopping = model_params.get("early_stopping", True)
        self.patience = model_params.get("patience", 10)
        self.min_delta = model_params.get("min_delta", 1e-6)
        self.use_lr_scheduler = model_params.get("lr_scheduler", True)
        self.lr_scheduler_patience = model_params.get("lr_scheduler_patience", 5)
        self.lr_scheduler_factor = model_params.get("lr_scheduler_factor", 0.5)
        self.gradient_clip_value = model_params.get("gradient_clip", 1.0)
        self.save_best_model = model_params.get("save_best_model", True)
        
        # Station grouping support
        self.station_column = config.get("station_column", "Stn Id")
        
        # Model will be initialized in fit() when we know input_size
        self.model = None
        self.input_size = None
    
    def fit(self, X: pd.DataFrame, y_temp: pd.Series, y_frost: pd.Series, **kwargs) -> "LSTMMultiTaskForecastModel":
        """Train the multi-task LSTM model.
        
        Args:
            X: Feature DataFrame.
            y_temp: Temperature target Series.
            y_frost: Frost target Series (0/1).
            **kwargs: Additional arguments (e.g., station_ids).
        
        Returns:
            Self for method chaining.
        """
        # Ensure attribute exists even if no checkpointing is requested
        self.checkpoint_dir = None
        self.feature_names = list(X.columns)
        self.input_size = len(self.feature_names)
        
        # Initialize model
        self.model = LSTMMultiTaskModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Prepare data - convert to numpy arrays and free original DataFrame memory
        X_array = X.values.astype(np.float32)
        y_temp_array = y_temp.values.astype(np.float32)
        y_frost_array = y_frost.values.astype(np.float32)
        
        # Free original DataFrame memory immediately
        del X, y_temp, y_frost
        import gc
        gc.collect()
        
        # Get station IDs if available
        station_ids = kwargs.get('station_ids', None)
        
        # Check for NaN values and handle them properly
        # Note: Data should already be cleaned in preprocessing (forward_fill, etc.)
        # and in prepare_features_and_targets (also forward_fill). If NaN still exists,
        # we use forward/backward fill to preserve time continuity instead of deleting rows.
        
        # First, remove rows with NaN targets (targets must be valid - cannot predict without target)
        valid_mask = ~(np.isnan(y_temp_array) | np.isnan(y_frost_array))
        if not valid_mask.all():
            if self.progress_logger:
                self.progress_logger.log(f"  ⚠️  Warning: Found NaN values in targets, removing rows", flush=True, detailed=True)
            X_array = X_array[valid_mask]
            y_temp_array = y_temp_array[valid_mask]
            y_frost_array = y_frost_array[valid_mask]
            if station_ids is not None:
                station_ids = np.asarray(station_ids)[valid_mask]
            if self.progress_logger:
                self.progress_logger.log(f"  Removed {np.sum(~valid_mask)} rows with NaN targets, remaining: {len(y_temp_array)}", flush=True, detailed=True)
            del valid_mask
            gc.collect()
        
        # Handle NaN in features using forward/backward fill to preserve time continuity
        # This is important for time series data (hourly collected) - deleting rows breaks continuity
        if np.isnan(X_array).any():
            n_nan_before = np.isnan(X_array).sum()
            # Convert to DataFrame for easier forward/backward fill
            import pandas as pd
            X_df = pd.DataFrame(X_array)
            
            # Forward fill within each station to preserve time continuity
            if station_ids is not None:
                station_ids_array = np.asarray(station_ids)
                X_df = X_df.groupby(station_ids_array).ffill()
            else:
                X_df = X_df.ffill()
            
            # If still NaN (e.g., at sequence start), use backward fill
            if X_df.isna().any().any():
                if station_ids is not None:
                    X_df = X_df.groupby(station_ids_array).bfill()
                else:
                    X_df = X_df.bfill()
            
            X_array = X_df.values.astype(np.float32)
            n_nan_after = np.isnan(X_array).sum()
            
            if n_nan_before > 0:
                if self.progress_logger:
                    self.progress_logger.log(f"  ⚠️  Warning: Found {n_nan_before} NaN values in features, filled using forward/backward fill", flush=True, detailed=True)
                if n_nan_after > 0:
                    if self.progress_logger:
                        self.progress_logger.log(f"     {n_nan_after} NaN values remain (likely at sequence boundaries), will be removed", flush=True, detailed=True)
                    # Only remove rows if NaN still exists after filling
                    nan_rows = np.isnan(X_array).any(axis=1)
                    if nan_rows.any():
                        valid_mask = ~nan_rows
                        X_array = X_array[valid_mask]
                        y_temp_array = y_temp_array[valid_mask]
                        y_frost_array = y_frost_array[valid_mask]
                        if station_ids is not None:
                            station_ids = np.asarray(station_ids)[valid_mask]
                        if self.progress_logger:
                            self.progress_logger.log(f"  Removed {nan_rows.sum()} rows with remaining NaN, final samples: {len(y_temp_array)}", flush=True, detailed=True)
                        del nan_rows, valid_mask
                        gc.collect()
            
            del X_df
            gc.collect()
        
        # Check for infinite values
        if np.isinf(X_array).any():
            print(f"  ⚠️  Warning: Found infinite values in features, clipping to finite range")
            X_array = np.clip(X_array, -1e6, 1e6)
        
        if np.isinf(y_temp_array).any() or np.isinf(y_frost_array).any():
            print(f"  ⚠️  Warning: Found infinite values in targets, clipping to finite range")
            y_temp_array = np.clip(y_temp_array, -1e6, 1e6)
            # Frost is a binary label (0/1); ensure bounds [0, 1]
            y_frost_array = np.clip(y_frost_array, 0.0, 1.0)
        if station_ids is not None:
            station_ids = np.asarray(station_ids)
            if len(station_ids) != len(X_array):
                print(f"  ⚠️  Warning: station_ids length ({len(station_ids)}) doesn't match X length ({len(X_array)}). Ignoring station_ids.")
                station_ids = None
        
        # Create datasets with station grouping support
        effective_seq_len = self.sequence_length
        # Standardize features and temperature target (keep frost label untouched)
        try:
            from sklearn.preprocessing import StandardScaler  # type: ignore
        except Exception:
            StandardScaler = None  # type: ignore
        if StandardScaler is not None:
            self._x_scaler = StandardScaler()
            X_array = self._x_scaler.fit_transform(X_array).astype(np.float32)
            self._y_temp_scaler = StandardScaler()
            y_temp_array = self._y_temp_scaler.fit_transform(y_temp_array.reshape(-1, 1)).astype(np.float32).ravel()
        else:
            # Fallback: simple mean/std scaling
            x_mean = X_array.mean(axis=0, keepdims=True)
            x_std = X_array.std(axis=0, keepdims=True) + 1e-6
            self._x_scaler = {"mean": x_mean.astype(np.float32), "std": x_std.astype(np.float32)}
            X_array = ((X_array - x_mean) / x_std).astype(np.float32)
            y_mean = float(y_temp_array.mean()); y_std = float(y_temp_array.std() + 1e-6)
            self._y_temp_scaler = {"mean": y_mean, "std": y_std}
            y_temp_array = ((y_temp_array - y_mean) / y_std).astype(np.float32)
        
        dataset = TimeSeriesDataset(X_array, y_temp_array, y_frost_array, 
                                   effective_seq_len, station_ids=station_ids)
        # Fallback if windowing yields no samples
        if len(dataset) == 0 and effective_seq_len > 24:
            if self.progress_logger:
                self.progress_logger.log(f"  ⚠️  Sequence dataset empty with length={effective_seq_len}, falling back to 24", flush=True, detailed=True)
            effective_seq_len = 24
            dataset = TimeSeriesDataset(X_array, y_temp_array, y_frost_array, effective_seq_len, station_ids=station_ids)
        if len(dataset) == 0 and effective_seq_len > 12:
            if self.progress_logger:
                self.progress_logger.log(f"  ⚠️  Sequence dataset still empty, falling back to 12", flush=True, detailed=True)
            effective_seq_len = 12
            dataset = TimeSeriesDataset(X_array, y_temp_array, y_frost_array, effective_seq_len, station_ids=station_ids)
        if len(dataset) == 0:
            raise ValueError("After sequence construction, dataset is empty. Please check data continuity and NaN handling.")
        
        # Free numpy arrays after dataset creation (dataset will keep its own copies)
        del X_array, y_temp_array, y_frost_array
        gc.collect()
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        n_samples = len(dataset)
        train_size = int(0.8 * n_samples)
        val_size = n_samples - train_size
        if train_size == 0 and n_samples >= 2:
            train_size = 1
            val_size = n_samples - train_size
        if val_size == 0 and n_samples >= 2:
            val_size = 1
            train_size = n_samples - val_size
        # Time-ordered split to avoid temporal leakage (use sequence order)
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        # Log dataset lengths and diagnostics for debugging
        if self.progress_logger:
            diag = dataset._seq_diagnostics
            self.progress_logger.log(
                f"  Dataset(seq_len={effective_seq_len}): total_sequences={n_samples}, train_sequences={len(train_dataset)}, val_sequences={len(val_dataset)}",
                flush=True,
                detailed=True,
            )
            if station_ids is not None and diag.get("total_gaps", 0) > 0:
                self.progress_logger.log(
                    f"  Sequence diagnostics: {diag['total_samples']:,} samples → {diag['total_sequences']:,} sequences "
                    f"(skipped {diag['skipped_stations']} stations, {diag['total_gaps']} gaps, {diag['total_contiguous_runs']} contiguous runs)",
                    flush=True,
                    detailed=True,
                )
        
        # Optimize DataLoader for GPU: use multiple workers and pin memory
        # Adaptive optimization based on dataset size and system resources
        cpu_count = os.cpu_count() or 1
        # For small datasets (<10K sequences), use smaller batch size to increase gradient updates
        # For large datasets, use configured batch size
        if n_samples < 10000:
            # Small dataset: reduce batch size to get more batches per epoch
            adaptive_batch_size = min(32, max(8, n_samples // 50))  # Target ~50 batches per epoch
        else:
            adaptive_batch_size = self.batch_size
        
        # Optimize num_workers based on CPU cores and dataset size
        # More workers for larger datasets, but cap at reasonable limit
        if n_samples < 5000:
            num_workers = min(4, cpu_count // 4)  # Small dataset: fewer workers
        elif n_samples < 50000:
            num_workers = min(12, cpu_count // 2)  # Medium dataset: moderate workers
        else:
            num_workers = min(16, cpu_count // 2)  # Large dataset: more workers
        
        pin_memory = torch.cuda.is_available()  # Pin memory if GPU available
        prefetch_factor = 2 if num_workers > 0 else None  # Prefetch 2 batches per worker
        
        # Adjust batch/drop_last for small datasets
        effective_batch_size = min(adaptive_batch_size, max(1, n_samples))
        train_drop_last = n_samples >= adaptive_batch_size
        # Optional weighted sampler for imbalanced classification (use end-of-sequence frost label)
        sampler = None
        if self.config.get("model_params", {}).get("use_weighted_sampler", True):
            frost_np_all = dataset.y_frost.cpu().numpy()
            seq_labels_all = np.array([float(frost_np_all[idx[-1]]) for idx in dataset.sequence_indices], dtype=np.float32)
            # For extremely imbalanced data, use more aggressive sampling weights
            pos_count = max(1.0, float((seq_labels_all > 0.5).sum()))
            neg_count = max(1.0, float(len(seq_labels_all) - (seq_labels_all > 0.5).sum()))
            imbalance_ratio_sampler = neg_count / pos_count
            
            # Use same strategy as pos_weight calculation
            if imbalance_ratio_sampler > 100:  # <1% positive
                pos_w_sampler = np.sqrt(imbalance_ratio_sampler)
            elif imbalance_ratio_sampler > 50:  # <2% positive
                pos_w_sampler = np.sqrt(imbalance_ratio_sampler) * 1.5
            else:
                pos_w_sampler = imbalance_ratio_sampler
            
            # For very imbalanced training set, further boost sampling weight
            train_seq_labels = seq_labels_all[train_indices]
            train_pos_ratio = (train_seq_labels > 0.5).sum() / len(train_seq_labels)
            if train_pos_ratio < 0.01:  # <1% positive in training set
                pos_w_sampler = pos_w_sampler * 2.5  # More aggressive sampling
            
            # Option: Use class-balanced batch sampling (ensure each batch has positive samples)
            use_class_balanced_batch = self.config.get("model_params", {}).get("use_class_balanced_batch", False)
            if use_class_balanced_batch and train_pos_ratio < 0.05:  # Only for very imbalanced data
                # Further boost positive sample weight
                pos_w_sampler = pos_w_sampler * 3.0
                if self.progress_logger:
                    self.progress_logger.log(
                        f"  ✅ Using class-balanced batch sampling (boosted weight={pos_w_sampler:.2f})",
                        flush=True
                    )
            
            weights_all = np.where(seq_labels_all > 0.5, pos_w_sampler, 1.0).astype(np.float32)
            train_weights = torch.from_numpy(weights_all[train_indices]).double()
            sampler = torch.utils.data.WeightedRandomSampler(weights=train_weights, num_samples=len(train_indices), replacement=True)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=effective_batch_size, 
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,  # Keep workers alive between epochs
            prefetch_factor=prefetch_factor,  # Limit prefetch to reduce memory
            drop_last=train_drop_last  # Avoid empty epoch when data is small
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=effective_batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor,
            drop_last=False  # Keep all validation data
        )
        # Log loader lengths to ensure non-zero batches
        if self.progress_logger:
            self.progress_logger.log(
                f"  DataLoader: train_batches≈{len(train_loader)}, val_batches≈{len(val_loader)}, batch_size={effective_batch_size}, drop_last={train_drop_last}",
                flush=True,
                detailed=True,
            )
        
        # Training setup
        criterion_temp = nn.MSELoss()
        # Compute pos_weight on sequence-end labels for class imbalance
        frost_np = dataset.y_frost.numpy()
        y_seq_last = np.array([float(frost_np[idx[-1]]) for idx in dataset.sequence_indices], dtype=np.float32)
        pos = max(1.0, float((y_seq_last > 0.5).sum()))
        neg = max(1.0, float(len(y_seq_last) - (y_seq_last > 0.5).sum()))
        imbalance_ratio = neg / pos
        
        # For extremely imbalanced data (<1% positive), use more aggressive weighting
        if imbalance_ratio > 100:  # <1% positive
            pos_weight_val = np.sqrt(imbalance_ratio)
            weight_strategy = "sqrt (very imbalanced)"
        elif imbalance_ratio > 50:  # <2% positive
            pos_weight_val = np.sqrt(imbalance_ratio) * 1.5
            weight_strategy = "sqrt*1.5 (moderately imbalanced)"
        else:
            pos_weight_val = imbalance_ratio
            weight_strategy = "linear"
        
        # Choose loss function based on configuration
        use_focal_loss = self.config.get("model_params", {}).get("use_focal_loss", False)
        
        if use_focal_loss:
            # Focal Loss: better for extremely imbalanced data
            focal_alpha = self.config.get("model_params", {}).get("focal_alpha", 0.25)
            focal_gamma = self.config.get("model_params", {}).get("focal_gamma", 2.0)
            criterion_frost = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            if self.progress_logger:
                self.progress_logger.log(
                    f"  ✅ Class imbalance: {pos:.0f} pos / {neg:.0f} neg (ratio={imbalance_ratio:.1f}:1, <{100/imbalance_ratio:.2f}% pos)",
                    flush=True
                )
                self.progress_logger.log(
                    f"  ✅ Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma}) for imbalanced classification",
                    flush=True
                )
        else:
            # BCEWithLogitsLoss with pos_weight
            pos_weight = torch.tensor([pos_weight_val], device=self.device, dtype=torch.float32)
            criterion_frost = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            if self.progress_logger:
                self.progress_logger.log(
                    f"  ✅ Class imbalance: {pos:.0f} pos / {neg:.0f} neg (ratio={imbalance_ratio:.1f}:1, <{100/imbalance_ratio:.2f}% pos)",
                    flush=True
                )
                self.progress_logger.log(
                    f"  ✅ Using pos_weight={pos_weight_val:.2f} ({weight_strategy}) for BCEWithLogitsLoss",
                    flush=True
                )
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = None
        if self.use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                min_lr=1e-6
            )
        
        # AMP scaler
        use_amp = self.use_amp and torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        if use_amp and self.progress_logger:
            self.progress_logger.log("  ✅ Using Mixed Precision Training (AMP) for faster training", flush=True)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training history for logging
        training_history = {
            'epoch': [],
            'train_loss_total': [],
            'train_loss_temp': [],
            'train_loss_frost': [],
            'val_loss_total': [],
            'val_loss_temp': [],
            'val_loss_frost': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Setup training utilities if not already set up
        checkpoint_dir = kwargs.get('checkpoint_dir', None)
        checkpoint_frequency = getattr(self, "checkpoint_frequency", None)
        if checkpoint_frequency is None:
            checkpoint_frequency = self.config.get("model_params", {}).get("checkpoint_frequency", 10)
        if checkpoint_dir and not self.checkpoint_manager:
            # Setup training tools if checkpoint_dir provided
            self.setup_training_tools(
                checkpoint_dir=checkpoint_dir,
                checkpoint_frequency=checkpoint_frequency,
                save_best=self.save_best_model,
                best_metric="val_loss_total",
                best_mode="min"
            )
            self.checkpoint_dir = Path(checkpoint_dir)  # For backward compatibility
        
        # Initialize training history if not already set up
        if not self.training_history:
            from src.models.utils import TrainingHistory
            # Multi-task model needs additional metrics
            self.training_history = TrainingHistory(metrics=[
                'train_loss_total', 'train_loss_temp', 'train_loss_frost',
                'val_loss_total', 'val_loss_temp', 'val_loss_frost',
                'learning_rate', 'epoch_time'
            ])
        
        # Initialize progress logger if not already set up
        if not self.progress_logger:
            from src.models.utils import ProgressLogger
            self.progress_logger = ProgressLogger()
        
        # Start training history tracking
        self.training_history.start_training()
        
        # Log training start
        self.progress_logger.log_training_start(
            model_name="LSTM Multi-task",
            device=str(self.device),
            config={
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "sequence_length": self.sequence_length
            }
        )
        
        # Training loop - use unified progress logger
        import time
        import sys
        epoch_start_time = time.time()
        
        # Print initial training info
        print(f"\n  🚀 Starting LSTM Multi-Task training")
        print(f"     Device: {self.device}")
        print(f"     Input size: {self.input_size}")
        print(f"     Hidden size: {self.hidden_size}")
        batch_info = f"{effective_batch_size}" if effective_batch_size == self.batch_size else f"{effective_batch_size} (adaptive, config: {self.batch_size})"
        print(f"     Batch size: {batch_info}")
        print(f"     Num workers: {num_workers}")
        print(f"     Epochs: {self.epochs}")
        print(f"     Sequence length: {effective_seq_len}")
        print(f"     Total sequences: {n_samples:,}")
        print(f"     Train sequences: {len(train_dataset):,} ({len(train_loader):,} batches)")
        print(f"     Val sequences: {len(val_dataset):,} ({len(val_loader):,} batches)")
        if hasattr(self, '_pos_weight') or use_focal_loss:
            if use_focal_loss:
                print(f"     Loss: Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
            else:
                print(f"     Frost pos_weight: {pos_weight_val:.2f} ({weight_strategy})")
        print(f"     Loss weights: temp={self.loss_weight_temp}, frost={self.loss_weight_frost}")
        sys.stdout.flush()
        
        self.model.train()
        # Use unified progress logger for epoch progress bar
        if self.progress_logger:
            epoch_pbar = self.progress_logger.get_tqdm(
                range(self.epochs),
                desc="Training",
                unit="epoch"
            )
        else:
            epoch_pbar = range(self.epochs)
        
        for epoch in epoch_pbar:
            # Ensure training mode at the start of every epoch (in case eval() was set during validation)
            self.model.train()
            epoch_iter_start = time.time()
            # Training phase with progress bar
            train_loss_total = 0.0
            train_loss_temp = 0.0
            train_loss_frost = 0.0
            train_batches = 0
            
            # Use unified progress logger for tqdm (automatically handles file output)
            if self.progress_logger:
                train_pbar = self.progress_logger.get_tqdm(
                    train_loader, 
                    desc=f"Epoch {epoch+1}/{self.epochs} [Train]",
                    unit="batch"
                )
            else:
                train_pbar = train_loader
            
            for batch_X, batch_y_temp, batch_y_frost in train_pbar:
                # Ensure training mode per-batch
                self.model.train()
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y_temp = batch_y_temp.to(self.device, non_blocking=True)
                batch_y_frost = batch_y_frost.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        temp_pred, frost_logit = self.model(batch_X)
                        loss_temp = criterion_temp(temp_pred, batch_y_temp)
                        frost_logit = frost_logit.view(-1)
                        batch_y_frost = batch_y_frost.view(-1)
                        loss_frost = criterion_frost(frost_logit, batch_y_frost)
                        loss_total = self.loss_weight_temp * loss_temp + self.loss_weight_frost * loss_frost
                    scaler.scale(loss_total).backward()
                    if self.gradient_clip_value is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    temp_pred, frost_logit = self.model(batch_X)
                    loss_temp = criterion_temp(temp_pred, batch_y_temp)
                    frost_logit = frost_logit.view(-1)
                    batch_y_frost = batch_y_frost.view(-1)
                    loss_frost = criterion_frost(frost_logit, batch_y_frost)
                    loss_total = self.loss_weight_temp * loss_temp + self.loss_weight_frost * loss_frost
                    loss_total.backward()
                    if self.gradient_clip_value is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                    optimizer.step()
                
                # Check for NaN loss (before deleting)
                loss_total_value = loss_total.item()
                loss_temp_value = loss_temp.item()
                loss_frost_value = loss_frost.item()
                
                # Memory optimization: delete batch data immediately after use
                del batch_X, batch_y_temp, batch_y_frost, temp_pred, frost_logit, loss_temp, loss_frost, loss_total
                
                if np.isnan(loss_total_value) or np.isinf(loss_total_value):
                    print(f"\n  ⚠️  Warning: Invalid loss value (total={loss_total_value}, temp={loss_temp_value}, frost={loss_frost_value}) at batch {train_batches}")
                    print(f"     This may indicate numerical instability. Consider:")
                    print(f"     - Reducing learning rate")
                    print(f"     - Checking for NaN/inf in input data")
                    print(f"     - Using gradient clipping")
                    # Skip this batch's loss
                    continue
                
                train_loss_total += loss_total_value
                train_loss_temp += loss_temp_value
                train_loss_frost += loss_frost_value
                train_batches += 1
                
                # No need to update progress bar manually when tqdm is disabled for file output
            
            # If no batches processed, warn and break to avoid spinning through epochs
            if train_batches == 0:
                if self.progress_logger:
                    self.progress_logger.log(
                        "  ⚠️  Warning: No training batches processed this epoch. Check DataLoader/dataset sizes.",
                        flush=True,
                        detailed=True,
                    )
                break
            avg_train_loss_total = train_loss_total / train_batches
            avg_train_loss_temp = train_loss_temp / train_batches
            avg_train_loss_frost = train_loss_frost / train_batches
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
                torch.cuda.empty_cache()
            
            # Validation phase with progress bar
            self.model.eval()
            val_loss_total = 0.0
            val_loss_temp = 0.0
            val_loss_frost = 0.0
            val_batches = 0
            
            # Use unified progress logger for tqdm (automatically handles file output)
            if self.progress_logger:
                val_pbar = self.progress_logger.get_tqdm(
                    val_loader,
                    desc=f"Epoch {epoch+1}/{self.epochs} [Val]",
                    unit="batch"
                )
            else:
                val_pbar = val_loader
            
            with torch.no_grad():
                for batch_X, batch_y_temp, batch_y_frost in val_pbar:
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y_temp = batch_y_temp.to(self.device, non_blocking=True)
                    batch_y_frost = batch_y_frost.to(self.device, non_blocking=True)
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            temp_pred, frost_logit = self.model(batch_X)
                            loss_temp = criterion_temp(temp_pred, batch_y_temp)
                            frost_logit = frost_logit.view(-1)
                            batch_y_frost = batch_y_frost.view(-1)
                            loss_frost = criterion_frost(frost_logit, batch_y_frost)
                            loss_total = self.loss_weight_temp * loss_temp + self.loss_weight_frost * loss_frost
                    else:
                        temp_pred, frost_logit = self.model(batch_X)
                        loss_temp = criterion_temp(temp_pred, batch_y_temp)
                        frost_logit = frost_logit.view(-1)
                        batch_y_frost = batch_y_frost.view(-1)
                        loss_frost = criterion_frost(frost_logit, batch_y_frost)
                        loss_total = self.loss_weight_temp * loss_temp + self.loss_weight_frost * loss_frost
                    val_loss_total += loss_total.item()
                    val_loss_temp += loss_temp.item()
                    val_loss_frost += loss_frost.item()
                    val_batches += 1
                    # Memory optimization: delete batch data immediately
                    del batch_X, batch_y_temp, batch_y_frost, temp_pred, frost_logit, loss_temp, loss_frost, loss_total
                    
                    # No need to update progress bar manually when tqdm is disabled for file output
            
            avg_val_loss_total = val_loss_total / val_batches
            avg_val_loss_temp = val_loss_temp / val_batches
            avg_val_loss_frost = val_loss_frost / val_batches
            
            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step(avg_val_loss_total)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_iter_start
            elapsed_time = time.time() - epoch_start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = self.epochs - (epoch + 1)
            estimated_remaining = avg_epoch_time * remaining_epochs
            
            # Record epoch in training history
            self.training_history.record_epoch(
                epoch=epoch + 1,
                train_loss_total=avg_train_loss_total,
                train_loss_temp=avg_train_loss_temp,
                train_loss_frost=avg_train_loss_frost,
                val_loss_total=avg_val_loss_total,
                val_loss_temp=avg_val_loss_temp,
                val_loss_frost=avg_val_loss_frost,
                learning_rate=current_lr,
                epoch_time=epoch_time
            )
            
            # Save checkpoint periodically
            if self.checkpoint_manager and self.checkpoint_manager.should_save_checkpoint(epoch + 1):
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch + 1,
                    model_state=self.model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler else None,
                    scaler_state=None,  # Multi-task model doesn't use scaler yet
                    metrics={
                        'train_loss_total': avg_train_loss_total,
                        'val_loss_total': avg_val_loss_total
                    },
                    training_history=self.training_history
                )
                self.progress_logger.log(f"  💾 Checkpoint saved: epoch {epoch + 1}", flush=True)
            
            # Early stopping logic
            if self.use_early_stopping:
                improved = False
                if avg_val_loss_total < best_val_loss - self.min_delta:
                    best_val_loss = avg_val_loss_total
                    patience_counter = 0
                    improved = True
                    if self.save_best_model:
                        best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # Update epoch progress bar if using tqdm
                if hasattr(epoch_pbar, 'set_postfix'):
                    epoch_pbar.set_postfix({
                        'train': f'{avg_train_loss_total:.6f}',
                        'val': f'{avg_val_loss_total:.6f}',
                        'temp': f'{avg_val_loss_temp:.6f}',
                        'frost': f'{avg_val_loss_frost:.6f}',
                        'lr': f'{current_lr:.6f}',
                        'patience': f'{patience_counter}/{self.patience}',
                        'ETA': f'{estimated_remaining/60:.1f}m'
                    })
                    if improved:
                        epoch_pbar.write(f"  ✅ Improved! Val Loss: {avg_val_loss_total:.6f} (Best: {best_val_loss:.6f})")
                
                # Early stopping
                if patience_counter >= self.patience:
                    if self.progress_logger:
                        self.progress_logger.log_early_stopping(epoch + 1, self.patience)
                    if self.save_best_model and best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
            else:
                # Update epoch progress bar if using tqdm
                if hasattr(epoch_pbar, 'set_postfix'):
                    epoch_pbar.set_postfix({
                        'train': f'{avg_train_loss_total:.6f}',
                        'val': f'{avg_val_loss_total:.6f}',
                        'temp': f'{avg_val_loss_temp:.6f}',
                        'frost': f'{avg_val_loss_frost:.6f}',
                        'lr': f'{current_lr:.6f}',
                        'ETA': f'{estimated_remaining/60:.1f}m'
                    })
        
        # Save training artifacts using unified utilities
        training_time = time.time() - epoch_start_time
        if self.checkpoint_dir and self.training_history and len(self.training_history) > 0:
            self.save_training_artifacts(self.checkpoint_dir)
        self.progress_logger.log_training_complete(training_time, len(self.training_history))
        
        # PROBABILITY CALIBRATION: Fit calibrator on validation set for frost probabilities
        # This improves Brier Score and ECE while maintaining discrimination (ROC-AUC, PR-AUC)
        use_calibration = self.config.get("model_params", {}).get("use_probability_calibration", True)
        calibration_method = self.config.get("model_params", {}).get("calibration_method", "platt")
        
        self._calibrator = None
        if use_calibration:
            try:
                from src.utils.calibration import ProbabilityCalibrator
                self.model.eval()
                y_true_frost_list: list[float] = []
                y_proba_frost_list: list[float] = []
                
                with torch.no_grad():
                    for batch_X, batch_y_temp, batch_y_frost in val_loader:
                        batch_X = batch_X.to(self.device, non_blocking=True)
                        _, frost_logit = self.model(batch_X)
                        proba = torch.sigmoid(frost_logit).detach().cpu().numpy()
                        y_proba_frost_list.extend(proba.reshape(-1).tolist())
                        y_true_frost_list.extend(batch_y_frost.cpu().numpy().reshape(-1).tolist())
                
                y_true_frost_np = np.array(y_true_frost_list, dtype=np.float32)
                y_proba_frost_np = np.array(y_proba_frost_list, dtype=np.float32)
                
                # Fit calibrator
                self._calibrator = ProbabilityCalibrator(method=calibration_method)
                self._calibrator.fit(y_proba_frost_np, y_true_frost_np)
                
                if self.progress_logger:
                    try:
                        from sklearn.metrics import brier_score_loss
                        from sklearn.calibration import calibration_curve
                        brier_before = brier_score_loss(y_true_frost_np, y_proba_frost_np)
                        y_proba_calibrated = self._calibrator.transform(y_proba_frost_np)
                        brier_after = brier_score_loss(y_true_frost_np, y_proba_calibrated)
                        
                        fraction_of_positives, mean_predicted_value = calibration_curve(
                            y_true_frost_np, y_proba_frost_np, n_bins=10, strategy='uniform'
                        )
                        ece_before = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                        
                        fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
                            y_true_frost_np, y_proba_calibrated, n_bins=10, strategy='uniform'
                        )
                        ece_after = np.mean(np.abs(fraction_of_positives_cal - mean_predicted_value_cal))
                        
                        self.progress_logger.log(
                            f"  ✅ Probability calibration ({calibration_method}): "
                            f"Brier {brier_before:.6f}→{brier_after:.6f} ({((brier_after-brier_before)/brier_before*100):+.1f}%), "
                            f"ECE {ece_before:.6f}→{ece_after:.6f} ({((ece_after-ece_before)/ece_before*100):+.1f}%)",
                            flush=True,
                            detailed=True
                        )
                    except Exception:
                        self.progress_logger.log(
                            f"  ✅ Probability calibration ({calibration_method}) applied",
                            flush=True,
                            detailed=True
                        )
            except Exception as e:
                if self.progress_logger:
                    self.progress_logger.log(
                        f"  ⚠️  Warning: Probability calibration failed: {e}. Using uncalibrated probabilities.",
                        flush=True,
                        detailed=True
                    )
                self._calibrator = None
        
        # Final memory cleanup
        del train_loader, val_loader, train_dataset, val_dataset, dataset
        if 'station_ids' in locals():
            del station_ids
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_fitted = True
        return self
    
    def predict_temp(self, X: pd.DataFrame, station_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict temperature.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Temperature predictions array.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        X_array = X.values.astype(np.float32)
        # Apply stored feature scaler
        if hasattr(self, "_x_scaler") and self._x_scaler is not None:
            try:
                X_array = self._x_scaler.transform(X_array).astype(np.float32)
            except AttributeError:
                X_array = ((X_array - self._x_scaler["mean"]) / self._x_scaler["std"]).astype(np.float32)
        
        # Build sequences by station if provided
        seqs = []
        if station_ids is not None:
            station_ids = np.asarray(station_ids)
            unique = np.unique(station_ids)
            for sid in unique:
                idx = np.where(station_ids == sid)[0]
                if len(idx) < self.sequence_length:
                    continue
                diffs = np.diff(idx)
                run_start = 0
                for j, d in enumerate(diffs, start=1):
                    if d != 1:
                        run = idx[run_start:j]
                        if len(run) >= self.sequence_length:
                            for k in range(0, len(run) - self.sequence_length + 1):
                                seqs.append(run[k:k + self.sequence_length])
                        run_start = j
                run = idx[run_start:]
                if len(run) >= self.sequence_length:
                    for k in range(0, len(run) - self.sequence_length + 1):
                        seqs.append(run[k:k + self.sequence_length])
        else:
            total = len(X_array)
            if total >= self.sequence_length:
                for i in range(0, total - self.sequence_length + 1):
                    seqs.append(np.arange(i, i + self.sequence_length))
        
        pred_series = np.full((len(X_array),), np.nan, dtype=np.float32)
        with torch.no_grad():
            for idx in seqs:
                sequence = X_array[idx]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                temp_pred, _ = self.model(sequence_tensor)
                pred_series[idx[-1]] = float(temp_pred.cpu().numpy())
        # Inverse-scale temperature to original units if scaler present
        if hasattr(self, "_y_temp_scaler") and self._y_temp_scaler is not None:
            try:
                pred_series = self._y_temp_scaler.inverse_transform(pred_series.reshape(-1, 1)).ravel().astype(np.float32)  # type: ignore
            except AttributeError:
                mean = self._y_temp_scaler["mean"]; std = self._y_temp_scaler["std"]
                pred_series = (pred_series * std + mean).astype(np.float32)
        # Fill missing by forward/back fill
        last = None
        for i in range(len(pred_series)):
            if not np.isnan(pred_series[i]):
                last = pred_series[i]
            elif last is not None:
                pred_series[i] = last
        if np.isnan(pred_series[0]):
            fv = np.flatnonzero(~np.isnan(pred_series))
            if fv.size > 0:
                pred_series[:fv[0]] = pred_series[fv[0]]
            else:
                pred_series[:] = 0.0
        return pred_series
    
    def predict_frost_proba(self, X: pd.DataFrame, station_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict frost probability."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        self.model.eval()
        X_array = X.values.astype(np.float32)
        if getattr(self, "_x_scaler", None) is not None:
            try:
                X_array = self._x_scaler.transform(X_array).astype(np.float32)
            except AttributeError:
                X_array = ((X_array - self._x_scaler["mean"]) / self._x_scaler["std"]).astype(np.float32)
        # Build sequences by station if provided (same as before)
        seqs = []
        if station_ids is not None:
            station_ids = np.asarray(station_ids)
            unique = np.unique(station_ids)
            for sid in unique:
                idx = np.where(station_ids == sid)[0]
                if len(idx) < self.sequence_length:
                    continue
                diffs = np.diff(idx)
                run_start = 0
                for j, d in enumerate(diffs, start=1):
                    if d != 1:
                        run = idx[run_start:j]
                        if len(run) >= self.sequence_length:
                            for k in range(0, len(run) - self.sequence_length + 1):
                                seqs.append(run[k:k + self.sequence_length])
                        run_start = j
                run = idx[run_start:]
                if len(run) >= self.sequence_length:
                    for k in range(0, len(run) - self.sequence_length + 1):
                        seqs.append(run[k:k + self.sequence_length])
        else:
            total = len(X_array)
            if total >= self.sequence_length:
                for i in range(0, total - self.sequence_length + 1):
                    seqs.append(np.arange(i, i + self.sequence_length))
        proba_series = np.full((len(X_array),), np.nan, dtype=np.float32)
        with torch.no_grad():
            for idx in range(len(seqs)):
                seq_idx = seqs[idx]
                sequence = X_array[seq_idx]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                _, frost_logit = self.model(sequence_tensor)
                prob = torch.sigmoid(frost_logit).detach().cpu().numpy().reshape(-1)[0]
                proba_series[seq_idx[-1]] = float(prob)
        # forward/back fill
        last = None
        for i in range(len(proba_series)):
            if not np.isnan(proba_series[i]):
                last = proba_series[i]
            elif last is not None:
                proba_series[i] = last
        if np.isnan(proba_series[0]):
            fv = np.flatnonzero(~np.isnan(proba_series))
            if fv.size > 0:
                proba_series[:fv[0]] = proba_series[fv[0]]
            else:
                proba_series[:] = 0.0
        
        # Apply probability calibration if available (improves Brier Score and ECE)
        if hasattr(self, "_calibrator") and self._calibrator is not None and self._calibrator.is_fitted:
            proba_series = self._calibrator.transform(proba_series)
        
        return proba_series
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions (returns temperature for compatibility).
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Temperature predictions array.
        """
        return self.predict_temp(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict frost probability.
        
        Args:
            X: Feature DataFrame.
        
        Returns:
            Frost probability predictions array.
        """
        return self.predict_frost_proba(X)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Multi-task LSTM doesn't provide direct feature importance.
        
        Returns:
            None (LSTM uses different importance metrics).
        """
        return None
    
    def save(self, path: Path) -> None:
        """Save model to disk.
        
        Args:
            path: Directory path to save model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        model_path = path / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        
        config_path = path / "config.json"
        config_to_save = {
            "feature_names": self.feature_names,
            "input_size": self.input_size,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "model_params": self.config.get("model_params", {}),
        }
        
        with open(config_path, 'w') as f:
            import json
            json.dump(config_to_save, f, indent=2)
        
        # Save scalers if available
        try:
            import pickle  # noqa: F401
            if getattr(self, "_x_scaler", None) is not None:
                with open(path / "x_scaler.pkl", "wb") as f:
                    import pickle
                    pickle.dump(self._x_scaler, f)
            if getattr(self, "_y_temp_scaler", None) is not None:
                with open(path / "y_temp_scaler.pkl", "wb") as f:
                    import pickle
                    pickle.dump(self._y_temp_scaler, f)
        except Exception:
            pass
        
        # Save probability calibrator if available
        if hasattr(self, "_calibrator") and self._calibrator is not None and self._calibrator.is_fitted:
            try:
                import pickle
                with open(path / "calibrator.pkl", "wb") as f:
                    pickle.dump(self._calibrator, f)
            except Exception as e:
                print(f"  ⚠️  Warning: Could not save calibrator: {e}")
        
        print(f"  ✅ Multi-task LSTM model saved to {path}")
    
    def load(self, path: Path) -> "LSTMMultiTaskForecastModel":
        """Load model from disk.
        
        Args:
            path: Directory path to load model from.
        
        Returns:
            Self for method chaining.
        """
        path = Path(path)
        
        config_path = path / "config.json"
        with open(config_path, 'r') as f:
            import json
            saved_config = json.load(f)
        
        self.feature_names = saved_config["feature_names"]
        self.input_size = saved_config["input_size"]
        self.sequence_length = saved_config["sequence_length"]
        self.hidden_size = saved_config["hidden_size"]
        self.num_layers = saved_config["num_layers"]
        self.dropout = saved_config["dropout"]
        
        # Initialize model
        self.model = LSTMMultiTaskModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Load weights
        model_path = path / "model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load scalers if present
        try:
            import pickle
            x_scaler_path = path / "x_scaler.pkl"
            y_temp_scaler_path = path / "y_temp_scaler.pkl"
            if x_scaler_path.exists():
                with open(x_scaler_path, "rb") as f:
                    self._x_scaler = pickle.load(f)
            if y_temp_scaler_path.exists():
                with open(y_temp_scaler_path, "rb") as f:
                    self._y_temp_scaler = pickle.load(f)
        except Exception:
            pass
        
        # Load probability calibrator if available
        calibrator_path = path / "calibrator.pkl"
        if calibrator_path.exists():
            try:
                import pickle
                with open(calibrator_path, "rb") as f:
                    self._calibrator = pickle.load(f)
            except Exception:
                self._calibrator = None
        
        self.is_fitted = True
        print(f"  ✅ Multi-task LSTM model loaded from {path}")
        
        return self

