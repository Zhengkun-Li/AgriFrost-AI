"""TCN (Temporal Convolutional Network) model implementation for frost forecasting."""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import os
import pandas as pd
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..base import BaseModel
from src.utils.losses import FocalLoss


if not TORCH_AVAILABLE:
    raise ImportError("PyTorch is required for TCN models. Please install torch.")

# Reuse TimeSeriesDataset from GRU/LSTM (same structure)
from .gru_model import TimeSeriesDataset


class Chomp1d(nn.Module):
    """Remove padding from the end of sequence."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Temporal block with dilated causal convolution, weight normalization, ReLU and dropout."""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, sequence_length).
        
        Returns:
            Output tensor with residual connection.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """Temporal Convolutional Network model."""
    
    def __init__(self, input_size: int, num_channels: list = [32, 32, 32], 
                 kernel_size: int = 3, dropout: float = 0.1):
        """Initialize TCN model.
        
        Args:
            input_size: Number of input features.
            num_channels: List of channel sizes for each layer.
            kernel_size: Size of convolutional kernel.
            dropout: Dropout rate.
        """
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence, features).
        
        Returns:
            Output tensor of shape (batch, 1).
        """
        # TCN expects (batch, features, sequence) for Conv1d
        # Input is (batch, sequence, features), so transpose
        x = x.transpose(1, 2)  # (batch, features, sequence)
        
        # Apply TCN layers
        out = self.network(x)
        
        # Take the last time step
        out = out[:, :, -1]  # (batch, channels)
        
        # Final linear layer
        output = self.fc(out)
        return output.squeeze(-1)


class TCNForecastModel(BaseModel):
    """TCN model for frost forecasting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize TCN model.
        
        Args:
            config: Model configuration dictionary.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        super().__init__(config)
        
        model_params = config.get("model_params", {})
        self.sequence_length = model_params.get("sequence_length", 24)
        self.num_channels = model_params.get("num_channels", [32, 32, 32])
        self.kernel_size = model_params.get("kernel_size", 3)
        self.dropout = model_params.get("dropout", 0.1)
        self.learning_rate = model_params.get("learning_rate", 0.0005)
        self.batch_size = model_params.get("batch_size", 32)
        self.epochs = model_params.get("epochs", 50)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Task type: 'classification' or 'regression' (default regression for backward compatibility)
        self.task_type = config.get("task_type", "regression")
        
        # Optimization parameters
        self.use_early_stopping = model_params.get("early_stopping", True)
        self.patience = model_params.get("patience", 10)
        self.min_delta = model_params.get("min_delta", 1e-6)
        self.use_lr_scheduler = model_params.get("lr_scheduler", True)
        self.lr_scheduler_patience = model_params.get("lr_scheduler_patience", 5)
        self.lr_scheduler_factor = model_params.get("lr_scheduler_factor", 0.5)
        self.gradient_clip_value = model_params.get("gradient_clip", None)
        self.save_best_model = model_params.get("save_best_model", True)
        self.val_frequency = model_params.get("val_frequency", 1)
        self.checkpoint_frequency = model_params.get("checkpoint_frequency", 10)
        self.use_amp = model_params.get("use_amp", False)
        
        # Training utilities
        self.checkpoint_dir = None
        
        # Station grouping support
        self.station_column = config.get("station_column", "Stn Id")
        
        # Model will be initialized in fit() when we know input_size
        self.model = None
        self.input_size = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, **kwargs) -> "TCNForecastModel":
        """Train the TCN model.
        
        Args:
            X: Feature DataFrame (training data).
            y: Target Series (training data).
            eval_set: Optional list of (X_val, y_val) tuples for validation.
                      If provided, uses external validation split instead of internal 0.8/0.2 split.
            **kwargs: Additional arguments:
                - checkpoint_dir: Optional directory for saving checkpoints
                - resume_from_checkpoint: Optional epoch number or 'latest' to resume from
                - station_ids: Optional array of station IDs for sequence grouping (training)
                - station_ids_val: Optional array of station IDs for validation set
        
        Returns:
            Self for method chaining.
        """
        # Ensure numpy is available (re-import to avoid scope issues)
        import numpy as np
        # Validate configuration
        from src.models.utils import ConfigValidator
        is_valid, error_msg = ConfigValidator.validate_model_config("tcn", self.config)
        if not is_valid:
            raise ValueError(f"Invalid TCN configuration: {error_msg}")
        
        self.feature_names = list(X.columns)
        self.input_size = len(self.feature_names)
        
        # Initialize model
        self.model = TCNModel(
            input_size=self.input_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)
        
        # Prepare data - convert to numpy arrays and free original DataFrame memory
        X_array = X.values.astype(np.float32)
        y_array = y.values.astype(np.float32)
        
        # Free original DataFrame memory immediately
        del X, y
        import gc
        gc.collect()
        
        # Get station IDs if available (to avoid cross-station sequences)
        station_ids = kwargs.get('station_ids', None)
        station_ids_val = kwargs.get('station_ids_val', None)
        
        # Handle external validation set if provided (fixes double-split issue)
        X_val_external = None
        y_val_external = None
        use_external_val = False
        if eval_set is not None and len(eval_set) > 0:
            # Use external validation set (from outer time split)
            X_val_external, y_val_external = eval_set[0]
            use_external_val = True
            if self.progress_logger:
                self.progress_logger.log(f"  ✅ Using external validation set (from time split): {len(X_val_external)} samples", flush=True, detailed=True)
        
        # Check for NaN values and handle them properly
        # Note: Data should already be cleaned in preprocessing (forward_fill, etc.)
        # and in prepare_features_and_targets (also forward_fill). If NaN still exists,
        # we use forward/backward fill to preserve time continuity instead of deleting rows.
        
        # First, remove rows with NaN targets (targets must be valid - cannot predict without target)
        if np.isnan(y_array).any():
            if self.progress_logger:
                self.progress_logger.log(f"  ⚠️  Warning: Found NaN values in targets, removing rows", flush=True, detailed=True)
            valid_mask = ~np.isnan(y_array)
            X_array = X_array[valid_mask]
            y_array = y_array[valid_mask]
            if station_ids is not None:
                station_ids = np.asarray(station_ids)[valid_mask]
            if self.progress_logger:
                self.progress_logger.log(f"  Removed {np.sum(~valid_mask)} rows with NaN targets, remaining: {len(y_array)}", flush=True, detailed=True)
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
                        y_array = y_array[valid_mask]
                        if station_ids is not None:
                            station_ids = np.asarray(station_ids)[valid_mask]
                        if self.progress_logger:
                            self.progress_logger.log(f"  Removed {nan_rows.sum()} rows with remaining NaN, final samples: {len(y_array)}", flush=True, detailed=True)
                        del nan_rows, valid_mask
                        gc.collect()
            
            del X_df
            gc.collect()
        
        # Check for infinite values
        if np.isinf(X_array).any():
            if self.progress_logger:
                self.progress_logger.log(f"  ⚠️  Warning: Found infinite values in features, clipping to finite range", flush=True, detailed=True)
            X_array = np.clip(X_array, -1e6, 1e6)
        
        if np.isinf(y_array).any():
            if self.progress_logger:
                self.progress_logger.log(f"  ⚠️  Warning: Found infinite values in targets, clipping to finite range", flush=True, detailed=True)
            y_array = np.clip(y_array, -1e6, 1e6)
        
        # Validate station_ids if provided
        if station_ids is not None:
            station_ids = np.asarray(station_ids)
            if len(station_ids) != len(X_array):
                if self.progress_logger:
                    self.progress_logger.log(f"  ⚠️  Warning: station_ids length ({len(station_ids)}) doesn't match X length ({len(X_array)}). Ignoring station_ids.", flush=True, detailed=True)
                station_ids = None
        
        # Standardize features and target for stable training
        # Fit on training data inside fit(); persist scalers for prediction/inference
        try:
            from sklearn.preprocessing import StandardScaler
        except Exception:
            StandardScaler = None
        
        self._x_scaler = None
        self._y_scaler = None
        if StandardScaler is not None:
            self._x_scaler = StandardScaler()
            X_array = self._x_scaler.fit_transform(X_array).astype(np.float32)
            if self.task_type != "classification":
                # Only scale targets for regression
                self._y_scaler = StandardScaler()
                y_array = self._y_scaler.fit_transform(y_array.reshape(-1, 1)).astype(np.float32).ravel()
        else:
            # Fallback: center-only if sklearn unavailable
            x_mean = X_array.mean(axis=0, keepdims=True)
            x_std = X_array.std(axis=0, keepdims=True) + 1e-6
            self._x_scaler = {"mean": x_mean.astype(np.float32), "std": x_std.astype(np.float32)}
            X_array = ((X_array - x_mean) / x_std).astype(np.float32)
            if self.task_type != "classification":
                y_mean = float(y_array.mean())
                y_std = float(y_array.std() + 1e-6)
                self._y_scaler = {"mean": y_mean, "std": y_std}
                y_array = ((y_array - y_mean) / y_std).astype(np.float32)
        
        # Create datasets with station grouping support
        effective_seq_len = self.sequence_length
        dataset = TimeSeriesDataset(X_array, y_array, effective_seq_len, station_ids=station_ids)
        # Fallback if windowing yields no samples (e.g., long horizon + gaps)
        if len(dataset) == 0 and effective_seq_len > 24:
            if self.progress_logger:
                self.progress_logger.log(f"  ⚠️  Sequence dataset empty with length={effective_seq_len}, falling back to 24", flush=True, detailed=True)
            effective_seq_len = 24
            dataset = TimeSeriesDataset(X_array, y_array, effective_seq_len, station_ids=station_ids)
        if len(dataset) == 0 and effective_seq_len > 12:
            if self.progress_logger:
                self.progress_logger.log(f"  ⚠️  Sequence dataset still empty, falling back to 12", flush=True, detailed=True)
            effective_seq_len = 12
            dataset = TimeSeriesDataset(X_array, y_array, effective_seq_len, station_ids=station_ids)
        if len(dataset) == 0:
            raise ValueError("After sequence construction, dataset is empty. Please check data continuity and NaN handling.")
        
        # Free numpy arrays after dataset creation (dataset will keep its own copies)
        del X_array, y_array
        gc.collect()
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n_samples = len(dataset)
        
        # CRITICAL FIX: Use external validation set if provided (from outer time split)
        # This avoids double-split issue and ensures temporal consistency
        if use_external_val:
            # Use all training data for training (no internal split)
            train_dataset = dataset
            train_indices = list(range(n_samples))
            
            # Prepare external validation set
            X_val_array = X_val_external.values.astype(np.float32)
            y_val_array = y_val_external.values.astype(np.float32)
            
            # Apply same preprocessing as training data
            # Remove NaN targets
            if np.isnan(y_val_array).any():
                valid_mask = ~np.isnan(y_val_array)
                X_val_array = X_val_array[valid_mask]
                y_val_array = y_val_array[valid_mask]
                if station_ids_val is not None:
                    station_ids_val = np.asarray(station_ids_val)[valid_mask]
            
            # Handle NaN in features (use same scaler from training)
            if np.isnan(X_val_array).any():
                import pandas as pd
                X_val_df = pd.DataFrame(X_val_array)
                if station_ids_val is not None:
                    X_val_df = X_val_df.groupby(station_ids_val).ffill().bfill()
                else:
                    X_val_df = X_val_df.ffill().bfill()
                X_val_array = X_val_df.values.astype(np.float32)
                # Remove rows with remaining NaN
                nan_rows = np.isnan(X_val_array).any(axis=1)
                if nan_rows.any():
                    valid_mask = ~nan_rows
                    X_val_array = X_val_array[valid_mask]
                    y_val_array = y_val_array[valid_mask]
                    if station_ids_val is not None:
                        station_ids_val = np.asarray(station_ids_val)[valid_mask]
            
            # Apply same scaling as training data
            if self._x_scaler is not None:
                if isinstance(self._x_scaler, dict):
                    X_val_array = ((X_val_array - self._x_scaler["mean"]) / self._x_scaler["std"]).astype(np.float32)
                else:
                    X_val_array = self._x_scaler.transform(X_val_array).astype(np.float32)
            
            if self.task_type != "classification" and self._y_scaler is not None:
                if isinstance(self._y_scaler, dict):
                    y_val_array = ((y_val_array - self._y_scaler["mean"]) / self._y_scaler["std"]).astype(np.float32)
                else:
                    y_val_array = self._y_scaler.transform(y_val_array.reshape(-1, 1)).astype(np.float32).ravel()
            
            # Create validation dataset
            val_dataset = TimeSeriesDataset(X_val_array, y_val_array, effective_seq_len, station_ids=station_ids_val)
            val_indices = list(range(len(val_dataset)))
            
            if self.progress_logger:
                self.progress_logger.log(f"  ✅ Using external validation: {len(val_dataset)} sequences (from time split)", flush=True, detailed=True)
        else:
            # Fallback: internal split if no external validation provided
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
            if self.progress_logger:
                self.progress_logger.log(f"  ⚠️  Using internal 0.8/0.2 split (no external validation provided)", flush=True, detailed=True)
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
        
        # If dataset is smaller than batch_size, avoid dropping the only batch
        train_drop_last = n_samples >= adaptive_batch_size
        effective_batch_size = min(adaptive_batch_size, max(1, n_samples))
        # Optional weighted sampler for imbalanced classification
        sampler = None
        if self.task_type == "classification" and self.config.get("model_params", {}).get("use_weighted_sampler", True):
            # Build per-sequence weight based on end-step label
            # Recompute sequence-end labels for all sequences, then index into train_indices
            all_y = dataset.y.cpu().numpy()
            all_seq_labels = np.array([float(all_y[idx[-1]]) for idx in dataset.sequence_indices], dtype=np.float32)
            
            # For extremely imbalanced data, use more aggressive sampling weights
            # Use the same pos_weight computed above, but ensure it's applied correctly
            pos_w = float(self._pos_weight.item()) if hasattr(self, "_pos_weight") else 1.0
            
            # For very imbalanced data (<1% positive), further boost positive sample weight
            # This ensures positive samples are sampled more frequently during training
            train_seq_labels = all_seq_labels[train_indices]
            train_pos_ratio = (train_seq_labels > 0.5).sum() / len(train_seq_labels)
            
            if train_pos_ratio < 0.01:  # <1% positive in training set
                # Very imbalanced: use more aggressive sampling (multiply by 2-3x)
                # This ensures positive samples appear in most batches
                pos_sampling_weight = pos_w * 2.5
                if self.progress_logger:
                    self.progress_logger.log(
                        f"  ✅ Very imbalanced training set ({train_pos_ratio*100:.2f}% pos), using aggressive sampling weight={pos_sampling_weight:.2f}",
                        flush=True
                    )
            else:
                pos_sampling_weight = pos_w
            
            sample_weights = np.where(all_seq_labels > 0.5, pos_sampling_weight, 1.0).astype(np.float32)
            train_weights = torch.from_numpy(sample_weights[train_indices]).double()
            
            # Option: Use class-balanced batch sampling (ensure each batch has positive samples)
            use_class_balanced_batch = self.config.get("model_params", {}).get("use_class_balanced_batch", False)
            if use_class_balanced_batch and train_pos_ratio < 0.05:  # Only for very imbalanced data
                # Create a custom sampler that ensures balanced batches
                # This is more complex, so we'll use a simpler approach: increase positive sample weight even more
                pos_sampling_weight = pos_sampling_weight * 3.0  # Further boost
                sample_weights = np.where(all_seq_labels > 0.5, pos_sampling_weight, 1.0).astype(np.float32)
                train_weights = torch.from_numpy(sample_weights[train_indices]).double()
                if self.progress_logger:
                    self.progress_logger.log(
                        f"  ✅ Using class-balanced batch sampling (boosted weight={pos_sampling_weight:.2f})",
                        flush=True
                    )
            
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
        if self.task_type == "classification":
            # Compute positive weight = N_neg / N_pos (avoid div by zero)
            y_np = dataset.y.numpy() if isinstance(dataset.y, torch.Tensor) else np.asarray(dataset.y, dtype=np.float32)
            y_seq_last = np.array([y_np[idx[-1]] for idx in dataset.sequence_indices], dtype=np.float32)
            pos = max(1.0, float((y_seq_last > 0.5).sum()))
            neg = max(1.0, float(len(y_seq_last) - (y_seq_last > 0.5).sum()))
            imbalance_ratio = neg / pos
            
            # For extremely imbalanced data (<1% positive), use more aggressive weighting
            # Option 1: Square root weighting (less aggressive than linear, more than sqrt)
            # Option 2: Use sqrt for very imbalanced, linear for moderately imbalanced
            if imbalance_ratio > 100:  # <1% positive
                # Very imbalanced: use sqrt to prevent excessive weight
                pos_weight_val = np.sqrt(imbalance_ratio)
                weight_strategy = "sqrt (very imbalanced)"
            elif imbalance_ratio > 50:  # <2% positive
                # Moderately imbalanced: use sqrt with multiplier
                pos_weight_val = np.sqrt(imbalance_ratio) * 1.5
                weight_strategy = "sqrt*1.5 (moderately imbalanced)"
            else:
                # Less imbalanced: use linear
                pos_weight_val = imbalance_ratio
                weight_strategy = "linear"
            
            # Choose loss function based on configuration
            use_focal_loss = self.config.get("model_params", {}).get("use_focal_loss", False)
            
            if use_focal_loss:
                # Focal Loss: better for extremely imbalanced data
                focal_alpha = self.config.get("model_params", {}).get("focal_alpha", 0.25)
                focal_gamma = self.config.get("model_params", {}).get("focal_gamma", 2.0)
                criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
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
                self._pos_weight = torch.tensor([pos_weight_val], device=self.device, dtype=torch.float32)
                criterion = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)
                
                if self.progress_logger:
                    self.progress_logger.log(
                        f"  ✅ Class imbalance: {pos:.0f} pos / {neg:.0f} neg (ratio={imbalance_ratio:.1f}:1, <{100/imbalance_ratio:.2f}% pos)",
                        flush=True
                    )
                    self.progress_logger.log(
                        f"  ✅ Using pos_weight={pos_weight_val:.2f} ({weight_strategy}) for BCEWithLogitsLoss",
                        flush=True
                    )
        else:
            criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Mixed precision training for faster training and lower memory usage
        use_amp = self.use_amp and torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        if use_amp:
            if self.progress_logger:
                self.progress_logger.log("  ✅ Using Mixed Precision Training (AMP) for faster training", flush=True)
        
        # Learning rate scheduler (using PyTorch built-in)
        scheduler = None
        if self.use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                min_lr=1e-6
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        start_epoch = 0  # Default: start from epoch 0
        # Note: Resume-from-checkpoint support can be implemented via CheckpointManager in the future.
        
        # Setup training utilities if not already set up
        checkpoint_dir = kwargs.get('checkpoint_dir', None)
        resume_from_checkpoint = kwargs.get('resume_from_checkpoint', None)
        
        # Validate training arguments
        from src.models.utils import ConfigValidator
        is_valid, error_msg = ConfigValidator.validate_training_args(
            "tcn", checkpoint_dir=checkpoint_dir
        )
        if not is_valid:
            raise ValueError(f"Invalid training arguments: {error_msg}")
        
        if checkpoint_dir and not self.checkpoint_manager:
            # Setup training tools if checkpoint_dir provided
            self.setup_training_tools(
                checkpoint_dir=checkpoint_dir,
                checkpoint_frequency=self.checkpoint_frequency,
                save_best=self.save_best_model,
                best_metric="val_loss",
                best_mode="min"
            )
            self.checkpoint_dir = Path(checkpoint_dir)  # For backward compatibility
        
        # Initialize training history if not already set up
        if not self.training_history:
            from src.models.utils import TrainingHistory
            self.training_history = TrainingHistory()
        
        # Initialize progress logger if not already set up
        if not self.progress_logger:
            from src.models.utils import ProgressLogger
            self.progress_logger = ProgressLogger()
        
        # Start training history tracking
        self.training_history.start_training()
        
        # Log training start
        self.progress_logger.log_training_start(
            model_name="TCN",
            device=str(self.device),
            config={
                "input_size": self.input_size,
                "num_channels": self.num_channels,
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
        print(f"\n  🚀 Starting TCN training")
        print(f"     Device: {self.device}")
        print(f"     Input size: {self.input_size}")
        print(f"     Num channels: {self.num_channels}")
        batch_info = f"{effective_batch_size}" if effective_batch_size == self.batch_size else f"{effective_batch_size} (adaptive, config: {self.batch_size})"
        print(f"     Batch size: {batch_info}")
        print(f"     Num workers: {num_workers}")
        print(f"     Epochs: {self.epochs}")
        print(f"     Sequence length: {effective_seq_len}")
        print(f"     Total sequences: {n_samples:,}")
        print(f"     Train sequences: {len(train_dataset):,} ({len(train_loader):,} batches)")
        print(f"     Val sequences: {len(val_dataset):,} ({len(val_loader):,} batches)")
        if self.task_type == "classification" and hasattr(self, "_pos_weight"):
            print(f"     Pos weight: {self._pos_weight.item():.2f}")
        sys.stdout.flush()
        
        self.model.train()
        # Use unified progress logger for epoch progress bar
        if self.progress_logger:
            epoch_pbar = self.progress_logger.get_tqdm(
                range(start_epoch, self.epochs),
                desc="Training",
                unit="epoch",
                initial=start_epoch
            )
        else:
            epoch_pbar = range(start_epoch, self.epochs)
        
        for epoch in epoch_pbar:
            epoch_iter_start = time.time()
            
            # Training phase with progress bar
            train_loss = 0.0
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
            
            for batch_X, batch_y in train_pbar:
                # Ensure training mode per batch
                self.model.train()
                batch_X = batch_X.to(self.device, non_blocking=True)  # Non-blocking transfer
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Mixed precision training
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(batch_X)
                        if self.task_type == "classification":
                            loss = criterion(outputs, batch_y)
                        else:
                            loss = criterion(outputs, batch_y)
                    
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping (using PyTorch built-in)
                    if self.gradient_clip_value is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(batch_X)
                    if self.task_type == "classification":
                        loss = criterion(outputs, batch_y)
                    else:
                        loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping (using PyTorch built-in)
                    if self.gradient_clip_value is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                    
                    optimizer.step()
                
                # Check for NaN loss
                loss_value = loss.item()
                if np.isnan(loss_value) or np.isinf(loss_value):
                    if self.progress_logger:
                        self.progress_logger.log(f"\n  ⚠️  Warning: Invalid loss value ({loss_value}) at batch {train_batches}", flush=True, detailed=True)
                        self.progress_logger.log(f"     This may indicate numerical instability. Consider:", flush=True, detailed=True)
                        self.progress_logger.log(f"     - Reducing learning rate", flush=True, detailed=True)
                        self.progress_logger.log(f"     - Checking for NaN/inf in input data", flush=True, detailed=True)
                        self.progress_logger.log(f"     - Using gradient clipping", flush=True, detailed=True)
                    # Skip this batch's loss
                    continue
                
                train_loss += loss_value
                train_batches += 1
                
                # Memory optimization: delete batch data immediately after use
                del batch_X, batch_y, outputs, loss
                
                # No need to update progress bar manually when tqdm is disabled for file output
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
                torch.cuda.empty_cache()
            
            # If no batches processed, warn and break to avoid spinning through epochs
            if train_batches == 0:
                if self.progress_logger:
                    self.progress_logger.log(
                        "  ⚠️  Warning: No training batches processed this epoch. Check DataLoader/dataset sizes.",
                        flush=True,
                        detailed=True,
                    )
                break
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
            
            # Validation phase (every N epochs for efficiency, but always on first and last epoch)
            should_validate = (epoch + 1) % self.val_frequency == 0 or epoch == 0 or epoch == self.epochs - 1
            
            if should_validate:
                val_loss = 0.0
                val_batches = 0
                self.model.eval()
                
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
                    for batch_X, batch_y in val_pbar:
                        batch_X = batch_X.to(self.device, non_blocking=True)
                        batch_y = batch_y.to(self.device, non_blocking=True)
                        
                        # Use mixed precision for validation too
                        if scaler is not None:
                            with torch.amp.autocast('cuda'):
                                outputs = self.model(batch_X)
                                batch_val_loss = criterion(outputs, batch_y).item()
                        else:
                            outputs = self.model(batch_X)
                            batch_val_loss = criterion(outputs, batch_y).item()
                        
                        val_loss += batch_val_loss
                        val_batches += 1
                        
                        # Memory optimization: delete batch data immediately
                        del batch_X, batch_y, outputs
                        
                        # No need to update progress bar manually when tqdm is disabled for file output
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            else:
                # Skip validation, use previous validation loss for early stopping
                avg_val_loss = best_val_loss if best_val_loss != float('inf') else float('inf')
            self.model.train()
            
            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step(avg_val_loss)
                new_lr = optimizer.param_groups[0]['lr']
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_iter_start
            elapsed_time = time.time() - epoch_start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = self.epochs - (epoch + 1)
            estimated_remaining = avg_epoch_time * remaining_epochs
            
            # Record epoch in training history
            self.training_history.record_epoch(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss if should_validate else None,
                learning_rate=current_lr,
                epoch_time=epoch_time
            )
            
            # Log epoch summary (detailed log only - epoch details are too verbose for brief log)
            if self.progress_logger:
                val_info = f"val_loss={avg_val_loss:.6f}" if should_validate else "val_loss=skipped"
                progress_info = f"train_loss={avg_train_loss:.6f}, {val_info}, lr={current_lr:.6e}"
                self.progress_logger.log(f"  Epoch {epoch+1}/{self.epochs} - {progress_info} - Time: {epoch_time:.1f}s, ETA: {estimated_remaining/60:.1f}m", flush=True, detailed=True)
            
            # Save checkpoint periodically
            if self.checkpoint_manager and self.checkpoint_manager.should_save_checkpoint(epoch + 1):
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch + 1,
                    model_state=self.model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler else None,
                    scaler_state=scaler.state_dict() if scaler else None,
                    metrics={'train_loss': avg_train_loss, 'val_loss': avg_val_loss},
                    training_history=self.training_history
                )
                self.progress_logger.log(f"  💾 Checkpoint saved: epoch {epoch + 1}", flush=True)
            
            # Early stopping logic
            if self.use_early_stopping:
                improved = False
                if avg_val_loss < best_val_loss - self.min_delta:
                    prev_best = best_val_loss
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    improved = True
                    # Save best model state
                    if self.save_best_model:
                        best_model_state = self.model.state_dict().copy()
                        # Also save via checkpoint manager if available
                        if self.checkpoint_manager:
                            self.checkpoint_manager.save_best_checkpoint(
                                epoch=epoch + 1,
                                model_state=best_model_state,
                                metric_value=avg_val_loss
                            )
                        # Log improvement
                        self.progress_logger.log_improvement("val_loss", avg_val_loss, best_val_loss)
                else:
                    patience_counter += 1
                
                # Update epoch progress bar if using tqdm
                if hasattr(epoch_pbar, 'set_postfix'):
                    epoch_pbar.set_postfix({
                        'train_loss': f'{avg_train_loss:.6f}',
                        'val_loss': f'{avg_val_loss:.6f}',
                        'lr': f'{current_lr:.6f}',
                        'patience': f'{patience_counter}/{self.patience}',
                        'ETA': f'{estimated_remaining/60:.1f}m'
                    })
                    if improved:
                        epoch_pbar.write(f"  ✅ Improved! Val Loss: {avg_val_loss:.6f} (Prev Best: {prev_best:.6f})")
                
                # Early stopping
                if patience_counter >= self.patience:
                    self.progress_logger.log_early_stopping(epoch + 1, self.patience)
                    if self.save_best_model and best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
            else:
                # Update epoch progress bar if using tqdm
                if hasattr(epoch_pbar, 'set_postfix'):
                    epoch_pbar.set_postfix({
                        'train_loss': f'{avg_train_loss:.6f}',
                        'val_loss': f'{avg_val_loss:.6f}',
                        'lr': f'{current_lr:.6f}',
                        'ETA': f'{estimated_remaining/60:.1f}m'
                    })
        
        # Save training artifacts using unified utilities
        if self.checkpoint_dir and self.training_history and len(self.training_history) > 0:
            self.save_training_artifacts(self.checkpoint_dir)
        
        # Threshold selection on validation set for classification task
        if self.task_type == "classification":
            try:
                self.model.eval()
                y_true_list: list[float] = []
                y_proba_list: list[float] = []
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device, non_blocking=True)
                        logits = self.model(batch_X)
                        proba = torch.sigmoid(logits).detach().cpu().numpy()
                        y_proba_list.extend(proba.reshape(-1).tolist())
                        y_true_list.extend(batch_y.cpu().numpy().reshape(-1).tolist())
                y_true_np = np.array(y_true_list, dtype=np.float32)
                y_proba_np = np.array(y_proba_list, dtype=np.float32)
                
                # PROBABILITY CALIBRATION: Fit calibrator on validation set
                # This improves Brier Score and ECE while maintaining discrimination (ROC-AUC, PR-AUC)
                use_calibration = self.config.get("model_params", {}).get("use_probability_calibration", True)
                calibration_method = self.config.get("model_params", {}).get("calibration_method", "platt")  # "platt" or "isotonic"
                
                self._calibrator = None
                if use_calibration:
                    try:
                        from src.utils.calibration import ProbabilityCalibrator
                        self._calibrator = ProbabilityCalibrator(method=calibration_method)
                        self._calibrator.fit(y_proba_np, y_true_np)
                        
                        # Apply calibration to validation probabilities for threshold selection
                        y_proba_calibrated = self._calibrator.transform(y_proba_np)
                        
                        if self.progress_logger:
                            # Compare before/after calibration
                            try:
                                from sklearn.metrics import brier_score_loss
                                brier_before = brier_score_loss(y_true_np, y_proba_np)
                                brier_after = brier_score_loss(y_true_np, y_proba_calibrated)
                                
                                # Calculate ECE if available
                                try:
                                    from sklearn.calibration import calibration_curve
                                    fraction_of_positives, mean_predicted_value = calibration_curve(
                                        y_true_np, y_proba_np, n_bins=10, strategy='uniform'
                                    )
                                    ece_before = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                                    
                                    fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
                                        y_true_np, y_proba_calibrated, n_bins=10, strategy='uniform'
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
                                    # Fallback without ECE
                                    self.progress_logger.log(
                                        f"  ✅ Probability calibration ({calibration_method}): "
                                        f"Brier Score {brier_before:.6f} → {brier_after:.6f} "
                                        f"({(brier_after-brier_before)/brier_before*100:+.1f}%)",
                                        flush=True,
                                        detailed=True
                                    )
                            except Exception:
                                self.progress_logger.log(
                                    f"  ✅ Probability calibration ({calibration_method}) applied",
                                    flush=True,
                                    detailed=True
                                )
                        
                        # Use calibrated probabilities for threshold selection
                        y_proba_np = y_proba_calibrated
                    except Exception as e:
                        if self.progress_logger:
                            self.progress_logger.log(
                                f"  ⚠️  Warning: Probability calibration failed: {e}. Using uncalibrated probabilities.",
                                flush=True,
                                detailed=True
                            )
                        self._calibrator = None
                
                # Check if validation set has positive samples
                n_pos = (y_true_np > 0.5).sum()
                if n_pos == 0:
                    if self.progress_logger:
                        self.progress_logger.log(f"  ⚠️  Warning: No positive samples in validation set, using default threshold=0.5", flush=True)
                    self.best_threshold = 0.5
                    return
                
                # For extremely imbalanced data, prioritize recall over precision
                # Strategy: Try to maximize F1 first, but if recall is too low, prioritize recall
                val_pos_ratio = n_pos / len(y_true_np)
                is_extremely_imbalanced = val_pos_ratio < 0.01  # <1% positive
                
                # Check if we should use PR-AUC optimization (better for imbalanced data)
                use_pr_auc_optimization = self.config.get("model_params", {}).get("use_pr_auc_threshold", False) and is_extremely_imbalanced
                
                best_f1 = -1.0
                best_thr = 0.5
                best_precision = 0.0
                best_recall = 0.0
                best_f1_recall = 0.0  # F1*recall for imbalanced data
                best_pr_auc = -1.0
                
                # Use finer-grained threshold search (50 points instead of 19)
                thresholds = np.linspace(0.01, 0.99, 50)
                
                # If using PR-AUC optimization, compute PR curve first
                if use_pr_auc_optimization:
                    try:
                        from sklearn.metrics import precision_recall_curve, auc
                        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true_np, y_proba_np)
                        pr_auc = auc(recall_curve, precision_curve)
                        # Find threshold that maximizes PR-AUC (by finding point closest to (1,1) on PR curve)
                        # Or use threshold that gives best F1 on PR curve
                        best_pr_auc = pr_auc
                        if self.progress_logger:
                            self.progress_logger.log(f"  📊 PR-AUC on validation set: {pr_auc:.4f}", flush=True)
                    except ImportError:
                        use_pr_auc_optimization = False
                        if self.progress_logger:
                            self.progress_logger.log(f"  ⚠️  sklearn not available, falling back to F1 optimization", flush=True)
                for thr in thresholds:
                    y_pred = (y_proba_np >= thr).astype(np.int32)
                    tp = ((y_pred == 1) & (y_true_np == 1)).sum()
                    fp = ((y_pred == 1) & (y_true_np == 0)).sum()
                    fn = ((y_pred == 0) & (y_true_np == 1)).sum()
                    
                    if tp + fp == 0:  # No positive predictions
                        continue
                    if tp + fn == 0:  # No actual positives
                        continue
                    
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    
                    # For extremely imbalanced data, use F1*recall as primary metric
                    # This ensures we get both good F1 and good recall
                    if is_extremely_imbalanced:
                        score = f1 * recall  # Prioritize recall while maintaining F1
                        if score > best_f1_recall:
                            best_f1_recall = score
                            best_f1 = f1
                            best_thr = float(thr)
                            best_precision = precision
                            best_recall = recall
                    else:
                        # For less imbalanced data, use F1
                        if f1 > best_f1:
                            best_f1 = f1
                            best_thr = float(thr)
                            best_precision = precision
                            best_recall = recall
                
                # Fallback: if no good threshold found, use Youden's J statistic (maximize TPR - FPR)
                if best_f1 <= 0.0:
                    if self.progress_logger:
                        self.progress_logger.log(f"  ⚠️  Warning: No threshold found with F1>0, using Youden's J", flush=True)
                    best_j = -1.0
                    for thr in thresholds:
                        y_pred = (y_proba_np >= thr).astype(np.int32)
                        tp = ((y_pred == 1) & (y_true_np == 1)).sum()
                        fp = ((y_pred == 1) & (y_true_np == 0)).sum()
                        tn = ((y_pred == 0) & (y_true_np == 0)).sum()
                        fn = ((y_pred == 0) & (y_true_np == 1)).sum()
                        tpr = tp / (tp + fn + 1e-8)
                        fpr = fp / (fp + tn + 1e-8)
                        j = tpr - fpr
                        if j > best_j:
                            best_j = j
                            best_thr = float(thr)
                            best_f1 = 0.0  # Mark as fallback
                
                self.best_threshold = best_thr
                if self.progress_logger:
                    if best_f1 > 0:
                        strategy_str = "F1*recall" if is_extremely_imbalanced else "F1"
                        self.progress_logger.log(
                            f"  Selected threshold={best_thr:.3f} maximizing {strategy_str}={best_f1:.4f}*{best_recall:.4f}={best_f1*best_recall:.4f} "
                            f"(P={best_precision:.3f}, R={best_recall:.3f}) on validation set",
                            flush=True
                        )
                    else:
                        self.progress_logger.log(f"  Selected threshold={best_thr:.3f} using Youden's J={best_j:.4f} on validation set", flush=True)
            except Exception:
                self.best_threshold = 0.5
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, station_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Make point predictions.
        
        Args:
            X: Feature DataFrame.
            station_ids: Optional station IDs aligned with X rows for boundary-safe windowing.
        
        Returns:
            Predictions array.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        X_array = X.values.astype(np.float32)
        # Apply stored feature scaler
        if hasattr(self, "_x_scaler") and self._x_scaler is not None:
            try:
                # sklearn scaler
                X_array = self._x_scaler.transform(X_array).astype(np.float32)
            except AttributeError:
                # dict scaler
                X_array = ((X_array - self._x_scaler["mean"]) / self._x_scaler["std"]).astype(np.float32)
        
        # Build boundary-safe sequences
        seqs = []
        if station_ids is not None:
            station_ids = np.asarray(station_ids)
            unique_stations = np.unique(station_ids)
            for sid in unique_stations:
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
        
        predictions = []
        with torch.no_grad():
            for idx in seqs:
                sequence = X_array[idx]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                pred = self.model(sequence_tensor)
                if self.task_type == "classification":
                    pred = torch.sigmoid(pred)
                predictions.append(pred.cpu().numpy())
        
        # Align length with X by padding the first available prediction within each segment
        pred_series = np.full((len(X_array),), np.nan, dtype=np.float32)
        pos = 0
        for seq_idx, pred in zip(seqs, predictions):
            pred_series[seq_idx[-1]] = float(pred)
        # Forward fill within available positions
        last = None
        for i in range(len(pred_series)):
            if not np.isnan(pred_series[i]):
                last = pred_series[i]
            elif last is not None:
                pred_series[i] = last
        # Back fill start if needed
        if np.isnan(pred_series[0]):
            first_valid = np.flatnonzero(~np.isnan(pred_series))
            if first_valid.size > 0:
                pred_series[:first_valid[0]] = pred_series[first_valid[0]]
            else:
                pred_series[:] = 0.0
        if self.task_type == "classification":
            thr = getattr(self, "best_threshold", 0.5)
            return (pred_series >= thr).astype(np.int32)
        else:
            # Inverse-scale predictions to original temperature units
            if hasattr(self, "_y_scaler") and self._y_scaler is not None:
                try:
                    pred_series = self._y_scaler.inverse_transform(pred_series.reshape(-1, 1)).ravel().astype(np.float32)
                except AttributeError:
                    mean = self._y_scaler["mean"]
                    std = self._y_scaler["std"]
                    pred_series = (pred_series * std + mean).astype(np.float32)
            return pred_series
    
    def predict_proba(self, X: pd.DataFrame, station_ids: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Return probabilities for classification task; None for regression."""
        if self.task_type != "classification":
            return None
        self.model.eval()
        X_array = X.values.astype(np.float32)
        if hasattr(self, "_x_scaler") and self._x_scaler is not None:
            try:
                X_array = self._x_scaler.transform(X_array).astype(np.float32)
            except AttributeError:
                X_array = ((X_array - self._x_scaler["mean"]) / self._x_scaler["std"]).astype(np.float32)
        # Build sequences
        seqs = []
        if station_ids is not None:
            station_ids = np.asarray(station_ids)
            unique_stations = np.unique(station_ids)
            for sid in unique_stations:
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
            for idx in seqs:
                sequence = X_array[idx]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                logits = self.model(sequence_tensor)
                proba = torch.sigmoid(logits).cpu().numpy().reshape(-1)[0]
                proba_series[idx[-1]] = float(proba)
        
        # Forward/back fill probabilities first (before calibration)
        last = None
        for i in range(len(proba_series)):
            if not np.isnan(proba_series[i]):
                last = proba_series[i]
            elif last is not None:
                proba_series[i] = last
        if np.isnan(proba_series[0]):
            first_valid = np.flatnonzero(~np.isnan(proba_series))
            if first_valid.size > 0:
                proba_series[:first_valid[0]] = proba_series[first_valid[0]]
            else:
                proba_series[:] = 0.0
        
        # Apply probability calibration if available (improves Brier Score and ECE)
        # Apply after forward/back fill to ensure all values are valid
        if hasattr(self, "_calibrator") and self._calibrator is not None and self._calibrator.is_fitted:
            proba_series = self._calibrator.transform(proba_series)
        
        return proba_series
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """TCN doesn't provide direct feature importance."""
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
        
        metadata = {
            "model_type": "tcn",
            "input_size": self.input_size,
            "sequence_length": self.sequence_length,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "feature_names": self.feature_names,
            "config": self.config
        }
        
        metadata_path = path / "metadata.pkl"
        import pickle
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        # Save scalers if available
        if hasattr(self, "_x_scaler") and self._x_scaler is not None:
            try:
                # sklearn scaler
                with open(path / "x_scaler.pkl", "wb") as f:
                    pickle.dump(self._x_scaler, f)
            except Exception:
                with open(path / "x_scaler_simple.pkl", "wb") as f:
                    pickle.dump(self._x_scaler, f)
        if hasattr(self, "_y_scaler") and self._y_scaler is not None:
            try:
                with open(path / "y_scaler.pkl", "wb") as f:
                    pickle.dump(self._y_scaler, f)
            except Exception:
                with open(path / "y_scaler_simple.pkl", "wb") as f:
                    pickle.dump(self._y_scaler, f)
        
        # Save probability calibrator if available
        if hasattr(self, "_calibrator") and self._calibrator is not None and self._calibrator.is_fitted:
            try:
                with open(path / "calibrator.pkl", "wb") as f:
                    pickle.dump(self._calibrator, f)
            except Exception as e:
                # If calibrator can't be pickled, save its parameters instead
                if self.progress_logger:
                    self.progress_logger.log(
                        f"  ⚠️  Warning: Could not save calibrator: {e}",
                        flush=True,
                        detailed=True
                    )
    
    @classmethod
    def load(cls, path: Path) -> "TCNForecastModel":
        """Load model from disk.
        
        Args:
            path: Directory path containing saved model.
        
        Returns:
            Loaded model instance.
        """
        import pickle
        path = Path(path)
        
        model_path = path / "model.pth"
        metadata_path = path / "metadata.pkl"
        
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        instance = cls(metadata["config"])
        instance.input_size = metadata["input_size"]
        instance.sequence_length = metadata["sequence_length"]
        instance.num_channels = metadata["num_channels"]
        instance.kernel_size = metadata["kernel_size"]
        instance.dropout = metadata["dropout"]
        instance.feature_names = metadata.get("feature_names", [])
        
        instance.model = TCNModel(
            input_size=instance.input_size,
            num_channels=instance.num_channels,
            kernel_size=instance.kernel_size,
            dropout=instance.dropout
        ).to(instance.device)
        
        instance.model.load_state_dict(torch.load(model_path, map_location=instance.device))
        instance.model.eval()
        instance.is_fitted = True
        
        # Load scalers if present
        import pickle
        x_scaler_path = None
        y_scaler_path = None
        for fn in ["x_scaler.pkl", "x_scaler_simple.pkl"]:
            if (path / fn).exists():
                x_scaler_path = path / fn
                break
        for fn in ["y_scaler.pkl", "y_scaler_simple.pkl"]:
            if (path / fn).exists():
                y_scaler_path = path / fn
                break
        if x_scaler_path is not None:
            with open(x_scaler_path, "rb") as f:
                instance._x_scaler = pickle.load(f)
        if y_scaler_path is not None:
            with open(y_scaler_path, "rb") as f:
                instance._y_scaler = pickle.load(f)
        
        # Load probability calibrator if available
        calibrator_path = path / "calibrator.pkl"
        if calibrator_path.exists():
            try:
                with open(calibrator_path, "rb") as f:
                    instance._calibrator = pickle.load(f)
            except Exception:
                instance._calibrator = None
        
        return instance

