"""Training history tracking utility (Refactored with schema injection and advanced APIs).

This is a refactored version of TrainingHistory that:
- Supports external schema injection (MetricSchema)
- Provides advanced APIs (best epoch, early stopping, summary)
- Supports DataFrame output for visualization
- Records raw event log
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import time
from dataclasses import dataclass, field

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from src.training.metrics import MetricSchema
    METRIC_SCHEMA_AVAILABLE = True
except ImportError:
    METRIC_SCHEMA_AVAILABLE = False


@dataclass
class TrainingEvent:
    """Single training event record."""
    type: str  # improve, lr_change, early_stop, etc.
    epoch: int
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


class TrainingHistory:
    """Track and manage training history with advanced APIs.
    
    **Key Improvements:**
    - External schema injection (MetricSchema)
    - Advanced APIs (best epoch, early stopping, summary)
    - DataFrame output for visualization
    - Raw event log
    
    **Event Types:**
    - improve: Metric improved
    - lr_change: Learning rate changed
    - early_stop: Early stopping triggered
    - checkpoint: Checkpoint saved
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        use_metric_schema: bool = True
    ):
        """Initialize training history tracker.
        
        - metrics can be from MetricSchema (external injection)
        - use_metric_schema for automatic metric validation
        
        Args:
            metrics: List of expected metric names. If None, uses MetricSchema.default_history_metrics().
            use_metric_schema: Whether to use MetricSchema for default metrics.
        """
        # Use MetricSchema if available and enabled
        if metrics is None:
            if use_metric_schema and METRIC_SCHEMA_AVAILABLE:
                metrics = MetricSchema.default_history_metrics()
            else:
                metrics = ['train_loss', 'val_loss', 'learning_rate', 'epoch_time']
        
        # Validate metrics if schema enabled
        if use_metric_schema and METRIC_SCHEMA_AVAILABLE:
            is_valid, error_msg = MetricSchema.validate_metrics(metrics)
            if not is_valid:
                # Log warning but don't fail (allow custom metrics via registry)
                import logging
                _logger = logging.getLogger(__name__)
                _logger.warning(f"Metric validation warning: {error_msg}")
        
        self.metrics = list(metrics)
        self.use_metric_schema = use_metric_schema and METRIC_SCHEMA_AVAILABLE
        
        self.history: Dict[str, List[Any]] = {
            'epoch': [],
            **{metric: [] for metric in metrics}
        }
        self.start_time: Optional[float] = None
        self.current_epoch: int = 0
        self.epoch_times: List[float] = []
        
        # Advanced tracking (new)
        self.best_metrics: Dict[str, Dict[str, Any]] = {}  # {metric_name: {value, epoch}}
        self.early_stopping: Optional[Dict[str, Any]] = None
        self.events: List[TrainingEvent] = []  # Raw event log
    
    def start_training(self) -> None:
        """Mark the start of training."""
        self.start_time = time.time()
        self._add_event('training_start', epoch=0, data={})
    
    def record_epoch(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        epoch_time: Optional[float] = None,
        **kwargs
    ) -> None:
        """Record metrics for a single epoch.
        
        Args:
            epoch: Epoch number (1-indexed).
            train_loss: Training loss (standard field).
            val_loss: Validation loss (standard field).
            learning_rate: Learning rate (standard field).
            epoch_time: Time taken for this epoch in seconds (standard field).
            **kwargs: Additional metrics to record (must be in self.metrics list).
        """
        self.current_epoch = epoch
        self.history['epoch'].append(epoch)
        
        # Record standard metrics
        if train_loss is not None and 'train_loss' in self.metrics:
            self.history['train_loss'].append(train_loss)
        if val_loss is not None and 'val_loss' in self.metrics:
            self.history['val_loss'].append(val_loss)
        if learning_rate is not None and 'learning_rate' in self.metrics:
            self.history['learning_rate'].append(learning_rate)
        
        # Record epoch_time as standard field
        if epoch_time is not None and 'epoch_time' in self.metrics:
            self.history['epoch_time'].append(epoch_time)
            self.epoch_times.append(epoch_time)
        
        # Record additional metrics (only if in metrics list)
        for key, value in kwargs.items():
            if key in self.metrics:
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
    
    # Advanced APIs (new)
    
    def set_best(
        self,
        metric: str,
        value: float,
        epoch: int,
        is_min: bool = True
    ) -> bool:
        """Set best metric value.
        
        **New Method:**
        - Tracks best epoch and value for each metric
        - Records improve event
        
        Args:
            metric: Metric name.
            value: Metric value.
            epoch: Epoch number.
            is_min: If True, lower is better. If False, higher is better.
        
        Returns:
            True if this is a new best, False otherwise.
        """
        is_improved = False
        
        if metric not in self.best_metrics:
            # First time seeing this metric
            is_improved = True
            self.best_metrics[metric] = {'value': value, 'epoch': epoch, 'is_min': is_min}
        else:
            current_best = self.best_metrics[metric]
            if is_min:
                if value < current_best['value']:
                    is_improved = True
                    current_best['value'] = value
                    current_best['epoch'] = epoch
            else:
                if value > current_best['value']:
                    is_improved = True
                    current_best['value'] = value
                    current_best['epoch'] = epoch
        
        if is_improved:
            self._add_event(
                'improve',
                epoch=epoch,
                data={'metric': metric, 'value': value, 'is_min': is_min}
            )
        
        return is_improved
    
    def mark_early_stopping(
        self,
        epoch: int,
        patience: int,
        reason: Optional[str] = None
    ) -> None:
        """Mark early stopping.
        
        **New Method:**
        - Records early stopping event
        - Stores early stopping metadata
        
        Args:
            epoch: Epoch where training stopped.
            patience: Patience value used.
            reason: Optional reason for early stopping.
        """
        self.early_stopping = {
            'epoch': epoch,
            'patience': patience,
            'reason': reason
        }
        self._add_event(
            'early_stop',
            epoch=epoch,
            data={'patience': patience, 'reason': reason}
        )
    
    def record_lr_change(
        self,
        epoch: int,
        old_lr: float,
        new_lr: float
    ) -> None:
        """Record learning rate change.
        
        **New Method:**
        - Records learning rate change event
        
        Args:
            epoch: Epoch number.
            old_lr: Old learning rate.
            new_lr: New learning rate.
        """
        self._add_event(
            'lr_change',
            epoch=epoch,
            data={'old_lr': old_lr, 'new_lr': new_lr}
        )
    
    def record_checkpoint(
        self,
        epoch: int,
        checkpoint_type: str = 'regular',
        **kwargs
    ) -> None:
        """Record checkpoint save.
        
        **New Method:**
        - Records checkpoint save event
        
        Args:
            epoch: Epoch number.
            checkpoint_type: Type of checkpoint ('regular', 'best', 'last').
            **kwargs: Additional checkpoint metadata.
        """
        self._add_event(
            'checkpoint',
            epoch=epoch,
            data={'checkpoint_type': checkpoint_type, **kwargs}
        )
    
    def _add_event(self, event_type: str, epoch: int, data: Dict[str, Any]) -> None:
        """Add event to event log.
        
        Args:
            event_type: Type of event.
            epoch: Epoch number.
            data: Event data.
        """
        event = TrainingEvent(
            type=event_type,
            epoch=epoch,
            timestamp=time.time(),
            data=data
        )
        self.events.append(event)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Get training summary as dictionary.
        
        **New Method:**
        - Provides high-level summary for Runner/metadata
        
        Returns:
            Dictionary with training summary.
        """
        summary = {
            'total_epochs': len(self.history['epoch']),
            'training_duration_seconds': self.training_duration_seconds,
            'current_epoch': self.current_epoch,
            'best_metrics': self.best_metrics.copy(),
        }
        
        if self.early_stopping:
            summary['early_stopping'] = self.early_stopping.copy()
        
        # Add latest metric values
        summary['latest_metrics'] = {}
        for metric in self.metrics:
            if metric in self.history and len(self.history[metric]) > 0:
                summary['latest_metrics'][metric] = self.history[metric][-1]
        
        # Add event counts
        summary['event_counts'] = {}
        for event in self.events:
            event_type = event.type
            summary['event_counts'][event_type] = summary['event_counts'].get(event_type, 0) + 1
        
        return summary
    
    def to_dataframe(self) -> Optional[Any]:
        """Convert history to pandas DataFrame.
        
        **New Method:**
        - Converts history to DataFrame for visualization
        - Easy to plot with matplotlib/plotly
        
        Returns:
            pandas DataFrame if pandas is available, None otherwise.
        """
        if not PANDAS_AVAILABLE:
            import logging
            _logger = logging.getLogger(__name__)
            _logger.warning("pandas not available. Cannot convert to DataFrame.")
            return None
        
        # Ensure all lists have the same length
        max_len = len(self.history['epoch'])
        data = {}
        for key, values in self.history.items():
            if len(values) < max_len:
                # Pad with None
                data[key] = list(values) + [None] * (max_len - len(values))
            else:
                data[key] = values
        
        return pd.DataFrame(data)
    
    
    def get_history(self) -> Dict[str, List[Any]]:
        """Get the complete training history.
        
        Returns:
            Dictionary mapping metric names to lists of values.
        """
        return self.history.copy()
    
    def get_latest(self, metric: str) -> Optional[Any]:
        """Get the latest value for a metric.
        
        Args:
            metric: Name of the metric.
        
        Returns:
            Latest value, or None if not available.
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return None
        return self.history[metric][-1]
    
    @property
    def training_duration_seconds(self) -> float:
        """Get training duration in seconds.
        
        Returns:
            Training duration (precise if epoch_times available).
        """
        if self.start_time is None:
            return 0.0
        
        # Use sum of epoch times if available (more precise)
        if self.epoch_times:
            return sum(self.epoch_times)
        
        # Fallback to wall-clock time
        return time.time() - self.start_time
    
    def save(self, path: Path) -> None:
        """Save training history to JSON file.
        
        Args:
            path: Path to save the history JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        history_data = self.history.copy()
        history_data['training_duration_seconds'] = self.training_duration_seconds
        history_data['total_epochs'] = len(self.history['epoch'])
        history_data['expected_metrics'] = self.metrics
        
        # Add advanced tracking data
        history_data['best_metrics'] = self.best_metrics
        if self.early_stopping:
            history_data['early_stopping'] = self.early_stopping
        
        # Add events (convert to dict for JSON serialization)
        history_data['events'] = [
            {
                'type': event.type,
                'epoch': event.epoch,
                'timestamp': event.timestamp,
                'data': event.data
            }
            for event in self.events
        ]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> "TrainingHistory":
        """Load training history from JSON file.
        
        Args:
            path: Path to the history JSON file.
        
        Returns:
            TrainingHistory instance with loaded data.
        """
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract metrics
        special_keys = {
            'epoch', 'training_duration_seconds', 'total_epochs',
            'expected_metrics', 'best_metrics', 'early_stopping', 'events'
        }
        
        if 'expected_metrics' in data:
            metrics = data['expected_metrics']
        else:
            # Fallback: infer from keys
            metrics = [k for k in data.keys() if k not in special_keys]
        
        instance = cls(metrics=metrics)
        
        # Load history
        instance.history = {
            'epoch': data.get('epoch', []),
            **{metric: data.get(metric, []) for metric in metrics if metric in data}
        }
        
        # Restore epoch times
        if 'epoch_time' in instance.history:
            instance.epoch_times = instance.history['epoch_time'].copy()
        
        # Restore metadata
        if 'epoch' in data and len(data['epoch']) > 0:
            instance.current_epoch = data['epoch'][-1]
        if 'training_duration_seconds' in data:
            instance.start_time = time.time() - data['training_duration_seconds']
        
        # Restore advanced tracking
        if 'best_metrics' in data:
            instance.best_metrics = data['best_metrics']
        if 'early_stopping' in data:
            instance.early_stopping = data['early_stopping']
        
        # Restore events
        if 'events' in data:
            instance.events = [
                TrainingEvent(
                    type=e['type'],
                    epoch=e['epoch'],
                    timestamp=e['timestamp'],
                    data=e['data']
                )
                for e in data['events']
            ]
        
        return instance
    
    def __len__(self) -> int:
        """Return the number of recorded epochs."""
        return len(self.history['epoch'])

