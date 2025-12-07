"""Progress logging utility for model training.

Features:
- Event-based callbacks for better integration with Runners
- Uses MetricSchema for unified metric names
- Pluggable handlers for custom output handling
- Path management via bind_files() method (Runner manages paths)
- tqdm control delegated to CLI/Runner
"""

from typing import Optional, Dict, Any, List, Callable
import sys
import time
from pathlib import Path
import logging

try:
    from src.training.metrics import MetricSchema
    METRIC_SCHEMA_AVAILABLE = True
except ImportError:
    METRIC_SCHEMA_AVAILABLE = False

_logger = logging.getLogger(__name__)


class ProgressLogger:
    """Log training progress with event-based interface.
    
    **Key Improvements:**
    - Event-based callbacks (on_training_start, on_epoch, etc.)
    - Pluggable handlers for custom output handling
    - Path management via bind_files() (Runner manages paths)
    - Uses MetricSchema for unified metric names
    - tqdm control delegated to CLI/Runner
    
    **Event Types:**
    - training_start: Training begins
    - epoch: Epoch completed
    - metric_improved: Metric improved
    - early_stopping: Early stopping triggered
    - training_complete: Training finished
    """
    
    def __init__(
        self,
        handlers: Optional[List[Callable]] = None,
        flush_interval: int = 10,
        max_log_size_mb: float = 100.0,
        use_metric_schema: bool = True
    ):
        """Initialize progress logger.
        
        - No log_file/detailed_log_file in __init__ (use bind_files())
        - No use_tqdm in __init__ (delegated to CLI/Runner)
        - Added handlers parameter for pluggable output handling
        - Added use_metric_schema for MetricSchema integration
        
        Args:
            handlers: List of handler functions called on each event.
                Each handler receives: (event_type: str, **kwargs)
            flush_interval: Flush output every N messages (default: 10).
            max_log_size_mb: Maximum log file size in MB before rotation (default: 100 MB).
            use_metric_schema: Whether to use MetricSchema for metric validation.
        """
        self.handlers: List[Callable] = handlers or []
        self.flush_interval = max(1, flush_interval)
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        self.message_count = 0
        self.pending_messages = 0
        
        # File paths (managed by Runner via bind_files)
        self.log_file: Optional[Path] = None
        self.detailed_log_file: Optional[Path] = None
        
        # Metric schema integration
        self.use_metric_schema = use_metric_schema and METRIC_SCHEMA_AVAILABLE
        if self.use_metric_schema:
            # Get standard metrics from schema
            self.standard_metrics = MetricSchema.STANDARD_METRICS
        else:
            self.standard_metrics = ['train_loss', 'val_loss', 'learning_rate', 'epoch_time']
    
    def bind_files(
        self,
        brief_path: Optional[Path] = None,
        detailed_path: Optional[Path] = None
    ) -> None:
        """Bind log file paths (called by Runner to manage paths).
        
        **New Method:**
        - Path management moved from __init__ to this method
        - Allows Runner to set paths after logger creation
        - Runner is responsible for path management
        - Immediately creates log files with headers to ensure they exist
        
        Args:
            brief_path: Path to brief log file (training.log).
            detailed_path: Path to detailed log file (training_detailed.log).
        """
        if brief_path:
            self.log_file = Path(brief_path)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            # Create log file immediately with header
            if not self.log_file.exists():
                from datetime import datetime
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 70 + "\n\n")
        if detailed_path:
            self.detailed_log_file = Path(detailed_path)
            self.detailed_log_file.parent.mkdir(parents=True, exist_ok=True)
            # Create detailed log file immediately with header
            if not self.detailed_log_file.exists():
                from datetime import datetime
                with open(self.detailed_log_file, 'w', encoding='utf-8') as f:
                    f.write(f"Detailed Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 70 + "\n\n")
    
    def add_handler(self, handler: Callable) -> None:
        """Add a handler function.
        
        Args:
            handler: Handler function that receives (event_type: str, **kwargs).
        """
        if handler not in self.handlers:
            self.handlers.append(handler)
    
    def remove_handler(self, handler: Callable) -> None:
        """Remove a handler function.
        
        Args:
            handler: Handler function to remove.
        """
        if handler in self.handlers:
            self.handlers.remove(handler)
    
    def _emit_event(self, event_type: str, **kwargs) -> None:
        """Emit an event to all handlers.
        
        Args:
            event_type: Type of event (training_start, epoch, etc.).
            **kwargs: Event data.
        """
        for handler in self.handlers:
            try:
                handler(event_type=event_type, **kwargs)
            except Exception as e:
                _logger.debug(f"Handler failed for event {event_type}: {e}")
    
    def log(self, message: str, flush: bool = False, detailed: bool = False) -> None:
        """Log a message.
        
        Args:
            message: Message to log.
            flush: Whether to flush immediately.
            detailed: If True, only write to detailed log. If False, write to both.
        """
        self.message_count += 1
        self.pending_messages += 1
        
        should_flush = flush or (self.pending_messages >= self.flush_interval)
        
        # Print to stdout
        print(message, flush=should_flush)
        
        # Write to brief log (if not detailed-only message)
        if self.log_file and not detailed:
            try:
                self._rotate_log_if_needed(self.log_file)
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
                    if should_flush:
                        f.flush()
            except Exception as e:
                _logger.debug(f"Failed to write to log file: {e}")
        
        # Write to detailed log (all messages)
        if self.detailed_log_file:
            try:
                self._rotate_log_if_needed(self.detailed_log_file)
                with open(self.detailed_log_file, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
                    if should_flush:
                        f.flush()
            except Exception as e:
                _logger.debug(f"Failed to write to detailed log file: {e}")
        
        if should_flush:
            self.pending_messages = 0
            sys.stdout.flush()
    
    def _rotate_log_if_needed(self, log_path: Path) -> None:
        """Rotate log file if it exceeds maximum size."""
        if not log_path.exists():
            return
        
        try:
            file_size = log_path.stat().st_size
            if file_size >= self.max_log_size_bytes:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                rotated_path = log_path.parent / f"{log_path.stem}_{timestamp}{log_path.suffix}"
                log_path.rename(rotated_path)
                _logger.info(f"Rotated log file {log_path} -> {rotated_path} (size: {file_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            _logger.debug(f"Failed to rotate log file {log_path}: {e}")
    
    # Event-based interface (new)
    
    def on_training_start(
        self,
        model_name: str,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Event: Training starts.
        
        Emits event to handlers and logs training information.
        
        Args:
            model_name: Name of the model.
            device: Device being used (e.g., "cuda", "cpu").
            config: Model configuration dictionary.
        """
        # Emit event
        self._emit_event(
            event_type='training_start',
            model_name=model_name,
            device=device,
            config=config
        )
        
        self.log(f"\n  🚀 Starting {model_name} training", flush=True)
        if device:
            self.log(f"     Device: {device}", flush=True)
        if config:
            for key, value in config.items():
                if isinstance(value, (int, float, str, bool)):
                    self.log(f"     {key}: {value}", flush=True)
    
    def on_epoch(
        self,
        epoch: int,
        total_epochs: int,
        metrics: Optional[Dict[str, Any]] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        epoch_time: Optional[float] = None,
        eta: Optional[float] = None,
        use_tqdm: bool = False,
        **kwargs
    ) -> None:
        """Event: Epoch completed.
        
        Uses MetricSchema for metric validation and emits event to handlers.
        
        Args:
            epoch: Current epoch number.
            total_epochs: Total number of epochs.
            metrics: Dictionary of metrics (preferred way). If provided, other parameters are ignored.
            train_loss: Training loss (standard metric). Used if metrics is None.
            val_loss: Validation loss (standard metric). Used if metrics is None.
            learning_rate: Current learning rate (standard metric). Used if metrics is None.
            epoch_time: Time taken for this epoch in seconds (standard metric). Used if metrics is None.
            eta: Estimated time remaining in seconds.
            use_tqdm: Whether to use tqdm progress bar (delegated to CLI/Runner).
            **kwargs: Additional metrics (validated against MetricSchema if enabled).
        """
        # If metrics dict provided, extract individual metrics
        if metrics is not None:
            train_loss = metrics.get('train_loss', train_loss)
            val_loss = metrics.get('val_loss', val_loss)
            learning_rate = metrics.get('learning_rate', learning_rate)
            epoch_time = metrics.get('epoch_time', epoch_time)
            eta = metrics.get('eta', eta)
            kwargs.update({k: v for k, v in metrics.items() if k not in ['train_loss', 'val_loss', 'learning_rate', 'epoch_time', 'eta']})
        
        # Validate metrics if schema enabled
        if self.use_metric_schema:
            all_metrics = MetricSchema.all_metrics()
            unknown_metrics = [k for k in kwargs.keys() if k not in all_metrics]
            if unknown_metrics:
                _logger.warning(f"Unknown metrics: {unknown_metrics}. Consider registering via MetricRegistry.")
        
        # Emit event
        event_data = {
            'epoch': epoch,
            'total_epochs': total_epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'epoch_time': epoch_time,
            'eta': eta,
            **kwargs
        }
        self._emit_event(event_type='epoch', **event_data)
        
        # Build log message
        parts = [f"Epoch {epoch}/{total_epochs}"]
        
        # Standard metrics
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.6f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.6f}")
        if learning_rate is not None:
            parts.append(f"learning_rate={learning_rate:.6e}")
        if epoch_time is not None:
            parts.append(f"epoch_time={epoch_time:.2f}s")
        if eta is not None:
            parts.append(f"ETA={eta/60:.1f}m")
        
        # Additional metrics
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                parts.append(f"{key}={value:.6f}")
            else:
                parts.append(f"{key}={value}")
        
        message = "  " + " - ".join(parts)
        self.log(message, flush=True, detailed=True)
    
    def on_metric_improved(
        self,
        metric_name: str,
        current_value: float,
        best_value: float,
        epoch: Optional[int] = None
    ) -> None:
        """Event: Metric improved.
        
        Emits event to handlers and logs improvement information.
        
        Args:
            metric_name: Name of the metric.
            current_value: Current metric value.
            best_value: Best metric value so far.
            epoch: Current epoch number (optional).
        """
        # Emit event
        self._emit_event(
            event_type='metric_improved',
            metric_name=metric_name,
            current_value=current_value,
            best_value=best_value,
            epoch=epoch
        )
        
        self.log(f"  ✅ Improved! {metric_name}: {current_value:.6f} (Best: {best_value:.6f})", flush=True)
    
    def on_early_stopping(
        self,
        epoch: int,
        patience: int,
        reason: Optional[str] = None
    ) -> None:
        """Event: Early stopping triggered.
        
        Emits event to handlers and logs early stopping information.
        
        Args:
            epoch: Epoch where training stopped.
            patience: Patience value used.
            reason: Optional reason for early stopping.
        """
        # Emit event
        self._emit_event(
            event_type='early_stopping',
            epoch=epoch,
            patience=patience,
            reason=reason
        )
        
        msg = f"  Early stopping at epoch {epoch} (patience={patience})"
        if reason:
            msg += f": {reason}"
        self.log(msg, flush=True)
    
    def on_training_complete(
        self,
        total_time: float,
        total_epochs: int,
        **kwargs
    ) -> None:
        """Event: Training completed.
        
        Emits event to handlers and logs completion information.
        
        Args:
            total_time: Total training time in seconds.
            total_epochs: Total number of epochs completed.
            **kwargs: Additional completion metadata.
        """
        # Emit event
        self._emit_event(
            event_type='training_complete',
            total_time=total_time,
            total_epochs=total_epochs,
            **kwargs
        )
        
        # Log completion
        self.log(f"  ✅ Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)", flush=True)
        self.log(f"     Total epochs: {total_epochs}", flush=True)
    
    # tqdm support (delegated to CLI/Runner)
    
    def get_tqdm(self, iterable, desc: str = "", use_tqdm: Optional[bool] = None, **kwargs):
        """Get a tqdm progress bar (control delegated to CLI/Runner).
        
        - use_tqdm parameter now passed at call time (not stored in instance)
        - CLI/Runner controls whether to use tqdm
        
        Args:
            iterable: Iterable to wrap.
            desc: Description for the progress bar.
            use_tqdm: Whether to use tqdm (default: auto-detect from isatty()).
            **kwargs: Additional arguments for tqdm.
        
        Returns:
            tqdm progress bar or the original iterable.
        """
        # Auto-detect if use_tqdm not specified
        if use_tqdm is None:
            use_tqdm = sys.stdout.isatty()
        
        if use_tqdm:
            try:
                from tqdm import tqdm
                kwargs.setdefault('leave', False)
                return tqdm(iterable, desc=desc, **kwargs)
            except ImportError:
                pass
        
        return iterable

