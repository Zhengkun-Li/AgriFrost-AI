"""Checkpoint management utility (Refactored with Runner path management).

This is a refactored version of CheckpointManager that:
- Paths managed by Runner (bind_dir() method)
- Standard naming (best.ckpt, last.ckpt)
- Enhanced resume mechanism
- Better integration with Runner
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import time
import pickle

_logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints with Runner path management.
    
    **Key Improvements:**
    - Path management via bind_dir() (Runner manages paths)
    - Standard naming (best.ckpt, last.ckpt)
    - Enhanced resume mechanism
    - Better integration with Runner
    
    **Standard Checkpoint Names:**
    - best.ckpt: Best model checkpoint
    - last.ckpt: Last checkpoint saved
    - checkpoint_epoch_{N}.ckpt: Regular checkpoints
    """
    
    def __init__(
        self,
        checkpoint_frequency: int = 10,
        save_best: bool = True,
        best_metric: str = "val_loss",
        best_mode: str = "min",
        keep_top_k: int = 3
    ):
        """Initialize checkpoint manager.
        
        - No checkpoint_dir in __init__ (use bind_dir())
        - Runner manages checkpoint directory
        
        Args:
            checkpoint_frequency: Save checkpoint every N epochs (0 = disabled).
            save_best: Whether to save the best model based on metric.
            best_metric: Metric name to use for determining best model.
            best_mode: "min" or "max" - whether lower or higher is better.
            keep_top_k: Number of top checkpoints to keep (default: 3, 0 = keep all).
        """
        self.checkpoint_frequency = checkpoint_frequency
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.keep_top_k = keep_top_k
        
        # Checkpoint directory (managed by Runner via bind_dir)
        self.checkpoint_dir: Optional[Path] = None
        
        # Internal state
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.checkpoint_count = 0
        self.checkpoint_history: List[Tuple[int, float]] = []
    
    def bind_dir(self, checkpoint_dir: Path) -> None:
        """Bind checkpoint directory (called by Runner to manage paths).
        
        **New Method:**
        - Path management moved from __init__ to this method
        - Allows Runner to set checkpoint directory after manager creation
        - Runner is responsible for path management
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def should_save_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint should be saved for this epoch.
        
        Args:
            epoch: Current epoch number.
        
        Returns:
            True if checkpoint should be saved.
        """
        if self.checkpoint_frequency <= 0:
            return False
        return epoch % self.checkpoint_frequency == 0
    
    def is_best(self, metric_value: float) -> bool:
        """Check if the current metric value is the best so far.
        
        Args:
            metric_value: Current metric value.
        
        Returns:
            True if this is the best value seen so far.
        """
        if self.best_value is None:
            return True
        
        if self.best_mode == "min":
            return metric_value < self.best_value
        else:
            return metric_value > self.best_value
    
    def update_best(self, epoch: int, metric_value: float) -> bool:
        """Update best metric value and epoch.
        
        Args:
            epoch: Current epoch number.
            metric_value: Current metric value.
        
        Returns:
            True if this is a new best value.
        """
        if self.is_best(metric_value):
            self.best_value = metric_value
            self.best_epoch = epoch
            return True
        return False
    
    def _ensure_cpu_compatible(self, model_state: Any) -> Any:
        """Ensure model state is CPU-compatible.
        
        Args:
            model_state: Model state to convert.
        
        Returns:
            CPU-compatible model state.
        """
        if isinstance(model_state, dict):
            try:
                import torch
                if any(isinstance(v, torch.Tensor) and v.is_cuda for v in model_state.values()):
                    return {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                            for k, v in model_state.items()}
            except ImportError:
                pass
        return model_state
    
    def save_checkpoint(
        self,
        epoch: int,
        model_state: Any,
        optimizer_state: Optional[Any] = None,
        scheduler_state: Optional[Any] = None,
        scaler_state: Optional[Any] = None,
        metrics: Optional[Dict[str, Any]] = None,
        training_history: Optional[Any] = None,
        custom_data: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save a checkpoint.
        
        **Standard Naming:**
        - Regular checkpoints: checkpoint_epoch_{epoch}.ckpt
        - Last checkpoint: last.ckpt (always updated)
        
        Args:
            epoch: Epoch number.
            model_state: Model state to save.
            optimizer_state: Optimizer state (optional).
            scheduler_state: Learning rate scheduler state (optional).
            scaler_state: Mixed precision scaler state (optional).
            metrics: Dictionary of current metrics.
            training_history: Training history object or dict.
            custom_data: Additional custom data to save.
        
        Returns:
            Path to the saved checkpoint file.
        
        Raises:
            ValueError: If checkpoint_dir is not set (call bind_dir() first).
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set. Call bind_dir() first.")
        
        # Ensure CPU compatibility
        model_state = self._ensure_cpu_compatible(model_state)
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.ckpt"
        
        checkpoint = {
            'epoch': epoch,
            'timestamp': time.time(),
            'model_state': model_state,
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state'] = optimizer_state
        if scheduler_state is not None:
            checkpoint['scheduler_state'] = scheduler_state
        if scaler_state is not None:
            checkpoint['scaler_state'] = scaler_state
        if metrics is not None:
            checkpoint['metrics'] = metrics
        if training_history is not None:
            if hasattr(training_history, 'get_history'):
                checkpoint['training_history'] = training_history.get_history()
            else:
                checkpoint['training_history'] = training_history
        if custom_data is not None:
            checkpoint.update(custom_data)
        
        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Update last.ckpt
        last_path = self.checkpoint_dir / "last.ckpt"
        with open(last_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.checkpoint_count += 1
        return checkpoint_path
    
    def save_best_checkpoint(
        self,
        epoch: int,
        model_state: Any,
        metric_value: float,
        **kwargs
    ) -> Optional[Path]:
        """Save checkpoint if this is the best model so far.
        
        **Standard Naming:**
        - Best checkpoint: best.ckpt (always updated)
        - Top-k checkpoints: best_top{N}_epoch{E}.ckpt
        
        Args:
            epoch: Epoch number.
            model_state: Model state to save.
            metric_value: Current metric value.
            **kwargs: Additional arguments passed to checkpoint.
        
        Returns:
            Path to saved checkpoint if saved, None otherwise.
        
        Raises:
            ValueError: If checkpoint_dir is not set (call bind_dir() first).
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set. Call bind_dir() first.")
        
        if not self.save_best:
            return None
        
        # Ensure CPU compatibility
        model_state = self._ensure_cpu_compatible(model_state)
        
        is_new_best = self.update_best(epoch, metric_value)
        
        # Always save best model (latest best)
        best_path = self.checkpoint_dir / "best.ckpt"
        checkpoint = {
            'epoch': epoch,
            'timestamp': time.time(),
            'model_state': model_state,
            'best_metric': self.best_metric,
            'best_value': metric_value,
            **kwargs
        }
        
        with open(best_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Track for top-k management
        if self.keep_top_k > 0:
            self.checkpoint_history.append((epoch, metric_value))
            
            # Sort by metric value (best first)
            if self.best_mode == "min":
                self.checkpoint_history.sort(key=lambda x: x[1])
            else:
                self.checkpoint_history.sort(key=lambda x: -x[1])
            
            # Keep only top-k
            if len(self.checkpoint_history) > self.keep_top_k:
                # Save top-k checkpoints
                for i, (e, v) in enumerate(self.checkpoint_history[:self.keep_top_k]):
                    if e != epoch:
                        top_k_path = self.checkpoint_dir / f"best_top{i+1}_epoch{e}.ckpt"
                        # Load existing checkpoint if available
                        existing_checkpoint = self.load_checkpoint(e)
                        if existing_checkpoint:
                            with open(top_k_path, 'wb') as f:
                                pickle.dump(existing_checkpoint, f)
                
                # Remove checkpoints not in top-k
                for e, v in self.checkpoint_history[self.keep_top_k:]:
                    for p in self.checkpoint_dir.glob(f"best_top*_epoch{e}.ckpt"):
                        try:
                            p.unlink()
                        except Exception:
                            pass
                
                # Update history to only top-k
                self.checkpoint_history = self.checkpoint_history[:self.keep_top_k]
        
        if is_new_best:
            return best_path
        return None
    
    def get_best_checkpoint_path(self) -> Path:
        """Get path to best checkpoint (standard naming).
        
        Returns:
            Path to best.ckpt.
        
        Raises:
            ValueError: If checkpoint_dir is not set.
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set. Call bind_dir() first.")
        return self.checkpoint_dir / "best.ckpt"
    
    def get_last_checkpoint_path(self) -> Path:
        """Get path to last checkpoint (standard naming).
        
        Returns:
            Path to last.ckpt.
        
        Raises:
            ValueError: If checkpoint_dir is not set.
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set. Call bind_dir() first.")
        return self.checkpoint_dir / "last.ckpt"
    
    def get_checkpoint_path(self, epoch: Optional[int] = None) -> Path:
        """Get path to a checkpoint file.
        
        Args:
            epoch: Epoch number. If None, returns best model path.
        
        Returns:
            Path to checkpoint file.
        
        Raises:
            ValueError: If checkpoint_dir is not set.
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set. Call bind_dir() first.")
        
        if epoch is None:
            return self.get_best_checkpoint_path()
        return self.checkpoint_dir / f"checkpoint_epoch_{epoch}.ckpt"
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoint files.
        
        Returns:
            List of checkpoint file paths, sorted by epoch.
        
        Raises:
            ValueError: If checkpoint_dir is not set.
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set. Call bind_dir() first.")
        
        checkpoints = []
        for path in self.checkpoint_dir.glob("checkpoint_epoch_*.ckpt"):
            checkpoints.append(path)
        return sorted(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    
    def load_checkpoint(self, epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Load a checkpoint from disk.
        
        **Standard Naming:**
        - If epoch is None: loads best.ckpt
        - Otherwise: loads checkpoint_epoch_{epoch}.ckpt
        
        Args:
            epoch: Epoch number to load. If None, loads the best model.
        
        Returns:
            Dictionary containing checkpoint data, or None if not found.
        
        Raises:
            ValueError: If checkpoint_dir is not set.
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set. Call bind_dir() first.")
        
        checkpoint_path = self.get_checkpoint_path(epoch)
        
        if not checkpoint_path.exists():
            return None
        
        return self.load_checkpoint_from_path(checkpoint_path)
    
    def load_checkpoint_from_path(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """Load checkpoint from a specific path.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        
        Returns:
            Dictionary containing checkpoint data, or None if not found.
        """
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Ensure model_state is CPU-compatible
            if 'model_state' in checkpoint:
                checkpoint['model_state'] = self._ensure_cpu_compatible(checkpoint['model_state'])
            
            # Update internal state
            if 'epoch' in checkpoint:
                self.best_epoch = checkpoint.get('epoch', 0)
            if 'best_value' in checkpoint:
                self.best_value = checkpoint.get('best_value')
            
            return checkpoint
        except Exception as e:
            _logger.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return None
    
    def resume_training(
        self,
        from_best: bool = False,
        from_last: bool = True,
        epoch: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Resume training from a checkpoint (enhanced).
        
        **Enhanced Resume Mechanism:**
        - from_best: Resume from best.ckpt
        - from_last: Resume from last.ckpt (default)
        - epoch: Resume from specific epoch checkpoint
        
        Args:
            from_best: Whether to resume from best checkpoint.
            from_last: Whether to resume from last checkpoint (default).
            epoch: Specific epoch to resume from (overrides from_best/from_last).
        
        Returns:
            Dictionary with resume information, or None if checkpoint not found.
        
        Raises:
            ValueError: If checkpoint_dir is not set or conflicting options.
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set. Call bind_dir() first.")
        
        # Determine checkpoint path
        if epoch is not None:
            checkpoint_path = self.get_checkpoint_path(epoch)
        elif from_best:
            checkpoint_path = self.get_best_checkpoint_path()
        elif from_last:
            checkpoint_path = self.get_last_checkpoint_path()
        else:
            raise ValueError("Must specify from_best, from_last, or epoch")
        
        checkpoint = self.load_checkpoint_from_path(checkpoint_path)
        if checkpoint is None:
            return None
        
        # Return full resume information
        resume_info = {
            'epoch': checkpoint.get('epoch', 0),
            'model_state': checkpoint.get('model_state'),
            'optimizer_state': checkpoint.get('optimizer_state'),
            'scheduler_state': checkpoint.get('scheduler_state'),
            'scaler_state': checkpoint.get('scaler_state'),
            'metrics': checkpoint.get('metrics', {}),
            'training_history': checkpoint.get('training_history'),
            'best_value': checkpoint.get('best_value'),
            'best_epoch': checkpoint.get('epoch', 0),
            'checkpoint_path': str(checkpoint_path),
        }
        
        # Update internal state
        self.best_epoch = resume_info['epoch']
        self.best_value = resume_info.get('best_value')
        
        return resume_info
    
    def get_checkpoint_metadata(self, epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get checkpoint metadata without loading full checkpoint.
        
        Args:
            epoch: Epoch number. If None, returns best model metadata.
        
        Returns:
            Dictionary with checkpoint metadata, or None if not found.
        """
        checkpoint = self.load_checkpoint(epoch)
        if checkpoint is None:
            return None
        
        metadata = {
            'epoch': checkpoint.get('epoch'),
            'timestamp': checkpoint.get('timestamp'),
            'best_metric': checkpoint.get('best_metric'),
            'best_value': checkpoint.get('best_value'),
            'has_optimizer_state': 'optimizer_state' in checkpoint,
            'has_scheduler_state': 'scheduler_state' in checkpoint,
            'has_scaler_state': 'scaler_state' in checkpoint,
            'has_metrics': 'metrics' in checkpoint,
            'has_training_history': 'training_history' in checkpoint,
        }
        
        if 'metrics' in checkpoint:
            metadata['metrics'] = checkpoint['metrics']
        
        return metadata
    
    def get_info(self) -> Dict[str, Any]:
        """Get checkpoint manager information.
        
        Returns:
            Dictionary with checkpoint manager state.
        """
        info = {
            'checkpoint_dir': str(self.checkpoint_dir) if self.checkpoint_dir else None,
            'checkpoint_frequency': self.checkpoint_frequency,
            'save_best': self.save_best,
            'best_metric': self.best_metric,
            'best_mode': self.best_mode,
            'keep_top_k': self.keep_top_k,
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'checkpoint_count': self.checkpoint_count,
            'top_k_checkpoints': self.checkpoint_history[:self.keep_top_k] if self.keep_top_k > 0 else [],
        }
        
        # Add checkpoint file paths if directory is set
        if self.checkpoint_dir and self.checkpoint_dir.exists():
            info['best_checkpoint_exists'] = self.get_best_checkpoint_path().exists()
            info['last_checkpoint_exists'] = self.get_last_checkpoint_path().exists()
            info['regular_checkpoints'] = len(self.list_checkpoints())
        
        return info

