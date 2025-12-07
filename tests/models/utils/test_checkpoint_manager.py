"""Unit tests for CheckpointManager."""

import pytest
import tempfile
import shutil
import pickle
from pathlib import Path
from src.models.utils.checkpoint_manager import CheckpointManager


class TestCheckpointManager:
    """Test CheckpointManager functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def checkpoint_mgr(self, temp_dir):
        """Create CheckpointManager instance."""
        mgr = CheckpointManager(
            checkpoint_frequency=5,
            save_best=True,
            best_metric="val_loss",
            best_mode="min",
            keep_top_k=3
        )
        mgr.bind_dir(temp_dir / "checkpoints")
        return mgr
    
    def test_init(self, checkpoint_mgr):
        """Test CheckpointManager initialization."""
        assert checkpoint_mgr.checkpoint_frequency == 5
        assert checkpoint_mgr.save_best is True
        assert checkpoint_mgr.best_metric == "val_loss"
        assert checkpoint_mgr.best_mode == "min"
        assert checkpoint_mgr.keep_top_k == 3
        assert checkpoint_mgr.best_value is None
        assert checkpoint_mgr.best_epoch == 0
        assert checkpoint_mgr.checkpoint_count == 0
    
    def test_should_save_checkpoint(self, checkpoint_mgr):
        """Test should_save_checkpoint."""
        assert checkpoint_mgr.should_save_checkpoint(5) is True
        assert checkpoint_mgr.should_save_checkpoint(10) is True
        assert checkpoint_mgr.should_save_checkpoint(7) is False
        assert checkpoint_mgr.should_save_checkpoint(0) is False
    
    def test_is_best_min(self, checkpoint_mgr):
        """Test is_best for min mode."""
        # First value is always best
        assert checkpoint_mgr.is_best(0.5) is True
        
        # Lower value is better
        checkpoint_mgr.best_value = 0.5
        assert checkpoint_mgr.is_best(0.4) is True
        assert checkpoint_mgr.is_best(0.6) is False
    
    def test_is_best_max(self, temp_dir):
        """Test is_best for max mode."""
        checkpoint_mgr = CheckpointManager(best_mode="max")
        checkpoint_mgr.bind_dir(temp_dir / "checkpoints")
        
        # First value is always best
        assert checkpoint_mgr.is_best(0.5) is True
        
        # Higher value is better
        checkpoint_mgr.best_value = 0.5
        assert checkpoint_mgr.is_best(0.6) is True
        assert checkpoint_mgr.is_best(0.4) is False
    
    def test_save_checkpoint_gpu_cpu_compatibility(self, checkpoint_mgr):
        """Test checkpoint save with GPU/CPU compatibility."""
        # Simulate GPU tensor (if torch available)
        try:
            import torch
            model_state = {
                'weight': torch.tensor([1.0, 2.0, 3.0]).cuda() if torch.cuda.is_available() else torch.tensor([1.0, 2.0, 3.0]),
                'bias': torch.tensor([0.5])
            }
        except ImportError:
            # If torch not available, use simple dict
            model_state = {'weight': [1.0, 2.0, 3.0], 'bias': [0.5]}
        
        checkpoint_path = checkpoint_mgr.save_checkpoint(
            epoch=5,
            model_state=model_state,
            optimizer_state={'lr': 0.001},
            metrics={'val_loss': 0.5}
        )
        
        assert checkpoint_path.exists()
        
        # Load and verify CPU compatibility
        loaded = checkpoint_mgr.load_checkpoint(epoch=5)
        assert loaded is not None
        assert loaded['epoch'] == 5
        assert 'model_state' in loaded
        assert 'metrics' in loaded
    
    def test_save_best_checkpoint(self, checkpoint_mgr):
        """Test save_best_checkpoint."""
        model_state = {'weight': [1.0, 2.0, 3.0]}
        
        # Save first best
        path1 = checkpoint_mgr.save_best_checkpoint(
            epoch=1,
            model_state=model_state,
            metric_value=0.5
        )
        assert path1 is not None
        assert checkpoint_mgr.best_value == 0.5
        assert checkpoint_mgr.best_epoch == 1
        
        # Save better value
        path2 = checkpoint_mgr.save_best_checkpoint(
            epoch=2,
            model_state=model_state,
            metric_value=0.4
        )
        assert path2 is not None
        assert checkpoint_mgr.best_value == 0.4
        assert checkpoint_mgr.best_epoch == 2
        
        # Save worse value (should not update best)
        path3 = checkpoint_mgr.save_best_checkpoint(
            epoch=3,
            model_state=model_state,
            metric_value=0.6
        )
        assert path3 is None  # Not saved as best
        assert checkpoint_mgr.best_value == 0.4  # Still best
        assert checkpoint_mgr.best_epoch == 2
    
    def test_best_k_checkpoints(self, checkpoint_mgr):
        """Test best-k checkpoint management."""
        model_state = {'weight': [1.0, 2.0, 3.0]}
        
        # Save multiple checkpoints
        values = [0.5, 0.4, 0.3, 0.35, 0.25]
        for i, val in enumerate(values):
            checkpoint_mgr.save_best_checkpoint(
                epoch=i + 1,
                model_state=model_state,
                metric_value=val
            )
        
        # Should keep top-k (keep_top_k=3, min mode = keep lowest 3)
        assert len(checkpoint_mgr.checkpoint_history) <= checkpoint_mgr.keep_top_k
    
    def test_resume_training(self, checkpoint_mgr):
        """Test resume_training."""
        model_state = {'weight': [1.0, 2.0, 3.0]}
        optimizer_state = {'lr': 0.001}
        scheduler_state = {'step': 5}
        
        # Save checkpoint
        checkpoint_mgr.save_checkpoint(
            epoch=10,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            metrics={'val_loss': 0.5}
        )
        
        # Resume
        resume_info = checkpoint_mgr.resume_training(epoch=10)
        assert resume_info is not None
        assert resume_info['epoch'] == 10
        assert 'model_state' in resume_info
        assert 'optimizer_state' in resume_info
        assert 'scheduler_state' in resume_info
        assert resume_info['model_state'] == model_state
    
    def test_get_checkpoint_metadata(self, checkpoint_mgr):
        """Test get_checkpoint_metadata."""
        model_state = {'weight': [1.0, 2.0, 3.0]}
        
        checkpoint_mgr.save_checkpoint(
            epoch=5,
            model_state=model_state,
            optimizer_state={'lr': 0.001},
            metrics={'val_loss': 0.5}
        )
        
        metadata = checkpoint_mgr.get_checkpoint_metadata(epoch=5)
        assert metadata is not None
        assert metadata['epoch'] == 5
        assert metadata['has_optimizer_state'] is True
        assert metadata['has_scheduler_state'] is False
        assert metadata['has_metrics'] is True
    
    def test_list_checkpoints(self, checkpoint_mgr):
        """Test list_checkpoints."""
        model_state = {'weight': [1.0, 2.0, 3.0]}
        
        # Save multiple checkpoints
        for epoch in [5, 10, 15]:
            checkpoint_mgr.save_checkpoint(
                epoch=epoch,
                model_state=model_state
            )
        
        checkpoints = checkpoint_mgr.list_checkpoints()
        assert len(checkpoints) == 3
        assert all(p.exists() for p in checkpoints)
    
    def test_get_checkpoint_path(self, checkpoint_mgr):
        """Test get_checkpoint_path."""
        # Best model path
        best_path = checkpoint_mgr.get_checkpoint_path()
        assert best_path.name == "best_model.pth"
        
        # Specific epoch path
        epoch_path = checkpoint_mgr.get_checkpoint_path(epoch=5)
        assert epoch_path.name == "checkpoint_epoch_5.pth"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

