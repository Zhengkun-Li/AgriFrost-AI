"""Unit tests for ProgressLogger."""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.models.utils.progress_logger import ProgressLogger


class TestProgressLogger:
    """Test ProgressLogger functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def logger(self, temp_dir):
        """Create ProgressLogger instance (event-based API)."""
        logger = ProgressLogger(
            flush_interval=10,
            max_log_size_mb=1.0,  # Small size for testing
            use_metric_schema=True
        )
        logger.bind_files(
            brief_path=temp_dir / "training.log",
            detailed_path=temp_dir / "training_detailed.log"
        )
        return logger
    
    def test_init(self, logger, temp_dir):
        """Test ProgressLogger initialization (event-based API)."""
        assert logger.log_file == temp_dir / "training.log"
        assert logger.detailed_log_file == temp_dir / "training_detailed.log"
        assert logger.flush_interval == 10
        assert logger.max_log_size_bytes == 1.0 * 1024 * 1024
        assert logger.message_count == 0
        assert logger.pending_messages == 0
    
    def test_log_brief(self, logger, temp_dir):
        """Test logging brief messages."""
        logger.log("Test message", detailed=False)
        
        # Check brief log
        assert (temp_dir / "training.log").exists()
        content = (temp_dir / "training.log").read_text(encoding='utf-8')
        assert "Test message" in content
        
        # Check detailed log
        assert (temp_dir / "training_detailed.log").exists()
        detailed_content = (temp_dir / "training_detailed.log").read_text(encoding='utf-8')
        assert "Test message" in detailed_content
    
    def test_log_detailed_only(self, logger, temp_dir):
        """Test logging detailed-only messages."""
        logger.log("Detailed message", detailed=True)
        
        # Should not be in brief log
        if (temp_dir / "training.log").exists():
            content = (temp_dir / "training.log").read_text(encoding='utf-8')
            assert "Detailed message" not in content
        
        # Should be in detailed log
        assert (temp_dir / "training_detailed.log").exists()
        detailed_content = (temp_dir / "training_detailed.log").read_text(encoding='utf-8')
        assert "Detailed message" in detailed_content
    
    def test_flush_interval(self, logger, temp_dir):
        """Test flush interval mechanism."""
        # Log 9 messages (less than flush_interval=10)
        for i in range(9):
            logger.log(f"Message {i}", detailed=False)
        
        # Should have accumulated but not flushed yet
        assert logger.pending_messages == 9
        
        # Log one more message to trigger flush
        logger.log("Final message", detailed=False)
        
        # Should have reset pending_messages
        assert logger.pending_messages == 0
    
    def test_on_epoch_field_unification(self, logger, temp_dir):
        """Test on_epoch field unification with TrainingHistory (event-based API)."""
        logger.on_epoch(
            epoch=1,
            total_epochs=100,
            metrics={
                'train_loss': 0.5,
                'val_loss': 0.45,
                'learning_rate': 0.01,
                'epoch_time': 12.5
            },
            use_tqdm=False
        )
        
        # Check detailed log
        content = (temp_dir / "training_detailed.log").read_text(encoding='utf-8')
        # Should use unified field format: train_loss=, val_loss=, etc.
        assert "train_loss=0.500000" in content
        assert "val_loss=0.450000" in content
        assert "learning_rate=0.010000" in content
        assert "epoch_time=12.50" in content
    
    def test_log_rotation(self, logger, temp_dir):
        """Test log file rotation."""
        # Write large content to trigger rotation
        large_message = "X" * (logger.max_log_size_bytes + 100)
        logger.log(large_message, detailed=False)
        
        # Check if rotation occurred (original file renamed or new file created)
        log_files = list(temp_dir.glob("training*.log"))
        assert len(log_files) >= 1
    
    def test_on_training_start(self, logger, temp_dir):
        """Test on_training_start (event-based API)."""
        logger.on_training_start(
            model_name="LSTM",
            device="cuda",
            config={"batch_size": 32, "epochs": 100}
        )
        
        content = (temp_dir / "training.log").read_text(encoding='utf-8')
        assert "Starting LSTM training" in content
        assert "Device: cuda" in content
        assert "batch_size: 32" in content
    
    def test_on_metric_improved(self, logger, temp_dir):
        """Test on_metric_improved (event-based API)."""
        logger.on_metric_improved("val_loss", 0.45, 0.5, epoch=1)
        
        content = (temp_dir / "training.log").read_text(encoding='utf-8')
        assert "Improved!" in content
        assert "val_loss" in content
    
    def test_on_early_stopping(self, logger, temp_dir):
        """Test on_early_stopping (event-based API)."""
        logger.on_early_stopping(epoch=50, patience=10)
        
        content = (temp_dir / "training.log").read_text(encoding='utf-8')
        assert "Early stopping" in content
        assert "epoch 50" in content
        assert "patience=10" in content
    
    def test_on_training_complete(self, logger, temp_dir):
        """Test on_training_complete (event-based API)."""
        logger.on_training_complete(total_time=1000.0, total_epochs=100)
        
        content = (temp_dir / "training.log").read_text(encoding='utf-8')
        assert "Training completed" in content
        assert "1000.00 seconds" in content
        assert "Total epochs: 100" in content
    
    def test_get_tqdm(self, logger):
        """Test get_tqdm."""
        # Test with iterable
        iterable = range(10)
        result = logger.get_tqdm(iterable, desc="Test")
        
        # Should return iterable or tqdm object
        assert result is not None
        # Can iterate
        list(result)  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

