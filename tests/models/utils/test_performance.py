"""Performance benchmark tests for model utils tools."""

import pytest
import tempfile
import time
import shutil
from pathlib import Path
from src.models.utils.progress_logger import ProgressLogger
from src.models.utils.training_history import TrainingHistory


class TestPerformance:
    """Performance benchmark tests."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_flush_optimization_performance(self, temp_dir):
        """Test flush optimization performance (reduced flush operations)."""
        # Old style: flush_interval=1 (flush every message)
        logger_old = ProgressLogger(flush_interval=1)
        logger_old.bind_files(brief_path=temp_dir / "old.log")
        
        # New style: flush_interval=10 (flush every 10 messages)
        logger_new = ProgressLogger(flush_interval=10)
        logger_new.bind_files(brief_path=temp_dir / "new.log")
        
        # Time old style
        start = time.time()
        for i in range(100):
            logger_old.log(f"Message {i}")
        old_time = time.time() - start
        
        # Time new style
        start = time.time()
        for i in range(100):
            logger_new.log(f"Message {i}")
        new_time = time.time() - start
        
        # New style should be faster (fewer flush operations)
        # Note: This is a relative test, actual times depend on system
        print(f"\n   Old style (flush_interval=1): {old_time:.4f}s")
        print(f"   New style (flush_interval=10): {new_time:.4f}s")
        print(f"   Speedup: {old_time / new_time:.2f}x")
    
    def test_log_rotation_performance(self, temp_dir):
        """Test log rotation performance."""
        logger = ProgressLogger(max_log_size_mb=0.001)  # 1KB for testing
        logger.bind_files(brief_path=temp_dir / "rotating.log")
        
        # Write enough to trigger rotation
        large_message = "X" * 2000  # 2KB
        start = time.time()
        logger.log(large_message, detailed=False)
        rotation_time = time.time() - start
        
        # Rotation should complete quickly (< 1 second)
        assert rotation_time < 1.0
        print(f"\n   Log rotation time: {rotation_time:.4f}s")
    
    def test_training_history_duration_precision(self, temp_dir):
        """Test training history duration precision."""
        history = TrainingHistory()
        history.start_training()
        
        # Record multiple epochs
        for i in range(100):
            history.record_epoch(
                epoch=i + 1,
                train_loss=0.5,
                epoch_time=10.0 + i * 0.1
            )
        
        # Save and check duration
        history.save(temp_dir / "history.json")
        
        import json
        with open(temp_dir / "history.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Duration should be sum of epoch_times
        expected_duration = sum(10.0 + i * 0.1 for i in range(100))
        actual_duration = data['training_duration_seconds']
        
        # Should be very close (within 0.1s)
        assert abs(actual_duration - expected_duration) < 0.1
        print(f"\n   Expected duration: {expected_duration:.2f}s")
        print(f"   Actual duration: {actual_duration:.2f}s")
        print(f"   Difference: {abs(actual_duration - expected_duration):.4f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

