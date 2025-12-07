"""Unit tests for TrainingHistory."""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from src.models.utils.training_history import TrainingHistory


class TestTrainingHistory:
    """Test TrainingHistory functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def history(self):
        """Create TrainingHistory instance with default metrics."""
        return TrainingHistory(metrics=['train_loss', 'val_loss', 'learning_rate', 'epoch_time'])
    
    def test_init(self, history):
        """Test TrainingHistory initialization."""
        assert history.metrics == ['train_loss', 'val_loss', 'learning_rate', 'epoch_time']
        assert 'epoch' in history.history
        assert 'train_loss' in history.history
        assert 'val_loss' in history.history
        assert 'learning_rate' in history.history
        assert 'epoch_time' in history.history
        assert history.start_time is None
        assert history.current_epoch == 0
    
    def test_start_training(self, history):
        """Test start_training."""
        history.start_training()
        assert history.start_time is not None
    
    def test_record_epoch_field_unification(self, history):
        """Test record_epoch field unification with ProgressLogger."""
        history.record_epoch(
            epoch=1,
            train_loss=0.5,
            val_loss=0.45,
            learning_rate=0.01,
            epoch_time=12.5  # Standard field (not in kwargs)
        )
        
        assert history.history['epoch'] == [1]
        assert history.history['train_loss'] == [0.5]
        assert history.history['val_loss'] == [0.45]
        assert history.history['learning_rate'] == [0.01]
        assert history.history['epoch_time'] == [12.5]
        assert len(history.epoch_times) == 1
        assert history.epoch_times[0] == 12.5
    
    def test_record_epoch_metrics_filtering(self, history):
        """Test that only expected metrics are recorded."""
        # Try to record metric not in metrics list
        history.record_epoch(
            epoch=1,
            train_loss=0.5,
            unknown_metric=999.0  # Not in metrics list
        )
        
        # Should not have unknown_metric in history
        assert 'unknown_metric' not in history.history
        assert history.history['train_loss'] == [0.5]
    
    def test_get_history(self, history):
        """Test get_history."""
        history.record_epoch(
            epoch=1,
            train_loss=0.5,
            val_loss=0.45
        )
        
        history_dict = history.get_history()
        assert isinstance(history_dict, dict)
        assert 'epoch' in history_dict
        assert 'train_loss' in history_dict
        # Should be a copy
        history_dict['new_key'] = 'value'
        assert 'new_key' not in history.history
    
    def test_get_latest(self, history):
        """Test get_latest."""
        history.record_epoch(epoch=1, train_loss=0.5)
        history.record_epoch(epoch=2, train_loss=0.4)
        
        assert history.get_latest('train_loss') == 0.4
        assert history.get_latest('nonexistent') is None
    
    def test_save_load_expected_metrics(self, history, temp_dir):
        """Test save and load with expected_metrics."""
        history.start_training()
        history.record_epoch(
            epoch=1,
            train_loss=0.5,
            val_loss=0.45,
            learning_rate=0.01,
            epoch_time=12.5
        )
        
        save_path = temp_dir / "training_history.json"
        history.save(save_path)
        
        # Check saved file contains expected_metrics
        with open(save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'expected_metrics' in data
        assert data['expected_metrics'] == ['train_loss', 'val_loss', 'learning_rate', 'epoch_time']
        assert 'training_duration_seconds' in data
        
        # Load and verify
        loaded_history = TrainingHistory.load(save_path)
        assert loaded_history.metrics == history.metrics
        assert loaded_history.history['train_loss'] == [0.5]
        assert loaded_history.history['epoch_time'] == [12.5]
    
    def test_load_filters_unknown_metrics(self, temp_dir):
        """Test that load filters out metrics not in expected_metrics."""
        # Create history file with extra fields
        history_data = {
            'epoch': [1, 2],
            'train_loss': [0.5, 0.4],
            'val_loss': [0.45, 0.4],
            'learning_rate': [0.01, 0.01],
            'epoch_time': [12.5, 11.0],
            'expected_metrics': ['train_loss', 'val_loss', 'learning_rate', 'epoch_time'],
            'temporary_field': [999, 888],  # Should be filtered
            'training_duration_seconds': 23.5,
            'total_epochs': 2
        }
        
        save_path = temp_dir / "training_history.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f)
        
        # Load and verify filtering
        loaded_history = TrainingHistory.load(save_path)
        assert 'temporary_field' not in loaded_history.history
        assert loaded_history.history['train_loss'] == [0.5, 0.4]
    
    def test_duration_precision(self, history, temp_dir):
        """Test duration precision using sum of epoch_times."""
        history.start_training()
        history.record_epoch(epoch=1, epoch_time=10.5)
        history.record_epoch(epoch=2, epoch_time=11.0)
        history.record_epoch(epoch=3, epoch_time=12.0)
        
        save_path = temp_dir / "training_history.json"
        history.save(save_path)
        
        # Check duration is sum of epoch_times
        with open(save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data['training_duration_seconds'] == 33.5  # 10.5 + 11.0 + 12.0
    
    def test_len(self, history):
        """Test __len__ method."""
        assert len(history) == 0
        history.record_epoch(epoch=1, train_loss=0.5)
        assert len(history) == 1
        history.record_epoch(epoch=2, train_loss=0.4)
        assert len(history) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

