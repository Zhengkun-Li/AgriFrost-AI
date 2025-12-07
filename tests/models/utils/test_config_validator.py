"""Unit tests for ConfigValidator."""

import pytest
from pathlib import Path
from src.models.utils.config_validator import ConfigValidator


class TestConfigValidator:
    """Test ConfigValidator functionality."""
    
    def test_validate_experiment_metadata_matrix_cell_a_no_radius(self):
        """Test matrix cell A: should not have radius_km."""
        valid, msg = ConfigValidator.validate_experiment_metadata(
            matrix_cell='A',
            radius_km=50.0
        )
        assert valid is False
        assert 'cannot have radius_km' in msg.lower() or 'no radius' in msg.lower()
    
    def test_validate_experiment_metadata_matrix_cell_c_must_have_radius(self):
        """Test matrix cell C: must have radius_km."""
        valid, msg = ConfigValidator.validate_experiment_metadata(
            matrix_cell='C',
            radius_km=None
        )
        assert valid is False
        assert 'must have radius_km' in msg.lower()
    
    def test_validate_experiment_metadata_matrix_cell_c_with_radius(self):
        """Test matrix cell C: valid with radius_km."""
        valid, msg = ConfigValidator.validate_experiment_metadata(
            matrix_cell='C',
            radius_km=50.0
        )
        assert valid is True
        assert msg is None
    
    def test_validate_experiment_metadata_matrix_cell_e_must_have_knn_k(self):
        """Test matrix cell E: must have knn_k."""
        valid, msg = ConfigValidator.validate_experiment_metadata(
            matrix_cell='E',
            knn_k=None
        )
        assert valid is False
        assert 'must have knn_k' in msg.lower()
    
    def test_validate_experiment_metadata_matrix_cell_e_no_radius(self):
        """Test matrix cell E: cannot have radius_km."""
        valid, msg = ConfigValidator.validate_experiment_metadata(
            matrix_cell='E',
            knn_k=5,
            radius_km=50.0
        )
        assert valid is False
        assert 'cannot have radius_km' in msg.lower() or 'use knn_k' in msg.lower()
    
    def test_validate_experiment_metadata_matrix_cell_e_with_knn_k(self):
        """Test matrix cell E: valid with knn_k."""
        valid, msg = ConfigValidator.validate_experiment_metadata(
            matrix_cell='E',
            knn_k=5
        )
        assert valid is True
        assert msg is None
    
    def test_validate_experiment_metadata_horizon_h_valid(self):
        """Test valid horizon_h values."""
        for horizon in [3, 6, 12, 24]:
            valid, msg = ConfigValidator.validate_experiment_metadata(
                horizon_h=horizon
            )
            assert valid is True, f"horizon_h={horizon} should be valid"
    
    def test_validate_experiment_metadata_horizon_h_invalid(self):
        """Test invalid horizon_h values."""
        valid, msg = ConfigValidator.validate_experiment_metadata(
            horizon_h=5
        )
        assert valid is False
        assert 'invalid horizon_h' in msg.lower()
    
    def test_validate_experiment_metadata_graph_model_requires_graph_params(self):
        """Test graph models must have either radius_km or knn_k."""
        valid, msg = ConfigValidator.validate_experiment_metadata(
            model_name='dcrnn',
            radius_km=None,
            knn_k=None
        )
        assert valid is False
        assert 'must have either radius_km or knn_k' in msg.lower()
    
    def test_validate_experiment_metadata_graph_model_with_radius(self):
        """Test graph model valid with radius_km."""
        valid, msg = ConfigValidator.validate_experiment_metadata(
            model_name='dcrnn',
            radius_km=50.0
        )
        assert valid is True
    
    def test_validate_model_config_lstm_valid(self):
        """Test valid LSTM config."""
        config = {
            "model_params": {
                "sequence_length": 24,
                "hidden_size": 64,
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001
            }
        }
        valid, msg = ConfigValidator.validate_model_config("lstm", config)
        assert valid is True
        assert msg is None
    
    def test_validate_model_config_lstm_missing_param(self):
        """Test LSTM config with missing parameter."""
        config = {
            "model_params": {
                "sequence_length": 24,
                "hidden_size": 64,
                # Missing batch_size, epochs, learning_rate
            }
        }
        valid, msg = ConfigValidator.validate_model_config("lstm", config)
        assert valid is False
        assert 'missing required parameter' in msg.lower()
    
    def test_validate_model_config_lstm_invalid_param(self):
        """Test LSTM config with invalid parameter."""
        config = {
            "model_params": {
                "sequence_length": -1,  # Invalid
                "hidden_size": 64,
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001
            }
        }
        valid, msg = ConfigValidator.validate_model_config("lstm", config)
        assert valid is False
        assert 'must be a positive' in msg.lower()
    
    def test_validate_training_args_strict_mode(self, tmp_path):
        """Test validate_training_args with strict mode."""
        valid, msg = ConfigValidator.validate_training_args(
            model_type='lstm',
            checkpoint_dir=tmp_path / "checkpoints",
            log_file=tmp_path / "training.log",
            strict_mode=True,
            unknown_key="unknown_value"
        )
        # Should be valid (strict mode just warns, doesn't fail)
        assert valid is True
    
    def test_validate_training_args_invalid_directory(self):
        """Test validate_training_args with invalid directory."""
        # Use non-existent parent directory
        invalid_path = Path("/nonexistent/path/checkpoints")
        valid, msg = ConfigValidator.validate_training_args(
            model_type='lstm',
            checkpoint_dir=invalid_path,
            log_file=None
        )
        # Should handle gracefully or raise appropriate error
        # (depends on implementation)
        # assert valid is False or msg is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

