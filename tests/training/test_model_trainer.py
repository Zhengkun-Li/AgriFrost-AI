"""Tests for model trainer."""

import pytest
import tempfile
import json
from pathlib import Path

from src.training.model_trainer import (
    check_models_exist,
    load_existing_results,
)


class TestModelTrainerHelpers:
    """Test cases for model trainer helper functions."""
    
    def test_check_models_exist_regular_model(self, tmp_path):
        """Test checking models exist for regular model type."""
        horizon_dir = tmp_path / "horizon_3h"
        horizon_dir.mkdir(parents=True)
        
        frost_model_dir = horizon_dir / "frost_classifier"
        temp_model_dir = horizon_dir / "temp_regressor"
        frost_model_dir.mkdir()
        temp_model_dir.mkdir()
        
        # Create model files
        (frost_model_dir / "model.pkl").touch()
        (temp_model_dir / "model.pkl").touch()
        
        assert check_models_exist(horizon_dir, "lightgbm") is True
    
    def test_check_models_exist_missing_models(self, tmp_path):
        """Test checking models exist when models are missing."""
        horizon_dir = tmp_path / "horizon_3h"
        horizon_dir.mkdir(parents=True)
        
        assert check_models_exist(horizon_dir, "lightgbm") is False
    
    def test_check_models_exist_partial_models(self, tmp_path):
        """Test checking models exist with only one model."""
        horizon_dir = tmp_path / "horizon_3h"
        horizon_dir.mkdir(parents=True)
        
        frost_model_dir = horizon_dir / "frost_classifier"
        frost_model_dir.mkdir()
        (frost_model_dir / "model.pkl").touch()
        
        # Missing temp model
        assert check_models_exist(horizon_dir, "lightgbm") is False
    
    def test_check_models_exist_multitask_model(self, tmp_path):
        """Test checking models exist for multitask model."""
        horizon_dir = tmp_path / "horizon_3h"
        horizon_dir.mkdir(parents=True)
        
        multitask_model_dir = horizon_dir / "multitask_model"
        multitask_model_dir.mkdir()
        (multitask_model_dir / "model.pth").touch()
        
        assert check_models_exist(horizon_dir, "lstm_multitask") is True
    
    def test_check_models_exist_multitask_frost_only(self, tmp_path):
        """Test checking multitask models with only frost classifier."""
        horizon_dir = tmp_path / "horizon_3h"
        horizon_dir.mkdir(parents=True)
        
        frost_model_dir = horizon_dir / "frost_classifier"
        frost_model_dir.mkdir()
        (frost_model_dir / "model.pth").touch()
        
        assert check_models_exist(horizon_dir, "lstm_multitask") is True
    
    def test_load_existing_results(self, tmp_path):
        """Test loading existing results."""
        horizon_dir = tmp_path / "horizon_3h"
        horizon_dir.mkdir(parents=True)
        
        frost_metrics = {
            "brier_score": 0.01,
            "ece": 0.005,
            "roc_auc": 0.99,
        }
        temp_metrics = {
            "mae": 1.5,
            "rmse": 2.0,
            "r2": 0.95,
        }
        
        with (horizon_dir / "frost_metrics.json").open("w") as f:
            json.dump(frost_metrics, f)
        
        with (horizon_dir / "temp_metrics.json").open("w") as f:
            json.dump(temp_metrics, f)
        
        result = load_existing_results(horizon_dir)
        
        assert result is not None
        assert result["frost_metrics"] == frost_metrics
        assert result["temp_metrics"] == temp_metrics
    
    def test_load_existing_results_missing_files(self, tmp_path):
        """Test loading results when files are missing."""
        horizon_dir = tmp_path / "horizon_3h"
        horizon_dir.mkdir(parents=True)
        
        result = load_existing_results(horizon_dir)
        assert result is None
    
    def test_load_existing_results_partial_files(self, tmp_path):
        """Test loading results when only one file exists."""
        horizon_dir = tmp_path / "horizon_3h"
        horizon_dir.mkdir(parents=True)
        
        frost_metrics = {"brier_score": 0.01}
        with (horizon_dir / "frost_metrics.json").open("w") as f:
            json.dump(frost_metrics, f)
        
        # Missing temp_metrics.json
        result = load_existing_results(horizon_dir)
        assert result is None

