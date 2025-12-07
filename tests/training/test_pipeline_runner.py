"""Tests for pipeline runner."""

import pytest
import tempfile
import yaml
from pathlib import Path

from src.training.pipeline_runner import (
    load_training_config,
    _resolve_path,
    _require_path,
    _build_pipeline_config,
    _infer_track,
    PipelineTrainingConfig,
    DataSection,
    LabelSection,
    TrainingSection,
)


class TestPipelineRunnerHelpers:
    """Test cases for pipeline runner helper functions."""
    
    def test_resolve_path_absolute(self, tmp_path):
        """Test resolving absolute path."""
        abs_path = tmp_path / "absolute" / "path"
        result = _resolve_path(abs_path, Path("/project"))
        assert result == abs_path
    
    def test_resolve_path_relative(self):
        """Test resolving relative path."""
        project_root = Path("/project")
        result = _resolve_path("relative/path", project_root)
        assert result == project_root / "relative" / "path"
    
    def test_resolve_path_none(self):
        """Test resolving None returns None."""
        result = _resolve_path(None, Path("/project"))
        assert result is None
    
    def test_require_path_valid(self, tmp_path):
        """Test requiring valid path."""
        valid_path = tmp_path / "valid"
        result = _require_path(valid_path, Path("/project"), "test_field")
        assert result == _resolve_path(valid_path, Path("/project"))
    
    def test_require_path_none(self):
        """Test requiring None path raises ValueError."""
        with pytest.raises(ValueError, match="must be provided"):
            _require_path(None, Path("/project"), "test_field")
    
    def test_build_pipeline_config(self):
        """Test building pipeline config."""
        data_cfg = DataSection(
            source=Path("data/raw/test.csv"),
            matrix_cell="B",
        )
        label_cfg = LabelSection(horizons=[3, 6, 12, 24])
        
        config = _build_pipeline_config(data_cfg, label_cfg)
        
        assert "cleaning" in config
        assert "feature_engineering" in config
        assert "labels" in config
        assert config["labels"]["threshold"] == 0.0
    
    def test_infer_track_b_d(self):
        """Test inferring track for B/D matrix cells."""
        assert _infer_track("B") == "top175_features"
        assert _infer_track("D") == "top175_features"
        assert _infer_track("b") == "top175_features"
        assert _infer_track("d") == "top175_features"
    
    def test_infer_track_a_c_e(self):
        """Test inferring track for A/C/E matrix cells."""
        assert _infer_track("A") == "raw"
        assert _infer_track("C") == "raw"
        assert _infer_track("E") == "raw"
    
    def test_infer_track_none(self):
        """Test inferring track with None returns default."""
        assert _infer_track(None) == "top175_features"


class TestLoadTrainingConfig:
    """Test cases for load_training_config."""
    
    def test_load_config_minimal(self, tmp_path):
        """Test loading minimal config."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        
        config = load_training_config(
            config_path=None,
            project_root=project_root,
            cli_overrides={
                "data_path": str(tmp_path / "data.csv"),
                "model": "lightgbm",
                "output_dir": str(tmp_path / "output"),
            }
        )
        
        assert isinstance(config, PipelineTrainingConfig)
        assert config.data.source == Path(tmp_path / "data.csv")
        assert config.training.model == "lightgbm"
    
    def test_load_config_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        
        config_dir = project_root / "config" / "pipeline"
        config_dir.mkdir(parents=True)
        
        config_path = config_dir / "test.yaml"
        config_yaml = {
            "data": {
                "source": "data/raw/test.csv",
                "matrix_cell": "B",
            },
            "labels": {
                "horizons": [3, 6, 12],
                "frost_threshold": 0.0,
            },
            "training": {
                "model": "lightgbm",
                "output_dir": "experiments/test",
            },
        }
        with config_path.open("w") as f:
            yaml.dump(config_yaml, f)
        
        config = load_training_config(
            config_path=config_path,
            project_root=project_root,
        )
        
        assert config.data.matrix_cell == "B"
        assert config.labels.horizons == [3, 6, 12]
        assert config.training.model == "lightgbm"
    
    def test_load_config_cli_overrides(self, tmp_path):
        """Test CLI overrides take precedence."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        
        config_dir = project_root / "config" / "pipeline"
        config_dir.mkdir(parents=True)
        
        config_path = config_dir / "test.yaml"
        config_yaml = {
            "data": {
                "source": "data/raw/test.csv",
                "matrix_cell": "A",
            },
            "labels": {
                "horizons": [3, 6],
            },
            "training": {
                "model": "xgboost",
                "output_dir": "experiments/test",
            },
        }
        with config_path.open("w") as f:
            yaml.dump(config_yaml, f)
        
        config = load_training_config(
            config_path=config_path,
            project_root=project_root,
            cli_overrides={
                "matrix_cell": "B",
                "horizons": [12, 24],
                "model": "lightgbm",
            }
        )
        
        # CLI overrides should take precedence
        assert config.data.matrix_cell == "B"
        assert config.labels.horizons == [12, 24]
        assert config.training.model == "lightgbm"
    
    def test_load_config_matrix_cell_inference(self, tmp_path):
        """Test matrix_cell inference from output_dir."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        
        config = load_training_config(
            config_path=None,
            project_root=project_root,
            cli_overrides={
                "data_path": str(tmp_path / "data.csv"),
                "model": "lightgbm",
                "output_dir": "experiments/B/model_run",
            }
        )
        
        # Should infer matrix_cell from output_dir
        assert config.data.matrix_cell == "B"
    
    def test_load_config_auto_cleaning_config(self, tmp_path):
        """Test auto-selection of cleaning config based on matrix_cell."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        
        # Create cleaning config files
        config_dir = project_root / "config"
        config_dir.mkdir()
        (config_dir / "data_cleaning_raw.yaml").touch()
        (config_dir / "data_cleaning_fe.yaml").touch()
        
        config = load_training_config(
            config_path=None,
            project_root=project_root,
            cli_overrides={
                "data_path": str(tmp_path / "data.csv"),
                "matrix_cell": "A",  # Should use raw config
                "model": "lightgbm",
                "output_dir": str(tmp_path / "output"),
            }
        )
        
        assert config.data.cleaning.get("config_path") == "config/data_cleaning_raw.yaml"
        
        # Test for B (feature-engineered)
        config_b = load_training_config(
            config_path=None,
            project_root=project_root,
            cli_overrides={
                "data_path": str(tmp_path / "data.csv"),
                "matrix_cell": "B",
                "model": "lightgbm",
                "output_dir": str(tmp_path / "output"),
            }
        )
        
        assert config_b.data.cleaning.get("config_path") == "config/data_cleaning_fe.yaml"
    
    def test_load_config_auto_feature_engineering(self, tmp_path):
        """Test auto-determination of feature engineering based on matrix_cell."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        
        # A/C should have feature_engineering disabled
        config_a = load_training_config(
            config_path=None,
            project_root=project_root,
            cli_overrides={
                "data_path": str(tmp_path / "data.csv"),
                "matrix_cell": "A",
                "model": "lightgbm",
                "output_dir": str(tmp_path / "output"),
            }
        )
        
        assert config_a.data.feature_engineering.get("enabled") is False
        
        # B/D should have feature_engineering enabled
        config_b = load_training_config(
            config_path=None,
            project_root=project_root,
            cli_overrides={
                "data_path": str(tmp_path / "data.csv"),
                "matrix_cell": "B",
                "model": "lightgbm",
                "output_dir": str(tmp_path / "output"),
            }
        )
        
        assert config_b.data.feature_engineering.get("enabled") is True

