"""Tests for path utility functions."""

import pytest
import tempfile
from pathlib import Path

from src.utils.path_utils import ensure_dir, get_project_root, get_data_dir


class TestPathUtils:
    """Test cases for path utility functions."""
    
    def test_ensure_dir_exists(self, temp_dir):
        """Test ensure_dir when directory exists."""
        result = ensure_dir(temp_dir)
        assert result == temp_dir
        assert temp_dir.exists()
        assert temp_dir.is_dir()
    
    def test_ensure_dir_creates(self):
        """Test ensure_dir when directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new" / "nested" / "dir"
            result = ensure_dir(new_dir)
            assert result == new_dir
            assert new_dir.exists()
            assert new_dir.is_dir()
    
    def test_ensure_dir_none(self):
        """Test ensure_dir with None raises ValueError."""
        with pytest.raises(ValueError, match="path cannot be None"):
            ensure_dir(None)
    
    def test_ensure_dir_string_path(self, temp_dir):
        """Test ensure_dir accepts string path."""
        new_dir = temp_dir / "string_path"
        result = ensure_dir(str(new_dir))
        assert isinstance(result, Path)
        assert result == new_dir
        assert new_dir.exists()
    
    def test_get_project_root(self):
        """Test get_project_root returns correct path."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()
        assert (root / "src").exists()
        assert (root / "tests").exists()
    
    def test_get_data_dir_raw(self):
        """Test get_data_dir with 'raw'."""
        data_dir = get_data_dir("raw")
        assert isinstance(data_dir, Path)
        assert data_dir.name == "raw"
        assert data_dir.parent.name == "data"
    
    def test_get_data_dir_processed(self):
        """Test get_data_dir with 'processed'."""
        data_dir = get_data_dir("processed")
        assert isinstance(data_dir, Path)
        assert data_dir.name == "processed"
    
    def test_get_data_dir_interim(self):
        """Test get_data_dir with 'interim'."""
        data_dir = get_data_dir("interim")
        assert isinstance(data_dir, Path)
        assert data_dir.name == "interim"
    
    def test_get_data_dir_external(self):
        """Test get_data_dir with 'external'."""
        data_dir = get_data_dir("external")
        assert isinstance(data_dir, Path)
        assert data_dir.name == "external"
    
    def test_get_data_dir_invalid(self):
        """Test get_data_dir with invalid type raises ValueError."""
        with pytest.raises(ValueError, match="data_type must be one of"):
            get_data_dir("invalid")

