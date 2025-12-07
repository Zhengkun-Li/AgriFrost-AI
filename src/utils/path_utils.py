"""Path utility functions."""

import logging
from pathlib import Path
from typing import Optional, Union

_logger = logging.getLogger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not.
    
    Args:
        path: Path to directory (can be Path object or string).
    
    Returns:
        Path object.
    
    Raises:
        ValueError: If path is None.
        OSError: If directory creation fails.
    """
    if path is None:
        raise ValueError("path cannot be None")
    
    path_obj = Path(path) if not isinstance(path, Path) else path
    
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
        _logger.debug(f"Ensured directory exists: {path_obj}")
    except OSError as e:
        _logger.error(f"Failed to create directory {path_obj}: {e}")
        raise
    
    return path_obj


def get_project_root() -> Path:
    """Get project root directory.
    
    Returns:
        Path to project root (where this file is located).
    """
    # Assuming src/utils/ is at project_root/src/utils/
    return Path(__file__).parent.parent.parent


def get_data_dir(data_type: str = "raw") -> Path:
    """Get data directory path.
    
    Args:
        data_type: Type of data ("raw", "processed", "interim", "external").
    
    Returns:
        Path to data directory.
    
    Raises:
        ValueError: If data_type is not a valid type.
    """
    valid_types = ["raw", "processed", "interim", "external"]
    if data_type not in valid_types:
        raise ValueError(
            f"data_type must be one of {valid_types}, got {data_type}"
        )
    
    root = get_project_root()
    return root / "data" / data_type

