"""Configuration validation with unified schema (migrated from models/utils).

This module provides configuration validation that can be used across the entire
project. Eventually will be enhanced with pydantic if available.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging

# Import from old location for backward compatibility
from src.models.utils.config_validator import ConfigValidator

_logger = logging.getLogger(__name__)

# Re-export for easy migration
__all__ = ['ConfigValidator', 'validate_experiment_metadata', 'validate_model_config']


def validate_experiment_metadata(
    matrix_cell: Optional[str] = None,
    track: Optional[str] = None,
    horizon_h: Optional[int] = None,
    model_name: Optional[str] = None,
    radius_km: Optional[float] = None,
    knn_k: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """Validate experiment metadata (convenience function).
    
    This is a wrapper around ConfigValidator.validate_experiment_metadata()
    for easier access from config/schema.
    
    Args:
        matrix_cell: Matrix cell identifier (A, B, C, D, E).
        track: Feature engineering track.
        horizon_h: Forecast horizon in hours.
        model_name: Model name.
        radius_km: Radius in km.
        knn_k: k value for kNN.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    return ConfigValidator.validate_experiment_metadata(
        matrix_cell=matrix_cell,
        track=track,
        horizon_h=horizon_h,
        model_name=model_name,
        radius_km=radius_km,
        knn_k=knn_k
    )


def validate_model_config(
    model_type: str,
    config: Dict[str, Any],
    task_type: str = "classification"
) -> Tuple[bool, Optional[str]]:
    """Validate model configuration (convenience function).
    
    Args:
        model_type: Type of model.
        config: Configuration dictionary.
        task_type: Task type (classification or regression).
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    return ConfigValidator.validate_model_config(
        model_type=model_type,
        config=config,
        task_type=task_type
    )

