"""Model registry for configuration-driven runners."""

from __future__ import annotations

import logging
from typing import Dict, Type, Any

from src.training.model_config import get_model_class as legacy_get_model_class

_logger = logging.getLogger(__name__)

_MODEL_REGISTRY: Dict[str, Type[Any]] = {}


def register_model(name: str, cls: Type[Any]) -> None:
    """Register/override a model class.
    
    Args:
        name: Model name.
        cls: Model class.
    
    Raises:
        ValueError: If name is empty or cls is not a class.
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Model name must be a non-empty string, got {name}")
    
    if not isinstance(cls, type):
        raise ValueError(f"Model class must be a class type, got {type(cls)}")
    
    normalized = name.lower()
    if normalized in _MODEL_REGISTRY:
        _logger.warning(f"Overwriting existing model registration: {normalized}")
    
    _MODEL_REGISTRY[normalized] = cls
    _logger.debug(f"Registered model: {normalized}")


def get_model_class(name: str):
    """Return model class by name, falling back to legacy mapping.
    
    Args:
        name: Model name.
    
    Returns:
        Model class.
    
    Raises:
        ValueError: If name is empty or model not found (after checking legacy mapping).
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Model name must be a non-empty string, got {name}")
    
    normalized = name.lower()
    if normalized in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[normalized]
    
    # Fallback to legacy mapping
    try:
        return legacy_get_model_class(name)
    except (ValueError, KeyError) as e:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available models: {available}. "
            f"Legacy mapping also failed: {e}"
        ) from e

