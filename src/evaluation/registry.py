"""Evaluation strategy registry."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple, Optional

_logger = logging.getLogger(__name__)

# Updated handler signature to support custom parameters (e.g., radius_km for spatial tracks)
EvaluationHandler = Callable[..., Any]  # Supports *args, **kwargs for flexibility

_EVALUATION_REGISTRY: Dict[str, EvaluationHandler] = {}


def register_evaluation_strategy(name: str, handler: EvaluationHandler) -> None:
    """Register a strategy handler.
    
    Args:
        name: Strategy name.
        handler: Handler function.
    
    Raises:
        ValueError: If name is empty or handler is not callable.
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Strategy name must be a non-empty string, got {name}")
    
    if not callable(handler):
        raise ValueError(f"Handler must be callable, got {type(handler)}")
    
    normalized = name.lower()
    if normalized in _EVALUATION_REGISTRY:
        _logger.warning(f"Overwriting existing evaluation strategy: {normalized}")
    
    _EVALUATION_REGISTRY[normalized] = handler
    _logger.debug(f"Registered evaluation strategy: {normalized}")


def get_evaluation_handler(name: str) -> EvaluationHandler:
    """Return handler for a strategy.
    
    Args:
        name: Strategy name.
    
    Returns:
        Handler function.
    
    Raises:
        ValueError: If name is empty or strategy not found.
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Strategy name must be a non-empty string, got {name}")
    
    normalized = name.lower()
    if normalized not in _EVALUATION_REGISTRY:
        available = list(_EVALUATION_REGISTRY.keys())
        raise ValueError(
            f"Unknown evaluation strategy: {name}. "
            f"Available strategies: {available}"
        )
    
    return _EVALUATION_REGISTRY[normalized]

