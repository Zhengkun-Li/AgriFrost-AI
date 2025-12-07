"""Centralized metric schema definition and registry.

This module provides a unified metric schema that is shared across:
- ProgressLogger: For logging training progress
- TrainingHistory: For tracking training metrics
- ModelTrainers: For reporting metrics during training

This ensures consistency and prevents metric name mismatches.
"""

from typing import List, Set, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class MetricType(Enum):
    """Metric type classification."""
    STANDARD = "standard"  # train_loss, val_loss, learning_rate, epoch_time
    CLASSIFICATION = "classification"  # accuracy, precision, recall, f1, ece
    REGRESSION = "regression"  # mae, rmse, r2
    MULTITASK = "multitask"  # temp_loss, frost_loss
    GRAPH = "graph"  # graph_loss, node_loss
    CUSTOM = "custom"  # Any user-defined metrics


@dataclass
class MetricDefinition:
    """Definition of a single metric."""
    name: str
    metric_type: MetricType
    description: str = ""
    required: bool = False
    default_value: Optional[float] = None


class MetricSchema:
    """Centralized metric schema for training system.
    
    This class provides a unified interface for defining and accessing metrics
    that are used across ProgressLogger, TrainingHistory, and model trainers.
    
    **Standard Metrics:**
    - train_loss: Training loss
    - val_loss: Validation loss
    - learning_rate: Learning rate
    - epoch_time: Time taken per epoch (seconds)
    
    **Extended Metrics:**
    - Classification: accuracy, precision, recall, f1, ece (Expected Calibration Error)
    - Regression: mae, rmse, r2
    - Multi-task: temp_loss, frost_loss
    - Graph: graph_loss, node_loss
    """
    
    # Standard metrics (always present)
    STANDARD_METRICS: List[str] = [
        'train_loss',
        'val_loss',
        'learning_rate',
        'epoch_time'
    ]
    
    # Classification metrics
    CLASSIFICATION_METRICS: List[str] = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'ece'  # Expected Calibration Error
    ]
    
    # Regression metrics
    REGRESSION_METRICS: List[str] = [
        'mae',  # Mean Absolute Error
        'rmse',  # Root Mean Squared Error
        'r2'  # R-squared
    ]
    
    # Multi-task metrics
    MULTITASK_METRICS: List[str] = [
        'temp_loss',
        'frost_loss'
    ]
    
    # Graph neural network metrics
    GRAPH_METRICS: List[str] = [
        'graph_loss',
        'node_loss',
        'edge_loss'
    ]
    
    @classmethod
    def default_history_metrics(cls) -> List[str]:
        """Get default metrics for TrainingHistory.
        
        Returns:
            List of standard metric names.
        """
        return cls.STANDARD_METRICS.copy()
    
    @classmethod
    def classification_metrics(cls) -> List[str]:
        """Get metrics for classification tasks.
        
        Returns:
            List of classification metric names.
        """
        return cls.STANDARD_METRICS + cls.CLASSIFICATION_METRICS
    
    @classmethod
    def regression_metrics(cls) -> List[str]:
        """Get metrics for regression tasks.
        
        Returns:
            List of regression metric names.
        """
        return cls.STANDARD_METRICS + cls.REGRESSION_METRICS
    
    @classmethod
    def multitask_metrics(cls) -> List[str]:
        """Get metrics for multi-task learning.
        
        Returns:
            List of multi-task metric names.
        """
        return cls.STANDARD_METRICS + cls.MULTITASK_METRICS
    
    @classmethod
    def graph_metrics(cls) -> List[str]:
        """Get metrics for graph neural networks.
        
        Returns:
            List of graph metric names.
        """
        return cls.STANDARD_METRICS + cls.GRAPH_METRICS
    
    @classmethod
    def all_metrics(cls) -> List[str]:
        """Get all available metrics.
        
        Returns:
            List of all metric names.
        """
        return (
            cls.STANDARD_METRICS +
            cls.CLASSIFICATION_METRICS +
            cls.REGRESSION_METRICS +
            cls.MULTITASK_METRICS +
            cls.GRAPH_METRICS
        )
    
    @classmethod
    def validate_metrics(cls, metrics: List[str]) -> tuple[bool, Optional[str]]:
        """Validate that all metrics are known.
        
        Args:
            metrics: List of metric names to validate.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        all_known = set(cls.all_metrics())
        unknown = set(metrics) - all_known
        
        if unknown:
            return False, f"Unknown metrics: {sorted(unknown)}. Available: {sorted(all_known)}"
        
        return True, None
    
    @classmethod
    def get_metric_type(cls, metric_name: str) -> MetricType:
        """Get the type of a metric.
        
        Args:
            metric_name: Name of the metric.
            
        Returns:
            MetricType enum value.
        """
        if metric_name in cls.STANDARD_METRICS:
            return MetricType.STANDARD
        elif metric_name in cls.CLASSIFICATION_METRICS:
            return MetricType.CLASSIFICATION
        elif metric_name in cls.REGRESSION_METRICS:
            return MetricType.REGRESSION
        elif metric_name in cls.MULTITASK_METRICS:
            return MetricType.MULTITASK
        elif metric_name in cls.GRAPH_METRICS:
            return MetricType.GRAPH
        else:
            return MetricType.CUSTOM


class MetricRegistry:
    """Global metric registry for dynamic metric registration.
    
    This allows model trainers to register custom metrics that are then
    available to ProgressLogger and TrainingHistory.
    """
    
    _registry: Dict[str, MetricDefinition] = {}
    _frozen: bool = False
    
    @classmethod
    def register(
        cls,
        name: str,
        metric_type: MetricType = MetricType.CUSTOM,
        description: str = "",
        required: bool = False
    ) -> None:
        """Register a custom metric.
        
        Args:
            name: Metric name.
            metric_type: Type of metric.
            description: Description of the metric.
            required: Whether this metric is required.
            
        Raises:
            ValueError: If registry is frozen or metric already exists.
        """
        if cls._frozen:
            raise ValueError("Metric registry is frozen. Cannot register new metrics.")
        
        if name in cls._registry:
            raise ValueError(f"Metric '{name}' already registered.")
        
        cls._registry[name] = MetricDefinition(
            name=name,
            metric_type=metric_type,
            description=description,
            required=required
        )
    
    @classmethod
    def get_registered_metrics(cls) -> List[str]:
        """Get all registered custom metrics.
        
        Returns:
            List of registered metric names.
        """
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a metric is registered.
        
        Args:
            name: Metric name.
            
        Returns:
            True if metric is registered.
        """
        return name in cls._registry
    
    @classmethod
    def freeze(cls) -> None:
        """Freeze the registry to prevent further registration."""
        cls._frozen = True
    
    @classmethod
    def unfreeze(cls) -> None:
        """Unfreeze the registry to allow registration."""
        cls._frozen = False
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered metrics."""
        cls._registry.clear()
        cls._frozen = False

