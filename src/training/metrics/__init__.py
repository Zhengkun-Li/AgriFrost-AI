"""Training metrics schema and registry.

This module provides centralized metric schema management for the training system.
It ensures consistency across ProgressLogger, TrainingHistory, and model trainers.
"""

from .schema import MetricSchema, MetricRegistry

__all__ = ['MetricSchema', 'MetricRegistry']

