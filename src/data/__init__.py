"""Data processing module for frost risk forecasting."""

from .loaders import DataLoader
from .cleaners import DataCleaner
from .feature_engineering import FeatureEngineer
from .spatial import NeighborhoodBuilder, SpatialAggregator
from .feature_config import FeaturePipelineConfig, SpatialAggregationConfig
from .pipeline import DataPipeline, DatasetBundle
from .frost_labels import FrostLabelGenerator
from .preprocessors import FeaturePreprocessor, preprocess_with_loso
from .feature_selection import FeatureSelector

__all__ = [
    # Core classes
    "DataLoader",
    "DataCleaner",
    "FeatureEngineer",
    "FeaturePipelineConfig",
    "SpatialAggregationConfig",
    "DataPipeline",
    "DatasetBundle",
    # Label generation
    "FrostLabelGenerator",
    # Preprocessing
    "FeaturePreprocessor",
    "preprocess_with_loso",
    # Feature selection
    "FeatureSelector",
    # Spatial aggregation
    "NeighborhoodBuilder",
    "SpatialAggregator",
]

