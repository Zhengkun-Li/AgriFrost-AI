"""Feature engineering orchestrator built on modular builders."""

from typing import Dict, Optional, List
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_config import FeaturePipelineConfig
from .features import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    add_derived_features,
    add_radiation_features,
    add_wind_features,
    add_humidity_features,
    add_trend_features,
    add_station_features,
)
from .features.constants import STATION_ID_COL, DATE_COL
from .spatial import NeighborhoodBuilder, SpatialAggregator


class FeatureEngineer:
    """Create configurable feature sets for frost forecasting."""

    def __init__(self, default_config: Optional[FeaturePipelineConfig] = None):
        self.default_config = default_config or FeaturePipelineConfig()

    def build_feature_set(
        self,
        df: pd.DataFrame,
        config: Optional[Dict] = None,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Build features according to configuration dictionary or dataclass.
        
        Args:
            df: Input DataFrame.
            config: Feature configuration dictionary or FeaturePipelineConfig instance.
            inplace: If True, modify DataFrame in place (avoids copy). Default: False.
        
        Returns:
            DataFrame with features added.
        """
        # Validate input
        if df.empty:
            raise ValueError("Cannot build features: DataFrame is empty")
        
        if DATE_COL not in df.columns:
            raise ValueError(f"Missing required column '{DATE_COL}' for feature engineering")
        
        cfg = (
            config
            if isinstance(config, FeaturePipelineConfig)
            else FeaturePipelineConfig.from_dict(config or {})
        )
        
        # Use inplace operations if requested, otherwise copy
        if inplace:
            df_features = df
        else:
            df_features = df.copy()

        if cfg.time_features:
            df_features = add_time_features(df_features)

        numeric_cols = self._numeric_feature_candidates(df_features)

        # CRITICAL: Spatial aggregation MUST be done BEFORE lag/rolling for Track D (multi-station FE)
        # This prevents temporal leakage: we aggregate neighbor features first, then apply
        # temporal operations (lag/rolling) to both original and aggregated features.
        if cfg.spatial_aggregation.enabled:
            neighborhood_builder = NeighborhoodBuilder(
                metadata_path=cfg.spatial_aggregation.metadata_path or cfg.station_metadata_path,
                distance_threshold_km=cfg.spatial_aggregation.distance_threshold_km,
                k_neighbors=cfg.spatial_aggregation.k_neighbors,
                weight_method=cfg.spatial_aggregation.weight_method,
                wind_gated=cfg.spatial_aggregation.wind_gated,
            )
            spatial_aggregator = SpatialAggregator(
                neighborhood_builder=neighborhood_builder,
                aggregation_methods=cfg.spatial_aggregation.aggregation_methods,
                impute_missing=cfg.spatial_aggregation.impute_missing,
            )
            cache_path = None
            cache_dir = getattr(cfg.spatial_aggregation, "cache_dir", None)
            cache_key = getattr(cfg.spatial_aggregation, "cache_key", None)
            if cache_dir and cache_key:
                cache_dir = Path(cache_dir)
                cache_path = cache_dir / cache_key / "aggregated.parquet"
            # Apply spatial aggregation to raw variables (generates neighbor_* features)
            # This MUST happen before lag/rolling to prevent temporal leakage
            df_features = spatial_aggregator.aggregate_features(
                df_features,
                variable_columns=numeric_cols,  # Use raw numeric columns
                cache_path=cache_path,
                use_cache=getattr(cfg.spatial_aggregation, "use_cache", True),
                refresh_cache=getattr(cfg.spatial_aggregation, "refresh_cache", False),
            )
            # Update numeric_cols to include newly created neighbor features for lag/rolling
            numeric_cols = self._numeric_feature_candidates(df_features)

        # Lag features: applied AFTER spatial aggregation (for Track D)
        # This ensures lag features can be computed on both original and neighbor_* features
        if cfg.lag.enabled:
            lag_columns = cfg.lag.columns or numeric_cols
            df_features = add_lag_features(
                df_features,
                columns=lag_columns,
                lags=cfg.lag.lags,
            )

        # Rolling features: applied AFTER spatial aggregation (for Track D)
        # This ensures rolling features can be computed on both original and neighbor_* features
        # Rolling windows MUST NOT leak across stations (enforced by groupby in add_rolling_features)
        if cfg.rolling.enabled:
            rolling_columns = cfg.rolling.columns or numeric_cols
            df_features = add_rolling_features(
                df_features,
                columns=rolling_columns,
                windows=cfg.rolling.windows,
                functions=cfg.rolling.functions,
            )

        if cfg.derived_features:
            df_features = add_derived_features(df_features)
        if cfg.radiation_features:
            df_features = add_radiation_features(df_features)
        if cfg.wind_features:
            df_features = add_wind_features(df_features)
        if cfg.humidity_features:
            df_features = add_humidity_features(df_features)
        if cfg.trend_features:
            df_features = add_trend_features(df_features)
        if cfg.station_features:
            df_features = add_station_features(df_features, metadata_path=cfg.station_metadata_path)

        df_features = self._optimize_dtypes(df_features)
        return df_features

    def engineer_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Backward compatible alias for build_feature_set."""
        return self.build_feature_set(df, config)

    @staticmethod
    def _numeric_feature_candidates(df: pd.DataFrame) -> List[str]:
        """Get numeric columns excluding metadata columns (station ID, date)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numeric_cols if col not in {DATE_COL, STATION_ID_COL}]

    @staticmethod
    def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["float64"]).columns:
            try:
                df[col] = df[col].astype("float32")
            except (ValueError, OverflowError):
                pass
        for col in df.select_dtypes(include=["int64"]).columns:
            if col not in [STATION_ID_COL]:
                try:
                    df[col] = pd.to_numeric(df[col], downcast="integer")
                except (ValueError, OverflowError):
                    pass
        return df

