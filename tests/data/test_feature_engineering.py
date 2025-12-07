"""Tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np

from src.data.feature_engineering import FeatureEngineer
from src.data.features import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    add_derived_features,
)


class TestFeatureEngineer:
    """Test cases for FeatureEngineer."""

    def test_create_time_features(self, sample_dataframe):
        """Test time feature creation using features module."""
        df_features = add_time_features(sample_dataframe.copy())
        
        assert "hour" in df_features.columns
        assert "day_of_year" in df_features.columns
        assert "month" in df_features.columns
        assert "season" in df_features.columns
        assert "is_night" in df_features.columns
        assert "hour_sin" in df_features.columns
        assert "hour_cos" in df_features.columns
        
        # Check values
        assert df_features["hour"].min() >= 0
        assert df_features["hour"].max() <= 23
        assert df_features["day_of_year"].min() >= 1
        assert df_features["day_of_year"].max() <= 366

    def test_create_lag_features(self, sample_dataframe):
        """Test lag feature creation using features module."""
        df_features = add_lag_features(
            sample_dataframe.copy(),
            columns=["Air Temp (C)"],
            lags=[1, 3],
        )
        
        assert "Air Temp (C)_lag_1" in df_features.columns
        assert "Air Temp (C)_lag_3" in df_features.columns
        
        # First row should have NaN for lag_1
        assert pd.isna(df_features.loc[0, "Air Temp (C)_lag_1"])
        # Second row should have first row's value (within tolerance for float comparison)
        assert abs(df_features.loc[1, "Air Temp (C)_lag_1"] - df_features.loc[0, "Air Temp (C)"]) < 0.001

    def test_create_rolling_features(self, sample_dataframe):
        """Test rolling feature creation using features module."""
        df_features = add_rolling_features(
            sample_dataframe.copy(),
            columns=["Air Temp (C)"],
            windows=[6, 12],
            functions=["mean", "min"],
        )
        
        assert "Air Temp (C)_rolling_6h_mean" in df_features.columns
        assert "Air Temp (C)_rolling_6h_min" in df_features.columns
        assert "Air Temp (C)_rolling_12h_mean" in df_features.columns
        assert "Air Temp (C)_rolling_12h_min" in df_features.columns

    def test_create_derived_features(self, sample_dataframe):
        """Test derived feature creation using features module."""
        df_features = add_derived_features(sample_dataframe.copy())
        
        # Check if derived features are created (may depend on available columns)
        if "Air Temp (C)" in df_features.columns and "Dew Point (C)" in df_features.columns:
            assert "temp_dew_diff" in df_features.columns

    def test_build_feature_set(self, sample_dataframe, sample_feature_config):
        """Test complete feature set building."""
        engineer = FeatureEngineer()
        df_features = engineer.build_feature_set(sample_dataframe, sample_feature_config)
        
        # Should have more columns than original
        assert len(df_features.columns) > len(sample_dataframe.columns)
        
        # Should have time features
        assert "hour" in df_features.columns
        assert "season" in df_features.columns
        
        # Should have lag features
        assert "Air Temp (C)_lag_1" in df_features.columns
        assert "Air Temp (C)_lag_3" in df_features.columns
        
        # Should have rolling features
        assert "Air Temp (C)_rolling_6h_mean" in df_features.columns

    def test_build_feature_set_minimal_config(self, sample_dataframe):
        """Test feature building with minimal config."""
        engineer = FeatureEngineer()
        config = {
            "time_features": True,
            "lag_features": {"enabled": False},
            "rolling_features": {"enabled": False},
            "derived_features": False
        }
        df_features = engineer.build_feature_set(sample_dataframe, config)
        
        # Should only have time features
        assert "hour" in df_features.columns
        assert "Air Temp (C)_lag_1" not in df_features.columns

