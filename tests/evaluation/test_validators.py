"""Tests for cross-validation strategies."""

import pytest
import pandas as pd
import numpy as np

from src.evaluation.validators import CrossValidator


class TestCrossValidator:
    """Test cases for CrossValidator."""
    
    def test_time_split(self, sample_dataframe):
        """Test time-based split."""
        train, val, test = CrossValidator.time_split(
            sample_dataframe,
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(sample_dataframe)
        
        # Check ordering
        assert train["Date"].max() <= val["Date"].min()
        assert val["Date"].max() <= test["Date"].min()
    
    def test_time_split_missing_date_col(self):
        """Test time_split with missing date column."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Date column"):
            CrossValidator.time_split(df)
    
    def test_leave_one_station_out(self):
        """Test LOSO cross-validation with multiple stations."""
        # Create DataFrame with multiple stations (LOSO requires at least 2)
        df = pd.DataFrame({
            "Stn Id": [2, 2, 7, 7],
            "Date": pd.date_range("2020-01-01", periods=4, freq="h"),
            "value": [1, 2, 3, 4]
        })
        
        splits = CrossValidator.leave_one_station_out(df)
        
        # Should have splits for each unique station (2 stations)
        assert len(splits) == 2
        
        # Check each split
        for train, test in splits:
            test_station = test["Stn Id"].iloc[0]
            assert test_station not in train["Stn Id"].values
            assert len(test) > 0
    
    def test_leave_one_station_out_single_station(self, sample_dataframe):
        """Test LOSO cross-validation with single station (should raise error)."""
        # LOSO requires at least 2 stations
        with pytest.raises(ValueError, match="Need at least 2 stations"):
            CrossValidator.leave_one_station_out(sample_dataframe)
    
    def test_leave_one_station_out_multiple_stations(self):
        """Test LOSO with multiple stations."""
        df = pd.DataFrame({
            "Stn Id": [2, 2, 7, 7, 15, 15],
            "Date": pd.date_range("2020-01-01", periods=6, freq="h"),
            "value": [1, 2, 3, 4, 5, 6]
        })
        
        splits = CrossValidator.leave_one_station_out(df)
        
        assert len(splits) == 3  # Three stations
        
        # Check each split
        for train, test in splits:
            test_station = test["Stn Id"].iloc[0]
            assert test_station not in train["Stn Id"].values
    
    def test_group_kfold(self):
        """Test Group K-Fold."""
        try:
            from sklearn.model_selection import GroupKFold
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        df = pd.DataFrame({
            "Stn Id": [2, 2, 7, 7, 15, 15, 2, 2],
            "value": [1, 2, 3, 4, 5, 6, 7, 8]
        })
        
        splits = CrossValidator.group_kfold(df, n_splits=3)
        
        assert len(splits) == 3
        
        # Check that groups are not split
        for train, test in splits:
            train_stations = set(train["Stn Id"].unique())
            test_stations = set(test["Stn Id"].unique())
            assert train_stations.isdisjoint(test_stations)
    
    def test_time_series_split(self, sample_dataframe):
        """Test Time Series Split."""
        try:
            from sklearn.model_selection import TimeSeriesSplit
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        splits = CrossValidator.time_series_split(sample_dataframe, n_splits=3)
        
        assert len(splits) == 3
        
        # Check ordering
        for train, test in splits:
            assert train["Date"].max() <= test["Date"].min()

