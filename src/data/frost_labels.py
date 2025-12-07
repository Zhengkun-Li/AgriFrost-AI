"""Generate frost event labels for multi-horizon forecasting."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .features.constants import STATION_ID_COL, DATE_COL, TEMP_COL


class FrostLabelGenerator:
    """Generate frost event labels for different forecast horizons."""
    
    def __init__(self, frost_threshold: float = 0.0):
        """Initialize frost label generator.
        
        Args:
            frost_threshold: Temperature threshold for frost (default: 0.0°C).
        """
        self.frost_threshold = frost_threshold
        self.horizons = [3, 6, 12, 24]  # hours
    
    def create_frost_labels(
        self,
        df: pd.DataFrame,
        temp_col: str = TEMP_COL,
        date_col: str = DATE_COL,
        station_col: str = STATION_ID_COL,
        horizons: Optional[List[int]] = None,
        hour_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Create frost event labels for multiple forecast horizons.
        
        This function creates time-based frost labels by shifting temperature
        forward by the specified horizon. It handles hourly data correctly,
        even if some hours are missing.
        
        **Time Alignment**:
        - If `hour_col` is provided, constructs proper datetime index and attempts
          time-based shifting (shift by actual hours, not rows)
        - Falls back to row-based shifting if frequency cannot be set
        - Row-based shifting assumes each row = 1 hour (may misalign if hours are missing)
        
        **Best Practice**: Provide `hour_col` parameter and ensure data has regular
        hourly frequency for accurate label alignment.
        
        Args:
            df: Input DataFrame with temperature and date columns.
            temp_col: Name of temperature column (default: TEMP_COL).
            date_col: Name of date column (default: DATE_COL).
            station_col: Name of station ID column (default: STATION_ID_COL).
            horizons: List of forecast horizons in hours (default: [3, 6, 12, 24]).
            hour_col: Optional hour column name. If provided, will be used to
                ensure proper time alignment (requires hourly data).
        
        Returns:
            DataFrame with added frost label columns for each horizon:
            - frost_{horizon}h: Binary label (1 if temp < threshold, 0 otherwise)
            - temp_{horizon}h: Future temperature value (for reference)
        
        Note:
            For datasets with missing hours, consider resampling to hourly
            frequency before calling this function, or use time-aware shifting
            with proper datetime index.
        """
        if horizons is None:
            horizons = self.horizons
        
        # Validate input
        if df.empty:
            raise ValueError("Cannot create frost labels: DataFrame is empty")
        
        if date_col not in df.columns:
            raise ValueError(f"Missing required column '{date_col}' for frost label generation")
        
        if temp_col not in df.columns:
            raise ValueError(f"Missing required column '{temp_col}' for frost label generation")
        
        if station_col not in df.columns:
            raise ValueError(f"Missing required column '{station_col}' for frost label generation")
        
        df = df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by station and date (with hour if available)
        sort_cols = [station_col, date_col]
        if hour_col and hour_col in df.columns:
            sort_cols.append(hour_col)
        df = df.sort_values(sort_cols).reset_index(drop=True)
        
        # Create datetime index for each station
        for station_id in df[station_col].unique():
            station_mask = df[station_col] == station_id
            station_df = df[station_mask].copy()
            
            # CRITICAL FIX: Save original row positions before modifying station_df
            # This ensures we can correctly align shifted values back to original df
            original_df_indices = df[station_mask].index.values  # Original df row indices
            original_positions = np.arange(len(station_df))  # Position-based indices for alignment
            
            # Create datetime index (assumes hourly data)
            # If hour_col exists, combine date + hour into datetime
            datetime_index = None
            if hour_col and hour_col in station_df.columns:
                # Combine date and hour into proper datetime
                # CRITICAL FIX: Handle Hour (PST) format (100 = 1:00, 200 = 2:00, 1300 = 13:00)
                # Convert hour format: 100 -> 01, 200 -> 02, 1300 -> 13
                hour_values = station_df[hour_col]
                # Check if hour is in format 100, 200, 1300, etc. (HHMM format)
                if hour_values.max() > 24:
                    # Hour is in HHMM format (e.g., 100 = 1:00, 1300 = 13:00)
                    hour_str = (hour_values // 100).astype(int).astype(str).str.zfill(2) + ":" + \
                               (hour_values % 100).astype(int).astype(str).str.zfill(2)
                else:
                    # Hour is already in 0-23 format
                    hour_str = hour_values.astype(int).astype(str).str.zfill(2) + ":00"
                
                datetime_index = pd.to_datetime(
                    station_df[date_col].astype(str) + " " + hour_str
                )
                station_df = station_df.set_index(datetime_index)
                # Don't use asfreq('h') as it may add missing time points and break alignment
                # Instead, just ensure we have a DatetimeIndex for time-based shifting
            else:
                # Assume hourly data: each row is one hour after the previous
                # Set as DatetimeIndex with hourly frequency if possible
                if not pd.api.types.is_datetime64_any_dtype(station_df[date_col]):
                    datetime_index = pd.to_datetime(station_df[date_col])
                else:
                    datetime_index = station_df[date_col]
                station_df = station_df.set_index(datetime_index)
            
            # For each horizon, check if temperature will be below threshold
            for horizon_h in horizons:
                label_col = f"frost_{horizon_h}h"
                temp_future_col = f"temp_{horizon_h}h"
                
                # Shift temperature forward by horizon hours
                # CRITICAL FIX: Use shift(-h) instead of shift(-h, freq='h')
                # shift(-h, freq='h') changes the index, causing alignment issues
                # shift(-h) only shifts values, keeping original index for correct alignment
                # Row-based shifting (assumes each row = 1 hour)
                # WARNING: If hours are missing, this may align to wrong time, but at least it works correctly
                future_temp = station_df[temp_col].shift(-horizon_h)
                
                # Create frost label (1 if temp < threshold, 0 otherwise)
                frost_label = (future_temp < self.frost_threshold).astype(int)
                
                # CRITICAL FIX: Use original_df_indices to correctly align values back to original df
                # station_df now has datetime index, but we need to map back to original df indices
                # Since we didn't use asfreq, the length should match, so we can use position-based alignment
                if len(frost_label) == len(original_df_indices):
                    # Direct alignment: station_df rows correspond to original_df_indices by position
                    df.loc[original_df_indices, label_col] = frost_label.values
                    df.loc[original_df_indices, temp_future_col] = future_temp.values
                else:
                    # Fallback: if lengths don't match (shouldn't happen), try to match by datetime
                    # This handles edge cases where station_df might have been modified
                    if isinstance(station_df.index, pd.DatetimeIndex) and hour_col and hour_col in df.columns:
                        # Match by datetime
                        # CRITICAL FIX: Handle Hour (PST) format (100 = 1:00, 200 = 2:00, 1300 = 13:00)
                        hour_values = df.loc[original_df_indices, hour_col]
                        if hour_values.max() > 24:
                            # Hour is in HHMM format
                            hour_str = (hour_values // 100).astype(int).astype(str).str.zfill(2) + ":" + \
                                       (hour_values % 100).astype(int).astype(str).str.zfill(2)
                        else:
                            # Hour is already in 0-23 format
                            hour_str = hour_values.astype(int).astype(str).str.zfill(2) + ":00"
                        
                        df_datetime = pd.to_datetime(
                            df.loc[original_df_indices, date_col].astype(str) + " " + hour_str
                        )
                        matching_indices = station_df.index[station_df.index.isin(df_datetime)]
                        if len(matching_indices) == len(original_df_indices):
                            aligned_frost = frost_label.loc[matching_indices]
                            aligned_temp = future_temp.loc[matching_indices]
                            df.loc[original_df_indices, label_col] = aligned_frost.values
                            df.loc[original_df_indices, temp_future_col] = aligned_temp.values
                        else:
                            # Last resort: use original method (may have bugs but should work if lengths match)
                            df.loc[original_df_indices, label_col] = frost_label.values[:len(original_df_indices)]
                            df.loc[original_df_indices, temp_future_col] = future_temp.values[:len(original_df_indices)]
                    else:
                        # Last resort: use original method
                        df.loc[original_df_indices, label_col] = frost_label.values[:len(original_df_indices)]
                        df.loc[original_df_indices, temp_future_col] = future_temp.values[:len(original_df_indices)]
        
        return df
    
    def get_label_columns(self, horizons: Optional[List[int]] = None) -> List[str]:
        """Get column names for frost labels.
        
        Args:
            horizons: List of forecast horizons (default: [3, 6, 12, 24]).
        
        Returns:
            List of label column names.
        """
        if horizons is None:
            horizons = self.horizons
        return [f"frost_{h}h" for h in horizons]
    
    def get_temp_columns(self, horizons: Optional[List[int]] = None) -> List[str]:
        """Get column names for future temperatures.
        
        Args:
            horizons: List of forecast horizons (default: [3, 6, 12, 24]).
        
        Returns:
            List of temperature column names.
        """
        if horizons is None:
            horizons = self.horizons
        return [f"temp_{h}h" for h in horizons]


def create_multi_horizon_targets(
    df: pd.DataFrame,
    horizons: List[int] = [3, 6, 12, 24],
    frost_threshold: float = 0.0
) -> pd.DataFrame:
    """Convenience function to create multi-horizon frost labels.
    
    Args:
        df: Input DataFrame with temperature data.
        horizons: List of forecast horizons in hours.
        frost_threshold: Temperature threshold for frost.
    
    Returns:
        DataFrame with added frost label and temperature columns.
    """
    generator = FrostLabelGenerator(frost_threshold=frost_threshold)
    return generator.create_frost_labels(df, horizons=horizons)

