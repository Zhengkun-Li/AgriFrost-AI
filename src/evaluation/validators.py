"""Cross-validation strategies for time series and spatial data."""

import logging
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

_logger = logging.getLogger(__name__)

# Import constants for column names
try:
    from src.data.features.constants import DATE_COL, STATION_ID_COL, HOUR_COL, HOUR_COL_ALT
except ImportError:
    # Fallback if constants not available
    DATE_COL = "Date"
    STATION_ID_COL = "Stn Id"
    HOUR_COL = "Hour (PST)"
    HOUR_COL_ALT = "Hour"

try:
    from sklearn.model_selection import GroupKFold, TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class CrossValidator:
    """Handle different cross-validation strategies for time series and spatial data."""
    
    @staticmethod
    def time_split(
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        date_col: str = "Date",
        split_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Time-based split (no data leakage).
        
        Args:
            df: Input DataFrame (must have date column).
            train_ratio: Proportion of data for training (must be in (0, 1]).
                Only used if split_date is None.
            val_ratio: Proportion of data for validation (must be in (0, 1]).
                Only used if split_date is None.
            date_col: Name of date column.
            split_date: Optional date string (e.g., "2020-01-01") to split at.
                If provided, train will be all data before split_date,
                val and test will be split from remaining data using val_ratio.
        
        Returns:
            Tuple of (train_df, val_df, test_df).
        
        Raises:
            ValueError: If DataFrame is empty, date column missing, or ratios invalid.
        """
        # Input validation
        if df.empty:
            raise ValueError("DataFrame cannot be empty for time split")
        
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        
        if not (0 < train_ratio <= 1):
            raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")
        
        if not (0 < val_ratio <= 1):
            raise ValueError(f"val_ratio must be in (0, 1], got {val_ratio}")
        
        if train_ratio + val_ratio >= 1:
            raise ValueError(f"train_ratio + val_ratio must be < 1, got {train_ratio + val_ratio}")
        
        # Sort by date (and hour if available) for proper temporal ordering
        sort_cols = [date_col]
        if HOUR_COL in df.columns:
            sort_cols.append(HOUR_COL)
        elif HOUR_COL_ALT in df.columns:
            sort_cols.append(HOUR_COL_ALT)
        
        df_sorted = df.sort_values(sort_cols).reset_index(drop=True)
        n = len(df_sorted)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Ensure we have at least some samples in each split
        if train_end == 0:
            raise ValueError(f"train_ratio {train_ratio} too small for dataset size {n}")
        if val_end == train_end:
            raise ValueError(f"val_ratio {val_ratio} too small for dataset size {n}")
        if val_end >= n:
            raise ValueError(f"train_ratio + val_ratio too large for dataset size {n}")
        
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
        
        _logger.debug(f"Time split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    @staticmethod
    def leave_one_station_out(
        df: pd.DataFrame,
        station_col: str = "Stn Id"
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Leave-One-Station-Out (LOSO) cross-validation.
        
        Args:
            df: Input DataFrame.
            station_col: Name of station ID column.
        
        Returns:
            List of (train_df, test_df) tuples, one for each station.
        
        Raises:
            ValueError: If DataFrame is empty or station column missing.
        """
        # Input validation
        if df.empty:
            raise ValueError("DataFrame cannot be empty for LOSO cross-validation")
        
        if station_col not in df.columns:
            raise ValueError(f"Station column '{station_col}' not found in DataFrame")
        
        stations = df[station_col].unique()
        if len(stations) < 2:
            raise ValueError(f"Need at least 2 stations for LOSO, found {len(stations)}")
        
        splits = []
        
        for test_station in stations:
            train_df = df[df[station_col] != test_station].copy()
            test_df = df[df[station_col] == test_station].copy()
            
            if train_df.empty:
                _logger.warning(f"Train set is empty for station {test_station}. Skipping.")
                continue
            if test_df.empty:
                _logger.warning(f"Test set is empty for station {test_station}. Skipping.")
                continue
            
            # Strict temporal sorting: Sort by date and hour within each split
            # This is critical for sequence models (LSTM/GRU/TCN) and prevents temporal leakage
            sort_cols = []
            if DATE_COL in train_df.columns:
                sort_cols.append(DATE_COL)
            if HOUR_COL in train_df.columns:
                sort_cols.append(HOUR_COL)
            elif HOUR_COL_ALT in train_df.columns:
                sort_cols.append(HOUR_COL_ALT)
            
            if sort_cols:
                train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
                test_df = test_df.sort_values(sort_cols).reset_index(drop=True)
                
                # Validate temporal ordering (train should come before test chronologically)
                if DATE_COL in train_df.columns and DATE_COL in test_df.columns:
                    train_max_date = pd.to_datetime(train_df[DATE_COL]).max()
                    test_min_date = pd.to_datetime(test_df[DATE_COL]).min()
                    if train_max_date >= test_min_date:
                        _logger.warning(
                            f"Potential temporal leakage for station {test_station}: "
                            f"train_max_date ({train_max_date}) >= test_min_date ({test_min_date}). "
                            f"This may indicate data leakage in LOSO split."
                        )
            else:
                _logger.debug(f"No date/hour columns found for temporal sorting in LOSO split for station {test_station}")
            
            splits.append((train_df, test_df))
            _logger.debug(f"LOSO split for station {test_station}: train={len(train_df)}, test={len(test_df)}")
        
        if not splits:
            raise ValueError("No valid LOSO splits generated. Check station column and data.")
        
        return splits
    
    @staticmethod
    def group_kfold(
        df: pd.DataFrame,
        n_splits: int = 5,
        group_col: str = "Stn Id"
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Group K-Fold cross-validation.
        
        Args:
            df: Input DataFrame.
            n_splits: Number of folds (must be >= 2).
            group_col: Column to group by (e.g., station ID).
        
        Returns:
            List of (train_df, test_df) tuples.
        
        Raises:
            ImportError: If scikit-learn is not available.
            ValueError: If DataFrame is empty, group column missing, or n_splits invalid.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for group_kfold. Install with: pip install scikit-learn")
        
        # Input validation
        if df.empty:
            raise ValueError("DataFrame cannot be empty for group k-fold")
        
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in DataFrame")
        
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        
        # Warning: group_kfold may not be suitable for spatial generalization tasks
        # if station embeddings or station-specific features are used in feature engineering
        if group_col == STATION_ID_COL:
            _logger.warning(
                f"Using group_kfold with station column '{group_col}' may not be suitable for "
                f"spatial generalization tasks. Consider using LOSO (leave_one_station_out) instead, "
                f"especially if station embeddings or station-specific features are used."
            )
        
        unique_groups = df[group_col].nunique()
        if unique_groups < n_splits:
            _logger.warning(
                f"Number of unique groups ({unique_groups}) is less than n_splits ({n_splits}). "
                f"Some folds may be empty or contain only one group."
            )
        
        from sklearn.model_selection import GroupKFold
        groups = df[group_col].values
        gkf = GroupKFold(n_splits=n_splits)
        
        splits = []
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                _logger.warning(f"Fold {fold_idx + 1} has empty train or validation set. Skipping.")
                continue
            
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            splits.append((train_df, test_df))
            _logger.debug(f"Fold {fold_idx + 1}: train={len(train_df)}, val={len(test_df)}")
        
        if not splits:
            raise ValueError("No valid splits generated. Check group column and n_splits.")
        
        return splits
    
    @staticmethod
    def time_series_split(
        df: pd.DataFrame,
        n_splits: int = 5,
        date_col: str = "Date"
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Time Series Split cross-validation.
        
        Args:
            df: Input DataFrame (must be sorted by date).
            n_splits: Number of splits (must be >= 2).
            date_col: Name of date column.
        
        Returns:
            List of (train_df, test_df) tuples.
        
        Raises:
            ImportError: If scikit-learn is not available.
            ValueError: If DataFrame is empty, date column missing, or n_splits invalid.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for time_series_split. Install with: pip install scikit-learn")
        
        # Input validation
        if df.empty:
            raise ValueError("DataFrame cannot be empty for time series split")
        
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        n = len(df_sorted)
        if n < n_splits + 1:
            raise ValueError(f"DataFrame too small ({n} rows) for {n_splits} splits. Need at least {n_splits + 1} rows.")
        
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        splits = []
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df_sorted)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                _logger.warning(f"Fold {fold_idx + 1} has empty train or validation set. Skipping.")
                continue
            
            train_df = df_sorted.iloc[train_idx].copy()
            test_df = df_sorted.iloc[test_idx].copy()
            splits.append((train_df, test_df))
            _logger.debug(f"Fold {fold_idx + 1}: train={len(train_df)}, val={len(test_df)}")
        
        if not splits:
            raise ValueError("No valid splits generated. Check date column and n_splits.")
        
        return splits

