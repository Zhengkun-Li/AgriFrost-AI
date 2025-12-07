"""Data cleaning utilities: QC flags, sentinel values, missing data handling."""

from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np

from .features.constants import STATION_ID_COL, DATE_COL, HOUR_COL, HOUR_COL_ALT, get_sort_columns

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


class DataCleaner:
    """Handle QC flags, sentinel values, and missing data imputation.
    
    QC Flag Rules (CIMIS):
    - Blank/Y: Keep (high quality)
    - M: Missing data -> mark as NaN
    - R: Rejected (extreme outlier) -> mark as NaN
    - S: Severe outlier -> mark as NaN
    - Q: Questionable -> configurable (default: mark as NaN)
    - P: Provisional -> configurable (default: mark as NaN)
    """

    # Default QC mapping: True means keep, False means mark as missing
    DEFAULT_QC_CONFIG = {
        "": True,  # Blank: keep
        "Y": True,  # Moderate outlier but accepted: keep
        "M": False,  # Missing: mark as NaN
        "R": False,  # Rejected: mark as NaN
        "S": False,  # Severe outlier: mark as NaN
        "Q": False,  # Questionable: mark as NaN (configurable)
        "P": False,  # Provisional: mark as NaN (configurable)
    }

    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "data_cleaning.yaml"

    def __init__(
        self,
        qc_config: Optional[Dict[str, bool]] = None,
        variable_qc_map: Optional[Dict[str, str]] = None,
        sentinel_values: Optional[List[float]] = None,
        imputation_config: Optional[Dict[str, Any]] = None,
        outlier_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path] = None,
    ):
        """Initialize cleaner with QC configuration.
        
        Args:
            qc_config: Custom QC flag mapping.
            variable_qc_map: Mapping between variable columns and QC columns.
            sentinel_values: Sentinel values to replace with NaN.
            imputation_config: Configuration for imputation strategy.
            outlier_config: Configuration for outlier handling.
            config: Full configuration dict overriding file-based config.
            config_path: Optional explicit path to YAML config file.
        """
        self._config = self._load_config(config=config, config_path=config_path)
        self.qc_config = qc_config or self._config.get("qc_mapping", self.DEFAULT_QC_CONFIG).copy()
        self.variable_qc_map = variable_qc_map or self._config.get("variable_qc_map", {}) or {}
        self.sentinel_values = sentinel_values or self._config.get("sentinel_values", [-6999, -9999])
        self.imputation_config = imputation_config or self._config.get(
            "impute", {"enabled": True, "strategy": "forward_fill", "columns": None}
        )
        self.outlier_config = outlier_config or self._config.get(
            "outlier", {"enabled": False, "method": "iqr", "factor": 1.5, "mode": "nan", "columns": None}
        )

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configuration files.")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_config(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        if config is not None:
            return json.loads(json.dumps(config))
        resolved_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        if resolved_path and resolved_path.exists():
            return self._load_yaml(resolved_path)
        return {}

    def get_config_snapshot(self) -> Dict[str, Any]:
        """Return effective configuration for reproducibility."""
        snapshot = {
            "qc_mapping": self.qc_config,
            "variable_qc_map": self.variable_qc_map,
            "sentinel_values": self.sentinel_values,
            "impute": self.imputation_config,
            "outlier": self.outlier_config,
        }
        return json.loads(json.dumps(snapshot))

    def get_config_hash(self) -> str:
        """Return hash of effective configuration."""
        snapshot = self.get_config_snapshot()
        serialized = json.dumps(snapshot, sort_keys=True).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _infer_variable_qc_map(
        self, 
        df: pd.DataFrame,
        qc_columns: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Infer mapping between variable columns and QC columns.
        
        Uses multiple strategies in order of preference:
        1. Named pattern matching (e.g., "qc_{variable}" or "{variable}_qc")
        2. Adjacency-based inference (fallback for CIMIS-style "var,qc,var,qc" pattern)
        
        Args:
            df: Input DataFrame.
            qc_columns: Optional list of QC column names to constrain the mapping.
                If provided, only these columns will be considered as QC columns.
                If None, all columns starting with "qc" will be considered.
        
        Returns:
            Dictionary mapping variable column names to their QC column names.
        """
        variable_qc_map: Dict[str, str] = {}
        columns = list(df.columns)
        
        # Determine QC columns set
        if qc_columns is not None:
            # Use provided qc_columns list (filter to existing columns only)
            qc_columns_set = {col for col in qc_columns if col in columns}
        else:
            # Auto-detect: all columns starting with "qc"
            qc_columns_set = {col for col in columns if isinstance(col, str) and col.lower().startswith("qc")}
        
        # Strategy 1: Pattern-based matching
        # Try common naming patterns: qc_{variable}, {variable}_qc, qc_{variable}_qc
        for var_col in columns:
            if var_col in qc_columns_set:
                continue
            if not isinstance(var_col, str):
                continue
            
            # Try different naming patterns
            var_lower = var_col.lower()
            patterns_to_try = [
                f"qc_{var_col}",  # qc_Air Temp (C)
                f"qc{var_col}",   # qcAir Temp (C)
                f"{var_col}_qc",  # Air Temp (C)_qc
                "qc",             # Generic "qc" column (fallback)
            ]
            
            # Also try simplified versions for complex column names
            # Extract key parts (e.g., "Air Temp" from "Air Temp (C)")
            if "(" in var_col:
                base_name = var_col.split("(")[0].strip()
                patterns_to_try.extend([
                    f"qc_{base_name}",
                    f"{base_name}_qc",
                ])
            
            for pattern in patterns_to_try:
                if pattern in columns and pattern in qc_columns_set:
                    variable_qc_map[var_col] = pattern
                    break
        
        # Strategy 2: Adjacency-based inference (CIMIS pattern: var,qc,var,qc...)
        # Only use this for columns not already mapped
        unmapped_vars = [col for col in columns if col not in variable_qc_map and col not in qc_columns_set]
        for idx, var_col in enumerate(unmapped_vars):
            col_idx = columns.index(var_col)
            next_idx = col_idx + 1
            if next_idx < len(columns):
                next_col = columns[next_idx]
                # Check if next column is a QC column (within qc_columns_set constraint)
                if isinstance(next_col, str) and next_col in qc_columns_set:
                    # Only add if not already mapped and next column is a QC column
                    if var_col not in variable_qc_map:
                        variable_qc_map[var_col] = next_col
        
        return variable_qc_map

    def apply_qc_filter(
        self,
        df: pd.DataFrame,
        variable_qc_map: Optional[Dict[str, str]] = None,
        qc_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Filter data based on QC flags.
        
        Args:
            df: Input DataFrame.
            variable_qc_map: Optional explicit mapping between variable columns and
                their QC columns. When None, falls back to inference.
            qc_columns: Optional list of QC column names to constrain the mapping.
                If provided, only these columns will be considered as QC columns.
                If None, all columns starting with "qc" will be considered.
        
        Returns:
            DataFrame with low-quality values marked as NaN.
        """
        df = df.copy()
        
        # If qc_columns is provided, use it to constrain the inference
        if variable_qc_map is None and qc_columns is None:
            # Use stored mapping or infer
            inferred_map = self.variable_qc_map or self._infer_variable_qc_map(df)
        elif variable_qc_map is not None:
            # Use explicit mapping
            inferred_map = variable_qc_map
        else:
            # Infer with qc_columns constraint
            inferred_map = self._infer_variable_qc_map(df, qc_columns=qc_columns)
        
        if not inferred_map:
            return df
        
        qc_lookup = {str(k).strip(): v for k, v in self.qc_config.items()}
        
        for var_col, qc_col in inferred_map.items():
            if var_col not in df.columns or qc_col not in df.columns:
                continue
            
            flags = (
                df[qc_col]
                .astype(str)
                .str.strip()
                .replace("nan", "")
            )
            keep_mask = flags.map(qc_lookup).fillna(True).astype(bool)
            df.loc[~keep_mask, var_col] = np.nan
        
        return df

    def handle_sentinels(self, df: pd.DataFrame, sentinel_values: Optional[List[float]] = None) -> pd.DataFrame:
        """Replace sentinel values with NaN.
        
        Args:
            df: Input DataFrame.
            sentinel_values: List of sentinel values to replace. Default: [-6999, -9999].
        
        Returns:
            DataFrame with sentinels replaced by NaN.
        """
        df = df.copy()
        
        if sentinel_values is None:
            sentinel_values = self.sentinel_values
        
        # Replace sentinel values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].replace(sentinel_values, np.nan)
        
        return df

    def handle_outliers(self, df: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = "iqr",
                       factor: float = 3.0,
                       mode: str = "nan") -> pd.DataFrame:
        """Remove or cap outliers using IQR or Z-score method.
        
        Args:
            df: Input DataFrame.
            columns: Columns to process. If None, process all numeric columns.
            method: Method to use ("iqr" or "zscore").
            factor: Factor for outlier detection (3.0 for z-score, 1.5 for IQR multiplier).
            mode: How to handle detected outliers ("nan" to set NaN, "clip" to clamp values).
        
        Returns:
            DataFrame with outliers handled.
        """
        df = df.copy()
        
        if columns is None:
            columns = self.outlier_config.get("columns")
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        mode = mode.lower()
        if mode not in {"nan", "clip"}:
            raise ValueError("mode must be either 'nan' or 'clip'")
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if pd.isna(IQR) or IQR == 0:
                    continue
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == "zscore":
                std = df[col].std()
                if std == 0 or pd.isna(std):
                    continue
                z_scores = np.abs((df[col] - df[col].mean()) / std)
                mask = z_scores > factor
            else:
                raise ValueError("method must be 'iqr' or 'zscore'")
            
            if mode == "clip" and method == "iqr":
                df[col] = df[col].clip(lower_bound, upper_bound)
            elif mode == "clip":
                # For z-score, compute bounds based on mean ± factor * std
                mean = df[col].mean()
                lower_bound = mean - factor * std
                upper_bound = mean + factor * std
                df[col] = df[col].clip(lower_bound, upper_bound)
            else:
                df.loc[mask, col] = np.nan
        
        return df

    def _sorted_time_columns(self, df: pd.DataFrame) -> List[str]:
        """Build ordered list of time columns available in dataframe.
        
        Uses constants from features.constants for consistency.
        """
        time_cols: List[str] = []
        if DATE_COL in df.columns:
            time_cols.append(DATE_COL)
        if HOUR_COL in df.columns:
            time_cols.append(HOUR_COL)
        elif HOUR_COL_ALT in df.columns:
            time_cols.append(HOUR_COL_ALT)
        return time_cols

    def _sort_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort dataframe by station and temporal columns if present.
        
        Uses get_sort_columns from features.constants for consistency.
        """
        # Detect hour column
        hour_col = None
        if HOUR_COL in df.columns:
            hour_col = HOUR_COL
        elif HOUR_COL_ALT in df.columns:
            hour_col = HOUR_COL_ALT
        
        sort_cols = get_sort_columns(
            station_col=STATION_ID_COL if STATION_ID_COL in df.columns else None,
            date_col=DATE_COL if DATE_COL in df.columns else None,
            hour_col=hour_col,
        )
        
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        return df

    def impute_missing(self, df: pd.DataFrame, 
                      strategy: Optional[str] = None,
                      columns: Optional[List[str]] = None,
                      **kwargs) -> pd.DataFrame:
        """Impute missing values using specified strategy.
        
        Args:
            df: Input DataFrame.
            strategy: Imputation strategy:
                - "forward_fill": Forward fill (default for time series)
                - "backward_fill": Backward fill
                - "mean": Fill with column mean
                - "median": Fill with column median
                - "interpolate": Linear interpolation
            columns: Columns to impute. If None, impute all numeric columns.
            **kwargs: Additional arguments for imputation method.
        
        Returns:
            DataFrame with imputed values.
        """
        df = df.copy()
        
        if columns is None:
            columns = self.imputation_config.get("columns")
            if columns is not None:
                columns = [col for col in columns if col in df.columns]
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if strategy is None:
            strategy = self.imputation_config.get("strategy", "forward_fill")
        
        # Ensure data is sorted by time for time-based imputation
        df = self._sort_by_time(df)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if strategy == "forward_fill":
                # Forward fill within each station
                if STATION_ID_COL in df.columns:
                    df[col] = df.groupby(STATION_ID_COL)[col].ffill(**kwargs)
                else:
                    df[col] = df[col].ffill(**kwargs)
                    
            elif strategy == "backward_fill":
                if STATION_ID_COL in df.columns:
                    df[col] = df.groupby(STATION_ID_COL)[col].bfill(**kwargs)
                else:
                    df[col] = df[col].bfill(**kwargs)
                    
            elif strategy == "mean":
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                
            elif strategy == "median":
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                
            elif strategy == "interpolate":
                if STATION_ID_COL in df.columns:
                    df[col] = df.groupby(STATION_ID_COL)[col].transform(
                        lambda x: x.interpolate(method="linear", **kwargs)
                    )
                else:
                    df[col] = df[col].interpolate(method="linear", **kwargs)
        
        return df

    def clean_pipeline(self, df: pd.DataFrame, 
                      apply_qc: Optional[bool] = None,
                      handle_sentinels: Optional[bool] = None,
                      handle_outliers: Optional[bool] = None,
                      impute_missing: Optional[bool] = None,
                      imputation_strategy: Optional[str] = None) -> pd.DataFrame:
        """Complete cleaning pipeline.
        
        Args:
            df: Input DataFrame.
            apply_qc: Whether to apply QC filtering.
            handle_sentinels: Whether to replace sentinel values.
            handle_outliers: Whether to remove outliers.
            impute_missing: Whether to impute missing values.
            imputation_strategy: Strategy for imputation.
        
        Returns:
            Cleaned DataFrame.
        """
        df_cleaned = df.copy()
        
        if apply_qc is None:
            apply_qc = True
        if handle_sentinels is None:
            handle_sentinels = True
        if handle_outliers is None:
            handle_outliers = bool(self.outlier_config.get("enabled", False))
        if impute_missing is None:
            impute_missing = bool(self.imputation_config.get("enabled", True))
        if imputation_strategy is None:
            imputation_strategy = self.imputation_config.get("strategy", "forward_fill")
        
        if apply_qc:
            df_cleaned = self.apply_qc_filter(df_cleaned, variable_qc_map=self.variable_qc_map)
        
        if handle_sentinels:
            df_cleaned = self.handle_sentinels(df_cleaned, sentinel_values=self.sentinel_values)
        
        if handle_outliers:
            df_cleaned = self.handle_outliers(
                df_cleaned,
                columns=self.outlier_config.get("columns"),
                method=self.outlier_config.get("method", "iqr"),
                factor=self.outlier_config.get("factor", 1.5),
                mode=self.outlier_config.get("mode", "nan"),
            )
        
        if impute_missing:
            df_cleaned = self.impute_missing(
                df_cleaned,
                strategy=imputation_strategy,
                columns=self.imputation_config.get("columns"),
            )
        
        return df_cleaned

