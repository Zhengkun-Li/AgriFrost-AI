"""Feature selection utilities for frost forecasting.

This module provides functions to select and filter features before training:
1. Remove low-importance features (based on feature importance analysis)
2. Remove highly correlated features
3. Remove features with high missing rates
4. Remove low-variance features
"""

import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json

_logger = logging.getLogger(__name__)


class FeatureSelector:
    """Feature selection utility class."""
    
    def __init__(
        self,
        min_importance: float = 0.0,
        max_correlation: float = 0.95,
        max_missing_rate: float = 0.5,
        min_variance: float = 0.0
    ):
        """Initialize feature selector.
        
        Args:
            min_importance: Minimum feature importance to keep (0-1)
            max_correlation: Maximum correlation to keep (0-1)
            max_missing_rate: Maximum missing rate to keep (0-1)
            min_variance: Minimum variance to keep
        """
        self.min_importance = min_importance
        self.max_correlation = max_correlation
        self.max_missing_rate = max_missing_rate
        self.min_variance = min_variance
        self.selected_features = None
        self.removed_features = {}
    
    def select_by_importance(
        self,
        X: pd.DataFrame,
        feature_importance: pd.DataFrame,
        top_k: Optional[int] = None,
        min_importance: Optional[float] = None
    ) -> List[str]:
        """Select features based on importance.
        
        Args:
            X: Feature DataFrame
            feature_importance: DataFrame with columns ['feature', 'importance'] or ['feature_name', 'importance']
            top_k: Keep top K features (if None, use min_importance)
            min_importance: Minimum importance to keep (if None, use self.min_importance)
        
        Returns:
            List of selected feature names
        """
        if min_importance is None:
            min_importance = self.min_importance
        
        # Normalize column names (handle both 'feature' and 'feature_name')
        if 'feature_name' in feature_importance.columns and 'feature' not in feature_importance.columns:
            feature_importance = feature_importance.rename(columns={'feature_name': 'feature'})
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        if top_k is not None:
            selected = feature_importance.head(top_k)['feature'].tolist()
        else:
            # Use min_importance threshold
            max_importance = feature_importance['importance'].max()
            threshold = max_importance * min_importance
            selected = feature_importance[feature_importance['importance'] >= threshold]['feature'].tolist()
        
        # Filter to only include features that exist in X
        selected = [f for f in selected if f in X.columns]
        
        # Track removed features
        removed = [f for f in X.columns if f not in selected]
        self.removed_features['low_importance'] = removed
        
        return selected
    
    def select_by_correlation(
        self,
        X: pd.DataFrame,
        max_correlation: Optional[float] = None,
        target_col: Optional[str] = None
    ) -> List[str]:
        """Remove highly correlated features.
        
        Args:
            X: Feature DataFrame
            max_correlation: Maximum correlation to keep (if None, use self.max_correlation)
            target_col: Optional target column to compute correlation with
        
        Returns:
            List of selected feature names
        """
        if max_correlation is None:
            max_correlation = self.max_correlation
        
        # Compute correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated feature pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = []
        for column in upper_triangle.columns:
            if column in to_drop:
                continue
            # Find features highly correlated with this one
            highly_correlated = upper_triangle[column][upper_triangle[column] > max_correlation]
            if len(highly_correlated) > 0:
                # Keep the feature with highest correlation to target (if provided)
                # Otherwise, keep the first one and drop others
                if target_col and target_col in X.columns:
                    # Compute correlation with target for all highly correlated features
                    target_corr = X[[column] + highly_correlated.index.tolist()].corrwith(X[target_col]).abs()
                    # Keep the one with highest correlation to target
                    keep_feature = target_corr.idxmax()
                    to_remove = [f for f in [column] + highly_correlated.index.tolist() if f != keep_feature]
                    to_drop.extend(to_remove)
                else:
                    # Drop all except the first one
                    to_drop.extend(highly_correlated.index.tolist())
        
        # Remove duplicates
        to_drop = list(set(to_drop))
        
        # Select features
        selected = [f for f in X.columns if f not in to_drop]
        
        # Track removed features
        self.removed_features['high_correlation'] = to_drop
        
        return selected
    
    def select_by_missing_rate(
        self,
        X: pd.DataFrame,
        max_missing_rate: Optional[float] = None
    ) -> List[str]:
        """Remove features with high missing rates.
        
        Args:
            X: Feature DataFrame
            max_missing_rate: Maximum missing rate to keep (if None, use self.max_missing_rate)
        
        Returns:
            List of selected feature names
        """
        if max_missing_rate is None:
            max_missing_rate = self.max_missing_rate
        
        # Compute missing rates
        missing_rates = X.isna().mean()
        
        # Select features with low missing rate
        selected = missing_rates[missing_rates <= max_missing_rate].index.tolist()
        
        # Track removed features
        removed = missing_rates[missing_rates > max_missing_rate].index.tolist()
        self.removed_features['high_missing'] = removed
        
        return selected
    
    def select_by_variance(
        self,
        X: pd.DataFrame,
        min_variance: Optional[float] = None
    ) -> List[str]:
        """Remove low-variance features.
        
        Args:
            X: Feature DataFrame
            min_variance: Minimum variance to keep (if None, use self.min_variance)
        
        Returns:
            List of selected feature names
        """
        if min_variance is None:
            min_variance = self.min_variance
        
        # Compute variance
        variances = X.var()
        
        # Select features with sufficient variance
        selected = variances[variances >= min_variance].index.tolist()
        
        # Track removed features
        removed = variances[variances < min_variance].index.tolist()
        self.removed_features['low_variance'] = removed
        
        return selected
    
    def select_features(
        self,
        X: pd.DataFrame,
        feature_importance: Optional[pd.DataFrame] = None,
        top_k: Optional[int] = None,
        min_importance: Optional[float] = None,
        remove_correlated: bool = True,
        remove_high_missing: bool = True,
        remove_low_variance: bool = False,
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Select features using multiple criteria.
        
        Args:
            X: Feature DataFrame
            feature_importance: DataFrame with columns ['feature', 'importance']
            top_k: Keep top K features (if None, use min_importance)
            min_importance: Minimum importance to keep (if None, use self.min_importance)
            remove_correlated: Whether to remove highly correlated features
            remove_high_missing: Whether to remove features with high missing rates
            remove_low_variance: Whether to remove low-variance features
            target_col: Optional target column for correlation analysis
        
        Returns:
            DataFrame with selected features
        """
        selected_features = list(X.columns)
        
        # Step 1: Remove high missing rate features
        if remove_high_missing:
            selected_features = self.select_by_missing_rate(X[selected_features])
            _logger.debug(f"After removing high missing rate: {len(selected_features)} features")
        
        # Step 2: Remove low variance features
        if remove_low_variance:
            selected_features = self.select_by_variance(X[selected_features])
            _logger.debug(f"After removing low variance: {len(selected_features)} features")
        
        # Step 3: Remove highly correlated features
        if remove_correlated:
            selected_features = self.select_by_correlation(
                X[selected_features],
                target_col=target_col
            )
            _logger.debug(f"After removing highly correlated: {len(selected_features)} features")
        
        # Step 4: Select by importance (if provided)
        if feature_importance is not None:
            # Normalize column names (handle both 'feature' and 'feature_name')
            if 'feature_name' in feature_importance.columns and 'feature' not in feature_importance.columns:
                feature_importance = feature_importance.rename(columns={'feature_name': 'feature'})
            
            # Filter importance to only include remaining features
            feature_importance_filtered = feature_importance[
                feature_importance['feature'].isin(selected_features)
            ]
            selected_features = self.select_by_importance(
                X[selected_features],
                feature_importance_filtered,
                top_k=top_k,
                min_importance=min_importance
            )
            _logger.debug(f"After selecting by importance: {len(selected_features)} features")
        
        # Store selected features
        self.selected_features = selected_features
        
        # Return selected features DataFrame
        return X[selected_features]
    
    def get_selection_report(self) -> Dict:
        """Get report on feature selection.
        
        Returns:
            Dictionary with selection statistics
        """
        report = {
            "n_selected": len(self.selected_features) if self.selected_features else 0,
            "removed_features": {
                category: len(features) for category, features in self.removed_features.items()
            },
            "removed_features_detail": self.removed_features
        }
        return report
    
    def save_selection_report(self, output_path: Path):
        """Save selection report to file.
        
        Args:
            output_path: Path to save report
        """
        report = self.get_selection_report()
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)


def select_features_from_importance(
    X: pd.DataFrame,
    feature_importance_path: Path,
    top_k: Optional[int] = None,
    min_importance_ratio: float = 0.01
) -> Tuple[pd.DataFrame, List[str]]:
    """Select features based on feature importance file.
    
    Args:
        X: Feature DataFrame
        feature_importance_path: Path to feature importance CSV file
        top_k: Keep top K features (if None, use min_importance_ratio)
        min_importance_ratio: Minimum importance ratio to keep (0-1)
    
    Returns:
        Tuple of (selected features DataFrame, removed feature names)
    """
    # Load feature importance
    feature_importance = pd.read_csv(feature_importance_path)
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Select features
    if top_k is not None:
        selected_features = feature_importance.head(top_k)['feature'].tolist()
    else:
        # Use min_importance_ratio threshold
        max_importance = feature_importance['importance'].max()
        threshold = max_importance * min_importance_ratio
        selected_features = feature_importance[
            feature_importance['importance'] >= threshold
        ]['feature'].tolist()
    
    # Filter to only include features that exist in X
    selected_features = [f for f in selected_features if f in X.columns]
    
    # Get removed features
    removed_features = [f for f in X.columns if f not in selected_features]
    
    return X[selected_features], removed_features


def remove_highly_correlated_features(
    X: pd.DataFrame,
    max_correlation: float = 0.95
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove highly correlated features.
    
    Args:
        X: Feature DataFrame
        max_correlation: Maximum correlation to keep (0-1)
    
    Returns:
        Tuple of (selected features DataFrame, removed feature names)
    """
    # Compute correlation matrix
    corr_matrix = X.corr().abs()
    
    # Find highly correlated feature pairs
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = []
    for column in upper_triangle.columns:
        if column in to_drop:
            continue
        # Find features highly correlated with this one
        highly_correlated = upper_triangle[column][upper_triangle[column] > max_correlation]
        if len(highly_correlated) > 0:
            # Drop all except the first one
            to_drop.extend(highly_correlated.index.tolist())
    
    # Remove duplicates
    to_drop = list(set(to_drop))
    
    # Select features
    selected_features = [f for f in X.columns if f not in to_drop]
    
    return X[selected_features], to_drop


def load_feature_selection_config(config_path: Path) -> Dict:
    """Load a feature selection configuration JSON."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Feature selection config not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def select_features_with_config(
    X: pd.DataFrame,
    config: Dict,
    feature_importance: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, FeatureSelector]:
    """Apply feature selection based on a configuration dictionary.
    
    Args:
        X: Feature DataFrame.
        config: Configuration dictionary (or JSON-loaded object) supporting:
            - min_importance / max_correlation / max_missing_rate / min_variance
            - remove_correlated / remove_high_missing / remove_low_variance (bool)
            - top_k / importance_threshold
            - importance_path: CSV path for feature importance
            - save_report / report_path
        feature_importance: Optional importance DataFrame (overrides importance_path).
    
    Returns:
        Tuple of (selected features DataFrame, FeatureSelector instance).
    """
    selector = FeatureSelector(
        min_importance=config.get("min_importance", 0.0),
        max_correlation=config.get("max_correlation", 0.95),
        max_missing_rate=config.get("max_missing_rate", 0.5),
        min_variance=config.get("min_variance", 0.0)
    )
    
    importance_df = feature_importance
    if importance_df is None and config.get("importance_path"):
        importance_path = Path(config["importance_path"])
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)
        else:
            _logger.warning(f"Importance file not found: {importance_path}")
    
    selected_df = selector.select_features(
        X,
        feature_importance=importance_df,
        top_k=config.get("top_k"),
        min_importance=config.get("importance_threshold"),
        remove_correlated=config.get("remove_correlated", True),
        remove_high_missing=config.get("remove_high_missing", True),
        remove_low_variance=config.get("remove_low_variance", False),
        target_col=config.get("target_column")
    )
    
    if config.get("save_report"):
        report_path = Path(config.get("report_path", "feature_selection_report.json"))
        selector.save_selection_report(report_path)
    
    return selected_df, selector

