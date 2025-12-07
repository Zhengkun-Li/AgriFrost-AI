"""Multi-horizon evaluation module for frost forecasting.

This module handles:
- Evaluation across multiple forecast horizons (3h, 6h, 12h, 24h)
- Aggregation of metrics across horizons
- Horizon-specific performance analysis
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pandas as pd
import numpy as np

_logger = logging.getLogger(__name__)

from src.evaluation.metrics import MetricsCalculator


class MultiHorizonEvaluator:
    """Evaluate model performance across multiple forecast horizons.
    
    This evaluator aggregates results from multiple horizons and provides
    comprehensive performance analysis across different forecast windows.
    """
    
    def __init__(
        self,
        horizons: List[int],
        output_dir: Optional[Path] = None
    ):
        """Initialize multi-horizon evaluator.
        
        Args:
            horizons: List of forecast horizons in hours (e.g., [3, 6, 12, 24]).
            output_dir: Optional output directory for saving results.
        """
        if not horizons:
            raise ValueError("horizons cannot be empty")
        if not all(isinstance(h, int) and h > 0 for h in horizons):
            raise ValueError(f"All horizons must be positive integers, got {horizons}")
        
        self.horizons = sorted(horizons)
        self.output_dir = Path(output_dir) if output_dir else None
        
        _logger.info(f"Initialized MultiHorizonEvaluator for horizons: {self.horizons}h")
    
    def evaluate(
        self,
        results_dict: Dict[int, Dict[str, Any]],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate results across all horizons.
        
        Args:
            results_dict: Dictionary mapping horizon -> metrics dictionary.
                Expected format:
                {
                    horizon: {
                        "classification": {...},  # Optional
                        "regression": {...},      # Optional
                        "frost_metrics": {...},   # Legacy format
                        "temp_metrics": {...}     # Legacy format
                    }
                }
            model_name: Optional model name for logging.
        
        Returns:
            Dictionary with aggregated metrics and per-horizon results:
            {
                "horizons": {
                    "3h": {...},
                    "6h": {...},
                    ...
                },
                "summary": {
                    "classification": {...},  # Aggregated across horizons
                    "regression": {...}       # Aggregated across horizons
                },
                "best_horizon": {...}  # Horizon with best performance
            }
        """
        if not results_dict:
            raise ValueError("results_dict cannot be empty")
        
        # Normalize results format
        normalized_results = {}
        for horizon in self.horizons:
            if horizon not in results_dict:
                _logger.warning(f"Results for {horizon}h horizon not found. Skipping.")
                continue
            
            horizon_results = results_dict[horizon]
            
            # Support both new structured format and legacy format
            if "classification" in horizon_results and "regression" in horizon_results:
                # New structured format
                normalized_results[horizon] = horizon_results
            elif "frost_metrics" in horizon_results and "temp_metrics" in horizon_results:
                # Legacy format: convert to structured format
                normalized_results[horizon] = {
                    "classification": horizon_results["frost_metrics"],
                    "regression": horizon_results["temp_metrics"]
                }
            else:
                _logger.warning(
                    f"Unrecognized result format for {horizon}h horizon. "
                    f"Expected 'classification'/'regression' or 'frost_metrics'/'temp_metrics'."
                )
                continue
        
        if not normalized_results:
            raise ValueError("No valid results found for any horizon")
        
        # Aggregate metrics across horizons
        aggregated = self._aggregate_metrics(normalized_results)
        
        # Find best horizon
        best_horizon = self._find_best_horizon(normalized_results)
        
        # Format results
        formatted_results = {}
        for horizon, metrics in normalized_results.items():
            formatted_results[f"{horizon}h"] = metrics
        
        results = {
            "horizons": formatted_results,
            "summary": aggregated,
            "best_horizon": best_horizon,
            "model_name": model_name
        }
        
        # Save results if output_dir is provided
        if self.output_dir:
            self._save_results(results, model_name)
        
        return results
    
    def _aggregate_metrics(
        self,
        normalized_results: Dict[int, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across all horizons.
        
        Args:
            normalized_results: Dictionary mapping horizon -> structured metrics.
        
        Returns:
            Dictionary with aggregated metrics for classification and regression.
        """
        classification_metrics_list = []
        regression_metrics_list = []
        
        for horizon, metrics in normalized_results.items():
            if "classification" in metrics:
                classification_metrics_list.append(metrics["classification"])
            if "regression" in metrics:
                regression_metrics_list.append(metrics["regression"])
        
        aggregated = {}
        
        if classification_metrics_list:
            aggregated["classification"] = self._aggregate_classification_metrics(
                classification_metrics_list
            )
        
        if regression_metrics_list:
            aggregated["regression"] = self._aggregate_regression_metrics(
                regression_metrics_list
            )
        
        return aggregated
    
    def _aggregate_classification_metrics(
        self,
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate classification metrics across horizons.
        
        Args:
            metrics_list: List of classification metrics dictionaries.
        
        Returns:
            Dictionary with mean and std for each metric.
        """
        # Collect all metric values
        metric_values = {}
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    if key not in metric_values:
                        metric_values[key] = []
                    metric_values[key].append(value)
        
        # Calculate mean and std
        aggregated = {}
        for key, values in metric_values.items():
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
                aggregated[f"{key}_min"] = float(np.min(values))
                aggregated[f"{key}_max"] = float(np.max(values))
        
        return aggregated
    
    def _aggregate_regression_metrics(
        self,
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate regression metrics across horizons.
        
        Args:
            metrics_list: List of regression metrics dictionaries.
        
        Returns:
            Dictionary with mean and std for each metric.
        """
        # Collect all metric values
        metric_values = {}
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    if key not in metric_values:
                        metric_values[key] = []
                    metric_values[key].append(value)
        
        # Calculate mean and std
        aggregated = {}
        for key, values in metric_values.items():
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
                aggregated[f"{key}_min"] = float(np.min(values))
                aggregated[f"{key}_max"] = float(np.max(values))
        
        return aggregated
    
    def _find_best_horizon(
        self,
        normalized_results: Dict[int, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """Find horizon with best overall performance.
        
        Args:
            normalized_results: Dictionary mapping horizon -> structured metrics.
        
        Returns:
            Dictionary with best horizon and metrics.
        """
        # Score each horizon (lower is better for most metrics)
        horizon_scores = {}
        
        for horizon, metrics in normalized_results.items():
            score = 0.0
            
            # Classification metrics (lower is better: brier_score, ece)
            if "classification" in metrics:
                cls_metrics = metrics["classification"]
                # Primary metrics: Brier Score (most important for probability calibration)
                if "brier_score" in cls_metrics:
                    score += cls_metrics["brier_score"]
                # Secondary: ECE (calibration error)
                if "ece" in cls_metrics:
                    score += cls_metrics["ece"] * 0.5
                # Higher is better: ROC-AUC, PR-AUC (subtract from 1)
                if "roc_auc" in cls_metrics:
                    score += (1.0 - cls_metrics["roc_auc"]) * 0.3
            
            # Regression metrics (lower is better: MAE, RMSE)
            if "regression" in metrics:
                reg_metrics = metrics["regression"]
                if "mae" in reg_metrics:
                    score += reg_metrics["mae"] * 0.1  # Normalized by typical temp range
                if "rmse" in reg_metrics:
                    score += reg_metrics["rmse"] * 0.1
            
            horizon_scores[horizon] = score
        
        if not horizon_scores:
            return {"horizon": None, "reason": "No valid metrics found"}
        
        best_horizon = min(horizon_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "horizon": best_horizon,
            "horizon_hours": f"{best_horizon}h",
            "score": horizon_scores[best_horizon],
            "metrics": normalized_results[best_horizon]
        }
    
    def _save_results(
        self,
        results: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> None:
        """Save evaluation results to file.
        
        Args:
            results: Results dictionary to save.
            model_name: Optional model name for filename.
        """
        if not self.output_dir:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"multi_horizon_evaluation"
        if model_name:
            filename += f"_{model_name}"
        filename += ".json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        
        _logger.info(f"Saved multi-horizon evaluation results to {output_path}")
    
    @staticmethod
    def load_results(results_path: Path) -> Dict[str, Any]:
        """Load multi-horizon evaluation results from file.
        
        Args:
            results_path: Path to results JSON file.
        
        Returns:
            Results dictionary.
        """
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)

