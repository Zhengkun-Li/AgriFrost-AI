"""Spatial sensitivity evaluation module for C/D/E tracks.

This module handles:
- Evaluation across different spatial aggregation parameters (radius, k)
- Sensitivity analysis for spatial features
- Optimal parameter selection
- Visualization data preparation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import pandas as pd
import numpy as np

_logger = logging.getLogger(__name__)

from src.evaluation.metrics import MetricsCalculator
from src.evaluation.multi_horizon_evaluator import MultiHorizonEvaluator


class SpatialSensitivityEvaluator:
    """Evaluate model sensitivity to spatial aggregation parameters.
    
    This evaluator analyzes how model performance varies with different
    spatial aggregation parameters (radius in km for radius-based aggregation,
    or k for k-nearest-neighbors).
    
    Typical use cases:
    - C/D tracks: Evaluate radius sensitivity (e.g., 25km, 50km, 75km, 100km)
    - Graph models: Evaluate k sensitivity for k-NN graph construction
    """
    
    def __init__(
        self,
        param_name: str = "radius_km",
        param_values: Optional[List[Union[int, float]]] = None,
        horizons: List[int] = [3, 6, 12, 24],
        output_dir: Optional[Path] = None
    ):
        """Initialize spatial sensitivity evaluator.
        
        Args:
            param_name: Name of spatial parameter (e.g., "radius_km", "k_neighbors").
            param_values: List of parameter values to evaluate.
                If None, uses default values based on param_name:
                - "radius_km": [25, 50, 75, 100]
                - "k_neighbors": [1, 3, 5, 7, 10]
            horizons: List of forecast horizons in hours.
            output_dir: Optional output directory for saving results.
        """
        if not param_name:
            raise ValueError("param_name cannot be empty")
        
        if param_values is None:
            # Default values based on parameter type
            if "radius" in param_name.lower():
                param_values = [25, 50, 75, 100]
            elif "k" in param_name.lower() or "neighbor" in param_name.lower():
                param_values = [1, 3, 5, 7, 10]
            else:
                raise ValueError(
                    f"Cannot infer default param_values for param_name '{param_name}'. "
                    f"Please provide param_values explicitly."
                )
        
        if not param_values:
            raise ValueError("param_values cannot be empty")
        
        if not all(isinstance(v, (int, float)) and v > 0 for v in param_values):
            raise ValueError(
                f"All param_values must be positive numbers, got {param_values}"
            )
        
        self.param_name = param_name
        self.param_values = sorted(param_values)
        self.horizons = sorted(horizons)
        self.output_dir = Path(output_dir) if output_dir else None
        
        _logger.info(
            f"Initialized SpatialSensitivityEvaluator for {param_name}: {self.param_values}, "
            f"horizons: {self.horizons}h"
        )
    
    def evaluate(
        self,
        results_dict: Dict[Union[int, float], Dict[int, Dict[str, Any]]],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate results across all parameter values and horizons.
        
        Args:
            results_dict: Dictionary mapping param_value -> horizon -> metrics.
                Expected format:
                {
                    radius_km: {
                        3: {"classification": {...}, "regression": {...}},
                        6: {...},
                        ...
                    },
                    ...
                }
            model_name: Optional model name for logging.
        
        Returns:
            Dictionary with sensitivity analysis:
            {
                "parameters": {
                    "25": {...},  # Per-parameter multi-horizon results
                    "50": {...},
                    ...
                },
                "sensitivity_analysis": {
                    "optimal_parameter": {...},
                    "parameter_comparison": {...},
                    "insights": [...]
                },
                "horizon_analysis": {
                    "3h": {...},  # Per-horizon parameter comparison
                    ...
                }
            }
        """
        if not results_dict:
            raise ValueError("results_dict cannot be empty")
        
        # Validate results structure
        for param_val in self.param_values:
            if param_val not in results_dict:
                _logger.warning(
                    f"Results for {self.param_name}={param_val} not found. Skipping."
                )
                continue
        
        # Evaluate each parameter value
        param_results = {}
        for param_val in self.param_values:
            if param_val not in results_dict:
                continue
            
            param_data = results_dict[param_val]
            if not param_data:
                _logger.warning(
                    f"No results found for {self.param_name}={param_val}. Skipping."
                )
                continue
            
            # Use MultiHorizonEvaluator for each parameter value
            evaluator = MultiHorizonEvaluator(
                horizons=self.horizons,
                output_dir=self.output_dir / f"{self.param_name}_{param_val}" if self.output_dir else None
            )
            
            param_results[str(param_val)] = evaluator.evaluate(
                param_data,
                model_name=f"{model_name}_{self.param_name}_{param_val}" if model_name else f"{self.param_name}_{param_val}"
            )
        
        if not param_results:
            raise ValueError(
                f"No valid results found for any {self.param_name} value"
            )
        
        # Generate sensitivity analysis
        sensitivity_analysis = self._generate_sensitivity_analysis(param_results)
        
        # Generate per-horizon analysis
        horizon_analysis = self._generate_horizon_analysis(param_results)
        
        results = {
            "parameters": param_results,
            "sensitivity_analysis": sensitivity_analysis,
            "horizon_analysis": horizon_analysis,
            "param_name": self.param_name,
            "param_values": [str(v) for v in self.param_values],
            "horizons": self.horizons,
            "model_name": model_name
        }
        
        # Save results if output_dir is provided
        if self.output_dir:
            self._save_results(results, model_name)
        
        return results
    
    def _generate_sensitivity_analysis(
        self,
        param_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate sensitivity analysis across all parameter values.
        
        Args:
            param_results: Dictionary mapping param_value -> multi-horizon results.
        
        Returns:
            Dictionary with sensitivity analysis and optimal parameter.
        """
        # Collect best horizon for each parameter value
        param_best_metrics = {}
        for param_val, results in param_results.items():
            best_horizon_info = results.get("best_horizon", {})
            if best_horizon_info and "metrics" in best_horizon_info:
                param_best_metrics[param_val] = best_horizon_info["metrics"]
        
        if not param_best_metrics:
            return {"error": "No valid metrics found for sensitivity analysis"}
        
        # Compare parameter values
        parameter_comparison = self._compare_parameters(param_best_metrics)
        
        # Find optimal parameter
        optimal_parameter = self._find_optimal_parameter(param_best_metrics)
        
        # Generate insights
        insights = self._generate_insights(param_best_metrics, optimal_parameter)
        
        return {
            "optimal_parameter": optimal_parameter,
            "parameter_comparison": parameter_comparison,
            "insights": insights
        }
    
    def _compare_parameters(
        self,
        param_best_metrics: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """Compare metrics across all parameter values.
        
        Args:
            param_best_metrics: Dictionary mapping param_value -> best horizon metrics.
        
        Returns:
            Dictionary with parameter-by-parameter comparison.
        """
        comparison = {}
        
        # Classification metrics comparison
        if all("classification" in metrics for metrics in param_best_metrics.values()):
            comparison["classification"] = {}
            cls_metrics = ["roc_auc", "pr_auc", "brier_score", "ece"]
            
            for metric in cls_metrics:
                param_values_dict = {}
                for param_val, metrics in param_best_metrics.items():
                    if metric in metrics.get("classification", {}):
                        param_values_dict[param_val] = metrics["classification"][metric]
                
                if param_values_dict:
                    comparison["classification"][metric] = {
                        "values": param_values_dict,
                        "best": (
                            max(param_values_dict.items(), key=lambda x: x[1])[0]
                            if metric in ["roc_auc", "pr_auc"]
                            else min(param_values_dict.items(), key=lambda x: x[1])[0]
                        ),
                        "worst": (
                            min(param_values_dict.items(), key=lambda x: x[1])[0]
                            if metric in ["roc_auc", "pr_auc"]
                            else max(param_values_dict.items(), key=lambda x: x[1])[0]
                        ),
                        "trend": self._analyze_trend(param_values_dict)
                    }
        
        # Regression metrics comparison
        if all("regression" in metrics for metrics in param_best_metrics.values()):
            comparison["regression"] = {}
            reg_metrics = ["mae", "rmse", "r2"]
            
            for metric in reg_metrics:
                param_values_dict = {}
                for param_val, metrics in param_best_metrics.items():
                    if metric in metrics.get("regression", {}):
                        param_values_dict[param_val] = metrics["regression"][metric]
                
                if param_values_dict:
                    comparison["regression"][metric] = {
                        "values": param_values_dict,
                        "best": (
                            min(param_values_dict.items(), key=lambda x: x[1])[0]
                            if metric in ["mae", "rmse"]
                            else max(param_values_dict.items(), key=lambda x: x[1])[0]
                        ),
                        "worst": (
                            max(param_values_dict.items(), key=lambda x: x[1])[0]
                            if metric in ["mae", "rmse"]
                            else min(param_values_dict.items(), key=lambda x: x[1])[0]
                        ),
                        "trend": self._analyze_trend(param_values_dict)
                    }
        
        return comparison
    
    def _analyze_trend(
        self,
        param_values_dict: Dict[str, float]
    ) -> str:
        """Analyze trend in metric values across parameter values.
        
        Args:
            param_values_dict: Dictionary mapping param_value -> metric_value.
        
        Returns:
            Trend description: "increasing", "decreasing", "stable", or "mixed".
        """
        if len(param_values_dict) < 2:
            return "insufficient_data"
        
        # Sort by parameter value
        sorted_items = sorted(
            param_values_dict.items(),
            key=lambda x: float(x[0])
        )
        values = [v for _, v in sorted_items]
        
        # Check for monotonic trends
        is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
        is_decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
        
        if is_increasing:
            return "increasing"
        elif is_decreasing:
            return "decreasing"
        
        # Check stability (small variation)
        if len(values) > 1:
            std = np.std(values)
            mean = np.mean(values)
            cv = std / mean if mean != 0 else std
            if cv < 0.05:  # Coefficient of variation < 5%
                return "stable"
        
        return "mixed"
    
    def _find_optimal_parameter(
        self,
        param_best_metrics: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """Find optimal parameter value.
        
        Args:
            param_best_metrics: Dictionary mapping param_value -> best horizon metrics.
        
        Returns:
            Dictionary with optimal parameter information.
        """
        param_scores = {}
        
        for param_val, metrics in param_best_metrics.items():
            score = 0.0
            
            # Classification score (primary: Brier Score)
            if "classification" in metrics:
                cls = metrics["classification"]
                if "brier_score" in cls:
                    score += cls["brier_score"]  # Lower is better
                if "roc_auc" in cls:
                    score += (1.0 - cls["roc_auc"]) * 0.3  # Higher is better
            
            # Regression score (MAE normalized)
            if "regression" in metrics:
                reg = metrics["regression"]
                if "mae" in reg:
                    score += reg["mae"] * 0.05  # Normalized
        
            param_scores[param_val] = score
        
        if not param_scores:
            return {
                "parameter": None,
                "reason": "No valid metrics found"
            }
        
        optimal_param = min(param_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "parameter": optimal_param,
            "parameter_value": float(optimal_param) if optimal_param else None,
            "score": param_scores[optimal_param],
            "metrics": param_best_metrics[optimal_param]
        }
    
    def _generate_insights(
        self,
        param_best_metrics: Dict[str, Dict[str, Dict[str, float]]],
        optimal_parameter: Dict[str, Any]
    ) -> List[str]:
        """Generate textual insights from sensitivity analysis.
        
        Args:
            param_best_metrics: Dictionary mapping param_value -> best horizon metrics.
            optimal_parameter: Optimal parameter information.
        
        Returns:
            List of insight strings.
        """
        insights = []
        
        if optimal_parameter.get("parameter"):
            insights.append(
                f"Optimal {self.param_name}: {optimal_parameter['parameter']}"
            )
        
        # Analyze trends
        if "classification" in param_best_metrics.get(
            list(param_best_metrics.keys())[0] if param_best_metrics else None, {}
        ):
            # Check Brier Score trend
            brier_scores = {}
            for param_val, metrics in param_best_metrics.items():
                if "classification" in metrics and "brier_score" in metrics["classification"]:
                    brier_scores[param_val] = metrics["classification"]["brier_score"]
            
            if brier_scores:
                trend = self._analyze_trend(brier_scores)
                if trend == "decreasing":
                    insights.append(
                        f"Brier Score decreases (improves) with increasing {self.param_name}"
                    )
                elif trend == "increasing":
                    insights.append(
                        f"Brier Score increases (worsens) with increasing {self.param_name}"
                    )
                elif trend == "stable":
                    insights.append(
                        f"Brier Score is relatively stable across {self.param_name} values"
                    )
        
        return insights
    
    def _generate_horizon_analysis(
        self,
        param_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate per-horizon analysis across all parameter values.
        
        Args:
            param_results: Dictionary mapping param_value -> multi-horizon results.
        
        Returns:
            Dictionary mapping horizon -> parameter comparison.
        """
        horizon_analysis = {}
        
        for horizon in self.horizons:
            horizon_key = f"{horizon}h"
            param_metrics = {}
            
            for param_val, results in param_results.items():
                if "horizons" in results and horizon_key in results["horizons"]:
                    param_metrics[param_val] = results["horizons"][horizon_key]
            
            if param_metrics:
                # Find best parameter for this horizon
                best_param = self._find_best_parameter_for_horizon(param_metrics)
                
                horizon_analysis[horizon_key] = {
                    "parameter_metrics": param_metrics,
                    "best_parameter": best_param
                }
        
        return horizon_analysis
    
    def _find_best_parameter_for_horizon(
        self,
        param_metrics: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """Find best parameter for a specific horizon.
        
        Args:
            param_metrics: Dictionary mapping param_value -> metrics for this horizon.
        
        Returns:
            Dictionary with best parameter information.
        """
        param_scores = {}
        
        for param_val, metrics in param_metrics.items():
            score = 0.0
            
            if "classification" in metrics:
                cls = metrics["classification"]
                if "brier_score" in cls:
                    score += cls["brier_score"]
                if "roc_auc" in cls:
                    score += (1.0 - cls["roc_auc"]) * 0.3
            
            if "regression" in metrics:
                reg = metrics["regression"]
                if "mae" in reg:
                    score += reg["mae"] * 0.05
            
            param_scores[param_val] = score
        
        if not param_scores:
            return {
                "parameter": None,
                "reason": "No valid metrics found"
            }
        
        best_param = min(param_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "parameter": best_param,
            "parameter_value": float(best_param) if best_param else None,
            "score": param_scores[best_param],
            "metrics": param_metrics[best_param]
        }
    
    def _save_results(
        self,
        results: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> None:
        """Save sensitivity evaluation results to file.
        
        Args:
            results: Results dictionary to save.
            model_name: Optional model name for filename.
        """
        if not self.output_dir:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"spatial_sensitivity_{self.param_name}"
        if model_name:
            filename += f"_{model_name}"
        filename += ".json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        
        _logger.info(f"Saved spatial sensitivity results to {output_path}")
        
        # Also save a summary markdown file
        self._save_summary_markdown(results, model_name)
    
    def _save_summary_markdown(
        self,
        results: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> None:
        """Save sensitivity evaluation summary as markdown.
        
        Args:
            results: Results dictionary.
            model_name: Optional model name for filename.
        """
        if not self.output_dir:
            return
        
        summary_path = self.output_dir / f"spatial_sensitivity_{self.param_name}_summary.md"
        
        param_name = self.param_name
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"# Spatial Sensitivity Analysis: {param_name}\n\n")
            
            if model_name:
                f.write(f"**Model Type:** {model_name}\n\n")
            
            # Optimal parameter
            optimal = results.get("sensitivity_analysis", {}).get("optimal_parameter", {})
            if optimal.get("parameter"):
                f.write(f"## Optimal {param_name}: {optimal['parameter']}\n\n")
            
            # Insights
            insights = results.get("sensitivity_analysis", {}).get("insights", [])
            if insights:
                f.write("## Key Insights\n\n")
                for insight in insights:
                    f.write(f"- {insight}\n")
                f.write("\n")
            
            # Per-horizon best parameters
            horizon_analysis = results.get("horizon_analysis", {})
            if horizon_analysis:
                f.write(f"## Best {param_name} per Horizon\n\n")
                f.write(f"| Horizon | Best {param_name} |\n")
                f.write(f"|---------|------------------|\n")
                for horizon, analysis in sorted(horizon_analysis.items()):
                    best = analysis.get("best_parameter", {}).get("parameter", "N/A")
                    f.write(f"| {horizon} | {best} |\n")
                f.write("\n")
        
        _logger.info(f"Saved spatial sensitivity summary markdown to {summary_path}")
    
    @staticmethod
    def load_results(results_path: Path) -> Dict[str, Any]:
        """Load spatial sensitivity evaluation results from file.
        
        Args:
            results_path: Path to results JSON file.
        
        Returns:
            Results dictionary.
        """
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)

