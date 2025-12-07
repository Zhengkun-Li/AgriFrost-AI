"""Matrix evaluation module for 2×2+1 framework.

This module handles:
- Evaluation across all matrix cells (A, B, C, D, E)
- Automatic matrix summary generation
- Comparison across different tracks
- Matrix visualization data preparation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pandas as pd
import numpy as np

_logger = logging.getLogger(__name__)

from src.evaluation.multi_horizon_evaluator import MultiHorizonEvaluator
from src.evaluation.metrics import MetricsCalculator


class MatrixEvaluator:
    """Evaluate models across the 2×2+1 matrix framework.
    
    Matrix cells:
    - A: Raw features + Single-station
    - B: Feature engineering + Single-station
    - C: Raw features + Multi-station (spatial aggregation)
    - D: Feature engineering + Multi-station (spatial aggregation)
    - E: Raw features + Graph neural network
    
    This evaluator aggregates results across all matrix cells and provides
    comprehensive comparison and summary.
    """
    
    # Matrix cell definitions
    RAW_CELLS = {"A", "C", "E"}  # Raw features only
    FE_CELLS = {"B", "D"}  # Feature engineering required
    SINGLE_STATION_CELLS = {"A", "B"}  # Single-station models
    MULTI_STATION_CELLS = {"C", "D", "E"}  # Multi-station models
    GRAPH_CELLS = {"E"}  # Graph neural network models
    
    def __init__(
        self,
        matrix_cells: List[str] = ["A", "B", "C", "D", "E"],
        horizons: List[int] = [3, 6, 12, 24],
        output_dir: Optional[Path] = None
    ):
        """Initialize matrix evaluator.
        
        Args:
            matrix_cells: List of matrix cells to evaluate (e.g., ["A", "B", "C", "D", "E"]).
            horizons: List of forecast horizons in hours.
            output_dir: Optional output directory for saving results.
        """
        if not matrix_cells:
            raise ValueError("matrix_cells cannot be empty")
        
        # Validate matrix cells
        valid_cells = {"A", "B", "C", "D", "E"}
        for cell in matrix_cells:
            if cell.upper() not in valid_cells:
                raise ValueError(
                    f"Invalid matrix cell: {cell}. "
                    f"Valid cells: {valid_cells}"
                )
        
        self.matrix_cells = [c.upper() for c in matrix_cells]
        self.horizons = sorted(horizons)
        self.output_dir = Path(output_dir) if output_dir else None
        
        _logger.info(
            f"Initialized MatrixEvaluator for cells: {self.matrix_cells}, "
            f"horizons: {self.horizons}h"
        )
    
    def evaluate(
        self,
        results_dict: Dict[str, Dict[int, Dict[str, Any]]],
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate results across all matrix cells and horizons.
        
        Args:
            results_dict: Dictionary mapping matrix_cell -> horizon -> metrics.
                Expected format:
                {
                    "A": {
                        3: {"classification": {...}, "regression": {...}},
                        6: {...},
                        ...
                    },
                    "B": {...},
                    ...
                }
            model_type: Optional model type for logging.
        
        Returns:
            Dictionary with comprehensive matrix evaluation:
            {
                "cells": {
                    "A": {...},  # Per-cell multi-horizon results
                    "B": {...},
                    ...
                },
                "matrix_summary": {
                    "best_cell": {...},
                    "comparison": {...},  # Cell-by-cell comparison
                    "insights": [...]  # Textual insights
                },
                "horizon_analysis": {
                    "3h": {...},  # Per-horizon cell comparison
                    ...
                }
            }
        """
        if not results_dict:
            raise ValueError("results_dict cannot be empty")
        
        # Validate results structure
        for cell in self.matrix_cells:
            if cell not in results_dict:
                _logger.warning(f"Results for matrix cell {cell} not found. Skipping.")
                continue
        
        # Evaluate each cell
        cell_results = {}
        for cell in self.matrix_cells:
            if cell not in results_dict:
                continue
            
            cell_data = results_dict[cell]
            if not cell_data:
                _logger.warning(f"No results found for matrix cell {cell}. Skipping.")
                continue
            
            # Use MultiHorizonEvaluator for each cell
            evaluator = MultiHorizonEvaluator(
                horizons=self.horizons,
                output_dir=self.output_dir / cell if self.output_dir else None
            )
            
            cell_results[cell] = evaluator.evaluate(cell_data, model_name=f"{model_type}_{cell}" if model_type else cell)
        
        if not cell_results:
            raise ValueError("No valid results found for any matrix cell")
        
        # Generate matrix summary
        matrix_summary = self._generate_matrix_summary(cell_results)
        
        # Generate per-horizon analysis
        horizon_analysis = self._generate_horizon_analysis(cell_results)
        
        results = {
            "cells": cell_results,
            "matrix_summary": matrix_summary,
            "horizon_analysis": horizon_analysis,
            "model_type": model_type,
            "matrix_cells": self.matrix_cells,
            "horizons": self.horizons
        }
        
        # Save results if output_dir is provided
        if self.output_dir:
            self._save_results(results, model_type)
        
        return results
    
    def _generate_matrix_summary(
        self,
        cell_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary comparing all matrix cells.
        
        Args:
            cell_results: Dictionary mapping cell -> multi-horizon results.
        
        Returns:
            Dictionary with matrix summary and insights.
        """
        # Collect best horizon for each cell
        cell_best_metrics = {}
        for cell, results in cell_results.items():
            best_horizon_info = results.get("best_horizon", {})
            if best_horizon_info and "metrics" in best_horizon_info:
                cell_best_metrics[cell] = best_horizon_info["metrics"]
        
        if not cell_best_metrics:
            return {"error": "No valid metrics found for comparison"}
        
        # Compare cells
        comparison = self._compare_cells(cell_best_metrics)
        
        # Find best overall cell
        best_cell = self._find_best_cell(cell_best_metrics)
        
        # Generate insights
        insights = self._generate_insights(cell_best_metrics, best_cell)
        
        return {
            "best_cell": best_cell,
            "comparison": comparison,
            "insights": insights
        }
    
    def _compare_cells(
        self,
        cell_best_metrics: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """Compare metrics across all cells.
        
        Args:
            cell_best_metrics: Dictionary mapping cell -> best horizon metrics.
        
        Returns:
            Dictionary with cell-by-cell comparison.
        """
        comparison = {}
        
        # Classification metrics comparison
        if all("classification" in metrics for metrics in cell_best_metrics.values()):
            comparison["classification"] = {}
            cls_metrics = ["roc_auc", "pr_auc", "brier_score", "ece", "f1"]
            
            for metric in cls_metrics:
                cell_values = {}
                for cell, metrics in cell_best_metrics.items():
                    if metric in metrics.get("classification", {}):
                        cell_values[cell] = metrics["classification"][metric]
                
                if cell_values:
                    comparison["classification"][metric] = {
                        "values": cell_values,
                        "best": max(cell_values.items(), key=lambda x: x[1])[0] if metric in ["roc_auc", "pr_auc", "f1"] else min(cell_values.items(), key=lambda x: x[1])[0],
                        "worst": min(cell_values.items(), key=lambda x: x[1])[0] if metric in ["roc_auc", "pr_auc", "f1"] else max(cell_values.items(), key=lambda x: x[1])[0]
                    }
        
        # Regression metrics comparison
        if all("regression" in metrics for metrics in cell_best_metrics.values()):
            comparison["regression"] = {}
            reg_metrics = ["mae", "rmse", "r2"]
            
            for metric in reg_metrics:
                cell_values = {}
                for cell, metrics in cell_best_metrics.items():
                    if metric in metrics.get("regression", {}):
                        cell_values[cell] = metrics["regression"][metric]
                
                if cell_values:
                    comparison["regression"][metric] = {
                        "values": cell_values,
                        "best": min(cell_values.items(), key=lambda x: x[1])[0] if metric in ["mae", "rmse"] else max(cell_values.items(), key=lambda x: x[1])[0],
                        "worst": max(cell_values.items(), key=lambda x: x[1])[0] if metric in ["mae", "rmse"] else min(cell_values.items(), key=lambda x: x[1])[0]
                    }
        
        return comparison
    
    def _find_best_cell(
        self,
        cell_best_metrics: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """Find best overall matrix cell.
        
        Args:
            cell_best_metrics: Dictionary mapping cell -> best horizon metrics.
        
        Returns:
            Dictionary with best cell information.
        """
        cell_scores = {}
        
        for cell, metrics in cell_best_metrics.items():
            score = 0.0
            
            # Classification score (primary: Brier Score, secondary: ROC-AUC)
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
                    score += reg["mae"] * 0.05  # Normalized by typical temp range
            
            cell_scores[cell] = score
        
        if not cell_scores:
            return {"cell": None, "reason": "No valid metrics found"}
        
        best_cell = min(cell_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "cell": best_cell,
            "score": cell_scores[best_cell],
            "metrics": cell_best_metrics[best_cell]
        }
    
    def _generate_insights(
        self,
        cell_best_metrics: Dict[str, Dict[str, Dict[str, float]]],
        best_cell: Dict[str, Any]
    ) -> List[str]:
        """Generate textual insights from matrix comparison.
        
        Args:
            cell_best_metrics: Dictionary mapping cell -> best horizon metrics.
            best_cell: Best cell information.
        
        Returns:
            List of insight strings.
        """
        insights = []
        
        if best_cell.get("cell"):
            insights.append(f"Best overall performance: Cell {best_cell['cell']}")
        
        # Compare raw vs feature engineering
        raw_cells = [c for c in self.matrix_cells if c in self.RAW_CELLS]
        fe_cells = [c for c in self.matrix_cells if c in self.FE_CELLS]
        
        if raw_cells and fe_cells:
            raw_brier = []
            fe_brier = []
            
            for cell in raw_cells:
                if cell in cell_best_metrics and "classification" in cell_best_metrics[cell]:
                    if "brier_score" in cell_best_metrics[cell]["classification"]:
                        raw_brier.append(cell_best_metrics[cell]["classification"]["brier_score"])
            
            for cell in fe_cells:
                if cell in cell_best_metrics and "classification" in cell_best_metrics[cell]:
                    if "brier_score" in cell_best_metrics[cell]["classification"]:
                        fe_brier.append(cell_best_metrics[cell]["classification"]["brier_score"])
            
            if raw_brier and fe_brier:
                avg_raw = np.mean(raw_brier)
                avg_fe = np.mean(fe_brier)
                if avg_fe < avg_raw:
                    insights.append("Feature engineering improves Brier Score (lower is better)")
                else:
                    insights.append("Raw features perform better on average")
        
        # Compare single-station vs multi-station
        single_cells = [c for c in self.matrix_cells if c in self.SINGLE_STATION_CELLS]
        multi_cells = [c for c in self.matrix_cells if c in self.MULTI_STATION_CELLS]
        
        if single_cells and multi_cells:
            insights.append(
                f"Single-station cells: {single_cells}, "
                f"Multi-station cells: {multi_cells}"
            )
        
        return insights
    
    def _generate_horizon_analysis(
        self,
        cell_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate per-horizon analysis across all cells.
        
        Args:
            cell_results: Dictionary mapping cell -> multi-horizon results.
        
        Returns:
            Dictionary mapping horizon -> cell comparison.
        """
        horizon_analysis = {}
        
        for horizon in self.horizons:
            horizon_key = f"{horizon}h"
            cell_metrics = {}
            
            for cell, results in cell_results.items():
                if "horizons" in results and horizon_key in results["horizons"]:
                    cell_metrics[cell] = results["horizons"][horizon_key]
            
            if cell_metrics:
                # Find best cell for this horizon
                best_cell = self._find_best_cell_for_horizon(cell_metrics)
                
                horizon_analysis[horizon_key] = {
                    "cell_metrics": cell_metrics,
                    "best_cell": best_cell
                }
        
        return horizon_analysis
    
    def _find_best_cell_for_horizon(
        self,
        cell_metrics: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """Find best cell for a specific horizon.
        
        Args:
            cell_metrics: Dictionary mapping cell -> metrics for this horizon.
        
        Returns:
            Dictionary with best cell information.
        """
        cell_scores = {}
        
        for cell, metrics in cell_metrics.items():
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
            
            cell_scores[cell] = score
        
        if not cell_scores:
            return {"cell": None, "reason": "No valid metrics found"}
        
        best_cell = min(cell_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "cell": best_cell,
            "score": cell_scores[best_cell],
            "metrics": cell_metrics[best_cell]
        }
    
    def _save_results(
        self,
        results: Dict[str, Any],
        model_type: Optional[str] = None
    ) -> None:
        """Save matrix evaluation results to file.
        
        Args:
            results: Results dictionary to save.
            model_type: Optional model type for filename.
        """
        if not self.output_dir:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = "matrix_evaluation"
        if model_type:
            filename += f"_{model_type}"
        filename += ".json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        
        _logger.info(f"Saved matrix evaluation results to {output_path}")
        
        # Also save a summary markdown file
        self._save_summary_markdown(results, model_type)
    
    def _save_summary_markdown(
        self,
        results: Dict[str, Any],
        model_type: Optional[str] = None
    ) -> None:
        """Save matrix evaluation summary as markdown.
        
        Args:
            results: Results dictionary.
            model_type: Optional model type for filename.
        """
        if not self.output_dir:
            return
        
        summary_path = self.output_dir / "matrix_summary.md"
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Matrix Evaluation Summary\n\n")
            
            if model_type:
                f.write(f"**Model Type:** {model_type}\n\n")
            
            # Best cell
            best_cell = results.get("matrix_summary", {}).get("best_cell", {})
            if best_cell.get("cell"):
                f.write(f"## Best Overall Performance: Cell {best_cell['cell']}\n\n")
            
            # Insights
            insights = results.get("matrix_summary", {}).get("insights", [])
            if insights:
                f.write("## Key Insights\n\n")
                for insight in insights:
                    f.write(f"- {insight}\n")
                f.write("\n")
            
            # Per-horizon best cells
            horizon_analysis = results.get("horizon_analysis", {})
            if horizon_analysis:
                f.write("## Best Cell per Horizon\n\n")
                f.write("| Horizon | Best Cell |\n")
                f.write("|---------|-----------|\n")
                for horizon, analysis in sorted(horizon_analysis.items()):
                    best = analysis.get("best_cell", {}).get("cell", "N/A")
                    f.write(f"| {horizon} | {best} |\n")
                f.write("\n")
        
        _logger.info(f"Saved matrix summary markdown to {summary_path}")
    
    @staticmethod
    def load_results(results_path: Path) -> Dict[str, Any]:
        """Load matrix evaluation results from file.
        
        Args:
            results_path: Path to results JSON file.
        
        Returns:
            Results dictionary.
        """
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)

