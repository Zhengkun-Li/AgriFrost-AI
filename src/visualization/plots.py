"""Plotting utilities for model predictions and analysis.

This module provides stateless plotting functions for:
- Model predictions visualization
- Training curve plotting (migrated from models/utils/curve_plotter)
- Reliability diagrams
- Feature analysis plots
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Any, Tuple
import pandas as pd
import numpy as np

_logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from src.evaluation.metrics import MetricsCalculator
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


class Plotter:
    """Create visualizations for model predictions and analysis."""
    
    def __init__(self, style: str = "matplotlib", figsize: tuple = (12, 6)):
        """Initialize plotter.
        
        Args:
            style: Plotting library to use ("matplotlib" or "plotly").
            figsize: Figure size for matplotlib (width, height).
        
        Raises:
            ImportError: If required plotting library is not available.
            ValueError: If style is not "matplotlib" or "plotly", or figsize is invalid.
        """
        # Input validation
        if style not in ["matplotlib", "plotly"]:
            raise ValueError(f"style must be 'matplotlib' or 'plotly', got {style}")
        
        if not isinstance(figsize, tuple) or len(figsize) != 2:
            raise ValueError(f"figsize must be a tuple of (width, height), got {figsize}")
        
        if figsize[0] <= 0 or figsize[1] <= 0:
            raise ValueError(f"figsize dimensions must be positive, got {figsize}")
        
        self.style = style
        self.figsize = figsize
        
        if style == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required. Install with: pip install matplotlib")
        if style == "plotly" and not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required. Install with: pip install plotly")
        
        _logger.debug(f"Initialized Plotter with style={style}, figsize={figsize}")
    
    def plot_predictions(self, 
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        dates: Optional[pd.Series] = None,
                        title: str = "Predictions vs Actual",
                        save_path: Optional[Path] = None,
                        show: bool = True) -> None:
        """Plot predictions against actual values.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            dates: Optional date index for time series plot.
            title: Plot title.
            save_path: Path to save figure.
            show: Whether to display the plot.
        
        Raises:
            ValueError: If inputs are empty or have incompatible shapes.
        """
        # Input validation
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError(f"Input arrays cannot be empty. y_true length: {len(y_true)}, y_pred length: {len(y_pred)}")
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true and y_pred must have the same length. Got {len(y_true)} and {len(y_pred)}")
        
        if dates is not None and len(dates) != len(y_true):
            raise ValueError(f"dates must have the same length as y_true. Got {len(dates)} and {len(y_true)}")
        
        _logger.debug(f"Plotting predictions for {len(y_true)} samples")
        
        if self.style == "matplotlib":
            self._plot_predictions_matplotlib(y_true, y_pred, dates, title, save_path, show)
        else:
            self._plot_predictions_plotly(y_true, y_pred, dates, title, save_path, show)
    
    def _plot_predictions_matplotlib(self, y_true, y_pred, dates, title, save_path, show):
        """Matplotlib implementation."""
        try:
            fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        except Exception as e:
            _logger.error(f"Failed to create matplotlib figure: {e}")
            raise
        
        # Time series plot
        if dates is not None:
            axes[0].plot(dates, y_true, label="Actual", alpha=0.7, linewidth=1)
            axes[0].plot(dates, y_pred, label="Predicted", alpha=0.7, linewidth=1)
            axes[0].set_xlabel("Date")
        else:
            x = np.arange(len(y_true))
            axes[0].plot(x, y_true, label="Actual", alpha=0.7, linewidth=1)
            axes[0].plot(x, y_pred, label="Predicted", alpha=0.7, linewidth=1)
            axes[0].set_xlabel("Sample Index")
        
        axes[0].set_ylabel("Temperature (°C)")
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        if dates is not None:
            # Use small markers for large datasets to avoid performance issues
            if len(residuals) > 10000:
                axes[1].plot(dates, residuals, '.', markersize=1, alpha=0.5)
            else:
                axes[1].scatter(dates, residuals, alpha=0.5, s=10)
            axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
            axes[1].set_xlabel("Date")
        else:
            x = np.arange(len(residuals))
            # Use small markers for large datasets to avoid performance issues
            if len(residuals) > 10000:
                axes[1].plot(x, residuals, '.', markersize=1, alpha=0.5)
            else:
                axes[1].scatter(x, residuals, alpha=0.5, s=10)
            axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
            axes[1].set_xlabel("Sample Index")
        
        axes[1].set_ylabel("Residuals (°C)")
        axes[1].set_title("Residuals")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            try:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                _logger.info(f"Saved prediction plot to {save_path}")
            except (IOError, OSError) as e:
                _logger.error(f"Failed to save plot to {save_path}: {e}")
                raise
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def _plot_predictions_plotly(self, y_true, y_pred, dates, title, save_path, show):
        """Plotly implementation."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(title, "Residuals"),
            vertical_spacing=0.1
        )
        
        x = dates if dates is not None else np.arange(len(y_true))
        
        # Time series plot
        fig.add_trace(
            go.Scatter(x=x, y=y_true, name="Actual", mode='lines', line=dict(width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y_pred, name="Predicted", mode='lines', line=dict(width=1)),
            row=1, col=1
        )
        
        # Residuals plot
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(x=x, y=residuals, name="Residuals", mode='markers', marker=dict(size=3)),
            row=2, col=1
        )
        # Add horizontal line with fallback for older plotly versions
        try:
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        except Exception:
            # Fallback for plotly < 4.12
            fig.add_trace(
                go.Scatter(
                    x=[x[0] if hasattr(x, '__iter__') and len(x) > 0 else 0, 
                       x[-1] if hasattr(x, '__iter__') and len(x) > 0 else 1],
                    y=[0, 0],
                    mode='lines',
                    line=dict(dash='dash', color='red', width=1),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Date" if dates is not None else "Sample Index", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Residuals (°C)", row=2, col=1)
        fig.update_layout(height=800, showlegend=True)
        
        if save_path:
            try:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(str(save_path))
                _logger.info(f"Saved prediction plot to {save_path}")
            except (IOError, OSError) as e:
                _logger.error(f"Failed to save plot to {save_path}: {e}")
                raise
        
        if show:
            fig.show()
    
    def plot_feature_importance(self,
                                importance: pd.DataFrame,
                                top_n: int = 20,
                                title: str = "Feature Importance",
                                save_path: Optional[Path] = None,
                                show: bool = True,
                                use_percentage: bool = True) -> None:
        """Plot feature importance.
        
        Args:
            importance: DataFrame with 'feature' and 'importance' columns.
                       If 'importance_pct' column exists, will use it when use_percentage=True.
            top_n: Number of top features to show (must be > 0).
            title: Plot title.
            save_path: Path to save figure.
            show: Whether to display the plot.
            use_percentage: Whether to use percentage (importance_pct) if available.
                          Default True, as percentage is more common and interpretable.
        
        Raises:
            ValueError: If DataFrame is empty, missing required columns, or top_n <= 0.
        """
        # Input validation
        if importance.empty:
            raise ValueError("importance DataFrame cannot be empty")
        
        required_cols = ['feature', 'importance']
        missing_cols = [col for col in required_cols if col not in importance.columns]
        if missing_cols:
            raise ValueError(f"importance DataFrame must have columns {required_cols}. Missing: {missing_cols}")
        
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        
        if top_n > len(importance):
            _logger.warning(f"top_n ({top_n}) is larger than number of features ({len(importance)}). Using all features.")
            top_n = len(importance)
        
        # Determine which column to use for plotting
        if use_percentage and 'importance_pct' in importance.columns:
            importance_col = 'importance_pct'
            xlabel = 'Importance (%)'
            if title == "Feature Importance":
                title = "Feature Importance (%)"
        else:
            importance_col = 'importance'
            xlabel = 'Importance'
        
        _logger.debug(f"Plotting top {top_n} feature importances using {importance_col}")
        
        if self.style == "matplotlib":
            self._plot_importance_matplotlib(importance, top_n, title, save_path, show, importance_col, xlabel)
        else:
            self._plot_importance_plotly(importance, top_n, title, save_path, show, importance_col, xlabel)
    
    def _plot_importance_matplotlib(self, importance, top_n, title, save_path, show, importance_col='importance', xlabel='Importance'):
        """Matplotlib implementation."""
        # Handle NaN values in importance column
        importance_clean = importance.dropna(subset=[importance_col])
        if importance_clean.empty:
            raise ValueError(f"All {importance_col} values are NaN")
        
        # Get top N features (sort by original importance column for consistency)
        sort_col = 'importance_pct' if importance_col == 'importance_pct' and 'importance_pct' in importance_clean.columns else 'importance'
        top_features = importance_clean.nlargest(top_n, sort_col)
        
        # Truncate long feature names to avoid plot distortion
        top_features = top_features.copy()
        if 'feature' in top_features.columns:
            top_features['feature'] = top_features['feature'].apply(
                lambda s: s[:40] + "..." if isinstance(s, str) and len(s) > 40 else s
            )
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features[importance_col].values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars if using percentage
        if importance_col == 'importance_pct':
            for i, (idx, row) in enumerate(top_features.iterrows()):
                value = row[importance_col]
                ax.text(value, i, f' {value:.2f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            try:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                _logger.info(f"Saved prediction plot to {save_path}")
            except (IOError, OSError) as e:
                _logger.error(f"Failed to save plot to {save_path}: {e}")
                raise
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def _plot_importance_plotly(self, importance, top_n, title, save_path, show, importance_col='importance', xlabel='Importance'):
        """Plotly implementation."""
        # Handle NaN values in importance column
        importance_clean = importance.dropna(subset=[importance_col])
        if importance_clean.empty:
            raise ValueError(f"All {importance_col} values are NaN")
        
        # Get top N features (sort by original importance column for consistency)
        sort_col = 'importance_pct' if importance_col == 'importance_pct' and 'importance_pct' in importance_clean.columns else 'importance'
        top_features = importance_clean.nlargest(top_n, sort_col)
        
        # Truncate long feature names to avoid plot distortion
        top_features = top_features.copy()
        if 'feature' in top_features.columns:
            top_features['feature'] = top_features['feature'].apply(
                lambda s: s[:40] + "..." if isinstance(s, str) and len(s) > 40 else s
            )
        
        fig = go.Figure()
        importance_values = top_features[importance_col].values
        fig.add_trace(go.Bar(
            x=importance_values,
            y=top_features['feature'].values,
            orientation='h',
            text=[f'{v:.2f}{"%" if importance_col == "importance_pct" else ""}' for v in importance_values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title='Feature',
            height=max(600, top_n * 30),
            yaxis={'autorange': 'reversed'}
        )
        
        if save_path:
            try:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(str(save_path))
                _logger.info(f"Saved prediction plot to {save_path}")
            except (IOError, OSError) as e:
                _logger.error(f"Failed to save plot to {save_path}: {e}")
                raise
        
        if show:
            fig.show()
    
    def plot_metrics_comparison(self,
                               metrics: Dict[str, Dict[str, float]],
                               title: str = "Model Comparison",
                               save_path: Optional[Path] = None,
                               show: bool = True) -> None:
        """Plot metrics comparison across models.
        
        Args:
            metrics: Dictionary with model names as keys and metric dicts as values.
            title: Plot title.
            save_path: Path to save figure.
            show: Whether to display the plot.
        
        Raises:
            ValueError: If metrics dictionary is empty.
        """
        # Input validation
        if not metrics:
            raise ValueError("metrics dictionary cannot be empty")
        
        _logger.debug(f"Plotting metrics comparison for {len(metrics)} models")
        
        if self.style == "matplotlib":
            self._plot_metrics_matplotlib(metrics, title, save_path, show)
        else:
            self._plot_metrics_plotly(metrics, title, save_path, show)
    
    def _plot_metrics_matplotlib(self, metrics, title, save_path, show):
        """Matplotlib implementation."""
        import math
        
        models = list(metrics.keys())
        metric_names = set()
        for model_metrics in metrics.values():
            metric_names.update(model_metrics.keys())
        metric_names = sorted(list(metric_names))
        
        n_metrics = len(metric_names)
        
        # Auto-layout: use multiple rows if too many metrics
        cols = min(4, n_metrics)
        rows = math.ceil(n_metrics / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.tolist()
        else:
            axes = axes.flatten().tolist()
        
        # Hide extra subplots if we have more axes than metrics
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        for idx, metric in enumerate(metric_names):
            values = [metrics[model].get(metric, 0) for model in models]
            axes[idx].bar(models, values)
            axes[idx].set_title(metric.upper())
            axes[idx].set_ylabel('Value')
            axes[idx].tick_params(axis='x', rotation=45, ha='right')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            try:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                _logger.info(f"Saved prediction plot to {save_path}")
            except (IOError, OSError) as e:
                _logger.error(f"Failed to save plot to {save_path}: {e}")
                raise
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def _plot_metrics_plotly(self, metrics, title, save_path, show):
        """Plotly implementation."""
        models = list(metrics.keys())
        metric_names = set()
        for model_metrics in metrics.values():
            metric_names.update(model_metrics.keys())
        metric_names = sorted(list(metric_names))
        
        fig = make_subplots(
            rows=1, cols=len(metric_names),
            subplot_titles=[m.upper() for m in metric_names]
        )
        
        for idx, metric in enumerate(metric_names):
            values = [metrics[model].get(metric, 0) for model in models]
            fig.add_trace(
                go.Bar(x=models, y=values, name=metric),
                row=1, col=idx + 1
            )
        
        fig.update_layout(title=title, height=500, showlegend=False)
        
        if save_path:
            try:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(str(save_path))
                _logger.info(f"Saved prediction plot to {save_path}")
            except (IOError, OSError) as e:
                _logger.error(f"Failed to save plot to {save_path}: {e}")
                raise
        
        if show:
            fig.show()
    
    def plot_reliability_diagram(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        title: str = "Reliability Diagram",
        save_path: Optional[Path] = None,
        show: bool = True
    ) -> None:
        """Plot reliability diagram for probability calibration.
        
        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities.
            n_bins: Number of bins for calibration (must be > 0).
            title: Plot title.
            save_path: Path to save figure.
            show: Whether to display the plot.
        
        Raises:
            ImportError: If MetricsCalculator is not available.
            ValueError: If inputs are empty, have incompatible shapes, or n_bins <= 0.
        """
        if not METRICS_AVAILABLE:
            raise ImportError("MetricsCalculator is required for reliability diagram")
        
        # Input validation
        y_true = np.asarray(y_true).flatten()
        y_proba = np.asarray(y_proba).flatten()
        
        if len(y_true) == 0 or len(y_proba) == 0:
            raise ValueError(f"Input arrays cannot be empty. y_true length: {len(y_true)}, y_proba length: {len(y_proba)}")
        
        if len(y_true) != len(y_proba):
            raise ValueError(f"y_true and y_proba must have the same length. Got {len(y_true)} and {len(y_proba)}")
        
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        
        # Check probability range
        if np.any(y_proba < 0) or np.any(y_proba > 1):
            _logger.warning("y_proba contains values outside [0, 1]. Clipping will be applied.")
            y_proba = np.clip(y_proba, 0, 1)
        
        _logger.debug(f"Plotting reliability diagram with {n_bins} bins for {len(y_true)} samples")
        
        if self.style == "matplotlib":
            self._plot_reliability_matplotlib(y_true, y_proba, n_bins, title, save_path, show)
        else:
            self._plot_reliability_plotly(y_true, y_proba, n_bins, title, save_path, show)
    
    def _plot_reliability_matplotlib(
        self, y_true, y_proba, n_bins, title, save_path, show
    ) -> None:
        """Matplotlib implementation of reliability diagram."""
        reliability_data = MetricsCalculator.calculate_reliability_data(
            y_true, y_proba, n_bins
        )
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # Plot reliability curve
        # Check both predicted_probs and actual_freqs for NaN
        valid_mask = ~(
            np.isnan(reliability_data["predicted_probs"]) |
            np.isnan(reliability_data["actual_freqs"])
        )
        ax.plot(
            reliability_data["predicted_probs"][valid_mask],
            reliability_data["actual_freqs"][valid_mask],
            'o-', label='Model', linewidth=2, markersize=8
        )
        
        # Add sample counts as text
        for i, (pred, actual, count) in enumerate(zip(
            reliability_data["predicted_probs"],
            reliability_data["actual_freqs"],
            reliability_data["counts"]
        )):
            if not np.isnan(pred) and count > 0:
                ax.annotate(
                    f'n={count}',
                    (pred, actual),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Observed Frequency', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Calculate and display ECE
        ece = MetricsCalculator.calculate_ece(y_true, y_proba, n_bins)
        ax.text(0.95, 0.05, f'ECE = {ece:.4f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            try:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                _logger.info(f"Saved prediction plot to {save_path}")
            except (IOError, OSError) as e:
                _logger.error(f"Failed to save plot to {save_path}: {e}")
                raise
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def _plot_reliability_plotly(
        self, y_true, y_proba, n_bins, title, save_path, show
    ) -> None:
        """Plotly implementation of reliability diagram."""
        reliability_data = MetricsCalculator.calculate_reliability_data(
            y_true, y_proba, n_bins
        )
        
        ece = MetricsCalculator.calculate_ece(y_true, y_proba, n_bins)
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='black', width=2)
        ))
        
        # Reliability curve
        # Check both predicted_probs and actual_freqs for NaN
        valid_mask = ~(
            np.isnan(reliability_data["predicted_probs"]) |
            np.isnan(reliability_data["actual_freqs"])
        )
        fig.add_trace(go.Scatter(
            x=reliability_data["predicted_probs"][valid_mask],
            y=reliability_data["actual_freqs"][valid_mask],
            mode='lines+markers',
            name='Model',
            line=dict(width=2),
            marker=dict(size=8),
            text=[f'n={c}' for c in reliability_data["counts"][valid_mask]],
            textposition='top center'
        ))
        
        fig.update_layout(
            title=f'{title} (ECE = {ece:.4f})',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Observed Frequency',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=800,
            height=800
        )
        
        if save_path:
            try:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(str(save_path))
                _logger.info(f"Saved prediction plot to {save_path}")
            except (IOError, OSError) as e:
                _logger.error(f"Failed to save plot to {save_path}: {e}")
                raise
        
        if show:
            fig.show()



# ============================================================================
# Training Curve Plotting (Stateless Functions)
# Migrated from models/utils/curve_plotter
# ============================================================================

def plot_training_curves(
    history: Union[Dict[str, List[Any]], Any],
    save_path: Path,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 150,
    backend: str = "matplotlib"
) -> bool:
    """Plot training curves from history (stateless function).
    
    **Stateless Design:**
    - No class instance needed
    - All parameters passed explicitly
    - Easy to use and test
    
    **TrainingHistory Integration:**
    - Accepts TrainingHistory instance directly (uses get_history())
    - Compatible with new TrainingHistory v2
    
    **Normalized Save Path:**
    - Saves to model_dir/curves/loss.png (standardized structure)
    
    Args:
        history: Training history dictionary or TrainingHistory instance.
        save_path: Path to save the plot (normalized to curves/ subdirectory).
        title: Plot title.
        figsize: Figure size (width, height).
        dpi: Resolution for saved figure.
        backend: Plotting backend ("matplotlib" or "plotly").
    
    Returns:
        True if plot was saved successfully, False otherwise.
    """
    # Handle TrainingHistory instance
    if hasattr(history, 'get_history'):
        history = history.get_history()
    
    # Normalize save path to curves/ subdirectory
    save_path = Path(save_path)
    if save_path.parent.name != 'curves':
        curves_dir = save_path.parent / "curves"
        save_path = curves_dir / save_path.name
    
    if backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
        return _plot_training_curves_matplotlib(history, save_path, title, figsize, dpi)
    elif backend == "plotly" and PLOTLY_AVAILABLE:
        return _plot_training_curves_plotly(history, save_path, title)
    else:
        _logger.warning(f"Plotting backend '{backend}' not available, skipping plot")
        return False


def _plot_training_curves_matplotlib(
    history: Dict[str, List[Any]],
    save_path: Path,
    title: str,
    figsize: Tuple[int, int],
    dpi: int
) -> bool:
    """Plot training curves using matplotlib (internal function)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Determine number of subplots based on available metrics
        metrics_to_plot = []
        if 'train_loss' in history or 'val_loss' in history:
            metrics_to_plot.append(('loss', ['train_loss', 'val_loss']))
        if 'learning_rate' in history:
            metrics_to_plot.append(('lr', ['learning_rate']))
        
        if len(metrics_to_plot) == 0:
            _logger.warning("No metrics to plot")
            return False
        
        n_plots = len(metrics_to_plot)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        epochs = history.get('epoch', list(range(1, len(history.get('train_loss', [])) + 1)))
        
        for idx, (plot_type, metric_names) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            if plot_type == 'loss':
                if 'train_loss' in history:
                    train_loss = [v for v in history['train_loss'] if v != float('inf')]
                    train_epochs = epochs[:len(train_loss)]
                    ax.plot(train_epochs, train_loss, label='Train Loss', 
                           marker='o', markersize=3, linewidth=1.5)
                
                if 'val_loss' in history:
                    val_loss = [(e, v) for e, v in zip(epochs, history['val_loss']) 
                               if v != float('inf')]
                    if val_loss:
                        val_epochs, val_losses = zip(*val_loss)
                        ax.plot(val_epochs, val_losses, label='Val Loss', 
                               marker='s', markersize=3, linewidth=1.5)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training and Validation Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            elif plot_type == 'lr':
                if 'learning_rate' in history:
                    lr = history['learning_rate']
                    ax.plot(epochs[:len(lr)], lr, label='Learning Rate', 
                           marker='o', markersize=3, linewidth=1.5, color='green')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Learning Rate')
                    ax.set_title('Learning Rate Schedule')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log')
        
        plt.suptitle(title, fontsize=14, y=0.995)
        plt.tight_layout()
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        _logger.warning(f"Failed to plot training curves: {e}", exc_info=True)
        return False


def _plot_training_curves_plotly(
    history: Dict[str, List[Any]],
    save_path: Path,
    title: str
) -> bool:
    """Plot training curves using plotly (internal function, placeholder)."""
    # Placeholder for plotly implementation
    return False


def plot_multitask_curves(
    history: Union[Dict[str, List[Any]], Any],
    save_path: Path,
    title: str = "Multi-task Training Curves",
    figsize: Tuple[int, int] = (10, 12),
    dpi: int = 150,
    backend: str = "matplotlib"
) -> bool:
    """Plot training curves for multi-task models (stateless function).
    
    Args:
        history: Training history dictionary with multi-task metrics.
        save_path: Path to save the plot.
        title: Plot title.
        figsize: Figure size.
        dpi: Resolution.
        backend: Plotting backend.
    
    Returns:
        True if successful.
    """
    # Handle TrainingHistory instance
    if hasattr(history, 'get_history'):
        history = history.get_history()
    
    if backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
        return _plot_multitask_curves_matplotlib(history, save_path, title, figsize, dpi)
    else:
        _logger.warning(f"Plotting backend '{backend}' not available, skipping plot")
        return False


def _plot_multitask_curves_matplotlib(
    history: Dict[str, List[Any]],
    save_path: Path,
    title: str,
    figsize: Tuple[int, int],
    dpi: int
) -> bool:
    """Plot multi-task curves using matplotlib (internal function)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        epochs = history.get('epoch', list(range(1, len(history.get('train_loss_total', [])) + 1)))
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Plot 1: Total loss
        if 'train_loss_total' in history and 'val_loss_total' in history:
            axes[0].plot(epochs, history['train_loss_total'], 
                       label='Train Loss (Total)', marker='o', markersize=3, linewidth=1.5)
            axes[0].plot(epochs, history['val_loss_total'], 
                       label='Val Loss (Total)', marker='s', markersize=3, linewidth=1.5)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Total Loss (Temperature + Frost)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Task-specific losses
        if 'train_loss_temp' in history:
            axes[1].plot(epochs, history['train_loss_temp'], 
                       label='Train Temp Loss', marker='o', markersize=3, linewidth=1.5, color='orange')
        if 'val_loss_temp' in history:
            axes[1].plot(epochs, history['val_loss_temp'], 
                       label='Val Temp Loss', marker='s', markersize=3, linewidth=1.5, color='orange', linestyle='--')
        if 'train_loss_frost' in history:
            axes[1].plot(epochs, history['train_loss_frost'], 
                       label='Train Frost Loss', marker='o', markersize=3, linewidth=1.5, color='blue')
        if 'val_loss_frost' in history:
            axes[1].plot(epochs, history['val_loss_frost'], 
                       label='Val Frost Loss', marker='s', markersize=3, linewidth=1.5, color='blue', linestyle='--')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Task-Specific Losses')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Learning rate
        if 'learning_rate' in history:
            axes[2].plot(epochs, history['learning_rate'], 
                       label='Learning Rate', marker='o', markersize=3, linewidth=1.5, color='green')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_title('Learning Rate Schedule')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].set_yscale('log')
        
        plt.suptitle(title, fontsize=14, y=0.995)
        plt.tight_layout()
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        _logger.warning(f"Failed to plot multi-task training curves: {e}", exc_info=True)
        return False
