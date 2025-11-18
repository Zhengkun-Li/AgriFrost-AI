"""Plotting utilities for model predictions and analysis."""

from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
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
        """
        self.style = style
        self.figsize = figsize
        
        if style == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required. Install with: pip install matplotlib seaborn")
        if style == "plotly" and not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required. Install with: pip install plotly")
    
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
        """
        if self.style == "matplotlib":
            self._plot_predictions_matplotlib(y_true, y_pred, dates, title, save_path, show)
        else:
            self._plot_predictions_plotly(y_true, y_pred, dates, title, save_path, show)
    
    def _plot_predictions_matplotlib(self, y_true, y_pred, dates, title, save_path, show):
        """Matplotlib implementation."""
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
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
            axes[1].scatter(dates, residuals, alpha=0.5, s=10)
            axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
            axes[1].set_xlabel("Date")
        else:
            x = np.arange(len(residuals))
            axes[1].scatter(x, residuals, alpha=0.5, s=10)
            axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
            axes[1].set_xlabel("Sample Index")
        
        axes[1].set_ylabel("Residuals (°C)")
        axes[1].set_title("Residuals")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
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
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_xaxes(title_text="Date" if dates is not None else "Sample Index", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Residuals (°C)", row=2, col=1)
        fig.update_layout(height=800, showlegend=True)
        
        if save_path:
            fig.write_html(str(save_path))
        
        if show:
            fig.show()
    
    def plot_feature_importance(self,
                                importance: pd.DataFrame,
                                top_n: int = 20,
                                title: str = "Feature Importance",
                                save_path: Optional[Path] = None,
                                show: bool = True) -> None:
        """Plot feature importance.
        
        Args:
            importance: DataFrame with 'feature' and 'importance' columns.
            top_n: Number of top features to show.
            title: Plot title.
            save_path: Path to save figure.
            show: Whether to display the plot.
        """
        if self.style == "matplotlib":
            self._plot_importance_matplotlib(importance, top_n, title, save_path, show)
        else:
            self._plot_importance_plotly(importance, top_n, title, save_path, show)
    
    def _plot_importance_matplotlib(self, importance, top_n, title, save_path, show):
        """Matplotlib implementation."""
        # Get top N features
        top_features = importance.nlargest(top_n, 'importance')
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'].values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _plot_importance_plotly(self, importance, top_n, title, save_path, show):
        """Plotly implementation."""
        top_features = importance.nlargest(top_n, 'importance')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_features['importance'].values,
            y=top_features['feature'].values,
            orientation='h'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=max(600, top_n * 30),
            yaxis={'autorange': 'reversed'}
        )
        
        if save_path:
            fig.write_html(str(save_path))
        
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
        """
        if self.style == "matplotlib":
            self._plot_metrics_matplotlib(metrics, title, save_path, show)
        else:
            self._plot_metrics_plotly(metrics, title, save_path, show)
    
    def _plot_metrics_matplotlib(self, metrics, title, save_path, show):
        """Matplotlib implementation."""
        models = list(metrics.keys())
        metric_names = set()
        for model_metrics in metrics.values():
            metric_names.update(model_metrics.keys())
        metric_names = sorted(list(metric_names))
        
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metric_names):
            values = [metrics[model].get(metric, 0) for model in models]
            axes[idx].bar(models, values)
            axes[idx].set_title(metric.upper())
            axes[idx].set_ylabel('Value')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
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
            fig.write_html(str(save_path))
        
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
            n_bins: Number of bins for calibration.
            title: Plot title.
            save_path: Path to save figure.
            show: Whether to display the plot.
        """
        if not METRICS_AVAILABLE:
            raise ImportError("MetricsCalculator is required for reliability diagram")
        
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
        valid_mask = ~np.isnan(reliability_data["predicted_probs"])
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
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
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
        valid_mask = ~np.isnan(reliability_data["predicted_probs"])
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
            fig.write_html(str(save_path))
        
        if show:
            fig.show()

