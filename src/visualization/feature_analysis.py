"""Feature analysis utilities."""

import logging
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from src.visualization.plots import Plotter

_logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Unified feature analysis tool."""
    
    @staticmethod
    def run_full_analysis(
        data_path: Path,
        output_dir: Path,
        model_dir: Optional[Path] = None,
        config: Optional[Dict] = None
    ) -> Dict[str, Path]:
        """Run full feature analysis (high-level API).
        
        Internal calls:
        - analyze_all_features
        - analyze_feature_importance (if model_dir provided)
        - generate_feature_report
        
        Args:
            data_path: Path to data file
            model_dir: Optional path to trained model directory
            output_dir: Output directory for analysis results
            config: Optional analysis configuration
        
        Returns:
            {
                'statistics': Path,
                'importance': Path,  # if model_dir provided
                'report': Path
            }
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 1. Analyze all features
        _logger.info("Analyzing all features...")
        stats_df = FeatureAnalyzer._analyze_all_features(data_path, config)
        stats_path = output_dir / "feature_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        results['statistics'] = stats_path
        _logger.info(f"Feature statistics saved to {stats_path}")
        
        # 2. Analyze feature importance (if model provided)
        if model_dir:
            _logger.info("Analyzing feature importance...")
            importance_df = FeatureAnalyzer._analyze_feature_importance(
                model_dir, output_dir
            )
            results['importance'] = output_dir / "feature_importance.csv"
            _logger.info(f"Feature importance saved to {results['importance']}")
        
        # 3. Generate report
        _logger.info("Generating feature report...")
        report_path = FeatureAnalyzer._generate_feature_report(
            results, output_dir
        )
        results['report'] = report_path
        _logger.info(f"Feature report saved to {report_path}")
        
        return results
    
    @staticmethod
    def compare_feature_sets(
        feature_sets: List[Dict[str, Path]],
        output_dir: Path
    ) -> pd.DataFrame:
        """Compare multiple feature sets (high-level API).
        
        Args:
            feature_sets: [{'name': 'raw', 'path': Path}, ...]
            output_dir: Output directory
        
        Returns:
            Comparison result DataFrame
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and analyze each feature set
        comparisons = []
        for feature_set in feature_sets:
            name = feature_set['name']
            path = Path(feature_set['path'])
            
            # Load data
            if path.suffix == '.csv':
                df = pd.read_csv(path)
            elif path.suffix == '.parquet':
                df = pd.read_parquet(path)
            else:
                _logger.warning(f"Unsupported file format: {path.suffix}")
                continue
            
            # Analyze
            stats = FeatureAnalyzer._analyze_all_features(path, config=None)
            stats['feature_set'] = name
            comparisons.append(stats)
        
        # Combine and compare
        comparison_df = pd.concat(comparisons, ignore_index=True)
        
        # Save
        comparison_path = output_dir / "feature_set_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        return comparison_df
    
    # Private methods (internal use)
    
    @staticmethod
    def _analyze_all_features(data_path: Path, config: Optional[Dict]) -> pd.DataFrame:
        """Analyze all features statistics (migrated from scripts)."""
        # Load data
        data_path = Path(data_path)
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        # Compute statistics
        stats = []
        for col in df.columns:
            if col in ["Date", "Stn Id", "Hour"]:
                continue
            
            col_stats = {
                'feature': col,
                'dtype': str(df[col].dtype),
                'count': df[col].count(),
                'missing': df[col].isna().sum(),
                'missing_pct': df[col].isna().mean() * 100,
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_stats.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median(),
                })
            
            stats.append(col_stats)
        
        return pd.DataFrame(stats)
    
    @staticmethod
    def _analyze_feature_importance(model_dir: Path, output_dir: Path) -> pd.DataFrame:
        """Analyze model feature importance (migrated from scripts)."""
        model_dir = Path(model_dir)
        
        # Look for feature importance file
        importance_files = [
            model_dir / "feature_importances.csv",
            model_dir / "feature_importance.csv",
            model_dir / "full_training" / "horizon_3h" / "feature_importances.csv",
        ]
        
        importance_path = None
        for path in importance_files:
            if path.exists():
                importance_path = path
                break
        
        if importance_path is None:
            raise FileNotFoundError(f"Feature importance file not found in {model_dir}")
        
        # Load importance
        importance_df = pd.read_csv(importance_path)
        
        # Use Plotter to visualize
        plotter = Plotter(style="matplotlib")
        plot_path = output_dir / "feature_importance_plot.png"
        plotter.plot_feature_importance(
            importance_df,
            top_n=20,
            title="Feature Importance",
            save_path=plot_path,
            show=False
        )
        
        return importance_df
    
    @staticmethod
    def _generate_feature_report(results: Dict, output_dir: Path) -> Path:
        """Generate feature analysis report (migrated from scripts)."""
        output_dir = Path(output_dir)
        
        # Create simple HTML report
        report_path = output_dir / "feature_analysis_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature Analysis Report</title>
        </head>
        <body>
            <h1>Feature Analysis Report</h1>
            <h2>Statistics</h2>
            <p>Feature statistics saved to: {results.get('statistics', 'N/A')}</p>
            {f"<h2>Feature Importance</h2><p>Feature importance saved to: {results.get('importance', 'N/A')}</p>" if 'importance' in results else ""}
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path

