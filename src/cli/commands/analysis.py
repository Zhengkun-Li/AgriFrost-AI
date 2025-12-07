"""Feature analysis commands."""

import click
import logging
import json
from pathlib import Path
from typing import List, Optional

from src.visualization.feature_analysis import FeatureAnalyzer
from src.cli.common import load_and_merge_config

_logger = logging.getLogger(__name__)


@click.group()
def analysis():
    """Feature analysis commands."""
    pass


@analysis.command()
@click.option('--data-path', type=click.Path(exists=True), required=True,
              help='Path to data file (CSV or Parquet)')
@click.option('--model-dir', type=click.Path(exists=True), default=None,
              help='Optional path to trained model directory (for feature importance)')
@click.option('--output-dir', type=click.Path(), required=True,
              help='Output directory for analysis results')
@click.option('--config', type=click.Path(exists=True), default=None,
              help='Optional analysis configuration file')
def full(data_path: str, model_dir: Optional[str], output_dir: str, config: Optional[str]):
    """Run full feature analysis (all-features + importance + report).
    
    Examples:
        python -m src.cli analysis full --data-path data/train.csv --output-dir analysis/features
        python -m src.cli analysis full --data-path data/train.csv --model-dir experiments/lightgbm_B_12h --output-dir analysis/features
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_dict = None
    if config:
        config_dict = load_and_merge_config(Path(config), {})
    
    try:
        results = FeatureAnalyzer.run_full_analysis(
            Path(data_path),
            output_dir,
            Path(model_dir) if model_dir else None,
            config_dict
        )
        
        click.echo(f"✅ Full analysis complete: {output_dir}")
        for key, path in results.items():
            click.echo(f"  {key}: {path}")
    except Exception as e:
        _logger.exception("Feature analysis failed")
        click.echo(f"❌ Analysis failed: {e}", err=True)
        raise click.Abort()


@analysis.command()
@click.option('--feature-sets', type=str, required=True,
              help='JSON string: [{"name": "raw", "path": "/path"}, ...]')
@click.option('--output-dir', type=click.Path(), required=True,
              help='Output directory for comparison results')
def compare_sets(feature_sets: str, output_dir: str):
    """Compare multiple feature sets.
    
    Examples:
        python -m src.cli analysis compare-sets --feature-sets '[{"name": "raw", "path": "data/raw.csv"}, {"name": "fe", "path": "data/features.csv"}]' --output-dir analysis/comparison
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse feature sets JSON
    try:
        sets_list = json.loads(feature_sets)
        sets_list = [{**d, 'path': Path(d['path'])} for d in sets_list]
    except json.JSONDecodeError as e:
        click.echo(f"❌ Invalid JSON format: {e}", err=True)
        raise click.Abort()
    
    try:
        comparison_df = FeatureAnalyzer.compare_feature_sets(
            sets_list,
            output_dir
        )
        
        click.echo(f"✅ Comparison complete: {output_dir}")
        click.echo(f"  Comparison saved to: {output_dir / 'feature_set_comparison.csv'}")
    except Exception as e:
        _logger.exception("Feature set comparison failed")
        click.echo(f"❌ Comparison failed: {e}", err=True)
        raise click.Abort()


@analysis.command(name="feature-importance")
@click.option('--model-dir', type=click.Path(exists=True), required=True,
              help='Path to trained model directory (e.g., experiments/lightgbm/raw/A/full_training/full_training/horizon_12h)')
@click.option('--task', type=click.Choice(['frost', 'temp', 'both']), default='both',
              help='Task to analyze (frost classification, temp regression, or both)')
@click.option('--output-dir', type=click.Path(), default=None,
              help='Output directory for feature importance files (default: model_dir/feature_importance)')
@click.option('--top-k', type=int, default=None,
              help='Show only top K most important features')
@click.option('--plot/--no-plot', default=True,
              help='Generate feature importance plots')
@click.option('--format', type=click.Choice(['csv', 'json', 'both']), default='both',
              help='Output format for feature importance data')
def feature_importance(
    model_dir: str,
    task: str,
    output_dir: Optional[str],
    top_k: Optional[int],
    plot: bool,
    format: str
):
    """Extract and analyze feature importance from trained models.
    
    Examples:
        # Analyze both frost and temp models
        python -m src.cli analysis feature-importance --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h
        
        # Analyze only frost classifier
        python -m src.cli analysis feature-importance --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h --task frost
        
        # Save to specific directory with plots
        python -m src.cli analysis feature-importance --model-dir experiments/lightgbm/raw/A/full_training/full_training/horizon_12h --output-dir results/feature_importance --plot --top-k 20
    """
    from src.models.registry import get_model_class
    from src.visualization.plots import Plotter
    import pandas as pd
    import numpy as np
    
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        click.echo(f"❌ Model directory not found: {model_dir}", err=True)
        raise click.Abort()
    
    # Try to extract feature names from training log or model config
    feature_names = None
    
    # Method 1: Try to get from training log
    training_log = model_dir / "training.log"
    if training_log.exists():
        try:
            with open(training_log, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            if "Feature list:" in log_content:
                for line in log_content.split('\n'):
                    if "Feature list:" in line:
                        feature_str = line.split("Feature list:")[1].strip()
                        feature_names = [f.strip() for f in feature_str.split(',')]
                        click.echo(f"✅ Extracted feature names from training log: {len(feature_names)} features")
                        break
        except Exception as e:
            _logger.debug(f"Could not extract feature names from training log: {e}")
    
    # Method 2: Try to get from model config (after fix)
    if feature_names is None:
        try:
            frost_model_path = model_dir / "frost_classifier"
            if frost_model_path.exists():
                config_path = frost_model_path / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    if 'feature_names' in config:
                        feature_names = config['feature_names']
                        click.echo(f"✅ Extracted feature names from model config: {len(feature_names)} features")
        except Exception as e:
            _logger.debug(f"Could not extract feature names from config: {e}")
    
    # Check if feature importance files already exist
    frost_importance_path = model_dir / "frost_feature_importance.csv"
    temp_importance_path = model_dir / "temp_feature_importance.csv"
    
    if not frost_importance_path.exists() and not temp_importance_path.exists():
        click.echo("⚠️  Feature importance files not found. Extracting from models...")
        click.echo("   (Feature importance is automatically saved during training)")
        
        # Try to load models and extract importance
        try:
            model_class = get_model_class("lightgbm")  # Try lightgbm first
            
            if task in ['frost', 'both']:
                frost_model_path = model_dir / "frost_classifier"
                if frost_model_path.exists():
                    frost_model = model_class.load(frost_model_path)
                    # Restore feature names if we found them
                    if feature_names is not None:
                        frost_model.feature_names = feature_names
                    
                    frost_importance = frost_model.get_feature_importance()
                    if frost_importance is not None:
                        total = frost_importance['importance'].sum()
                        if total > 0:
                            frost_importance = frost_importance.copy()
                            frost_importance['importance_pct'] = (frost_importance['importance'] / total * 100).round(2)
                            frost_importance['cumulative_pct'] = frost_importance['importance_pct'].cumsum()
                            frost_importance = frost_importance.sort_values('importance', ascending=False).reset_index(drop=True)
                        frost_importance.to_csv(frost_importance_path, index=False)
                        click.echo(f"✅ Extracted frost feature importance: {frost_importance_path}")
            
            if task in ['temp', 'both']:
                temp_model_path = model_dir / "temp_regressor"
                if temp_model_path.exists():
                    temp_model = model_class.load(temp_model_path)
                    # Restore feature names if we found them
                    if feature_names is not None:
                        temp_model.feature_names = feature_names
                    
                    temp_importance = temp_model.get_feature_importance()
                    if temp_importance is not None:
                        total = temp_importance['importance'].sum()
                        if total > 0:
                            temp_importance = temp_importance.copy()
                            temp_importance['importance_pct'] = (temp_importance['importance'] / total * 100).round(2)
                            temp_importance['cumulative_pct'] = temp_importance['importance_pct'].cumsum()
                            temp_importance = temp_importance.sort_values('importance', ascending=False).reset_index(drop=True)
                        temp_importance.to_csv(temp_importance_path, index=False)
                        click.echo(f"✅ Extracted temp feature importance: {temp_importance_path}")
        except Exception as e:
            _logger.exception(f"Error extracting feature importance: {e}")
            click.echo(f"⚠️  Could not extract feature importance from models: {e}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = model_dir / "feature_importance"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"\n📊 Analyzing feature importance")
    click.echo(f"📁 Model directory: {model_dir}")
    click.echo(f"📁 Output directory: {output_dir}")
    
    results = {}
    
    # Load and process frost importance
    if task in ['frost', 'both'] and frost_importance_path.exists():
        frost_importance = pd.read_csv(frost_importance_path)
        
        # Update feature names if we found them and current names are generic
        if feature_names is not None and frost_importance['feature'].str.startswith('feature_').any():
            click.echo("🔄 Updating feature names from generic to actual names...")
            # Create mapping from feature_0, feature_1, etc. to actual names
            feature_mapping = {f"feature_{i}": name for i, name in enumerate(feature_names)}
            frost_importance['feature'] = frost_importance['feature'].map(feature_mapping).fillna(frost_importance['feature'])
            # Save updated version
            frost_importance.to_csv(frost_importance_path, index=False)
            click.echo(f"✅ Updated feature names in {frost_importance_path}")
        
        if top_k is not None:
            frost_importance = frost_importance.head(top_k)
        results['frost'] = frost_importance
        
        # Save to output directory
        if format in ['csv', 'both']:
            output_csv = output_dir / "frost_feature_importance.csv"
            frost_importance.to_csv(output_csv, index=False)
            click.echo(f"✅ Saved CSV: {output_csv}")
        
        if format in ['json', 'both']:
            output_json = output_dir / "frost_feature_importance.json"
            frost_importance.to_json(output_json, orient='records', indent=2)
            click.echo(f"✅ Saved JSON: {output_json}")
        
        # Print summary
        click.echo(f"\n📊 Top 10 Most Important Features (Frost):")
        click.echo("=" * 80)
        for i, row in frost_importance.head(10).iterrows():
            click.echo(f"{i+1:2d}. {row['feature']:30s} | Importance: {row['importance']:10.4f} ({row['importance_pct']:5.2f}%) | Cumulative: {row['cumulative_pct']:5.2f}%")
        
        # Generate plots (both percentage and raw values)
        if plot:
            plotter = Plotter(style="matplotlib", figsize=(12, 8))
            
            # Percentage version
            plot_path_pct = output_dir / "frost_feature_importance_pct.png"
            plotter.plot_feature_importance(
                frost_importance.head(20),
                top_n=20,
                title=f"Feature Importance - Frost Classification (%)",
                save_path=plot_path_pct,
                show=False,
                use_percentage=True
            )
            click.echo(f"✅ Saved plot (percentage): {plot_path_pct}")
            
            # Raw values version
            plot_path_raw = output_dir / "frost_feature_importance_raw.png"
            plotter.plot_feature_importance(
                frost_importance.head(20),
                top_n=20,
                title=f"Feature Importance - Frost Classification (Raw Values)",
                save_path=plot_path_raw,
                show=False,
                use_percentage=False
            )
            click.echo(f"✅ Saved plot (raw values): {plot_path_raw}")
    
    # Load and process temp importance
    if task in ['temp', 'both'] and temp_importance_path.exists():
        temp_importance = pd.read_csv(temp_importance_path)
        
        # Update feature names if we found them and current names are generic
        if feature_names is not None and temp_importance['feature'].str.startswith('feature_').any():
            click.echo("🔄 Updating feature names from generic to actual names...")
            # Create mapping from feature_0, feature_1, etc. to actual names
            feature_mapping = {f"feature_{i}": name for i, name in enumerate(feature_names)}
            temp_importance['feature'] = temp_importance['feature'].map(feature_mapping).fillna(temp_importance['feature'])
            # Save updated version
            temp_importance.to_csv(temp_importance_path, index=False)
            click.echo(f"✅ Updated feature names in {temp_importance_path}")
        
        if top_k is not None:
            temp_importance = temp_importance.head(top_k)
        results['temp'] = temp_importance
        
        # Save to output directory
        if format in ['csv', 'both']:
            output_csv = output_dir / "temp_feature_importance.csv"
            temp_importance.to_csv(output_csv, index=False)
            click.echo(f"✅ Saved CSV: {output_csv}")
        
        if format in ['json', 'both']:
            output_json = output_dir / "temp_feature_importance.json"
            temp_importance.to_json(output_json, orient='records', indent=2)
            click.echo(f"✅ Saved JSON: {output_json}")
        
        # Print summary
        click.echo(f"\n📊 Top 10 Most Important Features (Temp):")
        click.echo("=" * 80)
        for i, row in temp_importance.head(10).iterrows():
            click.echo(f"{i+1:2d}. {row['feature']:30s} | Importance: {row['importance']:10.4f} ({row['importance_pct']:5.2f}%) | Cumulative: {row['cumulative_pct']:5.2f}%")
        
        # Generate plots (both percentage and raw values)
        if plot:
            plotter = Plotter(style="matplotlib", figsize=(12, 8))
            
            # Percentage version
            plot_path_pct = output_dir / "temp_feature_importance_pct.png"
            plotter.plot_feature_importance(
                temp_importance.head(20),
                top_n=20,
                title=f"Feature Importance - Temperature Regression (%)",
                save_path=plot_path_pct,
                show=False,
                use_percentage=True
            )
            click.echo(f"✅ Saved plot (percentage): {plot_path_pct}")
            
            # Raw values version
            plot_path_raw = output_dir / "temp_feature_importance_raw.png"
            plotter.plot_feature_importance(
                temp_importance.head(20),
                top_n=20,
                title=f"Feature Importance - Temperature Regression (Raw Values)",
                save_path=plot_path_raw,
                show=False,
                use_percentage=False
            )
            click.echo(f"✅ Saved plot (raw values): {plot_path_raw}")
    
    # Compare if both available
    if len(results) == 2 and 'frost' in results and 'temp' in results:
        click.echo(f"\n📊 Comparing Frost vs Temp Feature Importance...")
        
        frost_df = results['frost']
        temp_df = results['temp']
        
        # Merge for comparison
        comparison_df = frost_df[['feature', 'importance_pct']].merge(
            temp_df[['feature', 'importance_pct']],
            on='feature',
            suffixes=('_frost', '_temp'),
            how='outer'
        ).fillna(0)
        
        # Calculate correlation
        correlation = comparison_df['importance_pct_frost'].corr(comparison_df['importance_pct_temp'])
        click.echo(f"  Correlation between frost and temp importance: {correlation:.4f}")
        
        # Save comparison
        if format in ['csv', 'both']:
            comparison_path = output_dir / "frost_temp_importance_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            click.echo(f"✅ Saved comparison CSV: {comparison_path}")
        
        # Plot comparison (both percentage and raw values)
        if plot:
            import matplotlib.pyplot as plt
            
            # Percentage version
            fig_pct, (ax1_pct, ax2_pct) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot frost importance (percentage)
            frost_top = frost_df.head(15)
            y_pos_frost = np.arange(len(frost_top))
            ax1_pct.barh(y_pos_frost, frost_top['importance_pct'].values, align='center')
            ax1_pct.set_yticks(y_pos_frost)
            ax1_pct.set_yticklabels(frost_top['feature'].values)
            ax1_pct.invert_yaxis()
            ax1_pct.set_xlabel('Importance (%)')
            ax1_pct.set_title('Frost Classification')
            ax1_pct.grid(True, alpha=0.3, axis='x')
            
            # Plot temp importance (percentage)
            temp_top = temp_df.head(15)
            y_pos_temp = np.arange(len(temp_top))
            ax2_pct.barh(y_pos_temp, temp_top['importance_pct'].values, align='center')
            ax2_pct.set_yticks(y_pos_temp)
            ax2_pct.set_yticklabels(temp_top['feature'].values)
            ax2_pct.invert_yaxis()
            ax2_pct.set_xlabel('Importance (%)')
            ax2_pct.set_title('Temperature Regression')
            ax2_pct.grid(True, alpha=0.3, axis='x')
            
            plt.suptitle('Feature Importance Comparison: Frost vs Temp (%)', fontsize=14)
            plt.tight_layout()
            
            comparison_plot_path_pct = output_dir / "frost_temp_importance_comparison_pct.png"
            plt.savefig(comparison_plot_path_pct, dpi=300, bbox_inches='tight')
            plt.close(fig_pct)
            click.echo(f"✅ Saved comparison plot (percentage): {comparison_plot_path_pct}")
            
            # Raw values version
            fig_raw, (ax1_raw, ax2_raw) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot frost importance (raw values)
            ax1_raw.barh(y_pos_frost, frost_top['importance'].values, align='center')
            ax1_raw.set_yticks(y_pos_frost)
            ax1_raw.set_yticklabels(frost_top['feature'].values)
            ax1_raw.invert_yaxis()
            ax1_raw.set_xlabel('Importance (Raw Values)')
            ax1_raw.set_title('Frost Classification')
            ax1_raw.grid(True, alpha=0.3, axis='x')
            
            # Plot temp importance (raw values)
            ax2_raw.barh(y_pos_temp, temp_top['importance'].values, align='center')
            ax2_raw.set_yticks(y_pos_temp)
            ax2_raw.set_yticklabels(temp_top['feature'].values)
            ax2_raw.invert_yaxis()
            ax2_raw.set_xlabel('Importance (Raw Values)')
            ax2_raw.set_title('Temperature Regression')
            ax2_raw.grid(True, alpha=0.3, axis='x')
            
            plt.suptitle('Feature Importance Comparison: Frost vs Temp (Raw Values)', fontsize=14)
            plt.tight_layout()
            
            comparison_plot_path_raw = output_dir / "frost_temp_importance_comparison_raw.png"
            plt.savefig(comparison_plot_path_raw, dpi=300, bbox_inches='tight')
            plt.close(fig_raw)
            click.echo(f"✅ Saved comparison plot (raw values): {comparison_plot_path_raw}")
    
    click.echo(f"\n✅ Feature importance analysis complete!")
    click.echo(f"📁 Results saved to: {output_dir}")
