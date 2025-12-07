"""Evaluation commands."""

import click
import logging
from pathlib import Path
from typing import List, Optional

from src.training.pipeline_runner import EvaluationRunner, load_training_config
from src.evaluation.matrix_summary import (
    load_experiment_results,
    build_matrix_summary,
    build_spatial_sensitivity
)
from src.visualization.matrix_plots import (
    plot_matrix_summary,
    plot_spatial_sensitivity
)

_logger = logging.getLogger(__name__)


@click.group()
def evaluate():
    """Evaluation commands."""
    pass


@evaluate.command()
@click.option('--model-dir', type=click.Path(exists=True), required=True,
              help='Path to trained model directory')
@click.option('--output-dir', type=click.Path(), default=None,
              help='Output directory for evaluation results')
@click.option('--config', type=click.Path(exists=True), default=None,
              help='Path to evaluation config YAML')
@click.option('--project-root', type=click.Path(exists=True), default=None,
              help='Project root directory (default: auto-detect)')
def model(model_dir: str, output_dir: Optional[str], config: Optional[str], project_root: Optional[str]):
    """Evaluate a single model.
    
    Examples:
        python -m src.cli evaluate model --model-dir experiments/lightgbm_B_12h
        python -m src.cli evaluate model --model-dir experiments/lightgbm_B_12h --config config/evaluation.yaml
    """
    # Determine project root
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent
    else:
        project_root = Path(project_root)
    
    model_dir = Path(model_dir)
    
    # Check if run_metadata.json exists (required)
    metadata_path = model_dir / "run_metadata.json"
    if not metadata_path.exists():
        click.echo(f"❌ Error: {model_dir} missing run_metadata.json (invalid run)", err=True)
        raise click.Abort()
    
    # Load metadata to get experiment info
    from src.utils.metadata import ExperimentMetadata
    metadata = ExperimentMetadata.load(metadata_path)
    
    click.echo(f"Evaluating model: {metadata.model_name}")
    click.echo(f"  Matrix cell: {metadata.matrix_cell}")
    click.echo(f"  Track: {metadata.track}")
    click.echo(f"  Horizon: {metadata.horizon_h}h")
    
    # Build config for evaluation
    # If config file provided, use it; otherwise create minimal config
    if config:
        pipeline_config = load_training_config(Path(config), project_root, cli_overrides={})
    else:
        # Create minimal config from model directory structure
        # This is a fallback - ideally users should provide config
        click.echo("⚠️  Warning: No config provided. Using minimal config.", err=True)
        # For now, we'll need to construct a minimal config
        # This is complex, so we'll require config for now
        click.echo("❌ Error: --config is required for evaluation", err=True)
        raise click.Abort()
    
    # Set output directory
    if output_dir:
        if not pipeline_config.evaluation.tasks:
            # Create evaluation task if none exists
            from src.training.pipeline_runner import EvaluationTask
            pipeline_config.evaluation.tasks = [EvaluationTask(type="direct", params={})]
        pipeline_config.evaluation.tasks[0].params["output_dir"] = output_dir
    
    # Run evaluation
    runner = EvaluationRunner(pipeline_config, project_root)
    
    try:
        exit_code = runner.run()
        if exit_code == 0:
            click.echo(f"✅ Evaluation complete: {output_dir or model_dir}")
        else:
            click.echo(f"❌ Evaluation failed with exit code {exit_code}", err=True)
            raise click.Abort()
    except Exception as e:
        _logger.exception("Evaluation failed")
        click.echo(f"❌ Evaluation failed: {e}", err=True)
        raise click.Abort()


@evaluate.command()
@click.option('--model-dirs', type=click.Path(exists=True), multiple=True, required=True,
              help='Paths to trained model directories (can specify multiple)')
@click.option('--output-dir', type=click.Path(), required=True,
              help='Output directory for comparison results')
@click.option('--include-legacy', is_flag=True, default=False,
              help='Include legacy runs (matrix_cell=None) in comparison')
def compare(model_dirs: tuple, output_dir: str, include_legacy: bool):
    """Compare multiple models.
    
    ⚠️ Hard constraint: Only works through run_metadata.json + metrics.json, no path parsing.
    
    Examples:
        python -m src.cli evaluate compare --model-dirs experiments/lightgbm_B_12h experiments/xgboost_B_12h --output-dir comparison/
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir_paths = [Path(d) for d in model_dirs]
    
    click.echo(f"Comparing {len(model_dir_paths)} models...")
    
    # Load experiment results (enforced metadata-only)
    results_df = load_experiment_results(
        model_dir_paths,
        include_legacy=include_legacy
    )
    
    if results_df.empty:
        click.echo("❌ No valid experiment results found", err=True)
        raise click.Abort()
    
    # Build summary
    summary_df = build_matrix_summary(results_df)
    
    # Save summary
    summary_path = output_dir / "comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')
    click.echo(f"✅ Comparison summary saved to {summary_path}")
    
    # Plot comparison
    plot_path = output_dir / "comparison_plot.png"
    plot_matrix_summary(summary_df, plot_path)
    click.echo(f"✅ Comparison plot saved to {plot_path}")
    
    click.echo(f"✅ Comparison complete: {output_dir}")


@evaluate.command()
@click.option('--experiments-dir', type=click.Path(exists=True), required=True,
              help='Directory containing experiment results (will scan recursively)')
@click.option('--output-dir', type=click.Path(), required=True,
              help='Output directory for matrix summary')
@click.option('--filter-matrix-cell', type=click.Choice(['A', 'B', 'C', 'D', 'E']),
              help='Filter by matrix cell')
@click.option('--filter-track', type=str,
              help='Filter by track (e.g., raw, top175_features)')
@click.option('--filter-horizon', type=int,
              help='Filter by horizon (e.g., 3, 6, 12, 24)')
@click.option('--include-legacy', is_flag=True, default=False,
              help='Include legacy runs in summary')
def matrix(
    experiments_dir: str,
    output_dir: str,
    filter_matrix_cell: Optional[str],
    filter_track: Optional[str],
    filter_horizon: Optional[int],
    include_legacy: bool
):
    """Generate 2×2+1 matrix summary.
    
    ⚠️ Hard constraint: Only works through run_metadata.json + metrics.json, no path parsing.
    
    Automatically scans experiments_dir for all run_metadata.json files and builds matrix summary.
    
    Examples:
        python -m src.cli evaluate matrix --experiments-dir experiments/ --output-dir matrix_summary/
    """
    experiments_dir = Path(experiments_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan for all run_metadata.json files
    click.echo(f"Scanning {experiments_dir} for experiments...")
    metadata_files = list(experiments_dir.rglob("run_metadata.json"))
    
    if not metadata_files:
        click.echo(f"❌ No run_metadata.json files found in {experiments_dir}", err=True)
        raise click.Abort()
    
    click.echo(f"Found {len(metadata_files)} experiments")
    
    # Extract model directories (parent of run_metadata.json)
    model_dirs = [f.parent for f in metadata_files]
    
    # Build filter config
    filter_config = {}
    if filter_matrix_cell:
        filter_config["matrix_cell"] = filter_matrix_cell
    if filter_track:
        filter_config["track"] = filter_track
    if filter_horizon:
        filter_config["horizon_h"] = filter_horizon
    
    # Load results
    results_df = load_experiment_results(
        model_dirs,
        filter_config=filter_config if filter_config else None,
        include_legacy=include_legacy
    )
    
    if results_df.empty:
        click.echo("❌ No valid experiment results found after filtering", err=True)
        raise click.Abort()
    
    click.echo(f"Loaded {len(results_df)} experiment results")
    
    # Build matrix summary
    summary_df = build_matrix_summary(results_df)
    
    # Save summary
    summary_path = output_dir / "matrix_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')
    click.echo(f"✅ Matrix summary saved to {summary_path}")
    
    # Plot matrix summary
    plot_path = output_dir / "matrix_summary.png"
    plot_matrix_summary(summary_df, plot_path)
    click.echo(f"✅ Matrix summary plot saved to {plot_path}")
    
    # Build spatial sensitivity if applicable
    if "radius_km" in results_df.columns and results_df["radius_km"].notna().any():
        click.echo("Building spatial sensitivity (radius_km)...")
        sensitivity_df = build_spatial_sensitivity(results_df, spatial_param="radius_km")
        sensitivity_path = output_dir / "spatial_sensitivity_radius.csv"
        sensitivity_df.to_csv(sensitivity_path, index=False, encoding='utf-8')
        click.echo(f"✅ Spatial sensitivity (radius) saved to {sensitivity_path}")
        
        plot_sens_path = output_dir / "spatial_sensitivity_radius.png"
        plot_spatial_sensitivity(sensitivity_df, plot_sens_path, param_name="radius_km")
        click.echo(f"✅ Spatial sensitivity plot saved to {plot_sens_path}")
    
    if "knn_k" in results_df.columns and results_df["knn_k"].notna().any():
        click.echo("Building spatial sensitivity (knn_k)...")
        sensitivity_df = build_spatial_sensitivity(results_df, spatial_param="knn_k")
        sensitivity_path = output_dir / "spatial_sensitivity_knn.csv"
        sensitivity_df.to_csv(sensitivity_path, index=False, encoding='utf-8')
        click.echo(f"✅ Spatial sensitivity (knn) saved to {sensitivity_path}")
        
        plot_sens_path = output_dir / "spatial_sensitivity_knn.png"
        plot_spatial_sensitivity(sensitivity_df, plot_sens_path, param_name="knn_k")
        click.echo(f"✅ Spatial sensitivity plot saved to {plot_sens_path}")
    
    click.echo(f"✅ Matrix summary complete: {output_dir}")
