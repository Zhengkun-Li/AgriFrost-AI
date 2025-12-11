"""Training commands."""

import click
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from src.training.pipeline_runner import TrainingRunner, load_training_config
from src.cli.common import load_and_merge_config

_logger = logging.getLogger(__name__)


@click.group()
def train():
    """Training commands."""
    pass


@train.command()
@click.option('--model-name', '--model', type=str, required=True,
              help='Model name (e.g., lightgbm, xgboost, lstm)')
@click.option('--matrix-cell', type=click.Choice(['A', 'B', 'C', 'D', 'E']), required=True,
              help='Matrix cell (A/B/C/D/E)')
@click.option('--track', type=str, required=True,
              help='Feature track (e.g., raw, top175_features)')
@click.option('--horizon-h', '--horizon', type=int, required=True,
              help='Forecast horizon in hours (e.g., 3, 6, 12, 24)')
@click.option('--radius-km', type=float, default=None,
              help='Spatial radius in km (for C/D tracks)')
@click.option('--knn-k', type=int, default=None,
              help='KNN k parameter (for E track)')
@click.option('--config', type=click.Path(exists=True), default=None,
              help='Path to YAML config file')
@click.option('--output-dir', type=click.Path(), default=None,
              help='Output directory (default: experiments/{model}/{track}/{matrix_cell}/full_training)')
@click.option('--feature-selection-name', type=str, default=None,
              help='Suffix for output dir when using feature selection (e.g., "top15", "selected_90pct"). '
                   'Will be appended to output dir: experiments/{model}/{track}/{matrix_cell}/full_training_{suffix}')
@click.option('--data-path', type=click.Path(exists=True), default=None,
              help='Path to input data file')
@click.option('--sample-size', type=int, default=None,
              help='Sample size to limit memory usage (e.g., 1000000 for 1M samples)')
@click.option('--project-root', type=click.Path(exists=True), default=None,
              help='Project root directory (default: auto-detect)')
def single(
    model_name: str,
    matrix_cell: str,
    track: str,
    horizon_h: int,
    radius_km: Optional[float],
    knn_k: Optional[int],
    config: Optional[str],
    output_dir: Optional[str],
    feature_selection_name: Optional[str],
    data_path: Optional[str],
    sample_size: Optional[int],
    project_root: Optional[str]
):
    """Train a single model.
    
    ⚠️ Hard constraint: Must use TrainingRunner, cannot directly create model or call model.fit.
    
    Examples:
        # Full feature training (278 features)
        python -m src.cli train single --model-name lightgbm --matrix-cell B --track feature_engineering --horizon-h 12
        
        # Feature selection training (15 features)
        python -m src.cli train single --model-name lightgbm --matrix-cell B --track feature_engineering --horizon-h 12 --feature-selection-name top15
        
        # Custom output directory
        python -m src.cli train single --model-name lstm --matrix-cell A --track raw --horizon-h 24 --output-dir experiments/lstm/custom/path
    """
    # Determine project root
    if project_root is None:
        # Auto-detect: go up from src/cli/commands/ to project root
        project_root = Path(__file__).parent.parent.parent.parent
    else:
        project_root = Path(project_root)
    
    # Build CLI overrides
    # CRITICAL FIX: Generate labels for all horizons [3, 6, 12, 24] even when training single horizon
    # This ensures labeled_data.parquet contains all horizon labels
    cli_overrides: Dict[str, Any] = {
        "model": model_name,
        "matrix_cell": matrix_cell,
        "track": track,  # Add track to cli_overrides so it can be used for config loading
        "horizons": [3, 6, 12, 24],  # Generate labels for all horizons
    }
    
    if output_dir:
        cli_overrides["output_dir"] = output_dir
    else:
        # Generate default output dir following 2×2+1 structure
        base_dir = f"experiments/{model_name}/{track}/{matrix_cell}/full_training"
        # Add feature selection suffix if provided
        if feature_selection_name:
            cli_overrides["output_dir"] = f"{base_dir}_{feature_selection_name}"
        else:
            cli_overrides["output_dir"] = base_dir
    
    if data_path:
        cli_overrides["data_path"] = data_path
    
    if sample_size:
        cli_overrides["sample_size"] = sample_size
    
    # Add spatial parameters to feature_engineering config if needed
    if radius_km is not None or knn_k is not None:
        if "feature_engineering" not in cli_overrides:
            cli_overrides["feature_engineering"] = {}
        if "spatial" not in cli_overrides["feature_engineering"]:
            cli_overrides["feature_engineering"]["spatial"] = {}
        if radius_km is not None:
            cli_overrides["feature_engineering"]["spatial"]["radius_km"] = radius_km
        if knn_k is not None:
            cli_overrides["feature_engineering"]["spatial"]["knn_k"] = knn_k
    
    # Load and merge config
    config_path = Path(config) if config else None
    merged_config = load_and_merge_config(config_path, cli_overrides)
    
    # Load training config (this handles YAML parsing and creates PipelineTrainingConfig)
    pipeline_config = load_training_config(
        config_path=config_path,
        project_root=project_root,
        cli_overrides=merged_config
    )
    
    # Create and run TrainingRunner (hard constraint: must use TrainingRunner)
    runner = TrainingRunner(pipeline_config, project_root)
    
    try:
        exit_code = runner.run()
        if exit_code == 0:
            # Get output directory from config
            output_path = pipeline_config.training.output_dir
            click.echo(f"✅ Training complete: {output_path}")
            click.echo(f"   Metadata saved to: {output_path / 'full_training' / f'horizon_{horizon_h}h' / 'run_metadata.json'}")
        else:
            click.echo(f"❌ Training failed with exit code {exit_code}", err=True)
            raise click.Abort()
    except Exception as e:
        _logger.exception("Training failed")
        click.echo(f"❌ Training failed: {e}", err=True)
        raise click.Abort()


@train.command()
@click.option('--config', type=click.Path(exists=True), required=True,
              help='Path to YAML config file with matrix section')
@click.option('--project-root', type=click.Path(exists=True), default=None,
              help='Project root directory (default: auto-detect)')
def matrix(config: str, project_root: Optional[str]):
    """Run matrix experiments from config.
    
    ⚠️ Hard constraint: Must use TrainingRunner, cannot directly create model.
    
    Config format (YAML):
        matrix:
          matrix_cells: ["A", "B", "C", "D", "E"]
          tracks: ["raw", "top175_features"]
          horizons: [3, 6, 12, 24]
          radius_km: [25, 50, 75, 100]  # for C/D
          knn_k: [1, 3, 5]  # for E
        
        model_name: lightgbm
        data:
          source: data/raw/...
    
    Examples:
        python -m src.cli train matrix --config config/pipeline/matrix_experiments.yaml
    """
    # Determine project root
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent.parent
    else:
        project_root = Path(project_root)
    
    config_path = Path(config)
    
    # Load config
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract matrix config
    matrix_config = config_dict.get("matrix", {})
    matrix_cells = matrix_config.get("matrix_cells", ["A", "B", "C", "D", "E"])
    tracks = matrix_config.get("tracks", ["raw", "top175_features"])
    horizons = matrix_config.get("horizons", [3, 6, 12, 24])
    radius_km_list = matrix_config.get("radius_km", [])
    knn_k_list = matrix_config.get("knn_k", [])
    model_name = config_dict.get("model_name", "lightgbm")
    
    click.echo(f"Running matrix experiments:")
    click.echo(f"  Model: {model_name}")
    click.echo(f"  Matrix cells: {matrix_cells}")
    click.echo(f"  Tracks: {tracks}")
    click.echo(f"  Horizons: {horizons}")
    if radius_km_list:
        click.echo(f"  Radius (km): {radius_km_list}")
    if knn_k_list:
        click.echo(f"  KNN k: {knn_k_list}")
    
    # Run experiments (Cartesian product)
    success_count = 0
    total_count = 0
    
    for matrix_cell in matrix_cells:
        for track in tracks:
            for horizon_h in horizons:
                # Determine spatial parameters based on matrix_cell
                if matrix_cell in ["C", "D"]:
                    # C/D use radius_km list
                    spatial_params = radius_km_list if radius_km_list else [None]
                elif matrix_cell == "E":
                    # E uses knn_k list
                    spatial_params = knn_k_list if knn_k_list else [None]
                else:
                    # A/B ignore spatial params
                    spatial_params = [None]
                
                for spatial_param in spatial_params:
                    total_count += 1
                    
                    # Build CLI overrides for this experiment
                    cli_overrides = {
                        "model": model_name,
                        "matrix_cell": matrix_cell,
                        "horizons": [horizon_h],
                    }
                    
                    # Set output dir following 2×2+1 structure
                    if matrix_cell in ["C", "D"] and spatial_param is not None:
                        cli_overrides["output_dir"] = f"experiments/{model_name}/{track}/{matrix_cell}/full_training/radius_{spatial_param}km"
                        cli_overrides["feature_engineering"] = {
                            "spatial": {"radius_km": spatial_param}
                        }
                    elif matrix_cell == "E" and spatial_param is not None:
                        cli_overrides["output_dir"] = f"experiments/{model_name}/{track}/{matrix_cell}/full_training/knn_{spatial_param}"
                        cli_overrides["feature_engineering"] = {
                            "spatial": {"knn_k": spatial_param}
                        }
                    else:
                        cli_overrides["output_dir"] = f"experiments/{model_name}/{track}/{matrix_cell}/full_training"
                    
                    # Merge with base config
                    merged_config = load_and_merge_config(config_path, cli_overrides)
                    
                    # Load training config
                    pipeline_config = load_training_config(
                        config_path=config_path,
                        project_root=project_root,
                        cli_overrides=merged_config
                    )
                    
                    # Run training
                    click.echo(f"\n[{total_count}] Training: {model_name} / {matrix_cell} / {track} / {horizon_h}h" + 
                              (f" / radius={spatial_param}km" if spatial_param else "") +
                              (f" / knn={spatial_param}" if matrix_cell == "E" and spatial_param else ""))
                    
                    try:
                        runner = TrainingRunner(pipeline_config, project_root)
                        exit_code = runner.run()
                        if exit_code == 0:
                            success_count += 1
                            click.echo(f"  ✅ Success")
                        else:
                            click.echo(f"  ❌ Failed (exit code {exit_code})", err=True)
                    except Exception as e:
                        _logger.exception(f"Training failed for {matrix_cell}/{track}/{horizon_h}h")
                        click.echo(f"  ❌ Failed: {e}", err=True)
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Matrix experiments complete: {success_count}/{total_count} successful")
    click.echo(f"{'='*60}")
