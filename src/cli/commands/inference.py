"""Inference commands."""

import click
import logging
from pathlib import Path
from typing import List, Optional

from src.inference.predictor import FrostPredictor

_logger = logging.getLogger(__name__)


@click.group()
def inference():
    """Inference commands."""
    pass


@inference.command()
@click.option('--model-dir', type=click.Path(exists=True), required=True,
              help='Path to trained model directory')
@click.option('--input', type=click.Path(exists=True), required=True,
              help='Path to input CSV file')
@click.option('--output', type=click.Path(), required=True,
              help='Path to save predictions CSV')
@click.option('--horizon-h', '--horizon', type=int, multiple=True,
              help='Forecast horizons (can specify multiple, e.g., --horizon-h 3 --horizon-h 12)')
def predict(model_dir: str, input: str, output: str, horizon_h: tuple):
    """Generate frost predictions from trained model.
    
    Examples:
        python -m src.cli inference predict --model-dir experiments/lightgbm_B_12h --input data/test.csv --output predictions.csv
    """
    model_dir = Path(model_dir)
    input_path = Path(input)
    output_path = Path(output)
    
    # Check if model directory has run_metadata.json (should exist for standard runs)
    metadata_path = model_dir / "run_metadata.json"
    if metadata_path.exists():
        from src.utils.metadata import ExperimentMetadata
        metadata = ExperimentMetadata.load(metadata_path)
        click.echo(f"Model: {metadata.model_name}")
        click.echo(f"  Matrix cell: {metadata.matrix_cell}")
        click.echo(f"  Track: {metadata.track}")
        click.echo(f"  Horizon: {metadata.horizon_h}h")
    
    # Initialize predictor
    try:
        predictor = FrostPredictor(model_dir)
    except Exception as e:
        _logger.exception("Failed to load model")
        click.echo(f"❌ Failed to load model: {e}", err=True)
        raise click.Abort()
    
    # Predict
    horizons = list(horizon_h) if horizon_h else None
    
    try:
        predictor.predict_from_file(input_path, output_path, horizons)
        click.echo(f"✅ Predictions saved to: {output_path}")
        
        # Show sample prediction message
        import pandas as pd
        results_df = pd.read_csv(output_path, encoding='utf-8')
        if len(results_df) > 0:
            sample = results_df.iloc[0]
            horizon = horizons[0] if horizons else 3
            message = FrostPredictor.format_prediction_message(
                sample.get('frost_proba', 0),
                sample.get('temperature', 0),
                horizon
            )
            click.echo(f"\nSample prediction: {message}")
    except Exception as e:
        _logger.exception("Prediction failed")
        click.echo(f"❌ Prediction failed: {e}", err=True)
        raise click.Abort()
