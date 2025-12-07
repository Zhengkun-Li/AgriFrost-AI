#!/usr/bin/env python3
"""Quick test script for deep learning models with small dataset.

This script tests deep learning models (LSTM, GRU, TCN) using a small subset of data
to verify they work correctly without consuming too much memory.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.training.data_preparation import prepare_features_and_targets, load_and_prepare_data
from src.training.model_config import get_model_config, get_model_class
from src.data import DataPipeline
from src.training.pipeline_runner import load_training_config, _build_pipeline_config

def test_deep_model(model_name: str, horizon: int = 3, max_samples: int = 10000):
    """Test a deep learning model with small dataset.
    
    Args:
        model_name: Model name (lstm, gru, tcn)
        horizon: Forecast horizon in hours
        max_samples: Maximum number of samples to use
    """
    logger.info(f"="*80)
    logger.info(f"Testing {model_name.upper()} model")
    logger.info(f"="*80)
    
    try:
        # Load training config for Matrix A (simple, no spatial aggregation)
        config_path = project_root / "config" / "pipeline" / "matrix_a.yaml"
        
        logger.info("Loading training configuration...")
        training_config = load_training_config(
            config_path=str(config_path),
            project_root=project_root,
            cli_overrides={"matrix_cell": "A", "model": model_name}
        )
        
        # Build pipeline config and create pipeline
        pipeline_config = _build_pipeline_config(training_config.data, training_config.labels)
        pipeline = DataPipeline(config=pipeline_config)
        
        # Load and prepare data with labels
        logger.info(f"Loading data and generating labels (max {max_samples} samples)...")
        data_path = training_config.data.source
        
        # Use pipeline.run to generate labels
        bundle = pipeline.run(
            data_path=data_path,
            horizons=training_config.labels.horizons
        )
        
        # Get processed data with labels
        processed_data = bundle.data
        
        # Limit data size after processing
        if len(processed_data) > max_samples:
            logger.info(f"Sampling {max_samples} rows from {len(processed_data)} total rows")
            processed_data = processed_data.sample(n=max_samples, random_state=42).sort_index()
        
        logger.info(f"Using {len(processed_data)} samples for testing")
        
        # Prepare features and targets
        logger.info("Preparing features and targets...")
        X, y_frost, y_temp = prepare_features_and_targets(
            df=processed_data,
            horizon=horizon,
            track=training_config.data.track or "raw"
        )
        
        # Extract station IDs if available (for deep learning models that need sequence grouping)
        station_ids = None
        if 'Stn Id' in processed_data.columns:
            # Get station IDs for samples that are in X
            station_ids = processed_data.loc[X.index, 'Stn Id'].values if hasattr(X, 'index') else processed_data['Stn Id'].values[:len(X)]
        elif 'station_id' in processed_data.columns:
            station_ids = processed_data.loc[X.index, 'station_id'].values if hasattr(X, 'index') else processed_data['station_id'].values[:len(X)]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Frost target shape: {y_frost.shape}")
        logger.info(f"Temp target shape: {y_temp.shape}")
        
        # Split into train/validation (80/20)
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_val = X[split_idx:]
        y_frost_train = y_frost[:split_idx]
        y_frost_val = y_frost[split_idx:]
        y_temp_train = y_temp[:split_idx]
        y_temp_val = y_temp[split_idx:]
        station_ids_train = station_ids[:split_idx] if station_ids is not None else None
        station_ids_val = station_ids[split_idx:] if station_ids is not None else None
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        
        # Get model config (use small config for quick test)
        logger.info(f"Getting model configuration...")
        config = get_model_config(model_name, horizon, "classification", for_loso=True)
        config['model_params']['epochs'] = 5  # Very short training for test
        config['model_params']['batch_size'] = 32
        config['model_params']['patience'] = 3
        
        # Get model class
        model_class = get_model_class(model_name)
        
        # Test classification model
        logger.info(f"Testing {model_name} classification model...")
        model_frost = model_class(config)
        
        # Train
        logger.info("Training classification model...")
        model_frost.fit(
            X_train, 
            y_frost_train,
            validation_data=(X_val, y_frost_val),
            station_ids=station_ids_train,
            validation_station_ids=station_ids_val
        )
        
        # Predict
        logger.info("Making predictions...")
        predictions = model_frost.predict(X_val[:100])  # Small subset for speed
        proba = model_frost.predict_proba(X_val[:100])
        
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Probabilities shape: {proba.shape}")
        logger.info(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        logger.info(f"Probability range: [{proba.min():.4f}, {proba.max():.4f}]")
        
        # Test regression model
        logger.info(f"Testing {model_name} regression model...")
        config_reg = get_model_config(model_name, horizon, "regression", for_loso=True)
        config_reg['model_params']['epochs'] = 5
        config_reg['model_params']['batch_size'] = 32
        config_reg['model_params']['patience'] = 3
        
        model_temp = model_class(config_reg)
        
        logger.info("Training regression model...")
        model_temp.fit(
            X_train,
            y_temp_train,
            validation_data=(X_val, y_temp_val),
            station_ids=station_ids_train,
            validation_station_ids=station_ids_val
        )
        
        temp_predictions = model_temp.predict(X_val[:100])
        logger.info(f"Temperature predictions shape: {temp_predictions.shape}")
        logger.info(f"Temperature prediction range: [{temp_predictions.min():.2f}, {temp_predictions.max():.2f}]")
        
        logger.info(f"✅ {model_name.upper()} test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ {model_name.upper()} test FAILED: {e}", exc_info=True)
        return False


def main():
    """Run quick tests for all deep learning models."""
    models_to_test = ['lstm', 'gru', 'tcn']
    horizon = 3
    max_samples = 5000  # Very small dataset for quick test
    
    logger.info("Starting deep learning models quick test")
    logger.info(f"Using {max_samples} samples, horizon {horizon}h")
    logger.info("")
    
    results = {}
    for model_name in models_to_test:
        try:
            success = test_deep_model(model_name, horizon, max_samples)
            results[model_name] = success
            logger.info("")
        except KeyboardInterrupt:
            logger.warning("Test interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}", exc_info=True)
            results[model_name] = False
    
    # Summary
    logger.info("="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    for model_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{model_name.upper()}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    logger.info(f"\nTotal: {passed}/{total} models passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

