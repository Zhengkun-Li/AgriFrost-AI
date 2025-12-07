"""Unified prediction interface."""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import json

from src.models.base import BaseModel
from src.models.registry import get_model_class
from src.data import DataPipeline
from src.training.data_preparation import prepare_features_and_targets
from src.utils.metadata import ExperimentMetadata

_logger = logging.getLogger(__name__)


class FrostPredictor:
    """Unified prediction interface."""
    
    def __init__(self, model_dir: Path, config: Optional[Dict] = None):
        """Initialize predictor.
        
        Args:
            model_dir: Path to model directory (horizon directory, e.g., horizon_12h/)
            config: Optional config override
        """
        self.model_dir = Path(model_dir)
        self.frost_model, self.temp_model, self.config = self._load_models()
        
        # Initialize data pipeline (if feature engineering was used during training)
        self.data_pipeline = self._load_data_pipeline()
    
    def _load_models(self) -> Tuple[BaseModel, BaseModel, Dict]:
        """Load frost and temperature models and config.
        
        Returns:
            Tuple of (frost_model, temp_model, config)
        """
        # Load metadata first to get model info
        metadata_path = self.model_dir / "run_metadata.json"
        if metadata_path.exists():
            metadata = ExperimentMetadata.load(metadata_path)
            model_type = metadata.model_name
            track = metadata.track
            matrix_cell = metadata.matrix_cell
            horizon_h = metadata.horizon_h
        else:
            # Fallback: try to infer from config
            config_path = self.model_dir / "config.yaml"
            if not config_path.exists():
                config_path = self.model_dir / "config.json"
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.suffix == ".yaml":
                        import yaml
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                model_type = config.get("model_type", "lightgbm")
                track = config.get("track", "raw")
                matrix_cell = config.get("matrix_cell", "A")
            else:
                raise FileNotFoundError(
                    f"Neither run_metadata.json nor config file found in {self.model_dir}. "
                    f"Cannot determine model type."
                )
        
        # Build config dict (include metadata for spatial aggregation config)
        config = {
            "model_type": model_type,
            "track": track,
            "matrix_cell": matrix_cell,
            "horizon_h": horizon_h if metadata_path.exists() else None
        }
        
        # Add metadata fields to config (for radius_km, etc.)
        if metadata_path.exists():
            config["radius_km"] = metadata.radius_km if hasattr(metadata, 'radius_km') else None
            config["knn_k"] = metadata.knn_k if hasattr(metadata, 'knn_k') else None
        
        # Load models from frost_classifier and temp_regressor directories
        frost_model_dir = self.model_dir / "frost_classifier"
        temp_model_dir = self.model_dir / "temp_regressor"
        
        if not frost_model_dir.exists():
            raise FileNotFoundError(f"Frost model directory not found: {frost_model_dir}")
        if not temp_model_dir.exists():
            raise FileNotFoundError(f"Temperature model directory not found: {temp_model_dir}")
        
        # Load model class and load models
        model_cls = get_model_class(model_type)
        frost_model = model_cls.load(frost_model_dir)
        temp_model = model_cls.load(temp_model_dir)
        
        return frost_model, temp_model, config
    
    def _load_data_pipeline(self) -> Optional[DataPipeline]:
        """Load data pipeline (if training used feature engineering).
        
        CRITICAL: For Matrix Cell C/D with spatial aggregation, this ensures
        inference uses the same distance_threshold_km as training.
        """
        # Check if pipeline config exists
        pipeline_config_path = self.model_dir / "pipeline_config.yaml"
        if pipeline_config_path.exists():
            import yaml
            with open(pipeline_config_path, 'r', encoding='utf-8') as f:
                pipeline_config = yaml.safe_load(f)
            
            # Log spatial aggregation config for debugging
            fe_config = pipeline_config.get("feature_engineering", {})
            spatial_agg = fe_config.get("spatial_aggregation", {})
            if spatial_agg.get("enabled"):
                distance = spatial_agg.get("distance_threshold_km")
                _logger.info(
                    f"✅ Loaded DataPipeline with spatial_aggregation: "
                    f"enabled=True, distance_threshold_km={distance}km"
                )
            else:
                _logger.debug("Loaded DataPipeline without spatial_aggregation")
            
            return DataPipeline(pipeline_config)
        else:
            _logger.warning(
                f"⚠️  pipeline_config.yaml not found in {self.model_dir}. "
                f"Inference may not correctly generate neighbor features for Matrix Cell C/D."
            )
        return None
    
    def predict(
        self,
        input_data: pd.DataFrame,
        horizons: Optional[List[int]] = None,
        return_proba: bool = True
    ) -> Dict[str, np.ndarray]:
        """Execute prediction.
        
        Args:
            input_data: Input data (raw or feature-engineered)
            horizons: Forecast horizons (default: use model config)
            return_proba: Whether to return probabilities
        
        Returns:
            {
                'temperature': np.ndarray,
                'frost_proba': np.ndarray,  # if return_proba
                'horizons': List[int]
            }
        """
        # 1. Apply feature engineering if pipeline exists
        # CRITICAL: For Matrix Cell C (raw + spatial aggregation), we need to use DataPipeline
        # to correctly generate neighbor features with the same radius_km as training.
        # Since DataPipeline.run() requires data_path, we save to temp file for processing.
        if self.data_pipeline:
            import tempfile
            import shutil
            
            # Get pipeline config to extract feature_engineering config
            pipeline_config_path = self.model_dir / "pipeline_config.yaml"
            if pipeline_config_path.exists():
                import yaml
                with open(pipeline_config_path, 'r', encoding='utf-8') as f:
                    pipeline_config = yaml.safe_load(f)
                
                # Use FeatureEngineer with the complete config from pipeline
                # This ensures spatial_aggregation config is correctly applied
                from src.data.feature_engineering import FeatureEngineer
                feature_engineer = FeatureEngineer()
                
                # Get feature_engineering config from pipeline_config
                fe_config = pipeline_config.get("feature_engineering", {})
                
                # CRITICAL: Use spatial_aggregation config from pipeline_config.yaml
                # This ensures inference uses the same distance_threshold_km as training
                spatial_agg = fe_config.get("spatial_aggregation", {})
                
                # Ensure spatial_aggregation config is present for Matrix Cell C/D
                if self.config.get("matrix_cell") in ["C", "D"]:
                    if "spatial_aggregation" not in fe_config:
                        fe_config["spatial_aggregation"] = {}
                    
                    # Priority: pipeline_config.yaml > run_metadata.json
                    if not spatial_agg.get("enabled") and self.config.get("radius_km"):
                        # Fallback to metadata if pipeline config doesn't have spatial_aggregation
                        fe_config["spatial_aggregation"]["enabled"] = True
                        _logger.info(
                            f"Inference: Using radius_km={self.config.get('radius_km')} from run_metadata.json "
                            f"(pipeline_config.yaml missing spatial_aggregation config)"
                        )
                    elif spatial_agg.get("enabled"):
                        distance = spatial_agg.get("distance_threshold_km")
                        _logger.info(
                            f"Inference: Using spatial_aggregation config from pipeline_config.yaml: "
                            f"distance_threshold_km={distance}km"
                        )
                        # Verify consistency with metadata
                        if self.config.get("radius_km") and distance != self.config.get("radius_km"):
                            _logger.warning(
                                f"Inference: distance_threshold_km mismatch - "
                                f"pipeline_config.yaml: {distance}km, "
                                f"run_metadata.json: {self.config.get('radius_km')}km. "
                                f"Using pipeline_config.yaml value."
                            )
                    
                    # Ensure distance_threshold_km is set
                    if "distance_threshold_km" not in fe_config["spatial_aggregation"]:
                        if self.config.get("radius_km"):
                            fe_config["spatial_aggregation"]["distance_threshold_km"] = self.config.get("radius_km")
                        else:
                            _logger.warning(
                                f"Inference: Matrix Cell {self.config.get('matrix_cell')} requires distance_threshold_km, "
                                f"but neither pipeline_config.yaml nor run_metadata.json provides it. "
                                f"Spatial aggregation may not work correctly."
                            )
                
                if fe_config:
                    df = feature_engineer.build_feature_set(input_data, fe_config)
                else:
                    df = input_data
            else:
                # Fallback: use data_pipeline directly if available
                # Save to temp file and use DataPipeline.run()
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                    input_data.to_csv(tmp_file.name, index=False)
                    tmp_path = Path(tmp_file.name)
                
                try:
                    # Use DataPipeline.run() with generate_labels=False
                    bundle = self.data_pipeline.run(
                        data_path=tmp_path,
                        horizons=[],
                        use_feature_engineering=None,
                        feature_config=self.data_pipeline.default_feature_config,
                        generate_labels=False
                    )
                    df = bundle.data
                finally:
                    # Clean up temp file
                    if tmp_path.exists():
                        tmp_path.unlink()
        else:
            df = input_data
        
        # 2. Prepare features and targets (for horizon-specific prediction)
        # Get horizon from config or use first horizon in list
        horizon = horizons[0] if horizons else self.config.get("horizon_h", 3)
        if horizon is None:
            horizon = 3  # Default fallback
        track = self.config.get("track", "raw")
        
        X, _, _ = prepare_features_and_targets(
            df,
            horizon,
            track=track,
            model_type=self.config.get("model_type", "lightgbm")
        )
        
        # CRITICAL: Verify feature names match training feature names
        # This ensures consistency between training and inference
        if hasattr(self.frost_model, 'feature_names') and self.frost_model.feature_names:
            expected_features = self.frost_model.feature_names
            actual_features = list(X.columns)
            
            # Check if feature names match (in order)
            if actual_features != expected_features:
                # Try reordering to match expected features
                missing_features = set(expected_features) - set(actual_features)
                extra_features = set(actual_features) - set(expected_features)
                
                if missing_features:
                    _logger.warning(
                        f"Missing features in inference data: {len(missing_features)} features. "
                        f"First 5: {list(missing_features)[:5]}"
                    )
                if extra_features:
                    _logger.warning(
                        f"Extra features in inference data: {len(extra_features)} features. "
                        f"First 5: {list(extra_features)[:5]}"
                    )
                
                # Reorder and select only expected features
                available_features = [f for f in expected_features if f in actual_features]
                if len(available_features) < len(expected_features):
                    _logger.error(
                        f"Cannot match features: only {len(available_features)}/{len(expected_features)} "
                        f"expected features found. Prediction may fail."
                    )
                    raise ValueError(
                        f"Feature mismatch: expected {len(expected_features)} features, "
                        f"but only {len(available_features)} found. Missing: {list(missing_features)[:10]}"
                    )
                # Reorder X to match expected feature order
                X = X[available_features]
            else:
                # Ensure order matches exactly (already correct, but make sure)
                X = X[expected_features]
        
        # 3. Call models.predict() - use separate models for frost and temperature
        temp_pred = self.temp_model.predict(X)
        frost_proba = None
        if return_proba and hasattr(self.frost_model, "predict_proba"):
            frost_proba = self.frost_model.predict_proba(X)
        
        return {
            'temperature': temp_pred,
            'frost_proba': frost_proba,
            'horizons': horizons or self.config.get("horizons", [3, 6, 12, 24])
        }
    
    def predict_from_file(
        self,
        input_path: Path,
        output_path: Path,
        horizons: Optional[List[int]] = None
    ) -> None:
        """Predict from file and save to file.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save predictions
            horizons: Forecast horizons
        """
        # Load input data
        df = pd.read_csv(input_path)
        
        # Store input columns for output
        input_columns = df.columns.tolist()
        
        # Predict
        results = self.predict(df, horizons)
        
        # Format output - preserve input columns and add predictions
        output_df = df.copy()
        output_df['temperature'] = results['temperature']
        
        if results['frost_proba'] is not None:
            output_df['frost_proba'] = results['frost_proba']
            # Add binary prediction: frost_pred = 1 if frost_proba >= 0.5, else 0
            output_df['frost_pred'] = (results['frost_proba'] >= 0.5).astype(int)
        else:
            output_df['frost_proba'] = None
            output_df['frost_pred'] = None
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        
        _logger.info(f"Predictions saved to {output_path}")
        _logger.info(f"Output columns: {list(output_df.columns)}")
    
    @staticmethod
    def format_prediction_message(
        frost_proba: float,
        temperature: float,
        horizon_h: int
    ) -> str:
        """Format prediction message.
        
        Returns:
            "There is a 30% chance of frost in the next 3 hours, predicted temperature: 4.50 °C"
        """
        proba_pct = int(frost_proba * 100) if frost_proba is not None else 0
        return (
            f"There is a {proba_pct}% chance of frost in the next {horizon_h} hours, "
            f"predicted temperature: {temperature:.2f} °C"
        )

