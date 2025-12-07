"""Configuration-driven training runner built on DataPipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time
import hashlib

import pandas as pd
import numpy as np
import yaml

_logger = logging.getLogger(__name__)

from src.utils.path_utils import ensure_dir
from src.utils.metadata import ExperimentMetadata
from src.data import DataPipeline
from src.data.features.constants import DATE_COL, STATION_ID_COL, TEMP_COL
from src.training.data_preparation import prepare_features_and_targets
from src.training.model_trainer import train_models_for_horizon
from src.training.loso_evaluator import perform_loso_evaluation
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.registry import (
    register_evaluation_strategy,
    get_evaluation_handler,
)
from src.models.registry import get_model_class


@dataclass
class DataSection:
    source: Path
    matrix_cell: str
    track: Optional[str] = None  # Track name: "raw" or "feature_engineering"
    sample_size: Optional[int] = None
    cleaning: Dict[str, Any] = field(default_factory=dict)
    feature_engineering: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelSection:
    horizons: List[int]
    frost_threshold: float = 0.0


@dataclass
class TrainingSection:
    model: str
    output_dir: Path
    feature_selection: Optional[Path] = None
    feature_selection_config: Optional[Path] = None
    top_k: Optional[int] = None


@dataclass
class EvaluationTask:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSection:
    tasks: List[EvaluationTask] = field(default_factory=list)


@dataclass
class InferenceSection:
    model_dir: Path
    output_dir: Path
    model_type: Optional[str] = None
    horizons: Optional[List[int]] = None


@dataclass
class PipelineTrainingConfig:
    data: DataSection
    labels: LabelSection
    training: Optional[TrainingSection] = None
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    inference: Optional[InferenceSection] = None


def _resolve_path(path: Optional[Any], project_root: Path) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = project_root / p
    return p


def _require_path(path_value: Any, project_root: Path, field_name: str) -> Path:
    resolved = _resolve_path(path_value, project_root)
    if resolved is None:
        raise ValueError(f"{field_name} must be provided in pipeline config.")
    return resolved


def _build_pipeline_config(data_cfg: DataSection, label_cfg: LabelSection) -> Dict[str, Any]:
    return {
        "cleaning": data_cfg.cleaning,
        "feature_engineering": data_cfg.feature_engineering,
        "labels": {"threshold": label_cfg.frost_threshold},
    }


def _infer_track(matrix_cell: Optional[str]) -> str:
    """Infer feature engineering track from matrix cell.
    
    Args:
        matrix_cell: Matrix cell identifier (A, B, C, D, E, etc.).
    
    Returns:
        Track name: "raw" for raw features, "feature_engineering" for feature engineering.
        Note: Supports legacy "top175_features" but defaults to "feature_engineering".
    
    Raises:
        ValueError: If matrix_cell is not recognized.
    """
    if not matrix_cell:
        raise ValueError("matrix_cell must be provided to infer track")
    
    matrix_cell_upper = matrix_cell.upper()
    
    # Define matrix cells by feature engineering requirement
    RAW_CELLS = {"A", "C", "E"}  # Raw features only
    FE_CELLS = {"B", "D"}  # Feature engineering required
    
    if matrix_cell_upper in RAW_CELLS:
        return "raw"
    elif matrix_cell_upper in FE_CELLS:
        # Return "feature_engineering" (new naming) instead of "top175_features" (legacy)
        return "feature_engineering"
    else:
        raise ValueError(
            f"Unknown matrix cell: {matrix_cell}. "
            f"Supported cells: RAW={RAW_CELLS}, FE={FE_CELLS}"
        )


def _load_model_from_dir(model_path: Path, fallback_type: Optional[str]) -> Tuple[Any, str]:
    model_type = fallback_type
    config_path = model_path / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", model_type)
    if model_type is None:
        raise ValueError(f"Model type could not be inferred for {model_path}")
    model_cls = get_model_class(model_type)
    return model_cls.load(model_path), model_type


def load_training_config(
    config_path: Optional[Path],
    project_root: Path,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> PipelineTrainingConfig:
    """Load training config from YAML with CLI argument overrides.
    
    Strategy:
    1. If config_path is explicitly provided, use it
    2. If not, try to load matrix_{cell}.yaml based on matrix_cell from CLI
    3. If matrix_{cell}.yaml doesn't exist, fall back to default.yaml
    4. This allows each matrix cell (A, B, C, D) to have its own explicit configuration
    
    Args:
        config_path: Optional path to YAML config. If None, will auto-select based on matrix_cell.
        project_root: Project root directory.
        cli_overrides: Dict of CLI argument overrides (model, matrix_cell, output_dir, etc.).
    
    Returns:
        PipelineTrainingConfig instance.
    """
    cli = cli_overrides or {}
    
    # Determine matrix_cell early (from CLI) to potentially select matrix-specific config
    matrix_cell_from_cli = cli.get("matrix_cell")
    
    # Load config: explicit path > matrix-specific config > default
    if config_path is None:
        # Try to load matrix-specific config if matrix_cell is known
        if matrix_cell_from_cli:
            matrix_cell_upper = matrix_cell_from_cli.upper()
            if matrix_cell_upper in {"A", "B", "C", "D", "E"}:
                matrix_config = project_root / "config" / "pipeline" / f"matrix_{matrix_cell_upper.lower()}.yaml"
                if matrix_config.exists():
                    config_path = matrix_config
                    _logger.info(f"📋 Auto-selected matrix-specific config: {config_path.relative_to(project_root)}")
        
        # Fall back to default.yaml if no matrix-specific config found
        if config_path is None:
            default_config = project_root / "config" / "pipeline" / "default.yaml"
            if default_config.exists():
                config_path = default_config
            else:
                # If default.yaml doesn't exist, use empty dict
                raw = {}
    else:
        config_path = Path(config_path)
    
    # Load YAML if we have a config path
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}
    
    # Merge CLI overrides into config sections
    data_section = raw.get("data", {}).copy()
    if "data_path" in cli:
        data_section["source"] = str(cli["data_path"])
    elif "data" in cli and cli["data"]:
        data_section["source"] = str(cli["data"])
    if "sample_size" in cli:
        data_section["sample_size"] = cli["sample_size"]
    
    # Determine final matrix_cell: CLI override > YAML > default
    # Note: We avoid inferring from output_dir path to prevent false positives (e.g., "AINet" contains "A")
    matrix_cell = cli.get("matrix_cell")
    if not matrix_cell:
        matrix_cell = data_section.get("matrix_cell")
    if not matrix_cell:
        # Only infer from output_dir if it's a known pattern: experiments/model/{CELL}/...
        output_dir_str = cli.get("output_dir")
        if output_dir_str:
            output_parts = str(output_dir_str).split("/")
            # Look for single-letter cells in specific positions (not in model names)
            for i, part in enumerate(output_parts):
                if part.upper() in {"A", "B", "C", "D", "E"} and len(part) == 1:
                    # Only accept if it's in a reasonable position (not part of a model name)
                    if i > 0 and output_parts[i-1] not in ["experiments", "results"]:
                        matrix_cell = part.upper()
                        break
    matrix_cell = matrix_cell or "B"  # Default to B if still not found
    
    # Auto-select cleaning config based on matrix_cell if not specified
    if not data_section.get("cleaning", {}).get("config_path"):
        if matrix_cell in {"A", "C"}:
            cleaning_path = "config/data_cleaning_raw.yaml"
        elif matrix_cell == "E":
            cleaning_path = "config/data_cleaning_graph.yaml"
        else:  # B, D
            cleaning_path = "config/data_cleaning_fe.yaml"
        if "cleaning" not in data_section:
            data_section["cleaning"] = {}
        data_section["cleaning"]["config_path"] = cleaning_path
    
    # Handle direct radius_km/knn_k CLI parameters (for convenience)
    # Convert them to feature_engineering.spatial format
    if "radius_km" in cli or "knn_k" in cli:
        if "feature_engineering" not in cli:
            cli["feature_engineering"] = {}
        if "spatial" not in cli["feature_engineering"]:
            cli["feature_engineering"]["spatial"] = {}
        if "radius_km" in cli:
            cli["feature_engineering"]["spatial"]["radius_km"] = cli["radius_km"]
        if "knn_k" in cli:
            cli["feature_engineering"]["spatial"]["knn_k"] = cli["knn_k"]
    
    # Handle feature_engineering at top level (for CLI convenience, e.g., Matrix Cell C)
    # CLI may pass feature_engineering at top level, merge it into data.feature_engineering
    if "feature_engineering" in cli and cli["feature_engineering"]:
        if "feature_engineering" not in data_section:
            data_section["feature_engineering"] = {}
        # Deep merge top-level feature_engineering into data.feature_engineering
        top_level_fe = cli["feature_engineering"]
        if isinstance(top_level_fe, dict):
            for key, value in top_level_fe.items():
                if key == "enabled":
                    # Only set enabled if not already set in data_section
                    if "enabled" not in data_section["feature_engineering"]:
                        data_section["feature_engineering"]["enabled"] = value
                elif isinstance(value, dict) and key in data_section["feature_engineering"]:
                    # Deep merge nested dicts (e.g., spatial)
                    if isinstance(data_section["feature_engineering"][key], dict):
                        data_section["feature_engineering"][key].update(value)
                    else:
                        data_section["feature_engineering"][key] = value
                else:
                    # Overwrite or add new keys (e.g., spatial)
                    data_section["feature_engineering"][key] = value
    
    # Initialize feature_engineering section if not present
    if "feature_engineering" not in data_section:
        data_section["feature_engineering"] = {}
    
    # Get track: Priority: CLI > YAML config > infer from matrix_cell
    track = cli.get("track")
    if not track:
        # Try to get track from YAML config (e.g., from matrix_a.yaml, matrix_b.yaml, etc.)
        track = data_section.get("track")
    if not track:
        # Fall back to inference from matrix_cell (backward compatibility)
        track = "feature_engineering" if matrix_cell in {"B", "D"} else "raw"
    
    # Get feature_engineering.enabled: Priority: YAML config > infer from track
    # If YAML explicitly sets enabled=True/False, use that; otherwise infer from track
    fe_enabled = data_section.get("feature_engineering", {}).get("enabled")
    if fe_enabled is None:
        # Infer from track if not explicitly set in YAML
        fe_enabled = track not in {"raw", None}
    
    # Ensure feature_engineering section exists and set enabled
    if "feature_engineering" not in data_section:
        data_section["feature_engineering"] = {}
    
    # Preserve existing feature_engineering config (especially spatial config for Matrix Cell C/D)
    existing_config = data_section["feature_engineering"].copy()
    # Set enabled flag (CLI can override, but YAML config is preferred for initial setup)
    if cli.get("feature_engineering") and isinstance(cli["feature_engineering"], dict):
        # CLI explicitly sets feature_engineering.enabled
        if "enabled" in cli["feature_engineering"]:
            fe_enabled = cli["feature_engineering"]["enabled"]
    
    data_section["feature_engineering"]["enabled"] = fe_enabled
    # Restore all other config (especially spatial for Matrix Cell C/D)
    for key, value in existing_config.items():
        if key not in {"enabled", "_track"}:
            data_section["feature_engineering"][key] = value
    
    # Store track in config for later use (e.g., in TrainingRunner)
    data_section["track"] = track
    data_section["feature_engineering"]["_track"] = track
    
    # CRITICAL: For Matrix Cell C/D, if radius_km is provided via CLI, enable spatial aggregation
    # Check if spatial.radius_km exists in feature_engineering config
    if matrix_cell in {"C", "D"}:
        fe_spatial = data_section.get("feature_engineering", {}).get("spatial", {})
        radius_km = fe_spatial.get("radius_km")
        
        if radius_km is not None and radius_km > 0:
            # Enable spatial aggregation and map radius_km to distance_threshold_km
            if "spatial_aggregation" not in data_section["feature_engineering"]:
                data_section["feature_engineering"]["spatial_aggregation"] = {}
            
            data_section["feature_engineering"]["spatial_aggregation"]["enabled"] = True
            data_section["feature_engineering"]["spatial_aggregation"]["distance_threshold_km"] = radius_km
            
            _logger.info(
                f"✅ Matrix Cell {matrix_cell}: Enabled spatial aggregation with radius_km={radius_km}km"
            )

    spatial_cfg = data_section.get("feature_engineering", {}).get("spatial_aggregation")
    if spatial_cfg and spatial_cfg.get("enabled"):
        default_cache_dir = project_root / "experiments" / "graph_cache"
        if not spatial_cfg.get("cache_dir"):
            spatial_cfg["cache_dir"] = str(default_cache_dir)
        if "use_cache" not in spatial_cfg:
            spatial_cfg["use_cache"] = True
        if "refresh_cache" not in spatial_cfg:
            spatial_cfg["refresh_cache"] = False
        if not spatial_cfg.get("cache_key"):
            source = data_section.get("source")
            source_hash = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8] if source else "nosrc"
            radius_value = spatial_cfg.get("distance_threshold_km") or data_section.get("feature_engineering", {}).get("spatial", {}).get("radius_km")
            radius_part = f"radius_{int(radius_value)}km" if radius_value else "radius_0km"
            sample_size = data_section.get("sample_size")
            sample_part = f"sample_{sample_size}" if sample_size else "full"
            track_part = track or data_section.get("track") or "raw"
            cache_key = f"{matrix_cell}_{track_part}_{radius_part}_{sample_part}_{source_hash}"
            spatial_cfg["cache_key"] = cache_key.strip()
    
    # Get feature_engineering.config_path: Priority: YAML config > auto-select based on track
    # If YAML explicitly sets config_path, use it; otherwise auto-select
    if "config_path" not in data_section.get("feature_engineering", {}) or data_section["feature_engineering"].get("config_path") is None:
        if fe_enabled and track in {"top175_features", "feature_engineering"}:
            # Auto-select config_path based on track name
            if track == "top175_features":
                # Legacy: use top175.yaml if it exists, otherwise use feature_engineering.yaml
                legacy_path = project_root / "config" / "feature_engineering" / "top175.yaml"
                if legacy_path.exists():
                    data_section["feature_engineering"]["config_path"] = "config/feature_engineering/top175.yaml"
                else:
                    data_section["feature_engineering"]["config_path"] = "config/feature_engineering/feature_engineering.yaml"
            else:
                # New: use feature_engineering.yaml
                data_section["feature_engineering"]["config_path"] = "config/feature_engineering/feature_engineering.yaml"
    elif not fe_enabled and "config_path" in data_section.get("feature_engineering", {}):
        # Remove config_path for raw track to avoid loading non-existent config
        # But keep spatial aggregation config if provided via CLI
        data_section["feature_engineering"].pop("config_path", None)
    
    labels_section = raw.get("labels", {}).copy()
    if "horizons" in cli:
        labels_section["horizons"] = cli["horizons"]
    if "frost_threshold" in cli:
        labels_section["frost_threshold"] = cli["frost_threshold"]
    
    training_section = raw.get("training", {}).copy()
    if "model" in cli:
        training_section["model"] = cli["model"]
    if "output_dir" in cli:
        training_section["output_dir"] = str(cli["output_dir"])
    elif not training_section.get("output_dir"):
        # Generate default output dir if not specified
        model_name = cli.get("model") or training_section.get("model", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_section["output_dir"] = f"experiments/{model_name}/{matrix_cell}/{timestamp}"
    if "feature_selection" in cli:
        training_section["feature_selection"] = cli["feature_selection"]
    if "top_k_features" in cli:
        training_section["top_k"] = cli["top_k_features"]
    
    evaluation_section = raw.get("evaluation", {}).copy()
    # Convert legacy loso flags to tasks format if needed
    if "loso" in cli:
        if not isinstance(evaluation_section.get("tasks"), list):
            evaluation_section["tasks"] = []
        # Find or create loso task
        loso_task = None
        for task in evaluation_section["tasks"]:
            if task.get("type") == "loso":
                loso_task = task
                break
        if loso_task is None:
            loso_task = {"type": "loso"}
            evaluation_section["tasks"].append(loso_task)
        loso_task["enabled"] = cli["loso"]
    if "save_loso_models" in cli:
        if not isinstance(evaluation_section.get("tasks"), list):
            evaluation_section["tasks"] = []
        loso_task = next((t for t in evaluation_section["tasks"] if t.get("type") == "loso"), None)
        if loso_task is None:
            loso_task = {"type": "loso"}
            evaluation_section["tasks"].append(loso_task)
        loso_task["save_models"] = cli["save_loso_models"]
    if "save_loso_worst_n" in cli:
        if not isinstance(evaluation_section.get("tasks"), list):
            evaluation_section["tasks"] = []
        loso_task = next((t for t in evaluation_section["tasks"] if t.get("type") == "loso"), None)
        if loso_task is None:
            loso_task = {"type": "loso"}
            evaluation_section["tasks"].append(loso_task)
        loso_task["save_worst_n"] = cli["save_loso_worst_n"]
    if "save_loso_horizons" in cli:
        if not isinstance(evaluation_section.get("tasks"), list):
            evaluation_section["tasks"] = []
        loso_task = next((t for t in evaluation_section["tasks"] if t.get("type") == "loso"), None)
        if loso_task is None:
            loso_task = {"type": "loso"}
            evaluation_section["tasks"].append(loso_task)
        loso_task["save_horizons"] = cli["save_loso_horizons"]
    if "resume_loso" in cli:
        if not isinstance(evaluation_section.get("tasks"), list):
            evaluation_section["tasks"] = []
        loso_task = next((t for t in evaluation_section["tasks"] if t.get("type") == "loso"), None)
        if loso_task is None:
            loso_task = {"type": "loso"}
            evaluation_section["tasks"].append(loso_task)
        loso_task["resume"] = cli["resume_loso"]
    
    # Build config objects
    data_source = data_section.get("source")
    if data_source:
        data_source = _require_path(data_source, project_root, "data.source")
    else:
        # Default data source
        data_source = project_root / "data" / "raw" / "frost-risk-forecast-challenge" / "stations"
        if not data_source.exists():
            data_source = project_root / "data" / "raw" / "frost-risk-forecast-challenge" / "cimis_all_stations.csv.gz"
    
    data = DataSection(
        source=data_source,
        matrix_cell=matrix_cell,
        track=track,  # Set track explicitly from config/CLI/inference
        sample_size=data_section.get("sample_size"),
        cleaning=data_section.get("cleaning", {}),
        feature_engineering=data_section.get("feature_engineering", {}),
    )
    labels = LabelSection(
        horizons=labels_section.get("horizons", [3, 6, 12, 24]),
        frost_threshold=labels_section.get("frost_threshold", 0.0),
    )
    
    training = None
    if training_section:
        output_dir = training_section.get("output_dir")
        if output_dir:
            output_dir = _require_path(output_dir, project_root, "training.output_dir")
        else:
            raise ValueError("training.output_dir is required (specify via --output or YAML)")
        
        training = TrainingSection(
            model=training_section["model"],
            output_dir=output_dir,
            feature_selection=_resolve_path(training_section.get("feature_selection"), project_root),
            feature_selection_config=_resolve_path(training_section.get("feature_selection_config"), project_root),
            top_k=training_section.get("top_k"),
        )

    tasks: List[EvaluationTask] = []
    if isinstance(evaluation_section.get("tasks"), list):
        for task in evaluation_section["tasks"]:
            if not task:
                continue
            task_type = task.get("type")
            if not task_type:
                continue
            # Skip disabled tasks
            if not task.get("enabled", True):
                continue
            params = {k: v for k, v in task.items() if k not in ("type", "enabled")}
            tasks.append(EvaluationTask(type=task_type, params=params))
    else:
        # Backwards compatibility for legacy direct/loso keys
        if evaluation_section.get("direct"):
            tasks.append(EvaluationTask(type="direct", params=evaluation_section["direct"]))
        if evaluation_section.get("loso") and evaluation_section["loso"].get("enabled", False):
            loso_params = {k: v for k, v in evaluation_section["loso"].items() if k != "enabled"}
            tasks.append(EvaluationTask(type="loso", params=loso_params))
    evaluation = EvaluationSection(tasks=tasks)

    inference = None
    inference_section = raw.get("inference")
    if inference_section:
        inference = InferenceSection(
            model_dir=_require_path(inference_section.get("model_dir"), project_root, "inference.model_dir"),
            output_dir=_require_path(inference_section.get("output_dir"), project_root, "inference.output_dir"),
            model_type=inference_section.get("model_type"),
            horizons=inference_section.get("horizons"),
        )

    return PipelineTrainingConfig(
        data=data,
        labels=labels,
        training=training,
        evaluation=evaluation,
        inference=inference,
    )


class TrainingRunner:
    """Execute training pipeline based on configuration."""

    def __init__(self, config: PipelineTrainingConfig, project_root: Path):
        if config.training is None:
            raise ValueError("Training configuration is required for TrainingRunner.")
        self.config = config
        self.project_root = project_root
        # Try to get track from config (set by load_training_config from CLI), otherwise infer
        # Check if track is stored in config data section
        fe_config = self.config.data.feature_engineering
        if isinstance(fe_config, dict):
            self.track = fe_config.get("_track")  # Track may be stored here
        else:
            self.track = None
        # If not found in config, infer from matrix_cell (legacy behavior)
        if not self.track:
            self.track = _infer_track(self.config.data.matrix_cell)

    def run(self) -> int:
        output_dir = self.config.training.output_dir
        ensure_dir(output_dir)
        
        # Setup experiment-level log file
        experiment_log_file = output_dir / "experiment.log"
        experiment_log_file.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        experiment_start_time = datetime.now()
        
        def write_experiment_log(message: str):
            """Helper to append messages to experiment log file."""
            with open(experiment_log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        
        with open(experiment_log_file, 'w', encoding='utf-8') as f:
            f.write(f"Experiment Log - {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n")
            f.write(f"Model: {self.config.training.model}\n")
            f.write(f"Matrix Cell: {self.config.data.matrix_cell}\n")
            f.write(f"Track: {self.track}\n")
            f.write(f"Horizons: {self.config.labels.horizons}\n")
            f.write(f"Output Directory: {output_dir}\n")
            f.write("=" * 70 + "\n\n")
        _logger.info(f"Experiment log initialized at {experiment_log_file}")
        
        pipeline_config = _build_pipeline_config(self.config.data, self.config.labels)

        # Validate spatial parameters for Matrix Cell C/D
        if self.config.data.matrix_cell in ["C", "D"]:
            fe_config = self.config.data.feature_engineering
            if isinstance(fe_config, dict):
                spatial_config = fe_config.get("spatial", {})
                radius_km = spatial_config.get("radius_km") if spatial_config else None
                if radius_km is None:
                    _logger.warning(
                        f"⚠️  Matrix Cell {self.config.data.matrix_cell} requires radius_km parameter, "
                        f"but it was not provided. Spatial aggregation may not work correctly."
                    )
                else:
                    _logger.info(
                        f"✅ Matrix Cell {self.config.data.matrix_cell}: Using radius_km={radius_km} for spatial aggregation"
                    )
        
        # Log feature_engineering config for debugging
        fe_config = self.config.data.feature_engineering
        if isinstance(fe_config, dict):
            spatial_config = fe_config.get("spatial", {})
            if spatial_config:
                _logger.debug(f"Feature engineering config includes spatial: {spatial_config}")
            if "spatial_aggregation" in fe_config:
                _logger.debug(f"Feature engineering config includes spatial_aggregation: {fe_config.get('spatial_aggregation')}")

        pipeline = DataPipeline(pipeline_config)
        bundle = pipeline.run(
            data_path=self.config.data.source,
            horizons=self.config.labels.horizons,
            use_feature_engineering=None,
            feature_config=self.config.data.feature_engineering,
            sample_size=self.config.data.sample_size,
        )
        df = bundle.data
        
        # DEBUG: Verify neighbor features in DataFrame
        neighbor_cols_in_bundle = [c for c in df.columns if 'neighbor' in c.lower()]
        _logger.info(
            f"🔍 DEBUG: DataPipeline.run() returned DataFrame with {len(df.columns)} columns, "
            f"{len(neighbor_cols_in_bundle)} neighbor columns"
        )
        if len(neighbor_cols_in_bundle) > 0:
            _logger.info(f"🔍 DEBUG: Neighbor feature examples: {neighbor_cols_in_bundle[:5]}")
        else:
            _logger.warning(
                f"⚠️  DEBUG WARNING: No neighbor features in DataFrame returned by DataPipeline.run()! "
                f"This may indicate spatial aggregation did not execute or failed."
            )
        
        # Input validation: Check if DataFrame is empty
        if df.empty:
            raise ValueError(f"DataPipeline returned empty DataFrame. Check data source: {self.config.data.source}")
        
        # Strict column validation: Check for required columns
        required_cols = {DATE_COL, STATION_ID_COL}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise KeyError(
                f"DataPipeline output missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)[:20]}..."  # Show first 20 columns
            )
        
        # Check for temperature column (needed for label generation)
        if TEMP_COL not in df.columns:
            _logger.warning(
                f"Temperature column '{TEMP_COL}' not found. "
                f"Label generation may fail. Available columns: {list(df.columns)[:20]}..."
            )
        
        # Validate that we have feature columns (non-id, non-date columns)
        feature_cols = [col for col in df.columns 
                       if col not in [DATE_COL, STATION_ID_COL, TEMP_COL] 
                       and not col.startswith('frost_') and not col.startswith('temp_')]
        if len(feature_cols) == 0:
            _logger.warning(
                "No feature columns found in DataFrame. "
                "This may indicate an issue with feature engineering or data loading."
            )

        metadata_path = output_dir / "data_run_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(bundle.run_metadata, f, indent=2)
        _logger.info(f"Recorded data pipeline metadata at {metadata_path}")

        labeled_path = output_dir / "labeled_data.parquet"
        ensure_dir(labeled_path.parent)
        df.to_parquet(labeled_path)
        _logger.info(f"Saved labeled data to {labeled_path}")
        
        # Log data loading information
        write_experiment_log("\n[Data Loading]")
        write_experiment_log(f"  ✅ Data loaded successfully")
        write_experiment_log(f"  📊 Total samples: {len(df):,}")
        write_experiment_log(f"  📊 Total features: {len(feature_cols)}")
        write_experiment_log(f"  📊 Feature list: {', '.join(feature_cols[:50])}" + (f" ... (+{len(feature_cols)-50} more)" if len(feature_cols) > 50 else ""))
        if len(feature_cols) > 50:
            # Log remaining features on additional lines
            for i in range(50, len(feature_cols), 50):
                remaining = feature_cols[i:i+50]
                write_experiment_log(f"                    {', '.join(remaining)}")
        write_experiment_log(f"  📊 Stations: {df[STATION_ID_COL].nunique() if STATION_ID_COL in df.columns else 'N/A'}")
        write_experiment_log(f"  📊 Date range: {df[DATE_COL].min() if DATE_COL in df.columns else 'N/A'} to {df[DATE_COL].max() if DATE_COL in df.columns else 'N/A'}")
        write_experiment_log(f"  📊 Labeled data saved to: {labeled_path}")
        
        # Log label statistics for each horizon
        write_experiment_log("\n[Label Statistics]")
        for h in [3, 6, 12, 24]:
            frost_col = f"frost_{h}h"
            if frost_col in df.columns:
                frost_count = (df[frost_col] == 1).sum()
                frost_rate = frost_count / len(df[df[frost_col].notna()]) * 100 if len(df[df[frost_col].notna()]) > 0 else 0
                write_experiment_log(f"  {h}h: {frost_count:,} frost events ({frost_rate:.2f}%)")

        feature_selection = self._load_feature_selection()

        results = {}
        # CRITICAL FIX: Verify all requested horizons have labels
        # Labels are generated for all horizons [3, 6, 12, 24] in pipeline.run() above
        # But we only train the requested horizon(s) from config
        required_horizons = set(self.config.labels.horizons)
        available_horizons = set()
        for h in [3, 6, 12, 24]:
            if f"frost_{h}h" in df.columns:
                available_horizons.add(h)
        
        missing_horizons = required_horizons - available_horizons
        if missing_horizons:
            _logger.warning(
                f"Missing labels for horizons: {sorted(missing_horizons)}. "
                f"Available horizons: {sorted(available_horizons)}. "
                f"Will only train horizons with available labels."
            )
        
        # Train only horizons that have labels
        training_horizons = [h for h in self.config.labels.horizons if h in available_horizons]
        if not training_horizons:
            raise ValueError(
                f"None of requested horizons {sorted(required_horizons)} have labels. "
                f"Available horizons: {sorted(available_horizons)}. "
                f"Please regenerate labels for all requested horizons."
            )
        
        write_experiment_log("\n[Training]")
        for horizon in training_horizons:
            horizon_start_time = time.time()
            write_experiment_log(f"\n  Training horizon: {horizon}h")
            
            result = train_models_for_horizon(
                df,
                horizon,
                output_dir,
                model_type=self.config.training.model,
                skip_if_exists=True,
                feature_selection=feature_selection,
                track=self.track,
                matrix_cell=self.config.data.matrix_cell,
                experiment_log_file=experiment_log_file,  # Pass log file for logging
            )
            results[horizon] = result
            
            horizon_elapsed = time.time() - horizon_start_time
            
            # Log training results
            if result:
                frost_metrics = result.get("frost_metrics", {})
                temp_metrics = result.get("temp_metrics", {})
                
                write_experiment_log(f"    ✅ Training completed in {horizon_elapsed:.2f} seconds ({horizon_elapsed/60:.2f} minutes)")
                write_experiment_log(f"    📊 Frost Metrics:")
                write_experiment_log(f"       ROC-AUC: {frost_metrics.get('roc_auc', 'N/A'):.4f}" if isinstance(frost_metrics.get('roc_auc'), (int, float)) else f"       ROC-AUC: {frost_metrics.get('roc_auc', 'N/A')}")
                write_experiment_log(f"       PR-AUC: {frost_metrics.get('pr_auc', 'N/A'):.4f}" if isinstance(frost_metrics.get('pr_auc'), (int, float)) else f"       PR-AUC: {frost_metrics.get('pr_auc', 'N/A')}")
                write_experiment_log(f"       Brier Score: {frost_metrics.get('brier_score', 'N/A'):.4f}" if isinstance(frost_metrics.get('brier_score'), (int, float)) else f"       Brier Score: {frost_metrics.get('brier_score', 'N/A')}")
                write_experiment_log(f"    📊 Temp Metrics:")
                write_experiment_log(f"       MAE: {temp_metrics.get('mae', 'N/A'):.2f}°C" if isinstance(temp_metrics.get('mae'), (int, float)) else f"       MAE: {temp_metrics.get('mae', 'N/A')}")
                write_experiment_log(f"       RMSE: {temp_metrics.get('rmse', 'N/A'):.2f}°C" if isinstance(temp_metrics.get('rmse'), (int, float)) else f"       RMSE: {temp_metrics.get('rmse', 'N/A')}")
                write_experiment_log(f"       R²: {temp_metrics.get('r2', 'N/A'):.4f}" if isinstance(temp_metrics.get('r2'), (int, float)) else f"       R²: {temp_metrics.get('r2', 'N/A')}")
                
                out_parts = set(output_dir.parts)
                base_dir = output_dir if self.config.data.matrix_cell in out_parts else (output_dir / self.config.data.matrix_cell)
                horizon_model_dir = base_dir / "full_training" / f"horizon_{horizon}h"
                write_experiment_log(f"    📁 Model saved to: {horizon_model_dir}")
            
            # Create and save ExperimentMetadata for each horizon (required)
            # Models are saved in: base_dir/full_training/horizon_{horizon}h/
            # where base_dir matches train_models_for_horizon logic
            # train_models_for_horizon uses: base_dir = output_dir if matrix_cell in output_dir.parts else (output_dir / matrix_cell)
            out_parts = set(output_dir.parts)
            base_dir = output_dir if self.config.data.matrix_cell in out_parts else (output_dir / self.config.data.matrix_cell)
            horizon_model_dir = base_dir / "full_training" / f"horizon_{horizon}h"
            
            # Always save metadata (required by hard constraint), even if training was skipped
            # The directory may not exist yet if training was skipped, so create it
            horizon_model_dir.mkdir(parents=True, exist_ok=True)
            
            metadata = ExperimentMetadata(
                matrix_cell=self.config.data.matrix_cell,
                track=self.track,
                model_name=self.config.training.model,
                horizon_h=horizon,
                radius_km=None,  # Will be set if C/D track or graph model with radius
                knn_k=None,  # Will be set if E track or graph model with knn
                training_scope="full_training"
            )
            
            # Extract spatial parameters from config if available
            # Priority: model config (for graph models) > feature_engineering config
            
            # Check if model is a graph model and extract graph_type/graph_param
            model_name = self.config.training.model
            graph_models = ["dcrnn", "gat_lstm", "graphwavenet", "st_gcn"]
            if model_name in graph_models:
                # Graph models: get graph_type and graph_param from model config
                from src.training.model_config import get_model_config
                model_config = get_model_config(model_name, horizon=horizon)
                model_params = model_config.get("model_params", {})
                graph_type = model_params.get("graph_type", None)
                graph_param = model_params.get("graph_param", None)
                
                if graph_type == "radius" and graph_param is not None:
                    metadata.radius_km = float(graph_param)
                    if self.progress_logger:
                        self.progress_logger.log(
                            f"  ✅ Graph model ({model_name}): setting radius_km={metadata.radius_km} from model config",
                            flush=True,
                            detailed=True
                        )
                elif graph_type == "knn" and graph_param is not None:
                    metadata.knn_k = int(graph_param)
                    if self.progress_logger:
                        self.progress_logger.log(
                            f"  ✅ Graph model ({model_name}): setting knn_k={metadata.knn_k} from model config",
                            flush=True,
                            detailed=True
                        )
            
            # Fallback: Try to get from feature_engineering config (for non-graph models or if model config didn't provide)
            if metadata.radius_km is None and self.config.data.matrix_cell in ["C", "D"]:
                fe_config = self.config.data.feature_engineering
                if isinstance(fe_config, dict):
                    spatial_config = fe_config.get("spatial", {})
                    if isinstance(spatial_config, dict):
                        metadata.radius_km = spatial_config.get("radius_km")
            if metadata.knn_k is None and self.config.data.matrix_cell == "E":
                fe_config = self.config.data.feature_engineering
                if isinstance(fe_config, dict):
                    spatial_config = fe_config.get("spatial", {})
                    if isinstance(spatial_config, dict):
                        metadata.knn_k = spatial_config.get("knn_k")
            
            # Always save metadata (hard constraint: all experiments must write run_metadata.json)
            metadata.save(horizon_model_dir)
            _logger.info(f"Saved experiment metadata to {horizon_model_dir / 'run_metadata.json'}")
            
            # Save pipeline config for inference (critical for Matrix Cell C with spatial aggregation)
            # Transform config to match what DataPipeline actually uses (map spatial.radius_km to spatial_aggregation)
            pipeline_config_to_save = pipeline_config.copy()
            fe_config = pipeline_config_to_save.get("feature_engineering", {})
            if "spatial" in fe_config and "spatial_aggregation" not in fe_config:
                # Map spatial.radius_km to spatial_aggregation.distance_threshold_km
                spatial_cfg = fe_config.get("spatial", {})
                if spatial_cfg:
                    spatial_agg_cfg = {"enabled": spatial_cfg.get("enabled", True)}
                    if "radius_km" in spatial_cfg:
                        spatial_agg_cfg["distance_threshold_km"] = spatial_cfg["radius_km"]
                    if "distance_threshold_km" in spatial_cfg:
                        spatial_agg_cfg["distance_threshold_km"] = spatial_cfg["distance_threshold_km"]
                    if spatial_cfg.get("k_neighbors"):
                        spatial_agg_cfg["k_neighbors"] = spatial_cfg["k_neighbors"]
                    if spatial_cfg.get("weight_method"):
                        spatial_agg_cfg["weight_method"] = spatial_cfg["weight_method"]
                    if spatial_cfg.get("aggregation_methods"):
                        spatial_agg_cfg["aggregation_methods"] = spatial_cfg["aggregation_methods"]
                    if spatial_cfg.get("metadata_path"):
                        spatial_agg_cfg["metadata_path"] = spatial_cfg["metadata_path"]
                    fe_config["spatial_aggregation"] = spatial_agg_cfg
                    # Keep spatial config for backward compatibility, but mark as mapped
                    fe_config["_spatial_mapped_to_spatial_aggregation"] = True
            
            pipeline_config_path = horizon_model_dir / "pipeline_config.yaml"
            ensure_dir(pipeline_config_path.parent)
            import yaml
            with pipeline_config_path.open("w", encoding="utf-8") as f:
                yaml.dump(pipeline_config_to_save, f, default_flow_style=False, allow_unicode=True)
            _logger.info(f"Saved pipeline config to {pipeline_config_path}")

        summary = self._build_summary(results)
        summary_path = output_dir / "full_training" / "summary.json"
        ensure_dir(summary_path.parent)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        _logger.info(f"Summary saved to {summary_path}")

        # Check if LOSO evaluation task is enabled
        loso_task = next((t for t in self.config.evaluation.tasks if t.type == "loso"), None)
        if loso_task and loso_task.params.get("enabled", True):
            write_experiment_log("\n[LOSO Evaluation]")
            write_experiment_log("  Starting LOSO cross-validation...")
            loso_start_time = time.time()
            loso_results = self._run_loso(df, output_dir, feature_selection, loso_task.params)
            loso_elapsed = time.time() - loso_start_time
            write_experiment_log(f"  ✅ LOSO evaluation completed in {loso_elapsed:.2f} seconds ({loso_elapsed/60:.2f} minutes)")
            
            # Log detailed LOSO results: per-station metrics and summary statistics
            _log_loso_results(write_experiment_log, loso_results, self.config.labels.horizons, output_dir)

        # Log experiment completion
        experiment_end_time = datetime.now()
        total_elapsed = (experiment_end_time - experiment_start_time).total_seconds()
        write_experiment_log(f"\n[Experiment Complete]")
        write_experiment_log(f"  End time: {experiment_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        write_experiment_log(f"  Total duration: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
        write_experiment_log(f"  Summary saved to: {summary_path}")

        return 0

    def _load_feature_selection(self) -> Optional[Dict[str, Any]]:
        if self.config.training.feature_selection_config:
            with self.config.training.feature_selection_config.open("r", encoding="utf-8") as f:
                return json.load(f)
        if self.config.training.top_k:
            importance_path = self.config.training.feature_selection or (
                self.project_root / "experiments" / "lightgbm" / "feature_importance" / "feature_importance_3h_all.csv"
            )
            if importance_path.exists():
                return {
                    "enabled": True,
                    "method": "importance",
                    "top_k": self.config.training.top_k,
                    "importance_path": str(importance_path),
                    "save_report": True,
                }
            _logger.warning(f"Importance file not found: {importance_path}")
        if self.config.training.feature_selection and self.config.training.feature_selection.exists():
            with self.config.training.feature_selection.open("r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _build_summary(self, results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        now = datetime.now()
        summary = {
            "model_type": self.config.training.model,
            "horizons": self.config.labels.horizons,
            "timestamp": now.isoformat(),
            "results": {},
        }
        for horizon, result in results.items():
            summary["results"][f"{horizon}h"] = {
                "frost_metrics": result.get("frost_metrics", {}),
                "temp_metrics": result.get("temp_metrics", {}),
            }
        return summary

    def _run_loso(self, df: pd.DataFrame, output_dir: Path, feature_selection: Optional[Dict[str, Any]], loso_params: Dict[str, Any]):
        labeled_path = output_dir / "labeled_data.parquet"
        if not labeled_path.exists():
            df.to_parquet(labeled_path)

        loso_results = perform_loso_evaluation(
            labeled_path,
            self.config.labels.horizons,
            output_dir,
            model_type=self.config.training.model,
            frost_threshold=self.config.labels.frost_threshold,
            resume=loso_params.get("resume", False),
            feature_selection=feature_selection,
            save_models=loso_params.get("save_models", False),
            save_worst_n=loso_params.get("save_worst_n"),
            save_horizons=loso_params.get("save_horizons"),
        )

        loso_dir = output_dir / "loso"
        ensure_dir(loso_dir)
        with (loso_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(loso_results.get("summary", {}), f, indent=2, default=str)
        _logger.info(f"LOSO summary saved to {loso_dir / 'summary.json'}")
        
        return loso_results


class EvaluationRunner:
    """Execute evaluation tasks defined in configuration."""

    def __init__(self, config: PipelineTrainingConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.track = _infer_track(self.config.data.matrix_cell)
        self._dataset_cache: Optional[Tuple[pd.DataFrame, Dict[str, Any]]] = None

    def run(self) -> int:
        if not self.config.evaluation.tasks:
            _logger.warning("No evaluation tasks configured.")
            return 0
        dataset = self._get_dataset()
        for task in self.config.evaluation.tasks:
            handler = get_evaluation_handler(task.type)
            handler(self, dataset, task.params)
        return 0

    def _get_dataset(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self._dataset_cache is None:
            pipeline = DataPipeline(_build_pipeline_config(self.config.data, self.config.labels))
            bundle = pipeline.run(
                data_path=self.config.data.source,
                horizons=self.config.labels.horizons,
                use_feature_engineering=None,
                feature_config=self.config.data.feature_engineering,
                sample_size=self.config.data.sample_size,
            )
            df = bundle.data
            
            # Input validation
            if df.empty:
                raise ValueError(f"DataPipeline returned empty DataFrame. Check data source: {self.config.data.source}")
            
            self._dataset_cache = (df, bundle.run_metadata)
        return self._dataset_cache

    # Evaluation task handlers -------------------------------------------------

    def handle_loso(self, dataset: Tuple[pd.DataFrame, Dict[str, Any]], params: Dict[str, Any]) -> None:
        df, _ = dataset
        output_dir = params.get("output_dir")
        if output_dir:
            output_dir = _resolve_path(output_dir, self.project_root)
        elif self.config.training:
            output_dir = self.config.training.output_dir
        else:
            output_dir = self.project_root / "experiments" / "evaluation"
        ensure_dir(output_dir)

        labeled_path = output_dir / "evaluation_labeled.parquet"
        df.to_parquet(labeled_path)

        loso_params = {
            "save_models": params.get("save_models", False),
            "resume": params.get("resume", False),
            "save_worst_n": params.get("save_worst_n"),
            "save_horizons": params.get("save_horizons"),
        }

        perform_loso_evaluation(
            labeled_path,
            params.get("horizons", self.config.labels.horizons),
            output_dir,
            model_type=params.get("model_type") or (self.config.training.model if self.config.training else "lightgbm"),
            frost_threshold=self.config.labels.frost_threshold,
            resume=loso_params["resume"],
            feature_selection=None,
            save_models=loso_params["save_models"],
            save_worst_n=loso_params["save_worst_n"],
            save_horizons=loso_params["save_horizons"],
        )

    def handle_direct(self, dataset: Tuple[pd.DataFrame, Dict[str, Any]], params: Dict[str, Any]) -> None:
        df, metadata = dataset
        horizons = params.get("horizons", self.config.labels.horizons)
        track = params.get("track", self.track)

        if params.get("model_root"):
            model_root = _resolve_path(params["model_root"], self.project_root)
        elif self.config.training:
            base_dir = self.config.training.output_dir
            matrix_cell = self.config.data.matrix_cell
            model_root = base_dir if matrix_cell in base_dir.parts else base_dir / matrix_cell
            model_root = model_root / "full_training"
        else:
            raise ValueError("model_root must be provided for direct evaluation without training section.")

        output_dir = params.get("output_dir")
        if output_dir:
            output_dir = _resolve_path(output_dir, self.project_root)
        elif self.config.training:
            output_dir = self.config.training.output_dir / "evaluation"
        else:
            output_dir = self.project_root / "experiments" / "evaluation"
        ensure_dir(output_dir)

        base_model_type = params.get("model_type") or (self.config.training.model if self.config.training else None)

        for horizon in horizons:
            horizon_dir = model_root / f"horizon_{horizon}h"
            frost_model_path = horizon_dir / "frost_classifier"
            temp_model_path = horizon_dir / "temp_regressor"
            if not frost_model_path.exists() or not temp_model_path.exists():
                _logger.warning(f"Missing models for {horizon}h horizon under {horizon_dir}, skipping.")
                continue

            X, y_frost, y_temp = prepare_features_and_targets(
                df,
                horizon,
                track=track,
                model_type=base_model_type,
            )

            frost_model, resolved_type = _load_model_from_dir(frost_model_path, base_model_type)
            temp_model, _ = _load_model_from_dir(temp_model_path, resolved_type)

            frost_pred = frost_model.predict(X)
            frost_proba = frost_model.predict_proba(X) if hasattr(frost_model, "predict_proba") else None
            temp_pred = temp_model.predict(X)

            frost_metrics = MetricsCalculator.calculate_all_metrics(
                y_frost.values,
                frost_pred,
                frost_proba,
                task_type="classification",
            )
            temp_metrics = MetricsCalculator.calculate_all_metrics(
                y_temp.values,
                temp_pred,
                None,
                task_type="regression",
            )

            save_dir = output_dir / f"horizon_{horizon}h"
            ensure_dir(save_dir)
            with (save_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "frost_metrics": frost_metrics,
                        "temp_metrics": temp_metrics,
                        "n_samples": len(X),
                        "model_type": resolved_type,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            
            # Create and save ExperimentMetadata (required - replaces old dict-based metadata)
            # Try to load existing metadata from model directory
            try:
                existing_metadata_path = horizon_dir / "run_metadata.json"
                if existing_metadata_path.exists():
                    # Load existing metadata to preserve spatial parameters
                    experiment_metadata = ExperimentMetadata.load(existing_metadata_path)
                    # Update with evaluation-specific info
                    experiment_metadata.horizon_h = horizon
                else:
                    # Create new metadata if not found (legacy case)
                    experiment_metadata = ExperimentMetadata(
                        matrix_cell=self.config.data.matrix_cell,
                        track=track,
                        model_name=resolved_type or "unknown",
                        horizon_h=horizon,
                        radius_km=None,
                        knn_k=None,
                        training_scope="evaluation"
                    )
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                # Fallback: create new metadata if loading fails
                _logger.warning(f"Could not load existing metadata from {horizon_dir}: {e}. Creating new metadata.")
                experiment_metadata = ExperimentMetadata(
                    matrix_cell=self.config.data.matrix_cell,
                    track=track,
                    model_name=resolved_type or "unknown",
                    horizon_h=horizon,
                    radius_km=None,
                    knn_k=None,
                    training_scope="evaluation"
                )
            
            # Save metadata to evaluation output directory
            experiment_metadata.save(save_dir)
            _logger.info(f"Saved experiment metadata to {save_dir / 'run_metadata.json'}")


register_evaluation_strategy("loso", lambda runner, dataset, params: runner.handle_loso(dataset, params))
register_evaluation_strategy("direct", lambda runner, dataset, params: runner.handle_direct(dataset, params))


def _log_loso_results(write_experiment_log, loso_results: Dict[str, Any], horizons: List[int], output_dir: Path):
    """Log detailed LOSO evaluation results to experiment log.
    
    This function logs:
    - Summary statistics (mean ± SD) across all stations for each horizon
    - Per-station metrics (first 10 stations to avoid too much output)
    
    Args:
        write_experiment_log: Function to write to experiment log.
        loso_results: LOSO evaluation results dictionary.
        horizons: List of forecast horizons.
        output_dir: Output directory path.
    """
    summary = loso_results.get("summary", {})
    station_metrics = loso_results.get("station_metrics", [])
    n_stations = loso_results.get("n_stations", len(station_metrics))
    
    write_experiment_log(f"\n  📊 LOSO Results Summary (across {n_stations} stations):")
    
    for horizon in horizons:
        horizon_key = f"{horizon}h"
        if horizon_key not in summary:
            continue
        
        h_summary = summary[horizon_key]
        frost_summary = h_summary.get("frost_metrics", {})
        temp_summary = h_summary.get("temp_metrics", {})
        
        write_experiment_log(f"\n    Horizon {horizon}h:")
        write_experiment_log(f"      Frost Metrics:")
        
        # Brier Score
        if "brier_score" in frost_summary:
            bs_mean = frost_summary["brier_score"].get("mean", np.nan)
            bs_std = frost_summary["brier_score"].get("std", np.nan)
            if not np.isnan(bs_mean):
                write_experiment_log(f"        Brier Score: {bs_mean:.4f} ± {bs_std:.4f}")
        
        # ECE
        if "ece" in frost_summary:
            ece_mean = frost_summary["ece"].get("mean", np.nan)
            ece_std = frost_summary["ece"].get("std", np.nan)
            if not np.isnan(ece_mean):
                write_experiment_log(f"        Expected Calibration Error (ECE): {ece_mean:.4f} ± {ece_std:.4f}")
        
        # ROC-AUC
        if "roc_auc" in frost_summary:
            roc_mean = frost_summary["roc_auc"].get("mean", np.nan)
            roc_std = frost_summary["roc_auc"].get("std", np.nan)
            if not np.isnan(roc_mean):
                write_experiment_log(f"        ROC-AUC (discrimination): {roc_mean:.4f} ± {roc_std:.4f}")
        
        # PR-AUC
        if "pr_auc" in frost_summary:
            pr_mean = frost_summary["pr_auc"].get("mean", np.nan)
            pr_std = frost_summary["pr_auc"].get("std", np.nan)
            if not np.isnan(pr_mean):
                write_experiment_log(f"        PR-AUC (discrimination): {pr_mean:.4f} ± {pr_std:.4f}")
        
        write_experiment_log(f"      Temp Metrics:")
        if "mae" in temp_summary:
            mae_mean = temp_summary["mae"].get("mean", np.nan)
            mae_std = temp_summary["mae"].get("std", np.nan)
            if not np.isnan(mae_mean):
                write_experiment_log(f"        MAE: {mae_mean:.2f} ± {mae_std:.2f}°C")
        if "rmse" in temp_summary:
            rmse_mean = temp_summary["rmse"].get("mean", np.nan)
            rmse_std = temp_summary["rmse"].get("std", np.nan)
            if not np.isnan(rmse_mean):
                write_experiment_log(f"        RMSE: {rmse_mean:.2f} ± {rmse_std:.2f}°C")
        if "r2" in temp_summary:
            r2_mean = temp_summary["r2"].get("mean", np.nan)
            r2_std = temp_summary["r2"].get("std", np.nan)
            if not np.isnan(r2_mean):
                write_experiment_log(f"        R²: {r2_mean:.4f} ± {r2_std:.4f}")
    
    # Log per-station metrics for detailed analysis
    write_experiment_log(f"\n  📊 Per-Station Metrics (first 10 stations):")
    for station_result in station_metrics[:10]:  # Log first 10 stations to avoid too much output
        station_id = station_result.get("station_id", "N/A")
        write_experiment_log(f"\n    Station {station_id}:")
        for horizon in horizons:
            horizon_key = f"{horizon}h"
            if horizon_key not in station_result.get("horizons", {}):
                continue
            h_metrics = station_result["horizons"][horizon_key]
            frost_metrics = h_metrics.get("frost_metrics", {})
            temp_metrics = h_metrics.get("temp_metrics", {})
            
            write_experiment_log(f"      {horizon}h:")
            if "brier_score" in frost_metrics and not np.isnan(frost_metrics.get("brier_score", np.nan)):
                write_experiment_log(f"        Brier: {frost_metrics['brier_score']:.4f}, ECE: {frost_metrics.get('ece', np.nan):.4f}, ROC-AUC: {frost_metrics.get('roc_auc', np.nan):.4f}, PR-AUC: {frost_metrics.get('pr_auc', np.nan):.4f}")
            if "mae" in temp_metrics and not np.isnan(temp_metrics.get("mae", np.nan)):
                write_experiment_log(f"        MAE: {temp_metrics['mae']:.2f}°C, RMSE: {temp_metrics.get('rmse', np.nan):.2f}°C, R²: {temp_metrics.get('r2', np.nan):.4f}")
    
    if len(station_metrics) > 10:
        write_experiment_log(f"    ... (and {len(station_metrics) - 10} more stations, see station_metrics.csv for details)")
    
    write_experiment_log(f"\n  📁 Detailed metrics saved to: {output_dir / 'loso' / 'station_metrics.csv'}")


class InferenceRunner:
    """Run inference using pipeline configuration."""

    def __init__(self, config: PipelineTrainingConfig, project_root: Path):
        if config.inference is None:
            raise ValueError("Inference configuration required.")
        self.config = config
        self.project_root = project_root
        self.track = _infer_track(self.config.data.matrix_cell)
        self._dataset_cache: Optional[Tuple[pd.DataFrame, Dict[str, Any]]] = None

    def run(self) -> int:
        df, metadata = self._get_dataset()
        horizons = self.config.inference.horizons or self.config.labels.horizons
        results = []

        for horizon in horizons:
            X, _, _ = prepare_features_and_targets(
                df,
                horizon,
                track=self.track,
                model_type=self.config.training.model if self.config.training else None,
            )
            frost_model_path = self.config.inference.model_dir / f"horizon_{horizon}h" / "frost_classifier"
            temp_model_path = self.config.inference.model_dir / f"horizon_{horizon}h" / "temp_regressor"
            if not frost_model_path.exists() or not temp_model_path.exists():
                _logger.warning(f"Missing models for {horizon}h horizon under {self.config.inference.model_dir}, skipping.")
                continue
            frost_model, base_type = _load_model_from_dir(
                frost_model_path,
                self.config.inference.model_type or (self.config.training.model if self.config.training else None),
            )
            temp_model, _ = _load_model_from_dir(temp_model_path, base_type)

            frost_proba = frost_model.predict_proba(X) if hasattr(frost_model, "predict_proba") else frost_model.predict(X)
            temp_pred = temp_model.predict(X)

            frost_series = pd.Series(frost_proba).reset_index(drop=True)
            temp_series = pd.Series(temp_pred).reset_index(drop=True)

            horizon_result = pd.DataFrame({
                "horizon": horizon,
                "frost_probability": frost_series,
                "predicted_temperature": temp_series,
            })
            results.append(horizon_result)

        if not results:
            _logger.error("No inference outputs generated.")
            return 1

        predictions = pd.concat(results, ignore_index=True)
        ensure_dir(self.config.inference.output_dir)
        predictions_path = self.config.inference.output_dir / "predictions.parquet"
        predictions.to_parquet(predictions_path)
        with (self.config.inference.output_dir / "data_run_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        _logger.info(f"Inference results saved to {predictions_path}")
        return 0

    def _get_dataset(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self._dataset_cache is None:
            pipeline = DataPipeline(_build_pipeline_config(self.config.data, self.config.labels))
            bundle = pipeline.run(
                data_path=self.config.data.source,
                horizons=self.config.labels.horizons,
                use_feature_engineering=None,
                feature_config=self.config.data.feature_engineering,
                sample_size=self.config.data.sample_size,
            )
            df = bundle.data
            
            # Input validation
            if df.empty:
                raise ValueError(f"DataPipeline returned empty DataFrame. Check data source: {self.config.data.source}")
            
            self._dataset_cache = (df, bundle.run_metadata)
        return self._dataset_cache

