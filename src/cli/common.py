"""Common utilities for CLI commands."""

import warnings
from pathlib import Path
from typing import Optional, Dict
import json
import yaml


def warn_legacy_script(script_name: str, replacement: str):
    """Print unified DeprecationWarning for legacy scripts.
    
    Args:
        script_name: Old script name (e.g., "train_frost_forecast.py")
        replacement: New CLI command (e.g., "python -m src.cli train single ...")
    """
    warnings.warn(
        f"{script_name} is deprecated. "
        f"Use '{replacement}' instead. "
        f"This script will be removed in v0.3.",
        DeprecationWarning,
        stacklevel=3  # Point to caller, not inside warn_legacy_script
    )


def load_and_merge_config(
    config_path: Optional[Path],
    cli_overrides: Dict
) -> Dict:
    """Load config file and merge CLI overrides.
    
    Priority: CLI arguments > Config file > Default values
    
    Args:
        config_path: Path to config file (YAML or JSON)
        cli_overrides: CLI argument overrides
    
    Returns:
        Merged config dictionary
    
    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If YAML format is invalid
        json.JSONDecodeError: If JSON format is invalid
    """
    config = {}
    
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f) or {}
            elif config_path.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # CLI arguments override config
    config.update(cli_overrides)
    
    return config

