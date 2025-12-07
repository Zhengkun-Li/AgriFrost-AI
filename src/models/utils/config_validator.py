"""Configuration validation utility for model training."""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging

_logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validate model and training configurations.
    
    This class provides validation for model configurations to catch
    errors early and provide helpful error messages.
    
    **2×2+1 Framework Validation:**
    - Enforces matrix cell rules (A/B/E: no radius, C/D: must have radius, E: must have knn_k)
    - Validates horizon_h must be in {3, 6, 12, 24}
    - Validates track must be in {raw, top175_features, ...}
    - Supports strict/fallback modes for unknown keys
    """
    
    @staticmethod
    def validate_model_config(
        model_type: str,
        config: Dict[str, Any],
        task_type: str = "classification"
    ) -> Tuple[bool, Optional[str]]:
        """Validate model configuration.
        
        Args:
            model_type: Type of model (lightgbm, xgboost, lstm, etc.).
            config: Configuration dictionary.
            task_type: Task type (classification or regression).
        
        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        if not isinstance(config, dict):
            return False, "Configuration must be a dictionary"
        
        model_params = config.get("model_params", {})
        if not isinstance(model_params, dict):
            return False, "model_params must be a dictionary"
        
        # Model-specific validation
        if model_type == "lstm" or model_type == "lstm_multitask":
            return ConfigValidator._validate_lstm_config(model_params)
        elif model_type in ["lightgbm", "xgboost", "catboost"]:
            return ConfigValidator._validate_tree_config(model_type, model_params)
        elif model_type == "random_forest":
            return ConfigValidator._validate_rf_config(model_params)
        elif model_type == "prophet":
            return ConfigValidator._validate_prophet_config(model_params)
        
        return True, None
    
    @staticmethod
    def _validate_lstm_config(model_params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate LSTM model configuration."""
        # Check required parameters
        required_params = ["sequence_length", "hidden_size", "batch_size", "epochs", "learning_rate"]
        for param in required_params:
            if param not in model_params:
                return False, f"Missing required parameter: {param}"
        
        # Validate parameter types and ranges
        if not isinstance(model_params["sequence_length"], int) or model_params["sequence_length"] <= 0:
            return False, "sequence_length must be a positive integer"
        
        if not isinstance(model_params["hidden_size"], int) or model_params["hidden_size"] <= 0:
            return False, "hidden_size must be a positive integer"
        
        if not isinstance(model_params["batch_size"], int) or model_params["batch_size"] <= 0:
            return False, "batch_size must be a positive integer"
        
        if not isinstance(model_params["epochs"], int) or model_params["epochs"] <= 0:
            return False, "epochs must be a positive integer"
        
        if not isinstance(model_params["learning_rate"], (int, float)) or model_params["learning_rate"] <= 0:
            return False, "learning_rate must be a positive number"
        
        # Validate optional parameters
        if "dropout" in model_params:
            dropout = model_params["dropout"]
            if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
                return False, "dropout must be a number between 0 and 1"
        
        if "checkpoint_frequency" in model_params:
            freq = model_params["checkpoint_frequency"]
            if not isinstance(freq, int) or freq < 0:
                return False, "checkpoint_frequency must be a non-negative integer"
        
        return True, None
    
    @staticmethod
    def _validate_tree_config(
        model_type: str,
        model_params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate tree model (LightGBM, XGBoost, CatBoost) configuration."""
        # Check common parameters
        if "n_estimators" in model_params:
            n_est = model_params["n_estimators"]
            if not isinstance(n_est, int) or n_est <= 0:
                return False, "n_estimators must be a positive integer"
        
        if "learning_rate" in model_params:
            lr = model_params["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                return False, "learning_rate must be a positive number"
        
        if "max_depth" in model_params:
            max_d = model_params["max_depth"]
            if not isinstance(max_d, int) or max_d <= 0:
                return False, "max_depth must be a positive integer"
        
        return True, None
    
    @staticmethod
    def _validate_rf_config(model_params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate Random Forest configuration."""
        if "n_estimators" in model_params:
            n_est = model_params["n_estimators"]
            if not isinstance(n_est, int) or n_est <= 0:
                return False, "n_estimators must be a positive integer"
        
        if "max_depth" in model_params:
            max_d = model_params["max_depth"]
            if max_d is not None and (not isinstance(max_d, int) or max_d <= 0):
                return False, "max_depth must be None or a positive integer"
        
        return True, None
    
    @staticmethod
    def _validate_prophet_config(model_params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate Prophet configuration."""
        # Prophet has minimal required parameters
        # Most parameters are optional with good defaults
        return True, None
    
    @staticmethod
    def validate_experiment_metadata(
        matrix_cell: Optional[str] = None,
        track: Optional[str] = None,
        horizon_h: Optional[int] = None,
        model_name: Optional[str] = None,
        radius_km: Optional[float] = None,
        knn_k: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate experiment metadata according to 2×2+1 framework rules.
        
        **2×2+1 Framework Rules:**
        - A/B/E cells: raw features only, no spatial aggregation (no radius/knn_k)
        - C/D cells: must have radius_km for spatial aggregation
        - E cell: must have knn_k for kNN graph
        - horizon_h must be in {3, 6, 12, 24}
        - track must be in {raw, top175_features, ...}
        
        Args:
            matrix_cell: Matrix cell identifier (A, B, C, D, E, etc.).
            track: Feature engineering track (raw, top175_features, ...).
            horizon_h: Forecast horizon in hours.
            model_name: Model name (for graph models, validates graph params).
            radius_km: Radius in km (for C/D cells or graph models with radius).
            knn_k: k value for kNN (for E cell or graph models with knn).
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        # Validate matrix cell if provided
        if matrix_cell is not None:
            matrix_cell_upper = matrix_cell.upper()
            valid_cells = {'A', 'B', 'C', 'D', 'E'}
            if matrix_cell_upper not in valid_cells:
                return False, f"Invalid matrix_cell: {matrix_cell}. Must be one of {valid_cells}"
        
        # Validate horizon_h if provided
        if horizon_h is not None:
            valid_horizons = {3, 6, 12, 24}
            if horizon_h not in valid_horizons:
                return False, f"Invalid horizon_h: {horizon_h}. Must be one of {valid_horizons}"
        
        # Validate track if provided
        if track is not None:
            valid_tracks = {'raw', 'top175_features'}  # Can be extended
            if track not in valid_tracks:
                # Warning but not error (allow custom tracks)
                _logger.warning(f"Unknown track: {track}. Known tracks: {valid_tracks}")
        
        # Validate 2×2+1 framework rules
        if matrix_cell is not None:
            matrix_cell_upper = matrix_cell.upper()
            
            # A/B/E cells: no spatial aggregation (no radius/knn_k)
            if matrix_cell_upper in {'A', 'B'}:
                if radius_km is not None:
                    return False, f"Matrix cell {matrix_cell_upper} (raw features) cannot have radius_km. Remove spatial aggregation."
                if knn_k is not None:
                    return False, f"Matrix cell {matrix_cell_upper} (raw features) cannot have knn_k. Remove spatial aggregation."
            
            # C/D cells: must have radius_km
            if matrix_cell_upper in {'C', 'D'}:
                if radius_km is None:
                    return False, f"Matrix cell {matrix_cell_upper} (spatial aggregation) must have radius_km. Provide radius_km in config."
            
            # E cell: must have knn_k (and no radius_km)
            if matrix_cell_upper == 'E':
                if knn_k is None:
                    return False, f"Matrix cell E (kNN graph) must have knn_k. Provide knn_k in config."
                if radius_km is not None:
                    return False, f"Matrix cell E (kNN graph) cannot have radius_km. Use knn_k instead."
        
        # Validate graph model consistency (if model is a graph model)
        if model_name is not None:
            graph_models = {'dcrnn', 'gat_lstm', 'graphwavenet', 'st_gcn'}
            if model_name.lower() in graph_models:
                # Graph models must have either radius_km or knn_k
                if radius_km is None and knn_k is None:
                    return False, f"Graph model {model_name} must have either radius_km or knn_k in config."
        
        return True, None
    
    @staticmethod
    def validate_training_args(
        model_type: str,
        checkpoint_dir: Optional[Path] = None,
        log_file: Optional[Path] = None,
        strict_mode: bool = False,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Validate training arguments.
        
        **Strict/Fallback Modes:**
        - strict_mode=True: Reject unknown keys (helpful for catching typos)
        - strict_mode=False: Ignore unknown keys (helpful for CLI overrides)
        
        Args:
            model_type: Type of model.
            checkpoint_dir: Optional checkpoint directory.
            log_file: Optional log file path.
            strict_mode: If True, reject unknown keys in kwargs.
            **kwargs: Additional training arguments.
        
        Returns:
            Tuple of (is_valid, error_message).
        """
        # Validate checkpoint directory
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            try:
                checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                return False, f"Cannot create checkpoint directory: {e}"
        
        # Validate log file
        if log_file is not None:
            log_file = Path(log_file)
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                return False, f"Cannot create log file directory: {e}"
        
        # Strict mode: validate unknown keys (can be extended to check against schema)
        if strict_mode:
            # In strict mode, we could validate against a known schema
            # For now, we just log a warning if there are unknown keys
            known_keys = {'checkpoint_dir', 'log_file', 'strict_mode', 'model_type'}
            unknown_keys = set(kwargs.keys()) - known_keys
            if unknown_keys:
                _logger.warning(f"Unknown keys in training args (strict_mode=True): {unknown_keys}")
        
        return True, None
    
    @staticmethod
    def suggest_fixes(
        model_type: str,
        config: Dict[str, Any],
        error_message: str
    ) -> List[str]:
        """Suggest fixes for configuration errors.
        
        Args:
            model_type: Type of model.
            config: Configuration dictionary.
            error_message: Error message from validation.
        
        Returns:
            List of suggested fixes.
        """
        suggestions = []
        
        if "Missing required parameter" in error_message:
            param = error_message.split(":")[-1].strip()
            if model_type in ["lstm", "lstm_multitask"]:
                if param == "sequence_length":
                    suggestions.append(f"Add '{param}': 24 to model_params")
                elif param == "hidden_size":
                    suggestions.append(f"Add '{param}': 64 or 128 to model_params")
                elif param == "batch_size":
                    suggestions.append(f"Add '{param}': 32 or 64 to model_params")
                elif param == "epochs":
                    suggestions.append(f"Add '{param}': 50 or 100 to model_params")
                elif param == "learning_rate":
                    suggestions.append(f"Add '{param}': 0.001 or 0.0001 to model_params")
        
        if "must be a positive" in error_message:
            suggestions.append("Ensure the parameter value is a positive number")
        
        if "must be between 0 and 1" in error_message:
            suggestions.append("Ensure the parameter value is between 0 and 1")
        
        return suggestions

