"""Model configuration module for frost forecasting training.

This module handles:
- Model parameter configuration for different model types
- Model class selection
- Resource-aware configuration adjustment
"""

import os
from typing import Dict, Tuple, Optional, Type
from pathlib import Path

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_resource_aware_config() -> Tuple[int, int]:
    """Get resource-aware configuration based on available memory.
    
    Returns:
        Tuple of (hidden_size, batch_size) for LSTM models.
    """
    if not PSUTIL_AVAILABLE:
        # Default configuration if psutil is not available
        # Prefer larger capacity on modern machines
        return 128, 128
    
    mem_gb = psutil.virtual_memory().total / (1024**3)
    if mem_gb >= 32:
        return 128, 128  # Increased batch size for better GPU utilization
    elif mem_gb >= 16:
        return 128, 128
    else:
        return 64, 64  # Increased minimums


def get_model_params(
    model_type: str,
    task_type: str = "classification",
    max_workers: Optional[int] = None,
    for_loso: bool = False
) -> Dict:
    """Get model parameters for a specific model type and task.
    
    Args:
        model_type: Model type (lightgbm, xgboost, catboost, etc.).
        task_type: Task type (classification or regression).
        max_workers: Maximum number of workers (auto-determined if None).
        for_loso: Whether this is for LOSO evaluation (smaller config).
    
    Returns:
        Dictionary of model parameters.
    """
    # Validate model type
    supported_models = [
        "lightgbm", "xgboost", "catboost", "random_forest",
        "ensemble", "lstm", "lstm_multitask", "prophet",
        "extratrees", "linear_regression", "ridge", "elasticnet", "logreg", "gru", "tcn", "persistence",
        "dcrnn", "st_gcn", "gat_lstm", "graphwavenet"  # Graph neural network models
    ]
    if model_type not in supported_models:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {', '.join(supported_models)}"
        )
    
    if max_workers is None:
        max_workers = min(8, max(1, os.cpu_count() // 4))
    
    # Adjust parameters for LOSO (smaller config to save memory)
    if for_loso:
        n_estimators = 50
        max_depth = 6
        num_leaves = 31
        hidden_size, batch_size = get_resource_aware_config()
        if hidden_size > 64:
            hidden_size = 64
        if batch_size > 32:
            batch_size = 32
        epochs = 50
        patience = 8
    else:
        n_estimators = 200
        max_depth = 8
        num_leaves = 63
        hidden_size, batch_size = get_resource_aware_config()
        epochs = 100
        patience = 10
    
    if model_type == "lightgbm":
        if task_type == "classification":
            return {
                "n_estimators": n_estimators,
                "learning_rate": 0.05,
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": max_workers,
                "force_col_wise": True,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            }
        else:  # regression
            return {
                "n_estimators": n_estimators,
                "learning_rate": 0.05,
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": max_workers,
                "force_col_wise": True,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            }
    
    elif model_type == "xgboost":
        objective = "binary:logistic" if task_type == "classification" else "reg:squarederror"
        return {
            "n_estimators": n_estimators,
            "learning_rate": 0.05,
            "max_depth": max_depth,
            "random_state": 42,
            "n_jobs": max_workers,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "tree_method": "hist",
            "objective": objective,
        }
    
    elif model_type == "catboost":
        return {
            "iterations": n_estimators,
            "learning_rate": 0.05,
            "depth": max_depth,
            "random_state": 42,
            "thread_count": max_workers,
            "subsample": 0.8,
            "colsample_bylevel": 0.8,
            "l2_leaf_reg": 0.1,
            "verbose": False,
        }
    elif model_type == "extratrees":
        return {
            "n_estimators": n_estimators,
            "max_depth": None if not for_loso else max_depth,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": max_workers,
        }
    
    elif model_type == "random_forest":
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": max_workers,
        }
    
    elif model_type == "ensemble":
        base_n_estimators = 150 if not for_loso else 30
        return {
            "lightgbm": {
                "n_estimators": base_n_estimators,
                "learning_rate": 0.05,
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": max_workers,
                "force_col_wise": True,
            },
            "xgboost": {
                "n_estimators": base_n_estimators,
                "learning_rate": 0.05,
                "max_depth": max_depth,
                "random_state": 42,
                "n_jobs": max_workers,
                "tree_method": "hist",
                "objective": "binary:logistic" if task_type == "classification" else "reg:squarederror",
            },
            "catboost": {
                "iterations": base_n_estimators,
                "learning_rate": 0.05,
                "depth": max_depth,
                "random_state": 42,
                "thread_count": max_workers,
                "verbose": False,
            }
        }
    
    elif model_type == "lstm":
        return {
            "sequence_length": 24,
            "hidden_size": hidden_size,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.0003,  # Faster convergence with scaling
            "batch_size": batch_size,
            "epochs": 120,
            "early_stopping": True,
            "patience": 20,  # Increased from 10 to allow more training for small datasets
            "min_delta": 1e-6,
            "lr_scheduler": True,
            "lr_scheduler_patience": 5,
            "lr_scheduler_factor": 0.5,
            "gradient_clip": 1.0,  # Gradient clipping to prevent explosion
            "save_best_model": True,
            "use_amp": True,  # Enable mixed precision training for 1.5-2x speedup
            "use_weighted_sampler": True,
            "val_frequency": 1,  # Validate every epoch for better monitoring and threshold selection
            "checkpoint_frequency": 10,  # Save checkpoint every 10 epochs (0 = disabled)
            # Advanced options for imbalanced data
            "use_focal_loss": False,  # Use Focal Loss instead of BCEWithLogitsLoss (better for extreme imbalance)
            "focal_alpha": 0.25,  # Focal Loss alpha parameter (weight for rare class)
            "focal_gamma": 2.0,  # Focal Loss gamma parameter (focusing parameter, higher = more focus on hard examples)
            "use_class_balanced_batch": False,  # Further boost positive sample sampling (for <5% positive)
            "use_pr_auc_threshold": False,  # Use PR-AUC optimization for threshold selection (better for imbalanced data)
            # Probability calibration options (improves Brier Score and ECE)
            "use_probability_calibration": True,  # Enable probability calibration (Platt scaling or Isotonic regression)
            "calibration_method": "platt",  # "platt" (logistic regression) or "isotonic" (isotonic regression)
        }
    elif model_type == "gru":
        return {
            "sequence_length": 24,
            "hidden_size": hidden_size,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.0001,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": True,
            "patience": patience,
            "min_delta": 1e-6,
            "lr_scheduler": True,
            "lr_scheduler_patience": 5,
            "lr_scheduler_factor": 0.5,
            "save_best_model": True,
            "use_amp": True,
            "val_frequency": 5,
            "checkpoint_frequency": 10,
        }
    elif model_type == "tcn":
        return {
            "sequence_length": 24,
            "num_channels": [32, 32, 32],
            "kernel_size": 3,
            "dropout": 0.1,
            "learning_rate": 0.0005,
            "batch_size": max(32, batch_size),
            "epochs": epochs,
            "early_stopping": True,
            "patience": patience,
            "min_delta": 1e-6,
            "lr_scheduler": True,
            "lr_scheduler_patience": 5,
            "checkpoint_frequency": 10,
            "use_amp": True,
        }
    elif model_type == "dcrnn":
        # Optimized batch size for graph models (larger for better GPU utilization)
        graph_batch_size = min(batch_size * 2, 128) if not for_loso else batch_size
        return {
            "sequence_length": 24,
            "hidden_size": hidden_size,
            "num_layers": 2,
            "num_diffusion_steps": 1,  # Reduced from 2 to 1 for faster training
            "dropout": 0.2,
            "learning_rate": 0.0003,
            "batch_size": graph_batch_size,  # Increased batch size
            "epochs": epochs,
            "early_stopping": True,
            "patience": patience,
            "min_delta": 1e-6,
            "lr_scheduler": True,
            "lr_scheduler_patience": 5,
            "gradient_clip": 1.0,
            "use_amp": True,
            "use_probability_calibration": True,
            "calibration_method": "platt",
            # Graph-specific parameters
            "graph_type": "radius",  # 'radius' or 'knn'
            "graph_param": 50.0,  # Radius in km or k for kNN
            "edge_weight": "gaussian",  # 'gaussian', 'distance', 'binary', 'learnable'
        }
    elif model_type == "st_gcn":
        return {
            "sequence_length": 24,
            "hidden_channels": hidden_size,
            "num_blocks": 2,
            "kernel_size": 3,
            "dropout": 0.2,
            "learning_rate": 0.0003,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": True,
            "patience": patience,
            "min_delta": 1e-6,
            "lr_scheduler": True,
            "lr_scheduler_patience": 5,
            "gradient_clip": 1.0,
            "use_amp": True,
            "use_probability_calibration": True,
            "calibration_method": "platt",
            # Graph-specific parameters
            "graph_type": "radius",
            "graph_param": 50.0,
            "edge_weight": "gaussian",
        }
    elif model_type == "gat_lstm":
        return {
            "sequence_length": 24,
            "hidden_size": hidden_size,
            "num_gat_layers": 2,
            "num_lstm_layers": 2,
            "num_heads": 4,
            "dropout": 0.2,
            "learning_rate": 0.0003,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": True,
            "patience": patience,
            "min_delta": 1e-6,
            "lr_scheduler": True,
            "lr_scheduler_patience": 5,
            "gradient_clip": 1.0,
            "use_amp": True,
            "use_probability_calibration": True,
            "calibration_method": "platt",
            # Graph-specific parameters
            "graph_type": "radius",
            "graph_param": 50.0,
            "edge_weight": "gaussian",
        }
    elif model_type == "graphwavenet":
        return {
            "sequence_length": 24,
            "hidden_channels": hidden_size,
            "num_blocks": 4,
            "kernel_size": 2,
            "dropout": 0.2,
            "learning_rate": 0.0003,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping": True,
            "patience": patience,
            "min_delta": 1e-6,
            "lr_scheduler": True,
            "lr_scheduler_patience": 5,
            "gradient_clip": 1.0,
            "use_amp": True,
            "use_probability_calibration": True,
            "calibration_method": "platt",
            # Graph-specific parameters
            "graph_type": "radius",
            "graph_param": 50.0,
            "edge_weight": "gaussian",
        }
    elif model_type in ["linear_regression", "ridge", "elasticnet", "logreg"]:
        # base params for linear models; classification vs regression handled in class
        base = {
            "n_jobs": max_workers,
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "max_iter": 200
        }
        return base
    
    elif model_type == "lstm_multitask":
        return {
            "sequence_length": 24,
            "hidden_size": hidden_size,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.0003,
            "batch_size": batch_size,
            "epochs": 120,
            "early_stopping": True,
            "patience": 10,
            "min_delta": 1e-6,
            "lr_scheduler": True,
            "lr_scheduler_patience": 5,
            "lr_scheduler_factor": 0.5,
            "gradient_clip": 1.0,  # Gradient clipping to prevent explosion
            "save_best_model": True,
            "loss_weight_temp": 1.0,
            "loss_weight_frost": 1.0,
            "use_amp": True,  # Enable mixed precision training for 1.5-2x speedup
            "use_weighted_sampler": True,
            "val_frequency": 5,  # Validate every 5 epochs instead of every epoch (faster training)
        }
    
    elif model_type == "prophet":
        return {
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": True,
            "seasonality_mode": "multiplicative",
        }
    elif model_type == "persistence":
        # only needs threshold/scale; temp column name handled in model
        return {
            "frost_threshold": 0.0,
            "scale": 2.0,
            "temp_column": "Air Temp (C)",
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_model_class(model_type: str):
    """Get model class for a specific model type.
    
    Args:
        model_type: Model type (lightgbm, xgboost, catboost, etc.).
    
    Returns:
        Model class.
    """
    if model_type == "lightgbm":
        from src.models.ml.lightgbm_model import LightGBMModel
        return LightGBMModel
    elif model_type == "xgboost":
        from src.models.ml.xgboost_model import XGBoostModel
        return XGBoostModel
    elif model_type == "catboost":
        from src.models.ml.catboost_model import CatBoostModel
        return CatBoostModel
    elif model_type == "random_forest":
        from src.models.ml.random_forest_model import RandomForestModel
        return RandomForestModel
    elif model_type == "ensemble":
        from src.models.ml.ensemble_model import EnsembleModel
        return EnsembleModel
    elif model_type == "lstm":
        from src.models.deep.lstm_model import LSTMForecastModel
        return LSTMForecastModel
    elif model_type == "lstm_multitask":
        from src.models.deep.lstm_multitask_model import LSTMMultiTaskForecastModel
        return LSTMMultiTaskForecastModel
    elif model_type == "prophet":
        from src.models.traditional.prophet_model import ProphetModel
        return ProphetModel
    elif model_type == "extratrees":
        from src.models.ml.extratrees_model import ExtraTreesModel
        return ExtraTreesModel
    elif model_type in ["linear_regression", "ridge", "elasticnet", "logreg"]:
        from src.models.ml.lightgbm_model import LightGBMModel  # placeholder for import order
        from src.models.ml.linear_model import LinearModel
        return LinearModel
    elif model_type == "gru":
        from src.models.deep.gru_model import GRUForecastModel
        return GRUForecastModel
    elif model_type == "tcn":
        from src.models.deep.tcn_model import TCNForecastModel
        return TCNForecastModel
    elif model_type == "dcrnn":
        from src.models.graph.dcrnn_model import DCRNNForecastModel
        return DCRNNForecastModel
    elif model_type == "st_gcn":
        from src.models.graph.st_gcn_model import STGCNForecastModel
        return STGCNForecastModel
    elif model_type == "gat_lstm":
        from src.models.graph.gat_lstm_model import GATLSTMForecastModel
        return GATLSTMForecastModel
    elif model_type == "graphwavenet":
        from src.models.graph.graphwavenet_model import GraphWaveNetForecastModel
        return GraphWaveNetForecastModel
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_model_config(
    model_type: str,
    horizon: int,
    task_type: str = "classification",
    max_workers: Optional[int] = None,
    for_loso: bool = False,
    station_id: Optional[int] = None
) -> Dict:
    """Get complete model configuration.
    
    Args:
        model_type: Model type (lightgbm, xgboost, catboost, etc.).
        horizon: Forecast horizon in hours.
        task_type: Task type (classification or regression).
        max_workers: Maximum number of workers (auto-determined if None).
        for_loso: Whether this is for LOSO evaluation.
        station_id: Optional station ID for LOSO (used in model name).
    
    Returns:
        Dictionary with complete model configuration.
    """
    model_params = get_model_params(model_type, task_type, max_workers, for_loso)
    
    # Create model name
    if for_loso and station_id is not None:
        model_name = f"{task_type}_{horizon}h_station_{station_id}"
    else:
        model_name = f"{task_type}_{horizon}h"
    
    config = {
        "model_name": model_name,
        "model_type": model_type,
        "task_type": task_type,
        "model_params": model_params
    }
    
    # Add model-specific config
    if model_type == "ensemble":
        config["base_models"] = ["lightgbm", "xgboost", "catboost"]
        config["ensemble_method"] = "mean"
    elif model_type in ["lstm", "lstm_multitask"]:
        config["date_column"] = "Date"
        # Horizon-specific sequence length mapping
        try:
            seq_map = {3: 24, 6: 48, 12: 72, 24: 168}
            seq_len = seq_map.get(int(horizon), int(config["model_params"].get("sequence_length", 24)))
            config["model_params"]["sequence_length"] = seq_len
        except Exception:
            pass
        if model_type == "lstm_multitask":
            config["task_type"] = "multitask"
    elif model_type == "prophet":
        config["date_column"] = "Date"
        config["target_column"] = f"{task_type.split('_')[0]}_{horizon}h"
        config["regressor_columns"] = []
    
    return config

