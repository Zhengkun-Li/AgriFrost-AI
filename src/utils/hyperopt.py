"""Hyperparameter optimization utilities."""

import copy
import logging
from typing import Dict, Any, Callable, Optional
import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, space_eval
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    space_eval = None  # Placeholder if hyperopt not available

try:
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class HyperparameterOptimizer:
    """Hyperparameter optimization using Hyperopt."""
    
    def __init__(self, model_class, config_template: Dict[str, Any], max_evals: int = 50):
        """Initialize optimizer.
        
        Args:
            model_class: Model class to optimize.
            config_template: Template configuration dictionary.
            max_evals: Maximum number of evaluations.
        
        Raises:
            ImportError: If hyperopt is not available.
            ValueError: If max_evals <= 0.
        """
        if not HYPEROPT_AVAILABLE:
            raise ImportError("Hyperopt is required. Install with: pip install hyperopt")
        
        if max_evals <= 0:
            raise ValueError(f"max_evals must be positive, got {max_evals}")
        
        self.model_class = model_class
        self.config_template = config_template
        self.max_evals = max_evals
        self.trials = Trials()
        self.best_config = None
        self.best_score = None
        self.space = None  # Store search space for decoding trial parameters
        
        _logger.debug(f"Initialized HyperparameterOptimizer with max_evals={max_evals}")
    
    def optimize(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 space: Dict[str, Any],
                 objective_func: Optional[Callable] = None,
                 metric: str = "neg_mean_absolute_error",
                 cv: int = 3) -> Dict[str, Any]:
        """Optimize hyperparameters.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            space: Hyperparameter search space (using Hyperopt syntax).
            objective_func: Custom objective function. If None, uses cross-validation.
            metric: Metric to optimize (for sklearn cross-validation).
            cv: Number of cross-validation folds.
        
        Returns:
            Best hyperparameter configuration.
        
        Raises:
            ValueError: If inputs are invalid (empty DataFrame/Series, invalid cv).
        """
        # Input validation
        if X.empty:
            raise ValueError("Feature DataFrame X cannot be empty")
        
        if len(y) == 0:
            raise ValueError("Target Series y cannot be empty")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got {len(X)} and {len(y)}")
        
        if cv <= 0:
            raise ValueError(f"cv must be positive, got {cv}")
        
        if objective_func is None:
            def objective_func(params):
                return self._default_objective(X, y, params, metric, cv)
        
        _logger.info(f"Starting hyperparameter optimization with {self.max_evals} evaluations")
        
        def objective(params):
            try:
                score = objective_func(params)
                return {'loss': -score, 'status': STATUS_OK}
            except (ValueError, TypeError, AttributeError) as e:
                # Specific errors that indicate invalid parameters
                _logger.debug(f"Trial failed with specific error: {e}")
                return {'loss': float('inf'), 'status': STATUS_FAIL, 'error': str(e)}
            except Exception as e:
                # Unexpected errors
                _logger.warning(f"Trial failed with unexpected error: {e}")
                return {'loss': float('inf'), 'status': STATUS_FAIL, 'error': str(e)}
        
        # Store space for later parameter decoding
        self.space = space
        
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=self.trials
        )
        
        self.best_config = best
        # Robust extraction of successful trials with error handling
        successful_trials = []
        for t in self.trials.trials:
            try:
                if t.get('result', {}).get('status') == STATUS_OK:
                    successful_trials.append(t['result']['loss'])
            except (KeyError, TypeError, AttributeError):
                # Skip trials with invalid structure
                continue
        
        if successful_trials:
            self.best_score = -min(successful_trials)
            _logger.info(f"Optimization completed. Best score: {self.best_score:.4f}")
        else:
            _logger.warning("No successful trials found during optimization.")
            self.best_score = None
        
        return best
    
    def _default_objective(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any], metric: str, cv: int) -> float:
        """Default objective function using cross-validation.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            params: Hyperparameters to evaluate.
            metric: Metric to optimize.
            cv: Number of CV folds.
        
        Returns:
            Average CV score.
        
        Raises:
            ImportError: If scikit-learn is not available.
            ValueError: If metric is not supported.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for default objective")
        
        # Update config with hyperparameters (deep copy to avoid modifying template)
        config = copy.deepcopy(self.config_template)
        if "model_params" not in config:
            config["model_params"] = {}
        
        # Convert Hyperopt params to model params
        for key, value in params.items():
            if key.startswith("model_"):
                param_key = key.replace("model_", "")
                config["model_params"][param_key] = value
            else:
                config[key] = value
        
        # Cross-validation
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                # Create and train model for this fold
                model_temp = self.model_class(config)
                model_temp.fit(X_train, y_train)
                y_pred = model_temp.predict(X_val)
                
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                if metric == "neg_mean_absolute_error":
                    score = -mean_absolute_error(y_val, y_pred)
                elif metric == "neg_mean_squared_error":
                    score = -mean_squared_error(y_val, y_pred)
                elif metric == "r2":
                    score = r2_score(y_val, y_pred)
                else:
                    _logger.warning(f"Unknown metric '{metric}', using neg_mean_absolute_error")
                    score = -mean_absolute_error(y_val, y_pred)
                
                scores.append(score)
            except Exception as e:
                _logger.debug(f"Fold {fold_idx + 1} failed: {e}")
                # Skip failed folds instead of adding inf values
                # This prevents np.mean from returning inf when some folds succeed
                continue
        
        if not scores:
            raise ValueError("All CV folds failed. Check model configuration and data.")
        
        # Compute mean only from successful folds (exclude failed ones)
        return np.mean(scores)
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get best configuration found during optimization.
        
        Returns:
            Best configuration dictionary.
        
        Raises:
            ValueError: If optimization has not been run yet.
        """
        if self.best_config is None:
            raise ValueError("Optimization has not been run yet. Call optimize() first.")
        
        # Deep copy to avoid modifying template
        config = copy.deepcopy(self.config_template)
        if "model_params" not in config:
            config["model_params"] = {}
        
        for key, value in self.best_config.items():
            if key.startswith("model_"):
                param_key = key.replace("model_", "")
                config["model_params"][param_key] = value
            else:
                config[key] = value
        
        return config
    
    def get_trials_summary(self) -> pd.DataFrame:
        """Get summary of all trials with decoded parameters.
        
        Returns:
            DataFrame with trial results. Empty DataFrame if no trials available.
            Parameters are decoded from hyperopt's internal representation to actual values.
        """
        results = []
        for trial in self.trials.trials:
            # Robust error handling: check if trial structure is valid
            try:
                if trial.get('result', {}).get('status') == STATUS_OK:
                    # Decode hyperopt's internal parameter representation to actual values
                    vals = trial.get('misc', {}).get('vals', {})
                    if self.space is not None and space_eval is not None:
                        try:
                            decoded_params = space_eval(self.space, vals)
                        except Exception as e:
                            _logger.debug(f"Failed to decode trial parameters: {e}. Using raw vals.")
                            decoded_params = vals
                    else:
                        # If space not available or hyperopt not available, use raw vals
                        decoded_params = vals
                    
                    results.append({
                        'loss': trial['result']['loss'],
                        'params': decoded_params
                    })
            except (KeyError, TypeError, AttributeError) as e:
                # Skip trials with invalid structure
                _logger.debug(f"Skipping trial with invalid structure: {e}")
                continue
        
        if not results:
            _logger.warning("No successful trials to summarize.")
        
        return pd.DataFrame(results)

