"""Probability calibration utilities for improving Brier Score and ECE."""

import logging
import numpy as np
from typing import Optional, Tuple

_logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Probability calibrator using Platt Scaling or Isotonic Regression."""
    
    def __init__(self, method: str = "platt"):
        """Initialize calibrator.
        
        Args:
            method: Calibration method - "platt" (logistic regression) or "isotonic" (isotonic regression)
        
        Raises:
            ValueError: If method is not "platt" or "isotonic".
        """
        if method not in ["platt", "isotonic"]:
            raise ValueError(f"Unknown calibration method: {method}. Must be 'platt' or 'isotonic'.")
        self.method = method
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibrator on validation probabilities and true labels.
        
        Args:
            y_prob: Uncalibrated probabilities (1D array)
            y_true: True binary labels (1D array)
        
        Returns:
            Self for method chaining.
        
        Raises:
            ValueError: If inputs are empty or have incompatible shapes.
        """
        # Input validation
        if len(y_prob) == 0 or len(y_true) == 0:
            raise ValueError(f"Input arrays cannot be empty. y_prob length: {len(y_prob)}, y_true length: {len(y_true)}")
        
        if len(y_prob) != len(y_true):
            raise ValueError(f"y_prob and y_true must have the same length. Got {len(y_prob)} and {len(y_true)}")
        
        # Check if probabilities are in valid range and clip if needed
        if np.any(y_prob < 0) or np.any(y_prob > 1):
            _logger.warning(f"Some probabilities are outside [0, 1] range. Clipping will be applied.")
            y_prob = np.clip(y_prob, 0.0, 1.0)  # Clip in fit to ensure isotonic regression gets valid inputs
        
        try:
            if self.method == "platt":
                from sklearn.linear_model import LogisticRegression
                # Platt scaling: fit logistic regression on logit(prob)
                # Avoid log(0) and log(1) by clipping
                y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
                logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
                self.calibrator = LogisticRegression()
                self.calibrator.fit(logits.reshape(-1, 1), y_true)
            elif self.method == "isotonic":
                from sklearn.isotonic import IsotonicRegression
                # Isotonic regression: non-parametric calibration
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(y_prob, y_true)
            else:
                raise ValueError(f"Unknown calibration method: {self.method}")
            
            self.is_fitted = True
            _logger.debug(f"Fitted {self.method} calibrator on {len(y_prob)} samples")
        except ImportError as e:
            # Fallback if sklearn not available
            _logger.warning(f"sklearn not available: {e}. Calibration will be disabled.")
            self.calibrator = None
            self.is_fitted = False
        
        return self
    
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities.
        
        Args:
            y_prob: Uncalibrated probabilities (1D array)
        
        Returns:
            Calibrated probabilities (1D array)
        
        Raises:
            ValueError: If input is empty.
        """
        if not self.is_fitted or self.calibrator is None:
            _logger.debug("Calibrator not fitted or unavailable. Returning original probabilities.")
            return y_prob
        
        # Input validation
        if len(y_prob) == 0:
            raise ValueError("Input probability array cannot be empty")
        
        try:
            if self.method == "platt":
                # Apply Platt scaling
                y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
                logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
                calibrated = self.calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
            elif self.method == "isotonic":
                # Apply isotonic regression
                calibrated = self.calibrator.transform(y_prob)
            else:
                return y_prob
            
            # Ensure probabilities are in [0, 1]
            calibrated = np.clip(calibrated, 0.0, 1.0)
            return calibrated
        except (ValueError, AttributeError, TypeError) as e:
            # Specific errors that can occur during calibration
            _logger.warning(f"Calibration failed: {e}. Returning original probabilities.")
            return y_prob
        except Exception as e:
            # Unexpected errors
            _logger.error(f"Unexpected error during calibration: {e}. Returning original probabilities.", exc_info=True)
            return y_prob
    
    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            y_prob: Uncalibrated probabilities (1D array)
            y_true: True binary labels (1D array)
        
        Returns:
            Calibrated probabilities (1D array)
        """
        return self.fit(y_prob, y_true).transform(y_prob)

