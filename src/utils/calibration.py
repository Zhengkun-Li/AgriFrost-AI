"""Probability calibration utilities for improving Brier Score and ECE."""

import numpy as np
from typing import Optional, Tuple


class ProbabilityCalibrator:
    """Probability calibrator using Platt Scaling or Isotonic Regression."""
    
    def __init__(self, method: str = "platt"):
        """Initialize calibrator.
        
        Args:
            method: Calibration method - "platt" (logistic regression) or "isotonic" (isotonic regression)
        """
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
        """
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
        except ImportError:
            # Fallback if sklearn not available
            self.calibrator = None
            self.is_fitted = False
        
        return self
    
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities.
        
        Args:
            y_prob: Uncalibrated probabilities (1D array)
        
        Returns:
            Calibrated probabilities (1D array)
        """
        if not self.is_fitted or self.calibrator is None:
            return y_prob
        
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
        except Exception:
            # Fallback to original probabilities if calibration fails
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

