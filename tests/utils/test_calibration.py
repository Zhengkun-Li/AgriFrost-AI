"""Tests for probability calibration utilities."""

import pytest
import numpy as np

from src.utils.calibration import ProbabilityCalibrator


class TestProbabilityCalibrator:
    """Test cases for ProbabilityCalibrator."""
    
    def test_init_platt(self):
        """Test initialization with 'platt' method."""
        calibrator = ProbabilityCalibrator(method="platt")
        assert calibrator.method == "platt"
        assert not calibrator.is_fitted
        assert calibrator.calibrator is None
    
    def test_init_isotonic(self):
        """Test initialization with 'isotonic' method."""
        calibrator = ProbabilityCalibrator(method="isotonic")
        assert calibrator.method == "isotonic"
    
    def test_init_invalid_method(self):
        """Test initialization with invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown calibration method"):
            ProbabilityCalibrator(method="invalid")
    
    def test_fit_empty_prob(self):
        """Test fit with empty probabilities raises ValueError."""
        calibrator = ProbabilityCalibrator(method="platt")
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            calibrator.fit(np.array([]), np.array([0, 1]))
    
    def test_fit_empty_true(self):
        """Test fit with empty true labels raises ValueError."""
        calibrator = ProbabilityCalibrator(method="platt")
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            calibrator.fit(np.array([0.5, 0.6]), np.array([]))
    
    def test_fit_mismatched_lengths(self):
        """Test fit with mismatched lengths raises ValueError."""
        calibrator = ProbabilityCalibrator(method="platt")
        with pytest.raises(ValueError, match="must have the same length"):
            calibrator.fit(np.array([0.5, 0.6]), np.array([0, 1, 0]))
    
    def test_fit_platt(self):
        """Test fit with Platt scaling."""
        try:
            calibrator = ProbabilityCalibrator(method="platt")
            y_prob = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
            y_true = np.array([0, 0, 1, 1, 1])
            
            result = calibrator.fit(y_prob, y_true)
            assert result is calibrator  # Method chaining
            assert calibrator.is_fitted
            assert calibrator.calibrator is not None
        except ImportError:
            pytest.skip("scikit-learn not available")
    
    def test_fit_isotonic(self):
        """Test fit with isotonic regression."""
        try:
            calibrator = ProbabilityCalibrator(method="isotonic")
            y_prob = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
            y_true = np.array([0, 0, 1, 1, 1])
            
            calibrator.fit(y_prob, y_true)
            assert calibrator.is_fitted
            assert calibrator.calibrator is not None
        except ImportError:
            pytest.skip("scikit-learn not available")
    
    def test_fit_out_of_range_warning(self, caplog):
        """Test fit with probabilities outside [0, 1] issues warning."""
        try:
            calibrator = ProbabilityCalibrator(method="platt")
            y_prob = np.array([-0.1, 0.5, 1.5])  # Out of range
            y_true = np.array([0, 1, 1])
            
            calibrator.fit(y_prob, y_true)
            assert "outside [0, 1] range" in caplog.text
        except ImportError:
            pytest.skip("scikit-learn not available")
    
    def test_transform_not_fitted(self):
        """Test transform when not fitted returns original."""
        calibrator = ProbabilityCalibrator(method="platt")
        y_prob = np.array([0.5, 0.6])
        
        result = calibrator.transform(y_prob)
        np.testing.assert_array_equal(result, y_prob)
    
    def test_transform_empty(self):
        """Test transform with empty array raises ValueError."""
        try:
            calibrator = ProbabilityCalibrator(method="platt")
            y_prob = np.array([0.3, 0.4, 0.5])
            y_true = np.array([0, 1, 1])
            calibrator.fit(y_prob, y_true)
            
            with pytest.raises(ValueError, match="Input probability array cannot be empty"):
                calibrator.transform(np.array([]))
        except ImportError:
            pytest.skip("scikit-learn not available")
    
    def test_transform_platt(self):
        """Test transform with Platt scaling."""
        try:
            calibrator = ProbabilityCalibrator(method="platt")
            y_prob_train = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
            y_true_train = np.array([0, 0, 1, 1, 1])
            calibrator.fit(y_prob_train, y_true_train)
            
            y_prob_test = np.array([0.35, 0.55, 0.65])
            result = calibrator.transform(y_prob_test)
            
            assert len(result) == len(y_prob_test)
            assert np.all(result >= 0)
            assert np.all(result <= 1)
        except ImportError:
            pytest.skip("scikit-learn not available")
    
    def test_fit_transform(self):
        """Test fit_transform in one step."""
        try:
            calibrator = ProbabilityCalibrator(method="platt")
            y_prob = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
            y_true = np.array([0, 0, 1, 1, 1])
            
            result = calibrator.fit_transform(y_prob, y_true)
            
            assert len(result) == len(y_prob)
            assert np.all(result >= 0)
            assert np.all(result <= 1)
            assert calibrator.is_fitted
        except ImportError:
            pytest.skip("scikit-learn not available")

