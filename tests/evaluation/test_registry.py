"""Tests for evaluation registry."""

import pytest

from src.evaluation.registry import (
    register_evaluation_strategy,
    get_evaluation_handler,
)


def dummy_handler(runner, data, config):
    """Dummy evaluation handler for testing."""
    pass


class TestEvaluationRegistry:
    """Test cases for evaluation registry."""
    
    def test_register_strategy(self):
        """Test registering an evaluation strategy."""
        register_evaluation_strategy("test_strategy", dummy_handler)
        
        handler = get_evaluation_handler("test_strategy")
        assert handler is dummy_handler
    
    def test_register_strategy_lowercase(self):
        """Test that strategy names are normalized to lowercase."""
        register_evaluation_strategy("UPPERCASE_STRATEGY", dummy_handler)
        
        # Should be accessible via lowercase
        handler1 = get_evaluation_handler("uppercase_strategy")
        assert handler1 is dummy_handler
        
        # Should also be accessible via mixed case
        handler2 = get_evaluation_handler("UPPERCASE_STRATEGY")
        assert handler2 is dummy_handler
    
    def test_register_empty_name(self):
        """Test registering with empty name raises ValueError."""
        with pytest.raises(ValueError, match="Strategy name must be a non-empty string"):
            register_evaluation_strategy("", dummy_handler)
    
    def test_register_non_string_name(self):
        """Test registering with non-string name raises ValueError."""
        with pytest.raises(ValueError, match="Strategy name must be a non-empty string"):
            register_evaluation_strategy(123, dummy_handler)
    
    def test_register_non_callable_handler(self):
        """Test registering with non-callable handler raises ValueError."""
        with pytest.raises(ValueError, match="Handler must be callable"):
            register_evaluation_strategy("test", "not_callable")
    
    def test_register_overwrite_warning(self, caplog):
        """Test that overwriting existing strategy logs warning."""
        register_evaluation_strategy("overwrite_test", dummy_handler)
        register_evaluation_strategy("overwrite_test", dummy_handler)
        
        assert "Overwriting existing evaluation strategy" in caplog.text
    
    def test_get_handler_not_found(self):
        """Test getting non-existent handler raises ValueError."""
        with pytest.raises(ValueError, match="Unknown evaluation strategy"):
            get_evaluation_handler("nonexistent_strategy")
    
    def test_get_handler_shows_available(self):
        """Test error message shows available strategies."""
        # Register a strategy
        register_evaluation_strategy("available_test", dummy_handler)
        
        try:
            get_evaluation_handler("nonexistent")
        except ValueError as e:
            error_msg = str(e)
            assert "available_test" in error_msg.lower() or "available strategies" in error_msg.lower()
    
    def test_get_handler_empty_name(self):
        """Test getting handler with empty name raises ValueError."""
        with pytest.raises(ValueError, match="Strategy name must be a non-empty string"):
            get_evaluation_handler("")
    
    def test_get_handler_non_string_name(self):
        """Test getting handler with non-string name raises ValueError."""
        with pytest.raises(ValueError, match="Strategy name must be a non-empty string"):
            get_evaluation_handler(123)
    
    def test_multiple_strategies(self):
        """Test registering and retrieving multiple strategies."""
        def handler1(runner, data, config):
            pass
        
        def handler2(runner, data, config):
            pass
        
        register_evaluation_strategy("strategy_1", handler1)
        register_evaluation_strategy("strategy_2", handler2)
        
        assert get_evaluation_handler("strategy_1") is handler1
        assert get_evaluation_handler("strategy_2") is handler2

