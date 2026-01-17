"""
Tests for the Neural Optimizer layer.
"""

import pytest


def test_strategies_exist():
    """Test that optimization strategies are defined."""
    from aot_gpt.neural.strategies import get_all_strategies
    
    strategies = get_all_strategies()
    
    assert len(strategies) > 0
    assert any(s.name == "Multiply-to-Shift" for s in strategies)


def test_get_strategy_by_name():
    """Test retrieving a strategy by name."""
    from aot_gpt.neural.strategies import get_strategy
    
    strategy = get_strategy("Multiply-to-Shift")
    
    assert strategy is not None
    assert strategy.name == "Multiply-to-Shift"
    assert "shift" in strategy.description.lower()


def test_get_strategy_case_insensitive():
    """Test that strategy lookup is case insensitive."""
    from aot_gpt.neural.strategies import get_strategy
    
    s1 = get_strategy("Multiply-to-Shift")
    s2 = get_strategy("multiply-to-shift")
    s3 = get_strategy("MULTIPLY-TO-SHIFT")
    
    assert s1 == s2 == s3


def test_strategies_for_pattern():
    """Test finding strategies for a pattern."""
    from aot_gpt.neural.strategies import get_strategies_for_pattern
    
    strategies = get_strategies_for_pattern("multiply")
    
    assert len(strategies) > 0
    assert any("multiply" in s.name.lower() or "strength" in s.strategy_type.value 
               for s in strategies)


def test_format_strategies():
    """Test formatting strategies for prompts."""
    from aot_gpt.neural.strategies import format_strategies_for_prompt
    
    formatted = format_strategies_for_prompt()
    
    assert "Multiply-to-Shift" in formatted
    assert "Loop-Unroll" in formatted
    assert "Speedup" in formatted
