"""
Tests for the Tribunal Verifier layer.
"""

import pytest
from z3 import BitVec


def test_prove_equivalence_identical():
    """Test that identical functions are verified as equivalent."""
    from atlas.verifier import prove_equivalence
    
    def func(x, y):
        return x + y
    
    result = prove_equivalence(func, func, num_inputs=2)
    
    assert result.verified is True
    assert "VERIFIED" in result.message


def test_prove_equivalence_correct_optimization():
    """Test verifying a correct optimization (multiply to shift)."""
    from atlas.verifier import prove_equivalence
    
    def original(x, y):
        return (x * 2) + (y * 2)
    
    def optimized(x, y):
        return (x << 1) + (y << 1)
    
    result = prove_equivalence(original, optimized, num_inputs=2)
    
    assert result.verified is True


def test_prove_equivalence_buggy_optimization():
    """Test detecting a bug in optimization."""
    from atlas.verifier import prove_equivalence
    
    def original(x, y):
        return (x * 2) + (y * 2)
    
    def buggy(x, y):
        return x << 1  # Bug: forgot y!
    
    result = prove_equivalence(original, buggy, num_inputs=2)
    
    assert result.verified is False
    assert "BUG FOUND" in result.message or "differ" in result.message.lower()
    assert result.counter_example is not None


def test_prove_equivalence_distributive():
    """Test distributive property optimization."""
    from atlas.verifier import prove_equivalence
    
    def original(x, y):
        return (x * 3) + (y * 3)
    
    def optimized(x, y):
        return (x + y) * 3
    
    result = prove_equivalence(original, optimized, num_inputs=2)
    
    assert result.verified is True


def test_prove_equivalence_bitwidth():
    """Test verification with different bit widths."""
    from atlas.verifier import prove_equivalence
    
    def func(x, y):
        return x + y
    
    # 8-bit
    result8 = prove_equivalence(func, func, num_inputs=2, bit_width=8)
    assert result8.verified is True
    
    # 64-bit
    result64 = prove_equivalence(func, func, num_inputs=2, bit_width=64)
    assert result64.verified is True
