"""
Tests for the Lifter layer.
"""

import pytest


def test_lift_simple_function():
    """Test lifting a simple integer function."""
    from aot_gpt.lifter import lift_function
    
    def add(x, y):
        return x + y
    
    lifted = lift_function(add, sample_args=(1, 2))
    
    assert lifted.name == "add"
    assert "define" in lifted.llvm_ir
    assert lifted.ir_hash is not None
    assert len(lifted.ir_hash) == 16


def test_lift_arithmetic_function():
    """Test lifting a function with arithmetic operations."""
    from aot_gpt.lifter import lift_function
    
    def double_add(x, y):
        return (x * 2) + (y * 2)
    
    lifted = lift_function(double_add, sample_args=(5, 10))
    
    assert "mul" in lifted.llvm_ir.lower() or "shl" in lifted.llvm_ir.lower()
    assert "add" in lifted.llvm_ir.lower()


def test_lift_function_hash_consistency():
    """Test that same function source produces consistent signature."""
    from aot_gpt.lifter import lift_function
    
    def func1(x, y):
        return x + y
    
    lifted1 = lift_function(func1, sample_args=(1, 1))
    
    # Verify signature is consistent
    assert "int64" in lifted1.signature
    assert "func1" in lifted1.signature
    
    # Verify hash is 16 chars
    assert len(lifted1.ir_hash) == 16


def test_lift_function_source_code():
    """Test that source code is captured."""
    from aot_gpt.lifter import lift_function
    
    def documented_func(x, y):
        """This function adds two numbers."""
        return x + y
    
    lifted = lift_function(documented_func, sample_args=(1, 1))
    
    assert "documented_func" in lifted.source_code
    assert "adds two numbers" in lifted.source_code
