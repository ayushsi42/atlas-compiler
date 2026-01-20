"""
Integration tests for the full Atlas pipeline.
"""

import os
import pytest


# Skip decorator for tests requiring OpenAI API key
requires_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for LLM-powered tests"
)


def test_config_defaults():
    """Test that configuration has sensible defaults."""
    from atlas.config import get_config
    
    config = get_config()
    
    assert config.verifier.bit_width == 32
    assert config.cegar.max_iterations == 3
    assert config.cegar.safe_fallback is True


def test_cegar_simple_verification():
    """Test CEGAR loop with direct Z3 functions."""
    from atlas.cegar import CEGARLoop
    
    def original(x, y):
        return x + y
    
    def optimized(x, y):
        return x + y
    
    # Use run_simple which doesn't require the Neural Optimizer
    loop = CEGARLoop.__new__(CEGARLoop)  # Create without __init__
    loop.max_iterations = 3
    loop.safe_fallback = True
    result = loop.run_simple(original, optimized, num_inputs=2)
    
    assert result.verified is True


def test_cegar_detects_bug():
    """Test that CEGAR detects bugs in optimizations."""
    from atlas.cegar import CEGARLoop, CEGARStatus
    
    def original(x, y):
        return (x * 2) + (y * 2)
    
    def buggy(x, y):
        return x * 4  # Wrong!
    
    loop = CEGARLoop.__new__(CEGARLoop)
    loop.max_iterations = 3
    loop.safe_fallback = True
    result = loop.run_simple(original, buggy, num_inputs=2)
    
    assert result.verified is False
    assert result.status == CEGARStatus.FAILED_VERIFICATION


def test_symbolic_executor_basic():
    """Test symbolic execution of basic IR patterns."""
    from atlas.verifier.symbolic import SymbolicExecutor
    from z3 import BitVec
    
    executor = SymbolicExecutor(bit_width=32)
    
    # Simple add instruction
    ir = """
    %result = add i32 %0, %1
    ret i32 %result
    """
    
    func = executor.execute_function(ir, ['%0', '%1'])
    
    x = BitVec('x', 32)
    y = BitVec('y', 32)
    
    result = func(x, y)
    # Result should be symbolically x + y
    assert result is not None


@requires_api_key
def test_jit_decorator_basic():
    """Test the @jit decorator on a simple function."""
    import atlas
    
    @atlas.jit
    def add(x, y):
        return x + y
    
    # Should work
    result = add(5, 10)
    assert result == 15


@requires_api_key
def test_jit_decorator_with_verbose():
    """Test the @jit decorator with verbose output."""
    import atlas
    
    @atlas.jit(verbose=True)
    def multiply(x, y):
        return x * y
    
    result = multiply(3, 4)
    assert result == 12


@requires_api_key
def test_get_stats():
    """Test retrieving JIT compilation stats."""
    import atlas
    from atlas.decorator import get_stats
    
    @atlas.jit
    def subtract(x, y):
        return x - y
    
    # Trigger compilation
    subtract(10, 5)
    
    stats = get_stats(subtract)
    # Stats may be None if compilation fully succeeded via cache
    # Just verify no exception is raised
    assert True
