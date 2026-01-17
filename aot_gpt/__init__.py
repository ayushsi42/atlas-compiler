"""
AOT-GPT: A Verifiable Neural JIT Compiler

This package provides the @jit decorator that:
1. Lifts Python functions to LLVM IR
2. Optimizes using AI (Neural Core)
3. Verifies correctness using Z3 (Tribunal)
4. Compiles to native machine code (Executor)
"""

from aot_gpt.decorator import jit

__version__ = "0.1.0"
__all__ = ["jit"]
