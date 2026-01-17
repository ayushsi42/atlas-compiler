"""
Atlas: A Verifiable Neural JIT Compiler

This package provides the @jit decorator that:
1. Lifts Python functions to LLVM IR
2. Optimizes using AI (Neural Core)
3. Verifies correctness using Z3 (Tribunal)
4. Compiles to native machine code (Executor)

Module compilation:
- atlas.module(*funcs) - compile multiple functions together
- atlas.compile_module(funcs) - compile a list of functions
- @atlas.inline - mark function for inlining
"""

from atlas.decorator import jit, get_stats, get_original
from atlas.module import module, compile_module, inline, ModuleCompiler

__version__ = "0.1.0"
__all__ = [
    # Core
    "jit",
    "get_stats",
    "get_original",
    # Module compilation
    "module",
    "compile_module",
    "inline",
    "ModuleCompiler",
]
