"""
The Executor - LLVM MCJIT compilation and execution.
"""

from aot_gpt.executor.runtime import compile_and_execute, CompiledFunction

__all__ = ["compile_and_execute", "CompiledFunction"]
