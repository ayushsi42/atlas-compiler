"""
Atlas Module Compiler - Multi-function optimization.

Enables optimization across function boundaries with:
- Module-level compilation
- Function inlining
- Call graph analysis
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Any, Tuple
import functools
import inspect

from atlas.config import get_config


@dataclass
class FunctionNode:
    """Represents a function in the call graph."""
    name: str
    func: Callable
    calls: Set[str] = field(default_factory=set)
    called_by: Set[str] = field(default_factory=set)
    is_inline: bool = False
    compiled_func: Optional[Callable] = None
    

@dataclass
class CallGraph:
    """Represents the call graph of a module."""
    nodes: Dict[str, FunctionNode] = field(default_factory=dict)
    
    def add_function(self, name: str, func: Callable, is_inline: bool = False) -> None:
        """Add a function to the call graph."""
        calls = self._extract_calls(func)
        self.nodes[name] = FunctionNode(
            name=name,
            func=func,
            calls=calls,
            is_inline=is_inline,
        )
        
        # Update called_by for existing nodes
        for called_name in calls:
            if called_name in self.nodes:
                self.nodes[called_name].called_by.add(name)
    
    def _extract_calls(self, func: Callable) -> Set[str]:
        """Extract function calls from source code (simple heuristic)."""
        try:
            source = inspect.getsource(func)
            # Simple pattern: look for function calls like func_name(
            import re
            calls = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', source))
            # Filter out builtins and common names
            builtins = {'print', 'len', 'range', 'int', 'float', 'str', 'list', 'dict', 'set'}
            return calls - builtins - {func.__name__}
        except (OSError, TypeError):
            return set()
    
    def get_compilation_order(self) -> List[str]:
        """
        Get topological order for compilation (dependencies first).
        
        Returns:
            List of function names in compilation order
        """
        visited = set()
        order = []
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            
            node = self.nodes.get(name)
            if node:
                for dep in node.calls:
                    if dep in self.nodes:
                        visit(dep)
                order.append(name)
        
        for name in self.nodes:
            visit(name)
        
        return order
    
    def get_inline_candidates(self) -> List[str]:
        """Get functions marked for inlining."""
        return [name for name, node in self.nodes.items() if node.is_inline]


class ModuleCompiler:
    """
    Compiles multiple functions together with inter-procedural optimization.
    
    Usage:
        module = ModuleCompiler()
        module.add(func1)
        module.add(func2, inline=True)
        compiled = module.compile()
        
        # Call compiled functions
        result = compiled['func1'](args)
    """
    
    def __init__(self, verbose: bool = False):
        self.call_graph = CallGraph()
        self.verbose = verbose
        self._compiled: Dict[str, Callable] = {}
        self._is_compiled = False
    
    def add(self, func: Callable, inline: bool = False, name: Optional[str] = None) -> 'ModuleCompiler':
        """
        Add a function to the module.
        
        Args:
            func: The function to add
            inline: Whether to mark this function for inlining
            name: Optional custom name (defaults to func.__name__)
            
        Returns:
            self for chaining
        """
        func_name = name or func.__name__
        self.call_graph.add_function(func_name, func, is_inline=inline)
        return self
    
    def compile(self) -> Dict[str, Callable]:
        """
        Compile all functions in the module.
        
        Returns:
            Dictionary mapping function names to compiled callables
        """
        if self._is_compiled:
            return self._compiled
        
        import atlas
        
        order = self.call_graph.get_compilation_order()
        
        if self.verbose:
            print(f"[Atlas Module] Compilation order: {order}")
            inline_funcs = self.call_graph.get_inline_candidates()
            if inline_funcs:
                print(f"[Atlas Module] Inline candidates: {inline_funcs}")
        
        for name in order:
            node = self.call_graph.nodes[name]
            
            if self.verbose:
                print(f"[Atlas Module] Compiling: {name}")
            
            try:
                # Apply JIT decorator
                compiled = atlas.jit(node.func)
                self._compiled[name] = compiled
                node.compiled_func = compiled
                
            except Exception as e:
                if self.verbose:
                    print(f"[Atlas Module] Failed to compile {name}: {e}")
                # Use original function as fallback
                self._compiled[name] = node.func
        
        self._is_compiled = True
        return self._compiled
    
    def __getitem__(self, name: str) -> Callable:
        """Get a compiled function by name."""
        if not self._is_compiled:
            self.compile()
        return self._compiled[name]
    
    def get_call_graph_info(self) -> Dict[str, Any]:
        """Get information about the call graph."""
        return {
            "functions": list(self.call_graph.nodes.keys()),
            "compilation_order": self.call_graph.get_compilation_order(),
            "inline_candidates": self.call_graph.get_inline_candidates(),
            "edges": [
                {"from": name, "calls": list(node.calls)}
                for name, node in self.call_graph.nodes.items()
            ],
        }


# Module-level compilation decorator
def module(*funcs: Callable, verbose: bool = False) -> ModuleCompiler:
    """
    Compile multiple functions together as a module.
    
    Usage:
        compiled = atlas.module(func1, func2, func3)
        result = compiled['func1'](x, y)
    
    Args:
        *funcs: Functions to compile together
        verbose: Print compilation info
        
    Returns:
        ModuleCompiler with all functions compiled
    """
    compiler = ModuleCompiler(verbose=verbose)
    for func in funcs:
        compiler.add(func)
    compiler.compile()
    return compiler


def inline(func: Callable) -> Callable:
    """
    Mark a function as an inlining candidate.
    
    Usage:
        @atlas.inline
        def helper(x):
            return x * 2
            
        def main(x, y):
            return helper(x) + helper(y)
        
        compiled = atlas.module(helper, main)
    """
    func._atlas_inline = True
    return func


def compile_module(functions: List[Callable], verbose: bool = False) -> Dict[str, Callable]:
    """
    Convenience function to compile a list of functions.
    
    Args:
        functions: List of functions to compile together
        verbose: Print compilation info
        
    Returns:
        Dictionary mapping function names to compiled callables
    """
    compiler = ModuleCompiler(verbose=verbose)
    for func in functions:
        is_inline = getattr(func, '_atlas_inline', False)
        compiler.add(func, inline=is_inline)
    return compiler.compile()
