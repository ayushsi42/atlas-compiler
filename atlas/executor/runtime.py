"""
Runtime - LLVM MCJIT compilation and execution.

Compiles LLVM IR to native machine code and executes it.
"""

import ctypes
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import llvmlite.binding as llvm

from atlas.config import get_config


# Note: LLVM initialization is now handled automatically by llvmlite


@dataclass
class CompiledFunction:
    """Represents a compiled native function."""
    name: str
    address: int
    signature: Tuple[Any, ...]  # ctypes signature
    engine: Any  # Keep engine alive
    
    def __call__(self, *args):
        """Call the compiled function."""
        # Create ctypes function from address
        func_type = ctypes.CFUNCTYPE(self.signature[0], *self.signature[1:])
        func = func_type(self.address)
        return func(*args)


class LLVMCompiler:
    """
    Compiles LLVM IR to native machine code using MCJIT.
    """
    
    def __init__(self, opt_level: Optional[int] = None):
        config = get_config()
        self.opt_level = opt_level if opt_level is not None else config.executor.opt_level
        
        # Create target machine
        target = llvm.Target.from_default_triple()
        self._target_machine = target.create_target_machine(
            opt=self.opt_level,
            reloc='pic',
            codemodel='small',
        )
    
    def compile(self, ir_string: str, function_name: str) -> CompiledFunction:
        """
        Compile LLVM IR to native code.
        
        Args:
            ir_string: The LLVM IR code
            function_name: Name of the function to compile
            
        Returns:
            CompiledFunction that can be called directly
        """
        # Parse the IR
        try:
            mod = llvm.parse_assembly(ir_string)
            mod.verify()
        except Exception as e:
            raise ValueError(f"Failed to parse LLVM IR: {e}") from e
        
        # Optimize the module
        if self.opt_level > 0:
            pmb = llvm.PassManagerBuilder()
            pmb.opt_level = self.opt_level
            
            pm = llvm.ModulePassManager()
            pmb.populate(pm)
            pm.run(mod)
        
        # Create execution engine
        engine = llvm.create_mcjit_compiler(mod, self._target_machine)
        engine.finalize_object()
        
        # Get function address
        func_addr = engine.get_function_address(function_name)
        if func_addr == 0:
            raise ValueError(f"Function '{function_name}' not found in compiled module")
        
        # Infer ctypes signature from IR (simplified: assume int64 for all)
        signature = self._infer_signature(ir_string, function_name)
        
        return CompiledFunction(
            name=function_name,
            address=func_addr,
            signature=signature,
            engine=engine,  # Keep engine alive
        )
    
    def _infer_signature(self, ir_string: str, function_name: str) -> Tuple[Any, ...]:
        """
        Infer ctypes signature from LLVM IR.
        
        For MVP, we assume i64 return and i64 arguments.
        """
        import re
        
        # Find function definition
        pattern = rf'define\s+(\w+)\s+@{re.escape(function_name)}\s*\(([^)]*)\)'
        match = re.search(pattern, ir_string)
        
        if not match:
            # Default: int64 return, two int64 args
            return (ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
        
        ret_type_str = match.group(1)
        args_str = match.group(2)
        
        # Map LLVM types to ctypes
        type_map = {
            'i64': ctypes.c_int64,
            'i32': ctypes.c_int32,
            'i16': ctypes.c_int16,
            'i8': ctypes.c_int8,
            'i1': ctypes.c_bool,
            'void': None,
            'float': ctypes.c_float,
            'double': ctypes.c_double,
        }
        
        ret_type = type_map.get(ret_type_str, ctypes.c_int64)
        
        # Count arguments
        arg_types = []
        for arg in args_str.split(','):
            arg = arg.strip()
            if arg:
                # Extract type (first word)
                arg_type_str = arg.split()[0] if arg.split() else 'i64'
                arg_type = type_map.get(arg_type_str, ctypes.c_int64)
                if arg_type:
                    arg_types.append(arg_type)
        
        return (ret_type, *arg_types)
    
    def get_assembly(self, ir_string: str) -> str:
        """
        Get native assembly from LLVM IR (for visualization).
        """
        try:
            mod = llvm.parse_assembly(ir_string)
            mod.verify()
            return self._target_machine.emit_assembly(mod)
        except Exception as e:
            return f"Error generating assembly: {e}"


def compile_and_execute(
    ir_string: str,
    function_name: str,
    *args,
    opt_level: Optional[int] = None
) -> Any:
    """
    Compile LLVM IR and immediately execute with given arguments.
    
    Args:
        ir_string: The LLVM IR code
        function_name: Name of the function to call
        *args: Arguments to pass to the function
        opt_level: LLVM optimization level (0-3)
        
    Returns:
        The function's return value
    """
    compiler = LLVMCompiler(opt_level=opt_level)
    compiled = compiler.compile(ir_string, function_name)
    return compiled(*args)


def get_native_assembly(ir_string: str, opt_level: int = 3) -> str:
    """
    Convert LLVM IR to native assembly string.
    
    Useful for the Streamlit dashboard visualization.
    """
    compiler = LLVMCompiler(opt_level=opt_level)
    return compiler.get_assembly(ir_string)
