"""
Lifter Translator - Converts Python functions to LLVM IR using Numba.
"""

import hashlib
import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any

from numba import njit
from numba.core import types as numba_types


@dataclass
class LiftedFunction:
    """Represents a function that has been lifted to LLVM IR."""
    original_func: Callable
    llvm_ir: str
    ir_hash: str
    signature: str
    arg_types: Tuple[Any, ...]
    return_type: Any
    source_code: str
    compiled_func: Callable  # The njit-compiled function
    
    @property
    def name(self) -> str:
        return self.original_func.__name__


def _get_source_code(func: Callable) -> str:
    """Extract source code from a function."""
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        return ""


def _infer_signature_str(func: Callable, sample_args: Optional[Tuple] = None) -> str:
    """
    Create a Numba signature string from sample arguments.
    
    Supports:
    - int64, int32, float64, float32
    - boolean
    - 1D and 2D NumPy arrays
    """
    from atlas.lifter.types import infer_type, infer_types_from_args, AtlasType
    
    sig = inspect.signature(func)
    num_params = len(sig.parameters)
    
    if sample_args:
        # Infer from sample arguments using new type system
        type_infos = infer_types_from_args(sample_args)
        type_strs = [t.signature_str for t in type_infos]
        
        # Infer return type (default to first arg type for numeric, or int64)
        if type_infos:
            first_type = type_infos[0]
            if first_type.is_array:
                # Array functions typically return scalar of same dtype
                return_type = first_type.dtype.value if first_type.dtype else "int64"
            else:
                return_type = first_type.signature_str
        else:
            return_type = "int64"
    else:
        # Default to int64 for all params (fallback)
        type_strs = ["int64"] * num_params
        return_type = "int64"
    
    return f"{return_type}({', '.join(type_strs)})"


def _compute_ir_hash(ir: str) -> str:
    """Compute a hash of the LLVM IR for caching."""
    return hashlib.sha256(ir.encode()).hexdigest()[:16]


def lift_function(
    func: Callable,
    sample_args: Optional[Tuple] = None,
    arg_types: Optional[Tuple] = None
) -> LiftedFunction:
    """
    Lift a Python function to LLVM IR using Numba.
    
    Args:
        func: The Python function to lift
        sample_args: Optional sample arguments for type inference
        arg_types: Optional explicit Numba type tuple
    
    Returns:
        LiftedFunction containing the LLVM IR and metadata
    
    Raises:
        ValueError: If the function cannot be compiled
    """
    source_code = _get_source_code(func)
    
    # Create signature string
    sig_str = _infer_signature_str(func, sample_args)
    
    # Compile using njit with the signature
    try:
        # Compile the function with Numba
        compiled = njit(sig_str)(func)
        
        # Force compilation by inspecting types
        # This triggers the actual compilation
        compiled_signatures = compiled.signatures
        
        # Get the LLVM IR
        # inspect_llvm returns a dict of overloads
        llvm_dict = compiled.inspect_llvm()
        
        # Get the first (and likely only) overload's IR
        if llvm_dict:
            llvm_ir = list(llvm_dict.values())[0]
        else:
            # Fallback: trigger compilation with sample args
            if sample_args:
                _ = compiled(*sample_args)
                llvm_dict = compiled.inspect_llvm()
                llvm_ir = list(llvm_dict.values())[0] if llvm_dict else ""
            else:
                llvm_ir = ""
        
        # Compute hash for caching
        ir_hash = _compute_ir_hash(llvm_ir)
        
        # Infer types for metadata using new type system
        from atlas.lifter.types import infer_types_from_args
        
        num_params = len(inspect.signature(func).parameters)
        if sample_args:
            type_infos = infer_types_from_args(sample_args)
            arg_types_tuple = tuple(t.numba_type for t in type_infos)
            # Infer return type from first arg
            if type_infos:
                return_type = type_infos[0].numba_type if not type_infos[0].is_array else (
                    type_infos[0].dtype and type_infos[0].numba_type.dtype or numba_types.int64
                )
            else:
                return_type = numba_types.int64
        else:
            arg_types_tuple = tuple(numba_types.int64 for _ in range(num_params))
            return_type = numba_types.int64
        
        # Build full signature string for display
        full_sig = f"{func.__name__}({', '.join(str(t) for t in arg_types_tuple)}) -> {return_type}"
        
        return LiftedFunction(
            original_func=func,
            llvm_ir=llvm_ir,
            ir_hash=ir_hash,
            signature=full_sig,
            arg_types=arg_types_tuple,
            return_type=return_type,
            source_code=source_code,
            compiled_func=compiled,
        )
        
    except Exception as e:
        raise ValueError(f"Failed to lift function '{func.__name__}': {e}") from e


def extract_function_ir(lifted: LiftedFunction) -> str:
    """
    Extract just the function body IR from the full module IR.
    
    This strips out metadata and focuses on the actual function logic.
    """
    lines = lifted.llvm_ir.split('\n')
    func_lines = []
    in_function = False
    func_name = lifted.name
    
    for line in lines:
        if 'define' in line and func_name in line:
            in_function = True
        if in_function:
            func_lines.append(line)
            if line.strip() == '}':
                break
    
    return '\n'.join(func_lines)

