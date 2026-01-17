"""
Atlas Type System - Extended type support for the lifter.

Provides type inference for NumPy arrays, booleans, and additional numeric types.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple, Union
import numpy as np

from numba.core import types as numba_types


class AtlasType(Enum):
    """Supported Atlas types."""
    # Scalars
    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BOOL = "boolean"
    
    # Arrays (1D)
    ARRAY_INT32 = "int32[:]"
    ARRAY_INT64 = "int64[:]"
    ARRAY_FLOAT32 = "float32[:]"
    ARRAY_FLOAT64 = "float64[:]"
    ARRAY_BOOL = "boolean[:]"
    
    # Arrays (2D)
    ARRAY_2D_INT32 = "int32[:,:]"
    ARRAY_2D_INT64 = "int64[:,:]"
    ARRAY_2D_FLOAT32 = "float32[:,:]"
    ARRAY_2D_FLOAT64 = "float64[:,:]"
    
    # Unknown/fallback
    UNKNOWN = "unknown"


@dataclass
class TypeInfo:
    """Detailed type information."""
    atlas_type: AtlasType
    numba_type: Any
    is_array: bool = False
    ndim: int = 0
    dtype: Optional[AtlasType] = None
    
    @property
    def signature_str(self) -> str:
        """Get the Numba signature string for this type."""
        return self.atlas_type.value


def infer_type(value: Any) -> TypeInfo:
    """
    Infer the Atlas type from a Python value.
    
    Args:
        value: A Python value (int, float, bool, numpy array, etc.)
        
    Returns:
        TypeInfo with the inferred type
    """
    # NumPy arrays
    if isinstance(value, np.ndarray):
        return _infer_array_type(value)
    
    # Booleans (must check before int since bool is subclass of int)
    if isinstance(value, bool):
        return TypeInfo(
            atlas_type=AtlasType.BOOL,
            numba_type=numba_types.boolean,
        )
    
    # Integers
    if isinstance(value, (int, np.integer)):
        if isinstance(value, np.int32):
            return TypeInfo(
                atlas_type=AtlasType.INT32,
                numba_type=numba_types.int32,
            )
        return TypeInfo(
            atlas_type=AtlasType.INT64,
            numba_type=numba_types.int64,
        )
    
    # Floats
    if isinstance(value, (float, np.floating)):
        if isinstance(value, np.float32):
            return TypeInfo(
                atlas_type=AtlasType.FLOAT32,
                numba_type=numba_types.float32,
            )
        return TypeInfo(
            atlas_type=AtlasType.FLOAT64,
            numba_type=numba_types.float64,
        )
    
    # Unknown - default to int64
    return TypeInfo(
        atlas_type=AtlasType.INT64,
        numba_type=numba_types.int64,
    )


def _infer_array_type(arr: np.ndarray) -> TypeInfo:
    """Infer type information from a NumPy array."""
    dtype = arr.dtype
    ndim = arr.ndim
    
    # Map dtype to Atlas type
    dtype_map = {
        np.dtype('int32'): (AtlasType.INT32, numba_types.int32),
        np.dtype('int64'): (AtlasType.INT64, numba_types.int64),
        np.dtype('float32'): (AtlasType.FLOAT32, numba_types.float32),
        np.dtype('float64'): (AtlasType.FLOAT64, numba_types.float64),
        np.dtype('bool'): (AtlasType.BOOL, numba_types.boolean),
    }
    
    element_info = dtype_map.get(dtype, (AtlasType.INT64, numba_types.int64))
    element_atlas, element_numba = element_info
    
    if ndim == 1:
        # 1D array
        array_type_map = {
            AtlasType.INT32: AtlasType.ARRAY_INT32,
            AtlasType.INT64: AtlasType.ARRAY_INT64,
            AtlasType.FLOAT32: AtlasType.ARRAY_FLOAT32,
            AtlasType.FLOAT64: AtlasType.ARRAY_FLOAT64,
            AtlasType.BOOL: AtlasType.ARRAY_BOOL,
        }
        array_type = array_type_map.get(element_atlas, AtlasType.ARRAY_INT64)
        numba_array_type = numba_types.Array(element_numba, 1, 'C')
        
        return TypeInfo(
            atlas_type=array_type,
            numba_type=numba_array_type,
            is_array=True,
            ndim=1,
            dtype=element_atlas,
        )
    
    elif ndim == 2:
        # 2D array
        array_type_map = {
            AtlasType.INT32: AtlasType.ARRAY_2D_INT32,
            AtlasType.INT64: AtlasType.ARRAY_2D_INT64,
            AtlasType.FLOAT32: AtlasType.ARRAY_2D_FLOAT32,
            AtlasType.FLOAT64: AtlasType.ARRAY_2D_FLOAT64,
        }
        array_type = array_type_map.get(element_atlas, AtlasType.ARRAY_2D_INT64)
        numba_array_type = numba_types.Array(element_numba, 2, 'C')
        
        return TypeInfo(
            atlas_type=array_type,
            numba_type=numba_array_type,
            is_array=True,
            ndim=2,
            dtype=element_atlas,
        )
    
    # Fallback for higher dimensions - treat as 1D
    numba_array_type = numba_types.Array(element_numba, ndim, 'C')
    return TypeInfo(
        atlas_type=AtlasType.UNKNOWN,
        numba_type=numba_array_type,
        is_array=True,
        ndim=ndim,
        dtype=element_atlas,
    )


def to_numba_type(atlas_type: AtlasType) -> Any:
    """
    Convert an Atlas type to the corresponding Numba type.
    
    Args:
        atlas_type: The Atlas type enum value
        
    Returns:
        The corresponding Numba type
    """
    type_map = {
        AtlasType.INT32: numba_types.int32,
        AtlasType.INT64: numba_types.int64,
        AtlasType.FLOAT32: numba_types.float32,
        AtlasType.FLOAT64: numba_types.float64,
        AtlasType.BOOL: numba_types.boolean,
        AtlasType.ARRAY_INT32: numba_types.Array(numba_types.int32, 1, 'C'),
        AtlasType.ARRAY_INT64: numba_types.Array(numba_types.int64, 1, 'C'),
        AtlasType.ARRAY_FLOAT32: numba_types.Array(numba_types.float32, 1, 'C'),
        AtlasType.ARRAY_FLOAT64: numba_types.Array(numba_types.float64, 1, 'C'),
        AtlasType.ARRAY_BOOL: numba_types.Array(numba_types.boolean, 1, 'C'),
        AtlasType.ARRAY_2D_INT32: numba_types.Array(numba_types.int32, 2, 'C'),
        AtlasType.ARRAY_2D_INT64: numba_types.Array(numba_types.int64, 2, 'C'),
        AtlasType.ARRAY_2D_FLOAT32: numba_types.Array(numba_types.float32, 2, 'C'),
        AtlasType.ARRAY_2D_FLOAT64: numba_types.Array(numba_types.float64, 2, 'C'),
    }
    return type_map.get(atlas_type, numba_types.int64)


def infer_types_from_args(args: Tuple) -> Tuple[TypeInfo, ...]:
    """
    Infer types for a tuple of arguments.
    
    Args:
        args: Tuple of Python values
        
    Returns:
        Tuple of TypeInfo objects
    """
    return tuple(infer_type(arg) for arg in args)


def format_signature(func_name: str, arg_types: Tuple[TypeInfo, ...], return_type: TypeInfo) -> str:
    """
    Format a human-readable function signature.
    
    Args:
        func_name: The function name
        arg_types: Tuple of argument TypeInfo
        return_type: Return type TypeInfo
        
    Returns:
        Formatted signature string like "func(int64, float64[:]) -> float64"
    """
    arg_strs = [t.signature_str for t in arg_types]
    return f"{func_name}({', '.join(arg_strs)}) -> {return_type.signature_str}"
