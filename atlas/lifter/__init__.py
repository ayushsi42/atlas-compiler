"""
The Lifter Service - Translates Python to LLVM IR.
"""

from atlas.lifter.translator import lift_function, LiftedFunction, extract_function_ir
from atlas.lifter.cache import IRCache
from atlas.lifter.types import AtlasType, TypeInfo, infer_type, infer_types_from_args

__all__ = [
    "lift_function",
    "LiftedFunction",
    "extract_function_ir",
    "IRCache",
    # Type system
    "AtlasType",
    "TypeInfo",
    "infer_type",
    "infer_types_from_args",
]
