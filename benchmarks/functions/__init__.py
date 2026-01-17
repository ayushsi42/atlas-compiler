"""
Benchmark Functions Package

Contains benchmark functions organized by category:
- numerical: matrix operations, dot product, polynomial evaluation
- reductions: sum, min, max, average
- bitwise: popcount, bit reverse, parity
"""

from benchmarks.functions.numerical import (
    dot_product,
    polynomial_eval,
    double_add,
    triple_multiply,
)
from benchmarks.functions.reductions import (
    array_sum,
    array_min,
    array_max,
)
from benchmarks.functions.bitwise import (
    popcount,
    is_power_of_two,
    bit_parity,
)

__all__ = [
    # Numerical
    "dot_product",
    "polynomial_eval",
    "double_add",
    "triple_multiply",
    # Reductions
    "array_sum",
    "array_min",
    "array_max",
    # Bitwise
    "popcount",
    "is_power_of_two",
    "bit_parity",
]
