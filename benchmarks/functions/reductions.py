"""
Reduction Benchmark Functions

Functions for testing loop and accumulation optimizations.
"""


def array_sum(n: int) -> int:
    """
    Sum of first n integers: 1 + 2 + ... + n
    
    Optimization target: Closed form n*(n+1)/2
    """
    total = 0
    for i in range(1, n + 1):
        total += i
    return total


def array_min(a: int, b: int, c: int) -> int:
    """
    Find minimum of three values.
    
    Optimization target: Branch-to-select, min intrinsic.
    """
    if a < b:
        if a < c:
            return a
        else:
            return c
    else:
        if b < c:
            return b
        else:
            return c


def array_max(a: int, b: int, c: int) -> int:
    """
    Find maximum of three values.
    
    Optimization target: Branch-to-select, max intrinsic.
    """
    if a > b:
        if a > c:
            return a
        else:
            return c
    else:
        if b > c:
            return b
        else:
            return c


def abs_value(x: int) -> int:
    """
    Absolute value.
    
    Optimization target: Branchless abs using bit manipulation.
    """
    if x < 0:
        return -x
    return x


def clamp(x: int, lo: int, hi: int) -> int:
    """
    Clamp value to range [lo, hi].
    
    Optimization target: min(max(x, lo), hi)
    """
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
