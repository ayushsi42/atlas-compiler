"""
Numerical Benchmark Functions

Functions for testing numerical optimization strategies.
"""


def double_add(x: int, y: int) -> int:
    """
    Classic double-add: (x * 2) + (y * 2)
    
    Optimization target: Distributive property + strength reduction
    Expected: (x + y) << 1
    """
    return (x * 2) + (y * 2)


def triple_multiply(x: int, y: int, z: int) -> int:
    """
    Triple multiply: x * y * z
    
    Tests associativity and reordering.
    """
    return x * y * z


def dot_product(a: int, b: int, c: int, d: int) -> int:
    """
    Simple 2-element dot product: a*c + b*d
    
    Optimization target: FMA (Fused Multiply-Add) or SIMD.
    """
    return a * c + b * d


def polynomial_eval(x: int, a: int, b: int, c: int) -> int:
    """
    Polynomial: a*x^2 + b*x + c
    
    Optimization target: Horner's method, strength reduction.
    Expected: x * (a*x + b) + c
    """
    return a * x * x + b * x + c


def power_of_two_mult(x: int, n: int) -> int:
    """
    Multiply by power of 2: x * (2^n) where n is known at compile time.
    
    For benchmarking, we use x * 8 (n=3)
    Optimization target: Left shift
    Expected: x << 3
    """
    return x * 8


def divide_by_power_of_two(x: int) -> int:
    """
    Divide unsigned by power of 2: x / 4
    
    Optimization target: Right shift
    Expected: x >> 2 (for non-negative values)
    """
    # In Python, // is floor division
    return x // 4


def modulo_power_of_two(x: int) -> int:
    """
    Modulo by power of 2: x % 8
    
    Optimization target: Bitwise AND
    Expected: x & 7
    """
    return x % 8
