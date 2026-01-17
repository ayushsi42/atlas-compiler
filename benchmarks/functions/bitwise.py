"""
Bitwise Benchmark Functions

Functions for testing bit manipulation optimizations.
"""


def popcount(x: int) -> int:
    """
    Population count: count the number of 1 bits.
    
    Optimization target: CPU popcnt instruction.
    """
    count = 0
    # Process 32 bits
    for _ in range(32):
        count += x & 1
        x = x >> 1
    return count


def is_power_of_two(x: int) -> int:
    """
    Check if x is a power of 2 (return 1 if true, 0 if false).
    
    Optimization target: (x & (x-1)) == 0 && x != 0
    """
    if x <= 0:
        return 0
    if (x & (x - 1)) == 0:
        return 1
    return 0


def bit_parity(x: int) -> int:
    """
    Compute parity (XOR of all bits): 1 if odd number of 1s, 0 otherwise.
    
    Optimization target: Parallel XOR reduction.
    """
    # Simple XOR approach for 32-bit
    x = x ^ (x >> 16)
    x = x ^ (x >> 8)
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    return x & 1


def next_power_of_two(x: int) -> int:
    """
    Round up to next power of 2.
    
    Optimization target: Bit manipulation sequence.
    """
    x = x - 1
    x = x | (x >> 1)
    x = x | (x >> 2)
    x = x | (x >> 4)
    x = x | (x >> 8)
    x = x | (x >> 16)
    return x + 1


def leading_zeros(x: int) -> int:
    """
    Count leading zeros in a 32-bit integer.
    
    Optimization target: CPU lzcnt/clz instruction.
    """
    if x == 0:
        return 32
    
    count = 0
    # Check upper half
    if (x & 0xFFFF0000) == 0:
        count += 16
        x = x << 16
    if (x & 0xFF000000) == 0:
        count += 8
        x = x << 8
    if (x & 0xF0000000) == 0:
        count += 4
        x = x << 4
    if (x & 0xC0000000) == 0:
        count += 2
        x = x << 2
    if (x & 0x80000000) == 0:
        count += 1
    return count
