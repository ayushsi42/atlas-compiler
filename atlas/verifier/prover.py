"""
Z3 Equivalence Prover - The core of the Tribunal.

Proves that optimized code is functionally equivalent to original code.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any, List
from z3 import (
    Solver, BitVec, BitVecVal, sat, unsat, unknown,
    simplify, And, Or, Not, If,
    SignExt,
    ULT, ULE, UGT, UGE,
)

from atlas.config import get_config


@dataclass
class VerificationResult:
    """Result of equivalence verification."""
    verified: bool
    message: str
    counter_example: Optional[dict] = None
    iterations_checked: int = 0
    time_ms: float = 0.0


def prove_equivalence(
    original_logic: Callable,
    optimized_logic: Callable,
    num_inputs: int = 2,
    bit_width: int = 32
) -> VerificationResult:
    """
    Prove that two functions produce equivalent outputs for all inputs.
    
    Uses Z3 to search for a counter-example (input where outputs differ).
    If no counter-example exists (UNSAT), the functions are equivalent.
    
    Args:
        original_logic: Function taking Z3 BitVec inputs, returns Z3 expression
        optimized_logic: Function taking Z3 BitVec inputs, returns Z3 expression
        num_inputs: Number of input variables
        bit_width: Bit width for integers (default 32)
        
    Returns:
        VerificationResult indicating success or failure with counter-example
    """
    import time
    start_time = time.time()
    
    config = get_config()
    
    # Create solver with timeout
    s = Solver()
    s.set("timeout", config.verifier.timeout_ms)
    
    # Create symbolic input variables
    inputs = [BitVec(f'x{i}', bit_width) for i in range(num_inputs)]
    
    try:
        # Compute symbolic outputs
        res_original = original_logic(*inputs)
        res_optimized = optimized_logic(*inputs)
        
        # The key equation: Find ANY input where outputs differ
        s.add(res_original != res_optimized)
        
        result = s.check()
        elapsed_ms = (time.time() - start_time) * 1000
        
        if result == unsat:
            # No counter-example exists = functions are equivalent
            return VerificationResult(
                verified=True,
                message=f"VERIFIED: Mathematically identical for all 2^{bit_width * num_inputs} inputs.",
                iterations_checked=1,
                time_ms=elapsed_ms,
            )
        elif result == sat:
            # Found a counter-example!
            model = s.model()
            counter_example = {
                f'x{i}': model[inputs[i]].as_long() if model[inputs[i]] is not None else 0
                for i in range(num_inputs)
            }
            
            # Compute actual values for the counter-example
            return VerificationResult(
                verified=False,
                message=f"BUG FOUND: Functions differ at inputs {counter_example}",
                counter_example=counter_example,
                iterations_checked=1,
                time_ms=elapsed_ms,
            )
        else:
            # Unknown (timeout or complexity limit)
            return VerificationResult(
                verified=False,
                message="UNKNOWN: Verification timed out or hit complexity limit. "
                        "Consider simplifying the function.",
                iterations_checked=1,
                time_ms=elapsed_ms,
            )
            
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return VerificationResult(
            verified=False,
            message=f"ERROR: Verification failed with exception: {e}",
            iterations_checked=0,
            time_ms=elapsed_ms,
        )


def prove_equivalence_with_preconditions(
    original_logic: Callable,
    optimized_logic: Callable,
    preconditions: Callable,
    num_inputs: int = 2,
    bit_width: int = 32
) -> VerificationResult:
    """
    Prove equivalence under specified preconditions.
    
    Useful for optimizations that only apply to certain input ranges
    (e.g., positive integers only).
    
    Args:
        original_logic: Original function
        optimized_logic: Optimized function  
        preconditions: Function that returns Z3 constraint on inputs
        num_inputs: Number of input variables
        bit_width: Bit width for integers
        
    Returns:
        VerificationResult
    """
    import time
    start_time = time.time()
    
    config = get_config()
    
    s = Solver()
    s.set("timeout", config.verifier.timeout_ms)
    
    inputs = [BitVec(f'x{i}', bit_width) for i in range(num_inputs)]
    
    try:
        # Apply preconditions
        pre = preconditions(*inputs)
        s.add(pre)
        
        # Compute outputs
        res_original = original_logic(*inputs)
        res_optimized = optimized_logic(*inputs)
        
        # Check for difference
        s.add(res_original != res_optimized)
        
        result = s.check()
        elapsed_ms = (time.time() - start_time) * 1000
        
        if result == unsat:
            return VerificationResult(
                verified=True,
                message="VERIFIED: Equivalent for all inputs satisfying preconditions.",
                iterations_checked=1,
                time_ms=elapsed_ms,
            )
        elif result == sat:
            model = s.model()
            counter_example = {
                f'x{i}': model[inputs[i]].as_long() if model[inputs[i]] is not None else 0
                for i in range(num_inputs)
            }
            return VerificationResult(
                verified=False,
                message=f"BUG FOUND: Functions differ at {counter_example} (within preconditions)",
                counter_example=counter_example,
                iterations_checked=1,
                time_ms=elapsed_ms,
            )
        else:
            return VerificationResult(
                verified=False,
                message="UNKNOWN: Verification timed out.",
                iterations_checked=1,
                time_ms=elapsed_ms,
            )
            
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return VerificationResult(
            verified=False,
            message=f"ERROR: {e}",
            iterations_checked=0,
            time_ms=elapsed_ms,
        )


# =============================================================================
# Verification Checks for Common Safety Properties
# =============================================================================

def check_no_overflow(
    func_logic: Callable,
    num_inputs: int = 2,
    bit_width: int = 32
) -> VerificationResult:
    """
    Check if a function can produce integer overflow.
    
    Note: This checks if the mathematical result exceeds bit bounds,
    not if overflow happens (which is defined behavior in LLVM).
    """
    import time
    start_time = time.time()
    
    s = Solver()
    inputs = [BitVec(f'x{i}', bit_width) for i in range(num_inputs)]
    
    # Extend to double width to check for overflow
    double_width = bit_width * 2
    extended_inputs = [SignExt(bit_width, inp) for inp in inputs]
    
    try:
        result = func_logic(*extended_inputs)
        
        # Check if result exceeds original bit width bounds
        max_val = BitVecVal((1 << (bit_width - 1)) - 1, double_width)
        min_val = BitVecVal(-(1 << (bit_width - 1)), double_width)
        
        s.add(Or(result > max_val, result < min_val))
        
        check = s.check()
        elapsed_ms = (time.time() - start_time) * 1000
        
        if check == unsat:
            return VerificationResult(
                verified=True,
                message="VERIFIED: No integer overflow possible.",
                time_ms=elapsed_ms,
            )
        elif check == sat:
            model = s.model()
            counter_example = {
                f'x{i}': model[inputs[i]].as_long() if model[inputs[i]] is not None else 0
                for i in range(num_inputs)
            }
            return VerificationResult(
                verified=False,
                message=f"OVERFLOW POSSIBLE: At inputs {counter_example}",
                counter_example=counter_example,
                time_ms=elapsed_ms,
            )
        else:
            return VerificationResult(
                verified=False,
                message="UNKNOWN: Overflow check timed out.",
                time_ms=elapsed_ms,
            )
            
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return VerificationResult(
            verified=False,
            message=f"ERROR: {e}",
            time_ms=elapsed_ms,
        )


def check_division_safety(
    numerator_logic: Callable,
    denominator_logic: Callable,
    num_inputs: int = 2,
    bit_width: int = 32
) -> VerificationResult:
    """
    Check if division by zero is possible.
    """
    import time
    start_time = time.time()
    
    s = Solver()
    inputs = [BitVec(f'x{i}', bit_width) for i in range(num_inputs)]
    
    try:
        denominator = denominator_logic(*inputs)
        s.add(denominator == 0)
        
        result = s.check()
        elapsed_ms = (time.time() - start_time) * 1000
        
        if result == unsat:
            return VerificationResult(
                verified=True,
                message="VERIFIED: Division by zero not possible.",
                time_ms=elapsed_ms,
            )
        elif result == sat:
            model = s.model()
            counter_example = {
                f'x{i}': model[inputs[i]].as_long() if model[inputs[i]] is not None else 0
                for i in range(num_inputs)
            }
            return VerificationResult(
                verified=False,
                message=f"DIVISION BY ZERO: Possible at inputs {counter_example}",
                counter_example=counter_example,
                time_ms=elapsed_ms,
            )
        else:
            return VerificationResult(
                verified=False,
                message="UNKNOWN: Check timed out.",
                time_ms=elapsed_ms,
            )
            
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return VerificationResult(
            verified=False,
            message=f"ERROR: {e}",
            time_ms=elapsed_ms,
        )


def check_array_bounds(
    index_logic: Callable,
    size_logic: Callable,
    num_inputs: int = 2,
    bit_width: int = 32
) -> VerificationResult:
    """
    Check if array access is always within bounds.
    
    Verifies that 0 <= index < size for all possible inputs.
    
    Args:
        index_logic: Function returning the array index expression
        size_logic: Function returning the array size expression
        num_inputs: Number of input variables
        bit_width: Bit width for integers
        
    Returns:
        VerificationResult indicating if bounds are always respected
    """
    import time
    start_time = time.time()
    
    s = Solver()
    inputs = [BitVec(f'x{i}', bit_width) for i in range(num_inputs)]
    
    try:
        index = index_logic(*inputs)
        size = size_logic(*inputs)
        
        # Check for out-of-bounds: index < 0 (signed) OR index >= size (unsigned)
        # For unsigned, we use UGE (unsigned greater or equal)
        zero = BitVecVal(0, bit_width)
        
        # Signed negative check (index as signed < 0)
        # Using the sign bit: if MSB is 1, it's negative in signed interpretation
        msb_mask = BitVecVal(1 << (bit_width - 1), bit_width)
        is_negative = (index & msb_mask) != zero
        
        # Unsigned bounds check: index >= size
        is_too_large = UGE(index, size)
        
        # Out of bounds if either condition is true
        s.add(Or(is_negative, is_too_large))
        
        result = s.check()
        elapsed_ms = (time.time() - start_time) * 1000
        
        if result == unsat:
            return VerificationResult(
                verified=True,
                message="VERIFIED: Array access always within bounds.",
                time_ms=elapsed_ms,
            )
        elif result == sat:
            model = s.model()
            counter_example = {
                f'x{i}': model[inputs[i]].as_long() if model[inputs[i]] is not None else 0
                for i in range(num_inputs)
            }
            idx_val = model.eval(index).as_long() if model.eval(index) is not None else 0
            sz_val = model.eval(size).as_long() if model.eval(size) is not None else 0
            return VerificationResult(
                verified=False,
                message=f"OUT OF BOUNDS: index={idx_val}, size={sz_val} at inputs {counter_example}",
                counter_example=counter_example,
                time_ms=elapsed_ms,
            )
        else:
            return VerificationResult(
                verified=False,
                message="UNKNOWN: Bounds check timed out.",
                time_ms=elapsed_ms,
            )
            
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return VerificationResult(
            verified=False,
            message=f"ERROR: {e}",
            time_ms=elapsed_ms,
        )


def check_termination(
    loop_counter_logic: Callable,
    loop_bound: int,
    num_inputs: int = 2,
    bit_width: int = 32
) -> VerificationResult:
    """
    Check if a loop terminates within a bounded number of iterations.
    
    Verifies that the loop counter reaches the exit condition within
    'loop_bound' iterations for all inputs.
    
    Args:
        loop_counter_logic: Function that returns the loop counter after N steps
        loop_bound: Maximum number of iterations allowed
        num_inputs: Number of input variables
        bit_width: Bit width for integers
        
    Returns:
        VerificationResult indicating if termination is guaranteed
    """
    import time
    start_time = time.time()
    
    s = Solver()
    inputs = [BitVec(f'x{i}', bit_width) for i in range(num_inputs)]
    
    try:
        # The loop_counter_logic should return the counter value after loop_bound iterations
        # If the counter hasn't reached 0 (or exit condition), the loop doesn't terminate
        final_counter = loop_counter_logic(*inputs, loop_bound)
        
        # Check if there exists an input where counter > 0 after max iterations
        zero = BitVecVal(0, bit_width)
        s.add(final_counter > zero)
        
        result = s.check()
        elapsed_ms = (time.time() - start_time) * 1000
        
        if result == unsat:
            return VerificationResult(
                verified=True,
                message=f"VERIFIED: Loop terminates within {loop_bound} iterations.",
                time_ms=elapsed_ms,
            )
        elif result == sat:
            model = s.model()
            counter_example = {
                f'x{i}': model[inputs[i]].as_long() if model[inputs[i]] is not None else 0
                for i in range(num_inputs)
            }
            return VerificationResult(
                verified=False,
                message=f"NON-TERMINATION: Loop may not terminate within {loop_bound} iterations at inputs {counter_example}",
                counter_example=counter_example,
                time_ms=elapsed_ms,
            )
        else:
            return VerificationResult(
                verified=False,
                message="UNKNOWN: Termination check timed out.",
                time_ms=elapsed_ms,
            )
            
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return VerificationResult(
            verified=False,
            message=f"ERROR: {e}",
            time_ms=elapsed_ms,
        )


def check_null_safety(
    ptr_logic: Callable,
    num_inputs: int = 2,
    bit_width: int = 64
) -> VerificationResult:
    """
    Check if a pointer can be null before dereference.
    
    Verifies that the pointer expression is never zero (null) for any input.
    
    Args:
        ptr_logic: Function returning the pointer expression
        num_inputs: Number of input variables
        bit_width: Bit width for pointers (default 64 for modern systems)
        
    Returns:
        VerificationResult indicating if null dereference is possible
    """
    import time
    start_time = time.time()
    
    s = Solver()
    inputs = [BitVec(f'x{i}', bit_width) for i in range(num_inputs)]
    
    try:
        ptr = ptr_logic(*inputs)
        null = BitVecVal(0, bit_width)
        
        # Check if pointer can be null
        s.add(ptr == null)
        
        result = s.check()
        elapsed_ms = (time.time() - start_time) * 1000
        
        if result == unsat:
            return VerificationResult(
                verified=True,
                message="VERIFIED: Pointer is never null.",
                time_ms=elapsed_ms,
            )
        elif result == sat:
            model = s.model()
            counter_example = {
                f'x{i}': model[inputs[i]].as_long() if model[inputs[i]] is not None else 0
                for i in range(num_inputs)
            }
            return VerificationResult(
                verified=False,
                message=f"NULL POINTER: Possible at inputs {counter_example}",
                counter_example=counter_example,
                time_ms=elapsed_ms,
            )
        else:
            return VerificationResult(
                verified=False,
                message="UNKNOWN: Null check timed out.",
                time_ms=elapsed_ms,
            )
            
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return VerificationResult(
            verified=False,
            message=f"ERROR: {e}",
            time_ms=elapsed_ms,
        )

