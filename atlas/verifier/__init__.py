"""
The Tribunal Verifier - Formal verification using Z3.
"""

from atlas.verifier.prover import (
    prove_equivalence,
    prove_equivalence_with_preconditions,
    VerificationResult,
    check_no_overflow,
    check_division_safety,
    check_array_bounds,
    check_termination,
    check_null_safety,
)
from atlas.verifier.symbolic import SymbolicExecutor

__all__ = [
    "prove_equivalence",
    "prove_equivalence_with_preconditions",
    "VerificationResult",
    "SymbolicExecutor",
    # Safety checks
    "check_no_overflow",
    "check_division_safety",
    "check_array_bounds",
    "check_termination",
    "check_null_safety",
]
