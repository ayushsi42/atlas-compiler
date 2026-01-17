"""
The Tribunal Verifier - Formal verification using Z3.
"""

from aot_gpt.verifier.prover import prove_equivalence, VerificationResult
from aot_gpt.verifier.symbolic import SymbolicExecutor

__all__ = ["prove_equivalence", "VerificationResult", "SymbolicExecutor"]
