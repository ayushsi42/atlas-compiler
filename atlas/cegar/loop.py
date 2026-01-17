"""
CEGAR Loop - Counter-Example Guided Abstraction Refinement.

The feedback loop that refines optimizations when verification fails.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Callable

from z3 import BitVec

from atlas.config import get_config
from atlas.neural.agent import NeuralOptimizer, OptimizationResult
from atlas.verifier.prover import prove_equivalence, VerificationResult
from atlas.verifier.symbolic import ir_to_z3_function


class CEGARStatus(Enum):
    """Status of the CEGAR loop."""
    SUCCESS = "success"
    FAILED_MAX_RETRIES = "failed_max_retries"
    FAILED_VERIFICATION = "failed_verification"
    FALLBACK = "fallback"


@dataclass
class CEGARIteration:
    """Record of a single CEGAR iteration."""
    iteration: int
    optimized_ir: str
    verification_result: VerificationResult
    counter_example: Optional[dict] = None


@dataclass
class CEGARResult:
    """Final result of the CEGAR loop."""
    status: CEGARStatus
    original_ir: str
    final_ir: str
    verified: bool
    iterations: List[CEGARIteration] = field(default_factory=list)
    total_time_ms: float = 0.0
    strategy_used: Optional[str] = None
    
    @property
    def used_fallback(self) -> bool:
        return self.status == CEGARStatus.FALLBACK
    
    @property
    def num_refinements(self) -> int:
        return len(self.iterations) - 1 if self.iterations else 0


class CEGARLoop:
    """
    Counter-Example Guided Abstraction Refinement loop.
    
    Orchestrates the optimization-verification cycle:
    1. Get optimized IR from Neural Optimizer
    2. Verify equivalence with Z3
    3. If failed, feed counter-example back to LLM
    4. Repeat until verified or max retries reached
    5. Fall back to original if all retries fail
    """
    
    def __init__(
        self,
        optimizer: Optional[NeuralOptimizer] = None,
        max_iterations: Optional[int] = None
    ):
        config = get_config()
        self.optimizer = optimizer or NeuralOptimizer()
        self.max_iterations = max_iterations or config.cegar.max_iterations
        self.safe_fallback = config.cegar.safe_fallback
    
    def run(
        self,
        original_ir: str,
        signature: str,
        param_names: List[str]
    ) -> CEGARResult:
        """
        Run the CEGAR loop.
        
        Args:
            original_ir: The original LLVM IR
            signature: Function signature string
            param_names: List of parameter names for symbolic execution
            
        Returns:
            CEGARResult with final optimized (or fallback) IR
        """
        import time
        start_time = time.time()
        
        iterations: List[CEGARIteration] = []
        current_ir = original_ir
        last_counter_example = None
        last_error = None
        strategy_used = None
        
        # Get original as Z3 function
        try:
            original_z3_func = ir_to_z3_function(original_ir, param_names)
        except Exception as e:
            # Can't symbolically execute original, use fallback
            elapsed_ms = (time.time() - start_time) * 1000
            return CEGARResult(
                status=CEGARStatus.FALLBACK,
                original_ir=original_ir,
                final_ir=original_ir,
                verified=False,
                total_time_ms=elapsed_ms,
            )
        
        for i in range(self.max_iterations):
            # Step 1: Optimize
            if i == 0:
                # First iteration: get fresh optimization
                opt_result = self.optimizer.optimize(current_ir, signature)
                current_ir = opt_result.optimized_ir
                strategy_used = opt_result.strategy_used
            else:
                # Subsequent iterations: refine with counter-example
                current_ir = self.optimizer.refine_with_counterexample(
                    original_ir=original_ir,
                    failed_optimization=current_ir,
                    counter_example=str(last_counter_example),
                    error_message=last_error or "Functional equivalence failed",
                )
            
            # Step 2: Verify
            try:
                optimized_z3_func = ir_to_z3_function(current_ir, param_names)
                verification = self._verify(
                    original_z3_func,
                    optimized_z3_func,
                    num_inputs=len(param_names),
                )
            except Exception as e:
                # Symbolic execution failed for optimized IR
                verification = VerificationResult(
                    verified=False,
                    message=f"Failed to symbolically execute optimized IR: {e}",
                    counter_example=None,
                )
            
            # Record iteration
            iterations.append(CEGARIteration(
                iteration=i + 1,
                optimized_ir=current_ir,
                verification_result=verification,
                counter_example=verification.counter_example,
            ))
            
            # Step 3: Check result
            if verification.verified:
                # Success!
                elapsed_ms = (time.time() - start_time) * 1000
                return CEGARResult(
                    status=CEGARStatus.SUCCESS,
                    original_ir=original_ir,
                    final_ir=current_ir,
                    verified=True,
                    iterations=iterations,
                    total_time_ms=elapsed_ms,
                    strategy_used=strategy_used,
                )
            
            # Prepare for next iteration
            last_counter_example = verification.counter_example
            last_error = verification.message
        
        # Max iterations reached without success
        elapsed_ms = (time.time() - start_time) * 1000
        
        if self.safe_fallback:
            # Fall back to original code
            return CEGARResult(
                status=CEGARStatus.FALLBACK,
                original_ir=original_ir,
                final_ir=original_ir,  # Fallback to original
                verified=True,  # Original is trivially equivalent to itself
                iterations=iterations,
                total_time_ms=elapsed_ms,
                strategy_used=strategy_used,
            )
        else:
            # Return unverified optimization
            return CEGARResult(
                status=CEGARStatus.FAILED_MAX_RETRIES,
                original_ir=original_ir,
                final_ir=current_ir,
                verified=False,
                iterations=iterations,
                total_time_ms=elapsed_ms,
                strategy_used=strategy_used,
            )
    
    def _verify(
        self,
        original_func: Callable,
        optimized_func: Callable,
        num_inputs: int,
    ) -> VerificationResult:
        """Run Z3 verification."""
        config = get_config()
        return prove_equivalence(
            original_logic=original_func,
            optimized_logic=optimized_func,
            num_inputs=num_inputs,
            bit_width=config.verifier.bit_width,
        )
    
    def run_simple(
        self,
        original_logic: Callable,
        optimized_logic: Callable,
        num_inputs: int = 2,
    ) -> CEGARResult:
        """
        Simplified CEGAR for direct Z3 functions (no IR parsing).
        
        Useful for testing and the demo in plan.md.
        """
        import time
        start_time = time.time()
        
        verification = prove_equivalence(
            original_logic=original_logic,
            optimized_logic=optimized_logic,
            num_inputs=num_inputs,
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        iter_record = CEGARIteration(
            iteration=1,
            optimized_ir="<direct function>",
            verification_result=verification,
            counter_example=verification.counter_example,
        )
        
        return CEGARResult(
            status=CEGARStatus.SUCCESS if verification.verified else CEGARStatus.FAILED_VERIFICATION,
            original_ir="<direct function>",
            final_ir="<direct function>",
            verified=verification.verified,
            iterations=[iter_record],
            total_time_ms=elapsed_ms,
        )
