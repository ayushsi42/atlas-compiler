"""
Optimization strategies for the Neural Optimizer.

For MVP, these are hardcoded patterns instead of RAG from a vector DB.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class StrategyType(Enum):
    """Types of optimization strategies."""
    STRENGTH_REDUCTION = "strength_reduction"
    LOOP_UNROLLING = "loop_unrolling"
    SIMD_VECTORIZATION = "simd_vectorization"
    CONSTANT_FOLDING = "constant_folding"
    DEAD_CODE_ELIMINATION = "dead_code_elimination"
    COMMON_SUBEXPRESSION = "common_subexpression"


@dataclass
class OptimizationStrategy:
    """Represents an optimization strategy."""
    name: str
    strategy_type: StrategyType
    description: str
    applicable_patterns: List[str]
    expected_speedup: str
    risk_level: str  # LOW, MEDIUM, HIGH
    examples: List[str]
    
    def to_string(self) -> str:
        """Format strategy for LLM prompt."""
        return f"""
Strategy: {self.name}
Type: {self.strategy_type.value}
Description: {self.description}
Expected Speedup: {self.expected_speedup}
Risk Level: {self.risk_level}
Applicable To: {', '.join(self.applicable_patterns)}
"""


# Predefined optimization strategies
STRATEGIES = [
    OptimizationStrategy(
        name="Multiply-to-Shift",
        strategy_type=StrategyType.STRENGTH_REDUCTION,
        description="Replace multiplication by powers of 2 with left bit shifts. "
                    "x * 2 becomes x << 1, x * 4 becomes x << 2, etc.",
        applicable_patterns=["multiply by constant", "power of 2 multiplication"],
        expected_speedup="1.5x-2x",
        risk_level="LOW",
        examples=[
            "mul i32 %x, 2  ->  shl i32 %x, 1",
            "mul i32 %x, 8  ->  shl i32 %x, 3",
        ]
    ),
    OptimizationStrategy(
        name="Divide-to-Shift",
        strategy_type=StrategyType.STRENGTH_REDUCTION,
        description="Replace division by powers of 2 with right bit shifts (for unsigned) "
                    "or arithmetic shifts (for signed positive values).",
        applicable_patterns=["divide by constant", "power of 2 division"],
        expected_speedup="2x-4x",
        risk_level="MEDIUM",
        examples=[
            "udiv i32 %x, 2  ->  lshr i32 %x, 1",
            "sdiv i32 %x, 4  ->  ashr i32 %x, 2  (only for positive)",
        ]
    ),
    OptimizationStrategy(
        name="Loop-Unroll-4x",
        strategy_type=StrategyType.LOOP_UNROLLING,
        description="Unroll loops by a factor of 4, reducing loop overhead and enabling "
                    "better instruction pipelining.",
        applicable_patterns=["simple loop", "accumulator loop", "reduction loop"],
        expected_speedup="2x-4x",
        risk_level="MEDIUM",
        examples=[
            "Loop body executed once per iteration -> 4 copies with index increment by 4",
        ]
    ),
    OptimizationStrategy(
        name="Distributive-Factoring",
        strategy_type=StrategyType.COMMON_SUBEXPRESSION,
        description="Apply distributive property to factor out common terms. "
                    "a*c + b*c becomes (a+b)*c, reducing multiplications.",
        applicable_patterns=["distributive arithmetic", "common factor"],
        expected_speedup="1.2x-1.5x",
        risk_level="LOW",
        examples=[
            "(x * 2) + (y * 2)  ->  (x + y) * 2  ->  (x + y) << 1",
        ]
    ),
    OptimizationStrategy(
        name="Constant-Propagation",
        strategy_type=StrategyType.CONSTANT_FOLDING,
        description="Evaluate constant expressions at compile time and propagate "
                    "known values through the code.",
        applicable_patterns=["constant arithmetic", "known values"],
        expected_speedup="1.1x-1.3x",
        risk_level="LOW",
        examples=[
            "%a = add i32 2, 3  ; computed as 5 at compile time",
        ]
    ),
    OptimizationStrategy(
        name="SIMD-Vector-4",
        strategy_type=StrategyType.SIMD_VECTORIZATION,
        description="Vectorize operations using 128-bit SIMD (4x i32). Requires "
                    "aligned memory and independent loop iterations.",
        applicable_patterns=["array processing", "element-wise operations", "reduction"],
        expected_speedup="3x-4x",
        risk_level="HIGH",
        examples=[
            "4 scalar add instructions -> single vector add",
        ]
    ),
]


def get_all_strategies() -> List[OptimizationStrategy]:
    """Get all available optimization strategies."""
    return STRATEGIES.copy()


def get_strategy(name: str) -> Optional[OptimizationStrategy]:
    """Get a specific strategy by name."""
    for s in STRATEGIES:
        if s.name.lower() == name.lower():
            return s
    return None


def get_strategies_for_pattern(pattern: str) -> List[OptimizationStrategy]:
    """Find strategies applicable to a given pattern."""
    pattern_lower = pattern.lower()
    matching = []
    for s in STRATEGIES:
        for p in s.applicable_patterns:
            if pattern_lower in p.lower() or p.lower() in pattern_lower:
                matching.append(s)
                break
    return matching


def format_strategies_for_prompt() -> str:
    """Format all strategies for inclusion in LLM prompt."""
    lines = ["Available Optimization Strategies:", "=" * 40]
    for i, s in enumerate(STRATEGIES, 1):
        lines.append(f"\n{i}. {s.name}")
        lines.append(f"   Type: {s.strategy_type.value}")
        lines.append(f"   {s.description}")
        lines.append(f"   Speedup: {s.expected_speedup}, Risk: {s.risk_level}")
    return "\n".join(lines)
