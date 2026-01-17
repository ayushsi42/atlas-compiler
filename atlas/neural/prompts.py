"""
LangChain prompt templates for the Neural Optimizer.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# System message for all neural optimizer interactions
SYSTEM_MESSAGE = """You are an expert LLVM IR optimizer. Your task is to analyze and optimize LLVM IR code 
to improve performance while maintaining functional equivalence.

You understand:
- LLVM IR syntax and semantics
- CPU architecture optimizations (SIMD, cache locality, branch prediction)
- Compiler optimization techniques (loop unrolling, strength reduction, vectorization)
- Integer arithmetic patterns and their efficient implementations

CRITICAL: Your optimizations MUST be functionally equivalent to the original code.
Any incorrect optimization will be caught by the formal verifier."""


# Intent Analysis Prompt
INTENT_ANALYSIS_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE),
    HumanMessagePromptTemplate.from_template("""Analyze the following LLVM IR code and identify:
1. What algorithm or computation it represents
2. Its computational complexity (Big-O)
3. Potential performance bottlenecks

LLVM IR:
```
{llvm_ir}
```

Original Python function signature: {signature}

Provide a concise analysis in the following format:
ALGORITHM: [name of algorithm or computation pattern]
COMPLEXITY: [Big-O complexity]
BOTTLENECKS: [list of performance bottlenecks]
OPTIMIZATION_POTENTIAL: [HIGH/MEDIUM/LOW]""")
])


# Strategy Selection Prompt
STRATEGY_SELECTION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE),
    HumanMessagePromptTemplate.from_template("""Based on the analysis of the code:

Algorithm: {algorithm}
Complexity: {complexity}
Bottlenecks: {bottlenecks}

And the available optimization patterns:
{available_strategies}

Select the most appropriate optimization strategy. Consider:
1. Expected performance improvement
2. Safety of the transformation
3. Applicability to integer arithmetic

Respond with:
SELECTED_STRATEGY: [strategy name]
REASONING: [why this strategy is best]
EXPECTED_SPEEDUP: [estimated speedup factor, e.g., "2x-4x"]""")
])


# Code Generation Prompt
CODE_GENERATION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE),
    HumanMessagePromptTemplate.from_template("""Transform the following LLVM IR using the specified optimization strategy.

ORIGINAL LLVM IR:
```
{original_ir}
```

OPTIMIZATION STRATEGY: {strategy_name}
STRATEGY DESCRIPTION: {strategy_description}

RULES:
1. Output ONLY valid LLVM IR code
2. Maintain the same function signature
3. Ensure functional equivalence for all possible inputs
4. Use only integer operations (no floating point)
5. Do not add new function calls or external dependencies

Provide the optimized LLVM IR:""")
])


# CEGAR Refinement Prompt (used when verification fails)
CEGAR_REFINEMENT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE),
    HumanMessagePromptTemplate.from_template("""Your previous optimization was INCORRECT. The formal verifier found a bug.

ORIGINAL LLVM IR:
```
{original_ir}
```

YOUR PREVIOUS OPTIMIZATION:
```
{failed_optimization}
```

COUNTER-EXAMPLE (input that breaks equivalence):
{counter_example}

The verifier reported: {error_message}

Analyze your mistake and provide a CORRECTED optimization that:
1. Fixes the specific bug identified
2. Maintains functional equivalence for ALL inputs
3. Still improves performance where possible

Provide the corrected LLVM IR:""")
])
