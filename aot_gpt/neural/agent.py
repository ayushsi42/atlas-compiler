"""
Neural Optimizer Agent - Multi-step AI optimization pipeline.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from aot_gpt.config import get_config
from aot_gpt.neural.prompts import (
    INTENT_ANALYSIS_TEMPLATE,
    STRATEGY_SELECTION_TEMPLATE,
    CODE_GENERATION_TEMPLATE,
    CEGAR_REFINEMENT_TEMPLATE,
)
from aot_gpt.neural.strategies import (
    OptimizationStrategy,
    get_strategy,
    get_all_strategies,
    format_strategies_for_prompt,
)


@dataclass
class IntentAnalysis:
    """Result of intent analysis."""
    algorithm: str
    complexity: str
    bottlenecks: str
    optimization_potential: str
    raw_response: str


@dataclass
class StrategySelection:
    """Result of strategy selection."""
    strategy: OptimizationStrategy
    reasoning: str
    expected_speedup: str
    raw_response: str


@dataclass
class OptimizationResult:
    """Final result of the optimization process."""
    original_ir: str
    optimized_ir: str
    strategy_used: str
    intent_analysis: IntentAnalysis
    iterations: int
    success: bool
    error_message: Optional[str] = None


class NeuralOptimizer:
    """
    Multi-step neural optimizer agent.
    
    Pipeline:
    1. Analyze intent (what algorithm is this?)
    2. Select optimization strategy
    3. Generate optimized code
    """
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        config = get_config()
        self.model = model or config.neural.model
        self.api_key = api_key or config.neural.api_key
        self.temperature = config.neural.temperature
        
        self._llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
        )
        self._parser = StrOutputParser()
    
    def analyze_intent(self, llvm_ir: str, signature: str) -> IntentAnalysis:
        """
        Step 1: Analyze the intent of the code.
        
        Args:
            llvm_ir: The LLVM IR to analyze
            signature: The function signature string
            
        Returns:
            IntentAnalysis with algorithm identification
        """
        chain = INTENT_ANALYSIS_TEMPLATE | self._llm | self._parser
        
        response = chain.invoke({
            "llvm_ir": llvm_ir,
            "signature": signature,
        })
        
        # Parse structured response
        algorithm = self._extract_field(response, "ALGORITHM", "Unknown")
        complexity = self._extract_field(response, "COMPLEXITY", "Unknown")
        bottlenecks = self._extract_field(response, "BOTTLENECKS", "None identified")
        potential = self._extract_field(response, "OPTIMIZATION_POTENTIAL", "MEDIUM")
        
        return IntentAnalysis(
            algorithm=algorithm,
            complexity=complexity,
            bottlenecks=bottlenecks,
            optimization_potential=potential,
            raw_response=response,
        )
    
    def select_strategy(self, analysis: IntentAnalysis) -> StrategySelection:
        """
        Step 2: Select the best optimization strategy.
        
        Args:
            analysis: The intent analysis result
            
        Returns:
            StrategySelection with chosen strategy
        """
        chain = STRATEGY_SELECTION_TEMPLATE | self._llm | self._parser
        
        response = chain.invoke({
            "algorithm": analysis.algorithm,
            "complexity": analysis.complexity,
            "bottlenecks": analysis.bottlenecks,
            "available_strategies": format_strategies_for_prompt(),
        })
        
        # Parse response
        strategy_name = self._extract_field(response, "SELECTED_STRATEGY", "Multiply-to-Shift")
        reasoning = self._extract_field(response, "REASONING", "Default selection")
        speedup = self._extract_field(response, "EXPECTED_SPEEDUP", "1.5x")
        
        # Get the actual strategy object
        strategy = get_strategy(strategy_name)
        if strategy is None:
            # Fallback to first strategy
            strategy = get_all_strategies()[0]
        
        return StrategySelection(
            strategy=strategy,
            reasoning=reasoning,
            expected_speedup=speedup,
            raw_response=response,
        )
    
    def generate_optimized(
        self,
        original_ir: str,
        strategy: OptimizationStrategy
    ) -> str:
        """
        Step 3: Generate optimized LLVM IR.
        
        Args:
            original_ir: The original LLVM IR
            strategy: The optimization strategy to apply
            
        Returns:
            Optimized LLVM IR string
        """
        chain = CODE_GENERATION_TEMPLATE | self._llm | self._parser
        
        response = chain.invoke({
            "original_ir": original_ir,
            "strategy_name": strategy.name,
            "strategy_description": strategy.description,
        })
        
        # Extract LLVM IR code block
        optimized_ir = self._extract_code_block(response)
        return optimized_ir if optimized_ir else response
    
    def refine_with_counterexample(
        self,
        original_ir: str,
        failed_optimization: str,
        counter_example: str,
        error_message: str
    ) -> str:
        """
        CEGAR refinement: fix optimization based on counter-example.
        
        Args:
            original_ir: The original LLVM IR
            failed_optimization: The incorrect optimized IR
            counter_example: Input values that broke equivalence
            error_message: Verifier error message
            
        Returns:
            Corrected LLVM IR string
        """
        chain = CEGAR_REFINEMENT_TEMPLATE | self._llm | self._parser
        
        response = chain.invoke({
            "original_ir": original_ir,
            "failed_optimization": failed_optimization,
            "counter_example": counter_example,
            "error_message": error_message,
        })
        
        optimized_ir = self._extract_code_block(response)
        return optimized_ir if optimized_ir else response
    
    def optimize(self, llvm_ir: str, signature: str) -> OptimizationResult:
        """
        Run the full optimization pipeline.
        
        Args:
            llvm_ir: The LLVM IR to optimize
            signature: The function signature
            
        Returns:
            OptimizationResult with all details
        """
        try:
            # Step 1: Analyze intent
            intent = self.analyze_intent(llvm_ir, signature)
            
            # Check if optimization is worth it
            if intent.optimization_potential == "LOW":
                return OptimizationResult(
                    original_ir=llvm_ir,
                    optimized_ir=llvm_ir,  # Return unchanged
                    strategy_used="None (low potential)",
                    intent_analysis=intent,
                    iterations=0,
                    success=True,
                )
            
            # Step 2: Select strategy
            selection = self.select_strategy(intent)
            
            # Step 3: Generate optimized code
            optimized_ir = self.generate_optimized(llvm_ir, selection.strategy)
            
            return OptimizationResult(
                original_ir=llvm_ir,
                optimized_ir=optimized_ir,
                strategy_used=selection.strategy.name,
                intent_analysis=intent,
                iterations=1,
                success=True,
            )
            
        except Exception as e:
            # Return original on error
            return OptimizationResult(
                original_ir=llvm_ir,
                optimized_ir=llvm_ir,
                strategy_used="None (error)",
                intent_analysis=IntentAnalysis(
                    algorithm="Unknown",
                    complexity="Unknown",
                    bottlenecks="Unknown",
                    optimization_potential="Unknown",
                    raw_response="",
                ),
                iterations=0,
                success=False,
                error_message=str(e),
            )
    
    def _extract_field(self, text: str, field: str, default: str) -> str:
        """Extract a field value from structured LLM response."""
        pattern = rf"{field}:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else default
    
    def _extract_code_block(self, text: str) -> Optional[str]:
        """Extract code from markdown code block."""
        # Try to find LLVM IR code block
        pattern = r"```(?:llvm)?\s*\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no code block, look for lines starting with LLVM keywords
        lines = text.split('\n')
        ir_lines = []
        in_ir = False
        for line in lines:
            if line.strip().startswith(('define', ';', '@', '%', 'declare')):
                in_ir = True
            if in_ir:
                ir_lines.append(line)
                if line.strip() == '}':
                    break
        
        return '\n'.join(ir_lines) if ir_lines else None
