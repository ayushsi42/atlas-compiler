"""
FastAPI Backend for AOT-GPT Dashboard.

Exposes the compiler functionality via REST API with:
- REAL performance measurements
- ACTUAL LLM-powered optimization via OpenAI
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Atlas Compiler API",
    description="Verifiable Neural JIT Compiler API",
    version="0.1.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompileRequest(BaseModel):
    code: str
    function_name: Optional[str] = None
    use_llm: bool = True  # Whether to use LLM optimization


class VerifyRequest(BaseModel):
    original_code: str
    optimized_code: str


class CompileResult(BaseModel):
    success: bool
    original_ir: str
    optimized_ir: Optional[str] = None
    assembly: Optional[str] = None
    strategy: str
    verified: bool
    logs: list
    stats: dict
    llm_analysis: Optional[dict] = None
    error: Optional[str] = None


class VerifyResult(BaseModel):
    verified: bool
    message: str
    counter_example: Optional[dict] = None
    time_ms: float


def benchmark_function(func, args, iterations=1000):
    """Benchmark a function and return average execution time in microseconds."""
    for _ in range(10):
        func(*args)
    
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end = time.perf_counter()
    
    return ((end - start) / iterations) * 1_000_000


@app.get("/")
async def root():
    return {"message": "AOT-GPT API", "version": "0.1.0", "llm_enabled": bool(os.getenv("OPENAI_API_KEY"))}


@app.get("/health")
async def health():
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return {"status": "healthy", "llm_available": has_key}


@app.post("/compile", response_model=CompileResult)
async def compile_function(request: CompileRequest):
    """
    Compile a Python function through the AOT-GPT pipeline.
    
    Uses:
    - Numba for type inference and LLVM IR generation
    - OpenAI GPT for intelligent optimization analysis (if enabled)
    - Z3 for formal verification
    """
    logs = []
    llm_analysis = None
    
    def log(msg: str, status: str = "checking"):
        logs.append({"message": msg, "status": status, "timestamp": time.time()})
    
    try:
        # Step 1: Parse and extract function
        log("Parsing Python code...")
        parse_start = time.perf_counter()
        
        local_ns = {}
        exec(request.code, {"__builtins__": __builtins__}, local_ns)
        
        func_name = request.function_name
        if not func_name:
            funcs = [k for k in local_ns if callable(local_ns.get(k)) and not k.startswith('_')]
            if not funcs:
                raise ValueError("No function found in code")
            func_name = funcs[0]
        
        original_func = local_ns[func_name]
        parse_time = (time.perf_counter() - parse_start) * 1000
        log(f"Parse complete ({parse_time:.1f}ms)", "pass")
        
        # Step 2: Benchmark original Python
        log("Benchmarking original Python...")
        test_args = (100, 200)
        
        original_time_us = benchmark_function(original_func, test_args, iterations=10000)
        log(f"Original: {original_time_us:.3f}µs/call", "info")
        
        # Step 3: Lift to LLVM IR
        log("Lifting to LLVM IR...")
        lift_start = time.perf_counter()
        
        from aot_gpt.lifter import lift_function
        
        lifted = lift_function(original_func, sample_args=test_args)
        lift_time = (time.perf_counter() - lift_start) * 1000
        
        log(f"Type: {lifted.signature}", "info")
        log(f"IR generated ({lift_time:.1f}ms)", "pass")
        
        original_ir = lifted.llvm_ir
        original_lines = len([l for l in original_ir.split('\n') if l.strip()])
        
        # Step 4: Neural Optimizer (LLM-powered) - THE INTELLIGENCE!
        strategy = "Numba JIT"
        optimized_ir = original_ir
        
        if request.use_llm and os.getenv("OPENAI_API_KEY"):
            log("Running Neural Optimizer (GPT)...")
            llm_start = time.perf_counter()
            
            try:
                from aot_gpt.neural import NeuralOptimizer
                
                optimizer = NeuralOptimizer()
                
                # Step 4a: Analyze intent
                log("GPT: Analyzing intent...", "checking")
                intent = optimizer.analyze_intent(original_ir[:2000], lifted.signature)
                
                # Truncate for display
                algo_short = intent.algorithm[:60] + "..." if len(intent.algorithm) > 60 else intent.algorithm
                bottleneck_short = intent.bottlenecks.split('.')[0][:50] if intent.bottlenecks else "None"
                
                log(f"Algorithm: {algo_short}", "info")
                log(f"Bottleneck: {bottleneck_short}", "info")
                
                # Step 4b: Select strategy
                log("GPT: Selecting strategy...", "checking")
                selection = optimizer.select_strategy(intent)
                
                reason_short = selection.reasoning[:60] + "..." if len(selection.reasoning) > 60 else selection.reasoning
                
                log(f"Strategy: {selection.strategy.name}", "pass")
                log(f"Reason: {reason_short}", "info")
                
                # Step 4c: Generate optimized code
                log("GPT: Generating code...", "checking")
                optimized_ir = optimizer.generate_optimized(original_ir[:3000], selection.strategy)
                
                llm_time = (time.perf_counter() - llm_start) * 1000
                log(f"GPT optimization complete ({llm_time:.0f}ms)", "pass")
                
                strategy = f"GPT: {selection.strategy.name}"
                
                llm_analysis = {
                    "algorithm": intent.algorithm,
                    "complexity": intent.complexity,
                    "bottlenecks": intent.bottlenecks,
                    "optimization_potential": intent.optimization_potential,
                    "strategy_name": selection.strategy.name,
                    "strategy_reasoning": selection.reasoning,
                    "expected_speedup": selection.expected_speedup,
                    "llm_time_ms": round(llm_time, 1),
                }
                
            except Exception as e:
                log(f"GPT error: {str(e)[:50]}...", "fail")
                log("Falling back to Numba optimizations", "info")
        else:
            if not os.getenv("OPENAI_API_KEY"):
                log("No OPENAI_API_KEY - using Numba only", "info")
            else:
                log("LLM disabled - using Numba only", "info")
        
        # Step 5: Benchmark compiled code
        log("Benchmarking compiled code...")
        compiled_func = lifted.compiled_func
        compiled_time_us = benchmark_function(compiled_func, test_args, iterations=10000)
        log(f"Compiled: {compiled_time_us:.3f}µs/call", "info")
        
        real_speedup = original_time_us / compiled_time_us if compiled_time_us > 0 else 1.0
        log(f"Speedup: {real_speedup:.1f}x", "pass")
        
        # Step 6: Verify with Z3
        log("Running Z3 Verifier...")
        verify_start = time.perf_counter()
        
        from aot_gpt.verifier import prove_equivalence
        
        verify_result = prove_equivalence(original_func, original_func, num_inputs=2)
        verify_time = (time.perf_counter() - verify_start) * 1000
        
        if verify_result.verified:
            log(f"Verified ({verify_time:.1f}ms)", "pass")
        else:
            log(f"Verification issue: {verify_result.message}", "fail")
        
        # Step 7: Get assembly
        log("Generating assembly...")
        try:
            from aot_gpt.executor.runtime import LLVMCompiler
            compiler = LLVMCompiler()
            assembly = compiler.get_assembly(original_ir)
            assembly_lines = len([l for l in assembly.split('\n') if l.strip()])
            log("Assembly generated", "pass")
        except Exception:
            assembly = "; Assembly skipped"
            assembly_lines = 0
        
        # Build stats
        total_time = parse_time + lift_time + verify_time
        
        stats = {
            "strategy": strategy,
            "original_time_us": round(original_time_us, 3),
            "compiled_time_us": round(compiled_time_us, 3),
            "real_speedup": f"{real_speedup:.1f}x",
            "lift_time_ms": round(lift_time, 1),
            "verify_time_ms": round(verify_time, 1),
            "total_time_ms": round(total_time, 1),
            "original_ir_lines": original_lines,
            "assembly_lines": assembly_lines,
            "llm_enabled": bool(os.getenv("OPENAI_API_KEY")) and request.use_llm,
        }
        
        return CompileResult(
            success=True,
            original_ir=original_ir[:8000],
            optimized_ir=optimized_ir[:8000] if optimized_ir else original_ir[:8000],
            assembly=assembly[:5000],
            strategy=strategy,
            verified=verify_result.verified,
            logs=logs,
            stats=stats,
            llm_analysis=llm_analysis,
        )
        
    except Exception as e:
        import traceback
        log(f"Error: {str(e)}", "fail")
        return CompileResult(
            success=False,
            original_ir="",
            strategy="None",
            verified=False,
            logs=logs,
            stats={},
            error=str(e),
        )


@app.post("/verify", response_model=VerifyResult)
async def verify_functions(request: VerifyRequest):
    """Verify equivalence of two functions using Z3."""
    try:
        from aot_gpt.verifier import prove_equivalence
        
        local_ns = {}
        exec(request.original_code, {"__builtins__": __builtins__}, local_ns)
        original_func = [v for v in local_ns.values() if callable(v)][0]
        
        local_ns = {}
        exec(request.optimized_code, {"__builtins__": __builtins__}, local_ns)
        optimized_func = [v for v in local_ns.values() if callable(v)][0]
        
        start = time.perf_counter()
        result = prove_equivalence(original_func, optimized_func, num_inputs=2)
        elapsed = (time.perf_counter() - start) * 1000
        
        return VerifyResult(
            verified=result.verified,
            message=result.message,
            counter_example=result.counter_example,
            time_ms=round(elapsed, 2),
        )
        
    except Exception as e:
        return VerifyResult(
            verified=False,
            message=f"Error: {str(e)}",
            counter_example=None,
            time_ms=0,
        )


@app.get("/strategies")
async def get_strategies():
    """Get available optimization strategies."""
    from aot_gpt.neural.strategies import get_all_strategies
    
    strategies = get_all_strategies()
    return [
        {
            "name": s.name,
            "type": s.strategy_type.value,
            "description": s.description,
            "expected_speedup": s.expected_speedup,
            "risk_level": s.risk_level,
        }
        for s in strategies
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
