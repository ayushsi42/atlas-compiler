"""
Atlas Benchmark Runner

Runs benchmarks and measures optimization speedups.

Usage:
    python -m benchmarks.runner
    python -m benchmarks.runner --output results/report.json
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Any
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    description: str
    python_time_us: float  # Microseconds
    compiled_time_us: float
    speedup: float
    verified: bool
    strategy: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    timestamp: str
    total_benchmarks: int
    passed: int
    failed: int
    avg_speedup: float
    results: List[BenchmarkResult]


def benchmark_function(func: Callable, args: tuple, iterations: int = 10000) -> float:
    """
    Benchmark a function and return average execution time in microseconds.
    """
    # Warmup
    for _ in range(100):
        func(*args)
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    elapsed = time.perf_counter() - start
    
    return (elapsed / iterations) * 1_000_000  # Convert to microseconds


def run_benchmark(
    name: str,
    func: Callable,
    args: tuple,
    description: str = "",
    use_atlas: bool = True
) -> BenchmarkResult:
    """
    Run a single benchmark comparing Python vs Atlas-compiled.
    """
    # Measure Python baseline
    python_time = benchmark_function(func, args)
    
    if use_atlas:
        try:
            import atlas
            
            # Create JIT-compiled version
            @atlas.jit
            def compiled_func(*a):
                return func(*a)
            
            # Trigger compilation
            _ = compiled_func(*args)
            
            # Get stats
            stats = atlas.get_stats(compiled_func)
            
            # Measure compiled version
            compiled_time = benchmark_function(compiled_func, args)
            
            speedup = python_time / compiled_time if compiled_time > 0 else 1.0
            
            return BenchmarkResult(
                name=name,
                description=description,
                python_time_us=round(python_time, 3),
                compiled_time_us=round(compiled_time, 3),
                speedup=round(speedup, 2),
                verified=stats.verified if stats else False,
                strategy=stats.optimization_strategy if stats else None,
            )
            
        except Exception as e:
            return BenchmarkResult(
                name=name,
                description=description,
                python_time_us=round(python_time, 3),
                compiled_time_us=python_time,
                speedup=1.0,
                verified=False,
                error=str(e),
            )
    else:
        return BenchmarkResult(
            name=name,
            description=description,
            python_time_us=round(python_time, 3),
            compiled_time_us=python_time,
            speedup=1.0,
            verified=False,
        )


def get_all_benchmarks() -> List[Dict[str, Any]]:
    """Get all benchmark definitions."""
    from benchmarks.functions.numerical import (
        double_add, triple_multiply, dot_product, polynomial_eval,
        power_of_two_mult, divide_by_power_of_two, modulo_power_of_two,
    )
    from benchmarks.functions.reductions import (
        array_sum, array_min, array_max, abs_value, clamp,
    )
    from benchmarks.functions.bitwise import (
        popcount, is_power_of_two, bit_parity, next_power_of_two, leading_zeros,
    )
    
    return [
        # Numerical
        {"name": "double_add", "func": double_add, "args": (42, 17), 
         "desc": "(x*2) + (y*2) -> (x+y)<<1"},
        {"name": "triple_multiply", "func": triple_multiply, "args": (3, 5, 7),
         "desc": "x * y * z"},
        {"name": "dot_product", "func": dot_product, "args": (1, 2, 3, 4),
         "desc": "a*c + b*d (2-elem dot)"},
        {"name": "polynomial_eval", "func": polynomial_eval, "args": (5, 2, 3, 1),
         "desc": "a*x^2 + b*x + c"},
        {"name": "power_of_two_mult", "func": power_of_two_mult, "args": (42, 3),
         "desc": "x * 8 -> x << 3"},
        {"name": "divide_by_power_of_two", "func": divide_by_power_of_two, "args": (100,),
         "desc": "x // 4 -> x >> 2"},
        {"name": "modulo_power_of_two", "func": modulo_power_of_two, "args": (100,),
         "desc": "x % 8 -> x & 7"},
        
        # Reductions
        {"name": "array_sum", "func": array_sum, "args": (100,),
         "desc": "Sum 1..n"},
        {"name": "array_min", "func": array_min, "args": (5, 3, 7),
         "desc": "min(a, b, c)"},
        {"name": "array_max", "func": array_max, "args": (5, 3, 7),
         "desc": "max(a, b, c)"},
        {"name": "abs_value", "func": abs_value, "args": (-42,),
         "desc": "abs(x)"},
        {"name": "clamp", "func": clamp, "args": (50, 0, 100),
         "desc": "clamp(x, lo, hi)"},
        
        # Bitwise
        {"name": "popcount", "func": popcount, "args": (0b10110101,),
         "desc": "Count 1 bits"},
        {"name": "is_power_of_two", "func": is_power_of_two, "args": (256,),
         "desc": "Check power of 2"},
        {"name": "bit_parity", "func": bit_parity, "args": (0b10110101,),
         "desc": "XOR all bits"},
        {"name": "next_power_of_two", "func": next_power_of_two, "args": (100,),
         "desc": "Round to next 2^n"},
        {"name": "leading_zeros", "func": leading_zeros, "args": (256,),
         "desc": "Count leading zeros"},
    ]


def run_all_benchmarks(use_atlas: bool = True) -> BenchmarkSuite:
    """Run all benchmarks and return results."""
    from datetime import datetime
    
    benchmarks = get_all_benchmarks()
    results = []
    
    for bench in benchmarks:
        result = run_benchmark(
            name=bench["name"],
            func=bench["func"],
            args=bench["args"],
            description=bench.get("desc", ""),
            use_atlas=use_atlas,
        )
        results.append(result)
        print(f"  {result.name}: {result.speedup:.2f}x speedup")
    
    passed = sum(1 for r in results if r.error is None)
    failed = len(results) - passed
    avg_speedup = sum(r.speedup for r in results) / len(results) if results else 1.0
    
    return BenchmarkSuite(
        timestamp=datetime.now().isoformat(),
        total_benchmarks=len(results),
        passed=passed,
        failed=failed,
        avg_speedup=round(avg_speedup, 2),
        results=results,
    )


def generate_markdown_report(suite: BenchmarkSuite, output_path: Path) -> None:
    """Generate a Markdown report from benchmark results."""
    lines = [
        "# Atlas Benchmark Report",
        "",
        f"**Date**: {suite.timestamp}",
        f"**Total Benchmarks**: {suite.total_benchmarks}",
        f"**Passed**: {suite.passed} | **Failed**: {suite.failed}",
        f"**Average Speedup**: {suite.avg_speedup}x",
        "",
        "## Results",
        "",
        "| Benchmark | Description | Python (µs) | Compiled (µs) | Speedup | Verified |",
        "|-----------|-------------|-------------|---------------|---------|----------|",
    ]
    
    for r in suite.results:
        verified = "✅" if r.verified else "❌"
        lines.append(
            f"| {r.name} | {r.description} | {r.python_time_us} | "
            f"{r.compiled_time_us} | {r.speedup}x | {verified} |"
        )
    
    lines.extend([
        "",
        "## Summary",
        "",
        f"Atlas achieved an average **{suite.avg_speedup}x** speedup across all benchmarks.",
    ])
    
    output_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run Atlas benchmarks")
    parser.add_argument("--output", "-o", type=str, default="benchmarks/results/report.json",
                        help="Output path for JSON report")
    parser.add_argument("--no-atlas", action="store_true",
                        help="Run without Atlas compilation (baseline only)")
    parser.add_argument("--markdown", "-m", action="store_true",
                        help="Also generate Markdown report")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Atlas Benchmark Suite")
    print("=" * 50)
    print()
    
    suite = run_all_benchmarks(use_atlas=not args.no_atlas)
    
    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(asdict(suite), f, indent=2)
    
    print()
    print(f"Results saved to: {output_path}")
    
    # Generate Markdown if requested
    if args.markdown:
        md_path = output_path.with_suffix(".md")
        generate_markdown_report(suite, md_path)
        print(f"Markdown report: {md_path}")
    
    print()
    print("=" * 50)
    print(f"Average Speedup: {suite.avg_speedup}x")
    print(f"Passed: {suite.passed}/{suite.total_benchmarks}")
    print("=" * 50)


if __name__ == "__main__":
    main()
