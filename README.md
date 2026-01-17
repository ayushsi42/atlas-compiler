<div align="center">

# ⚡ Atlas Compiler

### Verifiable Neural JIT

**AI-optimized Python with formal verification**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A JIT compiler that uses GPT to optimize Python → LLVM IR, then formally verifies correctness with Z3 before compiling to native code.*

</div>

---

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Set API key
export OPENAI_API_KEY="your-key"

# Run backend
cd backend && python3 -m uvicorn server:app --reload --port 8000

# Run frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** → Click **Compile** → See real optimization

---

## How It Works

```
Python → Numba → LLVM IR → GPT Optimization → Z3 Verification → Native Code
```

| Stage | What Happens |
|-------|-------------|
| **Lift** | Python → LLVM IR via Numba |
| **Optimize** | GPT analyzes and suggests optimizations |
| **Verify** | Z3 proves mathematical equivalence |
| **Compile** | LLVM MCJIT → native machine code |

---

## Example

```python
import atlas

@atlas.jit
def double_add(x, y):
    return (x * 2) + (y * 2)

# Optimized to: (x + y) << 1
# Verified correct by Z3
# Compiled to native code
```

**Result:** `(x*2) + (y*2)` → `(x+y) << 1` (shift instead of multiply)

---

## Features

### Type Support
- Scalars: `int32`, `int64`, `float32`, `float64`, `boolean`
- Arrays: 1D and 2D NumPy arrays

### Optimization Strategies (11 total)
- Strength reduction: multiply-to-shift, divide-to-shift, modulo-to-bitwise
- Loop optimizations: unrolling, invariant hoisting
- Advanced: FMA fusion, branch-to-select, SIMD vectorization

### Module Compilation
```python
import atlas

compiled = atlas.module(func1, func2, func3)
result = compiled['func1'](x, y)
```

### Safety Verification
- Equivalence checking
- Overflow detection
- Division-by-zero checks
- Array bounds verification
- Termination analysis

---

## Benchmarks

```bash
python3 -m benchmarks.runner --markdown
```

17 benchmarks covering numerical, reduction, and bitwise operations.

---

## Tech Stack

- **Numba** — Type inference & IR generation
- **llvmlite** — LLVM Python bindings
- **Z3** — SMT solver for verification
- **LangChain + OpenAI** — LLM optimization
- **FastAPI** — Backend API
- **React + Vite** — Frontend

---

## Testing

```bash
pytest tests/ -v  # 21/21 pass
```

---

## FAQ

### How is this different from prompting an LLM to optimize my code?

When you ask ChatGPT to optimize code, you get a "trust me" answer—it might be faster, but it might also have bugs. Atlas is different:

| Aspect | LLM-only | Atlas |
|--------|----------|-------|
| **Correctness** | Hope it's right | Z3 *proves* mathematical equivalence |
| **Bugs** | Possible | Rejected if not provably correct |
| **Counterexamples** | None | Shows exact inputs where optimization fails |
| **Output** | Python code | Native machine code |

The LLM is **creative** (suggests optimizations), but Z3 is the **judge** (rejects incorrect ones).

### Doesn't the LLM call slow down compilation?

Yes, but the speedup is in **runtime**, not compile time:

| Phase | Python | Atlas |
|-------|--------|-------|
| Compile | 0s | ~2s (LLM + verify) |
| Run 1M times | 500ms | 50ms |
| **Total** | 500ms | 2050ms for 1M calls |

After ~5M calls, Atlas wins. For hot loops in ML/simulations, the compile cost is negligible.

### Can't I do formal verification with just Python + Z3?

Yes, for simple cases! A basic equivalence check is ~20 lines:

```python
from z3 import *
x, y = BitVecs('x y', 32)
s = Solver()
s.add((x*2)+(y*2) != (x+y)<<1)
print("Equivalent" if s.check() == unsat else "Bug found")
```

But Atlas adds:
- **LLM-generated optimizations** (you don't have to write them)
- **CEGAR loop** (auto-refines on verification failure)
- **Native compilation** (LLVM, not Python bytecode)
- **11 optimization strategies** for the LLM to choose from

A simple Z3 script is a proof-of-concept; Atlas is an end-to-end compiler.

---

## License

MIT
