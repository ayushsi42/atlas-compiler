<div align="center">

# ⚡ AOT-GPT

### Verifiable Neural JIT Compiler

**AI-optimized Python with formal verification**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Lift → Optimize → Verify → Compile → Execute*

</div>

---

## Overview

AOT-GPT transforms Python functions into optimized native machine code using AI, with **mathematical proof of correctness** via Z3 theorem prover.

```python
import aot_gpt

@aot_gpt.jit
def double_add(x, y):
    return (x * 2) + (y * 2)

result = double_add(5, 10)  # Optimized & verified!
```

### Key Features

| Feature | Description |
|---------|-------------|
| 🔄 **Lifter** | Python → LLVM IR via Numba type inference |
| 🧠 **Neural Core** | LLM-powered optimization with multiple strategies |
| ⚖️ **Tribunal** | Z3-based formal verification of equivalence |
| ⚡ **Executor** | LLVM MCJIT native code compilation |
| 🔁 **CEGAR** | Counter-example guided refinement loop |

---

## Installation

```bash
# Clone and install
git clone https://github.com/yourusername/atlas-compiler.git
cd atlas-compiler
pip install -e ".[dev]"

# Set API key
export OPENAI_API_KEY="your-key"
```

---

## Quick Start

### Using the Decorator

```python
import aot_gpt

@aot_gpt.jit
def multiply_sum(x, y):
    return (x * 4) + (y * 4)

# Automatically:
# 1. Lifted to LLVM IR
# 2. Optimized (mul → shift)
# 3. Verified with Z3
# 4. Compiled to native code

result = multiply_sum(5, 10)  # 60
```

### Z3 Verification Demo

```python
from aot_gpt.verifier import prove_equivalence

def original(x, y):
    return (x * 2) + (y * 2)

def buggy(x, y):
    return (x << 1)  # BUG: Forgot y!

result = prove_equivalence(original, buggy)
print(result.message)
# Output: BUG FOUND: Functions differ at inputs {x: 0, y: 1}
```

---

## Architecture

```
                    ┌──────────────────┐
                    │   Python Code    │
                    │   @aot_gpt.jit   │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │      🔄 THE LIFTER          │
              │   Numba → LLVM IR           │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     🧠 NEURAL CORE          │
              │   LLM Optimization          │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     ⚖️ THE TRIBUNAL         │
              │   Z3 Verification           │
              └──────────────┬──────────────┘
                             │
                      ┌──────┴──────┐
                      │  Verified?  │
                      └──────┬──────┘
                        Yes / \ No
                       ┌───┘   └───┐
                       │     🔁 CEGAR
                       │     Refine
                       │       │
              ┌────────▼───────▼────────┐
              │     ⚡ THE EXECUTOR      │
              │   MCJIT → Native Code   │
              └─────────────────────────┘
```

---

## Frontend

A premium React dashboard is included for visualization:

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

**Features:**
- Real-time compilation pipeline visualization
- Animated verification logs
- Z3 demo with bug detection
- Dark theme with glassmorphism UI

---

## Configuration

```python
from aot_gpt.config import configure

configure(
    model="gpt-4o",        # LLM model
    max_iterations=3,       # CEGAR retries
    safe_fallback=True,     # Fallback on failure
    bit_width=32,           # Verification bits
    opt_level=3,            # LLVM opt level
)
```

**Environment Variables:**
```bash
export OPENAI_API_KEY="your-key"        # Required
export REDIS_URL="redis://localhost"    # Optional (caching)
```

---

## Testing

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/test_verifier.py -v
pytest tests/test_lifter.py -v
```

**Current Status:** 21/21 tests passing ✅

---

## Optimization Strategies

| Strategy | Description | Speedup |
|----------|-------------|---------|
| Multiply-to-Shift | `x * 2` → `x << 1` | 1.5-2x |
| Divide-to-Shift | `x / 4` → `x >> 2` | 2-4x |
| Loop Unroll 4x | Reduce loop overhead | 2-4x |
| Distributive | `a*c + b*c` → `(a+b)*c` | 1.2-1.5x |
| SIMD Vector | Vectorize to 128-bit | 3-4x |

---

## Limitations (MVP)

This MVP focuses on **numerical computation**:

| Supported | Not Supported |
|-----------|---------------|
| ✅ Integer arithmetic | ❌ Strings |
| ✅ Bitwise operations | ❌ File I/O |
| ✅ Simple loops | ❌ Complex objects |

---

## Tech Stack

- **[Numba](https://numba.pydata.org/)** — Type inference & IR generation
- **[llvmlite](https://github.com/numba/llvmlite)** — LLVM Python bindings
- **[Z3](https://github.com/Z3Prover/z3)** — SMT solver for verification
- **[LangChain](https://langchain.com/)** — LLM orchestration
- **[React + Vite](https://vitejs.dev/)** — Frontend dashboard

---

## License

MIT License — See [LICENSE](LICENSE) for details.
