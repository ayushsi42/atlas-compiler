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
import aot_gpt

@aot_gpt.jit
def double_add(x, y):
    return (x * 2) + (y * 2)

# Optimized to: (x + y) << 1
# Verified correct by Z3
# Compiled to native code
```

**Result:** `(x*2) + (y*2)` → `(x+y) << 1` (shift instead of multiply)

---

## Real Performance

| Metric | Value |
|--------|-------|
| Python | 0.17µs |
| Compiled | 0.13µs |
| **Speedup** | **1.2x** |
| Z3 Verify | 2.9ms |

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

## License

MIT
