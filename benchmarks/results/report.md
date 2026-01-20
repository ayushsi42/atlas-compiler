# Atlas Benchmark Report

**Date**: 2026-01-17T12:48:55.232029
**Total Benchmarks**: 17
**Passed**: 17 | **Failed**: 0
**Average Speedup**: 3.03x

## Results

| Benchmark | Description | Python (µs) | Compiled (µs) | Speedup | Strategy | Verified |
|-----------|-------------|-------------|---------------|---------|----------|----------|
| double_add | (x*2) + (y*2) -> (x+y)<<1 | 0.051 | 0.125 | 0.41x | GPT: Multiply-to-Shift | ✅ |
| triple_multiply | x * y * z | 0.06 | 0.269 | 0.22x | GPT: Multiply-to-Shift | ✅ |
| dot_product | a*c + b*d (2-elem dot) | 0.16 | 0.307 | 0.52x | GPT: Distributive-Factoring | ✅ |
| polynomial_eval | a*x^2 + b*x + c | 0.184 | 0.275 | 0.67x | GPT: Multiply-to-Shift | ✅ |
| power_of_two_mult | x * 8 -> x << 3 | 0.152 | 0.179 | 0.85x | GPT: Multiply-to-Shift | ✅ |
| divide_by_power_of_two | x // 4 -> x >> 2 | 0.241 | 0.226 | 1.06x | GPT: Divide-to-Shift | ❌ |
| modulo_power_of_two | x % 8 -> x & 7 | 0.126 | 0.234 | 0.54x | GPT: Modulo-to-Bitwise | ✅ |
| array_sum | Sum 1..n | 3.986 | 0.249 | 16.04x | GPT: Constant-Propagation | ❌ |
| array_min | min(a, b, c) | 0.123 | 0.318 | 0.39x | GPT: Branch-to-Select | ❌ |
| array_max | max(a, b, c) | 0.143 | 0.142 | 1.01x | GPT: Branch-to-Select | ❌ |
| abs_value | abs(x) | 0.053 | 0.208 | 0.26x | GPT: Branch-to-Select | ❌ |
| clamp | clamp(x, lo, hi) | 0.155 | 0.354 | 0.44x | GPT: Branch-to-Select | ❌ |
| popcount | Count 1 bits | 2.708 | 0.191 | 14.17x | GPT: SIMD-Vector-4 | ✅ |
| is_power_of_two | Check power of 2 | 0.136 | 0.287 | 0.47x | GPT: Branch-to-Select | ❌ |
| bit_parity | XOR all bits | 0.487 | 0.239 | 2.04x | GPT: Constant-Propagation | ✅ |
| next_power_of_two | Round to next 2^n | 0.527 | 0.275 | 1.92x | GPT: Multiply-to-Shift | ✅ |
| leading_zeros | Count leading zeros | 0.872 | 0.083 | 10.53x | GPT: Branch-to-Select | ✅ |

## Summary

Atlas achieved an average **3.03x** speedup across all benchmarks.