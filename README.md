# cuBLASLt

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.jool.space/cuBLASLt.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.jool.space/cuBLASLt.jl/dev/)
[![Build Status](https://github.com/jool-space/cuBLASLt.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/cuBLASLt.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jool-space/cuBLASLt.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jool-space/cuBLASLt.jl)

A developer-facing layer over the `cuBLAS` package's `libcublasLt` bindings.
Matrix multiplication, with every knob exposed as arguments.
Per-call compute types (TF32 without `math_mode!`), block-scaled narrow
types (MXFP8/MXFP4/NVFP4) on input *and* output, fused epilogues
(bias/ReLU/GELU and their gradients), strided batching, and plan caching.

```julia
using cuBLASLt: MatmulPlan, plan_matmul, matmul!, plan_candidates

# planless: derives a plan from the arguments and hits the plan cache
matmul!(D, A, B; compute = :tf32)
matmul!(D, transpose(A), B)         # orientation from Transpose/Adjoint wrappers
matmul!(D3, PermutedDimsArray(A3, (2, 1, 3)), B3)   # batched transpose, 3-d arrays
matmul!(D, A, B; activation = :gelu, bias = b, aux = pre)   # fused epilogue
matmul!(dX, dY, transpose(W); epilogue = :dgelu_bgrad, aux = pre, bgrad = db)

# candidate plans for user-owned autotuning: benchmark, keep the winner
plans = plan_candidates(D, A, B; count = 8)

# planned: build once from prototype arguments — the same signature the plan
# is applied with — then apply; plans are callable
plan = plan_matmul(D, transpose(A), B; workspace = ws)
plan(D, transpose(A), B; workspace = ws)

# or fully explicit, no arrays needed; everything that affects algorithm
# selection lives in the plan, everything resolved at execution time is an
# apply argument
plan = MatmulPlan(; M, N, K, typeA = Float8_E4M3FN, typeB = Float8_E4M3FN,
                  typeD = Float32, transA = 'T', lda = K, ldb = K,
                  scale_modeA = :vec32_ue8m0, scale_modeB = :vec32_ue8m0)
plan(D, A, B; scaleA = sA, scaleB = sB)
```
