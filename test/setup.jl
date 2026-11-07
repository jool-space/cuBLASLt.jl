# Shared bindings loaded into every test worker via runtests.jl's `init_code`.
using cuBLASLt
using cuBLASLt: MatmulPlan, plan_matmul, matmul!, matmul_supported, plan_candidates
using BitPacking
using CUDACore
using cuBLAS: CUBLASError
using LinearAlgebra
using Microfloats
using NNlib: NNlib
using Random
using Test

# CPU reference for D = α ⋅ op(A) ⋅ op(B) + β ⋅ C, in Float64
op_ref(A, trans::Char) = trans == 'N' ? A : trans == 'T' ? transpose(A) : A'
matmul_ref(A, B, transA, transB, α, β, C) =
    α .* (Float64.(op_ref(A, transA)) * Float64.(op_ref(B, transB))) .+ β .* Float64.(C)

# 0-dimensional device scalar for pointer_mode = :device
device_scalar(x::T) where {T} = fill!(CuArray{T}(undef), x)

# Architecture gates for narrow-type support (per the cuBLASLt docs):
#   CC ≥ 8.9         — FP8 with tensor-wide scaling (implicit or :scalar_f32)
#   9.0 ≤ CC < 10.0  — the Hopper f32 schemes: :outer_vec_f32, :vec128_f32,
#                      :blk128x128_f32
#   CC ≥ 10.0        — :vec32_ue8m0 MXFP8 and :vec16_ue4m3 NVFP4 block scaling
#   10.0 ≤ CC < 12.0 — :per_batch_scalar_f32 (claimed; support undocumented,
#                      refused on sm_89 and sm_121)
#   10.0 ≤ CC < 11.0 — the experimental MN×K4 modes :vec32_mn_k4_ue8m0 and
#                      :vec128_mn_k4_ue8m0 (cuBLASLt ≥ 13.4)
# Tests skip outside these ranges and must plan + pass inside them.
const CC = CUDACore.capability(CUDACore.device())

# Build a plan, or return `nothing` when cuBLASLt has no algorithm for the
# configuration on this GPU (arch-dependent compute-type support). The library
# reports this either as an empty heuristic result (our `UnsupportedConfigError`)
# or as CUBLAS_STATUS_NOT_SUPPORTED from the heuristic call itself. Narrow-type
# tests must not use this — they gate on `CC` and require the plan to build.
function try_plan(; kwargs...)
    try
        return MatmulPlan(; kwargs...)
    catch err
        err isa cuBLASLt.UnsupportedConfigError && return nothing
        err isa CUBLASError && return nothing
        rethrow()
    end
end
