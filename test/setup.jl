# Shared bindings loaded into every test worker via runtests.jl's `init_code`.
using cuBLASLt
using cuBLASLt: MatmulPlan, plan_matmul, matmul!, plan_candidates
using CUDACore
using CUDACore: cudaDataType
using cuBLAS: CUBLASError
using LinearAlgebra
using Microfloats
using NNlib: NNlib
using Random
using Test

# remove once https://github.com/JuliaGPU/CUDA.jl/pull/3180 gets merged
for (T, n) in (
    (:Float8_E4M3FN,  28),
    (:Float8_E5M2,    29),
    (:Float8_E8M0FNU, 30),
    (:Float6_E2M3FN,  31),
    (:Float6_E3M2FN,  32),
    (:Float4_E2M1FN,  33),
)
    @eval Base.convert(::Type{CUDACore.cudaDataType}, ::Type{$T}) =
        reinterpret(CUDACore.cudaDataType, Cint($n))
end

# CPU reference for D = α ⋅ op(A) ⋅ op(B) + β ⋅ C, in Float64
op_ref(A, trans::Char) = trans == 'N' ? A : trans == 'T' ? transpose(A) : A'
matmul_ref(A, B, transA, transB, α, β, C) =
    α .* (Float64.(op_ref(A, transA)) * Float64.(op_ref(B, transB))) .+ β .* Float64.(C)

# 0-dimensional device scalar for pointer_mode = :device
device_scalar(x::T) where {T} = fill!(CuArray{T}(undef), x)

# Architecture gates for narrow-type support (per the cuBLASLt docs):
#   CC ≥ 8.9  — FP8 with tensor-wide scaling (implicit or :scalar_f32)
#   CC ≥ 9.0  — :vec128_f32 / :blk128x128_f32 block scaling
#   CC ≥ 10.0 — :vec32_ue8m0 MXFP8 and :vec16_ue4m3 NVFP4 block scaling
# Tests skip below these thresholds and must plan + pass at or above them.
const CC = CUDACore.capability(CUDACore.device())

# Build a plan, or return `nothing` when cuBLASLt has no algorithm for the
# configuration on this GPU (arch-dependent compute-type support). The library
# reports this either as an empty heuristic result (our ArgumentError) or as
# CUBLAS_STATUS_NOT_SUPPORTED from the heuristic call itself. Narrow-type
# tests must not use this — they gate on `CC` and require the plan to build.
function try_plan(; kwargs...)
    try
        return MatmulPlan(; kwargs...)
    catch err
        err isa ArgumentError && occursin("no algorithm", err.msg) && return nothing
        err isa CUBLASError && return nothing
        rethrow()
    end
end
