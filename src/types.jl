# symbol ↔ enum tables, and the version gates that go with them

const COMPUTE_TYPES = (;
    f32       = CUBLAS_COMPUTE_32F,
    tf32      = CUBLAS_COMPUTE_32F_FAST_TF32,
    fast_f16  = CUBLAS_COMPUTE_32F_FAST_16F,
    fast_bf16 = CUBLAS_COMPUTE_32F_FAST_16BF,
    bf16x9    = CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
    f16       = CUBLAS_COMPUTE_16F,
    f64       = CUBLAS_COMPUTE_64F,
)

const SCALE_MODES = (;
    scalar_f32     = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
    vec32_ue8m0    = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
    vec16_ue4m3    = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
    vec128_f32     = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F,
    blk128x128_f32 = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
)

symbol_list(t::NamedTuple) = join((":$k" for k in keys(t)), ", ")

function compute_type(s::Symbol)
    haskey(COMPUTE_TYPES, s) ||
        throw(ArgumentError("unknown compute type :$s; expected one of $(symbol_list(COMPUTE_TYPES))"))
    return COMPUTE_TYPES[s]
end

function scale_mode(s::Symbol)
    s === :none && return nothing
    haskey(SCALE_MODES, s) ||
        throw(ArgumentError("unknown scale mode :$s; expected :none or one of $(symbol_list(SCALE_MODES))"))
    return SCALE_MODES[s]
end

# α/β (and the descriptor's scale type) follow the compute type
scale_eltype(compute::Symbol) =
    compute === :f64 ? Float64 :
    compute === :f16 ? Float16 :
    Float32

function operation(c::Char)
    c == 'N' && return CUBLAS_OP_N
    c == 'T' && return CUBLAS_OP_T
    c == 'C' && return CUBLAS_OP_C
    throw(ArgumentError("unknown operation '$c'; expected 'N', 'T', or 'C'"))
end

to_datatype(t::cudaDataType) = t
to_datatype(::Type{T}) where {T} = convert(cudaDataType, T)

"""
    cuBLASLt.version() -> VersionNumber

The version of the loaded libcublasLt.
"""
function version()
    v = Int(cublasLtGetVersion())
    return VersionNumber(v ÷ 10000, (v ÷ 100) % 100, v % 100)
end

function require_version(v::VersionNumber, what)
    version() >= v || throw(ArgumentError(
        "$what requires cuBLASLt ≥ $v, but the loaded library is $(version()). " *
        "Upgrade the CUDA toolkit/runtime to use this feature."))
end

function check_feature_support(compute::Symbol, scaleA::Symbol, scaleB::Symbol)
    compute === :bf16x9 && require_version(v"12.9", "compute type :bf16x9 (BF16x9 FP32 emulation)")
    for s in (scaleA, scaleB)
        # SCALAR_32F predates the scale-mode attribute (it is the implicit default
        # for FP8 scale pointers), so only block modes hard-require 12.8.
        s in (:none, :scalar_f32) ||
            require_version(v"12.8", "block scale mode :$s")
    end
    return nothing
end
