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

const EPILOGUES = (;
    relu          = CUBLASLT_EPILOGUE_RELU,
    relu_aux      = CUBLASLT_EPILOGUE_RELU_AUX,
    bias          = CUBLASLT_EPILOGUE_BIAS,
    relu_bias     = CUBLASLT_EPILOGUE_RELU_BIAS,
    relu_aux_bias = CUBLASLT_EPILOGUE_RELU_AUX_BIAS,
    drelu         = CUBLASLT_EPILOGUE_DRELU,
    drelu_bgrad   = CUBLASLT_EPILOGUE_DRELU_BGRAD,
    gelu          = CUBLASLT_EPILOGUE_GELU,
    gelu_aux      = CUBLASLT_EPILOGUE_GELU_AUX,
    gelu_bias     = CUBLASLT_EPILOGUE_GELU_BIAS,
    gelu_aux_bias = CUBLASLT_EPILOGUE_GELU_AUX_BIAS,
    dgelu         = CUBLASLT_EPILOGUE_DGELU,
    dgelu_bgrad   = CUBLASLT_EPILOGUE_DGELU_BGRAD,
    bgrada        = CUBLASLT_EPILOGUE_BGRADA,
    bgradb        = CUBLASLT_EPILOGUE_BGRADB,
)

symbol_list(t::NamedTuple) = join((":$k" for k in keys(t)), ", ")

function epilogue_enum(s::Symbol)
    s === :none && return nothing
    haskey(EPILOGUES, s) ||
        throw(ArgumentError("unknown epilogue :$s; expected :none or one of $(symbol_list(EPILOGUES))"))
    return EPILOGUES[s]
end

# What each epilogue reads/writes at apply time. `bias` is an input vector,
# `bgrad` an output vector — both bind the same BIAS_POINTER, split by name so
# a call site can't accidentally overwrite an input. `aux` is an output for
# the forward *_aux epilogues and an input for the d* backward ones.
function epilogue_io(s::Symbol)
    bias  = s in (:bias, :relu_bias, :relu_aux_bias, :gelu_bias, :gelu_aux_bias)
    bgrad = s in (:drelu_bgrad, :dgelu_bgrad, :bgrada, :bgradb)
    aux   = s in (:relu_aux, :relu_aux_bias, :gelu_aux, :gelu_aux_bias,
                  :drelu, :drelu_bgrad, :dgelu, :dgelu_bgrad)
    return (; bias, bgrad, aux)
end

# the aux buffer holds GELU pre-activations (a real matrix, dtype settable) or
# a ReLU sign bit-mask (1 bit per element, ld counted in bits)
aux_kind(s::Symbol) =
    s in (:gelu_aux, :gelu_aux_bias, :dgelu, :dgelu_bgrad) ? :gelu :
    s in (:relu_aux, :relu_aux_bias, :drelu, :drelu_bgrad) ? :relu : :none

function compute_type(s::Symbol)
    haskey(COMPUTE_TYPES, s) ||
        throw(ArgumentError("unknown compute type :$s; expected one of $(symbol_list(COMPUTE_TYPES))"))
    return COMPUTE_TYPES[s]
end

function scale_mode_enum(s::Symbol)
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

function check_feature_support(compute::Symbol, scale_modeA::Symbol, scale_modeB::Symbol,
                               scale_modeC::Symbol, scale_modeD::Symbol,
                               out_scale_modeD::Symbol)
    compute === :bf16x9 && require_version(v"12.9", "compute type :bf16x9 (BF16x9 FP32 emulation)")
    for s in (scale_modeA, scale_modeB, scale_modeC, scale_modeD)
        # SCALAR_32F predates the scale-mode attribute (it is the implicit default
        # for FP8 scale pointers), so only block modes hard-require 12.8.
        s in (:none, :scalar_f32) ||
            require_version(v"12.8", "block scale mode :$s")
    end
    # the D_OUT_SCALE attributes (kernel-computed output block scales) are new
    out_scale_modeD === :none ||
        require_version(v"12.9", "block-scaled output (out_scale_modeD = :$out_scale_modeD)")
    return nothing
end
