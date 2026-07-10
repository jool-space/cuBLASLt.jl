module cuBLASLt

# Tier-2 layer over the cuBLAS package's libcublasLt bindings. One operation:
#
#     D = α ⋅ op(A) ⋅ op(B) + β ⋅ C
#
# with per-call compute types, block-scaled narrow types, strided batching,
# and plan caching. Everything that affects algorithm selection lives in a
# `MatmulPlan`; everything resolved at execution time is an argument of the
# plan application `plan(D, A, B; ...)`. `plan_matmul(D, A, B; ...)` derives a
# plan from prototype arguments; `matmul!` composes the two through a cache.

using CUDACore: CUDACore, CuArray, CuContext, CuPtr, CU_NULL, cudaDataType

using LinearAlgebra: Adjoint, Transpose

using cuBLAS: cublasStatus_t, CUBLAS_STATUS_SUCCESS,
    cublasOperation_t, CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C,
    cublasComputeType_t,
    CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_TF32,
    CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_COMPUTE_32F_FAST_16BF,
    CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
    CUBLAS_COMPUTE_16F, CUBLAS_COMPUTE_64F,
    cublasLtHandle_t, cublasLtCreate, cublasLtGetVersion,
    cublasLtMatmulDesc_t, cublasLtMatmulDescCreate, cublasLtMatmulDescDestroy,
    cublasLtMatmulDescSetAttribute,
    CUBLASLT_MATMUL_DESC_TRANSA, CUBLASLT_MATMUL_DESC_TRANSB,
    CUBLASLT_MATMUL_DESC_POINTER_MODE,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
    CUBLASLT_MATMUL_DESC_A_SCALE_MODE, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
    CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
    CUBLASLT_MATMUL_DESC_C_SCALE_MODE, CUBLASLT_MATMUL_DESC_D_SCALE_MODE,
    CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE,
    CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
    CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
    cublasLtEpilogue_t,
    CUBLASLT_EPILOGUE_RELU, CUBLASLT_EPILOGUE_RELU_AUX,
    CUBLASLT_EPILOGUE_BIAS, CUBLASLT_EPILOGUE_RELU_BIAS, CUBLASLT_EPILOGUE_RELU_AUX_BIAS,
    CUBLASLT_EPILOGUE_DRELU, CUBLASLT_EPILOGUE_DRELU_BGRAD,
    CUBLASLT_EPILOGUE_GELU, CUBLASLT_EPILOGUE_GELU_AUX,
    CUBLASLT_EPILOGUE_GELU_BIAS, CUBLASLT_EPILOGUE_GELU_AUX_BIAS,
    CUBLASLT_EPILOGUE_DGELU, CUBLASLT_EPILOGUE_DGELU_BGRAD,
    CUBLASLT_EPILOGUE_BGRADA, CUBLASLT_EPILOGUE_BGRADB,
    cublasLtPointerMode_t, CUBLASLT_POINTER_MODE_HOST, CUBLASLT_POINTER_MODE_DEVICE,
    cublasLtMatmulMatrixScale_t,
    CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
    CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
    CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
    CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F,
    CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
    cublasLtMatrixLayout_t, cublasLtMatrixLayoutCreate, cublasLtMatrixLayoutDestroy,
    cublasLtMatrixLayoutSetAttribute,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
    cublasLtMatmulPreference_t, cublasLtMatmulPreferenceCreate,
    cublasLtMatmulPreferenceDestroy, cublasLtMatmulPreferenceSetAttribute,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
    cublasLtMatmulAlgo_t, cublasLtMatmulHeuristicResult_t,
    cublasLtMatmulAlgoGetHeuristic, cublasLtMatmul

public MatmulPlan, plan_matmul, matmul!, plan_candidates
public handle, ltptr, ltstride, ltdata, ltscale, scale_mode, activation_symbol,
    version, empty_plan_cache!

include("types.jl")
include("handle.jl")
include("plan.jl")
include("cache.jl")
include("matmul.jl")

end
