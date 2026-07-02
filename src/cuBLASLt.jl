module cuBLASLt

# Tier-2 layer over the cuBLAS package's libcublasLt bindings. One operation:
#
#     D = α ⋅ op(A) ⋅ op(B) + β ⋅ C
#
# with per-call compute types, block-scaled narrow types, strided batching,
# and plan caching. Everything that affects algorithm selection lives in a
# `MatmulPlan`; everything resolved at execution time is a `matmul!` argument.

using CUDACore: CUDACore, CuArray, CuContext, CuPtr, cudaDataType,
                StridedCuMatrix, StridedCuVecOrMat

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

export MatmulPlan, matmul!

public handle, ltptr, version, empty_plan_cache!

include("types.jl")
include("handle.jl")
include("plan.jl")
include("cache.jl")
include("matmul.jl")

end # module
