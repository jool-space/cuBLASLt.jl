# MatmulPlan: descriptor + four layouts + heuristic-chosen algo + workspace
# requirement. Everything that affects algorithm selection lives here;
# everything resolved at execution time is a `matmul!` argument.

set_desc!(desc::cublasLtMatmulDesc_t, attr, val::T) where {T} =
    cublasLtMatmulDescSetAttribute(desc, attr, Ref(val), sizeof(T))
set_layout!(layout::cublasLtMatrixLayout_t, attr, val::T) where {T} =
    cublasLtMatrixLayoutSetAttribute(layout, attr, Ref(val), sizeof(T))
set_pref!(pref::cublasLtMatmulPreference_t, attr, val::T) where {T} =
    cublasLtMatmulPreferenceSetAttribute(pref, attr, Ref(val), sizeof(T))

const NULL_DESC = cublasLtMatmulDesc_t(C_NULL)
const NULL_LAYOUT = cublasLtMatrixLayout_t(C_NULL)
const NULL_PREF = cublasLtMatmulPreference_t(C_NULL)

"""
    MatmulPlan(; M, N, K, typeA, typeB, typeD, kwargs...)

An execution plan for `D = α ⋅ op(A) ⋅ op(B) + β ⋅ C`, where `op(A)` is
`M × K`, `op(B)` is `K × N`, and `C`/`D` are `M × N` (column-major).

The type parameter is the element type of `α`/`β` (`Float32`, or `Float16`/
`Float64` for the `:f16`/`:f64` compute types).

# Keywords

  - `M`, `N`, `K`: shapes of `op(A) ⋅ op(B) = D` (required).
  - `typeA`, `typeB`, `typeD`, `typeC = typeD`: Julia types or `cudaDataType`s.
  - `transA`, `transB`: `'N'`, `'T'`, or `'C'` (default `'N'`).
  - `lda`, `ldb`, `ldc`, `ldd`: leading dimensions of the stored matrices.
  - `batch = 1`: strided-batch count; `strideA`/`strideB`/`strideC`/`strideD`
    are element strides between batch entries (default: densely packed).
  - `compute = :f32`: one of $(symbol_list(COMPUTE_TYPES)).
  - `scaleA`, `scaleB`: scale *mode* (affects algorithm choice); `:none` or one
    of $(symbol_list(SCALE_MODES)). The scale *pointers* are `matmul!` arguments.
  - `pointer_mode = :host`: `:host` (α/β are `Number`s) or `:device`
    (α/β are 0-dimensional `CuArray`s, read at execution time; graph-capture safe).
  - `alignA`, `alignB`, `alignC`, `alignD` (default 256): minimum byte alignment
    the corresponding operand pointer is guaranteed to have at call time, as a
    power of two. Alignment gates algorithm validity — pass the true alignment
    when operands will be views into larger arrays.
  - `epilogue`: reserved for v0.2; must be `nothing`.
  - `max_workspace = 32 << 20`: workspace-size ceiling for the heuristic; the
    chosen algorithm's actual requirement is stored as `plan.workspace_size`.
"""
mutable struct MatmulPlan{T}
    const desc::cublasLtMatmulDesc_t
    const layoutA::cublasLtMatrixLayout_t
    const layoutB::cublasLtMatrixLayout_t
    const layoutC::cublasLtMatrixLayout_t
    const layoutD::cublasLtMatrixLayout_t
    const algo::cublasLtMatmulAlgo_t
    const workspace_size::Int
    const M::Int
    const N::Int
    const K::Int
    const batch::Int
    const transA::Char
    const transB::Char
    const typeA::cudaDataType
    const typeB::cudaDataType
    const typeC::cudaDataType
    const typeD::cudaDataType
    const compute::Symbol
    const scaleA::Symbol
    const scaleB::Symbol
    const pointer_mode::Symbol
    const alignA::Int
    const alignB::Int
    const alignC::Int
    const alignD::Int
    # guards call-time descriptor mutation (scale pointers); taken only when a
    # scale mode is active
    const lock::ReentrantLock
end

function unsafe_destroy!(plan::MatmulPlan)
    cublasLtMatrixLayoutDestroy(plan.layoutD)
    cublasLtMatrixLayoutDestroy(plan.layoutC)
    cublasLtMatrixLayoutDestroy(plan.layoutB)
    cublasLtMatrixLayoutDestroy(plan.layoutA)
    cublasLtMatmulDescDestroy(plan.desc)
    return nothing
end

function MatmulPlan(;
        M::Integer, N::Integer, K::Integer,
        typeA, typeB, typeD, typeC = typeD,
        transA::Char = 'N', transB::Char = 'N',
        lda::Integer = transA == 'N' ? M : K,
        ldb::Integer = transB == 'N' ? K : N,
        ldc::Integer = M, ldd::Integer = M,
        batch::Integer = 1,
        strideA::Integer = lda * (transA == 'N' ? K : M),
        strideB::Integer = ldb * (transB == 'N' ? N : K),
        strideC::Integer = ldc * N,
        strideD::Integer = ldd * N,
        compute::Symbol = :f32,
        scaleA::Symbol = :none, scaleB::Symbol = :none,
        pointer_mode::Symbol = :host,
        alignA::Integer = 256, alignB::Integer = 256,
        alignC::Integer = 256, alignD::Integer = 256,
        epilogue = nothing,
        max_workspace::Integer = 32 << 20)
    epilogue === nothing ||
        throw(ArgumentError("epilogues are reserved for v0.2; pass epilogue = nothing"))
    M >= 1 && N >= 1 && K >= 1 ||
        throw(DimensionMismatch("invalid shape M=$M, N=$N, K=$K"))
    batch >= 1 || throw(DimensionMismatch("invalid batch count $batch"))
    pointer_mode in (:host, :device) ||
        throw(ArgumentError("pointer_mode must be :host or :device, got :$pointer_mode"))
    for (name, align) in ((:alignA, alignA), (:alignB, alignB),
                          (:alignC, alignC), (:alignD, alignD))
        ispow2(align) || throw(ArgumentError(
            "$name must be a power of two (bytes), got $align"))
    end

    opA, opB = operation(transA), operation(transB)
    ct = compute_type(compute)
    modeA, modeB = scale_mode(scaleA), scale_mode(scaleB)
    check_feature_support(compute, scaleA, scaleB)

    scaleT = scale_eltype(compute)
    dtA, dtB = to_datatype(typeA), to_datatype(typeB)
    dtC, dtD = to_datatype(typeC), to_datatype(typeD)

    rowsA, colsA = transA == 'N' ? (M, K) : (K, M)
    rowsB, colsB = transB == 'N' ? (K, N) : (N, K)
    lda >= rowsA || throw(DimensionMismatch("lda = $lda < $rowsA rows of stored A"))
    ldb >= rowsB || throw(DimensionMismatch("ldb = $ldb < $rowsB rows of stored B"))
    ldc >= M || throw(DimensionMismatch("ldc = $ldc < M = $M"))
    ldd >= M || throw(DimensionMismatch("ldd = $ldd < M = $M"))

    descref = Ref(NULL_DESC)
    la, lb, lc, ld = Ref(NULL_LAYOUT), Ref(NULL_LAYOUT), Ref(NULL_LAYOUT), Ref(NULL_LAYOUT)
    prefref = Ref(NULL_PREF)
    try
        cublasLtMatmulDescCreate(descref, ct, to_datatype(scaleT))
        desc = descref[]
        set_desc!(desc, CUBLASLT_MATMUL_DESC_TRANSA, opA)
        set_desc!(desc, CUBLASLT_MATMUL_DESC_TRANSB, opB)
        pointer_mode === :device &&
            set_desc!(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, CUBLASLT_POINTER_MODE_DEVICE)
        # the scale-mode attribute only exists on ≥ 12.8; below that, only the
        # implicit-default :scalar_f32 can be active (block modes threw above)
        if version() >= v"12.8"
            modeA === nothing || set_desc!(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, modeA)
            modeB === nothing || set_desc!(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, modeB)
        end

        cublasLtMatrixLayoutCreate(la, dtA, UInt64(rowsA), UInt64(colsA), Int64(lda))
        cublasLtMatrixLayoutCreate(lb, dtB, UInt64(rowsB), UInt64(colsB), Int64(ldb))
        cublasLtMatrixLayoutCreate(lc, dtC, UInt64(M), UInt64(N), Int64(ldc))
        cublasLtMatrixLayoutCreate(ld, dtD, UInt64(M), UInt64(N), Int64(ldd))
        if batch > 1
            for (l, stride) in ((la, strideA), (lb, strideB), (lc, strideC), (ld, strideD))
                set_layout!(l[], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, Cint(batch))
                set_layout!(l[], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, Int64(stride))
            end
        end

        cublasLtMatmulPreferenceCreate(prefref)
        set_pref!(prefref[], CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, Csize_t(max_workspace))
        # the heuristic assumes 256-byte-aligned operands unless told otherwise;
        # algorithms requiring more alignment than promised are filtered out here
        set_pref!(prefref[], CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, UInt32(alignA))
        set_pref!(prefref[], CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, UInt32(alignB))
        set_pref!(prefref[], CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, UInt32(alignC))
        set_pref!(prefref[], CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, UInt32(alignD))

        res = Vector{cublasLtMatmulHeuristicResult_t}(undef, 1)
        cnt = Ref{Cint}(0)
        cublasLtMatmulAlgoGetHeuristic(handle(), desc, la[], lb[], lc[], ld[],
                                       prefref[], Cint(1), res, cnt)
        if cnt[] == 0 || res[1].state != CUBLAS_STATUS_SUCCESS
            throw(ArgumentError(
                "cuBLASLt found no algorithm for " *
                config_string(M, N, K, batch, transA, transB, dtA, dtB, dtC, dtD,
                              compute, scaleA, scaleB) *
                ". This type/scale-mode combination is likely unsupported on " *
                "$(CUDACore.name(CUDACore.device())) with cuBLASLt $(version())."))
        end

        plan = MatmulPlan{scaleT}(desc, la[], lb[], lc[], ld[],
                                  res[1].algo, Int(res[1].workspaceSize),
                                  M, N, K, batch, transA, transB,
                                  dtA, dtB, dtC, dtD,
                                  compute, scaleA, scaleB, pointer_mode,
                                  Int(alignA), Int(alignB), Int(alignC), Int(alignD),
                                  ReentrantLock())
        descref[] = NULL_DESC  # ownership transferred; disarm the catch block
        la[] = lb[] = lc[] = ld[] = NULL_LAYOUT
        return finalizer(unsafe_destroy!, plan)
    finally
        prefref[] == NULL_PREF || cublasLtMatmulPreferenceDestroy(prefref[])
        for l in (ld, lc, lb, la)
            l[] == NULL_LAYOUT || cublasLtMatrixLayoutDestroy(l[])
        end
        descref[] == NULL_DESC || cublasLtMatmulDescDestroy(descref[])
    end
end

function config_string(M, N, K, batch, transA, transB, dtA, dtB, dtC, dtD,
                       compute, scaleA, scaleB)
    str = "$(M)×$(K)$(transA == 'N' ? "" : "ᵀ") ⋅ $(K)×$(N)$(transB == 'N' ? "" : "ᵀ")" *
          " → $(M)×$(N)"
    batch > 1 && (str *= " ×$batch")
    str *= " [$dtA ⋅ $dtB + $dtC → $dtD, compute $compute"
    scaleA === :none || (str *= ", scaleA $scaleA")
    scaleB === :none || (str *= ", scaleB $scaleB")
    return str * "]"
end

function Base.show(io::IO, plan::MatmulPlan)
    print(io, "MatmulPlan(",
          config_string(plan.M, plan.N, plan.K, plan.batch, plan.transA, plan.transB,
                        plan.typeA, plan.typeB, plan.typeC, plan.typeD,
                        plan.compute, plan.scaleA, plan.scaleB))
    plan.pointer_mode === :device && print(io, ", device α/β")
    print(io, ", workspace ", Base.format_bytes(plan.workspace_size), ")")
end
