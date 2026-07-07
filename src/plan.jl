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

const DEFAULT_MAX_WORKSPACE = 32 << 20

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
  - `scale_modeA`, `scale_modeB`: scale *mode* (affects algorithm choice);
    `:none` or one of $(symbol_list(SCALE_MODES)). The scale *arrays* are
    apply-time arguments (`scaleA`/`scaleB`).
  - `pointer_mode = :host`: `:host` (α/β are `Number`s) or `:device`
    (α/β are 0-dimensional `CuArray`s, read at execution time; graph-capture safe).
  - `alignA`, `alignB`, `alignC`, `alignD` (default 256): minimum byte alignment
    the corresponding operand pointer is guaranteed to have at call time, as a
    power of two. Alignment gates algorithm validity — pass the true alignment
    when operands will be views into larger arrays.
  - `epilogue = :none`: fused epilogue, one of $(symbol_list(EPILOGUES)).
    Forward: `D = act(α ⋅ op(A) ⋅ op(B) + β ⋅ C + bias)`, with `*_aux` variants
    stashing the activation input (GELU: values; ReLU: a sign bit-mask) for the
    backward pass. Backward: `:drelu`/`:dgelu` multiply the matmul result
    elementwise by `act′(aux)`; the `*_bgrad` variants additionally reduce the
    result into a bias-gradient vector, and `:bgrada`/`:bgradb` reduce operand
    A/B over K into one. The `bias`/`bgrad`/`aux` *pointers* are apply-time
    arguments.
  - `bias_type`: element type of the bias/bgrad vector (defaults to cuBLASLt's
    rule: D's type, or BF16/FP16 for FP8 matmuls).
  - `aux_type`, `aux_ld`, `aux_stride`: aux-buffer element type (GELU only),
    leading dimension (elements; *bits* for the ReLU bit-mask, divisible by
    128), and batch stride.
  - `scale_modeC`, `scale_modeD`: like `scale_modeA/B`, for the C input and
    the D output (quantization scale); `scaleC`/`scaleD` are apply-time.
  - `out_scale_modeD`: block-scale mode of the *kernel-computed output scales*
    (MXFP8/NVFP4 output); the `out_scaleD` array they are written to is
    apply-time. Requires cuBLASLt ≥ 12.9.
  - `fast_accum = false`: FP8 fast accumulation.
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
    const scale_modeA::Symbol
    const scale_modeB::Symbol
    const scale_modeC::Symbol
    const scale_modeD::Symbol
    const out_scale_modeD::Symbol
    const epilogue::Symbol
    const fast_accum::Bool
    const pointer_mode::Symbol
    const alignA::Int
    const alignB::Int
    const alignC::Int
    const alignD::Int
    # guards call-time descriptor mutation (scale/bias/aux/amax pointers);
    # taken only when the plan has apply-time descriptor writes
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

MatmulPlan(; kwargs...) = only(matmul_plans(1; kwargs...))

# Shared builder behind `MatmulPlan` (count = 1) and `plan_candidates`: one
# heuristic query asking for `count` algorithms, one plan per result. Each
# plan owns its descriptor + layouts (they are destroyed per-plan by the
# finalizer), so the descriptor construction repeats per candidate.
function matmul_plans(count::Integer;
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
        scale_modeA::Symbol = :none, scale_modeB::Symbol = :none,
        scale_modeC::Symbol = :none, scale_modeD::Symbol = :none,
        out_scale_modeD::Symbol = :none,
        epilogue::Symbol = :none,
        bias_type = nothing, bias_stride = nothing,
        aux_type = nothing, aux_ld = nothing, aux_stride = nothing,
        fast_accum::Bool = false,
        pointer_mode::Symbol = :host,
        alignA::Integer = 256, alignB::Integer = 256,
        alignC::Integer = 256, alignD::Integer = 256,
        max_workspace::Integer = DEFAULT_MAX_WORKSPACE)
    count >= 1 || throw(ArgumentError("invalid candidate count $count"))
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

    # the operand-reducing epilogues need K-major storage of the reduced
    # operand; cuBLASLt only reports this at launch (NOT_SUPPORTED), so catch
    # it at plan time
    epilogue === :bgrada && transA != 'N' && throw(ArgumentError(
        "epilogue :bgrada requires transA = 'N' (the K-reduction over stored A)"))
    epilogue === :bgradb && transB != 'T' && throw(ArgumentError(
        "epilogue :bgradb requires transB = 'T' (the K-reduction over stored B)"))

    opA, opB = operation(transA), operation(transB)
    ct = compute_type(compute)
    modeA, modeB = scale_mode_enum(scale_modeA), scale_mode_enum(scale_modeB)
    modeC, modeD = scale_mode_enum(scale_modeC), scale_mode_enum(scale_modeD)
    out_modeD = scale_mode_enum(out_scale_modeD)
    ep = epilogue_enum(epilogue)
    check_feature_support(compute, scale_modeA, scale_modeB,
                          scale_modeC, scale_modeD, out_scale_modeD)

    scaleT = scale_eltype(compute)
    dtA, dtB = to_datatype(typeA), to_datatype(typeB)
    dtC, dtD = to_datatype(typeC), to_datatype(typeD)

    rowsA, colsA = transA == 'N' ? (M, K) : (K, M)
    rowsB, colsB = transB == 'N' ? (K, N) : (N, K)
    lda >= rowsA || throw(DimensionMismatch("lda = $lda < $rowsA rows of stored A"))
    ldb >= rowsB || throw(DimensionMismatch("ldb = $ldb < $rowsB rows of stored B"))
    ldc >= M || throw(DimensionMismatch("ldc = $ldc < M = $M"))
    ldd >= M || throw(DimensionMismatch("ldd = $ldd < M = $M"))

    # scale-pointer attributes whose mode is set must be non-NULL during the
    # heuristic (cuBLASLt validates them there; NULL means "scale = 1", which
    # only exists for tensor-wide scaling). The real pointers are apply-time
    # arguments, so aim the descriptor at a throwaway buffer for the heuristic
    # call; apply re-points the descriptor before every launch.
    scale_ptr_attrs = Tuple(attr for (mode, attr) in (
        (modeA, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER),
        (modeB, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER),
        (modeC, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER),
        (modeD, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER),
        (out_modeD, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER),
    ) if mode !== nothing)

    plans = MatmulPlan{scaleT}[]
    results = cublasLtMatmulHeuristicResult_t[]
    descref = Ref(NULL_DESC)
    la, lb, lc, ld = Ref(NULL_LAYOUT), Ref(NULL_LAYOUT), Ref(NULL_LAYOUT), Ref(NULL_LAYOUT)
    prefref = Ref(NULL_PREF)
    try
        for i in 1:count
            cublasLtMatmulDescCreate(descref, ct, to_datatype(scaleT))
            desc = descref[]
            set_desc!(desc, CUBLASLT_MATMUL_DESC_TRANSA, opA)
            set_desc!(desc, CUBLASLT_MATMUL_DESC_TRANSB, opB)
            pointer_mode === :device &&
                set_desc!(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, CUBLASLT_POINTER_MODE_DEVICE)
            # the scale-mode attributes only exist on ≥ 12.8; below that, only
            # the implicit-default :scalar_f32 can be active (block modes threw
            # in check_feature_support)
            if version() >= v"12.8"
                modeA === nothing || set_desc!(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, modeA)
                modeB === nothing || set_desc!(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, modeB)
                modeC === nothing || set_desc!(desc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE, modeC)
                modeD === nothing || set_desc!(desc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, modeD)
            end
            out_modeD === nothing ||
                set_desc!(desc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, out_modeD)
            ep === nothing || set_desc!(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, ep)
            fast_accum && set_desc!(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, Int8(1))
            bias_type === nothing ||
                set_desc!(desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, to_datatype(bias_type))
            bias_stride === nothing ||
                set_desc!(desc, CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, Int64(bias_stride))
            aux_type === nothing ||
                set_desc!(desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, to_datatype(aux_type))
            aux_ld === nothing ||
                set_desc!(desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, Int64(aux_ld))
            aux_stride === nothing ||
                set_desc!(desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE, Int64(aux_stride))

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

            if i == 1
                cublasLtMatmulPreferenceCreate(prefref)
                set_pref!(prefref[], CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, Csize_t(max_workspace))
                # the heuristic assumes 256-byte-aligned operands unless told
                # otherwise; algorithms requiring more than promised are filtered
                set_pref!(prefref[], CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, UInt32(alignA))
                set_pref!(prefref[], CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, UInt32(alignB))
                set_pref!(prefref[], CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, UInt32(alignC))
                set_pref!(prefref[], CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, UInt32(alignD))

                res = Vector{cublasLtMatmulHeuristicResult_t}(undef, count)
                cnt = Ref{Cint}(0)
                if isempty(scale_ptr_attrs)
                    cublasLtMatmulAlgoGetHeuristic(handle(), desc, la[], lb[], lc[], ld[],
                                                   prefref[], Cint(count), res, cnt)
                else
                    placeholder = CuArray{UInt8}(undef, 16)  # scale pointers must be 16B-aligned
                    for attr in scale_ptr_attrs
                        set_desc!(desc, attr, ltptr(placeholder))
                    end
                    GC.@preserve placeholder begin
                        cublasLtMatmulAlgoGetHeuristic(handle(), desc, la[], lb[], lc[], ld[],
                                                       prefref[], Cint(count), res, cnt)
                    end
                    for attr in scale_ptr_attrs
                        set_desc!(desc, attr, CU_NULL)
                    end
                end
                append!(results, (r for r in view(res, 1:Int(cnt[]))
                                  if r.state == CUBLAS_STATUS_SUCCESS))
                if isempty(results)
                    throw(ArgumentError(
                        "cuBLASLt found no algorithm for " *
                        config_string(M, N, K, batch, transA, transB, dtA, dtB, dtC, dtD,
                                      compute, scale_modeA, scale_modeB, scale_modeD,
                                      out_scale_modeD, epilogue) *
                        ". This configuration is likely unsupported on " *
                        "$(CUDACore.name(CUDACore.device())) with cuBLASLt $(version())."))
                end
            end
            i <= length(results) || break

            plan = MatmulPlan{scaleT}(descref[], la[], lb[], lc[], ld[],
                                      results[i].algo, Int(results[i].workspaceSize),
                                      M, N, K, batch, transA, transB,
                                      dtA, dtB, dtC, dtD,
                                      compute, scale_modeA, scale_modeB,
                                      scale_modeC, scale_modeD, out_scale_modeD,
                                      epilogue, fast_accum, pointer_mode,
                                      Int(alignA), Int(alignB), Int(alignC), Int(alignD),
                                      ReentrantLock())
            finalizer(unsafe_destroy!, plan)
            push!(plans, plan)
            descref[] = NULL_DESC  # ownership transferred; disarm the cleanup
            la[] = lb[] = lc[] = ld[] = NULL_LAYOUT
        end
        return plans
    finally
        prefref[] == NULL_PREF || cublasLtMatmulPreferenceDestroy(prefref[])
        for l in (ld, lc, lb, la)
            l[] == NULL_LAYOUT || cublasLtMatrixLayoutDestroy(l[])
        end
        descref[] == NULL_DESC || cublasLtMatmulDescDestroy(descref[])
    end
end

function config_string(M, N, K, batch, transA, transB, dtA, dtB, dtC, dtD,
                       compute, scale_modeA, scale_modeB, scale_modeD,
                       out_scale_modeD, epilogue)
    str = "$(M)×$(K)$(transA == 'N' ? "" : "ᵀ") ⋅ $(K)×$(N)$(transB == 'N' ? "" : "ᵀ")" *
          " → $(M)×$(N)"
    batch > 1 && (str *= " ×$batch")
    str *= " [$dtA ⋅ $dtB + $dtC → $dtD, compute $compute"
    scale_modeA === :none || (str *= ", A scales $scale_modeA")
    scale_modeB === :none || (str *= ", B scales $scale_modeB")
    scale_modeD === :none || (str *= ", D scale $scale_modeD")
    out_scale_modeD === :none || (str *= ", D out-scales $out_scale_modeD")
    epilogue === :none || (str *= ", epilogue $epilogue")
    return str * "]"
end

function Base.show(io::IO, plan::MatmulPlan)
    print(io, "MatmulPlan(",
          config_string(plan.M, plan.N, plan.K, plan.batch, plan.transA, plan.transB,
                        plan.typeA, plan.typeB, plan.typeC, plan.typeD,
                        plan.compute, plan.scale_modeA, plan.scale_modeB,
                        plan.scale_modeD, plan.out_scale_modeD, plan.epilogue))
    plan.pointer_mode === :device && print(io, ", device α/β")
    plan.fast_accum && print(io, ", fast accum")
    print(io, ", workspace ", Base.format_bytes(plan.workspace_size), ")")
end
