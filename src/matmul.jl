# matmul! entry points: the plan path (the primitive everything else reduces
# to) and the planless convenience that derives a plan key and hits the cache.

"""
    cuBLASLt.ltptr(x) -> CuPtr{Cvoid}

The device pointer cuBLASLt reads an operand through. Defaults to
`reinterpret(CuPtr{Cvoid}, pointer(x))`; overload this (or unwrap parents) to
hand over custom storage wrappers without this package knowing their types.
"""
ltptr(x) = reinterpret(CuPtr{Cvoid}, pointer(x))

# alignment class of a device pointer, capped at cuBLASLt's 256-byte default
ptr_alignment(p::CuPtr{Cvoid}) = 1 << min(8, trailing_zeros(UInt(p)))

function check_alignment(plan::MatmulPlan, A, B, C, D)
    for (name, x, need) in ((:A, A, plan.alignA), (:B, B, plan.alignB),
                            (:C, C, plan.alignC), (:D, D, plan.alignD))
        have = ptr_alignment(ltptr(x))
        have >= need || throw(ArgumentError(
            "operand $name is $have-byte aligned, but the plan promised the " *
            "heuristic ≥ $need bytes (alignment gates algorithm validity; " *
            "rebuild the plan with align$name = $have)"))
    end
    return nothing
end

"""
    matmul!(D, A, B, plan::MatmulPlan;
            α = true, β = false, C = D,
            scaleA = nothing, scaleB = nothing, workspace = nothing)

Compute `D = α ⋅ op(A) ⋅ op(B) + β ⋅ C` according to `plan`.

Operands are anything accepted by [`cuBLASLt.ltptr`](@ref). `α`/`β` are host
`Number`s (`pointer_mode = :host`) or 0-dimensional `CuArray`s
(`pointer_mode = :device`). `scaleA`/`scaleB` are device scale arrays, required
iff the plan's corresponding scale mode is not `:none`. For the block modes
(`:vec32_ue8m0`, `:vec16_ue4m3`) cuBLASLt reads scales through its tiled
layout of 128(outer)×4(inner)-entry tiles: data must be swizzled accordingly
and the allocation padded to whole tiles, or the kernel reads out of bounds.
`workspace` is an optional preallocated buffer of at least
`plan.workspace_size` bytes; by default one is allocated from the
stream-ordered pool.
"""
function matmul!(D, A, B, plan::MatmulPlan{T};
                 α = true, β = false, C = D,
                 scaleA = nothing, scaleB = nothing,
                 workspace = nothing) where {T}
    if plan.scaleA === :none
        scaleA === nothing || throw(ArgumentError(
            "plan has scaleA = :none but a scaleA pointer was passed"))
    else
        scaleA !== nothing || throw(ArgumentError(
            "plan has scaleA = :$(plan.scaleA); pass the device scale array as scaleA"))
    end
    if plan.scaleB === :none
        scaleB === nothing || throw(ArgumentError(
            "plan has scaleB = :none but a scaleB pointer was passed"))
    else
        scaleB !== nothing || throw(ArgumentError(
            "plan has scaleB = :$(plan.scaleB); pass the device scale array as scaleB"))
    end

    check_alignment(plan, A, B, C, D)

    ws = if workspace === nothing
        CuArray{UInt8}(undef, plan.workspace_size)
    else
        sizeof(workspace) >= plan.workspace_size || throw(ArgumentError(
            "workspace of $(sizeof(workspace)) bytes is smaller than the plan's " *
            "requirement of $(plan.workspace_size) bytes"))
        workspace
    end

    if plan.pointer_mode === :host
        α isa Number && β isa Number || throw(ArgumentError(
            "plan has pointer_mode = :host; α and β must be Numbers " *
            "(got $(typeof(α)) and $(typeof(β)))"))
        αarg, βarg = Ref(convert(T, α)), Ref(convert(T, β))
    else
        α isa CuArray{T,0} && β isa CuArray{T,0} || throw(ArgumentError(
            "plan has pointer_mode = :device; α and β must be CuArray{$T,0}s " *
            "(got $(typeof(α)) and $(typeof(β)))"))
        αarg, βarg = ltptr(α), ltptr(β)
    end

    GC.@preserve α β A B C D ws scaleA scaleB begin
        if plan.scaleA === :none && plan.scaleB === :none
            unsafe_matmul!(plan, αarg, βarg, A, B, C, D, ws)
        else
            # scale pointers live on the shared descriptor, so setting them and
            # launching must be atomic with respect to other users of this plan
            lock(plan.lock) do
                scaleA === nothing ||
                    set_desc!(plan.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, ltptr(scaleA))
                scaleB === nothing ||
                    set_desc!(plan.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, ltptr(scaleB))
                unsafe_matmul!(plan, αarg, βarg, A, B, C, D, ws)
            end
        end
    end
    return D
end

function unsafe_matmul!(plan::MatmulPlan, α, β, A, B, C, D, ws)
    cublasLtMatmul(handle(), plan.desc,
                   α, ltptr(A), plan.layoutA, ltptr(B), plan.layoutB,
                   β, ltptr(C), plan.layoutC, ltptr(D), plan.layoutD,
                   Ref(plan.algo), ltptr(ws), Csize_t(sizeof(ws)),
                   CUDACore.stream())
    return nothing
end

"""
    matmul!(D::StridedCuMatrix, A::StridedCuVecOrMat, B::StridedCuVecOrMat;
            transA = 'N', transB = 'N', compute = :f32,
            α = true, β = false, C = D)

Planless convenience: derives a plan key from the arguments, hits the plan
cache, and dispatches to the plan path.
"""
function matmul!(D::StridedCuMatrix, A::StridedCuVecOrMat, B::StridedCuVecOrMat;
                 transA::Char = 'N', transB::Char = 'N', compute::Symbol = :f32,
                 α = true, β = false, C::StridedCuMatrix = D)
    m, n = size(D)
    k = size(A, transA == 'N' ? 2 : 1)
    size(A, transA == 'N' ? 1 : 2) == m || throw(DimensionMismatch(
        "op(A) must be $m×$k for a $m×$n output; A is $(join(size(A), "×")) with transA = '$transA'"))
    (size(B, transB == 'N' ? 1 : 2), size(B, transB == 'N' ? 2 : 1)) == (k, n) ||
        throw(DimensionMismatch(
            "op(B) must be $k×$n; B is $(join(size(B), "×")) with transB = '$transB'"))
    size(C) == (m, n) || throw(DimensionMismatch(
        "C must be $m×$n like D; got $(join(size(C), "×"))"))

    lda, ldb = max(1, stride(A, 2)), max(1, stride(B, 2))
    ldc, ldd = max(1, stride(C, 2)), max(1, stride(D, 2))
    pointer_mode =
        α isa Number && β isa Number ? :host :
        α isa CuArray{<:Any,0} && β isa CuArray{<:Any,0} ? :device :
        throw(ArgumentError(
            "α and β must both be Numbers (host) or both 0-dimensional CuArrays " *
            "(device); got $(typeof(α)) and $(typeof(β))"))

    alignA, alignB = ptr_alignment(ltptr(A)), ptr_alignment(ltptr(B))
    alignC, alignD = ptr_alignment(ltptr(C)), ptr_alignment(ltptr(D))

    key = (CUDACore.device(), m, n, k, eltype(A), eltype(B), eltype(C), eltype(D),
           transA, transB, lda, ldb, ldc, ldd, alignA, alignB, alignC, alignD,
           compute, pointer_mode)
    plan = cached_plan(key) do
        MatmulPlan(; M = m, N = n, K = k,
                   typeA = eltype(A), typeB = eltype(B),
                   typeC = eltype(C), typeD = eltype(D),
                   transA, transB, lda, ldb, ldc, ldd,
                   alignA, alignB, alignC, alignD,
                   compute, pointer_mode)
    end
    return matmul!(D, A, B, plan; α, β, C)
end
