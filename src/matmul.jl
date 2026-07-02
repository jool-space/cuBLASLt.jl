# The planning tiers. `plan_matmul` and plan application share one signature:
# every call-time argument doubles as the prototype its plan-time counterpart
# is derived from (α/β types → pointer mode, C → typeC/ldc, scale arrays →
# scale modes, workspace → max_workspace). `matmul!` is the composition of the
# two through the plan cache.

"""
    cuBLASLt.ltptr(x) -> CuPtr{Cvoid}

The device pointer cuBLASLt reads an operand's storage through. Defaults to
`reinterpret(CuPtr{Cvoid}, pointer(x))`.
"""
ltptr(x) = reinterpret(CuPtr{Cvoid}, pointer(x))

"""
    cuBLASLt.ltdata(x) -> storage array

The storage array behind an operand — the thing whose eltype, size, strides,
and pointer describe what cuBLASLt reads. Identity by default; block-scaled
container types overload this to hand over their payload without this package
knowing their layout.
"""
ltdata(x) = x

"""
    cuBLASLt.ltscale(x) -> device scale array or nothing

The device scale array an operand carries, `nothing` by default. Operand types
that bundle their scales overload this so [`plan_matmul`](@ref) and plan
application extract scale pointers without explicit `scaleA`/`scaleB`
arguments (which, when passed, take precedence).
"""
ltscale(x) = nothing

"""
    cuBLASLt.scale_mode(x) -> Symbol

The scale mode implied by an operand's type, `:none` by default. Block-scaled
container types overload this (block shape + scale eltype → one of
$(symbol_list(SCALE_MODES))).
"""
scale_mode(x) = :none

# operand orientation from LinearAlgebra wrappers; anything raw is 'N': the
# stored matrix, trusted
unwrap_op(x) = 'N', x
unwrap_op(x::Transpose) = 'T', parent(x)
unwrap_op(x::Adjoint) = 'C', parent(x)

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

# the explicit scaleA/scaleB argument wins over what the operand carries
operand_scale(x, explicit) = explicit === nothing ? ltscale(x) : explicit

# scale-mode inference from a bare scale array; operands that carry their mode
# by type (scale_mode) never reach this
infer_scale_mode(name, ::Nothing) = :none
infer_scale_mode(name, s::CuArray{Float32}) =
    length(s) == 1 ? :scalar_f32 : throw(ArgumentError(
        "Float32 scales of length $(length(s)) are ambiguous " *
        "(:vec128_f32 or :blk128x128_f32); pass scale_mode$name explicitly"))
infer_scale_mode(name, s) = throw(ArgumentError(
    "cannot infer a scale mode from a $(typeof(s)); pass scale_mode$name explicitly"))

# Plan-time facts derived from prototype arguments. `kws` overrides that feed
# back into the derivation (transA/transB, scale modes) are consumed here; the
# rest merely pass through to the MatmulPlan kwargs constructor, where
# splatting them after the derived NamedTuple makes them override it.
function derived_plan_kwargs(D, A, B, α, β, C, scaleA, scaleB, workspace, kws)
    wrapA, pA = unwrap_op(A)
    wrapB, pB = unwrap_op(B)
    transA = get(kws, :transA, wrapA)
    transB = get(kws, :transB, wrapB)
    wrapA != 'N' && transA != wrapA && throw(ArgumentError(
        "A is a $(nameof(typeof(A))) (transA = '$wrapA') but transA = '$transA' was passed"))
    wrapB != 'N' && transB != wrapB && throw(ArgumentError(
        "B is a $(nameof(typeof(B))) (transB = '$wrapB') but transB = '$transB' was passed"))

    dataA, dataB = ltdata(pA), ltdata(pB)
    dataC, dataD = ltdata(C), ltdata(D)

    ndims(dataD) in (2, 3) || throw(DimensionMismatch(
        "D must be a matrix or a 3-dimensional batch; got ndims = $(ndims(dataD))"))
    batch = size(dataD, 3)
    if batch > 1 || ndims(dataD) == 3
        for (name, x) in ((:A, dataA), (:B, dataB), (:C, dataC))
            ndims(x) == 3 && size(x, 3) == batch || throw(DimensionMismatch(
                "D batches $batch matrices but $name is $(join(size(x), "×"))"))
        end
    end

    m, n = size(dataD, 1), size(dataD, 2)
    k = size(dataA, transA == 'N' ? 2 : 1)
    size(dataA, transA == 'N' ? 1 : 2) == m || throw(DimensionMismatch(
        "op(A) must be $m×$k for a $m×$n output; A is $(join(size(dataA), "×")) with transA = '$transA'"))
    (size(dataB, transB == 'N' ? 1 : 2), size(dataB, transB == 'N' ? 2 : 1)) == (k, n) ||
        throw(DimensionMismatch(
            "op(B) must be $k×$n; B is $(join(size(dataB), "×")) with transB = '$transB'"))
    (size(dataC, 1), size(dataC, 2)) == (m, n) || throw(DimensionMismatch(
        "C must be $m×$n like D; got $(join(size(dataC), "×"))"))

    pointer_mode =
        α isa Number && β isa Number ? :host :
        α isa CuArray{<:Any,0} && β isa CuArray{<:Any,0} ? :device :
        throw(ArgumentError(
            "α and β must both be Numbers (host) or both 0-dimensional CuArrays " *
            "(device); got $(typeof(α)) and $(typeof(β))"))

    scale_modeA = haskey(kws, :scale_modeA) ? kws[:scale_modeA] :
        scale_mode(pA) !== :none ? scale_mode(pA) :
        infer_scale_mode(:A, operand_scale(pA, scaleA))
    scale_modeB = haskey(kws, :scale_modeB) ? kws[:scale_modeB] :
        scale_mode(pB) !== :none ? scale_mode(pB) :
        infer_scale_mode(:B, operand_scale(pB, scaleB))

    derived = (;
        M = m, N = n, K = k,
        typeA = eltype(dataA), typeB = eltype(dataB),
        typeC = eltype(dataC), typeD = eltype(dataD),
        transA, transB,
        lda = max(1, stride(dataA, 2)), ldb = max(1, stride(dataB, 2)),
        ldc = max(1, stride(dataC, 2)), ldd = max(1, stride(dataD, 2)),
        batch,
        alignA = ptr_alignment(ltptr(dataA)), alignB = ptr_alignment(ltptr(dataB)),
        alignC = ptr_alignment(ltptr(dataC)), alignD = ptr_alignment(ltptr(dataD)),
        compute = :f32, pointer_mode, scale_modeA, scale_modeB,
        max_workspace = workspace === nothing ? DEFAULT_MAX_WORKSPACE : sizeof(workspace))
    batch > 1 && (derived = (; derived...,
        strideA = stride(dataA, 3), strideB = stride(dataB, 3),
        strideC = stride(dataC, 3), strideD = stride(dataD, 3)))
    return derived
end

"""
    plan_matmul(D, A, B; α = true, β = false, C = D,
                scaleA = nothing, scaleB = nothing, workspace = nothing,
                kws...) -> MatmulPlan

Build a [`MatmulPlan`](@ref) for `D = α ⋅ op(A) ⋅ op(B) + β ⋅ C` from
prototype arguments — the same signature the returned plan is applied with,
so every call-time argument doubles as the prototype its plan-time
counterpart is derived from:

  - shapes, element types, leading dimensions, batching, and pointer
    alignments come from the arrays (via [`cuBLASLt.ltdata`](@ref));
  - `Transpose`/`Adjoint` wrappers on `A`/`B` set `transA`/`transB`;
  - the types of `α`/`β` set `pointer_mode` (`Number`s → `:host`,
    0-dimensional `CuArray`s → `:device`);
  - `scaleA`/`scaleB` arrays (or scales carried by the operands, see
    [`cuBLASLt.ltscale`](@ref) and [`cuBLASLt.scale_mode`](@ref)) set the
    scale modes;
  - `workspace` sets `max_workspace = sizeof(workspace)`, guaranteeing the
    chosen algorithm fits the buffer.

Remaining `kws` override anything derived and are passed through to
`MatmulPlan` (e.g. `compute`, `scale_modeA`, `max_workspace`).

Alignment caveat: the plan promises the heuristic the *prototypes'* pointer
alignments, and application checks that later operands keep the promise —
plan with your worst-aligned representative.
"""
function plan_matmul(D, A, B; α = true, β = false, C = D,
                     scaleA = nothing, scaleB = nothing, workspace = nothing,
                     kws...)
    derived = derived_plan_kwargs(D, A, B, α, β, C, scaleA, scaleB, workspace, kws)
    return MatmulPlan(; derived..., kws...)
end

# a raw operand is trusted as the stored matrix; a wrapped one is unwrapped
# and must agree with the plan's orientation
apply_operand(x, trans::Char, name::Symbol) = x
function apply_operand(x::Union{Transpose,Adjoint}, trans::Char, name::Symbol)
    w, p = unwrap_op(x)
    w == trans || throw(ArgumentError(
        "operand $name is a $(nameof(typeof(x))) but the plan has trans$name = '$trans'"))
    return p
end

"""
    (plan::MatmulPlan)(D, A, B; α = true, β = false, C = D,
                       scaleA = nothing, scaleB = nothing, workspace = nothing)

Compute `D = α ⋅ op(A) ⋅ op(B) + β ⋅ C` according to `plan`, in place.

Operands are anything [`cuBLASLt.ltdata`](@ref)/[`cuBLASLt.ltptr`](@ref)
accept; `Transpose`/`Adjoint` wrappers on `A`/`B` are unwrapped and checked
against the plan's orientation, raw arrays are trusted as the stored
matrices. `α`/`β` are host `Number`s (`pointer_mode = :host`) or
0-dimensional `CuArray`s (`pointer_mode = :device`). `scaleA`/`scaleB` are
device scale arrays, required iff the plan's corresponding scale mode is not
`:none` and the operand does not carry its own (see
[`cuBLASLt.ltscale`](@ref)). For the block modes (`:vec32_ue8m0`,
`:vec16_ue4m3`) cuBLASLt reads scales through its tiled layout of
128(outer)×4(inner)-entry tiles: data must be swizzled accordingly and the
allocation padded to whole tiles, or the kernel reads out of bounds.
`workspace` is an optional preallocated buffer of at least
`plan.workspace_size` bytes; by default one is allocated from the
stream-ordered pool and freed right after the launch. **Under graph capture,
always pass `workspace` explicitly** — the pool default is eager-safe only.
"""
function (plan::MatmulPlan{T})(D, A, B; α = true, β = false, C = D,
                               scaleA = nothing, scaleB = nothing,
                               workspace = nothing) where {T}
    pA = apply_operand(A, plan.transA, :A)
    pB = apply_operand(B, plan.transB, :B)
    sA = operand_scale(pA, scaleA)
    sB = operand_scale(pB, scaleB)

    if plan.scale_modeA === :none
        sA === nothing || throw(ArgumentError(
            "plan has scale_modeA = :none but scales for A were passed"))
    else
        sA !== nothing || throw(ArgumentError(
            "plan has scale_modeA = :$(plan.scale_modeA); pass the device scale array as scaleA"))
    end
    if plan.scale_modeB === :none
        sB === nothing || throw(ArgumentError(
            "plan has scale_modeB = :none but scales for B were passed"))
    else
        sB !== nothing || throw(ArgumentError(
            "plan has scale_modeB = :$(plan.scale_modeB); pass the device scale array as scaleB"))
    end

    dataA, dataB = ltdata(pA), ltdata(pB)
    dataC, dataD = ltdata(C), ltdata(D)
    check_alignment(plan, dataA, dataB, dataC, dataD)

    ws_owned = workspace === nothing
    ws = if ws_owned
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

    GC.@preserve α β A B C D pA pB ws sA sB begin
        if plan.scale_modeA === :none && plan.scale_modeB === :none
            unsafe_matmul!(plan, αarg, βarg, dataA, dataB, dataC, dataD, ws)
        else
            # scale pointers live on the shared descriptor, so setting them and
            # launching must be atomic with respect to other users of this plan
            lock(plan.lock) do
                sA === nothing ||
                    set_desc!(plan.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, ltptr(sA))
                sB === nothing ||
                    set_desc!(plan.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, ltptr(sB))
                unsafe_matmul!(plan, αarg, βarg, dataA, dataB, dataC, dataD, ws)
            end
        end
    end
    # stream-ordered free: queued behind the launch, back in the pool without
    # waiting for GC
    ws_owned && CUDACore.unsafe_free!(ws)
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
    matmul!(D, A, B; α = true, β = false, C = D,
            scaleA = nothing, scaleB = nothing, workspace = nothing, kws...)

Planless convenience: [`plan_matmul`](@ref) through the plan cache, then
apply. Same signature as both; `kws` override derived plan kwargs (e.g.
`compute = :tf32`, `transA = 'T'`).
"""
function matmul!(D, A, B; α = true, β = false, C = D,
                 scaleA = nothing, scaleB = nothing, workspace = nothing,
                 kws...)
    derived = derived_plan_kwargs(D, A, B, α, β, C, scaleA, scaleB, workspace, kws)
    key = (CUDACore.device(), derived, values(kws))
    plan = cached_plan(key) do
        MatmulPlan(; derived..., kws...)
    end
    return plan(D, A, B; α, β, C, scaleA, scaleB, workspace)
end
