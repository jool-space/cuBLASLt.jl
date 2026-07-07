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

"""
    cuBLASLt.activation_symbol(f) -> Symbol or nothing

The epilogue activation `f` stands for: `:relu` or `:gelu` (cuBLASLt's GELU
is the tanh approximation) — those two are the entire menu the library fuses.
Returns `nothing` for anything unfusable, so layer code can *branch* on
fusability (the branch constant-folds: this is dispatch on `typeof(f)`)
instead of catching exceptions; passing an unfusable `activation =` to
[`plan_matmul`](@ref)/[`matmul!`](@ref) is what throws. Symbols pass through,
`identity` means `:none`. Register a function with

    cuBLASLt.activation_symbol(::typeof(f)) = :relu   # or :gelu

so callers can pass `activation = f` (e.g. `NNlib.relu`).
"""
activation_symbol(f) = nothing
activation_symbol(s::Symbol) = s in (:none, :relu, :gelu) ? s : nothing
activation_symbol(::typeof(identity)) = :none

# activation ⊕ bias? ⊕ aux? → the epilogue enum's name for the composition
function compose_epilogue(act::Symbol, bias::Bool, aux::Bool)
    if act === :none
        aux && throw(ArgumentError(
            "an aux buffer requires an activation (the aux is what the " *
            "activation stashes for the backward pass)"))
        return bias ? :bias : :none
    end
    return Symbol(string(act), aux ? "_aux" : "", bias ? "_bias" : "")
end

# operand orientation from wrapper types; anything raw is 'N': the stored
# matrix, trusted. PermutedDimsArray carries its permutation as a type
# parameter, so a swap of the first two dims is the (only) Base spelling of a
# batched transpose — it never conjugates, so it never means 'C'.
unwrap_op(x) = 'N', x
unwrap_op(x::Transpose) = 'T', parent(x)
unwrap_op(x::Adjoint) = 'C', parent(x)
unwrap_op(x::PermutedDimsArray{<:Any,2,(1,2)}) = 'N', parent(x)
unwrap_op(x::PermutedDimsArray{<:Any,2,(2,1)}) = 'T', parent(x)
unwrap_op(x::PermutedDimsArray{<:Any,3,(1,2,3)}) = 'N', parent(x)
unwrap_op(x::PermutedDimsArray{<:Any,3,(2,1,3)}) = 'T', parent(x)
unwrap_op(::PermutedDimsArray{<:Any,N,perm}) where {N,perm} = throw(ArgumentError(
    "unsupported PermutedDimsArray permutation $perm; supported are (1,2)/(1,2,3) " *
    "(identity, 'N') and (2,1)/(2,1,3) (transposed storage, 'T')"))

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

# Plan-time facts derived from prototype arguments (`call` is the NamedTuple
# of apply-time keyword arguments plus `activation`). `kws` overrides that
# feed back into the derivation (transA/transB, scale modes, epilogue) are
# consumed here; the rest merely pass through to the MatmulPlan kwargs
# constructor, where splatting them after the derived NamedTuple makes them
# override it.
function derived_plan_kwargs(D, A, B, call, kws)
    (; α, β, C, scaleA, scaleB, workspace) = call
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

    for (name, x) in ((:A, dataA), (:B, dataB), (:C, dataC), (:D, dataD))
        stride(x, 1) == 1 || throw(ArgumentError(
            "$name has stride $(stride(x, 1)) in dimension 1; cuBLASLt layouts " *
            "are column-major and need unit stride within a column"))
    end

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
    scale_modeC = haskey(kws, :scale_modeC) ? kws[:scale_modeC] :
        infer_scale_mode(:C, call.scaleC)
    scale_modeD = haskey(kws, :scale_modeD) ? kws[:scale_modeD] :
        infer_scale_mode(:D, call.scaleD)
    # kernel-computed output block scales: mode from the D container's type
    # (a Microscaling-style D carries it) or an explicit kwarg — a bare array
    # can't imply a block geometry
    out_scale_modeD = haskey(kws, :out_scale_modeD) ? kws[:out_scale_modeD] :
        scale_mode(D) !== :none ? scale_mode(D) :
        operand_scale(D, call.out_scaleD) === nothing ? :none :
        throw(ArgumentError(
            "cannot infer out_scale_modeD from a bare out_scaleD array; " *
            "pass out_scale_modeD explicitly"))

    # epilogue: an explicit kwarg wins; else composed from the activation and
    # the bias/aux prototypes. The gradient epilogues are kwarg-only — a bgrad
    # prototype can't tell :drelu_bgrad from :bgrada/:bgradb.
    act = if call.activation === nothing
        :none
    else
        a = activation_symbol(call.activation)
        a === nothing && throw(ArgumentError(
            "activation $(call.activation) is not fusable; cuBLASLt epilogues fuse " *
            "exactly two activations, ReLU and GELU (tanh approximation). Either " *
            "register it (`cuBLASLt.activation_symbol(::typeof(f)) = :relu` or " *
            ":gelu) or apply it as a broadcast after the matmul"))
        a
    end
    epilogue = if haskey(kws, :epilogue)
        act === :none || throw(ArgumentError(
            "pass either activation or an explicit epilogue kwarg, not both"))
        kws[:epilogue]
    elseif call.bgrad !== nothing
        throw(ArgumentError(
            "gradient epilogues cannot be derived from a bgrad prototype; pass " *
            "epilogue = :drelu_bgrad, :dgelu_bgrad, :bgrada, or :bgradb"))
    else
        compose_epilogue(act, call.bias !== nothing, call.aux !== nothing)
    end

    epi_derived = (;)
    if epilogue !== :none
        bvec = call.bias !== nothing ? call.bias : call.bgrad
        bvec === nothing ||
            (epi_derived = (; epi_derived..., bias_type = eltype(bvec)))
        bvec !== nothing && batch > 1 && ndims(bvec) == 2 &&
            (epi_derived = (; epi_derived..., bias_stride = stride(bvec, 2)))
        if call.aux !== nothing
            aux = call.aux
            if aux_kind(epilogue) === :relu
                # the ReLU aux is a bit-mask; its ld/stride are counted in bits
                bits = 8 * sizeof(eltype(aux))
                epi_derived = (; epi_derived..., aux_ld = bits * stride(aux, 2))
                batch > 1 && (epi_derived = (; epi_derived...,
                    aux_stride = bits * stride(aux, 3)))
            else
                epi_derived = (; epi_derived..., aux_type = eltype(aux),
                               aux_ld = stride(aux, 2))
                batch > 1 && (epi_derived = (; epi_derived...,
                    aux_stride = stride(aux, 3)))
            end
        end
    end

    derived = (;
        epilogue, epi_derived...,
        M = m, N = n, K = k,
        typeA = eltype(dataA), typeB = eltype(dataB),
        typeC = eltype(dataC), typeD = eltype(dataD),
        transA, transB,
        lda = max(1, stride(dataA, 2)), ldb = max(1, stride(dataB, 2)),
        ldc = max(1, stride(dataC, 2)), ldd = max(1, stride(dataD, 2)),
        batch,
        alignA = ptr_alignment(ltptr(dataA)), alignB = ptr_alignment(ltptr(dataB)),
        alignC = ptr_alignment(ltptr(dataC)), alignD = ptr_alignment(ltptr(dataD)),
        compute = :f32, pointer_mode,
        scale_modeA, scale_modeB, scale_modeC, scale_modeD, out_scale_modeD,
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
  - `Transpose`/`Adjoint` wrappers on `A`/`B` set `transA`/`transB`, as does
    `PermutedDimsArray` with the first two dims swapped — `(2,1)` or, for a
    batched transpose, `(2,1,3)` — which sets `'T'` (identity permutations
    are accepted as `'N'`; anything else throws);
  - the types of `α`/`β` set `pointer_mode` (`Number`s → `:host`,
    0-dimensional `CuArray`s → `:device`);
  - `scaleA`/`scaleB`/`scaleC`/`scaleD` arrays (or scales carried by the
    operands, see [`cuBLASLt.ltscale`](@ref) and
    [`cuBLASLt.scale_mode`](@ref)) set the scale modes; `out_scaleD` (or a
    scale-carrying `D`) sets `out_scale_modeD` — kernel-computed output block
    scales for quantized output;
  - `activation` (a `Symbol` or a function registered via
    [`cuBLASLt.activation_symbol`](@ref)) composed with the presence of
    `bias`/`aux` prototypes sets the `epilogue`; the gradient epilogues
    (`:drelu_bgrad`, `:dgelu_bgrad`, `:bgrada`, `:bgradb`) are spelled with
    an explicit `epilogue` kwarg. Bias/aux prototypes set `bias_type`,
    `aux_type`, `aux_ld`, and the batch strides;
  - `workspace` sets `max_workspace = sizeof(workspace)`, guaranteeing the
    chosen algorithm fits the buffer.

Remaining `kws` override anything derived and are passed through to
`MatmulPlan` (e.g. `compute`, `scale_modeA`, `epilogue`, `max_workspace`).

Alignment caveat: the plan promises the heuristic the *prototypes'* pointer
alignments, and application checks that later operands keep the promise —
plan with your worst-aligned representative.
"""
function plan_matmul(D, A, B; α = true, β = false, C = D,
                     scaleA = nothing, scaleB = nothing,
                     scaleC = nothing, scaleD = nothing, out_scaleD = nothing,
                     activation = nothing, bias = nothing, bgrad = nothing,
                     aux = nothing, workspace = nothing, kws...)
    call = (; α, β, C, scaleA, scaleB, scaleC, scaleD, out_scaleD,
            activation, bias, bgrad, aux, workspace)
    derived = derived_plan_kwargs(D, A, B, call, kws)
    return MatmulPlan(; derived..., kws...)
end

"""
    plan_candidates(D, A, B; count = 8, kws...) -> Vector{MatmulPlan}
    plan_candidates(; count = 8, M, N, K, typeA, typeB, typeD, kws...) -> Vector{MatmulPlan}

The algorithm escape hatch: like [`plan_matmul`](@ref) (or the `MatmulPlan`
kwargs constructor in the second form), but returns up to `count` candidate
plans, heuristic-ranked. Each is independently callable — benchmark them and
keep the winner; that is the canonical cuBLASLt autotuning loop, with the
plan cache ownership on the caller's side.
"""
function plan_candidates(D, A, B; count::Integer = 8, α = true, β = false, C = D,
                         scaleA = nothing, scaleB = nothing,
                         scaleC = nothing, scaleD = nothing, out_scaleD = nothing,
                         activation = nothing, bias = nothing, bgrad = nothing,
                         aux = nothing, workspace = nothing, kws...)
    call = (; α, β, C, scaleA, scaleB, scaleC, scaleD, out_scaleD,
            activation, bias, bgrad, aux, workspace)
    derived = derived_plan_kwargs(D, A, B, call, kws)
    return matmul_plans(count; derived..., kws...)
end
plan_candidates(; count::Integer = 8, kws...) = matmul_plans(count; kws...)

# a raw operand is trusted as the stored matrix; a wrapped one is unwrapped
# and must agree with the plan's orientation
apply_operand(x, trans::Char, name::Symbol) = x
function apply_operand(x::Union{Transpose,Adjoint,PermutedDimsArray}, trans::Char, name::Symbol)
    w, p = unwrap_op(x)
    # 'N' wrappers (identity perms) are as trusted as raw arrays
    w == 'N' || w == trans || throw(ArgumentError(
        "operand $name is a $(nameof(typeof(x))) but the plan has trans$name = '$trans'"))
    return p
end

# scale arrays are mandatory iff the plan's corresponding mode is active
function check_scale_arg(name::Symbol, mode::Symbol, x)
    if mode === :none
        x === nothing || throw(ArgumentError(
            "plan has $name = :none but an array was passed"))
    else
        x !== nothing || throw(ArgumentError(
            "plan has $name = :$mode; pass the device array"))
    end
    return nothing
end

# epilogue vectors/buffers are mandatory iff the plan's epilogue uses them
function check_epilogue_arg(name::Symbol, needed::Bool, x, epilogue::Symbol)
    if needed
        x !== nothing || throw(ArgumentError(
            "plan has epilogue = :$epilogue; pass the device array as $name"))
    else
        x === nothing || throw(ArgumentError(
            "plan has epilogue = :$epilogue, which takes no $name"))
    end
    return nothing
end

"""
    (plan::MatmulPlan)(D, A, B; α = true, β = false, C = D,
                       scaleA = nothing, scaleB = nothing,
                       scaleC = nothing, scaleD = nothing,
                       out_scaleD = nothing, amaxD = nothing,
                       bias = nothing, bgrad = nothing, aux = nothing,
                       workspace = nothing)

Compute `D = α ⋅ op(A) ⋅ op(B) + β ⋅ C` according to `plan`, in place.

Operands are anything [`cuBLASLt.ltdata`](@ref)/[`cuBLASLt.ltptr`](@ref)
accept; `Transpose`/`Adjoint`/`PermutedDimsArray` wrappers on `A`/`B` are
unwrapped and checked against the plan's orientation, raw arrays (and
identity permutations) are trusted as the stored matrices. `α`/`β` are host `Number`s (`pointer_mode = :host`) or
0-dimensional `CuArray`s (`pointer_mode = :device`). `scaleA`/`scaleB` are
device scale arrays, required iff the plan's corresponding scale mode is not
`:none` and the operand does not carry its own (see
[`cuBLASLt.ltscale`](@ref)). For the block modes (`:vec32_ue8m0`,
`:vec16_ue4m3`) cuBLASLt reads scales through its tiled layout of
128(outer)×4(inner)-entry tiles: data must be swizzled accordingly and the
allocation padded to whole tiles, or the kernel reads out of bounds.
Epilogue arguments (mandatory iff the plan's epilogue uses them): `bias` is
the input vector of length `M` for the forward `*_bias` epilogues; `bgrad`
the *output* vector the gradient epilogues reduce into (length `M`, or `N`
for `:bgradb`); `aux` the activation-input stash — written by the forward
`*_aux` epilogues, read by the backward `d*` ones (GELU: a matrix of
pre-activations; ReLU: a bit-mask; both sides of a forward/backward pair must
agree on its dtype/ld/stride, which nothing checks for you).

Output quantization: `scaleD` (with `scale_modeD`) is the quantization scale
applied to a narrow `D`; `out_scaleD` (with `out_scale_modeD`) the array the
kernel *writes* computed block scales to (or carried by a scale-bearing `D`,
see [`cuBLASLt.ltscale`](@ref)); `amaxD` an optional 1-element `Float32`
device array the pre-quantization absmax of `D` is written to; `scaleC`
dequantizes a narrow `C`.

`workspace` is an optional preallocated buffer of at least
`plan.workspace_size` bytes; by default one is allocated from the
stream-ordered pool and freed right after the launch. **Under graph capture,
always pass `workspace` explicitly** — the pool default is eager-safe only.
"""
function (plan::MatmulPlan{T})(D, A, B; α = true, β = false, C = D,
                               scaleA = nothing, scaleB = nothing,
                               scaleC = nothing, scaleD = nothing,
                               out_scaleD = nothing, amaxD = nothing,
                               bias = nothing, bgrad = nothing, aux = nothing,
                               workspace = nothing) where {T}
    pA = apply_operand(A, plan.transA, :A)
    pB = apply_operand(B, plan.transB, :B)
    sA = operand_scale(pA, scaleA)
    sB = operand_scale(pB, scaleB)
    out_sD = operand_scale(D, out_scaleD)

    check_scale_arg(:scale_modeA, plan.scale_modeA, sA)
    check_scale_arg(:scale_modeB, plan.scale_modeB, sB)
    check_scale_arg(:scale_modeC, plan.scale_modeC, scaleC)
    check_scale_arg(:scale_modeD, plan.scale_modeD, scaleD)
    check_scale_arg(:out_scale_modeD, plan.out_scale_modeD, out_sD)

    io = epilogue_io(plan.epilogue)
    check_epilogue_arg(:bias, io.bias, bias, plan.epilogue)
    check_epilogue_arg(:bgrad, io.bgrad, bgrad, plan.epilogue)
    check_epilogue_arg(:aux, io.aux, aux, plan.epilogue)
    bvec = io.bias ? bias : bgrad
    if bvec !== nothing
        blen = plan.epilogue === :bgradb ? plan.N : plan.M
        size(bvec, 1) == blen || throw(DimensionMismatch(
            "epilogue :$(plan.epilogue) needs a length-$blen $(io.bias ? "bias" : "bgrad") " *
            "vector; got size $(join(size(bvec), "×"))"))
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

    # apply-time pointers live on the shared descriptor: (attr, array) pairs
    # to write, empty for a plain matmul
    writes = Tuple((attr, x) for (attr, x) in (
        (CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, sA),
        (CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, sB),
        (CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, scaleC),
        (CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, scaleD),
        (CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, out_sD),
        (CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, amaxD),
        (CUBLASLT_MATMUL_DESC_BIAS_POINTER, bvec),
        (CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, aux),
    ) if x !== nothing)

    GC.@preserve α β A B C D pA pB ws sA sB scaleC scaleD out_sD amaxD bvec aux begin
        if isempty(writes)
            unsafe_matmul!(plan, αarg, βarg, dataA, dataB, dataC, dataD, ws)
        else
            # setting descriptor pointers and launching must be atomic with
            # respect to other users of this plan
            lock(plan.lock) do
                for (attr, x) in writes
                    set_desc!(plan.desc, attr, ltptr(x))
                end
                unsafe_matmul!(plan, αarg, βarg, dataA, dataB, dataC, dataD, ws)
                # amax is the one *optional* pointer: un-set it so a later
                # apply without amaxD doesn't scribble on a freed array
                amaxD === nothing ||
                    set_desc!(plan.desc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, CU_NULL)
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
            scaleA = nothing, scaleB = nothing, scaleC = nothing,
            scaleD = nothing, out_scaleD = nothing, amaxD = nothing,
            activation = nothing, bias = nothing, bgrad = nothing,
            aux = nothing, workspace = nothing, kws...)

Planless convenience: [`plan_matmul`](@ref) through the plan cache, then
apply. Same signature as both; `kws` override derived plan kwargs (e.g.
`compute = :tf32`, `transA = 'T'`, `epilogue = :dgelu_bgrad`).
"""
function matmul!(D, A, B; α = true, β = false, C = D,
                 scaleA = nothing, scaleB = nothing,
                 scaleC = nothing, scaleD = nothing, out_scaleD = nothing,
                 amaxD = nothing, activation = nothing, bias = nothing,
                 bgrad = nothing, aux = nothing, workspace = nothing,
                 kws...)
    call = (; α, β, C, scaleA, scaleB, scaleC, scaleD, out_scaleD,
            activation, bias, bgrad, aux, workspace)
    derived = derived_plan_kwargs(D, A, B, call, kws)
    key = (CUDACore.device(), derived, values(kws))
    plan = cached_plan(key) do
        MatmulPlan(; derived..., kws...)
    end
    return plan(D, A, B; α, β, C, scaleA, scaleB, scaleC, scaleD,
                out_scaleD, amaxD, bias, bgrad, aux, workspace)
end
