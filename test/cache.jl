Random.seed!(4)

# The routing surface: the plan cache (positive *and* negative), the
# feasibility probe, and the workspace-from-a-space contract that makes an
# apply allocation-free.

@testset "matmul_supported" begin
    M, N, K = 64, 48, 32
    A, B = rand(Float32, M, K), rand(Float32, K, N)
    dA, dB = CuArray(A), CuArray(B)
    dD = CUDACore.zeros(Float32, M, N)

    @test matmul_supported(dD, dA, dB)
    @test matmul_supported(dD, dA, dB; compute = :tf32)

    # a true answer leaves the plan in the cache, so the launch that follows
    # pays no heuristic query — and must still agree with the reference
    matmul!(dD, dA, dB)
    @test Array(dD) ≈ matmul_ref(A, B, 'N', 'N', 1, 0, zeros(Float32, M, N)) rtol = 1e-5

    # malformed arguments still throw: `false` means "cuBLASLt has no
    # algorithm", never "you called this wrong"
    @test_throws DimensionMismatch matmul_supported(dD, dA, CuArray(rand(Float32, K + 1, N)))
    @test_throws ArgumentError matmul_supported(dD, dA, dB; compute = :nonsense)
end

@testset "unsupported configurations are cached, not re-queried" begin
    # MXFP8 block scaling needs CC ≥ 10.0; below that the heuristic reports no
    # algorithm — a well-formed request the library cannot serve, which is the
    # one failure the cache memoizes.
    if CC >= v"10.0" || cuBLASLt.version() < v"12.8"
        @info "skipping negative-cache test (needs a device below CC 10.0)" CC
    else
        M, N, K = 64, 32, 128
        dA = CuArray(Float8_E4M3FN.(rand(Float32, K, M)))
        dB = CuArray(Float8_E4M3FN.(rand(Float32, K, N)))
        dD = CUDACore.zeros(Float32, M, N)
        sA = fill!(CuArray{UInt8}(undef, 128 * cld(M, 128) * 4 * cld(K ÷ 32, 4)), 0x7f)
        sB = fill!(CuArray{UInt8}(undef, 128 * cld(N, 128) * 4 * cld(K ÷ 32, 4)), 0x7f)

        kws = (; transA = 'T', lda = K, ldb = K,
               scale_modeA = :vec32_ue8m0, scale_modeB = :vec32_ue8m0,
               scaleA = sA, scaleB = sB)

        @test !matmul_supported(dD, dA, dB; kws...)
        # the probe answers `false`; the launch still throws
        @test_throws cuBLASLt.UnsupportedConfigError matmul!(dD, dA, dB; kws...)
        @test any(v -> v isa cuBLASLt.UnsupportedConfigError, values(cuBLASLt.plan_cache))

        # the negative result is memoized: a repeat probe is a dict hit, not
        # another heuristic query — no new entry appears
        entries = length(cuBLASLt.plan_cache)
        @test !matmul_supported(dD, dA, dB; kws...)
        @test length(cuBLASLt.plan_cache) == entries
    end
end

# The plan is the capture seam: it knows its workspace requirement before any
# launch, so a caller can carve those bytes from wherever it keeps memory and
# hand them to the apply, which then allocates nothing. (Who owns that memory —
# an arena, a frame, a slab — is a question for the layer above; this package
# only moves pointers.)
@testset "caller-supplied workspace" begin
    M, N, K = 128, 96, 64
    A, B = rand(Float32, M, K), rand(Float32, K, N)
    dA, dB = CuArray(A), CuArray(B)
    dD = CUDACore.zeros(Float32, M, N)
    ref = matmul_ref(A, B, 'N', 'N', 1, 0, zeros(Float32, M, N))

    plan = plan_matmul(dD, dA, dB)

    # the requirement is known from the plan alone — no launch, no probing
    @test plan.workspace_size >= 0

    # one buffer, reused across applies: nothing is allocated, nothing freed
    ws = CUDACore.zeros(UInt8, max(plan.workspace_size, 1))
    for _ in 1:4
        fill!(dD, 0)
        plan(dD, dA, dB; workspace = ws)
        @test Array(dD) ≈ ref rtol = 1e-5
    end
    @test length(ws) == max(plan.workspace_size, 1)   # survived every apply

    # a slice of a larger slab works the same way (what an arena hands out)
    slab = CUDACore.zeros(UInt8, 64 << 20)
    fill!(dD, 0)
    plan(dD, dA, dB; workspace = view(slab, 1:max(plan.workspace_size, 1)))
    @test Array(dD) ≈ ref rtol = 1e-5

    # the planless form takes one too
    fill!(dD, 0)
    matmul!(dD, dA, dB; workspace = ws)
    @test Array(dD) ≈ ref rtol = 1e-5

    # too-small buffers are rejected rather than silently overrun
    if plan.workspace_size > 0
        @test_throws ArgumentError plan(dD, dA, dB;
            workspace = CUDACore.zeros(UInt8, plan.workspace_size - 1))
    end
end
