Random.seed!(1)

@testset "plain f32, all orientations" begin
    M, N, K = 48, 32, 64
    for transA in ('N', 'T'), transB in ('N', 'T')
        A = rand(Float32, transA == 'N' ? (M, K) : (K, M))
        B = rand(Float32, transB == 'N' ? (K, N) : (N, K))
        dA, dB = CuArray(A), CuArray(B)
        dD = CUDACore.zeros(Float32, M, N)

        plan = MatmulPlan(; M, N, K, typeA = Float32, typeB = Float32,
                          typeD = Float32, transA, transB)
        plan(dD, dA, dB)
        @test Array(dD) ≈ matmul_ref(A, B, transA, transB, 1, 0, zeros(Float32, M, N)) rtol = 1e-5
    end
end

@testset "α, β, and C ≠ D" begin
    M, N, K = 32, 16, 24
    A, B, C = rand(Float32, M, K), rand(Float32, K, N), rand(Float32, M, N)
    dA, dB, dC = CuArray.((A, B, C))
    dD = CUDACore.zeros(Float32, M, N)

    plan = MatmulPlan(; M, N, K, typeA = Float32, typeB = Float32, typeD = Float32)
    plan(dD, dA, dB; α = 2.0f0, β = 0.5f0, C = dC)
    @test Array(dD) ≈ matmul_ref(A, B, 'N', 'N', 2, 0.5, C) rtol = 1e-5
    @test Array(dC) ≈ C  # C is read-only when D does not alias it

    # in-place accumulation, D aliasing C
    D0 = rand(Float32, M, N)
    dD2 = CuArray(D0)
    plan(dD2, dA, dB; α = 1.0f0, β = 1.0f0)
    @test Array(dD2) ≈ matmul_ref(A, B, 'N', 'N', 1, 1, D0) rtol = 1e-5
end

@testset "device α/β" begin
    M, N, K = 16, 16, 32
    A, B = rand(Float32, M, K), rand(Float32, K, N)
    dA, dB = CuArray(A), CuArray(B)
    dD = CUDACore.zeros(Float32, M, N)

    plan = MatmulPlan(; M, N, K, typeA = Float32, typeB = Float32, typeD = Float32,
                      pointer_mode = :device)
    plan(dD, dA, dB; α = device_scalar(3.0f0), β = device_scalar(0.0f0))
    @test Array(dD) ≈ matmul_ref(A, B, 'N', 'N', 3, 0, zeros(Float32, M, N)) rtol = 1e-5
end

@testset "strided batching" begin
    M, N, K, batch = 16, 24, 32, 4
    A, B = rand(Float32, M, K, batch), rand(Float32, K, N, batch)
    dA, dB = CuArray(A), CuArray(B)
    dD = CUDACore.zeros(Float32, M, N, batch)

    plan = MatmulPlan(; M, N, K, typeA = Float32, typeB = Float32, typeD = Float32,
                      batch)
    plan(dD, dA, dB)
    D = Array(dD)
    for b in 1:batch
        @test D[:, :, b] ≈ Float32.(Float64.(A[:, :, b]) * Float64.(B[:, :, b])) rtol = 1e-5
    end
end

@testset "explicit workspace reuse" begin
    M = N = K = 64
    A, B = rand(Float32, M, K), rand(Float32, K, N)
    dA, dB = CuArray(A), CuArray(B)
    dD = CUDACore.zeros(Float32, M, N)

    plan = MatmulPlan(; M, N, K, typeA = Float32, typeB = Float32, typeD = Float32)
    ws = CuArray{UInt8}(undef, plan.workspace_size)
    for _ in 1:3
        plan(dD, dA, dB; workspace = ws)
    end
    @test Array(dD) ≈ Float32.(Float64.(A) * Float64.(B)) rtol = 1e-5
end

@testset "planless convenience" begin
    M, N, K = 40, 24, 56
    A, B, C = rand(Float32, K, M), rand(Float32, K, N), rand(Float32, M, N)
    dA, dB, dC = CuArray.((A, B, C))
    dD = CUDACore.zeros(Float32, M, N)

    matmul!(dD, dA, dB; transA = 'T', α = 2.0f0, β = 1.0f0, C = dC)
    @test Array(dD) ≈ matmul_ref(A, B, 'T', 'N', 2, 1, C) rtol = 1e-5

    # Transpose wrappers instead of the transA kwarg
    matmul!(dD, transpose(dA), dB; β = 0.0f0)
    @test Array(dD) ≈ matmul_ref(A, B, 'T', 'N', 1, 0, zeros(Float32, M, N)) rtol = 1e-5

    # views (non-dense leading dimension)
    P = CuArray(rand(Float32, 2M, K))
    vA = view(P, 1:M, 1:K)
    matmul!(dD, vA, dB; β = 0.0f0)
    @test Array(dD) ≈ Float32.(Float64.(Array(vA)) * Float64.(B)) rtol = 1e-5

    # dimension mismatches
    @test_throws DimensionMismatch matmul!(dD, dB, dA)
    @test_throws DimensionMismatch matmul!(dD, dA, dB; transA = 'T',
                                           C = CUDACore.zeros(Float32, M, 2N))
end

@testset "planless strided batching" begin
    M, N, K, batch = 16, 24, 32, 3
    A, B = rand(Float32, M, K, batch), rand(Float32, K, N, batch)
    dA, dB = CuArray(A), CuArray(B)
    dD = CUDACore.zeros(Float32, M, N, batch)

    matmul!(dD, dA, dB)
    D = Array(dD)
    for b in 1:batch
        @test D[:, :, b] ≈ Float32.(Float64.(A[:, :, b]) * Float64.(B[:, :, b])) rtol = 1e-5
    end

    @test_throws DimensionMismatch matmul!(dD, dA,
                                           CuArray(rand(Float32, K, N, batch + 1)))
end

@testset "planless with misaligned views" begin
    # a row-offset view is only element-aligned; the derived plan must promise
    # the heuristic that reduced alignment, not the 256-byte pool default
    M, N, K = 32, 16, 24
    P = CuArray(rand(Float32, M + 1, K))
    vA = view(P, 2:M+1, 1:K)  # 4-byte-aligned pointer, ld = M + 1
    B = rand(Float32, K, N)
    dB = CuArray(B)
    dD = CUDACore.zeros(Float32, M, N)

    matmul!(dD, vA, dB)
    @test Array(dD) ≈ Float32.(Float64.(Array(vA)) * Float64.(B)) rtol = 1e-5
end

@testset "graph capture with explicit workspace" begin
    M = N = K = 32
    A, B = rand(Float32, M, K), rand(Float32, K, N)
    dA, dB = CuArray(A), CuArray(B)
    dD = CUDACore.zeros(Float32, M, N)

    plan = MatmulPlan(; M, N, K, typeA = Float32, typeB = Float32, typeD = Float32)
    ws = CuArray{UInt8}(undef, plan.workspace_size)
    plan(dD, dA, dB; workspace = ws)  # warmup outside capture

    graph = CUDACore.capture(; throw_error = false) do
        plan(dD, dA, dB; workspace = ws)
    end
    if graph === nothing
        @warn "stream capture unavailable; skipping graph-capture smoke test"
    else
        exec = CUDACore.instantiate(graph)
        fill!(dD, 0)
        CUDACore.launch(exec)
        CUDACore.synchronize()
        @test Array(dD) ≈ matmul_ref(A, B, 'N', 'N', 1, 0, zeros(Float32, M, N)) rtol = 1e-5

        # replay after updating an input in place: same pointers, new values
        A2 = rand(Float32, M, K)
        copyto!(dA, A2)
        CUDACore.launch(exec)
        CUDACore.synchronize()
        @test Array(dD) ≈ matmul_ref(A2, B, 'N', 'N', 1, 0, zeros(Float32, M, N)) rtol = 1e-5
    end
end

@testset "mixed input/output types" begin
    M, N, K = 32, 32, 48
    A, B = rand(Float16, M, K), rand(Float16, K, N)
    dA, dB = CuArray(A), CuArray(B)
    dD = CUDACore.zeros(Float32, M, N)

    plan = MatmulPlan(; M, N, K, typeA = Float16, typeB = Float16, typeD = Float32)
    plan(dD, dA, dB)
    @test Array(dD) ≈ Float32.(Float64.(A) * Float64.(B)) rtol = 1e-3
end
