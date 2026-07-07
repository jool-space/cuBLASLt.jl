@testset "plan construction" begin
    plan = MatmulPlan(; M = 32, N = 16, K = 64, typeA = Float32, typeB = Float32,
                      typeD = Float32)
    @test plan isa MatmulPlan{Float32}
    @test plan.workspace_size >= 0
    @test occursin("32×64", sprint(show, plan))

    # scale type follows compute type
    @test MatmulPlan(; M = 8, N = 8, K = 8, typeA = Float64, typeB = Float64,
                     typeD = Float64, compute = :f64) isa MatmulPlan{Float64}
end

@testset "plan validation" begin
    kw = (; M = 32, N = 16, K = 64, typeA = Float32, typeB = Float32, typeD = Float32)

    @test_throws ArgumentError MatmulPlan(; kw..., compute = :f37)
    @test_throws ArgumentError MatmulPlan(; kw..., scale_modeA = :vec42_bogus)
    @test_throws ArgumentError MatmulPlan(; kw..., transA = 'X')
    @test_throws ArgumentError MatmulPlan(; kw..., pointer_mode = :remote)
    @test_throws ArgumentError MatmulPlan(; kw..., epilogue = :softmax)  # not fusable
    @test_throws ArgumentError MatmulPlan(; kw..., alignA = 3)  # not a power of two
    @test_throws DimensionMismatch MatmulPlan(; kw..., M = 0)
    @test_throws DimensionMismatch MatmulPlan(; kw..., lda = 1)  # < rows of stored A
    @test_throws ArgumentError MatmulPlan(; kw..., typeA = String)
end

@testset "plan_matmul derivation" begin
    M, N, K = 32, 16, 24
    dA, dB = CuArray(rand(Float32, M, K)), CuArray(rand(Float32, K, N))
    dD = CUDACore.zeros(Float32, M, N)

    plan = plan_matmul(dD, dA, dB)
    @test plan isa MatmulPlan{Float32}
    @test (plan.M, plan.N, plan.K) == (M, N, K)
    @test plan.transA == 'N' && plan.transB == 'N'
    @test plan.pointer_mode === :host

    # Transpose/Adjoint wrappers set the orientation
    dAt = CuArray(rand(Float32, K, M))
    @test plan_matmul(dD, transpose(dAt), dB).transA == 'T'
    @test plan_matmul(dD, dAt', dB).transA == 'C'
    # an agreeing transA kwarg is redundant but fine; a conflicting one throws
    @test plan_matmul(dD, transpose(dAt), dB; transA = 'T').transA == 'T'
    @test_throws ArgumentError plan_matmul(dD, transpose(dAt), dB; transA = 'N')

    # α/β prototypes set the pointer mode
    plan_dev = plan_matmul(dD, dA, dB; α = device_scalar(1.0f0),
                           β = device_scalar(0.0f0))
    @test plan_dev.pointer_mode === :device
    @test_throws ArgumentError plan_matmul(dD, dA, dB; α = device_scalar(1.0f0),
                                           β = 0.0f0)

    # a workspace prototype caps the heuristic's workspace budget
    ws = CuArray{UInt8}(undef, 1 << 20)
    @test plan_matmul(dD, dA, dB; workspace = ws).workspace_size <= sizeof(ws)

    # kws override derived plan kwargs
    @test plan_matmul(dD, dA, dB; compute = :tf32).compute === :tf32

    # scale arrays that don't pin down a mode demand an explicit one
    # (length-1 Float32 → :scalar_f32 inference is exercised in scaled.jl)
    @test_throws ArgumentError plan_matmul(dD, dA, dB;
                                           scaleA = CUDACore.zeros(Float32, 4, 4))
    @test_throws ArgumentError plan_matmul(dD, dA, dB;
                                           scaleA = CUDACore.zeros(Float16, 1))

    @test_throws DimensionMismatch plan_matmul(dD, dB, dA)
end

@testset "plan application argument validation" begin
    plan = MatmulPlan(; M = 8, N = 8, K = 8, typeA = Float32, typeB = Float32,
                      typeD = Float32)
    A, B, D = (CuArray(rand(Float32, 8, 8)) for _ in 1:3)

    # scale pointers only when the plan has a scale mode
    @test_throws ArgumentError plan(D, A, B; scaleA = A)

    # α/β must match the plan's pointer mode
    @test_throws ArgumentError plan(D, A, B; α = device_scalar(1.0f0),
                                    β = device_scalar(0.0f0))

    # undersized explicit workspace
    if plan.workspace_size > 0
        @test_throws ArgumentError plan(D, A, B;
                                        workspace = CuArray{UInt8}(undef, 1))
    end

    # operands less aligned than the plan promised the heuristic
    P = CuArray(rand(Float32, 9, 8))
    @test_throws ArgumentError plan(D, view(P, 2:9, 1:8), B)

    # a wrapped operand must agree with the plan's orientation
    @test_throws ArgumentError plan(D, transpose(A), B)

    dev_plan = MatmulPlan(; M = 8, N = 8, K = 8, typeA = Float32, typeB = Float32,
                          typeD = Float32, pointer_mode = :device)
    @test_throws ArgumentError dev_plan(D, A, B; α = 1.0f0, β = 0.0f0)
    # eltype of device scalars must match the plan's scale type
    @test_throws ArgumentError dev_plan(D, A, B; α = device_scalar(1.0),
                                        β = device_scalar(0.0))
end

@testset "plan cache" begin
    cuBLASLt.empty_plan_cache!()
    A, B = CuArray(rand(Float32, 16, 24)), CuArray(rand(Float32, 24, 8))
    D = CUDACore.zeros(Float32, 16, 8)

    matmul!(D, A, B)
    n = length(cuBLASLt.plan_cache)
    @test n == 1
    matmul!(D, A, B)
    @test length(cuBLASLt.plan_cache) == n  # same key, no new plan
    matmul!(D, A, B; compute = :tf32)
    @test length(cuBLASLt.plan_cache) == n + 1

    cuBLASLt.empty_plan_cache!()
    @test isempty(cuBLASLt.plan_cache)
end

@testset "version" begin
    v = cuBLASLt.version()
    @test v isa VersionNumber
    @test v >= v"11"
end

@testset "plan_candidates" begin
    M, N, K = 128, 128, 128
    A, B = rand(Float32, M, K), rand(Float32, K, N)
    dA, dB = CuArray.((A, B))
    dD = CUDACore.zeros(Float32, M, N)
    ref = Float64.(A) * Float64.(B)

    plans = plan_candidates(dD, dA, dB; count = 4)
    @test 1 <= length(plans) <= 4
    for plan in plans  # every candidate is a complete, runnable plan
        fill!(dD, 0)
        plan(dD, dA, dB)
        @test Array(dD) ≈ ref rtol = 1e-5
    end

    # kwargs-only form, no arrays needed to plan
    plans2 = plan_candidates(; count = 3, M, N, K, typeA = Float32,
                             typeB = Float32, typeD = Float32)
    @test 1 <= length(plans2) <= 3
    fill!(dD, 0)
    plans2[end](dD, dA, dB)
    @test Array(dD) ≈ ref rtol = 1e-5

    @test_throws ArgumentError plan_candidates(; count = 0, M, N, K,
                                               typeA = Float32, typeB = Float32,
                                               typeD = Float32)
end
