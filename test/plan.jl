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
    @test_throws ArgumentError MatmulPlan(; kw..., scaleA = :vec42_bogus)
    @test_throws ArgumentError MatmulPlan(; kw..., transA = 'X')
    @test_throws ArgumentError MatmulPlan(; kw..., pointer_mode = :remote)
    @test_throws ArgumentError MatmulPlan(; kw..., epilogue = :relu)
    @test_throws ArgumentError MatmulPlan(; kw..., alignA = 3)  # not a power of two
    @test_throws DimensionMismatch MatmulPlan(; kw..., M = 0)
    @test_throws DimensionMismatch MatmulPlan(; kw..., lda = 1)  # < rows of stored A
    @test_throws ArgumentError MatmulPlan(; kw..., typeA = String)
end

@testset "matmul! argument validation" begin
    plan = MatmulPlan(; M = 8, N = 8, K = 8, typeA = Float32, typeB = Float32,
                      typeD = Float32)
    A, B, D = (CuArray(rand(Float32, 8, 8)) for _ in 1:3)

    # scale pointers only when the plan has a scale mode
    @test_throws ArgumentError matmul!(D, A, B, plan; scaleA = A)

    # α/β must match the plan's pointer mode
    @test_throws ArgumentError matmul!(D, A, B, plan; α = device_scalar(1.0f0),
                                       β = device_scalar(0.0f0))

    # undersized explicit workspace
    if plan.workspace_size > 0
        @test_throws ArgumentError matmul!(D, A, B, plan;
                                           workspace = CuArray{UInt8}(undef, 1))
    end

    # operands less aligned than the plan promised the heuristic
    P = CuArray(rand(Float32, 9, 8))
    @test_throws ArgumentError matmul!(D, view(P, 2:9, 1:8), B, plan)

    dev_plan = MatmulPlan(; M = 8, N = 8, K = 8, typeA = Float32, typeB = Float32,
                          typeD = Float32, pointer_mode = :device)
    @test_throws ArgumentError matmul!(D, A, B, dev_plan; α = 1.0f0, β = 0.0f0)
    # eltype of device scalars must match the plan's scale type
    @test_throws ArgumentError matmul!(D, A, B, dev_plan; α = device_scalar(1.0),
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
