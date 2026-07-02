Random.seed!(2)

# Per-call compute types: the whole reason `math_mode!` dies.

const M, N, K = 64, 48, 128

function compute_case(compute, TA, TD; rtol, kwargs...)
    A, B = rand(TA, M, K), rand(TA, K, N)
    dA, dB = CuArray(A), CuArray(B)
    dD = CUDACore.zeros(TD, M, N)
    plan = try_plan(; M, N, K, typeA = TA, typeB = TA, typeD = TD, compute, kwargs...)
    if plan === nothing
        @info "skipping compute = :$compute (no algorithm on this GPU)"
        return
    end
    matmul!(dD, dA, dB, plan)
    @test Float64.(Array(dD)) ≈ Float64.(A) * Float64.(B) rtol = rtol
end

@testset "compute :f32" begin
    compute_case(:f32, Float32, Float32; rtol = 1e-5)
end

@testset "compute :tf32" begin
    # TF32 keeps FP32 range with 10 mantissa bits; ~1e-3 relative accuracy
    compute_case(:tf32, Float32, Float32; rtol = 5e-3)
end

@testset "compute :fast_f16" begin
    compute_case(:fast_f16, Float32, Float32; rtol = 5e-3)
end

@testset "compute :fast_bf16" begin
    # BF16 inputs carry ~8 mantissa bits
    compute_case(:fast_bf16, Float32, Float32; rtol = 5e-2)
end

@testset "compute :bf16x9" begin
    if cuBLASLt.version() >= v"12.9"
        # 3-way BF16 splitting recovers (better than) FP32 accuracy
        compute_case(:bf16x9, Float32, Float32; rtol = 1e-5)
    else
        @test_throws ArgumentError MatmulPlan(; M, N, K, typeA = Float32,
                                              typeB = Float32, typeD = Float32,
                                              compute = :bf16x9)
    end
end

@testset "compute :f16" begin
    compute_case(:f16, Float16, Float16; rtol = 5e-2)
end

@testset "compute :f64" begin
    compute_case(:f64, Float64, Float64; rtol = 1e-12)
end
