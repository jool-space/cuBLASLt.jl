Random.seed!(4)

# Fused epilogues. cuBLASLt's GELU is the tanh approximation; the references
# here spell it (and its derivative) out exactly.
gelu_ref(x) = 0.5x * (1 + tanh(0.7978845608028654 * (x + 0.044715x^3)))
function dgelu_ref(x)
    g = 0.7978845608028654 * (x + 0.044715x^3)
    t = tanh(g)
    return 0.5 * (1 + t) + 0.5x * (1 - t^2) * 0.7978845608028654 * (1 + 3 * 0.044715x^2)
end

# activation-function registration must be top-level (method definition)
myrelu(x) = max(x, zero(x))
cuBLASLt.activation_symbol(::typeof(myrelu)) = :relu

@testset "forward epilogues: relu/gelu/bias" begin
    M, N, K = 32, 16, 24
    A, B = rand(Float32, M, K) .- 0.5f0, rand(Float32, K, N) .- 0.5f0
    b = rand(Float32, M) .- 0.5f0
    dA, dB, db = CuArray.((A, B, b))
    dD = CUDACore.zeros(Float32, M, N)
    Z = Float64.(A) * Float64.(B)

    matmul!(dD, dA, dB; activation = :relu)
    @test Array(dD) ≈ max.(Z, 0) rtol = 1e-5

    matmul!(dD, dA, dB; bias = db)
    @test Array(dD) ≈ Z .+ b rtol = 1e-5

    matmul!(dD, dA, dB; activation = :relu, bias = db)
    @test Array(dD) ≈ max.(Z .+ b, 0) rtol = 1e-5

    matmul!(dD, dA, dB; activation = :gelu, bias = db)
    @test Array(dD) ≈ gelu_ref.(Z .+ b) rtol = 1e-4

    # registered functions stand for their symbol; unregistered ones throw
    matmul!(dD, dA, dB; activation = myrelu)
    @test Array(dD) ≈ max.(Z, 0) rtol = 1e-5
    matmul!(dD, dA, dB; activation = identity)
    @test Array(dD) ≈ Z rtol = 1e-5
    @test_throws ArgumentError matmul!(dD, dA, dB; activation = sin)
    @test_throws ArgumentError matmul!(dD, dA, dB; activation = :swish)
    # the registry answers fusability as a value for layer-level branching
    @test cuBLASLt.activation_symbol(myrelu) === :relu
    @test cuBLASLt.activation_symbol(sin) === nothing
    @test cuBLASLt.activation_symbol(:swish) === nothing
    @test cuBLASLt.activation_symbol(identity) === :none
    # the NNlib extension registers relu/gelu
    @test cuBLASLt.activation_symbol(NNlib.relu) === :relu
    @test cuBLASLt.activation_symbol(NNlib.gelu) === :gelu
    matmul!(dD, dA, dB; activation = NNlib.relu)
    @test Array(dD) ≈ max.(Z, 0) rtol = 1e-5
    @test_throws ArgumentError matmul!(dD, dA, dB; activation = :relu,
                                       epilogue = :relu_bias)

    # apply-time argument validation against the plan's epilogue
    plan = plan_matmul(dD, dA, dB; activation = :relu, bias = db)
    @test plan.epilogue === :relu_bias
    @test_throws ArgumentError plan(dD, dA, dB)                 # missing bias
    @test_throws ArgumentError plan(dD, dA, dB; bias = db, bgrad = db)
    @test_throws DimensionMismatch plan(dD, dA, dB; bias = CUDACore.zeros(Float32, M + 1))
    plain = plan_matmul(dD, dA, dB)
    @test_throws ArgumentError plain(dD, dA, dB; bias = db)     # epilogue :none
end

@testset "gelu_aux forward + dgelu_bgrad backward" begin
    M, N, K = 32, 16, 24  # gelu aux ld must be divisible by 8
    A, B = rand(Float32, M, K) .- 0.5f0, rand(Float32, K, N) .- 0.5f0
    b = rand(Float32, M) .- 0.5f0
    dA, dB, db = CuArray.((A, B, b))
    dD = CUDACore.zeros(Float32, M, N)
    daux = CUDACore.zeros(Float32, M, N)
    Zb = Float64.(A) * Float64.(B) .+ b

    matmul!(dD, dA, dB; activation = :gelu, bias = db, aux = daux)
    @test Array(dD) ≈ gelu_ref.(Zb) rtol = 1e-4
    @test Array(daux) ≈ Zb rtol = 1e-5  # the stashed pre-activation

    # backward through the same aux: G = (A2 B2) ⊙ gelu′(aux), bgrad = Σ_N G
    A2, B2 = rand(Float32, M, K) .- 0.5f0, rand(Float32, K, N) .- 0.5f0
    dA2, dB2 = CuArray.((A2, B2))
    dG = CUDACore.zeros(Float32, M, N)
    dbg = CUDACore.zeros(Float32, M)
    matmul!(dG, dA2, dB2; epilogue = :dgelu_bgrad, aux = daux, bgrad = dbg)
    Gref = (Float64.(A2) * Float64.(B2)) .* dgelu_ref.(Zb)
    @test Array(dG) ≈ Gref rtol = 1e-3
    @test Array(dbg) ≈ vec(sum(Gref, dims = 2)) rtol = 1e-3
end

@testset "relu_aux forward + drelu_bgrad backward" begin
    # the ReLU aux is a bit-mask; ld is counted in bits and must be divisible
    # by 128, so the byte rows are padded to 16
    M, N, K = 32, 16, 24
    A, B = rand(Float32, M, K) .- 0.5f0, rand(Float32, K, N) .- 0.5f0
    dA, dB = CuArray.((A, B))
    dD = CUDACore.zeros(Float32, M, N)
    dmask = CUDACore.zeros(UInt8, 16, N)
    Z = Float64.(A) * Float64.(B)

    matmul!(dD, dA, dB; activation = :relu, aux = dmask)
    @test Array(dD) ≈ max.(Z, 0) rtol = 1e-5

    A2, B2 = rand(Float32, M, K) .- 0.5f0, rand(Float32, K, N) .- 0.5f0
    dA2, dB2 = CuArray.((A2, B2))
    dG = CUDACore.zeros(Float32, M, N)
    dbg = CUDACore.zeros(Float32, M)
    matmul!(dG, dA2, dB2; epilogue = :drelu_bgrad, aux = dmask, bgrad = dbg)
    Gref = (Float64.(A2) * Float64.(B2)) .* (Z .> 0)
    @test Array(dG) ≈ Gref rtol = 1e-4
    @test Array(dbg) ≈ vec(sum(Gref, dims = 2)) rtol = 1e-4
end

@testset "bgrada/bgradb" begin
    # weight-gradient GEMMs: the bias gradient falls out of an operand the
    # matmul is already reading, reduced over K
    M, N, K = 16, 24, 32
    A, B = rand(Float32, M, K) .- 0.5f0, rand(Float32, K, N) .- 0.5f0
    dA, dB = CuArray.((A, B))
    dD = CUDACore.zeros(Float32, M, N)
    ref = Float64.(A) * Float64.(B)

    dbg = CUDACore.zeros(Float32, M)
    matmul!(dD, dA, dB; epilogue = :bgrada, bgrad = dbg)
    @test Array(dD) ≈ ref rtol = 1e-5
    @test Array(dbg) ≈ vec(sum(Float64.(A), dims = 2)) rtol = 1e-5

    # :bgradb reduces stored B over K, which needs transB = 'T'
    Bt = rand(Float32, N, K) .- 0.5f0
    dBt = CuArray(Bt)
    dbgN = CUDACore.zeros(Float32, N)
    matmul!(dD, dA, transpose(dBt); epilogue = :bgradb, bgrad = dbgN)
    @test Array(dD) ≈ Float64.(A) * Float64.(Bt)' rtol = 1e-5
    @test Array(dbgN) ≈ vec(sum(Float64.(Bt), dims = 2)) rtol = 1e-5

    # the K-major storage requirements are caught at plan time
    @test_throws ArgumentError matmul!(dD, dA, dB; epilogue = :bgradb, bgrad = dbgN)
    @test_throws ArgumentError matmul!(dD, transpose(CuArray(collect(A'))), dB;
                                       epilogue = :bgrada, bgrad = dbg)

    # bgrad length is validated against the epilogue's reduction axis
    @test_throws DimensionMismatch matmul!(dD, dA, transpose(dBt); epilogue = :bgradb,
                                           bgrad = CUDACore.zeros(Float32, M))
    # bgrad prototypes can't pick a gradient epilogue by themselves
    @test_throws ArgumentError matmul!(dD, dA, dB; bgrad = dbg)
end

@testset "batched bias" begin
    M, N, K, batch = 16, 24, 32, 3
    A, B = rand(Float32, M, K, batch), rand(Float32, K, N, batch)
    b = rand(Float32, M, batch)
    dA, dB, db = CuArray.((A, B, b))
    dD = CUDACore.zeros(Float32, M, N, batch)

    matmul!(dD, dA, dB; bias = db)
    D = Array(dD)
    for i in 1:batch
        @test D[:, :, i] ≈ Float64.(A[:, :, i]) * Float64.(B[:, :, i]) .+ b[:, i] rtol = 1e-5
    end
end
