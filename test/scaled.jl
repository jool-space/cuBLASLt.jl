Random.seed!(3)

# Block-scaled and tensor-wide-scaled narrow-type matmuls. FP8/FP4 kernels
# want the TN, K-major orientation: A stored K×M, B stored K×N, both with
# leading dimension K.
#
# Scale arrays here are deliberately *constant*: on sm100+ cuBLASLt reads
# block scales through a swizzled layout, and constant arrays are
# swizzle-invariant, so these tests need no swizzle machinery (and no
# Microscaling dependency). Any test with non-constant scales must swizzle.

const M, N, K = 64, 32, 128

fp8_operands(TA, TB) = (TA.(rand(Float32, K, M)), TB.(rand(Float32, K, N)))

@testset "fp8, tensor-wide :scalar_f32" begin
    A, B = fp8_operands(Float8_E4M3FN, Float8_E4M3FN)
    plan = try_plan(; M, N, K, typeA = Float8_E4M3FN, typeB = Float8_E4M3FN,
                    typeD = Float32, transA = 'T', lda = K, ldb = K,
                    scaleA = :scalar_f32, scaleB = :scalar_f32)
    if plan === nothing
        @info "skipping :scalar_f32 FP8 (unsupported on this GPU)"
    else
        dD = CUDACore.zeros(Float32, M, N)
        sA, sB = CuArray([2.0f0]), CuArray([0.5f0])
        matmul!(dD, CuArray(A), CuArray(B), plan; scaleA = sA, scaleB = sB)
        @test Array(dD) ≈ 2.0f0 * 0.5f0 * (Float64.(A)' * Float64.(B)) rtol = 1e-3
    end
end

@testset "fp8, unscaled and mixed element types" begin
    for (TA, TB) in ((Float8_E4M3FN, Float8_E4M3FN), (Float8_E4M3FN, Float8_E5M2),
                     (Float8_E5M2, Float8_E4M3FN))
        A, B = fp8_operands(TA, TB)
        plan = try_plan(; M, N, K, typeA = TA, typeB = TB, typeD = Float32,
                        transA = 'T', lda = K, ldb = K)
        if plan === nothing
            @info "skipping unscaled $TA × $TB (unsupported on this GPU)"
            continue
        end
        dD = CUDACore.zeros(Float32, M, N)
        matmul!(dD, CuArray(A), CuArray(B), plan)
        @test Array(dD) ≈ Float64.(A)' * Float64.(B) rtol = 1e-3
    end
end

@testset "E5M2 × E5M2 is rejected" begin
    res = try
        MatmulPlan(; M, N, K, typeA = Float8_E5M2, typeB = Float8_E5M2,
                   typeD = Float32, transA = 'T', lda = K, ldb = K)
    catch err
        err
    end
    @test !(res isa MatmulPlan)
end

@testset "mxfp8, :vec32_ue8m0" begin
    A, B = fp8_operands(Float8_E4M3FN, Float8_E4M3FN)
    plan = try_plan(; M, N, K, typeA = Float8_E4M3FN, typeB = Float8_E4M3FN,
                    typeD = Float32, transA = 'T', lda = K, ldb = K,
                    scaleA = :vec32_ue8m0, scaleB = :vec32_ue8m0)
    if plan === nothing
        @info "skipping :vec32_ue8m0 MXFP8 (unsupported on this GPU)"
    else
        dD = CUDACore.zeros(Float32, M, N)
        # E8M0 byte 127 = 2^0; one scale per 32 K-elements per output row/col
        sA = fill!(CuArray{UInt8}(undef, (K ÷ 32) * M), 0x7f)
        sB = fill!(CuArray{UInt8}(undef, (K ÷ 32) * N), 0x7f)
        matmul!(dD, CuArray(A), CuArray(B), plan; scaleA = sA, scaleB = sB)
        @test Array(dD) ≈ Float64.(A)' * Float64.(B) rtol = 1e-4
    end
end

@testset "nvfp4, :vec16_ue4m3" begin
    # raw-byte NVFP4 payloads: 0x22 packs two E2M1 values of 1.0 per byte, so
    # op(A)⋅op(B) is all-ones times all-ones and every D entry is exactly K
    R_4F_E2M1 = convert(cudaDataType, Float4_E2M1FN)
    plan = try_plan(; M, N, K, typeA = R_4F_E2M1, typeB = R_4F_E2M1,
                    typeD = Float32, transA = 'T', lda = K, ldb = K,
                    scaleA = :vec16_ue4m3, scaleB = :vec16_ue4m3)
    if plan === nothing
        @info "skipping :vec16_ue4m3 NVFP4 (unsupported on this GPU)"
    else
        dA = fill!(CuArray{UInt8}(undef, (K ÷ 2) * M), 0x22)
        dB = fill!(CuArray{UInt8}(undef, (K ÷ 2) * N), 0x22)
        # UE4M3 byte 0x38 = 1.0; one scale per 16 K-elements
        sA = fill!(CuArray{UInt8}(undef, (K ÷ 16) * M), 0x38)
        sB = fill!(CuArray{UInt8}(undef, (K ÷ 16) * N), 0x38)
        dD = CUDACore.zeros(Float32, M, N)
        matmul!(dD, dA, dB, plan; scaleA = sA, scaleB = sB)
        @test Array(dD) ≈ fill(Float32(K), M, N)
    end
end
