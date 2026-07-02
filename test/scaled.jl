Random.seed!(3)

# Block-scaled and tensor-wide-scaled narrow-type matmuls. FP8/FP4 kernels
# want the TN, K-major orientation: A stored K×M, B stored K×N, both with
# leading dimension K.
#
# Scale arrays here are deliberately *constant*: on sm100+ cuBLASLt reads
# block scales through a swizzled layout, and constant arrays are
# swizzle-invariant, so these tests need no swizzle machinery (and no
# Microscaling dependency). Any test with non-constant scales must swizzle.
#
# The swizzled layout stores scales in 128(outer)×4(inner)-entry tiles, and
# storage must be padded to whole tiles — undersized arrays make cuBLASLt
# read out of bounds. A constant fill over the padded allocation is both
# swizzle- and padding-invariant.

const M, N, K = 64, 32, 128

fp8_operands(TA, TB) = (TA.(rand(Float32, K, M)), TB.(rand(Float32, K, N)))

# `outer` is the output dimension (M for A, N for B); `inner` the number of
# scales along K (K ÷ block size)
block_scales(byte::UInt8, outer, inner) =
    fill!(CuArray{UInt8}(undef, 128 * cld(outer, 128) * 4 * cld(inner, 4)), byte)

@testset "fp8, tensor-wide :scalar_f32" begin
    if CC < v"8.9"
        @info "skipping :scalar_f32 FP8 (requires CC ≥ 8.9, device is $CC)"
    else
        A, B = fp8_operands(Float8_E4M3FN, Float8_E4M3FN)
        plan = MatmulPlan(; M, N, K, typeA = Float8_E4M3FN, typeB = Float8_E4M3FN,
                          typeD = Float32, transA = 'T', lda = K, ldb = K,
                          scale_modeA = :scalar_f32, scale_modeB = :scalar_f32)
        dD = CUDACore.zeros(Float32, M, N)
        sA, sB = CuArray([2.0f0]), CuArray([0.5f0])
        plan(dD, CuArray(A), CuArray(B); scaleA = sA, scaleB = sB)
        @test Array(dD) ≈ 2.0f0 * 0.5f0 * (Float64.(A)' * Float64.(B)) rtol = 1e-3

        # planless: length-1 Float32 scale prototypes infer :scalar_f32
        fill!(dD, 0)
        matmul!(dD, CuArray(A), CuArray(B); transA = 'T', scaleA = sA, scaleB = sB)
        @test Array(dD) ≈ 2.0f0 * 0.5f0 * (Float64.(A)' * Float64.(B)) rtol = 1e-3
    end
end

@testset "fp8, unscaled and mixed element types" begin
    if CC < v"8.9"
        @info "skipping unscaled FP8 (requires CC ≥ 8.9, device is $CC)"
    else
        for (TA, TB) in ((Float8_E4M3FN, Float8_E4M3FN), (Float8_E4M3FN, Float8_E5M2),
                         (Float8_E5M2, Float8_E4M3FN))
            A, B = fp8_operands(TA, TB)
            plan = MatmulPlan(; M, N, K, typeA = TA, typeB = TB, typeD = Float32,
                              transA = 'T', lda = K, ldb = K)
            dD = CUDACore.zeros(Float32, M, N)
            plan(dD, CuArray(A), CuArray(B))
            @test Array(dD) ≈ Float64.(A)' * Float64.(B) rtol = 1e-3
        end
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
    if CC < v"10.0"
        @info "skipping :vec32_ue8m0 MXFP8 (requires CC ≥ 10.0, device is $CC)"
    else
        A, B = fp8_operands(Float8_E4M3FN, Float8_E4M3FN)
        plan = MatmulPlan(; M, N, K, typeA = Float8_E4M3FN, typeB = Float8_E4M3FN,
                          typeD = Float32, transA = 'T', lda = K, ldb = K,
                          scale_modeA = :vec32_ue8m0, scale_modeB = :vec32_ue8m0)
        dD = CUDACore.zeros(Float32, M, N)
        # E8M0 byte 127 = 2^0; one scale per 32 K-elements per output row/col
        sA = block_scales(0x7f, M, K ÷ 32)
        sB = block_scales(0x7f, N, K ÷ 32)
        plan(dD, CuArray(A), CuArray(B); scaleA = sA, scaleB = sB)
        @test Array(dD) ≈ Float64.(A)' * Float64.(B) rtol = 1e-4
    end
end

@testset "nvfp4, :vec16_ue4m3" begin
    if CC < v"10.0"
        @info "skipping :vec16_ue4m3 NVFP4 (requires CC ≥ 10.0, device is $CC)"
    else
        # raw-byte NVFP4 payloads: 0x22 packs two E2M1 values of 1.0 per byte, so
        # op(A)⋅op(B) is all-ones times all-ones and every D entry is exactly K
        R_4F_E2M1 = convert(cudaDataType, Float4_E2M1FN)
        plan = MatmulPlan(; M, N, K, typeA = R_4F_E2M1, typeB = R_4F_E2M1,
                          typeD = Float32, transA = 'T', lda = K, ldb = K,
                          scale_modeA = :vec16_ue4m3, scale_modeB = :vec16_ue4m3)
        dA = fill!(CuArray{UInt8}(undef, (K ÷ 2) * M), 0x22)
        dB = fill!(CuArray{UInt8}(undef, (K ÷ 2) * N), 0x22)
        # UE4M3 byte 0x38 = 1.0; one scale per 16 K-elements
        sA = block_scales(0x38, M, K ÷ 16)
        sB = block_scales(0x38, N, K ÷ 16)
        dD = CUDACore.zeros(Float32, M, N)
        plan(dD, dA, dB; scaleA = sA, scaleB = sB)
        @test Array(dD) ≈ fill(Float32(K), M, N)
    end
end
