@testset "BitPacking operand traits" begin
    T = Float4_E2M1FN
    packed = NarrowArray{T}(fill(T(1), 16, 8, 2))
    data = cuBLASLt.ltdata(packed)

    @test Base.get_extension(cuBLASLt, :BitPackingExt) !== nothing
    @test data === packed
    @test eltype(data) === T
    @test size(data) == (16, 8, 2)
    @test ndims(data) == 3
    @test cuBLASLt.ltstride(data, 1) == 1
    @test cuBLASLt.ltstride(data, 2) == 16
    @test cuBLASLt.ltstride(data, 3) == 128
    @test cuBLASLt.ltstride(data, 4) == 256

    stepped = NarrowArray(@view parent(packed)[1:2:end, :, :])
    @test_throws ArgumentError cuBLASLt.ltstride(stepped, 1)

    dpacked = Narrow{T}.(CuArray(fill(T(1), 16, 8)))
    @test UInt(cuBLASLt.ltptr(dpacked)) == UInt(pointer(parent(dpacked)))
end
