module BitPackingExt

using BitPacking: NarrowArray
import cuBLASLt

cuBLASLt.ltptr(A::NarrowArray) = cuBLASLt.ltptr(parent(A))

function cuBLASLt.ltstride(A::NarrowArray{<:Any,N,L}, d::Integer) where {N,L}
    if d == 1
        physical = cuBLASLt.ltstride(parent(A), 1)
        physical == 1 || throw(ArgumentError(
            "a NarrowArray operand needs contiguous packed chunks in dimension 1; " *
            "its parent has stride $physical"))
        return 1
    end
    return L * cuBLASLt.ltstride(parent(A), d)
end

end # module
