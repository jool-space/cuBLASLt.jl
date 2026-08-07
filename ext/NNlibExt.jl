module NNlibExt

using cuBLASLt
using NNlib
using NNlib: BatchedTranspose, BatchedAdjoint, BatchedAdjOrTrans

cuBLASLt.activation_symbol(::typeof(relu)) = :relu
cuBLASLt.activation_symbol(::typeof(gelu)) = :gelu

# orientation traits for the batched wrappers: everything downstream (layouts,
# strides, batching) derives from the unwrapped parent, exactly as for
# PermutedDimsArray{<:Any,3,(2,1,3)}. 'C' ≡ 'T' for real element types.
cuBLASLt.unwrap_op(x::BatchedTranspose) = 'T', parent(x)
cuBLASLt.unwrap_op(x::BatchedAdjoint) = 'C', parent(x)
cuBLASLt.apply_operand(x::BatchedAdjOrTrans, trans::Char, name::Symbol) =
    cuBLASLt.checked_unwrap(x, trans, name)

end
