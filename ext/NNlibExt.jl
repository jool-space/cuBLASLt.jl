module NNlibExt

using cuBLASLt
using NNlib

cuBLASLt.activation_symbol(::typeof(relu)) = :relu
cuBLASLt.activation_symbol(::typeof(gelu)) = :gelu

end
