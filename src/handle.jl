# One cublasLtHandle_t per CuContext, lazily created, process-lifetime.
#
# Unlike the classic cuBLAS handle, an Lt handle carries no stream, pointer
# mode, or math mode — those all live in per-call descriptors — so there is no
# task-local state to re-sync and nothing to return to an idle pool.

const HANDLES = Dict{CuContext,cublasLtHandle_t}()
const HANDLE_LOCK = ReentrantLock()

"""
    cuBLASLt.handle() -> cublasLtHandle_t

The cuBLASLt handle for the current CUDA context.
"""
function handle()
    ctx = CUDACore.context()
    lock(HANDLE_LOCK) do
        get!(HANDLES, ctx) do
            ref = Ref{cublasLtHandle_t}()
            cublasLtCreate(ref)
            ref[]
        end
    end
end
