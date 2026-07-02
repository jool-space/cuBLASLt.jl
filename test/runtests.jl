import cuBLASLt
using CUDACore
using ParallelTestRunner

# Each test file runs in its own worker process; `setup.jl` provides the shared
# `using` block in every worker.
const init_code = quote
    include($(joinpath(@__DIR__, "setup.jl")))
end

testsuite = find_tests(@__DIR__)

args = parse_args(ARGS)
if filter_tests!(testsuite, args)
    if CUDACore.functional()
        @info "Running GPU tests" device = CUDACore.name(CUDACore.device()) capability = CUDACore.capability(CUDACore.device()) cublaslt = cuBLASLt.version()
    else
        @warn "CUDA not functional — GPU tests will fail"
    end
    # setup.jl is loaded into every worker via init_code; don't run it standalone.
    filter!(t -> first(t) != "setup", testsuite)
end

runtests(cuBLASLt, ARGS; init_code, testsuite)
