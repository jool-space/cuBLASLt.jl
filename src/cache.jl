# Plan cache: a memoized *construction*, keyed exactly (device × every plan
# kwarg), lock-protected. Exact-shape keying is deliberate: Lt heuristics are
# shape-dependent and pretending otherwise produces silently bad algorithm
# choices.
#
# Two properties worth stating, because both were once the other way round:
#
#   Unbounded. The cache holds one entry per distinct configuration the program
#   actually runs — bounded in practice by the program's shape diversity, and a
#   plan is a handful of host-side descriptors. The previous 256-entry bound
#   evicted *wholesale* (`empty!`), so a workload with 257 live configurations
#   thrashed permanently, rebuilding every plan on every call. A memo whose
#   pathological case is worse than no memo is not a memo. `empty_plan_cache!`
#   remains, for benchmarking hygiene.
#
#   Negative results are cached. An `UnsupportedConfigError` means "this device
#   and this library have no algorithm for this configuration" — a stable fact
#   about the world, reached only after a full (not cheap) heuristic query. A
#   router that probes cuBLASLt and falls back elsewhere would otherwise re-pay
#   that query on every call, forever. So the error is stored and rethrown on
#   hit. (Argument errors are *not* cached: those say the caller spelled
#   something wrong, and the answer can change with the next call's arguments.)

const plan_cache = Dict{Any,Union{MatmulPlan,UnsupportedConfigError}}()
const plan_cache_lock = ReentrantLock()

# Returns the cached plan, or the cached `UnsupportedConfigError` — the caller
# decides whether an unsupported configuration throws (`cached_plan`) or routes
# elsewhere (`matmul_supported`).
function cached_entry(f, key)
    lock(plan_cache_lock) do
        get!(plan_cache, key) do
            try
                f()
            catch e
                e isa UnsupportedConfigError || rethrow()
                e
            end
        end
    end
end

function cached_plan(f, key)
    entry = cached_entry(f, key)
    entry isa UnsupportedConfigError && throw(entry)
    return entry::MatmulPlan
end

"""
    cuBLASLt.empty_plan_cache!()

Empty the cache used by the planless `matmul!` methods (benchmarking hygiene;
freed plans are reclaimed by the GC).
"""
function empty_plan_cache!()
    lock(plan_cache_lock) do
        empty!(plan_cache)
    end
    return nothing
end
