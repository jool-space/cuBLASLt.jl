# Plan cache: Dict-backed, keyed on (device, every-plan-kwarg), lock-protected,
# bounded with wholesale eviction — deliberately dumb for v0.1. Exact-shape
# keying: Lt heuristics are shape-dependent and pretending otherwise produces
# silently bad algorithm choices.

const MAX_CACHED_PLANS = 256

const plan_cache = Dict{Any,MatmulPlan}()
const plan_cache_lock = ReentrantLock()

function cached_plan(f, key)
    lock(plan_cache_lock) do
        plan = get(plan_cache, key, nothing)
        if plan === nothing
            length(plan_cache) >= MAX_CACHED_PLANS && empty!(plan_cache)
            plan = plan_cache[key] = f()
        end
        return plan
    end
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
