# Raw FFI surface: every libstiles entry point this package calls.
#
# The symbols are resolved once with dlsym in __init__ and called through the
# resulting function pointers, so the library path is free to be decided at
# runtime (ccall's (:name, lib) form needs a constant). Nothing here allocates
# or interprets; the meaning lives in factor.jl.
#
# The handle is an opaque void* and every lifecycle call takes void**, so the
# Julia side keeps one Ref{Ptr{Cvoid}} per factorization and passes it
# throughout. Julia's collector does not move objects, so that slot's address
# is stable for as long as the owning object is reachable, which is exactly the
# contract the C API wants.

const SYMBOLS = Dict{Symbol,Ptr{Cvoid}}()

# Every symbol resolved at load time. A build without one of the optional
# entries still loads; the error surfaces only if that call is made.
const REQUIRED_SYMBOLS = [
    :sTiles_get_version, :sTiles_set_log_level, :sTiles_expert_user,
    :sTiles_set_tile_size, :sTiles_set_tile_type_mode,
    :sTiles_create, :sTiles_assign_graph_one_call, :sTiles_init_group,
    :sTiles_assign_values, :sTiles_bind, :sTiles_unbind, :sTiles_chol,
    :sTiles_selinv, :sTiles_freeGroup, :sTiles_quit,
    :sTiles_get_logdet, :sTiles_get_nnz_factor, :sTiles_get_selinv_elm,
    :sTiles_get_chol_elm, :sTiles_get_selinv_row, :sTiles_clear_selinv,
    :sTiles_get_chol_timing, :sTiles_get_selinv_timing,
    :sTiles_solve_LLT, :sTiles_solve_L, :sTiles_solve_LT,
]
const OPTIONAL_SYMBOLS = [
    :sTiles_set_ordering_mode, :sTiles_force_ND, :sTiles_return_tile_size,
    :sTiles_get_auto_tile_size, :sTiles_get_logical_element_perm,
    :sTiles_estimate_memory,
]

function _resolve_symbols(handle::Ptr{Cvoid})
    empty!(SYMBOLS)
    missing_syms = Symbol[]
    for name in REQUIRED_SYMBOLS
        p = Libdl.dlsym(handle, name; throw_error = false)
        p === nothing || p == C_NULL ? push!(missing_syms, name) : (SYMBOLS[name] = p)
    end
    for name in OPTIONAL_SYMBOLS
        p = Libdl.dlsym(handle, name; throw_error = false)
        (p === nothing || p == C_NULL) || (SYMBOLS[name] = p)
    end
    isempty(missing_syms) || throw(sTilesError(
        "the loaded libstiles is missing " * join(missing_syms, ", ") *
        "; it is too old for this package (loaded $(LIBPATH[]))"))
    return nothing
end

@inline function _fp(name::Symbol)
    p = get(SYMBOLS, name, C_NULL)
    p == C_NULL && throw(sTilesError(
        "the loaded libstiles does not export $name (loaded $(LIBPATH[]))"))
    return p
end

# -- version, logging, global configuration ---------------------------------
function _version()
    p = ccall(_fp(:sTiles_get_version), Cstring, ())
    return p == C_NULL ? "unknown" : unsafe_string(p)
end
_set_log_level(v::Integer)     = ccall(_fp(:sTiles_set_log_level), Cvoid, (Cint,), v)
_expert_user()                 = ccall(_fp(:sTiles_expert_user), Cvoid, ())
_set_tile_size(v::Integer)     = ccall(_fp(:sTiles_set_tile_size), Cvoid, (Cint,), v)
_set_tile_type_mode(v::Integer) = ccall(_fp(:sTiles_set_tile_type_mode), Cvoid, (Cint,), v)
_set_ordering_mode(v::Integer) = ccall(_fp(:sTiles_set_ordering_mode), Cvoid, (Cint,), v)
_force_ND(v::Integer)          = ccall(_fp(:sTiles_force_ND), Cvoid, (Cint,), v)
_auto_tile_size()              = ccall(_fp(:sTiles_get_auto_tile_size), Cint, ())

# -- lifecycle ---------------------------------------------------------------
_create(h, ng, calls, cores, ctype, ginv) =
    ccall(_fp(:sTiles_create), Cint,
          (Ptr{Ptr{Cvoid}}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Bool}),
          h, ng, calls, cores, ctype, ginv)

_assign_graph_one_call(g, c, h, n, nnz, row, col) =
    ccall(_fp(:sTiles_assign_graph_one_call), Cint,
          (Cint, Cint, Ptr{Ptr{Cvoid}}, Cint, Cint, Ptr{Cint}, Ptr{Cint}),
          g, c, h, n, nnz, row, col)

_init_group(g, h)   = ccall(_fp(:sTiles_init_group), Cint, (Cint, Ptr{Ptr{Cvoid}}), g, h)
_assign_values(g, c, h, x) =
    ccall(_fp(:sTiles_assign_values), Cint,
          (Cint, Cint, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}), g, c, h, x)
_bind(g, c, h)      = ccall(_fp(:sTiles_bind), Cint, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_unbind(g, c, h)    = ccall(_fp(:sTiles_unbind), Cint, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_chol(g, c, h)      = ccall(_fp(:sTiles_chol), Cint, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_selinv(g, c, h)    = ccall(_fp(:sTiles_selinv), Cint, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_clear_selinv(g, c, h) = ccall(_fp(:sTiles_clear_selinv), Cint, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_free_group(g)      = ccall(_fp(:sTiles_freeGroup), Cvoid, (Cint,), g)
_quit()             = ccall(_fp(:sTiles_quit), Cvoid, ())

# -- results -----------------------------------------------------------------
_get_logdet(g, c, h) = ccall(_fp(:sTiles_get_logdet), Cdouble, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_get_nnz_factor(g, c, h) = ccall(_fp(:sTiles_get_nnz_factor), Clonglong, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_get_chol_timing(g, c, h) = ccall(_fp(:sTiles_get_chol_timing), Cdouble, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_get_selinv_timing(g, c, h) = ccall(_fp(:sTiles_get_selinv_timing), Cdouble, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)

_get_selinv_elm(g, c, i, j, h) =
    ccall(_fp(:sTiles_get_selinv_elm), Cdouble,
          (Cint, Cint, Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, i, j, h)
_get_chol_elm(g, c, i, j, h) =
    ccall(_fp(:sTiles_get_chol_elm), Cdouble,
          (Cint, Cint, Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, i, j, h)
_get_selinv_row(g, c, node, nbrs, m, h) =
    ccall(_fp(:sTiles_get_selinv_row), Ptr{Cdouble},
          (Cint, Cint, Cint, Ptr{Cint}, Cint, Ptr{Ptr{Cvoid}}), g, c, node, nbrs, m, h)

_get_logical_element_perm(g, c, h, out) =
    ccall(_fp(:sTiles_get_logical_element_perm), Cint,
          (Cint, Cint, Ptr{Ptr{Cvoid}}, Ptr{Cint}), g, c, h, out)

# -- solves (B is column-major, original order, overwritten in place) --------
_solve_LLT(g, c, h, b, nrhs) =
    ccall(_fp(:sTiles_solve_LLT), Cint,
          (Cint, Cint, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Cint), g, c, h, b, nrhs)
_solve_L(g, c, h, b, nrhs) =
    ccall(_fp(:sTiles_solve_L), Cint,
          (Cint, Cint, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Cint), g, c, h, b, nrhs)
_solve_LT(g, c, h, b, nrhs) =
    ccall(_fp(:sTiles_solve_LT), Cint,
          (Cint, Cint, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Cint), g, c, h, b, nrhs)

# -- memory estimate (static, no handle) ------------------------------------
_estimate_memory(n, nnz, ts, variant, inv, nd) =
    ccall(_fp(:sTiles_estimate_memory), Cdouble,
          (Cint, Cint, Cint, Cint, Cint, Cint), n, nnz, ts, variant, inv, nd)
