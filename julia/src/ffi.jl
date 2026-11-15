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
#
# Symbols are resolved per loaded copy of the library (see Lib in library.jl):
# a factor calls only into its own copy, so the per-factor wrappers below take
# that copy first. Version, ordering setters and the memory estimate use the
# first copy.

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
    syms = Dict{Symbol,Ptr{Cvoid}}()
    missing_syms = Symbol[]
    for name in REQUIRED_SYMBOLS
        p = Libdl.dlsym(handle, name; throw_error = false)
        p === nothing || p == C_NULL ? push!(missing_syms, name) : (syms[name] = p)
    end
    for name in OPTIONAL_SYMBOLS
        p = Libdl.dlsym(handle, name; throw_error = false)
        (p === nothing || p == C_NULL) || (syms[name] = p)
    end
    isempty(missing_syms) || throw(sTilesError(
        "the loaded libstiles is missing " * join(missing_syms, ", ") *
        "; it is too old for this package (loaded $(LIBPATH[]))"))
    return syms
end

@inline function _fp(L::Lib, name::Symbol)
    p = get(L.syms, name, C_NULL)
    p == C_NULL && throw(sTilesError(
        "the loaded libstiles does not export $name (loaded $(LIBPATH[]))"))
    return p
end

@inline function _fp(name::Symbol)
    _ensure_loaded()
    return _fp(POOL[1], name)
end

# -- version, logging, global configuration ---------------------------------
function _version()
    p = ccall(_fp(:sTiles_get_version), Cstring, ())
    return p == C_NULL ? "unknown" : unsafe_string(p)
end
_set_log_level(L::Lib, v::Integer)     = ccall(_fp(L, :sTiles_set_log_level), Cvoid, (Cint,), v)
_expert_user(L::Lib)                 = ccall(_fp(L, :sTiles_expert_user), Cvoid, ())
_set_tile_size(L::Lib, v::Integer)     = ccall(_fp(L, :sTiles_set_tile_size), Cvoid, (Cint,), v)
_set_tile_type_mode(L::Lib, v::Integer) = ccall(_fp(L, :sTiles_set_tile_type_mode), Cvoid, (Cint,), v)
_set_ordering_mode(v::Integer) = ccall(_fp(:sTiles_set_ordering_mode), Cvoid, (Cint,), v)
_force_ND(v::Integer)          = ccall(_fp(:sTiles_force_ND), Cvoid, (Cint,), v)
_auto_tile_size()              = ccall(_fp(:sTiles_get_auto_tile_size), Cint, ())

# -- lifecycle ---------------------------------------------------------------
_create(L::Lib, h, ng, calls, cores, ctype, ginv) =
    ccall(_fp(L, :sTiles_create), Cint,
          (Ptr{Ptr{Cvoid}}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Bool}),
          h, ng, calls, cores, ctype, ginv)

_assign_graph_one_call(L::Lib, g, c, h, n, nnz, row, col) =
    ccall(_fp(L, :sTiles_assign_graph_one_call), Cint,
          (Cint, Cint, Ptr{Ptr{Cvoid}}, Cint, Cint, Ptr{Cint}, Ptr{Cint}),
          g, c, h, n, nnz, row, col)

_init_group(L::Lib, g, h)   = ccall(_fp(L, :sTiles_init_group), Cint, (Cint, Ptr{Ptr{Cvoid}}), g, h)
_assign_values(L::Lib, g, c, h, x) =
    ccall(_fp(L, :sTiles_assign_values), Cint,
          (Cint, Cint, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}), g, c, h, x)
_bind(L::Lib, g, c, h)      = ccall(_fp(L, :sTiles_bind), Cint, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_unbind(L::Lib, g, c, h)    = ccall(_fp(L, :sTiles_unbind), Cint, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_chol(L::Lib, g, c, h)      = ccall(_fp(L, :sTiles_chol), Cint, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_selinv(L::Lib, g, c, h)    = ccall(_fp(L, :sTiles_selinv), Cint, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_clear_selinv(L::Lib, g, c, h) = ccall(_fp(L, :sTiles_clear_selinv), Cint, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_free_group(L::Lib, g)      = ccall(_fp(L, :sTiles_freeGroup), Cvoid, (Cint,), g)
_quit(L::Lib)             = ccall(_fp(L, :sTiles_quit), Cvoid, ())

# -- results -----------------------------------------------------------------
_get_logdet(L::Lib, g, c, h) = ccall(_fp(L, :sTiles_get_logdet), Cdouble, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_get_nnz_factor(L::Lib, g, c, h) = ccall(_fp(L, :sTiles_get_nnz_factor), Clonglong, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_get_chol_timing(L::Lib, g, c, h) = ccall(_fp(L, :sTiles_get_chol_timing), Cdouble, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)
_get_selinv_timing(L::Lib, g, c, h) = ccall(_fp(L, :sTiles_get_selinv_timing), Cdouble, (Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, h)

_get_selinv_elm(L::Lib, g, c, i, j, h) =
    ccall(_fp(L, :sTiles_get_selinv_elm), Cdouble,
          (Cint, Cint, Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, i, j, h)
_get_chol_elm(L::Lib, g, c, i, j, h) =
    ccall(_fp(L, :sTiles_get_chol_elm), Cdouble,
          (Cint, Cint, Cint, Cint, Ptr{Ptr{Cvoid}}), g, c, i, j, h)
_get_selinv_row(L::Lib, g, c, node, nbrs, m, h) =
    ccall(_fp(L, :sTiles_get_selinv_row), Ptr{Cdouble},
          (Cint, Cint, Cint, Ptr{Cint}, Cint, Ptr{Ptr{Cvoid}}), g, c, node, nbrs, m, h)

_get_logical_element_perm(L::Lib, g, c, h, out) =
    ccall(_fp(L, :sTiles_get_logical_element_perm), Cint,
          (Cint, Cint, Ptr{Ptr{Cvoid}}, Ptr{Cint}), g, c, h, out)

# -- solves (B is column-major, original order, overwritten in place) --------
_solve_LLT(L::Lib, g, c, h, b, nrhs) =
    ccall(_fp(L, :sTiles_solve_LLT), Cint,
          (Cint, Cint, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Cint), g, c, h, b, nrhs)
_solve_L(L::Lib, g, c, h, b, nrhs) =
    ccall(_fp(L, :sTiles_solve_L), Cint,
          (Cint, Cint, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Cint), g, c, h, b, nrhs)
_solve_LT(L::Lib, g, c, h, b, nrhs) =
    ccall(_fp(L, :sTiles_solve_LT), Cint,
          (Cint, Cint, Ptr{Ptr{Cvoid}}, Ptr{Cdouble}, Cint), g, c, h, b, nrhs)

# -- memory estimate (static, no handle) ------------------------------------
_estimate_memory(n, nnz, ts, variant, inv, nd) =
    ccall(_fp(:sTiles_estimate_memory), Cdouble,
          (Cint, Cint, Cint, Cint, Cint, Cint), n, nnz, ts, variant, inv, nd)
