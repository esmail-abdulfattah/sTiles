"""
    sTiles

Julia interface to the sTiles sparse Cholesky / selected-inverse framework
(KAUST). It is a thin wrapper over the prebuilt `libstiles` shared library:
the matrix is marshalled across the C boundary and every computation happens
inside the solver, so results are identical to the Python and R bindings.

No build step and no compiler: the library for your platform is located (or
downloaded from the project's GitHub Release and cached) when the package
loads.

```julia
using sTiles, LinearAlgebra, SparseArrays

F = sTiles.cholesky(Q; cores = 4, inverse = true)   # Q: SparseMatrixCSC, SPD
logdet(F)                # log|Q|
x = F \\ b                # Q x = b
v = selinv_diag(F)       # diag(Q^-1), the marginal variances
close(F)
```

Reuse the analysis across matrices that share a sparsity pattern:

```julia
F = sTiles.analyze(Q0; cores = 4)
for theta in grid
    sTiles.factorize!(F, build_Q(theta))     # numeric phase only
    ll = ... logdet(F) ...
end
close(F)
```

Indices are 1-based and refer to the original ordering of the input matrix;
the solver's fill-reducing permutation is undone for you.
"""
module sTiles

using Libdl
using Downloads
using SparseArrays
using SparseArrays: getcolptr, rowvals, nonzeros
using LinearAlgebra
import p7zip_jll

export sTilesError, selinv!, selinv_diag, selinv_elm, selinv_row, solve, solve!

"""
    sTilesError(msg)

Raised for every failure reported by the solver or by this package's checks.
"""
struct sTilesError <: Exception
    msg::String
end
Base.showerror(io::IO, e::sTilesError) = print(io, "sTilesError: ", e.msg)

include("library.jl")
include("ffi.jl")
include("factor.jl")

function __init__()
    # Suppress the one-time banner libstiles prints on first use. Set before
    # the library loads so its getenv() sees it; export STILES_NO_BANNER=0 to
    # restore it.
    haskey(ENV, "STILES_NO_BANNER") || (ENV["STILES_NO_BANNER"] = "1")
    handle, path = _load_library()
    LIBHANDLE[] = handle
    LIBPATH[] = path
    _resolve_symbols(handle)
    return nothing
end

end # module
