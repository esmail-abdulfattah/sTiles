# sTiles.jl

Julia interface to the **sTiles** sparse Cholesky / selected-inverse framework.
A thin wrapper over the prebuilt `libstiles` shared library: the matrix is
marshalled across the C boundary and every computation happens inside the
solver, so results are identical to the Python and R bindings. No build step and no compiler on the user's machine.

```julia
using sTiles, LinearAlgebra, SparseArrays

F = sTiles.cholesky(Q; cores = 4, inverse = true)   # Q: sparse SPD matrix
logdet(F)                 # log|Q|
x = F \ b                 # Q x = b
v = selinv_diag(F)        # diag(Q^-1), the marginal variances
close(F)
```

## Install

```julia
julia> ]                        # package mode
pkg> add https://github.com/esmail-abdulfattah/sTiles:julia
```

The package fetches the matching solver from the project's GitHub Release the
first time it loads and caches it under `~/.cache/sTiles/<release>/` (the same
cache the Python and R packages use). To use a specific build instead, set
`STILES_LIB=/path/to/libstiles.so` before `using sTiles`. Search order and
every override are listed at the top of [`src/library.jl`](src/library.jl).
From a checkout, `pkg> dev path/to/julia` uses the package in place and finds
a library in a sibling `binaries/` tree or the repo `lib/` automatically.

Requires Julia 1.10 or later (tested on 1.10 and 1.11). Depends on standard libraries only.

## API

Everything user-facing lives in the `sTiles` module. The four `selinv_*`
verbs, `solve`, `solve!` and `sTilesError` are exported; the rest is reached
as `sTiles.name` so nothing clashes with `LinearAlgebra`.

| operation                          | Julia                                   |
| ---------------------------------- | --------------------------------------- |
| analyze + factorize (one shot)     | `sTiles.cholesky(Q; ...)`               |
| analysis only (phase 1)            | `sTiles.analyze(Q; ...)`                |
| numeric Cholesky (phase 2)         | `sTiles.factorize!(F)`                  |
| reuse analysis, new values         | `sTiles.factorize!(F, Q2)` / `sTiles.update!(F, Q2)` |
| log-determinant                    | `logdet(F)`, `logabsdet(F)`, `det(F)`   |
| selected inverse (phase 3)         | `selinv!(F)`                            |
| marginal variances, `diag(Q^-1)`   | `selinv_diag(F)`                        |
| one inverse entry `(Q^-1)[i, j]`   | `selinv_elm(F, i, j)`                   |
| inverse row over neighbours        | `selinv_row(F, i, nbrs)`                |
| solve `Q x = b` (multi-RHS)        | `F \ b`, `solve(F, b)`, `ldiv!(F, b)`   |
| forward / backward solve           | `solve(F, b; system = :L)` / `:Lt`      |
| entry of the factor `L[i, j]`      | `sTiles.chol_elm(F, i, j)`              |
| fill-reducing permutation          | `sTiles.permutation(F)`                 |
| `nnz(L)`, timings, state           | `nnz(F)`, `sTiles.chol_time(F)`, `sTiles.selinv_time(F)`, `sTiles.analyze_time(F)`, `sTiles.summary(F)` |
| free the handle                    | `close(F)`                              |
| solver version / location          | `sTiles.version()`, `sTiles.library_path()` |
| refresh the downloaded solver      | `sTiles.update_library()`, `sTiles.clear_cache()` |

Keyword arguments to `cholesky` and `analyze`: `cores` (worker threads,
default 1), `mode` (`:auto`, `:dense`, `:semisparse`, `:sparse`; default
`:auto`), `tile_size` (default 40, `-1` lets the solver choose), `inverse`
(reserve selected-inverse storage, default `false`), `log_level` (`-1` silent,
`0` timing markers, `1` info, `2` debug, `3` trace).

`F` is a `LinearAlgebra.Factorization`, so `size`, `det`, `logdet`, `\` and
`ldiv!` behave as they do for any factorization. Indices are 1-based and refer
to the original ordering of the input matrix; right-hand sides go in and come
out in that ordering. Only the lower triangle of `Q` is read.

### Two phases: analysis vs numeric

The analysis (ordering bake-off + tile layout) depends only on the sparsity
pattern and usually dominates the cost; the numeric Cholesky is cheap by
comparison. Split them to time them apart, or to sweep many matrices that
share one pattern:

```julia
F = sTiles.analyze(Q; cores = 4, inverse = true)   # once per pattern
sTiles.analyze_time(F)
for theta in grid
    sTiles.factorize!(F, build_Q(theta))           # once per set of values
    ll = ... logdet(F) ...
end
close(F)
```

The new matrix must have the **same stored pattern**: keep structural zeros
in place (a `SparseMatrixCSC` keeps stored zeros) so every matrix in the sweep
matches. A different pattern raises an `sTilesError`; build a new factor for it.

### Selected inverse

`Z = Q^-1` restricted to the pattern of the Cholesky factor, in one Takahashi
sweep. Build with `inverse = true`:

```julia
F = sTiles.cholesky(Q; inverse = true)
selinv!(F)                     # explicit + timeable; else lazy on first query
selinv_diag(F)                 # marginal variances, length n
selinv_elm(F, i, j)            # exact in-pattern, exactly 0.0 outside it
selinv_row(F, i, nbrs)         # (Q^-1)[i, k] for k in nbrs
```

### One factor at a time

`libstiles` keeps its thread teams in process-global state, so one factor may
be live per process. Close it (`close(F)`) before building the next one. A
factor that goes out of scope is released by the garbage collector, but
`close` is immediate and is what you want in a loop.

## Examples and tests

```bash
cd julia
julia --project=. examples/quickstart.jl
julia --project=. examples/timing_split.jl [matrix.mtx]
julia --project=. examples/selinv_access.jl
julia --project=. -e 'using Pkg; Pkg.test()'

# longer: matrix families, config sweep, RHS widths, reuse, threads, INLA graphs vs CHOLMOD
julia -t 8 --project=. test/extensive.jl --dump /tmp/stiles-dump path/to/precision.mtx
# then recompute the dumped cases with the Python package on the same library
STILES_LIB=... PYTHONPATH=../python python3 test/cross_check.py /tmp/stiles-dump
```

If Julia itself segfaults on `using Downloads` or `using SparseArrays`, an
`LD_LIBRARY_PATH` in the shell (Intel oneAPI `setvars.sh` is the usual source)
is shadowing Julia's bundled libraries; run with `env -u LD_LIBRARY_PATH julia`.
