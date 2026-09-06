# The user-facing layer: one live factorization, plus the verbs on it.

# tile_type_mode values (mirror stiles.h)
const MODE_DENSE      = 0
const MODE_SEMISPARSE = 1
const MODE_SPARSE     = 2
const MODE_AUTO       = 3

const MODE_NAMES = Dict(0 => :dense, 1 => :semisparse, 2 => :sparse, 3 => :auto)
const MODE_CODES = Dict(:dense => MODE_DENSE, :semisparse => MODE_SEMISPARSE,
                        :semi => MODE_SEMISPARSE, :sparse => MODE_SPARSE,
                        :auto => MODE_AUTO)

_resolve_mode(mode::Integer) = Int(mode)
_resolve_mode(mode::AbstractString) = _resolve_mode(Symbol(lowercase(mode)))
function _resolve_mode(mode::Symbol)
    haskey(MODE_CODES, mode) || throw(ArgumentError(
        "unknown mode :$mode; use one of $(sort(collect(keys(MODE_CODES))))"))
    return MODE_CODES[mode]
end

"""
    version() -> String

Version string of the loaded `libstiles`.
"""
version() = _version()

# ---------------------------------------------------------------------------
# Matrix -> lower-triangle COO, 0-based, canonical (row, col) order.
#
# The canonical order matters twice over: it lets a later set of values line up
# entry-for-entry with the pattern already analyzed, and it is the same order
# the Python and R bindings build, so all three feed libstiles identically.
# ---------------------------------------------------------------------------
function _lower_coo(Q::SparseMatrixCSC)
    n = size(Q, 1)
    n == size(Q, 2) || throw(ArgumentError("matrix must be square"))
    n <= typemax(Cint) || throw(ArgumentError(
        "n = $n exceeds the 32-bit indexing limit of libstiles"))

    cp, ri, nzv = getcolptr(Q), rowvals(Q), nonzeros(Q)

    # Count the lower-triangle entries per row, then place them by row: a
    # counting sort, O(nnz + n), which lands entries in (row, col) order
    # because the columns are scanned in increasing order.
    counts = zeros(Int, n + 1)
    nnz_low = 0
    @inbounds for j in 1:n, k in cp[j]:(cp[j+1]-1)
        i = ri[k]
        if i >= j
            counts[i] += 1
            nnz_low += 1
        end
    end
    nnz_low <= typemax(Cint) || throw(ArgumentError(
        "nnz = $nnz_low exceeds the 32-bit indexing limit of libstiles"))

    pos = Vector{Int}(undef, n + 1)
    acc = 1
    @inbounds for i in 1:n
        pos[i] = acc
        acc += counts[i]
    end

    row = Vector{Cint}(undef, nnz_low)
    col = Vector{Cint}(undef, nnz_low)
    val = Vector{Float64}(undef, nnz_low)
    @inbounds for j in 1:n, k in cp[j]:(cp[j+1]-1)
        i = ri[k]
        if i >= j
            p = pos[i]
            row[p] = i - 1                 # 0-based for libstiles
            col[p] = j - 1
            val[p] = nzv[k]
            pos[i] = p + 1
        end
    end
    return n, row, col, val
end

_lower_coo(Q::AbstractMatrix) = _lower_coo(sparse(Q))

"""Values only, in the order of an already-analyzed pattern."""
function _values_for(F, Q::AbstractMatrix)
    _, row, col, val = _lower_coo(Q)
    (length(val) == F.nnz && row == F.row && col == F.col) || throw(sTilesError(
        "this matrix has a different sparsity pattern; the analysis can only " *
        "be reused for the pattern it was built from (build a new factor, or " *
        "keep the structural zeros so every pattern in the sweep matches)"))
    return val
end

# ---------------------------------------------------------------------------
# One live factorization.
#
# libstiles keeps its thread teams and worker state in process-global storage,
# so exactly one factor can be live at a time -- the same restriction the
# Python and R bindings enforce.
# ---------------------------------------------------------------------------
const LIVE = Ref(0)
const LIVE_LOCK = ReentrantLock()

"""
    sTiles.Factor

A live sTiles factorization of one symmetric positive-definite matrix.

Build one with [`sTiles.cholesky`](@ref) (analyze and factorize) or
[`sTiles.analyze`](@ref) (analysis only), and release it with `close(F)`.
"""
mutable struct Factor <: LinearAlgebra.Factorization{Float64}
    handle::Base.RefValue{Ptr{Cvoid}}
    n::Int
    nnz::Int
    # libstiles RETAINS the index arrays it is handed and reads them for the
    # lifetime of the handle, so the factor owns them; dropping them here would
    # leave the library reading freed memory.
    row::Vector{Cint}
    col::Vector{Cint}
    values::Vector{Float64}
    group::Cint
    cores::Int
    mode::Int
    inverse::Bool
    factored::Bool
    selinv_done::Bool
    closed::Bool
    analyze_time::Float64
    lock::ReentrantLock
end

function _check(rc::Integer, what::AbstractString)
    rc == 0 || throw(sTilesError("$what failed (status $rc)"))
    return nothing
end

_require_open(F::Factor) =
    F.closed && throw(sTilesError("this factor has been closed"))

function _require_factored(F::Factor)
    _require_open(F)
    F.factored || throw(sTilesError(
        "not factorized yet: the analysis is done, the numeric Cholesky is " *
        "not; call sTiles.factorize!(F)"))
    return nothing
end

"""
    sTiles.analyze(Q; cores=1, mode=:auto, tile_size=40, inverse=false, log_level=-1)

Phase 1 only: the ordering bake-off and tile layout for `Q`'s sparsity pattern.
No numeric values are read and no Cholesky is run, so this is the cost paid once
per pattern; [`sTiles.factorize!`](@ref) then pays only the numeric cost, once
per set of values.

Only the lower triangle of `Q` is used, and `Q` must be symmetric positive
definite when it is factorized.

Keyword arguments

  * `cores`: worker threads for the factorization.
  * `mode`: `:auto`, `:dense`, `:semisparse` or `:sparse` tile regime.
  * `tile_size`: tile size, or `-1` to let the solver choose.
  * `inverse`: reserve storage for the selected inverse. Required by
    [`selinv_diag`](@ref), [`selinv_elm`](@ref) and [`selinv_row`](@ref).
  * `log_level`: verbosity of the solver's own logging (`-1` silent, `0` timing
    markers, `1` info, `2` debug, `3` trace). Errors are always shown.

```julia
F = sTiles.analyze(Q; cores = 4, inverse = true)
sTiles.factorize!(F)                 # numbers enter here
sTiles.update!(F, Q2)                # new values, same pattern, no re-analysis
```
"""
function analyze(Q::AbstractMatrix; cores::Integer = 1, mode = :auto,
                 tile_size::Integer = 40, inverse::Bool = false,
                 log_level::Integer = -1)
    m = _resolve_mode(mode)
    cores >= 1 || throw(ArgumentError("cores must be >= 1"))
    n, row, col, val = _lower_coo(Q)

    lock(LIVE_LOCK)
    try
        LIVE[] == 0 || throw(sTilesError(
            "sTiles handles one matrix at a time: another factor is still " *
            "open in this process. Close it first (close(F)), then build the " *
            "next one."))

        _set_log_level(log_level)
        _expert_user()                  # gates the setters below
        _set_tile_size(tile_size)
        _set_tile_type_mode(m)

        handle = Ref(Ptr{Cvoid}(C_NULL))
        # One group, and it gets the graph below: creating extra groups and
        # leaving them without one makes the team setup fail on the empty group.
        ng = Cint(1)
        calls = Cint[1]
        coresv = Cint[cores]
        ctype = Cint[0]                 # 0 = sparse factorization variant
        ginv = Bool[inverse]
        _check(_create(handle, ng, calls, coresv, ctype, ginv), "sTiles_create")

        F = Factor(handle, n, length(row), row, col, val, Cint(0), Int(cores), m,
                   inverse, false, false, false, 0.0, ReentrantLock())
        LIVE[] += 1
        finalizer(_finalize!, F)

        # Timed here: libstiles reports the Cholesky and selected-inverse times
        # but nothing for the analysis, which is usually the expensive phase.
        t0 = time()
        try
            _check(_assign_graph_one_call(F.group, 0, F.handle, n, F.nnz, F.row, F.col),
                   "sTiles_assign_graph_one_call")
            _check(_init_group(F.group, F.handle), "sTiles_init_group")
        catch
            _teardown!(F)
            rethrow()
        end
        F.analyze_time = time() - t0
        return F
    finally
        unlock(LIVE_LOCK)
    end
end

"""
    sTiles.cholesky(Q; cores=1, mode=:auto, tile_size=40, inverse=false, log_level=-1)

Analyze `Q` and run the numeric Cholesky, returning a live [`sTiles.Factor`](@ref).
Takes the same keyword arguments as [`sTiles.analyze`](@ref).

```julia
F = sTiles.cholesky(Q; cores = 4, inverse = true)
logdet(F)                 # log|Q|
x = F \\ b                 # Q x = b
v = selinv_diag(F)        # diag(Q^-1), the marginal variances
close(F)
```
"""
function cholesky(Q::AbstractMatrix; kwargs...)
    F = analyze(Q; kwargs...)
    try
        factorize!(F)
    catch
        close(F)
        rethrow()
    end
    return F
end

"""
    sTiles.factorize!(F) -> F
    sTiles.factorize!(F, Q) -> F

Run the numeric Cholesky, reusing the analysis. With `Q`, factor a matrix that
shares the analyzed sparsity pattern; without it, factor the values `F` was
built from. This is the phase to repeat in a sweep: the ordering and tile layout
are computed once, every later factorization pays only the numeric cost.
"""
function factorize!(F::Factor, Q::Union{AbstractMatrix,Nothing} = nothing)
    lock(F.lock)
    try
        _require_open(F)
        if Q !== nothing
            F.values = _values_for(F, Q)
        end
        _check(_assign_values(F.group, 0, F.handle, F.values), "sTiles_assign_values")
        _bind(F.group, 0, F.handle)
        rc = _chol(F.group, 0, F.handle)
        _unbind(F.group, 0, F.handle)
        rc == 0 || throw(sTilesError(
            "sTiles_chol failed (status $rc): matrix not positive definite?"))
        F.factored = true
        F.selinv_done = false
        return F
    finally
        unlock(F.lock)
    end
end

"""
    sTiles.update!(F, Q) -> F

New values, same sparsity pattern: re-run the numeric factorization while
reusing the analysis. Alias for `factorize!(F, Q)`.
"""
update!(F::Factor, Q::AbstractMatrix) = factorize!(F, Q)

"""
    sTiles.isfactored(F) -> Bool

Whether the numeric Cholesky has run.
"""
isfactored(F::Factor) = F.factored && !F.closed

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
"""
    logdet(F)

Log-determinant of the factorized matrix, `2 * sum(log.(diag(L)))`.
"""
function LinearAlgebra.logdet(F::Factor)
    lock(F.lock)
    try
        _require_factored(F)
        return _get_logdet(F.group, 0, F.handle)
    finally
        unlock(F.lock)
    end
end

# Q is positive definite, so the determinant is positive and its sign is +1.
LinearAlgebra.logabsdet(F::Factor) = (logdet(F), one(Float64))
LinearAlgebra.det(F::Factor) = exp(logdet(F))

"""
    nnz(F)

Number of stored non-zeros in the Cholesky factor `L`.
"""
function SparseArrays.nnz(F::Factor)
    lock(F.lock)
    try
        _require_open(F)
        return Int(_get_nnz_factor(F.group, 0, F.handle))
    finally
        unlock(F.lock)
    end
end

"""
    sTiles.chol_time(F) -> Float64

Numeric Cholesky time in seconds, as measured inside the library.
"""
function chol_time(F::Factor)
    lock(F.lock)
    try
        _require_factored(F)
        return _get_chol_timing(F.group, 0, F.handle)
    finally
        unlock(F.lock)
    end
end

"""
    sTiles.selinv_time(F) -> Float64

Selected-inverse time in seconds, as measured inside the library.
"""
function selinv_time(F::Factor)
    lock(F.lock)
    try
        _require_factored(F)
        return _get_selinv_timing(F.group, 0, F.handle)
    finally
        unlock(F.lock)
    end
end

"""
    sTiles.analyze_time(F) -> Float64

Wall-clock seconds spent in the analysis phase (ordering bake-off and tile
layout), measured around the call because the library does not time it.
"""
analyze_time(F::Factor) = F.analyze_time

"""
    sTiles.summary(F) -> NamedTuple

Dimensions, fill, mode, phase state and the library-measured timings. Timings
are `nothing` until the corresponding phase has run.
"""
function summary(F::Factor)
    _require_open(F)
    return (n = F.n, nnz = F.nnz, nnz_factor = nnz(F),
            mode = get(MODE_NAMES, F.mode, F.mode), cores = F.cores,
            inverse = F.inverse, factored = F.factored,
            analyze_time = F.analyze_time,
            chol_time = F.factored ? chol_time(F) : nothing,
            selinv_time = F.selinv_done ? selinv_time(F) : nothing,
            version = version(), library = library_path())
end

"""
    sTiles.chol_elm(F, i, j) -> Float64

Entry `L[i, j]` of the Cholesky factor, in the original ordering of the input
matrix (1-based).
"""
function chol_elm(F::Factor, i::Integer, j::Integer)
    lock(F.lock)
    try
        _require_factored(F)
        _checkindex(F, i); _checkindex(F, j)
        return _get_chol_elm(F.group, 0, i - 1, j - 1, F.handle)
    finally
        unlock(F.lock)
    end
end

function _checkindex(F::Factor, i::Integer)
    1 <= i <= F.n || throw(BoundsError(F, i))
    return nothing
end

function _ensure_selinv(F::Factor)
    F.inverse || throw(sTilesError(
        "build the factor with inverse = true to use the selected inverse"))
    _require_factored(F)
    if !F.selinv_done
        _bind(F.group, 0, F.handle)
        rc = _selinv(F.group, 0, F.handle)
        _unbind(F.group, 0, F.handle)
        _check(rc, "sTiles_selinv")
        F.selinv_done = true
    end
    return nothing
end

"""
    selinv!(F) -> F

Compute the selected inverse as its own timeable phase: `Z = Q^-1` restricted to
the pattern of the Cholesky factor, in one Takahashi sweep. Requires a factor
built with `inverse = true`. Idempotent, and computed lazily by the first
`selinv_*` query if you skip it.
"""
function selinv!(F::Factor)
    lock(F.lock)
    try
        _ensure_selinv(F)
        return F
    finally
        unlock(F.lock)
    end
end

"""
    selinv_diag(F) -> Vector{Float64}

Diagonal of the selected inverse, `diag(Q^-1)`: the marginal variances, in the
original ordering.
"""
function selinv_diag(F::Factor)
    lock(F.lock)
    try
        _ensure_selinv(F)
        out = Vector{Float64}(undef, F.n)
        @inbounds for i in 1:F.n
            out[i] = _get_selinv_elm(F.group, 0, i - 1, i - 1, F.handle)
        end
        return out
    finally
        unlock(F.lock)
    end
end

"""
    selinv_elm(F, i, j) -> Float64

Selected-inverse entry `(Q^-1)[i, j]` at any position (1-based, original
ordering). Returns the value when `(i, j)` lies in the factor pattern, the
pattern of `L + L'`, and exactly `0.0` outside it, where the entry is not stored
(for a GMRF the true value there is negligible). Both triangles work, since `Z`
is symmetric. Triggers the selected-inverse computation on first use.
"""
function selinv_elm(F::Factor, i::Integer, j::Integer)
    lock(F.lock)
    try
        _ensure_selinv(F)
        _checkindex(F, i); _checkindex(F, j)
        return _get_selinv_elm(F.group, 0, i - 1, j - 1, F.handle)
    finally
        unlock(F.lock)
    end
end

"""
    selinv_row(F, node, neighbors) -> Vector{Float64}

The values `(Q^-1)[node, k]` for each `k` in `neighbors` (1-based). Entries
outside the factor pattern come back as `0`.
"""
function selinv_row(F::Factor, node::Integer, neighbors::AbstractVector{<:Integer})
    lock(F.lock)
    try
        _ensure_selinv(F)
        _checkindex(F, node)
        nb = Vector{Cint}(undef, length(neighbors))
        @inbounds for (k, v) in enumerate(neighbors)
            _checkindex(F, v)
            nb[k] = v - 1
        end
        p = _get_selinv_row(F.group, 0, node - 1, nb, length(nb), F.handle)
        p == C_NULL && throw(sTilesError("sTiles_get_selinv_row returned null"))
        # The library owns that buffer and reuses it; copy before returning.
        return copy(unsafe_wrap(Array, p, length(nb); own = false))
    finally
        unlock(F.lock)
    end
end

"""
    sTiles.permutation(F) -> Vector{Int}

The fill-reducing permutation over the original nodes, 1-based (logical, with
any nested-dissection padding removed).
"""
function permutation(F::Factor)
    lock(F.lock)
    try
        _require_open(F)
        buf = Vector{Cint}(undef, F.n)
        m = _get_logical_element_perm(F.group, 0, F.handle, buf)
        m < 0 && throw(sTilesError("sTiles_get_logical_element_perm failed"))
        return [Int(buf[k]) + 1 for k in 1:m]
    finally
        unlock(F.lock)
    end
end

# ---------------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------------
const SOLVERS = Dict(:A => _solve_LLT, :L => _solve_L, :Lt => _solve_LT)

_solver(system::Symbol) = get(SOLVERS, system) do
    throw(ArgumentError("system must be :A, :L or :Lt, got :$system"))
end

"""
    solve!(F, B; system = :A) -> B

Solve in place, overwriting `B` (a length-`n` vector or an `n x nrhs` matrix of
`Float64`). `system = :A` solves `Q x = b`, `:L` the forward solve `L y = b`,
`:Lt` the backward solve `L' x = b`. Right-hand sides are in the original
ordering, in and out.
"""
function solve!(F::Factor, B::StridedVecOrMat{Float64}; system::Symbol = :A)
    fn = _solver(system)
    lock(F.lock)
    try
        _require_factored(F)
        size(B, 1) == F.n || throw(DimensionMismatch(
            "right-hand side has $(size(B, 1)) rows, expected $(F.n)"))
        stride(B, 1) == 1 && (ndims(B) == 1 || stride(B, 2) == size(B, 1)) ||
            throw(ArgumentError(
                "the right-hand side must be contiguous and column-major; " *
                "pass a plain Vector or Matrix (or copy the view first)"))
        nrhs = size(B, 2)
        _bind(F.group, 0, F.handle)
        rc = fn(F.group, 0, F.handle, B, nrhs)
        _unbind(F.group, 0, F.handle)
        _check(rc, "sTiles solve (:$system)")
        return B
    finally
        unlock(F.lock)
    end
end

"""
    solve(F, b; system = :A) -> x

Solve without touching `b`. `system = :A` solves `Q x = b` (the default), `:L`
solves `L y = b`, `:Lt` solves `L' x = b`.
"""
solve(F::Factor, b::AbstractVecOrMat; system::Symbol = :A) =
    solve!(F, _rhs_copy(b); system = system)

_rhs_copy(b::AbstractVector) = Vector{Float64}(b)
_rhs_copy(b::AbstractMatrix) = Matrix{Float64}(b)

Base.:\(F::Factor, b::AbstractVecOrMat) = solve(F, b)

LinearAlgebra.ldiv!(F::Factor, B::StridedVecOrMat{Float64}) = solve!(F, B)
# One method per rank: a single VecOrMat signature is ambiguous against
# LinearAlgebra's own ldiv!(::AbstractMatrix, ::Factorization, ::AbstractMatrix).
LinearAlgebra.ldiv!(Y::StridedVector{Float64}, F::Factor, B::AbstractVector) =
    solve!(F, copyto!(Y, B))
LinearAlgebra.ldiv!(Y::StridedMatrix{Float64}, F::Factor, B::AbstractMatrix) =
    solve!(F, copyto!(Y, B))

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
function _teardown!(F::Factor)
    F.closed && return nothing
    F.closed = true
    try
        _free_group(F.group)
    finally
        F.handle[] = C_NULL
        LIVE[] -= 1
        if LIVE[] <= 0
            LIVE[] = 0
            # Freeing a group releases its memory but not the worker teams, and
            # on Linux not this process's claim on a slice of the machine's
            # cores either; only quit does that. It is process-wide, so it may
            # run only once the last factor is gone. Building another factor
            # afterwards re-initializes cleanly.
            try
                _quit()
            catch
            end
        end
    end
    return nothing
end

"""
    close(F)

Release the factorization: frees the solver's memory and, once the last live
factor is closed, joins its worker threads. Safe to call more than once.
"""
function Base.close(F::Factor)
    lock(LIVE_LOCK)
    try
        _teardown!(F)
    finally
        unlock(LIVE_LOCK)
    end
    return nothing
end

# The finalizer deliberately takes no lock: locking from a finalizer can
# deadlock against the collector. It is safe without one, because a factor
# being finalized is unreachable, so no other task can be calling into it, and
# only one factor is ever live.
_finalize!(F::Factor) = (try _teardown!(F) catch end; nothing)

"""
    isopen(F) -> Bool

Whether the factorization is still live (not yet closed).
"""
Base.isopen(F::Factor) = !F.closed

Base.size(F::Factor) = (F.n, F.n)
Base.size(F::Factor, d::Integer) = d <= 2 ? F.n : 1
Base.eltype(::Type{Factor}) = Float64

function Base.show(io::IO, F::Factor)
    if F.closed
        print(io, "sTiles.Factor (closed)")
        return
    end
    phase = F.factored ? "factorized" : "analyzed, not factorized"
    print(io, "sTiles.Factor: $(F.n)x$(F.n), nnz(Q)=$(F.nnz), ",
          "mode=$(get(MODE_NAMES, F.mode, F.mode)), cores=$(F.cores), $phase")
    return
end

function Base.show(io::IO, ::MIME"text/plain", F::Factor)
    show(io, F)
    F.closed && return
    if F.factored
        print(io, "\n  nnz(L)   = ", nnz(F))
        print(io, "\n  logdet   = ", logdet(F))
        print(io, "\n  time (s) = ", round(F.analyze_time; digits = 4),
              " analyze, ", round(chol_time(F); digits = 4), " chol")
        F.selinv_done && print(io, ", ", round(selinv_time(F); digits = 4), " selinv")
    end
    return
end

# ---------------------------------------------------------------------------
# Memory estimate (no handle needed: call it before building a factor)
# ---------------------------------------------------------------------------
"""
    sTiles.estimate_memory(n, nnz; tile_size=0, variant=0, inverse=false,
                           nested_dissection=true) -> Float64

Estimated memory in GB for factorizing an `n x n` matrix with `nnz` non-zeros.
`variant` is 0 for the sparse factorization, 1 for full dense, 2 for scaled
dense; `inverse` triples the estimate. This is an early estimate from the matrix
parameters and typical fill ratios, meant for a fits-in-RAM check before
building the factor.
"""
function estimate_memory(n::Integer, nnz::Integer; tile_size::Integer = 0,
                         variant::Integer = 0, inverse::Bool = false,
                         nested_dissection::Bool = true)
    return _estimate_memory(n, nnz, tile_size, variant, inverse ? 1 : 0,
                            nested_dissection ? 1 : 0)
end
