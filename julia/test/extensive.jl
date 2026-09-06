# Extensive correctness check of sTiles.jl against dense LinearAlgebra and
# CHOLMOD, across matrix families, solver configurations, right-hand-side
# widths, pattern reuse, lifecycle and threads. Longer than runtests.jl and
# needs the INLA graphs in run/mtx for the last section, so it is run by hand:
#
#   julia -t 8 --project=. test/extensive.jl [--dump DIR] [mtx files...]
#
# --dump DIR writes a few matrices plus the Julia results to DIR so that
# test/cross_check.py can recompute them with pysTiles on the same library.
using sTiles
using LinearAlgebra, SparseArrays, Random, Printf

const FAILS = String[]
const NPASS = Ref(0)

function record(name, ok, detail = "")
    if ok
        NPASS[] += 1
        @printf("  PASS  %-58s %s\n", name, detail)
    else
        push!(FAILS, "$name  $detail")
        @printf("  FAIL  %-58s %s\n", name, detail)
    end
    return ok
end

relerr(a, b) = abs(a - b) / max(abs(b), 1e-300)
relresid(Q, x, b) = norm(Q * x - b) / max(norm(b), 1e-300)
maxrel(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1e-300)

# ---------------------------------------------------------------------------
# Matrix families
# ---------------------------------------------------------------------------
band(n, k; d = 2k + 2.0) = sparse(spdiagm([j => fill(j == 0 ? d : -1.0, n - abs(j)) for j in -k:k]...))

function grid2d(nx, ny; shift = 1e-3)
    T(m) = spdiagm(-1 => fill(-1.0, m - 1), 0 => fill(2.0, m), 1 => fill(-1.0, m - 1))
    return sparse(kron(sparse(I, ny, ny), T(nx)) + kron(T(ny), sparse(I, nx, nx)) + shift * I)
end

function grid3d(nx, ny, nz; shift = 1e-3)
    T(m) = spdiagm(-1 => fill(-1.0, m - 1), 0 => fill(2.0, m), 1 => fill(-1.0, m - 1))
    Ix, Iy, Iz = sparse(I, nx, nx), sparse(I, ny, ny), sparse(I, nz, nz)
    L = kron(Iz, kron(Iy, T(nx))) + kron(Iz, kron(T(ny), Ix)) + kron(T(nz), kron(Iy, Ix))
    return sparse(L + shift * I)
end

function randspd(n, density; rng = MersenneTwister(n + round(Int, 1000density)))
    A = sprandn(rng, n, n, density)
    return sparse(A' * A + n * I)
end

# GMRF-like: a band field plus a few dense "fixed effect" rows/columns.
function arrowhead(n, k; rng = MersenneTwister(7))
    Q = band(n - k, 1)
    B = randn(rng, n - k, k) .* 0.05
    C = Matrix(k * I + B' * B)
    full = [Matrix(Q) B; B' C]
    return sparse(full + 0.5 * I)
end

function blockdiag(nb, bs; rng = MersenneTwister(3))
    blocks = [begin A = randn(rng, bs, bs); sparse(A' * A + bs * I) end for _ in 1:nb]
    return blockdiag_(blocks)
end
blockdiag_(bs) = sparse(cat(bs...; dims = (1, 2)))

# Multiple random effects sharing a sparse design: precision of a two-level model.
function mixed_model(n1, n2, nobs; rng = MersenneTwister(11))
    Z1 = sparse(1:nobs, rand(rng, 1:n1, nobs), 1.0, nobs, n1)
    Z2 = sparse(1:nobs, rand(rng, 1:n2, nobs), 1.0, nobs, n2)
    Z = [Z1 Z2]
    return sparse(Z' * Z + blockdiag_([band(n1, 1), band(n2, 2)]))
end

# ---------------------------------------------------------------------------
# One full check of a factor against a reference (dense when small, CHOLMOD
# otherwise): logdet, solves (1 and 5 RHS, L/Lt composition, in-place), and
# the selected inverse (diag, every entry of pattern(Q), random rows, both
# triangles).
# ---------------------------------------------------------------------------
function check_factor(name, Q; cores = 1, mode = :auto, tile_size = 40,
                      inverse = true, allpattern = true, tol_ld = 1e-10,
                      tol_solve = 1e-10, tol_inv = 1e-8, rng = MersenneTwister(1))
    n = size(Q, 1)
    dense = n <= 1500
    t0 = time()
    F = sTiles.cholesky(Q; cores = cores, mode = mode, tile_size = tile_size,
                        inverse = inverse)
    try
        # reference
        if dense
            Qd = Matrix(Q)
            ld_ref = logdet(cholesky(Symmetric(Qd)))
            solve_ref = b -> Qd \ b
        else
            C = cholesky(Q)
            ld_ref = logdet(C)
            solve_ref = b -> C \ b
        end
        ok = true
        e = relerr(logdet(F), ld_ref)
        ok &= record("$name logdet", e < tol_ld, @sprintf("rel %.1e", e))

        b = randn(rng, n)
        x = F \ b
        r = relresid(Q, x, b)
        e = maxrel(x, solve_ref(b))
        ok &= record("$name solve nrhs=1", r < tol_solve && e < 1e-8,
                     @sprintf("resid %.1e vs ref %.1e", r, e))
        B = randn(rng, n, 5)
        X = solve(F, B)
        r = maximum(relresid(Q, X[:, k], B[:, k]) for k in 1:5)
        ok &= record("$name solve nrhs=5", r < tol_solve, @sprintf("resid %.1e", r))
        Y = copy(B); solve!(F, Y)
        ok &= record("$name solve! == solve", Y == X)
        y = solve(F, b; system = :L); x2 = solve(F, y; system = :Lt)
        e = maxrel(x2, x)
        ok &= record("$name L then Lt == full solve", e < 1e-12, @sprintf("rel %.1e", e))

        if inverse
            d = selinv_diag(F)
            if dense
                Qinv = inv(cholesky(Symmetric(Qd)))
                dref = diag(Qinv)
            else
                idx = unique(rand(rng, 1:n, 25))
                dref = [solve_ref(Matrix(I, n, n)[:, i])[i] for i in idx]
                d = d[idx]
            end
            e = maxrel(d, dref)
            ok &= record("$name selinv_diag" * (dense ? "" : " (25 samples)"), e < tol_inv,
                         @sprintf("rel %.1e", e))
            if dense && allpattern
                # every entry of pattern(Q) lies in pattern(L+L'), so all must match
                worst = 0.0; nz = 0
                rows, cols, _ = findnz(tril(Q))
                for (i, j) in zip(rows, cols)
                    z = selinv_elm(F, i, j)
                    z == selinv_elm(F, j, i) || (worst = Inf)
                    worst = max(worst, abs(z - Qinv[i, j]))
                    nz += z != 0
                end
                worst /= maximum(abs.(Qinv))
                ok &= record("$name selinv all pattern(Q) entries ($(length(rows)))",
                             worst < tol_inv, @sprintf("rel %.1e", worst))
                # entries reported as nonzero anywhere must match the dense inverse
                worst = 0.0; nnzZ = 0
                for j in 1:min(n, 120), i in j:min(n, 120)
                    z = selinv_elm(F, i, j)
                    if z != 0
                        nnzZ += 1
                        worst = max(worst, abs(z - Qinv[i, j]))
                    end
                end
                worst /= maximum(abs.(Qinv))
                ok &= record("$name selinv nonzeros match dense ($nnzZ in 120x120)",
                             worst < tol_inv, @sprintf("rel %.1e", worst))
            end
            if dense
                worst = 0.0
                for _ in 1:10
                    node = rand(rng, 1:n)
                    nb = unique(rand(rng, 1:n, 12))
                    r = selinv_row(F, node, nb)
                    for (k, v) in enumerate(nb)
                        r[k] == selinv_elm(F, node, v) || (worst = Inf)
                        r[k] != 0 && (worst = max(worst, abs(r[k] - Qinv[node, v])))
                    end
                end
                worst /= maximum(abs.(Qinv))
                ok &= record("$name selinv_row == selinv_elm, matches dense", worst < tol_inv,
                             @sprintf("rel %.1e", worst))
            end
        end
        s = sTiles.summary(F)
        # libstiles reports nnz(L) = 0 when the matrix fits in one tile
        # (n <= tile_size; the same from Python); the numbers above are still
        # right, so only flag it for larger n.
        nnz_ok = s.nnz_factor >= s.nnz || (n <= tile_size && s.nnz_factor == 0)
        ok &= record("$name summary", s.n == n && s.nnz == nnz(tril(Q)) && s.factored && nnz_ok,
                     "nnz(L)=$(s.nnz_factor) mode=$(s.mode) $(round(time()-t0; digits=2))s" *
                     (n <= tile_size && s.nnz_factor == 0 ? " (library quirk: nnz(L)=0 when n<=tile_size)" : ""))
        return ok
    finally
        close(F)
    end
end

# ===========================================================================
println("libstiles: ", sTiles.version(), "  at ", sTiles.library_path())
println("julia threads: ", Threads.nthreads())

dump_dir = nothing
args = copy(ARGS)
if (k = findfirst(==("--dump"), args)) !== nothing
    dump_dir = args[k + 1]; deleteat!(args, k:k+1); mkpath(dump_dir)
end
mtx_files = args

println("\n== A. matrix families vs dense reference ==")
families = [
    ("band1 n=300", band(300, 1)),
    ("band2 n=500", band(500, 2)),
    ("band5 n=400", band(400, 5)),
    ("grid2d 30x30", grid2d(30, 30)),
    ("grid3d 8x8x8", grid3d(8, 8, 8)),
    ("rand n=300 d=0.01", randspd(300, 0.01)),
    ("rand n=300 d=0.05", randspd(300, 0.05)),
    ("rand n=200 d=0.2", randspd(200, 0.2)),
    ("rand n=150 d=0.6 (dense)", randspd(150, 0.6)),
    ("fulldense n=120", sparse(let A = randn(MersenneTwister(5), 120, 120); A'A + 120I end)),
    ("arrowhead n=400 k=6", arrowhead(400, 6)),
    ("blockdiag 8x40", blockdiag(8, 40)),
    ("mixed_model 200+100", mixed_model(200, 100, 3000)),
    ("diagonal n=100", sparse(Diagonal(1.0 .+ (1:100)))),
    ("n=1", sparse(reshape([4.0], 1, 1))),
    ("n=2", sparse([2.0 -1.0; -1.0 2.0])),
    ("n=3", sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])),
]
for (name, Q) in families
    check_factor(name, Q)
end

println("\n== A2. input variants ==")
let Q = band(200, 2), ld = logdet(Matrix(Q)), rng = MersenneTwister(9)
    b = randn(rng, 200)
    xref = Matrix(Q) \ b
    variants = [
        ("lower-only storage", sparse(tril(Q))),
        ("Symmetric wrapper (upper)", Symmetric(sparse(triu(Q)), :U)),
        ("Int32 indices", SparseMatrixCSC{Float64,Int32}(Q)),
        ("Float32 values", SparseMatrixCSC{Float32,Int}(Q)),
        ("dense Matrix", Matrix(Q)),
        ("garbage upper triangle", let U = copy(Q); U[1, 3] = 99.0; U[2, 5] = -7.0; U end),
    ]
    for (name, M) in variants
        F = sTiles.cholesky(M)
        try
            e = relerr(logdet(F), ld); e2 = maxrel(F \ b, xref)
            record("variant $name", e < 1e-12 && e2 < 1e-10, @sprintf("logdet %.1e solve %.1e", e, e2))
        finally
            close(F)
        end
    end
    # explicit stored zeros stay in the pattern and are usable for reuse
    # sparse(I, J, V) keeps explicit zeros, so (10, 1) is a stored zero here
    rows, cols, vals = findnz(Q)
    Z = sparse([rows; 10; 1], [cols; 1; 10], [vals; 0.0; 0.0], 200, 200)
    F = sTiles.analyze(Z)
    try
        sTiles.factorize!(F)
        Z2 = copy(Z); Z2[10, 1] = 0.3; Z2[1, 10] = 0.3
        sTiles.factorize!(F, Z2)
        e = relerr(logdet(F), logdet(Matrix(Z2)))
        record("stored zero in pattern, later nonzero", nnz(tril(Z)) == sTiles.summary(F).nnz && e < 1e-12,
               @sprintf("nnz %d, logdet rel %.1e", sTiles.summary(F).nnz, e))
    finally
        close(F)
    end
end

println("\n== A3. ill-conditioned: sTiles vs CHOLMOD error, both vs exact ==")
let n = 400, rng = MersenneTwister(4)
    # exact solution known: Q x* = b with x* = ones
    for shift in (1e-2, 1e-6, 1e-10)
        Q = grid2d(20, 20; shift = shift)
        b = Q * ones(n)
        F = sTiles.cholesky(Q; inverse = true)
        try
            es = norm(F \ b .- 1) / sqrt(n)
            ec = norm(cholesky(Q) \ b .- 1) / sqrt(n)
            ld = relerr(logdet(F), logdet(cholesky(Q)))
            record(@sprintf("shift=%.0e cond~%.0e", shift, 8 / shift), es < 100 * max(ec, 1e-16) && ld < 1e-8,
                   @sprintf("err sTiles %.1e CHOLMOD %.1e logdet rel %.1e", es, ec, ld))
        finally
            close(F)
        end
    end
end

println("\n== B. configuration sweep: mode x cores x tile_size ==")
let mats = [("band2 n=600", band(600, 2)), ("rand n=300 d=0.05", randspd(300, 0.05)),
            ("arrowhead n=500 k=8", arrowhead(500, 8))]
    for (name, Q) in mats
        Qd = Matrix(Q); ld = logdet(cholesky(Symmetric(Qd))); dref = diag(inv(cholesky(Symmetric(Qd))))
        b = randn(MersenneTwister(2), size(Q, 1)); xref = Qd \ b
        worst_ld = 0.0; worst_x = 0.0; worst_d = 0.0; nconf = 0; bad = String[]
        for mode in (:auto, :dense, :semisparse, :sparse), cores in (1, 2, 4, 8), ts in (-1, 16, 40, 120)
            F = try
                sTiles.cholesky(Q; cores = cores, mode = mode, tile_size = ts, inverse = true)
            catch err
                push!(bad, "$mode/$cores/$ts ctor threw $(sprint(showerror, err))"); continue
            end
            try
                e1 = relerr(logdet(F), ld); e2 = maxrel(F \ b, xref); e3 = maxrel(selinv_diag(F), dref)
                worst_ld = max(worst_ld, e1); worst_x = max(worst_x, e2); worst_d = max(worst_d, e3)
                (e1 < 1e-10 && e2 < 1e-8 && e3 < 1e-8) || push!(bad, "$mode/$cores/$ts")
                nconf += 1
            catch err
                push!(bad, "$mode/$cores/$ts threw $(sprint(showerror, err))")
            finally
                close(F)
            end
        end
        record("$name: $nconf configs agree", isempty(bad),
               @sprintf("worst logdet %.1e solve %.1e selinv %.1e %s", worst_ld, worst_x, worst_d, join(bad, " ")))
    end
end

println("\n== C. right-hand-side widths ==")
let mats = [("band3 n=800", band(800, 3)), ("rand n=400 d=0.03", randspd(400, 0.03)),
            ("arrowhead n=600 k=10", arrowhead(600, 10))]
    for (name, Q) in mats, cores in (1, 4)
        n = size(Q, 1); Qd = Matrix(Q)
        F = sTiles.cholesky(Q; cores = cores)
        try
            worst = 0.0; bad = Int[]
            for nrhs in (1, 2, 3, 7, 16, 63, 64, 65, 128, 200, 257)
                B = randn(MersenneTwister(nrhs), n, nrhs)
                X = solve(F, B)
                r = maximum(relresid(Q, X[:, k], B[:, k]) for k in 1:nrhs)
                Y = solve(F, solve(F, B; system = :L); system = :Lt)
                r2 = maxrel(Y, X)
                worst = max(worst, r, r2)
                (r < 1e-10 && r2 < 1e-12) || push!(bad, nrhs)
            end
            record("$name cores=$cores nrhs sweep", isempty(bad), @sprintf("worst %.1e %s", worst, isempty(bad) ? "" : "bad: $bad"))
        finally
            close(F)
        end
    end
end

println("\n== D. pattern reuse ==")
let Q0 = mixed_model(300, 150, 5000), n = size(Q0, 1), rng = MersenneTwister(21)
    for cores in (1, 4)
        F = sTiles.analyze(Q0; cores = cores, inverse = true)
        try
            worst = 0.0; worst_d = 0.0
            for k in 1:20
                # same pattern, new values: random positive diagonal + scaled off-diagonal
                Qk = sparse(Q0 * (0.5 + rand(rng)) + Diagonal(rand(rng, n) .* 3))
                sTiles.factorize!(F, Qk)
                worst = max(worst, relerr(logdet(F), logdet(cholesky(Qk))))
                if k % 5 == 0
                    Qinv_d = diag(inv(cholesky(Symmetric(Matrix(Qk)))))
                    worst_d = max(worst_d, maxrel(selinv_diag(F), Qinv_d))
                end
            end
            record("20 value sets on one pattern, cores=$cores", worst < 1e-10 && worst_d < 1e-8,
                   @sprintf("worst logdet %.1e selinv %.1e", worst, worst_d))
            # a non-PD set in the middle must error, and the next valid set must be right
            Qbad = copy(Q0); Qbad[1, 1] = -5.0
            threw = try sTiles.factorize!(F, Qbad); false catch err; err isa sTilesError end
            sTiles.factorize!(F, Q0)
            e = relerr(logdet(F), logdet(cholesky(Q0)))
            e2 = relresid(Q0, F \ ones(n), ones(n))
            record("non-PD values error then recover, cores=$cores", threw && e < 1e-10 && e2 < 1e-10,
                   @sprintf("threw=%s logdet rel %.1e resid %.1e", threw, e, e2))
        finally
            close(F)
        end
    end
end

println("\n== E. lifecycle ==")
let rng = MersenneTwister(33), bad = 0
    for k in 1:60
        n = rand(rng, 5:250)
        Q = randspd(n, rand(rng, [0.02, 0.1, 0.5]); rng = rng)
        mode = rand(rng, [:auto, :dense, :semisparse, :sparse])
        F = sTiles.cholesky(Q; cores = rand(rng, 1:4), mode = mode, inverse = isodd(k))
        e = relerr(logdet(F), logdet(Matrix(Q)))
        isodd(k) && (e = max(e, maxrel(selinv_diag(F), diag(inv(Matrix(Q))))))
        close(F)
        e < 1e-8 || (bad += 1)
    end
    record("60 create/factor/close cycles, random n/mode/cores", bad == 0, "bad=$bad")

    # dropped without close: the finalizer must release it
    let
        F = sTiles.cholesky(band(50, 1)); F = nothing
        for _ in 1:20
            sTiles.LIVE[] == 0 && break
            GC.gc(true)
        end
        record("finalizer releases a dropped factor", sTiles.LIVE[] == 0, "LIVE=$(sTiles.LIVE[])")
    end
    F = sTiles.cholesky(band(50, 1))
    record("new factor after finalizer", relerr(logdet(F), logdet(Matrix(band(50, 1)))) < 1e-12)
    close(F)
    # a failed constructor leaves nothing live
    for M in (spdiagm(0 => [1.0, -1.0]), sprand(4, 5, 0.5))
        try sTiles.cholesky(M) catch end
    end
    record("failed constructors leave nothing live", sTiles.LIVE[] == 0)
end

println("\n== F. concurrent solves from Julia threads ==")
let Q = band(2000, 3), n = 2000, Qd = nothing
    F = sTiles.cholesky(Q; cores = 2, inverse = true)
    try
        if Threads.nthreads() == 1
            record("threads", true, "skipped: start julia with -t 8")
        else
            Bs = [randn(MersenneTwister(t), n, 3) for t in 1:32]
            ref = [solve(F, B) for B in Bs]
            out = Vector{Any}(undef, 32)
            Threads.@threads for t in 1:32
                out[t] = solve(F, Bs[t])
            end
            e = maximum(maxrel(out[t], ref[t]) for t in 1:32)
            record("32 solves from $(Threads.nthreads()) threads == serial", e == 0, @sprintf("max diff %.1e", e))
            # mixed operations concurrently: logdet / selinv / solve
            ld = logdet(F); d = selinv_diag(F)
            okv = fill(true, 32)
            Threads.@threads for t in 1:32
                okv[t] = (logdet(F) == ld) && (selinv_diag(F) == d) && (solve(F, Bs[t]) == ref[t])
            end
            record("mixed concurrent logdet/selinv/solve", all(okv))
        end
    finally
        close(F)
    end
end

# ---------------------------------------------------------------------------
# Dump for the cross-binding check (test/cross_check.py)
# ---------------------------------------------------------------------------
function dump_case(dir, name, Q, cores)
    n = size(Q, 1)
    open(joinpath(dir, "$name.mtx"), "w") do io
        println(io, "%%MatrixMarket matrix coordinate real symmetric")
        L = tril(Q)
        rows, cols, vals = findnz(L)
        println(io, n, " ", n, " ", length(vals))
        for (i, j, v) in zip(rows, cols, vals)
            @printf(io, "%d %d %.17g\n", i, j, v)
        end
    end
    F = sTiles.cholesky(Q; cores = cores, inverse = true)
    try
        b = collect(1.0:n)
        open(joinpath(dir, "$name.julia.txt"), "w") do io
            @printf(io, "cores %d\nlogdet %.17g\nnnz_factor %d\n", cores, logdet(F), nnz(F))
            for v in selinv_diag(F); @printf(io, "d %.17g\n", v); end
            for v in F \ b;         @printf(io, "x %.17g\n", v); end
        end
    finally
        close(F)
    end
end

# ---------------------------------------------------------------------------
# Real INLA precision matrices vs CHOLMOD
# ---------------------------------------------------------------------------
function read_mm(path)
    rows = Int[]; cols = Int[]; vals = Float64[]
    n = 0; m = 0; sym = false
    open(path) do io
        first = true; header = true
        for line in eachline(io)
            if first
                sym = occursin("symmetric", line); first = false; continue
            end
            (isempty(line) || startswith(line, "%")) && continue
            f = split(line)
            if header
                n = parse(Int, f[1]); m = parse(Int, f[2]); header = false; continue
            end
            push!(rows, parse(Int, f[1])); push!(cols, parse(Int, f[2]))
            push!(vals, length(f) >= 3 ? parse(Float64, f[3]) : 1.0)
        end
    end
    A = sparse(rows, cols, vals, n, m)
    if sym
        return sparse(A + A' - Diagonal(A))
    end
    return sparse((A + A') / 2)
end

if !isempty(mtx_files)
    println("\n== G. INLA precision matrices vs CHOLMOD ==")
    for path in mtx_files
        name = replace(basename(path), "inla_graph_" => "", ".mtx" => "")
        Q = read_mm(path)
        n = size(Q, 1)
        t0 = time()
        C = try cholesky(Q) catch err; println("  SKIP  $name: CHOLMOD failed ($(sprint(showerror, err)))"); continue end
        tC = time() - t0
        rng = MersenneTwister(1)
        F = sTiles.cholesky(Q; cores = 8, inverse = true)
        try
            ld = relerr(logdet(F), logdet(C))
            record("$name n=$n logdet vs CHOLMOD", ld < 1e-10, @sprintf("rel %.1e  (chol %.3fs sTiles, %.3fs CHOLMOD)", ld, sTiles.chol_time(F), tC))
            b = randn(rng, n)
            x = F \ b
            record("$name solve residual", relresid(Q, x, b) < 1e-10, @sprintf("resid %.1e vs CHOLMOD x: rel %.1e", relresid(Q, x, b), maxrel(x, C \ b)))
            B = randn(rng, n, 16); X = solve(F, B)
            r = maximum(relresid(Q, X[:, k], B[:, k]) for k in 1:16)
            record("$name solve nrhs=16", r < 1e-10, @sprintf("resid %.1e", r))
            y = solve(F, b; system = :L); x2 = solve(F, y; system = :Lt)
            record("$name L then Lt", maxrel(x2, x) < 1e-12, @sprintf("rel %.1e", maxrel(x2, x)))
            t1 = time(); d = selinv_diag(F); tsel = time() - t1
            idx = unique(rand(rng, 1:n, 30))
            E = zeros(n, length(idx)); for (k, i) in enumerate(idx); E[i, k] = 1.0; end
            Xi = C \ E
            dref = [Xi[i, k] for (k, i) in enumerate(idx)]
            e = maxrel(d[idx], dref)
            record("$name selinv_diag vs CHOLMOD columns (30)", e < 1e-8, @sprintf("rel %.1e (selinv %.3fs, diag read %.3fs)", e, sTiles.selinv_time(F), tsel))
            # off-diagonal in-pattern entries: pattern(Q) entries vs the CHOLMOD columns
            worst = 0.0; cnt = 0
            for (k, i) in enumerate(idx)
                nb = findall(!iszero, Q[:, i])
                r = selinv_row(F, i, nb)
                worst = max(worst, maximum(abs.(r .- Xi[nb, k])) / maximum(abs.(Xi[:, k])))
                cnt += length(nb)
            end
            record("$name selinv_row over pattern(Q) ($cnt entries)", worst < 1e-8, @sprintf("rel %.1e", worst))
            s = sTiles.summary(F)
            record("$name nnz(L) vs CHOLMOD", true, "sTiles $(s.nnz_factor) CHOLMOD $(nnz(C)) mode=$(s.mode)")
        finally
            close(F)
        end
        # after close: only one factor may be live, and dump_case builds its own
        if dump_dir !== nothing && n <= 100_000
            dump_case(dump_dir, name, Q, 8)
        end
    end
end

if dump_dir !== nothing
    println("\n== H. dump for cross-binding check -> $dump_dir ==")
    for (name, Q, c) in [("band2_600", band(600, 2), 1), ("rand300", randspd(300, 0.05), 1),
                         ("arrow500", arrowhead(500, 8), 1), ("mixed", mixed_model(300, 150, 5000), 1),
                         ("grid2d_40", grid2d(40, 40), 4), ("rand_dense150", randspd(150, 0.6), 4)]
        dump_case(dump_dir, name, Q, c)
        println("  wrote $name (cores=$c)")
    end
end

println("\n==================================================================")
@printf("%d checks passed, %d failed\n", NPASS[], length(FAILS))
for f in FAILS; println("  FAIL  ", f); end
exit(isempty(FAILS) ? 0 : 1)
