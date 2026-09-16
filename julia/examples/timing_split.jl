# Separate the analysis (once per sparsity pattern) from the numeric Cholesky
# (once per set of values): the loop an iterative fit wants.
#
#   julia --project=.. timing_split.jl [path/to/matrix.mtx]
using sTiles, LinearAlgebra, SparseArrays

function load_or_make(args)
    if !isempty(args)
        # Minimal Matrix Market reader for a symmetric coordinate file.
        rows = Int[]; cols = Int[]; vals = Float64[]
        n = 0
        open(args[1]) do io
            header = true
            for line in eachline(io)
                startswith(line, "%") && continue
                f = split(line)
                if header
                    n = parse(Int, f[1]); header = false; continue
                end
                push!(rows, parse(Int, f[1])); push!(cols, parse(Int, f[2]))
                push!(vals, parse(Float64, f[3]))
            end
        end
        Q = sparse(rows, cols, vals, n, n)
        return sparse(tril(Q) + tril(Q, -1)')
    end
    n = 20000
    return spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))
end

Q = load_or_make(ARGS)
println("n = ", size(Q, 1), ", nnz = ", nnz(Q))

F = sTiles.analyze(Q; cores = 4, inverse = true)     # phase 1
println("analysis (ordering + layout): ", round(sTiles.analyze_time(F); digits = 4), " s")

sTiles.factorize!(F)                                 # phase 2
println("numeric Cholesky:             ", round(sTiles.chol_time(F); digits = 4), " s",
        "   nnz(L) = ", nnz(F), "   logdet = ", logdet(F))

# A sweep over the same pattern pays only the numeric cost.
for scale in (1.5, 2.0, 2.5)
    sTiles.update!(F, scale * Q)
    println("  scale ", scale, ": chol ", round(sTiles.chol_time(F); digits = 4),
            " s, logdet ", logdet(F))
end

selinv!(F)                                           # phase 3
println("selected inverse:             ", round(sTiles.selinv_time(F); digits = 4), " s")
close(F)
