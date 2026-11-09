# Factorize your own matrix, read from a Matrix Market (.mtx) file.
#
#   julia 06_matrix_market.jl                  # uses ../data/grid40.mtx
#   julia 06_matrix_market.jl my_matrix.mtx
using sTiles, LinearAlgebra, SparseArrays, Printf

# A small reader for "coordinate real" files, so the example needs no extra
# package. (MatrixMarket.jl does the same with `mmread(path)`.)
function read_mtx(path)
    lines = filter(l -> !startswith(l, "%"), readlines(path))
    n, m, _ = parse.(Int, split(lines[1]))
    I, J, V = Int[], Int[], Float64[]
    for l in lines[2:end]
        f = split(l)
        push!(I, parse(Int, f[1])); push!(J, parse(Int, f[2])); push!(V, parse(Float64, f[3]))
    end
    return sparse(I, J, V, n, m)         # sTiles reads only the lower triangle
end

path = isempty(ARGS) ? joinpath(@__DIR__, "..", "data", "grid40.mtx") : ARGS[1]
Q = read_mtx(path)
@printf("%s: %d x %d, %d stored non-zeros\n", basename(path), size(Q)..., nnz(Q))

F = sTiles.cholesky(Q; cores = 2)
@printf("log|Q| = %.6f\n", logdet(F))
@printf("non-zeros in L = %d, analysis %.3f s, cholesky %.4f s\n",
        nnz(F), sTiles.analyze_time(F), sTiles.chol_time(F))
close(F)
