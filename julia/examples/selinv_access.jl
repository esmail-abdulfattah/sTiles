# Selected-inverse element access, checked against a dense inverse.
#
#   julia --project=.. selinv_access.jl
using sTiles, LinearAlgebra, SparseArrays, Random

n = 60
A = sprandn(MersenneTwister(1), n, n, 0.05)
Q = sparse(A' * A + n * I)
Qinv = inv(Matrix(Q))

F = sTiles.cholesky(Q; inverse = true)
selinv!(F)

# Diagonal: always in the pattern.
v = selinv_diag(F)
println("max |diag error| = ", maximum(abs.(v .- diag(Qinv))))

# Off-diagonal: exact when (i, j) is in the factor pattern, 0 outside it.
i, j = 5, 2
z = selinv_elm(F, i, j)
println("(Q^-1)[$i,$j]: sTiles = $z, dense = $(Qinv[i, j])",
        z == 0 ? "  (outside the factor pattern)" : "")

# A row over a neighbourhood, e.g. the non-zero pattern of column i of Q.
nbrs = findall(!iszero, Q[:, i])
r = selinv_row(F, i, nbrs)
println("row $i over ", length(nbrs), " neighbours: max error ",
        maximum(abs.(r .- Qinv[i, nbrs])))
close(F)
