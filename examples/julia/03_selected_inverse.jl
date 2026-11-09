# Read entries of Q^-1 without forming the dense inverse: the diagonal,
# single entries, and a row. Checked here against the dense inverse.
#
#   julia 03_selected_inverse.jl
using sTiles, LinearAlgebra, SparseArrays, Printf

n = 200
Q = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))
Qinv = inv(Matrix(Q))                    # only to check the answers

F = sTiles.cholesky(Q; cores = 2, inverse = true)
selinv!(F)                               # optional: otherwise runs on first query

d = selinv_diag(F)                       # all n variances
@printf("diagonal:   max error = %.1e\n", maximum(abs.(d - diag(Qinv))))

# One entry. Indices are 1-based, in the original ordering of Q.
@printf("Qinv[11, 12] = %.6f   (dense: %.6f)\n", selinv_elm(F, 11, 12), Qinv[11, 12])

# Entries are exact inside the pattern of the Cholesky factor, which sTiles
# stores in whole tiles, and 0 outside it. Ask only for entries you know are
# in the pattern: the diagonal and the non-zeros of Q are.
@printf("Qinv[11, 151] = %.1f   (dense: %.1e, outside the pattern)\n",
        selinv_elm(F, 11, 151), Qinv[11, 151])

# A row restricted to chosen columns, e.g. the neighbours of node 11.
println("row 11 at [10, 11, 12] = ", round.(selinv_row(F, 11, [10, 11, 12]); digits = 6))

close(F)
