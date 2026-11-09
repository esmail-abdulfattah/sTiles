# Factorize a sparse SPD matrix, then get the log-determinant, solve a
# linear system and read the marginal variances.
#
#   julia 01_quickstart.jl
using sTiles, LinearAlgebra, SparseArrays, Printf

# A 1000 x 1000 tridiagonal matrix: 4 on the diagonal, -1 next to it.
n = 1000
Q = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))

# inverse = true reserves room for the selected inverse (the variances below).
F = sTiles.cholesky(Q; cores = 2, inverse = true)

@printf("log|Q|            = %.6f\n", logdet(F))

b = ones(n)
x = F \ b                                # x = Q^-1 b
@printf("max |Q x - b|     = %.1e\n", maximum(abs.(Q * x - b)))

v = selinv_diag(F)                       # diag(Q^-1), the marginal variances
println("first variances   = ", round.(v[1:3]; digits = 6))

close(F)                                 # one factor at a time: free it first
