# Draw samples from N(0, Q^-1), the Gaussian with precision matrix Q.
#
# If Q = C C', then x = C' \ z with z ~ N(0, I) has covariance Q^-1.
# solve(F, z; system = :Lt) computes exactly that, in the original ordering
# of Q.
#
#   julia 05_sampling.jl
using sTiles, LinearAlgebra, SparseArrays, Printf, Random, Statistics

n = 1000
Q = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))

F = sTiles.cholesky(Q; cores = 2, inverse = true)

Z = randn(MersenneTwister(1), n, 10_000) # 10 000 draws, one per column
X = solve(F, Z; system = :Lt)            # each column ~ N(0, Q^-1)

# Check: the sample variances match the exact ones from the selected inverse.
exact = selinv_diag(F)
@printf("exact variance of x[1]   = %.4f\n", exact[1])
@printf("sample variance of x[1]  = %.4f\n", var(X[1, :]))

# With a mean mu, a draw from N(mu, Q^-1) is mu + X[:, k].
close(F)
