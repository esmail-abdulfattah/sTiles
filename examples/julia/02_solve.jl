# Solve Q x = b for one right-hand side, then for several at once.
#
#   julia 02_solve.jl
using sTiles, LinearAlgebra, SparseArrays, Printf, Random

n = 1000
Q = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))
F = sTiles.cholesky(Q; cores = 2)

# One right-hand side: a vector of length n.
b = collect(1.0:n)
x = F \ b
@printf("one rhs:   length(x) = %d, max |Q x - b| = %.1e\n",
        length(x), maximum(abs.(Q * x - b)))

# Several right-hand sides: an n x k matrix, one column each.
# All columns are solved in one call, which is faster than a loop.
B = randn(MersenneTwister(0), n, 5)
X = F \ B
@printf("five rhs:  size(X) = %s, max |Q X - B| = %.1e\n",
        size(X), maximum(abs.(Q * X - B)))

close(F)
