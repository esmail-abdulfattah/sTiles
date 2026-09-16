# Factorize a sparse SPD matrix, then read the log-determinant, solve a system
# and get the marginal variances.
#
#   julia --project=.. quickstart.jl
using sTiles, LinearAlgebra, SparseArrays

n = 1000
Q = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))
b = ones(n)

F = sTiles.cholesky(Q; cores = 2, inverse = true)
println(F)

println("logdet(Q)         = ", logdet(F))
x = F \ b
println("||Q x - b|| / ||b|| = ", norm(Q * x - b) / norm(b))

v = selinv_diag(F)                    # diag(Q^-1)
println("marginal variances: ", v[1:5], " ...")
println("(Q^-1)[3, 4]        = ", selinv_elm(F, 3, 4))

println(sTiles.summary(F))
close(F)
