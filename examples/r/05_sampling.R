# Draw samples from N(0, solve(Q)), the Gaussian with precision matrix Q.
#
# If Q = C t(C), then x = solve(t(C), z) with z ~ N(0, I) has covariance
# solve(Q). sTiles_solve(s, z, "Lt") computes exactly that, in the original
# ordering of Q.
#
#   Rscript 05_sampling.R
library(sTiles)
library(Matrix)

n <- 1000
Q <- bandSparse(n, k = c(0, 1), diagonals = list(rep(4, n), rep(-1, n - 1)),
                symmetric = TRUE)
set.seed(1)

s <- sTiles(Q, cores = 2, inverse = TRUE)

Z <- matrix(rnorm(n * 10000), n, 10000)  # 10 000 draws, one per column
X <- sTiles_solve(s, Z, system = "Lt")   # each column ~ N(0, solve(Q))

# Check: the sample variances match the exact ones from the selected inverse.
exact <- sTiles_selinv_diag(s)
cat(sprintf("exact variance of x[1]   = %.4f\n", exact[1]))
cat(sprintf("sample variance of x[1]  = %.4f\n", var(X[1, ])))

# With a mean mu, a draw from N(mu, solve(Q)) is mu + X[, k].
sTiles_close(s)
