# Factorize a sparse SPD matrix, then get the log-determinant, solve a
# linear system and read the marginal variances.
#
#   Rscript 01_quickstart.R
library(sTiles)
library(Matrix)

# A 1000 x 1000 tridiagonal matrix: 4 on the diagonal, -1 next to it.
n <- 1000
Q <- bandSparse(n, k = c(0, 1), diagonals = list(rep(4, n), rep(-1, n - 1)),
                symmetric = TRUE)

# inverse = TRUE reserves room for the selected inverse (the variances below).
s <- sTiles(Q, cores = 2, inverse = TRUE)

cat(sprintf("log|Q|            = %.6f\n", sTiles_logdet(s)))

b <- rep(1, n)
x <- sTiles_solve(s, b)                  # x = Q^-1 b
cat(sprintf("max |Q x - b|     = %.1e\n", max(abs(Q %*% x - b))))

v <- sTiles_selinv_diag(s)               # diag(Q^-1), the marginal variances
cat("first variances   =", round(v[1:3], 6), "\n")

sTiles_close(s)                          # free the factorization now
