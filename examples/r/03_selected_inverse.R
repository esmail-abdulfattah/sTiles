# Read entries of solve(Q) without forming the dense inverse: the diagonal,
# single entries, a row, and the whole inverse on the pattern of Q.
# Checked here against the dense inverse.
#
#   Rscript 03_selected_inverse.R
library(sTiles)
library(Matrix)

n <- 200
Q <- bandSparse(n, k = c(0, 1), diagonals = list(rep(4, n), rep(-1, n - 1)),
                symmetric = TRUE)
Qinv <- solve(as.matrix(Q))              # only to check the answers

s <- sTiles(Q, cores = 2, inverse = TRUE)
sTiles_selinv(s)                         # optional: otherwise runs on first query

d <- sTiles_selinv_diag(s)               # all n variances
cat(sprintf("diagonal:   max error = %.1e\n", max(abs(d - diag(Qinv)))))

# One entry. Indices are 1-based, in the original ordering of Q.
cat(sprintf("Qinv[11, 12] = %.6f   (dense: %.6f)\n",
            sTiles_selinv_elm(s, 11, 12), Qinv[11, 12]))

# Entries are exact inside the pattern of the Cholesky factor, which sTiles
# stores in whole tiles, and 0 outside it. Ask only for entries you know are
# in the pattern: the diagonal and the non-zeros of Q are.
cat(sprintf("Qinv[11, 151] = %.1f   (dense: %.1e, outside the pattern)\n",
            sTiles_selinv_elm(s, 11, 151), Qinv[11, 151]))

# A row restricted to chosen columns, e.g. the neighbours of node 11.
cat("row 11 at c(10, 11, 12) =", round(sTiles_selinv_row(s, 11, c(10, 11, 12)), 6), "\n")

# The inverse at every non-zero of Q, as a sparse matrix.
Z <- sTiles_selinv_matrix(s)
print(Z[1:3, 1:3])

sTiles_close(s)
