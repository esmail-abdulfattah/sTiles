# Analyze the sparsity pattern once, then factorize many matrices that
# share it. This is the loop of a hyperparameter search or an optimizer.
#
#   Rscript 04_reuse_analysis.R
library(sTiles)
library(Matrix)

n <- 100000
T <- bandSparse(n, k = c(0, 1), diagonals = list(rep(2, n), rep(-1, n - 1)),
                symmetric = TRUE)
I <- Diagonal(n)

build_Q <- function(kappa) T + kappa * I   # same non-zero pattern for every kappa

# Step 1, once: ordering and tile layout. They depend only on WHERE the
# non-zeros are, not on their values.
s <- sTiles_analyze(build_Q(1), cores = 2)
cat(sprintf("analysis:  %.3f s  (paid once)\n", sTiles_summary(s)$analyze_time))

# Step 2, per value set: numeric Cholesky only.
for (kappa in c(0.1, 1, 10)) {
    sTiles_update(s, build_Q(kappa))
    cat(sprintf("kappa = %5.1f   log|Q| = %12.4f   cholesky: %.4f s\n",
                kappa, sTiles_logdet(s), sTiles_summary(s)$chol_time))
}

sTiles_close(s)
