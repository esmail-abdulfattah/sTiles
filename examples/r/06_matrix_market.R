# Factorize your own matrix, read from a Matrix Market (.mtx) file.
#
#   Rscript 06_matrix_market.R                  # uses ../data/grid40.mtx
#   Rscript 06_matrix_market.R my_matrix.mtx
library(sTiles)
library(Matrix)

args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) > 0) args[1] else "../data/grid40.mtx"

# readMM keeps a "symmetric" file symmetric; sTiles reads only the lower
# triangle, so a full matrix works as well.
Q <- readMM(path)
cat(sprintf("%s: %d x %d, %d stored non-zeros\n",
            basename(path), nrow(Q), ncol(Q), length(Q@x)))

s <- sTiles(Q, cores = 2)
cat(sprintf("log|Q| = %.6f\n", sTiles_logdet(s)))
info <- sTiles_summary(s)                # print(info) shows the full report
cat(sprintf("non-zeros in L = %.0f, analysis %.3f s, cholesky %.4f s\n",
            info$nnz_factor, info$analyze_time, info$chol_time))
sTiles_close(s)
