# A small SPD test matrix: tridiagonal, diagonally dominant, so every solver
# path has something well conditioned to chew on.
spd_band <- function(n = 60) {
    Matrix::bandSparse(n, k = c(0, 1),
                       diagonals = list(rep(4, n), rep(-1, n - 1)),
                       symmetric = TRUE)
}
