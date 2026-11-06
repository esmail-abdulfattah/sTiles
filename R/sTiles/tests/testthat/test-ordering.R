## sTiles_ordering(): the standalone fill-reducing permutation. Skipped without
## a solver, and skipped when the loaded solver predates the entry point
## (releases up to Version_2026.09.04).

skip_no_ordering <- function() {
    skip_if_not(sTiles_available(), "no sTiles solver library installed")
    skip_if_not(sTiles:::.sTiles_has_ordering(),
                "loaded libstiles does not export sTiles_compute_ordering")
}

grid_spd <- function(g) {
    T1 <- Matrix::bandSparse(g, k = c(0, 1),
                             diagonals = list(rep(2, g), rep(-1, g - 1)),
                             symmetric = TRUE)
    I  <- Matrix::Diagonal(g)
    Matrix::kronecker(I, T1) + Matrix::kronecker(T1, I) + 0.5 * Matrix::Diagonal(g * g)
}

expect_ordering <- function(o, n) {
    expect_s3_class(o, "sTiles_ordering")
    expect_equal(o$n, n)
    expect_equal(sort(o$perm), seq_len(n))
    expect_equal(o$iperm[o$perm], seq_len(n))
    expect_true(o$winner %in% names(sTiles:::.sTiles_ordering_names))
    if (o$nblocks > 0) {
        expect_length(o$range, o$nblocks + 1)
        expect_length(o$parent, o$nblocks)
        expect_equal(o$range[1], 1L)
        expect_equal(o$range[o$nblocks + 1], n + 1L)
        expect_true(all(diff(o$range) > 0))
        expect_true(all(o$parent == 0 | o$parent > seq_len(o$nblocks)))
    } else {
        expect_length(o$range, 0)
        expect_length(o$parent, 0)
    }
}

test_that("a grid gets a valid permutation and a known winner", {
    skip_no_ordering()
    o <- sTiles_ordering(grid_spd(30))
    expect_ordering(o, 900L)
    expect_output(print(o), "sTiles ordering")
})

test_that("a band matrix gets a valid permutation", {
    skip_no_ordering()
    expect_ordering(sTiles_ordering(spd_band(500)), 500L)
})

test_that("tile_size ranks by tiles and cores are accepted", {
    skip_no_ordering()
    expect_ordering(sTiles_ordering(grid_spd(30), tile_size = 40, cores = 2), 900L)
})

test_that("tile_size = 0 either ranks by nnz(L) or says what is missing", {
    skip_no_ordering()
    o <- tryCatch(sTiles_ordering(spd_band(500), tile_size = 0),
                  error = function(e) {
                      expect_match(conditionMessage(e), "tile_size > 0")
                      NULL
                  })
    if (!is.null(o)) expect_ordering(o, 500L)
})

test_that("the reordered matrix is a symmetric permutation of the input", {
    skip_no_ordering()
    Q <- grid_spd(12)
    o <- sTiles_ordering(Q)
    P <- Q[o$iperm, o$iperm]
    expect_true(Matrix::isSymmetric(P))
    expect_equal(Matrix::nnzero(P), Matrix::nnzero(Q))
})
