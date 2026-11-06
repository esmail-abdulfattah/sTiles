## The numerics. Skipped wherever no solver library is installed, which is the
## case on a check machine that has not run sTiles_install_library().

skip_no_solver <- function()
    skip_if_not(sTiles_available(), "no sTiles solver library installed")

test_that("log-determinant matches Matrix", {
    skip_no_solver()
    Q <- spd_band(60)
    s <- sTiles(Q)
    on.exit(sTiles_close(s))
    ref <- as.numeric(Matrix::determinant(Q, logarithm = TRUE)$modulus)
    expect_equal(sTiles_logdet(s), ref, tolerance = 1e-8)
})

test_that("solve matches Matrix, for a vector and for several right-hand sides", {
    skip_no_solver()
    Q <- spd_band(60)
    s <- sTiles(Q)
    on.exit(sTiles_close(s))

    b <- as.double(seq_len(60))
    expect_equal(as.vector(sTiles_solve(s, b)),
                 as.vector(Matrix::solve(Q, b)), tolerance = 1e-8)

    B <- cbind(b, rev(b))
    expect_equal(as.matrix(sTiles_solve(s, B)),
                 as.matrix(Matrix::solve(Q, B)),
                 tolerance = 1e-8, ignore_attr = TRUE)
})

test_that("solve keeps the shape of b and refuses a transposed matrix", {
    skip_no_solver()
    Q <- spd_band(60)
    s <- sTiles(Q)
    on.exit(sTiles_close(s))
    b <- as.double(seq_len(60))

    expect_null(dim(sTiles_solve(s, b)))
    expect_identical(dim(sTiles_solve(s, matrix(b))), c(60L, 1L))
    expect_identical(dim(sTiles_solve(s, cbind(b, b, b))), c(60L, 3L))
    expect_equal(as.matrix(sTiles_solve(s, as.data.frame(cbind(b, b)))),
                 as.matrix(Matrix::solve(Q, cbind(b, b))),
                 tolerance = 1e-8, ignore_attr = TRUE)
    expect_error(sTiles_solve(s, t(cbind(b, b))), "60 rows")
})

test_that("half solves work in the original ordering, in every mode", {
    skip_no_solver()
    set.seed(2)
    n <- 150
    B <- Matrix::rsparsematrix(n, n, density = 0.02)
    Q <- Matrix::forceSymmetric(Matrix::crossprod(B) + Matrix::Diagonal(n) * 3)
    full <- as.matrix(Matrix::solve(Q))
    b <- rnorm(n)

    for (mode in c("auto", "dense", "semisparse", "sparse")) {
        s <- sTiles(Q, mode = mode, cores = 2L)
        y <- sTiles_solve(s, b, "L")
        expect_equal(sTiles_solve(s, y, "Lt"), sTiles_solve(s, b), tolerance = 1e-8)
        expect_equal(sum(y^2), sum(b * (full %*% b)), tolerance = 1e-8)
        # Columns of this are draws: their covariance is exactly the inverse.
        S <- sTiles_solve(s, diag(n), "Lt")
        sTiles_close(s)
        expect_equal(S %*% t(S), full, tolerance = 1e-8, ignore_attr = TRUE)
    }
})

test_that("selected inverse matches the true inverse on the factor pattern", {
    skip_no_solver()
    Q <- spd_band(60)
    s <- sTiles(Q, inverse = TRUE)
    on.exit(sTiles_close(s))
    sTiles_selinv(s)

    full <- as.matrix(Matrix::solve(Q))
    expect_equal(sTiles_selinv_diag(s), diag(full), tolerance = 1e-8)
    # Tridiagonal Q: (i, i+1) is in pattern(L + t(L)), so it must be exact.
    expect_equal(sTiles_selinv_elm(s, 5, 6), full[5, 6], tolerance = 1e-8)
    expect_equal(sTiles_selinv_elm(s, 6, 5), full[5, 6], tolerance = 1e-8)
    expect_equal(sTiles_selinv_row(s, 5, c(4, 5, 6)), full[5, c(4, 5, 6)],
                 tolerance = 1e-8)
})

test_that("selinv matrix matches the true inverse on the pattern of Q", {
    skip_no_solver()
    set.seed(1)
    n <- 120
    B <- Matrix::rsparsematrix(n, n, density = 0.02)
    Q <- Matrix::forceSymmetric(Matrix::crossprod(B) + Matrix::Diagonal(n) * 5)
    full <- as.matrix(Matrix::solve(Q))
    on_q <- as.matrix(Q) != 0

    for (mode in c("auto", "dense", "semisparse", "sparse")) {
        s <- sTiles(Q, inverse = TRUE, mode = mode, cores = 2L)
        Z <- sTiles_selinv_matrix(s)
        sTiles_close(s)
        expect_s4_class(Z, "dsCMatrix")
        expect_equal(as.matrix(Z) != 0, on_q, ignore_attr = TRUE)
        expect_equal(as.matrix(Z)[on_q], full[on_q], tolerance = 1e-8)
    }
})

test_that("selinv matrix follows new values", {
    skip_no_solver()
    Q <- spd_band(60)
    s <- sTiles_analyze(Q, inverse = TRUE)
    on.exit(sTiles_close(s))
    sTiles_factorize(s)
    Z1 <- sTiles_selinv_matrix(s)
    sTiles_update(s, 2 * Q)
    expect_equal(as.matrix(sTiles_selinv_matrix(s)), as.matrix(Z1) / 2,
                 tolerance = 1e-8, ignore_attr = TRUE)
})

test_that("selinv matrix refuses a handle built without inverse", {
    skip_no_solver()
    s <- sTiles(spd_band(60))
    on.exit(sTiles_close(s))
    expect_error(sTiles_selinv_matrix(s), "inverse=TRUE")
})

test_that("preprocessing is reused across value updates", {
    skip_no_solver()
    Q <- spd_band(60)
    s <- sTiles_analyze(Q)
    on.exit(sTiles_close(s))

    sTiles_factorize(s)
    d1 <- sTiles_logdet(s)
    sTiles_update(s, 2 * Q)
    d2 <- sTiles_logdet(s)

    # det(2 Q) = 2^n det(Q) for an n x n matrix.
    expect_equal(d2 - d1, 60 * log(2), tolerance = 1e-8)
})

test_that("a mismatched pattern is refused rather than silently wrong", {
    skip_no_solver()
    s <- sTiles_analyze(spd_band(60))
    on.exit(sTiles_close(s))
    expect_error(sTiles_factorize(s, spd_band(40)), "sparsity")
})

test_that("summary reports the phases", {
    skip_no_solver()
    Q <- spd_band(60)
    s <- sTiles_analyze(Q)
    on.exit(sTiles_close(s))

    before <- sTiles_summary(s)
    expect_s3_class(before, "sTiles_summary")
    expect_false(before$factored)
    expect_identical(before$n, 60L)

    sTiles_factorize(s)
    after <- sTiles_summary(s)
    expect_true(after$factored)
    expect_gte(after$nnz_factor, after$nnz)
})
