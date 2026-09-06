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
