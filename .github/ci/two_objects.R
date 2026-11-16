library(sTiles); library(Matrix)
sTiles_install_library()
cat("solver:", sTiles_version(), "\n")
gmrf <- function(n) bandSparse(n, k = c(0, 1), symmetric = TRUE,
                               diagonals = list(rep(4, n), rep(-1, n - 1)))
ld <- function(Q) as.numeric(determinant(as.matrix(Q))$modulus)
Q1 <- gmrf(2000); Q2 <- 2 * gmrf(500)
s  <- sTiles(Q1, cores = 2)
ss <- sTiles(Q2, cores = 2, inverse = TRUE)
stopifnot(abs(sTiles_logdet(s) - ld(Q1)) < 1e-8,
          abs(sTiles_logdet(ss) - ld(Q2)) < 1e-8)
sTiles_close(s)
v <- sTiles_selinv_diag(ss)
stopifnot(max(abs(v - diag(solve(as.matrix(Q2))))) < 1e-10)
sss <- sTiles(Q1, cores = 2)
stopifnot(abs(sTiles_logdet(sss) - ld(Q1)) < 1e-8,
          abs(sTiles_logdet(ss) - ld(Q2)) < 1e-8)
sTiles_close(ss); sTiles_close(sss)
Q3 <- gmrf(20)
s3 <- sTiles(Q3, mode = "sparse")
stopifnot(abs(sTiles_logdet(s3) - ld(Q3)) < 1e-10,
          max(abs(Q3 %*% sTiles_solve(s3, rep(1, 20)) - 1)) < 1e-10)
sTiles_close(s3)
cat("OK\n")
