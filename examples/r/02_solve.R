# Solve Q x = b for one right-hand side, then for several at once.
#
#   Rscript 02_solve.R
library(sTiles)
library(Matrix)

n <- 1000
Q <- bandSparse(n, k = c(0, 1), diagonals = list(rep(4, n), rep(-1, n - 1)),
                symmetric = TRUE)
s <- sTiles(Q, cores = 2)

# One right-hand side: a vector of length n.
b <- as.numeric(1:n)
x <- sTiles_solve(s, b)
cat(sprintf("one rhs:   length(x) = %d, max |Q x - b| = %.1e\n",
            length(x), max(abs(Q %*% x - b))))

# Several right-hand sides: an n x k matrix, one column each.
# All columns are solved in one call, which is faster than a loop.
set.seed(0)
B <- matrix(rnorm(n * 5), n, 5)
X <- sTiles_solve(s, B)
cat(sprintf("five rhs:  dim(X) = %d x %d, max |Q X - B| = %.1e\n",
            nrow(X), ncol(X), max(abs(Q %*% X - B))))

sTiles_close(s)
