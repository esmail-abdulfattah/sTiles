CHOLMOD raw per-core timing sweeps
================================================================================

Two CSV files, one row per matrix (89 SPD matrices). CHOLMOD (supernodal LL^T)
run on the Intel node (2x Xeon Gold, 40 cores). Times in seconds, matrix
loading EXCLUDED.

  cholmod.csv            NUMERIC factorization time (cholmod_factorize)
  symbolic_cholmod.csv   SYMBOLIC analysis time     (cholmod_analyze)

Columns
-------
matrix        matrix name
n             dimension
nnz           nonzeros, upper triangle as stored
c1 c2 c4 c8 c16 c32 c40
              the timed quantity at 1, 2, 4, 8, 16, 32, 40 threads (seconds);
              one matrix at 7 thread counts across its row.
best | min    fastest over the seven thread counts (cholmod.csv "best",
              symbolic_cholmod.csv "min").
best@c / min@c
              the thread count that achieved that minimum.

How to read it
--------------
- Per-matrix scaling: read across c1..c40 on the row.
- Number used in the paper: numeric -> "best" (see ../numeric.csv);
  symbolic -> "c40" (see ../symbolic.csv).
- Total analysis+factor = symbolic_cholmod c40 + cholmod c40.

Note: CHOLMOD has no internal threads; all parallelism comes from the
multithreaded BLAS (scale OMP_NUM_THREADS / MKL_NUM_THREADS in the launcher),
so the thread count in the column names is that BLAS budget.

Same layout as the other solver folders (pardiso/, mumps/, pastix/, sympack/).
