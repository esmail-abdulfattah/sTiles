PaStiX raw per-core timing sweeps
================================================================================

Two CSV files, one row per matrix (89 SPD matrices). PaStiX run on the Intel
node (2x Xeon Gold, 40 cores). Times in seconds, matrix loading EXCLUDED.

  pastix.csv            NUMERIC factorization time (pastix_task_numfact)
  symbolic_pastix.csv   SYMBOLIC analysis time     (pastix_task_analyze)

Columns
-------
matrix        matrix name
n             dimension
nnz           nonzeros, upper triangle as stored
c1 c2 c4 c8 c16 c32 c40
              the timed quantity at 1, 2, 4, 8, 16, 32, 40 threads (seconds);
              one matrix at 7 thread counts across its row.
best | min    fastest over the seven thread counts (pastix.csv "best",
              symbolic_pastix.csv "min").
best@c / min@c
              the thread count that achieved that minimum.

How to read it
--------------
- Per-matrix scaling: read across c1..c40 on the row.
- Number used in the paper: numeric -> "best" (see ../numeric.csv);
  symbolic -> "c40" (see ../symbolic.csv).
- Total analysis+factor = symbolic_pastix c40 + pastix c40.

Note: PaStiX uses its own threads (IPARM_THREAD_NBR), set to the column's
thread count.

Same layout as the other solver folders (pardiso/, mumps/, cholmod/, sympack/).
