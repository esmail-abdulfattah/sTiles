symPACK raw per-core timing sweeps
================================================================================

Two CSV files, one row per matrix (89 SPD matrices). symPACK run on the Intel
node (2x Xeon Gold, 40 cores). Times in seconds, matrix loading EXCLUDED.

  sympack.csv            NUMERIC factorization time
  symbolic_sympack.csv   SYMBOLIC analysis time

Columns
-------
matrix        matrix name
n             dimension
nnz           nonzeros, upper triangle as stored
c1 c2 c4 c8 c16 c32 c40
              the timed quantity at 1, 2, 4, 8, 16, 32, 40 threads (seconds);
              one matrix at 7 thread counts across its row.
best | min    fastest over the seven thread counts (sympack.csv "best",
              symbolic_sympack.csv "min").
best@c / min@c
              the thread count that achieved that minimum.

How to read it
--------------
- Per-matrix scaling: read across c1..c40 on the row.
- Number used in the paper: numeric -> "best" (see ../numeric.csv);
  symbolic -> "c40" (see ../symbolic.csv).
- Total analysis+factor = symbolic_sympack c40 + sympack c40.

Note: symPACK does NOT factor two of the matrices (pedigree and yU0G1u, both
sparse); their numeric entries are blank/zero, and the aggregate symPACK factor
total in the paper (912.7 s) is over the 87 matrices it solves.

Same layout as the other solver folders (pardiso/, mumps/, cholmod/, pastix/).
