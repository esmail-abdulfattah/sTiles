sTiles raw per-core timing sweeps
================================================================================

sTiles is run in several tile MODES, so it has more files than the other
solvers. One row per matrix (89 SPD matrices), Intel node (2x Xeon Gold, 40
cores), times in seconds, matrix loading EXCLUDED.

NUMERIC factorization time (one file per mode)
  stiles_auto.csv         AUTO: the per-matrix selector picks the mode (this is
                          the headline sTiles result used in the paper)
  stiles_dense.csv        forced dense tiling
  stiles_semisparse.csv   forced semisparse tiling (the new primitive)
  stiles_sparse.csv       forced non-uniform / supernodal sparse tiling

SYMBOLIC analysis time
  symbolic_stiles_auto.csv         AUTO mode analysis
  symbolic_stiles_semisparse.csv   semisparse mode analysis
  symbolic_stiles_sparse.csv       sparse mode analysis

Columns (every file)
--------------------
matrix        matrix name
n             dimension
nnz           nonzeros, upper triangle as stored
c1 c2 c4 c8 c16 c32 c40
              the timed quantity at 1, 2, 4, 8, 16, 32, 40 threads (seconds);
              one matrix at 7 thread counts across its row.
best | min    fastest over the seven thread counts (numeric files call it
              "best", symbolic files call it "min").
best@c / min@c
              the thread count that achieved that minimum.

How to read it
--------------
- The headline sTiles numbers in the paper come from AUTO mode:
    numeric  -> stiles_auto.csv "best"  (also copied into ../numeric.csv)
    symbolic -> symbolic sweep at c40   (in ../symbolic.csv; note the symbolic
                AUTO time in the paper is the parallel-symbolic default measured
                at 40 cores, which supersedes symbolic_stiles_auto.csv c40)
- The per-mode files (dense/semisparse/sparse) are used to check the router:
  "did AUTO pick the fastest of the three fixed modes?" (paper Section 7.3,
  Figures threshold_sensitivity / feature_scatter).
- Total analysis+factor (AUTO) = (symbolic AUTO c40) + (stiles_auto c40).

Same column layout as the competitor folders (pardiso/, mumps/, cholmod/,
pastix/, sympack/).
