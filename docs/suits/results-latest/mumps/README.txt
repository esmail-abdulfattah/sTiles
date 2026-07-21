MUMPS raw per-core timing sweeps
================================================================================

Two CSV files, one row per matrix (89 SPD matrices). Both are MUMPS run on the
Intel node (2x Xeon Gold, 40 cores) at several thread counts. Times are in
seconds and EXCLUDE matrix loading (timed separately).

  mumps.csv            NUMERIC factorization time (JOB=2)
  symbolic_mumps.csv   SYMBOLIC analysis time     (JOB=1, ordering + symbolic)

These are the per-core source data; the paper's wide comparison files
(../numeric.csv, ../symbolic.csv) pull MUMPS's column from here.

Columns
-------
matrix        matrix name
n             dimension
nnz           nonzeros, upper triangle as stored
c1 c2 c4 c8 c16 c32 c40
              the timed quantity at 1, 2, 4, 8, 16, 32, 40 threads (seconds).
              So each row is the same matrix solved at 7 thread counts.
best | min    the fastest of the seven (mumps.csv calls it "best",
              symbolic_mumps.csv calls it "min"); same meaning = the minimum
              over the thread counts.
best@c / min@c
              the thread count that achieved that minimum (1, 2, ... or 40).

How to read it
--------------
- A single matrix's scaling: read across c1..c40 on its row (time should fall
  as threads rise, then flatten or rise past the memory-bandwidth knee).
- The number used in the paper:
    numeric  comparison -> the "best" column (fastest over cores),  see ../numeric.csv
    symbolic comparison -> the "c40"  column (fixed 40 threads),    see ../symbolic.csv
- Total analysis+factor for a matrix = (symbolic_mumps c40) + (mumps c40).

Note: MUMPS threads via its threaded BLAS (set OMP_NUM_THREADS / MKL_NUM_THREADS
in the launcher); the thread count in the column names is that BLAS budget.

The same layout is used by every solver's files in this directory
(pardiso/, cholmod.csv, pastix.csv, sympack.csv, stiles_auto.csv for numeric;
symbolic_*.csv for symbolic).
