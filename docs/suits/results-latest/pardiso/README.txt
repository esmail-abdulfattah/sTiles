PARDISO raw per-core timing sweeps
================================================================================

Two CSV files, one row per matrix (89 SPD matrices). Both are PARDISO run on
the Intel node (2x Xeon Gold, 40 cores) at several thread counts. Times are in
seconds and EXCLUDE matrix loading (timed separately).

  pardiso.csv            NUMERIC factorization time (chol, phase 22)
  symbolic_pardiso.csv   SYMBOLIC analysis time     (reorder + symbolic, phase 11)

These are the per-core source data; the paper's wide comparison files
(../numeric.csv, ../symbolic.csv) pull PARDISO's column from here.

Columns
-------
matrix        matrix name
n             dimension
nnz           nonzeros, upper triangle as stored
c1 c2 c4 c8 c16 c32 c40
              the timed quantity at 1, 2, 4, 8, 16, 32, 40 threads (seconds).
              So each row is the same matrix solved at 7 thread counts.
best | min    the fastest of the seven (pardiso.csv calls it "best",
              symbolic_pardiso.csv calls it "min"); same meaning = the minimum
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
- Total analysis+factor for a matrix = (symbolic_pardiso c40) + (pardiso c40),
  e.g. Emilia_923: 8.12 + 24.76 = 32.88 s.

The same layout is used by every solver's files in this directory
(mumps.csv, cholmod.csv, pastix.csv, sympack.csv, stiles_auto.csv for numeric;
symbolic_*.csv for symbolic).
