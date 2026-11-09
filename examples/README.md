# sTiles examples

Six short programs. Each one is written in Python, R, Julia and C, and all four versions print the same numbers.

| #  | Example              | What it shows                                                         |
| -- | -------------------- | --------------------------------------------------------------------- |
| 01 | `quickstart`         | Factorize, then the log-determinant, a solve, and marginal variances |
| 02 | `solve`              | `Q x = b` for one right-hand side and for many at once                |
| 03 | `selected_inverse`   | Entries of `Q^-1` without the dense inverse: diagonal, entries, rows  |
| 04 | `reuse_analysis`     | Analyze a sparsity pattern once, factorize many value sets            |
| 05 | `sampling`           | Draws from `N(0, Q^-1)` with the triangular solve                     |
| 06 | `matrix_market`      | Factorize your own matrix from a `.mtx` file                          |

Walkthrough with outputs: <https://esmail-abdulfattah.github.io/sTiles/examples.html>

## Get the examples

```bash
git clone --depth 1 https://github.com/esmail-abdulfattah/sTiles.git
cd sTiles/examples
```

## Python

```bash
pip install sTiles
cd python
python 01_quickstart.py
```

## R

```r
install.packages("sTiles")           # from CRAN
sTiles::sTiles_install_library()     # once: downloads the solver
```

```bash
cd r
Rscript 01_quickstart.R
```

## Julia

```julia
pkg> add https://github.com/esmail-abdulfattah/sTiles:julia
```

```bash
cd julia
julia 01_quickstart.jl
```

## C

Download and unpack the archive for your platform from the [download page](https://esmail-abdulfattah.github.io/sTiles/download.html).
It holds `include/stiles.h`, `lib/libstiles.*` and `lib/pkgconfig/stiles.pc`.

```bash
cd c
make STILES=/path/to/libstiles-linux-x86_64     # builds all six
./01_quickstart
make STILES=/path/to/libstiles-linux-x86_64 run # builds and runs all six
```

`common.h` holds the helpers the C examples share: error checks, the test matrix, a residual and a timer. None of it is part of sTiles.
On Windows, see the end of `c/Makefile`.

## The test matrix

Examples 01 to 05 use a tridiagonal matrix with 4 on the diagonal and -1 next to it. It is easy to build in every language, and its inverse is known.
Example 06 reads `data/grid40.mtx`, a 1600 x 1600 grid Laplacian, or any `.mtx` file you pass on the command line.
