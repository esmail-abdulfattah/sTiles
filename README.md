# sTiles

High-performance **tile-based** framework for sparse **Cholesky factorization**,
**triangular solves**, and **selected inversion** of symmetric positive-definite
matrices, spanning the full spectrum from very sparse to fully dense.

Developed at King Abdullah University of Science and Technology (KAUST).

## Documentation

**Please follow [esmail-abdulfattah.github.io/sTiles](https://esmail-abdulfattah.github.io/sTiles/)
for installation, downloads, and examples for Python, R, Julia, and C/C++.**

## This repository

This repository holds the language bindings and the project website. The
`libstiles` engine itself is distributed as prebuilt binaries for Linux
(x86_64, arm64), macOS (Apple Silicon, Intel), and Windows (x86_64), published
as [release assets](https://github.com/esmail-abdulfattah/sTiles/releases/latest).
None of the packages needs a compiler or a build step. The Python and Julia
packages fetch the binary for your platform on first use; the R package asks
first, and installs it on request with `sTiles::sTiles_install_library()`.

```bash
pip install sTiles
```

```r
install.packages("sTiles", repos = c("https://esmail-abdulfattah.r-universe.dev", "https://cloud.r-project.org"))
```

```julia
pkg> add https://github.com/esmail-abdulfattah/sTiles:julia
```

## Contact

Esmail Abdul Fattah, <esmail.abdulfattah@kaust.edu.sa>
