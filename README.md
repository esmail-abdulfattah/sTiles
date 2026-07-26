# sTiles

High-performance **tile-based** framework for sparse **Cholesky factorization**,
**triangular solves**, and **selected inversion** of symmetric positive-definite
matrices, spanning the full spectrum from very sparse to fully dense.

Developed at King Abdullah University of Science and Technology (KAUST).

## Documentation

**Please follow [esmail-abdulfattah.github.io/sTiles](https://esmail-abdulfattah.github.io/sTiles/)
for installation, downloads, and examples for Python, R, and C/C++.**

## This repository

This repository holds the language bindings and the project website. The
`libstiles` engine itself is distributed as prebuilt binaries for Linux
(x86_64, arm64), macOS (Apple Silicon, Intel), and Windows (x86_64), published
as [release assets](https://github.com/esmail-abdulfattah/sTiles/releases/latest).
Both packages below fetch the binary for your platform automatically on first
use, so no compiler or build step is required.

```bash
pip install sTiles
```

```r
remotes::install_github("esmail-abdulfattah/sTiles", subdir = "R/sTiles")
```

## Contact

Esmail Abdul Fattah, <esmail.abdulfattah@kaust.edu.sa>
