# sTiles: A Sparse-Dense Tile-Based Computational Framework

## Overview

**sTiles** is an accelerated computational framework designed for the efficient factorization of sparse structured symmetric matrices. It employs a tile-based approach to optimize sparse Cholesky factorization by exploiting the structure of arrowhead matrices with variable bandwidths. The framework integrates permutation techniques to minimize fill-in, a static scheduler for efficient task orchestration, and GPU acceleration for enhanced performance.

## Key Features

- **Sparse-Dense Tile-Based Approach**: Efficiently partitions matrices into tiles to balance computation and memory footprint.
- **GPU Acceleration**: Leverages GPU hardware to significantly improve computation speed.
- **Structure-Aware Computation**: Focuses on nonzero tiles to optimize processing for structured sparse matrices.
- **Advanced Scheduling**: Uses a left-looking Cholesky factorization with tree reduction to maximize parallelism.
- **High-Performance Computing**: Achieves significant speedups over traditional sparse solvers such as CHOLMOD, SymPACK, MUMPS, and PARDISO.

## Performance Highlights

**sTiles** has demonstrated significant performance improvements:

- **8.41X / 9.34X / 5.07X / 11.08X** speedups compared to CHOLMOD, SymPACK, MUMPS, and PARDISO, respectively.
- **5X GPU Speedup** over a 32-core AMD EPYC CPU when executed on an NVIDIA A100 GPU.

## Installation

To install **sTiles**, clone the repository and build the library using the provided makefile:

```bash
git clone https://github.com/esmail-abdulfattah/sTiles.git
cd sTiles_<>
make
