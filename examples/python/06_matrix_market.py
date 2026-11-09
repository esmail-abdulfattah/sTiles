"""Factorize your own matrix, read from a Matrix Market (.mtx) file.

    python 06_matrix_market.py                  # uses ../data/grid40.mtx
    python 06_matrix_market.py my_matrix.mtx
"""
import sys
from pathlib import Path

import scipy.io
from sTiles import sTiles

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "../data/grid40.mtx")

# mmread expands a "symmetric" file to the full matrix; sTiles reads only the
# lower triangle, so either form works.
Q = scipy.io.mmread(path).tocsc()
print(f"{path.name}: {Q.shape[0]} x {Q.shape[1]}, {Q.nnz} non-zeros")

with sTiles(Q, cores=2) as s:
    print(f"log|Q| = {s.logdet:.6f}")
    info = s.summary()
    print(f"non-zeros in L = {info['nnz_factor']}, "
          f"analysis {info['analyze_time']:.3f} s, cholesky {info['chol_time']:.4f} s")
