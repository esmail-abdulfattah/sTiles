"""Factorize a sparse SPD matrix, then get the log-determinant, solve a
linear system and read the marginal variances.

    python 01_quickstart.py
"""
import numpy as np
import scipy.sparse as sp
from sTiles import sTiles

# A 1000 x 1000 tridiagonal matrix: 4 on the diagonal, -1 next to it.
n = 1000
Q = sp.diags([-1.0, 4.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc")

# inverse=True reserves room for the selected inverse (the variances below).
with sTiles(Q, cores=2, inverse=True) as s:
    print(f"log|Q|            = {s.logdet:.6f}")

    b = np.ones(n)
    x = s.solve(b)                       # x = Q^-1 b
    print(f"max |Q x - b|     = {np.abs(Q @ x - b).max():.1e}")

    v = s.selinv_diag()                  # diag(Q^-1), the marginal variances
    print(f"first variances   = {v[:3].round(6)}")
# Leaving the with-block frees the factorization.
