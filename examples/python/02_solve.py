"""Solve Q x = b for one right-hand side, then for several at once.

    python 02_solve.py
"""
import numpy as np
import scipy.sparse as sp
from sTiles import sTiles

n = 1000
Q = sp.diags([-1.0, 4.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc")

with sTiles(Q, cores=2) as s:
    # One right-hand side: a vector of length n.
    b = np.arange(1.0, n + 1)
    x = s.solve(b)
    print(f"one rhs:   x.shape = {x.shape}, max |Q x - b| = {np.abs(Q @ x - b).max():.1e}")

    # Several right-hand sides: an n x k matrix, one column each.
    # All columns are solved in one call, which is faster than a loop.
    B = np.random.default_rng(0).standard_normal((n, 5))
    X = s.solve(B)
    print(f"five rhs:  X.shape = {X.shape}, max |Q X - B| = {np.abs(Q @ X - B).max():.1e}")
