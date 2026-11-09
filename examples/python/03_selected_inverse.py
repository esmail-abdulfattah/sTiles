"""Read entries of Q^-1 without forming the dense inverse: the diagonal,
single entries, and a row. Checked here against numpy's dense inverse.

    python 03_selected_inverse.py
"""
import numpy as np
import scipy.sparse as sp
from sTiles import sTiles

n = 200
Q = sp.diags([-1.0, 4.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc")
Qinv = np.linalg.inv(Q.toarray())        # only to check the answers

with sTiles(Q, cores=2, inverse=True) as s:
    s.selinv()                           # optional: otherwise runs on first query

    d = s.selinv_diag()                  # all n variances
    print(f"diagonal:   max error = {np.abs(d - np.diag(Qinv)).max():.1e}")

    # One entry. Indices are 0-based, in the original ordering of Q.
    print(f"Qinv[10, 11] = {s.selinv_elm(10, 11):.6f}   (dense: {Qinv[10, 11]:.6f})")

    # Entries are exact inside the pattern of the Cholesky factor, which
    # sTiles stores in whole tiles, and 0 outside it. Ask only for entries
    # you know are in the pattern: the diagonal and the non-zeros of Q are.
    print(f"Qinv[10, 150] = {s.selinv_elm(10, 150):.1f}   (dense: {Qinv[10, 150]:.1e}, outside the pattern)")

    # A row restricted to chosen columns, e.g. the neighbours of node 10.
    row = s.selinv_row(10, [9, 10, 11])
    print(f"row 10 at [9, 10, 11] = {row.round(6)}")
