"""Draw samples from N(0, Q^-1), the Gaussian with precision matrix Q.

If Q = C C^T, then x = C^-T z with z ~ N(0, I) has covariance Q^-1.
solve(z, "Lt") computes exactly that, in the original ordering of Q.

    python 05_sampling.py
"""
import numpy as np
import scipy.sparse as sp
from sTiles import sTiles

n = 1000
Q = sp.diags([-1.0, 4.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc")
rng = np.random.default_rng(1)

with sTiles(Q, cores=2, inverse=True) as s:
    Z = rng.standard_normal((n, 10_000))  # 10 000 draws, one per column
    X = s.solve(Z, "Lt")                   # each column ~ N(0, Q^-1)

    # Check: the sample variances match the exact ones from the selected inverse.
    exact = s.selinv_diag()
    print(f"exact variance of x[0]   = {exact[0]:.4f}")
    print(f"sample variance of x[0]  = {X[0].var():.4f}")

    # With a mean mu, a draw from N(mu, Q^-1) is mu + X[:, k].
