"""Analyze the sparsity pattern once, then factorize many matrices that
share it. This is the loop of a hyperparameter search or an optimizer.

    python 04_reuse_analysis.py
"""
import scipy.sparse as sp
from sTiles import sTiles

n = 100_000
T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc")
I = sp.identity(n, format="csc")

def build_Q(kappa):
    return T + kappa * I                 # same non-zero pattern for every kappa

# Step 1, once: ordering and tile layout. They depend only on WHERE the
# non-zeros are, not on their values.
s = sTiles.analyze(build_Q(1.0), cores=2)
print(f"analysis:  {s.analyze_time:.3f} s  (paid once)")

# Step 2, per value set: numeric Cholesky only.
for kappa in [0.1, 1.0, 10.0]:
    s.update(build_Q(kappa))
    print(f"kappa = {kappa:5.1f}   log|Q| = {s.logdet:12.4f}   cholesky: {s.chol_time:.4f} s")

s.close()
