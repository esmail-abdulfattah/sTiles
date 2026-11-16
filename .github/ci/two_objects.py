import numpy as np
import scipy.sparse as sp
import sTiles


def g(n):
    return sp.diags([-np.ones(n - 1), 4 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csc")


def ld(Q):
    return np.linalg.slogdet(Q.toarray())[1]


Q1, Q2 = g(2000), 2 * g(500)
s = sTiles.sTiles(Q1, cores=2)
ss = sTiles.sTiles(Q2, cores=2, inverse=True)
assert abs(s.logdet - ld(Q1)) < 1e-8 and abs(ss.logdet - ld(Q2)) < 1e-8
s.close()
assert np.allclose(ss.selinv_diag(), np.diag(np.linalg.inv(Q2.toarray())), rtol=1e-10)
sss = sTiles.sTiles(Q1, cores=2)
assert abs(sss.logdet - ld(Q1)) < 1e-8 and abs(ss.logdet - ld(Q2)) < 1e-8
ss.close()
sss.close()
Q3 = g(20)
s3 = sTiles.sTiles(Q3, mode="sparse")
assert abs(s3.logdet - ld(Q3)) < 1e-10
assert np.allclose(Q3 @ s3.solve(np.ones(20)), 1.0)
s3.close()
print("OK")
