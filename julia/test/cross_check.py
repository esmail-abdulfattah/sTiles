"""Recompute the cases dumped by test/extensive.jl --dump DIR with pysTiles on
the same libstiles and compare: logdet, nnz(L), diag(Q^-1) and a solve.

    STILES_LIB=/path/to/libstiles.so PYTHONPATH=../python python3 cross_check.py DIR
"""
import glob, os, sys
import numpy as np
import scipy.io
from sTiles import sTiles, library_path

d = sys.argv[1]
print("pysTiles on", library_path())
nfail = 0
for mtx in sorted(glob.glob(os.path.join(d, "*.mtx"))):
    name = os.path.basename(mtx)[:-4]
    ref = os.path.join(d, name + ".julia.txt")
    if not os.path.exists(ref):
        continue
    cores = 1; ld = None; nnzL = None; dj = []; xj = []
    for line in open(ref):
        k, v = line.split()
        if k == "cores": cores = int(v)
        elif k == "logdet": ld = float(v)
        elif k == "nnz_factor": nnzL = int(v)
        elif k == "d": dj.append(float(v))
        elif k == "x": xj.append(float(v))
    Q = scipy.io.mmread(mtx).tocsc()
    n = Q.shape[0]
    s = sTiles(Q, cores=cores, inverse=True)
    lp = s.logdet
    dp = s.selinv_diag()
    xp = s.solve(np.arange(1.0, n + 1.0))
    nnzp = s.nnz_factor
    s.close()
    dj = np.array(dj); xj = np.array(xj)
    ed = np.max(np.abs(dp - dj)) / np.max(np.abs(dj))
    ex = np.max(np.abs(xp - xj)) / np.max(np.abs(xj))
    eld = abs(lp - ld) / abs(ld)
    exact = (lp == ld) and np.array_equal(dp, dj) and np.array_equal(xp, xj)
    # nnz(L) can legitimately differ between two runs: the ordering bake-off is
    # not deterministic on every matrix (spacetime varies by ~7% run to run in
    # the same language), so it is reported, not judged.
    ok = eld < 1e-13 and ed < 1e-12 and ex < 1e-12
    nfail += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name:14s} n={n:6d} cores={cores}  "
          f"logdet rel {eld:.1e}  diag rel {ed:.1e}  solve rel {ex:.1e}  "
          f"nnz(L) {'==' if nnzp == nnzL else f'{nnzp} vs julia {nnzL} (ordering not deterministic)'}  {'bit-identical' if exact else ''}")
print(f"{nfail} failures")
sys.exit(1 if nfail else 0)
