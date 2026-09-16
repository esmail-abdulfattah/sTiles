# sTiles (Python)

Python interface to the **sTiles** sparse Cholesky and selected-inverse
framework. Pure `ctypes` over the prebuilt `libstiles` shared object, so there
is no build step and no compiler on the user's machine.

```python
import scipy.sparse as sp
from sTiles import sTiles

with sTiles(Q, cores=4, mode="auto", inverse=True) as s:
    s.logdet             # log|Q|
    s.selinv_diag()      # diag(Q^-1), the marginal variances
    s.solve(b)           # Q x = b
```

Reuse the preprocessing, one symbolic analysis and many numeric
factorizations:

```python
s = sTiles.analyze(Q0, cores=4, inverse=True)   # ordering + tile layout
for theta in grid:
    s.factorize(build_Q(theta))                 # numeric Cholesky only
    ll = ... s.logdet ...
s.close()
```

## Install

```bash
pip install sTiles
```

The matching `libstiles` for your platform is downloaded from the project's
GitHub release the first time you use the package, and cached. To use a
specific build instead, set `STILES_LIB=/path/to/libstiles.so`. Use
`python -m sTiles info` to see which solver is loaded and from where,
`python -m sTiles update` to refresh it.

Documentation, the full API and the R and Julia interfaces:
<https://esmail-abdulfattah.github.io/sTiles/>
