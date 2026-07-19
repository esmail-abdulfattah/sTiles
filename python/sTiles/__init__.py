"""
sTiles -- Python bindings for the sTiles sparse Cholesky / selected-inverse
framework (KAUST).

Loads the prebuilt ``libstiles`` shared object via ctypes; no build step and no
compiler required on the user's machine.

Quick start
-----------
    import scipy.sparse as sp
    from sTiles import sTiles

    with sTiles(Q, cores=4, inverse=True) as s:
        s.logdet
        s.selinv_diag()      # diag(Q^-1)
        s.solve(b)           # Q x = b
"""

from .core import sTiles, factorize, version, library_path
from .core import (
    MODE_AUTO,
    MODE_DENSE,
    MODE_SEMISPARSE,
    MODE_SPARSE,
    sTilesError,
)

__all__ = [
    "sTiles",
    "factorize",
    "version",
    "library_path",
    "sTilesError",
    "MODE_AUTO",
    "MODE_DENSE",
    "MODE_SEMISPARSE",
    "MODE_SPARSE",
]

from ._version import __version__  # date-based CalVer, YYYY.M.D
