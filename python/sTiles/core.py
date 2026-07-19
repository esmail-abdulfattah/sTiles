"""
High-level Python interface to sTiles.

Typical use (compute a Cholesky, log-determinant, marginal variances, solve)::

    import numpy as np, scipy.sparse as sp
    from sTiles import sTiles

    Q = ...                       # symmetric positive-definite sparse matrix
    with sTiles(Q, cores=4, inverse=True) as s:
        print(s.logdet)           # log|Q|
        var = s.selinv_diag()     # diag(Q^-1)  -> marginal variances
        x   = s.solve(b)          # Q x = b

Reuse the preprocessing -- factor many matrices that share one sparsity pattern
without repeating the (expensive) symbolic analysis::

    s = sTiles(Q0, cores=4, inverse=True)
    for theta in grid:
        s.factorize(build_Q(theta))        # reuses ordering + layout
        loglik = ... s.logdet ...
    s.close()

Notes
-----
* The matrix must be symmetric positive-definite.  Only the lower triangle is
  read; the upper triangle (if present) is ignored.
* All indices exposed by this API are 0-based in the *original* ordering of the
  input matrix -- sTiles undoes its internal fill-reducing permutation for you.
* State inside libstiles is keyed by an integer ``group`` index.  Distinct live
  :class:`sTiles` objects must use distinct ``group`` values (default 0).
"""

from __future__ import annotations

import ctypes
from ctypes import byref, c_bool, c_int, c_void_p

import numpy as np

from . import _ffi
from ._ffi import lib

__all__ = ["sTiles", "factorize", "version", "library_path"]

# tile_type_mode values (mirror stiles.h)
MODE_DENSE = 0
MODE_SEMISPARSE = 1
MODE_SPARSE = 2
MODE_AUTO = 3
_MODE_ALIASES = {
    "dense": MODE_DENSE,
    "semisparse": MODE_SEMISPARSE,
    "semi": MODE_SEMISPARSE,
    "sparse": MODE_SPARSE,
    "auto": MODE_AUTO,
}


def version() -> str:
    raw = lib.sTiles_get_version()
    return raw.decode() if raw else "unknown"


def library_path() -> str:
    """Absolute path of the loaded libstiles shared object."""
    return _ffi.library_path


def _resolve_mode(mode) -> int:
    if isinstance(mode, str):
        try:
            return _MODE_ALIASES[mode.lower()]
        except KeyError:
            raise ValueError(
                f"unknown mode {mode!r}; use one of {sorted(_MODE_ALIASES)}"
            ) from None
    return int(mode)


def _lower_coo(Q):
    """
    Reduce a square symmetric matrix to its lower triangle in a canonical COO
    order (sorted by (row, col)), returned as int32 row/col and float64 values.

    Accepts anything SciPy can turn into a COO matrix, or a dense ndarray.
    """
    import scipy.sparse as sp

    if sp.issparse(Q):
        coo = sp.tril(Q).tocoo()
        row = np.ascontiguousarray(coo.row, dtype=np.int32)
        col = np.ascontiguousarray(coo.col, dtype=np.int32)
        val = np.ascontiguousarray(coo.data, dtype=np.float64)
        n = Q.shape[0]
    else:
        A = np.asarray(Q, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("matrix must be square 2-D")
        n = A.shape[0]
        ii, jj = np.nonzero(np.tril(A))
        row = np.ascontiguousarray(ii, dtype=np.int32)
        col = np.ascontiguousarray(jj, dtype=np.int32)
        val = np.ascontiguousarray(A[ii, jj], dtype=np.float64)

    if Q.shape[0] != Q.shape[1]:
        raise ValueError("matrix must be square")

    # Canonical (row, col) order so repeated factorizations of the same pattern
    # line up value-for-value.
    order = np.lexsort((col, row))
    return int(n), row[order].copy(), col[order].copy(), val[order].copy()


class sTilesError(RuntimeError):
    pass


class sTiles:
    """
    A live sTiles factorization of one symmetric positive-definite matrix.

    Parameters
    ----------
    Q : scipy.sparse matrix or 2-D ndarray
        Symmetric positive-definite matrix.  Only the lower triangle is used.
    cores : int
        Worker threads for the factorization (default 1).
    mode : {'auto','dense','semisparse','sparse'} or int
        Tile factorization regime (default 'auto').
    tile_size : int
        Tile size; use -1 for sTiles auto-detection (default 40).
    inverse : bool
        Reserve storage for the selected inverse.  Must be True to call
        :meth:`selinv_diag`, :meth:`selinv_elm`, or :meth:`selinv_row`
        (default False).
    group : int
        libstiles group index; use distinct values for concurrent instances.
    log_level : int
        Verbosity of libstiles' own logging (default -1 = silent).  Errors are
        always shown.  Use 0 for [TIME] markers, 1 = info, 2 = debug, 3 = trace.
    factorize : bool
        If True (default) run the numeric Cholesky as part of construction.  If
        False, only the preprocessing (symbolic analysis) runs, and you call
        :meth:`factorize` yourself -- letting you time the two phases apart.

    Notes
    -----
    Construction has two phases: **preprocessing** (ordering bake-off + tile
    layout, depends only on the sparsity pattern) and the **numeric** Cholesky.
    With ``factorize=False`` the constructor stops after preprocessing::

        s = sTiles(Q, inverse=True, factorize=False)   # time preprocessing
        s.factorize()                                   # time the numeric part
        s.chol_time                                     # library-measured numeric time
    """

    def __init__(self, Q, cores=1, mode="auto", tile_size=40, inverse=False,
                 group=0, log_level=-1, factorize=True):
        self._closed = False
        self._handle = c_void_p(None)
        self.group = int(group)
        self.want_inverse = bool(inverse)
        self._factored = False
        self._selinv_done = False
        self.cores = int(cores)
        self.mode = _resolve_mode(mode)

        n, row, col, val = _lower_coo(Q)
        self.n = n
        self.nnz = row.size
        self._row = row          # keep refs alive for the lib's lifetime
        self._col = col
        self._values = val

        lib.sTiles_set_log_level(int(log_level))

        # --- global configuration (expert mode gates the setters) ----------
        lib.sTiles_expert_user()
        lib.sTiles_set_tile_size(int(tile_size))
        lib.sTiles_set_tile_type_mode(self.mode)

        # --- create handle -------------------------------------------------
        # State inside libstiles is indexed by group; a handle must be created
        # with at least (group+1) groups for `group` to be a valid slot.  Only
        # our group gets a graph; the rest stay empty.
        ng = self.group + 1
        calls = (c_int * ng)(*([1] * ng))
        cores_arr = (c_int * ng)(*([self.cores] * ng))
        chol_type = (c_int * ng)(*([0] * ng))   # 0 = sparse factorization variant
        get_inv = (c_bool * ng)(*([self.want_inverse] * ng))
        rc = lib.sTiles_create(byref(self._handle), ng, calls, cores_arr,
                               chol_type, get_inv)
        self._check(rc, "sTiles_create")

        # --- PHASE 1: preprocessing (graph + ordering + tile layout) -------
        # Symbolic only -- no numeric values, no Cholesky.
        rc = lib.sTiles_assign_graph_one_call(
            self.group, 0, byref(self._handle), self.n, self.nnz,
            row.ctypes.data_as(_ffi.c_int_p), col.ctypes.data_as(_ffi.c_int_p))
        self._check(rc, "sTiles_assign_graph_one_call")

        rc = lib.sTiles_init_group(self.group, byref(self._handle))
        self._check(rc, "sTiles_init_group")

        # --- PHASE 2: numeric factorization (optional) ---------------------
        if factorize:
            self.factorize()

    # -- internals ----------------------------------------------------------
    def _check(self, rc, what):
        if rc is not None and rc != 0:
            raise sTilesError(f"{what} failed (status {rc})")

    def _require_open(self):
        if self._closed:
            raise sTilesError("this sTiles handle has been closed")

    def _require_factored(self):
        self._require_open()
        if not self._factored:
            raise sTilesError(
                "not factorized yet; call .factorize() (preprocessing is done, "
                "the numeric Cholesky is not)")

    def factorize(self, Q=None):
        """
        Run the numeric Cholesky (assign_values + chol), reusing the
        preprocessing.  This is PHASE 2 and is what you time separately from
        construction.  Pass ``Q`` to factor a matrix with the SAME sparsity
        pattern (reusing the preprocessing); otherwise the values captured at
        construction are used.  Returns ``self``.
        """
        self._require_open()
        if Q is not None:
            _, row, col, val = _lower_coo(Q)
            if row.size != self.nnz or not (
                np.array_equal(row, self._row) and np.array_equal(col, self._col)
            ):
                raise sTilesError(
                    "factorize(Q=) requires an identical sparsity pattern; "
                    "build a new sTiles for a different pattern")
            self._values = val
        val = self._values
        rc = lib.sTiles_assign_values(
            self.group, 0, byref(self._handle),
            val.ctypes.data_as(_ffi.c_double_p))
        self._check(rc, "sTiles_assign_values")
        lib.sTiles_bind(self.group, 0, byref(self._handle))
        rc = lib.sTiles_chol(self.group, 0, byref(self._handle))
        lib.sTiles_unbind(self.group, 0, byref(self._handle))
        self._check(rc, "sTiles_chol")
        self._factored = True
        self._selinv_done = False
        return self

    def _ensure_selinv(self):
        if not self.want_inverse:
            raise sTilesError(
                "construct sTiles(..., inverse=True) to use the selected inverse")
        self._require_factored()
        if not self._selinv_done:
            lib.sTiles_bind(self.group, 0, byref(self._handle))
            rc = lib.sTiles_selinv(self.group, 0, byref(self._handle))
            lib.sTiles_unbind(self.group, 0, byref(self._handle))
            self._check(rc, "sTiles_selinv")
            self._selinv_done = True

    # -- results ------------------------------------------------------------
    @property
    def is_factored(self) -> bool:
        """Whether the numeric Cholesky has run (PHASE 2)."""
        return self._factored

    @property
    def logdet(self) -> float:
        """log-determinant of Q (== 2 * sum(log diag(L)))."""
        self._require_factored()
        return float(lib.sTiles_get_logdet(self.group, 0, byref(self._handle)))

    @property
    def nnz_factor(self) -> int:
        """Number of stored non-zeros in the Cholesky factor L."""
        self._require_open()
        return int(lib.sTiles_get_nnz_factor(self.group, 0, byref(self._handle)))

    @property
    def chol_time(self) -> float:
        """Library-measured numeric Cholesky time in seconds (no Python overhead)."""
        self._require_factored()
        return float(lib.sTiles_get_chol_timing(self.group, 0, byref(self._handle)))

    @property
    def selinv_time(self) -> float:
        """Library-measured selected-inverse time in seconds (no Python overhead)."""
        self._require_factored()
        return float(lib.sTiles_get_selinv_timing(self.group, 0, byref(self._handle)))

    def summary(self) -> dict:
        """
        Structured snapshot: dimensions, fill, mode, phase state and
        library-measured timings.  Mirrors R's ``sTiles_summary(s)`` (returns a
        dict, e.g. ``s.summary()["chol_time"]``).  Timings are ``None`` until the
        corresponding phase has run.
        """
        self._require_open()
        modes = {0: "dense", 1: "semisparse", 2: "sparse", 3: "auto"}
        return {
            "n": self.n,
            "nnz": self.nnz,
            "nnz_factor": self.nnz_factor,
            "mode": modes.get(self.mode, self.mode),
            "cores": self.cores,
            "inverse": self.want_inverse,
            "factored": self._factored,
            "chol_time": self.chol_time if self._factored else None,
            "selinv_time": self.selinv_time if self._selinv_done else None,
            "version": version(),
            "library": library_path(),
        }

    def selinv(self):
        """
        Compute the selected inverse explicitly, as its own (timeable) phase --
        ``Z = Q^-1`` restricted to the pattern of the Cholesky factor.  Requires
        ``inverse=True``.  Idempotent; otherwise it is computed lazily on the
        first ``selinv_*`` query.  Returns ``self``.
        """
        self._ensure_selinv()
        return self

    def chol_elm(self, i: int, j: int) -> float:
        """Entry L[i, j] of the Cholesky factor (0-based, original order)."""
        self._require_factored()
        return float(lib.sTiles_get_chol_elm(
            self.group, 0, int(i), int(j), byref(self._handle)))

    def selinv_elm(self, i: int, j: int) -> float:
        """
        Selected-inverse entry ``(Q^-1)[i, j]`` at ANY position (0-based,
        original order).  Returns the value when ``(i, j)`` lies in the factor
        pattern (pattern of ``L+L^T``) and exactly ``0.0`` outside it.  Both
        triangles work (Z is symmetric).  Triggers the selected-inverse
        computation on first use.
        """
        self._ensure_selinv()
        return float(lib.sTiles_get_selinv_elm(
            self.group, 0, int(i), int(j), byref(self._handle)))

    def selinv_diag(self) -> np.ndarray:
        """
        Diagonal of the selected inverse, ``diag(Q^-1)`` -- the marginal
        variances.  Length-``n`` float64 array in original order.
        """
        self._ensure_selinv()
        out = np.empty(self.n, dtype=np.float64)
        get = lib.sTiles_get_selinv_elm
        h = byref(self._handle)
        g = self.group
        for i in range(self.n):
            out[i] = get(g, 0, i, i, h)
        return out

    def selinv_row(self, node: int, neighbors) -> np.ndarray:
        """
        Values ``(Q^-1)[node, k]`` for each ``k`` in ``neighbors`` (entries that
        fall outside the factor pattern come back as 0).
        """
        self._ensure_selinv()
        nb = np.ascontiguousarray(neighbors, dtype=np.int32)
        ptr = lib.sTiles_get_selinv_row(
            self.group, 0, int(node), nb.ctypes.data_as(_ffi.c_int_p),
            nb.size, byref(self._handle))
        if not ptr:
            raise sTilesError("sTiles_get_selinv_row returned null")
        vals = np.ctypeslib.as_array(ptr, shape=(nb.size,)).copy()
        return vals

    def permutation(self) -> np.ndarray:
        """
        Fill-reducing permutation over the original nodes, as a length-``n``
        int32 array (logical, ND-padding removed).
        """
        self._require_open()
        out = np.empty(self.n, dtype=np.int32)
        m = lib.sTiles_get_logical_element_perm(
            self.group, 0, byref(self._handle), out.ctypes.data_as(_ffi.c_int_p))
        if m < 0:
            raise sTilesError("sTiles_get_logical_element_perm failed")
        return out[:m]

    # -- solves -------------------------------------------------------------
    def _solve(self, fn, b):
        self._require_factored()
        B = np.asarray(b, dtype=np.float64)
        one_d = B.ndim == 1
        if one_d:
            B = B.reshape(self.n, 1)
        if B.shape[0] != self.n:
            raise ValueError(f"rhs has {B.shape[0]} rows, expected {self.n}")
        nrhs = B.shape[1]
        # libstiles wants column-major (each rhs contiguous); overwrite in place.
        work = np.asfortranarray(B, dtype=np.float64).copy(order="F")
        lib.sTiles_bind(self.group, 0, byref(self._handle))
        rc = fn(self.group, 0, byref(self._handle),
                work.ctypes.data_as(_ffi.c_double_p), nrhs)
        lib.sTiles_unbind(self.group, 0, byref(self._handle))
        self._check(rc, fn.__name__)
        return work[:, 0].copy() if one_d else np.ascontiguousarray(work)

    def solve(self, b, system="A") -> np.ndarray:
        """
        Solve with the factorization.  ``b`` is a length-n vector or n x nrhs
        matrix.  ``system``: ``"A"`` solves ``Q x = b`` (default), ``"L"`` solves
        ``L y = b`` (forward), ``"Lt"`` solves ``L^T x = b`` (backward).
        """
        fn = {"A": lib.sTiles_solve_LLT, "L": lib.sTiles_solve_L,
              "Lt": lib.sTiles_solve_LT}.get(system)
        if fn is None:
            raise ValueError(f"system must be 'A', 'L' or 'Lt', got {system!r}")
        return self._solve(fn, b)

    def solve_L(self, b) -> np.ndarray:
        """Forward solve ``L y = b`` (alias for ``solve(b, 'L')``)."""
        return self.solve(b, "L")

    def solve_LT(self, b) -> np.ndarray:
        """Backward solve ``L^T x = b`` (alias for ``solve(b, 'Lt')``)."""
        return self.solve(b, "Lt")

    # -- value reuse (same pattern, new values) ----------------------------
    def update_values(self, Q):
        """
        Re-factor a matrix that shares this object's sparsity pattern, reusing
        the symbolic analysis (ordering + layout).  Much cheaper than building
        a fresh :class:`sTiles`.  Alias for ``factorize(Q)``; raises if the
        pattern differs.
        """
        return self.factorize(Q)

    # -- lifecycle ----------------------------------------------------------
    def close(self):
        if not self._closed:
            try:
                lib.sTiles_freeGroup(self.group)
            finally:
                self._closed = True
                self._handle = c_void_p(None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        if self._closed:
            return "<sTiles closed>"
        phase = "factorized" if self._factored else "analyzed (not factorized)"
        return (f"<sTiles n={self.n}, nnz={self.nnz}, mode={self.mode}, "
                f"cores={self.cores}, {phase}>")


def factorize(Q, **kw) -> sTiles:
    """Convenience constructor -- identical to :class:`sTiles`."""
    return sTiles(Q, **kw)
