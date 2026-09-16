"""Cross-checks against dense NumPy on small SPD matrices, plus the lifecycle
and error paths. Mirrors the Julia runtests.jl. Run from the package root so
binaries/ is found, or with STILES_LIB set:   python -m pytest tests"""
import gc
import os

import numpy as np
import pytest
import scipy.sparse as sp

from sTiles import (MODE_AUTO, MODE_DENSE, MODE_SEMISPARSE, MODE_SPARSE,
                    library_path, sTiles, sTilesError, version)


def band_spd(n):
    """Tridiagonal SPD: diagonal 4, off-diagonals -1 (a 1-D GMRF precision)."""
    return sp.diags([[-1.0] * (n - 1), [4.0] * n, [-1.0] * (n - 1)],
                    [-1, 0, 1], format="csc")


def random_spd(n, density=0.05, seed=1):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=density, random_state=rng, format="csc",
                  data_rvs=rng.standard_normal)
    return (A.T @ A + n * sp.eye(n)).tocsc()


def logdet_dense(Q):
    sign, ld = np.linalg.slogdet(np.asarray(Q.todense()) if sp.issparse(Q) else Q)
    assert sign > 0
    return ld


@pytest.fixture(autouse=True)
def _no_live_handles():
    """Every test starts and ends with nothing live, or one failure cascades."""
    gc.collect()
    yield
    gc.collect()


def test_library():
    assert os.path.isfile(library_path())
    assert isinstance(version(), str) and version()
    want = os.environ.get("STILES_LIB")
    if want:   # the loader must not have silently fallen back to a cached copy
        assert os.path.realpath(library_path()) == os.path.realpath(want)


def test_cholesky_logdet_solve():
    n = 60
    Q = band_spd(n)
    Qd = Q.toarray()
    with sTiles(Q) as s:
        assert s.is_factored
        assert s.n == n and s.nnz == sp.tril(Q).nnz
        assert s.nnz_factor >= n
        assert s.logdet == pytest.approx(logdet_dense(Qd), rel=1e-12)

        b = np.arange(1.0, n + 1.0)
        x = s.solve(b)
        assert x.shape == (n,)
        assert x == pytest.approx(np.linalg.solve(Qd, b), rel=1e-12)
        assert np.linalg.norm(Q @ x - b) / np.linalg.norm(b) < 1e-12
        assert np.array_equal(b, np.arange(1.0, n + 1.0))     # untouched

        B = np.random.default_rng(2).standard_normal((n, 3))
        X = s.solve(B)
        assert X.shape == (n, 3)
        assert X == pytest.approx(np.linalg.solve(Qd, B), rel=1e-12)
        # a Fortran-ordered and a non-contiguous rhs give the same answer
        assert s.solve(np.asfortranarray(B)) == pytest.approx(X, rel=1e-14)
        assert s.solve(B[:, ::-1]) == pytest.approx(X[:, ::-1], rel=1e-14)

        # forward then backward must equal the full solve
        y = s.solve(b, system="L")
        assert s.solve(y, system="Lt") == pytest.approx(x, rel=1e-12)
        assert s.solve_LT(s.solve_L(b)) == pytest.approx(x, rel=1e-12)
        with pytest.raises(ValueError):
            s.solve(b, system="foo")
        with pytest.raises(ValueError):
            s.solve(np.ones(n + 1))

        info = s.summary()
        assert info["n"] == n and info["nnz"] == sp.tril(Q).nnz and info["factored"]
        assert isinstance(info["chol_time"], float) and info["chol_time"] >= 0
        assert info["selinv_time"] is None
        assert info["analyze_time"] >= 0
        assert info["library"] == library_path()
        assert "factorized" in repr(s)
        assert sorted(s.permutation()) == list(range(n))

        s.close()
        assert repr(s) == "<sTiles closed>"
        with pytest.raises(sTilesError):
            s.logdet
        s.close()                                            # idempotent


def test_selected_inverse():
    n = 40
    Q = random_spd(n)
    Qinv = np.linalg.inv(Q.toarray())
    with sTiles(Q, inverse=True) as s:
        assert s.selinv() is s
        assert isinstance(s.summary()["selinv_time"], float)
        d = s.selinv_diag()
        assert d.shape == (n,)
        assert d == pytest.approx(np.diag(Qinv), rel=1e-10)
        for i in (0, 6, n - 1):
            assert s.selinv_elm(i, i) == pytest.approx(Qinv[i, i], rel=1e-10)
        # every in-pattern entry matches the dense inverse; off-pattern is 0
        nmatch = 0
        for j in range(n):
            for i in range(j, n):
                z = s.selinv_elm(i, j)
                assert z == s.selinv_elm(j, i)
                if z != 0:
                    assert z == pytest.approx(Qinv[i, j], rel=1e-8)
                    nmatch += 1
        assert nmatch >= n
        r = s.selinv_row(4, [4, 0, 1])
        assert r[0] == pytest.approx(Qinv[4, 4], rel=1e-10)
        assert r[1] == s.selinv_elm(4, 0)
    with sTiles(Q) as s:                                     # no inverse storage
        with pytest.raises(sTilesError):
            s.selinv_diag()


def test_selinv_recomputed_after_update():
    n = 50
    Q = band_spd(n)
    with sTiles(Q, inverse=True) as s:
        d1 = s.selinv_diag()
        s.update(2.0 * Q)
        assert s.selinv_diag() == pytest.approx(d1 / 2.0, rel=1e-12)


def test_chol_elm():
    n = 30
    Q = band_spd(n)
    with sTiles(Q) as s:
        assert s.chol_elm(0, 0) > 0
        assert 2 * sum(np.log(s.chol_elm(i, i)) for i in range(n)) == \
            pytest.approx(s.logdet, rel=1e-10)


def test_analyze_then_factorize_reuse_pattern():
    n = 50
    Q1 = band_spd(n)
    s = sTiles.analyze(Q1)
    try:
        assert not s.is_factored
        assert "not factorized" in repr(s)
        assert s.summary()["chol_time"] is None
        with pytest.raises(sTilesError):
            s.logdet
        with pytest.raises(sTilesError):
            s.solve(np.ones(n))
        assert s.factorize() is s
        assert s.logdet == pytest.approx(logdet_dense(Q1), rel=1e-12)

        Q2 = 2.0 * Q1                                        # same pattern
        assert s.update(Q2) is s
        assert s.logdet == pytest.approx(logdet_dense(Q2), rel=1e-12)
        assert s.solve(np.ones(n)) == pytest.approx(
            np.linalg.solve(Q2.toarray(), np.ones(n)), rel=1e-12)
        s.update_values(Q1)                                  # deprecated alias
        assert s.logdet == pytest.approx(logdet_dense(Q1), rel=1e-12)

        Q3 = Q1 + sp.diags([[0.1] * (n - 2)] * 2, [-2, 2])   # other pattern
        with pytest.raises(sTilesError):
            s.factorize(Q3)
        assert s.logdet == pytest.approx(logdet_dense(Q1), rel=1e-12)  # still usable
        with pytest.raises(sTilesError):
            sTiles(Q1, factorize=False).factorize(band_spd(n + 1))
    finally:
        s.close()


@pytest.mark.parametrize("make", [
    lambda Q: Q.toarray(),                     # dense input
    lambda Q: Q.tocsr(),
    lambda Q: Q.tocoo(),
    lambda Q: Q.tolil(),
    lambda Q: sp.tril(Q).tocsc(),              # lower triangle only
    lambda Q: Q.astype(np.float32),
    lambda Q: (Q * 1).astype(np.int64),        # integer values (4, -1 exact)
])
def test_input_forms(make):
    n = 20
    Q = band_spd(n)
    ld = logdet_dense(Q)
    with sTiles(make(Q)) as s:
        assert s.logdet == pytest.approx(ld, rel=1e-6)


def test_upper_triangle_ignored():
    n = 20
    Q = band_spd(n).tolil()
    Q[0, 1] = 100.0
    with sTiles(Q.tocsc()) as s:
        assert s.logdet == pytest.approx(logdet_dense(band_spd(n)), rel=1e-12)


@pytest.mark.parametrize("mode", ["dense", "semisparse", "semi", "sparse", "auto",
                                  MODE_DENSE, MODE_SEMISPARSE, MODE_SPARSE, MODE_AUTO])
def test_modes(mode):
    n = 200
    Q = random_spd(n, density=0.02)
    with sTiles(Q, mode=mode) as s:
        assert s.logdet == pytest.approx(logdet_dense(Q), rel=1e-10)


@pytest.mark.parametrize("tile_size", [-1, 8, 40, 120])
def test_tile_sizes(tile_size):
    n = 300
    Q = band_spd(n)
    with sTiles(Q, tile_size=tile_size, cores=2) as s:
        assert s.logdet == pytest.approx(logdet_dense(Q), rel=1e-12)
        assert s.solve(np.ones(n)) == pytest.approx(
            np.linalg.solve(Q.toarray(), np.ones(n)), rel=1e-10)


def test_bad_inputs():
    with pytest.raises(ValueError):
        sTiles(band_spd(10), mode="bogus")
    with pytest.raises(ValueError):
        sTiles(sp.random(5, 6, 0.5, format="csc"))
    with pytest.raises(ValueError):
        sTiles(np.ones((3, 4)))
    with pytest.raises(ValueError):
        sTiles(np.ones(5))


def test_cores():
    n = 80
    Q = random_spd(n)
    with sTiles(Q, cores=2) as s:
        assert s.summary()["cores"] == 2
        assert s.solve(np.ones(n)) == pytest.approx(
            np.linalg.solve(Q.toarray(), np.ones(n)), rel=1e-10)


@pytest.mark.parametrize("nrhs", [1, 7, 64, 100, 257])
def test_rhs_widths(nrhs):
    n = 150
    Q = random_spd(n, density=0.03, seed=5)
    B = np.random.default_rng(nrhs).standard_normal((n, nrhs))
    with sTiles(Q, cores=2) as s:
        X = s.solve(B)
        assert X.shape == (n, nrhs)
        assert np.linalg.norm(Q @ X - B) / np.linalg.norm(B) < 1e-11


def test_errors_and_lifecycle():
    n = 10
    Q = band_spd(n)
    with sTiles(Q):
        with pytest.raises(sTilesError):
            sTiles(Q)                                        # one live handle
    with sTiles(Q) as s:                                     # clean re-init after quit
        assert s.logdet == pytest.approx(logdet_dense(Q), rel=1e-12)
    # not positive definite: clean error, nothing left live
    N = sp.diags([1.0, -1.0, 1.0]).tocsc()
    with pytest.raises(sTilesError):
        sTiles(N)
    with sTiles(Q) as s:
        assert s.logdet == pytest.approx(logdet_dense(Q), rel=1e-12)
    # non-PD values on a good pattern: update fails, and closing the handle
    # and building a new one always works
    with sTiles(Q) as s:
        with pytest.raises(sTilesError):
            s.update(-1.0 * Q)
    with sTiles(Q) as s:
        assert s.logdet == pytest.approx(logdet_dense(Q), rel=1e-12)


@pytest.mark.parametrize("n,cores,mode", [
    (300, 1, "sparse"), (300, 4, "dense"), (300, 4, "semisparse"), (2000, 4, "auto"),
    pytest.param(10, 1, "auto", marks=pytest.mark.xfail(
        reason="libstiles: the serial dense path keeps ss_abort set after a "
               "non-PD failure, so the handle reports -112 forever "
               "(single tile, or cores=1 dense/semisparse); close and rebuild")),
    pytest.param(300, 1, "dense", marks=pytest.mark.xfail(reason="same ss_abort poison")),
    pytest.param(300, 1, "semisparse", marks=pytest.mark.xfail(reason="same ss_abort poison")),
])
def test_handle_survives_nonpd_update(n, cores, mode):
    Q = band_spd(n)
    with sTiles(Q, cores=cores, mode=mode) as s:
        ld = s.logdet
        with pytest.raises(sTilesError):
            s.update(-1.0 * Q)
        s.update(Q)                         # good values again: must recover
        assert s.logdet == pytest.approx(ld, rel=1e-12)


def test_gc_releases_the_handle():
    n = 10
    Q = band_spd(n)
    s = sTiles(Q)
    del s                                                    # no close(): finalizer must free it
    gc.collect()
    with sTiles(Q) as t:
        assert t.logdet == pytest.approx(logdet_dense(Q), rel=1e-12)


def test_many_cycles():
    n = 30
    Q = band_spd(n)
    ld = logdet_dense(Q)
    for _ in range(40):
        with sTiles(Q, inverse=True) as s:
            assert s.logdet == pytest.approx(ld, rel=1e-12)
            assert s.selinv_diag()[0] > 0


def test_factorize_convenience():
    from sTiles import factorize
    n = 12
    with factorize(band_spd(n)) as s:
        assert s.logdet == pytest.approx(logdet_dense(band_spd(n)), rel=1e-12)


FERRIS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "ideas",
                      "adv_sTiles", "run", "mtx", "inla_graph_ferris.mtx")


@pytest.mark.skipif(not os.path.exists(os.environ.get("STILES_TEST_FERRIS", FERRIS)),
                    reason="ferris.mtx not available")
def test_ferris():
    import scipy.io
    Q = scipy.io.mmread(os.environ.get("STILES_TEST_FERRIS", FERRIS)).tocsc()
    with sTiles(Q, cores=4, inverse=True) as s:
        assert s.logdet == pytest.approx(296806.395604036, rel=1e-12)
        assert s.nnz_factor > Q.nnz
        b = np.arange(1.0, Q.shape[0] + 1.0)
        x = s.solve(b)
        assert np.linalg.norm(Q @ x - b) / np.linalg.norm(b) < 1e-10
        d = s.selinv_diag()
        assert np.all(d > 0)
        # marginal variance of node 0 also from a full solve column
        e0 = np.zeros(Q.shape[0]); e0[0] = 1.0
        assert d[0] == pytest.approx(s.solve(e0)[0], rel=1e-10)
