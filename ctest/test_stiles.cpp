// ---------------------------------------------------------------------------
// test_stiles.cpp — self-contained smoke test for a prebuilt libstiles.
//
// Links ONLY against libstiles (the .so/.dylib you downloaded from CI) and its
// public C header stiles.h. It builds a small sparse SPD matrix Q, factors it
// with sTiles, and cross-checks four things against a plain dense reference
// computed inside this file (so the test needs no BLAS/LAPACK of its own):
//
//   1. log|Q|                         (log-determinant)
//   2. diag(Q^-1)                     (marginal variances, via selected inverse)
//   3. Q x = b                        (a linear solve)
//   4. value reuse                    (re-factor same pattern, new numbers)
//
// The call sequence mirrors the validated pysTiles binding exactly. Prints
// PASS/FAIL per check and exits non-zero if any check fails.
//
// Build/run (see the Makefile in this directory):
//   make run LIB=../binaries/libstiles-linux-x86_64        # Linux x86_64
//   make run LIB=../binaries/libstiles-linux-arm64         # Linux arm64
//   make run LIB=../binaries/libstiles-macos-apple-arm64   # macOS Apple Silicon
// ---------------------------------------------------------------------------
#include "stiles.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using std::vector;

// ---- tiny dense reference (row-major n x n), plain C++, no external libs ----
struct Dense {
    int n;
    vector<double> a;                       // n*n, row-major
    Dense(int n_) : n(n_), a((size_t)n_ * n_, 0.0) {}
    double& operator()(int i, int j)       { return a[(size_t)i * n + j]; }
    double  operator()(int i, int j) const { return a[(size_t)i * n + j]; }
};

// Lower Cholesky factor L (A = L L^T), returns false if not SPD.
static bool cholesky(const Dense& A, Dense& L) {
    int n = A.n;
    for (int j = 0; j < n; ++j) {
        double d = A(j, j);
        for (int k = 0; k < j; ++k) d -= L(j, k) * L(j, k);
        if (d <= 0.0) return false;
        L(j, j) = std::sqrt(d);
        for (int i = j + 1; i < n; ++i) {
            double s = A(i, j);
            for (int k = 0; k < j; ++k) s -= L(i, k) * L(j, k);
            L(i, j) = s / L(j, j);
        }
    }
    return true;
}

// Solve A x = b given L (A = L L^T). x and b length n; x may alias b.
static void chol_solve(const Dense& L, const double* b, double* x) {
    int n = L.n;
    vector<double> y(n);
    for (int i = 0; i < n; ++i) {           // forward: L y = b
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= L(i, k) * y[k];
        y[i] = s / L(i, i);
    }
    for (int i = n - 1; i >= 0; --i) {      // backward: L^T x = y
        double s = y[i];
        for (int k = i + 1; k < n; ++k) s -= L(k, i) * x[k];
        x[i] = s / L(i, i);
    }
}

// ---- problem: a 2-D-grid-like SPD matrix (strictly diagonally dominant) -----
static Dense build_Q(int n, double diag_bump) {
    Dense Q(n);
    for (int i = 0; i < n; ++i) Q(i, i) = 4.0 + diag_bump * (i % 5);
    for (int i = 1; i < n; ++i) { Q(i, i - 1) = Q(i - 1, i) = -1.0; }   // nearest
    for (int i = 8; i < n; ++i) { Q(i, i - 8) = Q(i - 8, i) = -0.3; }   // "far" band
    return Q;
}

// Extract the LOWER triangle (i>=j) as 0-based COO, sorted by (row, col) —
// exactly what sTiles_assign_graph_one_call + sTiles_assign_values expect.
static void lower_coo(const Dense& Q, vector<int>& row, vector<int>& col,
                      vector<double>& val) {
    row.clear(); col.clear(); val.clear();
    for (int i = 0; i < Q.n; ++i)
        for (int j = 0; j <= i; ++j)
            if (Q(i, j) != 0.0) { row.push_back(i); col.push_back(j); val.push_back(Q(i, j)); }
}

// ---- one check: report and accumulate failures ------------------------------
static int g_fail = 0, g_total = 0;
static void check(const char* name, double err, double tol) {
    ++g_total;
    bool ok = std::isfinite(err) && err <= tol;
    std::printf("  [%s] %-26s  err=%.3e  (tol=%.0e)\n",
                ok ? "PASS" : "FAIL", name, err, tol);
    if (!ok) ++g_fail;
}

int main() {
    const int    n         = 120;
    const int    group     = 0;
    const int    call       = 0;
    const int    cores     = 1;      // 1 core: simple + deterministic
    const int    tile_size = 40;     // mirrors pysTiles default
    const int    mode      = 3;      // 3 = auto (mirrors pysTiles "auto")
    const double tol       = 1e-8;

    std::printf("libstiles smoke test\n");
    std::printf("  version: %s\n", sTiles_get_version());
    std::printf("  n=%d  cores=%d  tile_size=%d  mode=auto\n\n", n, cores, tile_size);

    // ---------- build the matrix + dense reference ----------
    Dense Q = build_Q(n, 0.1);
    Dense L(n);
    if (!cholesky(Q, L)) { std::printf("reference matrix not SPD (bug in test)\n"); return 2; }

    double ref_logdet = 0.0;
    for (int i = 0; i < n; ++i) ref_logdet += 2.0 * std::log(L(i, i));

    vector<double> ref_diag_inv(n);         // diag(Q^-1): solve Q z = e_i, take z[i]
    {
        vector<double> e(n, 0.0), z(n, 0.0);
        for (int i = 0; i < n; ++i) {
            e[i] = 1.0; chol_solve(L, e.data(), z.data()); ref_diag_inv[i] = z[i]; e[i] = 0.0;
        }
    }

    vector<int> row, col; vector<double> val;
    lower_coo(Q, row, col, val);
    const int nnz = (int)row.size();

    // ---------- sTiles: global config (expert mode gates the setters) --------
    sTiles_set_log_level(-1);               // -1 silences; set 1 to see lib errors
    sTiles_expert_user();
    sTiles_set_tile_size(tile_size);
    sTiles_set_tile_type_mode(mode);

    // ---------- create a 1-group / 1-call handle, inverse enabled ------------
    void* h = nullptr;
    const int  ng          = 1;
    int  calls_per_grp[1]  = {1};
    int  cores_per_grp[1]  = {cores};
    int  chol_type[1]      = {0};           // 0 = standard factorization variant
    bool want_inverse[1]   = {true};        // enable selected inverse (selinv)
    if (sTiles_create(&h, ng, calls_per_grp, cores_per_grp, chol_type, want_inverse) != 0) {
        std::printf("sTiles_create failed\n"); return 2;
    }

    // ---------- symbolic: graph + ordering + layout (retains row/col ptrs) ---
    if (sTiles_assign_graph_one_call(group, call, &h, n, nnz, row.data(), col.data()) != 0) {
        std::printf("sTiles_assign_graph_one_call failed\n"); return 2;
    }
    if (sTiles_init_group(group, &h) != 0) { std::printf("sTiles_init_group failed\n"); return 2; }

    // ---------- numeric factorization ----------------------------------------
    sTiles_assign_values(group, call, &h, val.data());
    sTiles_bind(group, call, &h);
    int rc_chol = sTiles_chol(group, call, &h);
    sTiles_unbind(group, call, &h);
    if (rc_chol != 0) { std::printf("sTiles_chol failed\n"); return 2; }

    // 1) log-determinant
    double logdet = sTiles_get_logdet(group, call, &h);
    check("logdet", std::fabs(logdet - ref_logdet) / std::fabs(ref_logdet), tol);

    // 2) diag(Q^-1) via selected inverse
    sTiles_bind(group, call, &h);
    int rc_inv = sTiles_selinv(group, call, &h);
    sTiles_unbind(group, call, &h);
    if (rc_inv != 0) { std::printf("sTiles_selinv failed\n"); return 2; }
    double max_dinv = 0.0;
    for (int i = 0; i < n; ++i) {
        double v = sTiles_get_selinv_elm(group, call, i, i, &h);
        max_dinv = std::fmax(max_dinv, std::fabs(v - ref_diag_inv[i]));
    }
    check("diag(Q^-1)", max_dinv, tol);

    // 3) solve Q x = b  (b overwritten in place with x, column-major)
    vector<double> b(n), x(n), ref_x(n);
    for (int i = 0; i < n; ++i) b[i] = 0.3 + std::sin(0.7 * i);
    chol_solve(L, b.data(), ref_x.data());
    x = b;
    sTiles_bind(group, call, &h);
    int rc_solve = sTiles_solve_LLT(group, call, &h, x.data(), 1);
    sTiles_unbind(group, call, &h);
    if (rc_solve != 0) { std::printf("sTiles_solve_LLT failed\n"); return 2; }
    // residual ||Q x - b|| / ||b||
    double rnum = 0.0, rden = 0.0;
    for (int i = 0; i < n; ++i) {
        double qx = 0.0;
        for (int j = 0; j < n; ++j) qx += Q(i, j) * x[j];
        rnum += (qx - b[i]) * (qx - b[i]);
        rden += b[i] * b[i];
    }
    check("solve  ||Qx-b||/||b||", std::sqrt(rnum / rden), tol);

    // 4) value reuse (INLA fast path): same pattern, new numbers, reuse symbolic
    Dense Q2 = build_Q(n, 0.1);
    for (int i = 0; i < n; ++i) Q2(i, i) += 0.5;      // shift the diagonal
    Dense L2(n); cholesky(Q2, L2);
    double ref_logdet2 = 0.0;
    for (int i = 0; i < n; ++i) ref_logdet2 += 2.0 * std::log(L2(i, i));
    vector<int> r2, c2; vector<double> v2; lower_coo(Q2, r2, c2, v2);
    sTiles_assign_values(group, call, &h, v2.data());
    sTiles_bind(group, call, &h);
    int rc_chol2 = sTiles_chol(group, call, &h);
    sTiles_unbind(group, call, &h);
    if (rc_chol2 != 0) { std::printf("sTiles_chol (reuse) failed\n"); return 2; }
    double logdet2 = sTiles_get_logdet(group, call, &h);
    check("reuse logdet", std::fabs(logdet2 - ref_logdet2) / std::fabs(ref_logdet2), tol);

    sTiles_freeGroup(group);

    std::printf("\n%s (%d/%d checks passed)\n",
                g_fail == 0 ? "ALL OK" : "FAILURES", g_total - g_fail, g_total);
    return g_fail == 0 ? 0 : 1;
}
