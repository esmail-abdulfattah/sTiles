// ---------------------------------------------------------------------------
// scale_stiles.cpp — does libstiles actually use multiple cores here?
//
// Factorizes ONE moderately heavy banded SPD matrix at increasing core counts
// and prints the Cholesky time + speedup, so you can see whether the library
// scales on this machine (Mac or Linux). Uses the library's own timer
// (sTiles_get_chol_timing), which measures the numeric factorization only
// (ordering/symbolic excluded). No correctness check here — test_stiles.cpp
// already validated that; this is purely a scaling probe.
//
// Build/run (see Makefile):
//   make scale LIB=libstiles-macos-apple-arm64                 # defaults
//   ./scale_stiles <maxcores> <n> <tile_size>                  # e.g. 8 4000 128
//
// A DENSE SPD matrix is used on purpose: tile Cholesky (what sTiles is) is
// compute-bound and parallelizes well on dense work, so this is the honest
// "does it use multiple cores" probe. (A sparse/banded matrix is memory-bound
// and barely scales — that's a property of the problem, not the library.)
// If the 1-core time is tiny (< ~0.3 s), bump n and re-run.
// ---------------------------------------------------------------------------
#include "stiles.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using std::vector;

// Lower-triangle COO (0-based, sorted by (row,col)) of a DENSE SPD matrix:
// every entry present, strongly diagonally dominant so it is safely SPD.
static void dense_lower_coo(int n, vector<int>& row, vector<int>& col,
                            vector<double>& val) {
    row.clear(); col.clear(); val.clear();
    const size_t nnz = (size_t)n * (n + 1) / 2;
    row.reserve(nnz); col.reserve(nnz); val.reserve(nnz);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            row.push_back(i); col.push_back(j);
            val.push_back(i == j ? (double)n + (i % 7)      // diag-dominant -> SPD
                                 : 1.0 / (1.0 + (i - j)));
        }
    }
}

// Factor the matrix on `cores` workers; return the library's chol time (s).
static double factor_time(int cores, int n, int nnz, int tile_size,
                          int* row, int* col, double* val, double* logdet_out) {
    sTiles_set_log_level(-1);
    sTiles_expert_user();
    sTiles_set_tile_size(tile_size);
    sTiles_set_tile_type_mode(0);                   // 0 = dense tiles

    void* h = nullptr;
    const int ng = 1;
    int  calls[1]   = {1};
    int  ncores[1]  = {cores};
    int  ctype[1]   = {0};
    bool inv[1]     = {false};                      // factor only, no selinv
    if (sTiles_create(&h, ng, calls, ncores, ctype, inv) != 0) return -1.0;
    if (sTiles_assign_graph_one_call(0, 0, &h, n, nnz, row, col) != 0) return -1.0;
    if (sTiles_init_group(0, &h) != 0) return -1.0;

    sTiles_assign_values(0, 0, &h, val);
    sTiles_bind(0, 0, &h);
    int rc = sTiles_chol(0, 0, &h);
    sTiles_unbind(0, 0, &h);
    double t = -1.0;
    if (rc == 0) {
        t = sTiles_get_chol_timing(0, 0, &h);
        *logdet_out = sTiles_get_logdet(0, 0, &h);
    }
    sTiles_freeGroup(0);
    return t;
}

int main(int argc, char** argv) {
    int hw       = (int)std::thread::hardware_concurrency(); if (hw < 1) hw = 8;
    int maxcores = argc > 1 ? std::atoi(argv[1]) : hw;
    int n        = argc > 2 ? std::atoi(argv[2]) : 4000;
    int tsz      = argc > 3 ? std::atoi(argv[3]) : 128;
    if (maxcores < 1) maxcores = 1;

    std::printf("libstiles scaling probe\n");
    std::printf("  version : %s\n", sTiles_get_version());
    std::printf("  matrix  : DENSE SPD  n=%d  tile_size=%d\n", n, tsz);
    std::printf("  hardware: %d logical cores  (sweeping up to %d)\n\n", hw, maxcores);

    vector<int> row, col; vector<double> val;
    dense_lower_coo(n, row, col, val);
    const int nnz = (int)row.size();

    // Core counts to try: 1, 2, 4, 8, ... capped at maxcores (always include maxcores).
    vector<int> sweep;
    for (int c = 1; c <= maxcores; c *= 2) sweep.push_back(c);
    if (sweep.empty() || sweep.back() != maxcores) sweep.push_back(maxcores);

    std::printf("  cores |  chol time (s) |  speedup |  parallel eff.\n");
    std::printf("  ------+----------------+----------+---------------\n");
    double t1 = -1.0;
    for (int c : sweep) {
        double ld = 0.0;
        double t = factor_time(c, n, nnz, tsz, row.data(), col.data(), val.data(), &ld);
        if (t <= 0.0) { std::printf("  %5d |  (factor failed)\n", c); continue; }
        if (c == 1) t1 = t;
        double sp  = (t1 > 0.0) ? t1 / t : 1.0;
        double eff = sp / (double)c * 100.0;
        std::printf("  %5d |  %13.4f |  %7.2fx |  %10.0f%%\n", c, t, sp, eff);
    }
    std::printf("\n(speedup vs 1 core; parallel efficiency = speedup / cores.)\n");
    if (t1 >= 0.0 && t1 < 0.3)
        std::printf("NOTE: 1-core time is small (%.3fs) — bump n for a clearer scaling signal.\n", t1);
    return 0;
}
