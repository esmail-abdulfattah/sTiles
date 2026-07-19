// ---------------------------------------------------------------------------
// run_mtx.cpp — factor a REAL matrix (Matrix Market .mtx) with libstiles.
//
// Loads a symmetric coordinate-real .mtx (e.g. the INLA group2 matrices),
// factors it at 1,2,4,8… cores, and reports chol time + speedup PLUS a
// self-validating correctness check that needs no dense reference:
//   * logdet is finite
//   * solve residual ||Q x - b|| / ||b||  (Q applied straight from the sparse
//     COO, so it works for any size)
//
// Build/run (see Makefile):
//   make mtx LIB=libstiles-macos-apple-arm64 MTX=inla_graph_ferris.mtx
//   ./run_mtx <matrix.mtx> [maxcores] [tile_size]
// ---------------------------------------------------------------------------
#include "stiles.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using std::vector;

struct Ent { int r, c; double v; };

// Load a Matrix Market "coordinate real symmetric" file into lower-triangle COO
// (0-based, sorted by (row,col)). Returns n, or -1 on failure.
static int load_mtx(const char* path, vector<int>& row, vector<int>& col,
                    vector<double>& val) {
    FILE* f = std::fopen(path, "r");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return -1; }
    char line[512];
    long nr = 0, nc = 0, nnz = 0;
    while (std::fgets(line, sizeof line, f)) {                 // skip % comments
        if (line[0] == '%' || line[0] == '\n') continue;
        if (std::sscanf(line, "%ld %ld %ld", &nr, &nc, &nnz) == 3) break;
    }
    if (nr <= 0 || nr != nc) { std::fclose(f); return -1; }
    vector<Ent> e; e.reserve(nnz);
    long i, j; double v;
    for (long k = 0; k < nnz && std::fscanf(f, "%ld %ld %lf", &i, &j, &v) == 3; ++k) {
        --i; --j;                                             // 1-based -> 0-based
        if (i < j) std::swap(i, j);                           // keep lower triangle
        e.push_back({(int)i, (int)j, v});
    }
    std::fclose(f);
    std::sort(e.begin(), e.end(), [](const Ent& a, const Ent& b) {
        return a.r != b.r ? a.r < b.r : a.c < b.c;            // (row, col) order
    });
    row.resize(e.size()); col.resize(e.size()); val.resize(e.size());
    for (size_t k = 0; k < e.size(); ++k) { row[k]=e[k].r; col[k]=e[k].c; val[k]=e[k].v; }
    return (int)nr;
}

// ||Q x - b|| / ||b|| with Q applied straight from the lower COO (symmetric).
static double residual(int n, int nnz, const int* row, const int* col,
                       const double* val, const double* x, const double* b) {
    vector<double> Qx(n, 0.0);
    for (int k = 0; k < nnz; ++k) {
        int i = row[k], j = col[k]; double v = val[k];
        Qx[i] += v * x[j];
        if (i != j) Qx[j] += v * x[i];                        // symmetric partner
    }
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) { double d = Qx[i]-b[i]; num += d*d; den += b[i]*b[i]; }
    return std::sqrt(num / den);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <matrix.mtx> [maxcores] [tile_size]\n", argv[0]); return 2; }
    const char* path = argv[1];
    int hw       = (int)std::thread::hardware_concurrency(); if (hw < 1) hw = 8;
    int maxcores = argc > 2 ? std::atoi(argv[2]) : hw;
    int tsz      = argc > 3 ? std::atoi(argv[3]) : 40;
    if (maxcores < 1) maxcores = 1;

    vector<int> row, col; vector<double> val;
    int n = load_mtx(path, row, col, val);
    if (n < 0) { std::fprintf(stderr, "failed to load %s\n", path); return 2; }
    const int nnz = (int)row.size();

    std::printf("libstiles real-matrix run\n");
    std::printf("  version : %s\n", sTiles_get_version());
    std::printf("  matrix  : %s   n=%d  nnz(lower)=%d  (%.1f nnz/row)\n",
                path, n, nnz, (double)nnz / n);
    std::printf("  hardware: %d logical cores  (sweeping up to %d, tile_size=%d)\n\n",
                hw, maxcores, tsz);

    // rhs for the residual check
    vector<double> b(n), x(n);
    for (int i = 0; i < n; ++i) b[i] = 1.0 + 0.1 * std::sin(0.5 * i);

    vector<int> sweep;
    for (int c = 1; c <= maxcores; c *= 2) sweep.push_back(c);
    if (sweep.empty() || sweep.back() != maxcores) sweep.push_back(maxcores);

    std::printf("  cores |  chol time (s) |  speedup |      logdet      | solve resid\n");
    std::printf("  ------+----------------+----------+------------------+------------\n");
    double t1 = -1.0;
    for (int c : sweep) {
        sTiles_set_log_level(-1);
        sTiles_expert_user();
        sTiles_set_tile_size(tsz);
        sTiles_set_tile_type_mode(3);                          // auto (route by structure)
        void* h = nullptr;
        int ng = 1, calls[1] = {1}, ncores[1] = {c}, ctype[1] = {0}; bool inv[1] = {false};
        if (sTiles_create(&h, ng, calls, ncores, ctype, inv) != 0) { std::printf("  %5d | create failed\n", c); continue; }
        if (sTiles_assign_graph_one_call(0,0,&h,n,nnz,row.data(),col.data()) != 0) { std::printf("  %5d | assign_graph failed\n", c); sTiles_freeGroup(0); continue; }
        if (sTiles_init_group(0,&h) != 0) { std::printf("  %5d | init failed\n", c); sTiles_freeGroup(0); continue; }
        sTiles_assign_values(0,0,&h,val.data());
        sTiles_bind(0,0,&h);
        int rc = sTiles_chol(0,0,&h);
        sTiles_unbind(0,0,&h);
        if (rc != 0) { std::printf("  %5d | chol failed\n", c); sTiles_freeGroup(0); continue; }
        double t  = sTiles_get_chol_timing(0,0,&h);
        double ld = sTiles_get_logdet(0,0,&h);
        // solve + residual (once is enough, but cheap; do it every row)
        x = b;
        sTiles_bind(0,0,&h);
        sTiles_solve_LLT(0,0,&h,x.data(),1);
        sTiles_unbind(0,0,&h);
        double res = residual(n, nnz, row.data(), col.data(), val.data(), x.data(), b.data());
        sTiles_freeGroup(0);
        if (c == 1) t1 = t;
        double sp = (t1 > 0.0) ? t1 / t : 1.0;
        std::printf("  %5d |  %13.4f |  %7.2fx |  %15.6f | %.2e\n", c, t, sp, ld, res);
    }
    std::printf("\n(logdet should match across core counts; solve residual should be ~1e-10 or better.)\n");
    return 0;
}
