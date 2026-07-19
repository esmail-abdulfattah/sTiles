/*
 * sTiles_glue.c -- thin R/C glue between R and libstiles.
 *
 * Wraps the opaque sTiles handle in an R external pointer with a finalizer, so
 * an R-side `sTiles` object owns exactly one live factorization and frees it
 * (sTiles_freeGroup) when garbage-collected or closed.
 *
 * Ownership note: sTiles_assign_graph_one_call RETAINS the row/col pointers
 * (it does not copy them) and reads them over the handle's lifetime, so this
 * glue keeps malloc'd copies in the ctx and frees them in the finalizer --
 * never handing R's own vector memory to the library.
 *
 * The numeric-value pointer, by contrast, is consumed during assign_values, so
 * passing REAL(x) straight through is safe.
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>
#include <string.h>

#include "stiles.h"      /* vendored copy of tools/include/stiles.h */

typedef struct {
    void*     handle;
    int       group;
    int       n;
    long long nnz;
    int       inverse;      /* selected-inverse storage reserved at create   */
    int       factored;     /* has a numeric Cholesky run yet?                */
    int       selinv_done;  /* has sTiles_selinv run since the last chol?     */
    int*      row;          /* ctx-owned copies handed to sTiles (retained)  */
    int*      col;
} sTiles_ctx;

static sTiles_ctx* ctx_from(SEXP ext) {
    if (TYPEOF(ext) != EXTPTRSXP)
        Rf_error("not an sTiles handle");
    sTiles_ctx* c = (sTiles_ctx*) R_ExternalPtrAddr(ext);
    if (!c) Rf_error("sTiles handle is NULL (already freed?)");
    return c;
}

static void require_factored(sTiles_ctx* c) {
    if (!c->factored)
        Rf_error("not factorized yet; run sTiles_factorize(s) "
                 "(preprocessing is done, the numeric Cholesky is not)");
}

static void sTiles_finalize(SEXP ext) {
    sTiles_ctx* c = (sTiles_ctx*) R_ExternalPtrAddr(ext);
    if (!c) return;
    if (c->handle) sTiles_freeGroup(c->group);   /* frees before we drop row/col */
    free(c->row);
    free(c->col);
    free(c);
    R_ClearExternalPtr(ext);
}

static void ensure_selinv(sTiles_ctx* c) {
    if (!c->inverse)
        Rf_error("this handle was built without inverse=TRUE; "
                 "the selected inverse is unavailable");
    require_factored(c);
    if (!c->selinv_done) {
        sTiles_bind(c->group, 0, &c->handle);
        int rc = sTiles_selinv(c->group, 0, &c->handle);
        sTiles_unbind(c->group, 0, &c->handle);
        if (rc != 0) Rf_error("sTiles_selinv failed (status %d)", rc);
        c->selinv_done = 1;
    }
}

/* PHASE 1 -- preprocessing (symbolic) only: create + assign_graph + init_group.
 * i,j are 0-based lower-triangle COO (INTSXP). No numeric values, no Cholesky.
 * This is the ordering bake-off + tile layout; time it separately from the
 * numeric factorization done by sTiles_factorize_R. */
SEXP sTiles_analyze_R(SEXP i_, SEXP j_, SEXP n_, SEXP cores_,
                      SEXP mode_, SEXP ts_, SEXP inv_, SEXP group_,
                      SEXP loglevel_) {
    const int n       = Rf_asInteger(n_);
    const int nnz     = LENGTH(i_);
    const int cores   = Rf_asInteger(cores_);
    const int mode    = Rf_asInteger(mode_);
    const int ts      = Rf_asInteger(ts_);
    const int inverse = Rf_asLogical(inv_) == TRUE;
    const int group   = Rf_asInteger(group_);
    const int loglvl  = Rf_asInteger(loglevel_);

    if (LENGTH(j_) != nnz)
        Rf_error("i and j must have equal length");

    sTiles_set_log_level(loglvl);
    sTiles_expert_user();
    sTiles_set_tile_size(ts);
    sTiles_set_tile_type_mode(mode);

    sTiles_ctx* c = (sTiles_ctx*) calloc(1, sizeof(sTiles_ctx));
    if (!c) Rf_error("out of memory");
    c->group = group; c->n = n; c->nnz = nnz; c->inverse = inverse;

    /* ctx-owned copies of the index arrays (sTiles retains these pointers). */
    c->row = (int*) malloc((size_t) nnz * sizeof(int));
    c->col = (int*) malloc((size_t) nnz * sizeof(int));
    if (!c->row || !c->col) { free(c->row); free(c->col); free(c);
                              Rf_error("out of memory"); }
    memcpy(c->row, INTEGER(i_), (size_t) nnz * sizeof(int));
    memcpy(c->col, INTEGER(j_), (size_t) nnz * sizeof(int));

    /* A handle must span at least (group+1) groups for `group` to be valid. */
    const int ng = group + 1;
    int* calls  = (int*) R_alloc(ng, sizeof(int));
    int* coresv = (int*) R_alloc(ng, sizeof(int));
    int* ctype  = (int*) R_alloc(ng, sizeof(int));
    for (int g = 0; g < ng; ++g) { calls[g] = 1; coresv[g] = cores; ctype[g] = 0; }
    /* sTiles_create wants const bool*; build a _Bool array. */
    _Bool* ginv_b = (_Bool*) R_alloc(ng, sizeof(_Bool));
    for (int g = 0; g < ng; ++g) ginv_b[g] = inverse ? 1 : 0;

    if (sTiles_create(&c->handle, ng, calls, coresv, ctype, ginv_b) != 0) {
        free(c->row); free(c->col); free(c);
        Rf_error("sTiles_create failed");
    }

    sTiles_assign_graph_one_call(group, 0, &c->handle, n, nnz, c->row, c->col);
    if (sTiles_init_group(group, &c->handle) != 0) {
        sTiles_freeGroup(group);
        free(c->row); free(c->col); free(c);
        Rf_error("sTiles_init_group failed");
    }

    SEXP ext = PROTECT(R_MakeExternalPtr(c, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ext, sTiles_finalize, TRUE);
    Rf_setAttrib(ext, R_ClassSymbol, Rf_mkString("sTiles_ptr"));
    UNPROTECT(1);
    return ext;
}

/* PHASE 2 -- numeric factorization only: assign_values + Cholesky. Reuses the
 * preprocessing from sTiles_analyze_R, so calling it repeatedly with new values
 * that share the pattern reuses that work. Times the numeric work alone. */
SEXP sTiles_factorize_R(SEXP ext, SEXP x_) {
    sTiles_ctx* c = ctx_from(ext);
    if (LENGTH(x_) != c->nnz)
        Rf_error("value vector length %d != pattern nnz %lld",
                 LENGTH(x_), c->nnz);
    sTiles_assign_values(c->group, 0, &c->handle, REAL(x_));
    sTiles_bind(c->group, 0, &c->handle);
    int rc = sTiles_chol(c->group, 0, &c->handle);
    sTiles_unbind(c->group, 0, &c->handle);
    if (rc != 0)
        Rf_error("sTiles_chol failed (status %d) -- matrix not positive-definite?",
                 rc);
    c->factored = 1;
    c->selinv_done = 0;
    return ext;
}

/* Library-measured numeric Cholesky time (seconds); excludes R overhead. */
SEXP sTiles_chol_time_R(SEXP ext) {
    sTiles_ctx* c = ctx_from(ext);
    require_factored(c);
    return Rf_ScalarReal(sTiles_get_chol_timing(c->group, 0, &c->handle));
}

/* Has a numeric factorization run? Read from the C handle (survives the R
 * copy-on-modify semantics that a list field would not). */
SEXP sTiles_is_factored_R(SEXP ext) {
    return Rf_ScalarLogical(ctx_from(ext)->factored);
}

/* Compute the selected inverse explicitly (its own timed phase). Idempotent:
 * a no-op if already computed since the last factorization. Requires the
 * handle to have been built with inverse=TRUE. */
SEXP sTiles_selinv_R(SEXP ext) {
    ensure_selinv(ctx_from(ext));
    return ext;
}

/* Library-measured selected-inverse time (seconds); excludes R overhead. */
SEXP sTiles_selinv_time_R(SEXP ext) {
    sTiles_ctx* c = ctx_from(ext);
    if (!c->selinv_done)
        Rf_error("selected inverse not computed yet; call sTiles_selinv(s)");
    return Rf_ScalarReal(sTiles_get_selinv_timing(c->group, 0, &c->handle));
}

SEXP sTiles_logdet_R(SEXP ext) {
    sTiles_ctx* c = ctx_from(ext);
    require_factored(c);
    return Rf_ScalarReal(sTiles_get_logdet(c->group, 0, &c->handle));
}

SEXP sTiles_nnz_factor_R(SEXP ext) {
    sTiles_ctx* c = ctx_from(ext);
    return Rf_ScalarReal((double) sTiles_get_nnz_factor(c->group, 0, &c->handle));
}

SEXP sTiles_dim_R(SEXP ext) {
    return Rf_ScalarInteger(ctx_from(ext)->n);
}

SEXP sTiles_selinv_diag_R(SEXP ext) {
    sTiles_ctx* c = ctx_from(ext);
    ensure_selinv(c);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, c->n));
    double* d = REAL(out);
    for (int i = 0; i < c->n; ++i)
        d[i] = sTiles_get_selinv_elm(c->group, 0, i, i, &c->handle);
    UNPROTECT(1);
    return out;
}

/* i, j are 1-based from R; converted to 0-based here. */
SEXP sTiles_selinv_elm_R(SEXP ext, SEXP i_, SEXP j_) {
    sTiles_ctx* c = ctx_from(ext);
    ensure_selinv(c);
    return Rf_ScalarReal(sTiles_get_selinv_elm(
        c->group, 0, Rf_asInteger(i_) - 1, Rf_asInteger(j_) - 1, &c->handle));
}

SEXP sTiles_chol_elm_R(SEXP ext, SEXP i_, SEXP j_) {
    sTiles_ctx* c = ctx_from(ext);
    require_factored(c);
    return Rf_ScalarReal(sTiles_get_chol_elm(
        c->group, 0, Rf_asInteger(i_) - 1, Rf_asInteger(j_) - 1, &c->handle));
}

/* neighbors: 1-based node indices from R. */
SEXP sTiles_selinv_row_R(SEXP ext, SEXP node_, SEXP neighbors_) {
    sTiles_ctx* c = ctx_from(ext);
    ensure_selinv(c);
    const int m = LENGTH(neighbors_);
    int* nb = (int*) R_alloc(m, sizeof(int));
    int* src = INTEGER(neighbors_);
    for (int k = 0; k < m; ++k) nb[k] = src[k] - 1;   /* 0-based */
    double* vals = sTiles_get_selinv_row(c->group, 0, Rf_asInteger(node_) - 1,
                                         nb, m, &c->handle);
    if (!vals) Rf_error("sTiles_get_selinv_row returned NULL");
    SEXP out = PROTECT(Rf_allocVector(REALSXP, m));
    memcpy(REAL(out), vals, (size_t) m * sizeof(double));
    free(vals);   /* libstiles malloc'd it */
    UNPROTECT(1);
    return out;
}

/* Solve Q X = B.  B is a column-major (n x nrhs) numeric matrix. */
SEXP sTiles_solve_R(SEXP ext, SEXP B_, SEXP which_) {
    sTiles_ctx* c = ctx_from(ext);
    require_factored(c);
    const int total = LENGTH(B_);
    if (total % c->n != 0)
        Rf_error("rhs length %d is not a multiple of n=%d", total, c->n);
    const int nrhs = total / c->n;
    const int which = Rf_asInteger(which_);   /* 0=LLT, 1=L, 2=LT */

    SEXP out = PROTECT(Rf_allocVector(REALSXP, total));
    memcpy(REAL(out), REAL(B_), (size_t) total * sizeof(double));
    double* work = REAL(out);

    sTiles_bind(c->group, 0, &c->handle);
    int rc;
    if      (which == 1) rc = sTiles_solve_L (c->group, 0, &c->handle, work, nrhs);
    else if (which == 2) rc = sTiles_solve_LT(c->group, 0, &c->handle, work, nrhs);
    else                 rc = sTiles_solve_LLT(c->group, 0, &c->handle, work, nrhs);
    sTiles_unbind(c->group, 0, &c->handle);
    if (rc != 0) { UNPROTECT(1); Rf_error("solve failed (status %d)", rc); }

    if (nrhs > 1) {
        SEXP dim = PROTECT(Rf_allocVector(INTSXP, 2));
        INTEGER(dim)[0] = c->n; INTEGER(dim)[1] = nrhs;
        Rf_setAttrib(out, R_DimSymbol, dim);
        UNPROTECT(1);
    }
    UNPROTECT(1);
    return out;
}

/* Logical fill-reducing permutation over original nodes (1-based for R). */
SEXP sTiles_perm_R(SEXP ext) {
    sTiles_ctx* c = ctx_from(ext);
    int* buf = (int*) R_alloc(c->n, sizeof(int));
    int m = sTiles_get_logical_element_perm(c->group, 0, &c->handle, buf);
    if (m < 0) Rf_error("sTiles_get_logical_element_perm failed");
    SEXP out = PROTECT(Rf_allocVector(INTSXP, m));
    int* o = INTEGER(out);
    for (int k = 0; k < m; ++k) o[k] = buf[k] + 1;    /* 1-based */
    UNPROTECT(1);
    return out;
}

SEXP sTiles_free_R(SEXP ext) {
    sTiles_finalize(ext);
    return R_NilValue;
}

SEXP sTiles_version_R(void) {
    const char* v = sTiles_get_version();
    return Rf_mkString(v ? v : "unknown");
}

/* ---- registration -------------------------------------------------------- */
static const R_CallMethodDef CallEntries[] = {
    {"sTiles_analyze_R",       (DL_FUNC) &sTiles_analyze_R,         9},
    {"sTiles_factorize_R",     (DL_FUNC) &sTiles_factorize_R,       2},
    {"sTiles_chol_time_R",     (DL_FUNC) &sTiles_chol_time_R,       1},
    {"sTiles_is_factored_R",   (DL_FUNC) &sTiles_is_factored_R,     1},
    {"sTiles_selinv_R",        (DL_FUNC) &sTiles_selinv_R,          1},
    {"sTiles_selinv_time_R",   (DL_FUNC) &sTiles_selinv_time_R,     1},
    {"sTiles_logdet_R",        (DL_FUNC) &sTiles_logdet_R,          1},
    {"sTiles_nnz_factor_R",    (DL_FUNC) &sTiles_nnz_factor_R,      1},
    {"sTiles_dim_R",           (DL_FUNC) &sTiles_dim_R,             1},
    {"sTiles_selinv_diag_R",   (DL_FUNC) &sTiles_selinv_diag_R,     1},
    {"sTiles_selinv_elm_R",    (DL_FUNC) &sTiles_selinv_elm_R,      3},
    {"sTiles_chol_elm_R",      (DL_FUNC) &sTiles_chol_elm_R,        3},
    {"sTiles_selinv_row_R",    (DL_FUNC) &sTiles_selinv_row_R,      3},
    {"sTiles_solve_R",         (DL_FUNC) &sTiles_solve_R,           3},
    {"sTiles_perm_R",          (DL_FUNC) &sTiles_perm_R,            1},
    {"sTiles_free_R",          (DL_FUNC) &sTiles_free_R,            1},
    {"sTiles_version_R",       (DL_FUNC) &sTiles_version_R,         0},
    {NULL, NULL, 0}
};

/* Named R_init_sTiles to match the package/DLL name "sTiles". */
void R_init_sTiles(DllInfo* dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
