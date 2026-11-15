/*
 * sTiles_glue.c -- thin R/C glue between R and libstiles.
 *
 * Wraps the opaque sTiles handle in an R external pointer with a finalizer, so
 * an R-side `sTiles` object owns exactly one live factorization and frees it
 * (sTiles_freeGroup) when garbage-collected or closed.
 *
 * One library copy per handle: libstiles keeps its state process-wide (memory
 * keyed by group number, thread teams, settings, sTiles_quit), so two handles
 * in one copy would share group 0. A file at a different path, loaded
 * RTLD_LOCAL, is a separate instance with its own globals, so each live handle
 * runs in a copy of its own and calls it through that copy's function table
 * (stiles_lib). The R side keeps the pool of copies (.sTiles_lib_for_handle).
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef int         (*fn_sTiles_assign_graph_one_call_t)(int, int, void**, int, int, int*, int*);
typedef int         (*fn_sTiles_assign_values_t)(int, int, void**, double*);
typedef int         (*fn_sTiles_bind_t)(int, int, void**);
typedef int         (*fn_sTiles_chol_t)(int, int, void**);
typedef int         (*fn_sTiles_create_t)(void**, int, const int*, const int*, const int*, const bool*);
typedef void        (*fn_sTiles_expert_user_t)(void);
typedef void        (*fn_sTiles_freeGroup_t)(int);
typedef double      (*fn_sTiles_get_chol_elm_t)(int, int, int, int, void**);
typedef double      (*fn_sTiles_get_chol_timing_t)(int, int, void**);
typedef double      (*fn_sTiles_get_logdet_t)(int, int, void**);
typedef int         (*fn_sTiles_get_logical_element_perm_t)(int, int, void**, int*);
typedef long long   (*fn_sTiles_get_nnz_factor_t)(int, int, void**);
typedef double      (*fn_sTiles_get_selinv_elm_t)(int, int, int, int, void**);
typedef double*     (*fn_sTiles_get_selinv_row_t)(int, int, int, int*, int, void**);
typedef double      (*fn_sTiles_get_selinv_timing_t)(int, int, void**);
typedef const char* (*fn_sTiles_get_version_t)(void);
typedef int         (*fn_sTiles_init_group_t)(int, void**);
typedef void        (*fn_sTiles_quit_t)(void);
typedef int         (*fn_sTiles_selinv_t)(int, int, void**);
typedef void        (*fn_sTiles_set_log_level_t)(int);
typedef void        (*fn_sTiles_set_tile_size_t)(int);
typedef void        (*fn_sTiles_set_tile_type_mode_t)(int);
typedef int         (*fn_sTiles_solve_L_t)(int, int, void**, double*, int);
typedef int         (*fn_sTiles_solve_LLT_t)(int, int, void**, double*, int);
typedef int         (*fn_sTiles_solve_LT_t)(int, int, void**, double*, int);
typedef int         (*fn_sTiles_unbind_t)(int, int, void**);
typedef int         (*fn_compute_ordering_t)(const int*, const int*, int, int, int, int,
                                             int, int, int*, int*, int*, int*, int*);

/* One loaded copy of libstiles. */
typedef struct {
    void* dl;                 /* dlopen / LoadLibrary handle */
    int   busy;               /* a live sTiles handle runs in this copy */
    fn_sTiles_assign_graph_one_call_t     assign_graph_one_call;
    fn_sTiles_assign_values_t             assign_values;
    fn_sTiles_bind_t                      bind;
    fn_sTiles_chol_t                      chol;
    fn_sTiles_create_t                    create;
    fn_sTiles_expert_user_t               expert_user;
    fn_sTiles_freeGroup_t                 freeGroup;
    fn_sTiles_get_chol_elm_t              get_chol_elm;
    fn_sTiles_get_chol_timing_t           get_chol_timing;
    fn_sTiles_get_logdet_t                get_logdet;
    fn_sTiles_get_logical_element_perm_t  get_logical_element_perm;
    fn_sTiles_get_nnz_factor_t            get_nnz_factor;
    fn_sTiles_get_selinv_elm_t            get_selinv_elm;
    fn_sTiles_get_selinv_row_t            get_selinv_row;
    fn_sTiles_get_selinv_timing_t         get_selinv_timing;
    fn_sTiles_get_version_t               get_version;
    fn_sTiles_init_group_t                init_group;
    fn_sTiles_quit_t                      quit;
    fn_sTiles_selinv_t                    selinv;
    fn_sTiles_set_log_level_t             set_log_level;
    fn_sTiles_set_tile_size_t             set_tile_size;
    fn_sTiles_set_tile_type_mode_t        set_tile_type_mode;
    fn_sTiles_solve_L_t                   solve_L;
    fn_sTiles_solve_LLT_t                 solve_LLT;
    fn_sTiles_solve_LT_t                  solve_LT;
    fn_sTiles_unbind_t                    unbind;
    fn_compute_ordering_t                 compute_ordering;  /* optional, see below */
} stiles_lib;

/* ---- loading a copy ------------------------------------------------------ */

#ifdef _WIN32
/* Altered search path: the first copy finds its sibling DLLs in its own
 * directory. Later copies live elsewhere and get those siblings by name, since
 * Windows reuses a dependency that is already loaded. */
static void* lib_open(const char* path) {
    return (void*) LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
}
static void lib_close(void* dl) { FreeLibrary((HMODULE) dl); }
static void* lib_sym(void* dl, const char* name) {
    FARPROC p = GetProcAddress((HMODULE) dl, name);
    void* v;
    memcpy(&v, &p, sizeof v);
    return v;
}
static const char* lib_error(void) {
    static char buf[64];
    snprintf(buf, sizeof buf, "error %lu", (unsigned long) GetLastError());
    return buf;
}
#else
static void* lib_open(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static void lib_close(void* dl) { dlclose(dl); }
static void* lib_sym(void* dl, const char* name) { return dlsym(dl, name); }
static const char* lib_error(void) { const char* e = dlerror(); return e ? e : "unknown error"; }
#endif

/* memcpy, not a cast: ISO C has no object-to-function pointer cast */
static int bind_one(void* dl, const char* name, void* dst) {
    void* v = lib_sym(dl, name);
    if (!v) return 0;
    memcpy(dst, &v, sizeof v);
    return 1;
}

/* NULL on success, else the first missing symbol. */
static const char* lib_bind(stiles_lib* L) {
#define REQ(field) if (!bind_one(L->dl, "sTiles_" #field, &L->field)) return "sTiles_" #field
    REQ(assign_graph_one_call); REQ(assign_values);   REQ(bind);
    REQ(chol);                  REQ(create);          REQ(expert_user);
    REQ(freeGroup);             REQ(get_chol_elm);    REQ(get_chol_timing);
    REQ(get_logdet);            REQ(get_logical_element_perm);
    REQ(get_nnz_factor);        REQ(get_selinv_elm);  REQ(get_selinv_row);
    REQ(get_selinv_timing);     REQ(get_version);     REQ(init_group);
    REQ(quit);                  REQ(selinv);          REQ(set_log_level);
    REQ(set_tile_size);         REQ(set_tile_type_mode);
    REQ(solve_L);               REQ(solve_LLT);       REQ(solve_LT);
    REQ(unbind);
#undef REQ
    /* Optional: absent from releases up to Version_2026.09.04. Left NULL there,
     * and sTiles_ordering() reports it instead of the whole package failing. */
    if (!bind_one(L->dl, "sTiles_compute_ordering", &L->compute_ordering))
        L->compute_ordering = NULL;
    return NULL;
}

static stiles_lib* lib_from(SEXP ext) {
    if (TYPEOF(ext) != EXTPTRSXP)
        Rf_error("not an sTiles library copy");
    stiles_lib* L = (stiles_lib*) R_ExternalPtrAddr(ext);
    if (!L) Rf_error("sTiles library copy already closed");
    return L;
}

/* Load the library file at `path` as a new copy. No finalizer: the R side
 * keeps every copy in its pool and closes them in .sTiles_unload. */
SEXP sTiles_lib_open_R(SEXP path_) {
    const char* path = CHAR(STRING_ELT(path_, 0));
    stiles_lib* L = (stiles_lib*) calloc(1, sizeof(stiles_lib));
    if (!L) Rf_error("out of memory");
    L->dl = lib_open(path);
    if (!L->dl) {
        free(L);
        Rf_error("sTiles: could not load '%s' (%s)", path, lib_error());
    }
    const char* missing = lib_bind(L);
    if (missing) {
        lib_close(L->dl);
        free(L);
        Rf_error("sTiles: could not resolve symbol '%s' in %s", missing, path);
    }
    SEXP ext = PROTECT(R_MakeExternalPtr(L, R_NilValue, R_NilValue));
    Rf_setAttrib(ext, R_ClassSymbol, Rf_mkString("sTiles_lib"));
    UNPROTECT(1);
    return ext;
}

SEXP sTiles_lib_busy_R(SEXP ext) {
    return Rf_ScalarLogical(lib_from(ext)->busy);
}

SEXP sTiles_lib_close_R(SEXP ext) {
    stiles_lib* L = (stiles_lib*) R_ExternalPtrAddr(ext);
    if (!L) return R_NilValue;
    if (L->busy) Rf_error("sTiles library copy still holds a live handle");
    lib_close(L->dl);
    free(L);
    R_ClearExternalPtr(ext);
    return R_NilValue;
}

/* ---- standalone ordering ------------------------------------------------- */

SEXP sTiles_has_ordering_R(SEXP lib_) {
    return Rf_ScalarLogical(lib_from(lib_)->compute_ordering != NULL);
}

/* i, j: 0-based lower-triangle COO from .sTiles_lower_coo (not retained by the
 * library, so R's own vectors are passed straight through). Returns a list;
 * every index in it is converted to 1-based here, root parents become 0. */
SEXP sTiles_ordering_R(SEXP lib_, SEXP i_, SEXP j_, SEXP n_, SEXP tile_, SEXP cores_) {
    fn_compute_ordering_t fn = lib_from(lib_)->compute_ordering;
    if (!fn)
        Rf_error("the loaded libstiles does not export sTiles_compute_ordering; "
                 "it needs a library newer than Version_2026.09.04 "
                 "(see sTiles_install_library)");
    const int n   = Rf_asInteger(n_);
    const int nnz = LENGTH(i_);
    if (LENGTH(j_) != nnz) Rf_error("i and j differ in length");

    SEXP perm   = PROTECT(Rf_allocVector(INTSXP, n));
    SEXP iperm  = PROTECT(Rf_allocVector(INTSXP, n));
    SEXP range  = PROTECT(Rf_allocVector(INTSXP, n + 1));
    SEXP parent = PROTECT(Rf_allocVector(INTSXP, n));
    int nb = 0;
    const int win = fn(INTEGER(i_), INTEGER(j_), n, nnz,
                       Rf_asInteger(tile_), Rf_asInteger(cores_),
                       /* ordering_mode 0 and dense_last 1: the candidate set
                        * and the dense-node preprocessing are the solver's own,
                        * never a user choice */
                       0, 1,
                       INTEGER(perm), INTEGER(iperm),
                       INTEGER(range), INTEGER(parent), &nb);
    if (win < 0) {
        UNPROTECT(4);
        if (Rf_asInteger(tile_) <= 0)
            Rf_error("sTiles_compute_ordering failed: tile_size <= 0 ranks by "
                     "exact nnz(L), which this solver build may not have; use "
                     "tile_size > 0");
        Rf_error("sTiles_compute_ordering failed (the library's error output "
                 "says why: check the input is the lower triangle with 0-based "
                 "indices in range)");
    }
    int* p = INTEGER(perm);  int* ip = INTEGER(iperm);
    for (int k = 0; k < n; ++k) { p[k] += 1; ip[k] += 1; }

    SEXP rng = PROTECT(Rf_allocVector(INTSXP, nb > 0 ? nb + 1 : 0));
    SEXP par = PROTECT(Rf_allocVector(INTSXP, nb > 0 ? nb : 0));
    if (nb > 0) {
        const int* r0 = INTEGER(range); const int* p0 = INTEGER(parent);
        int* r1 = INTEGER(rng);         int* p1 = INTEGER(par);
        for (int k = 0; k <= nb; ++k) r1[k] = r0[k] + 1;
        for (int k = 0; k < nb; ++k)  p1[k] = (p0[k] < 0) ? 0 : p0[k] + 1;
    }

    const char* names[] = {"n", "perm", "iperm", "winner", "nblocks", "range", "parent", ""};
    SEXP out = PROTECT(Rf_mkNamed(VECSXP, names));
    SET_VECTOR_ELT(out, 0, Rf_ScalarInteger(n));
    SET_VECTOR_ELT(out, 1, perm);
    SET_VECTOR_ELT(out, 2, iperm);
    SET_VECTOR_ELT(out, 3, Rf_ScalarInteger(win));
    SET_VECTOR_ELT(out, 4, Rf_ScalarInteger(nb));
    SET_VECTOR_ELT(out, 5, rng);
    SET_VECTOR_ELT(out, 6, par);
    UNPROTECT(7);
    return out;
}

/* ---- handles ------------------------------------------------------------- */

typedef struct {
    stiles_lib* L;          /* the copy this handle runs in                  */
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

/* The handle is the only one in its copy, so quitting here releases that
 * copy's worker threads and core claim without touching any other handle,
 * and leaves the copy clean for the next handle the pool gives it to. */
static void sTiles_finalize(SEXP ext) {
    sTiles_ctx* c = (sTiles_ctx*) R_ExternalPtrAddr(ext);
    if (!c) return;
    stiles_lib* L = c->L;
    if (c->handle) L->freeGroup(c->group);   /* frees before we drop row/col */
    free(c->row);
    free(c->col);
    free(c);
    R_ClearExternalPtr(ext);
    L->quit();
    L->busy = 0;
}

static void ensure_selinv(sTiles_ctx* c) {
    if (!c->inverse)
        Rf_error("this handle was built without inverse=TRUE; "
                 "the selected inverse is unavailable");
    require_factored(c);
    if (!c->selinv_done) {
        c->L->bind(c->group, 0, &c->handle);
        int rc = c->L->selinv(c->group, 0, &c->handle);
        c->L->unbind(c->group, 0, &c->handle);
        if (rc != 0) Rf_error("sTiles_selinv failed (status %d)", rc);
        c->selinv_done = 1;
    }
}

/* PHASE 1 -- preprocessing (symbolic) only: create + assign_graph + init_group.
 * i,j are 0-based lower-triangle COO (INTSXP). No numeric values, no Cholesky.
 * This is the ordering bake-off + tile layout; time it separately from the
 * numeric factorization done by sTiles_factorize_R. lib_ is a free copy from
 * the R-side pool. */
SEXP sTiles_analyze_R(SEXP lib_, SEXP i_, SEXP j_, SEXP n_, SEXP cores_,
                      SEXP mode_, SEXP ts_, SEXP inv_, SEXP group_,
                      SEXP loglevel_) {
    stiles_lib* L     = lib_from(lib_);
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
    if (L->busy)
        Rf_error("sTiles library copy already holds a live handle");
    if (group != 0)
        Rf_error("group must be 0: libstiles groups are not usable as separate "
                 "handles");

    L->set_log_level(loglvl);
    L->expert_user();
    L->set_tile_size(ts);
    L->set_tile_type_mode(mode);

    sTiles_ctx* c = (sTiles_ctx*) calloc(1, sizeof(sTiles_ctx));
    if (!c) Rf_error("out of memory");
    c->L = L;
    c->group = group; c->n = n; c->nnz = nnz; c->inverse = inverse;

    /* ctx-owned copies of the index arrays (sTiles retains these pointers). */
    c->row = (int*) malloc((size_t) nnz * sizeof(int));
    c->col = (int*) malloc((size_t) nnz * sizeof(int));
    if (!c->row || !c->col) { free(c->row); free(c->col); free(c);
                              Rf_error("out of memory"); }
    memcpy(c->row, INTEGER(i_), (size_t) nnz * sizeof(int));
    memcpy(c->col, INTEGER(j_), (size_t) nnz * sizeof(int));

    /* One group, and it gets the graph below. Creating extra groups and leaving
     * them empty makes Setup_all_teams fail on the empty one. */
    const int ng = 1;
    int* calls  = (int*) R_alloc(ng, sizeof(int));
    int* coresv = (int*) R_alloc(ng, sizeof(int));
    int* ctype  = (int*) R_alloc(ng, sizeof(int));
    for (int g = 0; g < ng; ++g) { calls[g] = 1; coresv[g] = cores; ctype[g] = 0; }
    /* sTiles_create wants const bool*; build a _Bool array. */
    _Bool* ginv_b = (_Bool*) R_alloc(ng, sizeof(_Bool));
    for (int g = 0; g < ng; ++g) ginv_b[g] = inverse ? 1 : 0;

    if (L->create(&c->handle, ng, calls, coresv, ctype, ginv_b) != 0) {
        free(c->row); free(c->col); free(c);
        L->quit();
        Rf_error("sTiles_create failed");
    }

    L->assign_graph_one_call(group, 0, &c->handle, n, nnz, c->row, c->col);
    if (L->init_group(group, &c->handle) != 0) {
        L->freeGroup(group);
        free(c->row); free(c->col); free(c);
        L->quit();
        Rf_error("sTiles_init_group failed");
    }

    /* prot = the copy's external pointer, so the copy outlives the handle */
    SEXP ext = PROTECT(R_MakeExternalPtr(c, R_NilValue, lib_));
    R_RegisterCFinalizerEx(ext, sTiles_finalize, TRUE);
    L->busy = 1;
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
    c->L->assign_values(c->group, 0, &c->handle, REAL(x_));
    c->L->bind(c->group, 0, &c->handle);
    int rc = c->L->chol(c->group, 0, &c->handle);
    c->L->unbind(c->group, 0, &c->handle);
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
    return Rf_ScalarReal(c->L->get_chol_timing(c->group, 0, &c->handle));
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
    return Rf_ScalarReal(c->L->get_selinv_timing(c->group, 0, &c->handle));
}

SEXP sTiles_logdet_R(SEXP ext) {
    sTiles_ctx* c = ctx_from(ext);
    require_factored(c);
    return Rf_ScalarReal(c->L->get_logdet(c->group, 0, &c->handle));
}

SEXP sTiles_nnz_factor_R(SEXP ext) {
    sTiles_ctx* c = ctx_from(ext);
    return Rf_ScalarReal((double) c->L->get_nnz_factor(c->group, 0, &c->handle));
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
        d[i] = c->L->get_selinv_elm(c->group, 0, i, i, &c->handle);
    UNPROTECT(1);
    return out;
}

/* i, j are 1-based from R; converted to 0-based here. */
SEXP sTiles_selinv_elm_R(SEXP ext, SEXP i_, SEXP j_) {
    sTiles_ctx* c = ctx_from(ext);
    ensure_selinv(c);
    return Rf_ScalarReal(c->L->get_selinv_elm(
        c->group, 0, Rf_asInteger(i_) - 1, Rf_asInteger(j_) - 1, &c->handle));
}

SEXP sTiles_chol_elm_R(SEXP ext, SEXP i_, SEXP j_) {
    sTiles_ctx* c = ctx_from(ext);
    require_factored(c);
    return Rf_ScalarReal(c->L->get_chol_elm(
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
    double* vals = c->L->get_selinv_row(c->group, 0, Rf_asInteger(node_) - 1,
                                        nb, m, &c->handle);
    if (!vals) Rf_error("sTiles_get_selinv_row returned NULL");
    SEXP out = PROTECT(Rf_allocVector(REALSXP, m));
    memcpy(REAL(out), vals, (size_t) m * sizeof(double));
    free(vals);   /* libstiles malloc'd it */
    UNPROTECT(1);
    return out;
}

/* Z at every stored entry of Q's lower triangle, in row/col order. */
SEXP sTiles_selinv_pattern_R(SEXP ext) {
    sTiles_ctx* c = ctx_from(ext);
    ensure_selinv(c);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, c->nnz));
    double* z = REAL(out);
    for (long long k = 0; k < c->nnz; ++k)
        z[k] = c->L->get_selinv_elm(c->group, 0, c->row[k], c->col[k],
                                    &c->handle);
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

    c->L->bind(c->group, 0, &c->handle);
    int rc;
    if      (which == 1) rc = c->L->solve_L (c->group, 0, &c->handle, work, nrhs);
    else if (which == 2) rc = c->L->solve_LT(c->group, 0, &c->handle, work, nrhs);
    else                 rc = c->L->solve_LLT(c->group, 0, &c->handle, work, nrhs);
    c->L->unbind(c->group, 0, &c->handle);
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
    int m = c->L->get_logical_element_perm(c->group, 0, &c->handle, buf);
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

SEXP sTiles_version_R(SEXP lib_) {
    const char* v = lib_from(lib_)->get_version();
    return Rf_mkString(v ? v : "unknown");
}

/* ---- registration -------------------------------------------------------- */
static const R_CallMethodDef CallEntries[] = {
    {"sTiles_lib_open_R",      (DL_FUNC) &sTiles_lib_open_R,        1},
    {"sTiles_lib_busy_R",      (DL_FUNC) &sTiles_lib_busy_R,        1},
    {"sTiles_lib_close_R",     (DL_FUNC) &sTiles_lib_close_R,       1},
    {"sTiles_analyze_R",       (DL_FUNC) &sTiles_analyze_R,        10},
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
    {"sTiles_selinv_pattern_R", (DL_FUNC) &sTiles_selinv_pattern_R, 1},
    {"sTiles_solve_R",         (DL_FUNC) &sTiles_solve_R,           3},
    {"sTiles_perm_R",          (DL_FUNC) &sTiles_perm_R,            1},
    {"sTiles_free_R",          (DL_FUNC) &sTiles_free_R,            1},
    {"sTiles_version_R",       (DL_FUNC) &sTiles_version_R,         1},
    {"sTiles_has_ordering_R",  (DL_FUNC) &sTiles_has_ordering_R,    1},
    {"sTiles_ordering_R",      (DL_FUNC) &sTiles_ordering_R,        6},
    {NULL, NULL, 0}
};

/* Named R_init_sTiles to match the package/DLL name "sTiles". */
void R_init_sTiles(DllInfo* dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
