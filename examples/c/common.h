/* Helpers shared by the C examples, kept out of them so each example shows
 * only the sTiles calls. Nothing here is part of sTiles. */
#ifndef STILES_EXAMPLES_COMMON_H
#define STILES_EXAMPLES_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* Stop with a message when an sTiles call returns a non-zero status. */
#define CHECK(call) do {                                                  \
        int rc_ = (call);                                                 \
        if (rc_ != 0) {                                                   \
            fprintf(stderr, "%s failed (status %d)\n", #call, rc_);       \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

/* A symmetric matrix as sTiles takes it: ONE triangle (here the lower one,
 * row >= col) in coordinate form, 0-based, listed row by row. */
typedef struct {
    int n, nnz;
    int *rows, *cols;
    double *vals;
} lower_coo;

/* n x n tridiagonal matrix with `diag` on the diagonal and `off` next to it. */
static inline lower_coo tridiagonal(int n, double diag, double off)
{
    lower_coo A = { n, 2 * n - 1, NULL, NULL, NULL };
    A.rows = malloc(A.nnz * sizeof *A.rows);
    A.cols = malloc(A.nnz * sizeof *A.cols);
    A.vals = malloc(A.nnz * sizeof *A.vals);
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (i > 0) { A.rows[k] = i; A.cols[k] = i - 1; A.vals[k] = off; k++; }
        A.rows[k] = i; A.cols[k] = i; A.vals[k] = diag; k++;
    }
    return A;
}

static inline void free_coo(lower_coo *A)
{
    free(A->rows); free(A->cols); free(A->vals);
}

/* max_i |(A x - b)_i|, using both triangles of the symmetric A. */
static inline double max_residual(const lower_coo *A, const double *x, const double *b)
{
    double *r = malloc(A->n * sizeof *r), worst = 0.0;
    for (int i = 0; i < A->n; i++) r[i] = -b[i];
    for (int k = 0; k < A->nnz; k++) {
        int i = A->rows[k], j = A->cols[k];
        r[i] += A->vals[k] * x[j];
        if (i != j) r[j] += A->vals[k] * x[i];
    }
    for (int i = 0; i < A->n; i++) if (fabs(r[i]) > worst) worst = fabs(r[i]);
    free(r);
    return worst;
}

/* Wall-clock seconds. */
static inline double now(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

#endif
