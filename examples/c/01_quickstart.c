/* Factorize a sparse SPD matrix, then get the log-determinant, solve a
 * linear system and read the marginal variances.
 *
 *   make 01_quickstart && ./01_quickstart
 */
#include "stiles.h"
#include "common.h"

int main(void)
{
    /* A 1000 x 1000 tridiagonal matrix: 4 on the diagonal, -1 next to it. */
    lower_coo Q = tridiagonal(1000, 4.0, -1.0);
    int n = Q.n;

    /* 0. Print errors only (the default also prints [TIME] progress rows). */
    sTiles_set_log_level(-1);

    /* 1. Create a handle: one group (one sparsity pattern) holding one call
     *    (one matrix), on 2 cores, standard Cholesky, with the selected
     *    inverse enabled. Every call below then names group 0, call 0. */
    void *h = NULL;
    int  calls[] = {1}, cores[] = {2}, variant[] = {0};
    bool inverse[] = {true};
    CHECK(sTiles_create(&h, 1, calls, cores, variant, inverse));

    /* 2. Analysis: hand over the pattern, then compute the ordering and tile
     *    layout. sTiles keeps the rows/cols arrays: free them only at the end. */
    CHECK(sTiles_assign_graph_one_call(0, 0, &h, n, Q.nnz, Q.rows, Q.cols));
    CHECK(sTiles_init_group(0, &h));

    /* 3. Numeric phase: values in the same order as rows/cols, then the
     *    Cholesky factorization and the selected inverse. Work runs between
     *    bind and unbind, which start and park the thread team. */
    CHECK(sTiles_assign_values(0, 0, &h, Q.vals));
    sTiles_bind(0, 0, &h);
    CHECK(sTiles_chol(0, 0, &h));
    CHECK(sTiles_selinv(0, 0, &h));
    sTiles_unbind(0, 0, &h);

    /* 4. Results. */
    printf("log|Q|            = %.6f\n", sTiles_get_logdet(0, 0, &h));

    double *b = malloc(n * sizeof *b), *x = malloc(n * sizeof *x);
    for (int i = 0; i < n; i++) b[i] = x[i] = 1.0;
    sTiles_bind(0, 0, &h);
    CHECK(sTiles_solve_LLT(0, 0, &h, x, 1));   /* x = Q^-1 b, in place */
    sTiles_unbind(0, 0, &h);
    printf("max |Q x - b|     = %.1e\n", max_residual(&Q, x, b));

    double *v = malloc(n * sizeof *v);           /* diag(Q^-1) */
    sTiles_get_selinv_diag(0, 0, &h, n, v);
    printf("first variances   = %.6f %.6f %.6f\n", v[0], v[1], v[2]);

    /* 5. Release the solver, then your arrays. */
    sTiles_quit();
    free(b); free(x); free(v);
    free_coo(&Q);
    return 0;
}
