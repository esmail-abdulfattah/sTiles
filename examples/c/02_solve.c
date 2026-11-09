/* Solve Q x = b for one right-hand side, then for several at once.
 *
 *   make 02_solve && ./02_solve
 */
#include "stiles.h"
#include "common.h"

int main(void)
{
    lower_coo Q = tridiagonal(1000, 4.0, -1.0);
    int n = Q.n;

    /* Handle, analysis and Cholesky, as in 01_quickstart.c (no inverse). */
    sTiles_set_log_level(-1);
    void *h = NULL;
    int  calls[] = {1}, cores[] = {2}, variant[] = {0};
    bool inverse[] = {false};
    CHECK(sTiles_create(&h, 1, calls, cores, variant, inverse));
    CHECK(sTiles_assign_graph_one_call(0, 0, &h, n, Q.nnz, Q.rows, Q.cols));
    CHECK(sTiles_init_group(0, &h));
    CHECK(sTiles_assign_values(0, 0, &h, Q.vals));
    sTiles_bind(0, 0, &h);
    CHECK(sTiles_chol(0, 0, &h));

    /* One right-hand side. The solve overwrites its input with the solution,
     * so keep a copy of b if you still need it. */
    double *b = malloc(n * sizeof *b), *x = malloc(n * sizeof *x);
    for (int i = 0; i < n; i++) b[i] = x[i] = i + 1.0;
    CHECK(sTiles_solve_LLT(0, 0, &h, x, 1));
    printf("one rhs:   max |Q x - b| = %.1e\n", max_residual(&Q, x, b));

    /* Several right-hand sides: an n x k array stored column by column
     * (column j starts at B + j*n), all solved in one call. */
    int k = 5;
    double *B = malloc((size_t)n * k * sizeof *B), *X = malloc((size_t)n * k * sizeof *X);
    for (int i = 0; i < n * k; i++) B[i] = X[i] = (double)rand() / RAND_MAX - 0.5;
    CHECK(sTiles_solve_LLT(0, 0, &h, X, k));
    double worst = 0.0;
    for (int j = 0; j < k; j++) {
        double r = max_residual(&Q, X + (size_t)j * n, B + (size_t)j * n);
        if (r > worst) worst = r;
    }
    printf("five rhs:  max |Q X - B| = %.1e\n", worst);

    sTiles_unbind(0, 0, &h);
    sTiles_quit();
    free(b); free(x); free(B); free(X);
    free_coo(&Q);
    return 0;
}
