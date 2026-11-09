/* Analyze the sparsity pattern once, then factorize many matrices that
 * share it. This is the loop of a hyperparameter search or an optimizer.
 *
 *   make 04_reuse_analysis && ./04_reuse_analysis
 */
#include "stiles.h"
#include "common.h"

int main(void)
{
    /* T: tridiagonal with 2 on the diagonal and -1 next to it. The matrices
     * below are Q(kappa) = T + kappa I: same non-zeros, different values. */
    lower_coo T = tridiagonal(100000, 2.0, -1.0);
    int n = T.n;
    double *vals = malloc(T.nnz * sizeof *vals);

    sTiles_set_log_level(-1);
    void *h = NULL;
    int  calls[] = {1}, cores[] = {2}, variant[] = {0};
    bool inverse[] = {false};
    CHECK(sTiles_create(&h, 1, calls, cores, variant, inverse));

    /* Step 1, once: ordering and tile layout. They depend only on WHERE the
     * non-zeros are, not on their values. */
    double t0 = now();
    CHECK(sTiles_assign_graph_one_call(0, 0, &h, n, T.nnz, T.rows, T.cols));
    CHECK(sTiles_init_group(0, &h));
    printf("analysis:  %.3f s  (paid once)\n", now() - t0);

    /* Step 2, per value set: new values in the same order, numeric Cholesky. */
    double kappas[] = {0.1, 1.0, 10.0};
    for (int r = 0; r < 3; r++) {
        for (int k = 0; k < T.nnz; k++)
            vals[k] = T.vals[k] + (T.rows[k] == T.cols[k] ? kappas[r] : 0.0);

        CHECK(sTiles_assign_values(0, 0, &h, vals));
        sTiles_bind(0, 0, &h);
        CHECK(sTiles_chol(0, 0, &h));
        sTiles_unbind(0, 0, &h);

        printf("kappa = %5.1f   log|Q| = %12.4f   cholesky: %.4f s\n", kappas[r],
               sTiles_get_logdet(0, 0, &h), sTiles_get_chol_timing(0, 0, &h));
    }

    sTiles_quit();
    free(vals);
    free_coo(&T);
    return 0;
}
