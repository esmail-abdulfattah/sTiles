/* Read entries of Q^-1 without forming the dense inverse: the diagonal and
 * single entries.
 *
 *   make 03_selected_inverse && ./03_selected_inverse
 */
#include "stiles.h"
#include "common.h"

int main(void)
{
    lower_coo Q = tridiagonal(200, 4.0, -1.0);
    int n = Q.n;

    /* The selected inverse needs inverse = true at creation. */
    sTiles_set_log_level(-1);
    void *h = NULL;
    int  calls[] = {1}, cores[] = {2}, variant[] = {0};
    bool inverse[] = {true};
    CHECK(sTiles_create(&h, 1, calls, cores, variant, inverse));
    CHECK(sTiles_assign_graph_one_call(0, 0, &h, n, Q.nnz, Q.rows, Q.cols));
    CHECK(sTiles_init_group(0, &h));
    CHECK(sTiles_assign_values(0, 0, &h, Q.vals));
    sTiles_bind(0, 0, &h);
    CHECK(sTiles_chol(0, 0, &h));
    CHECK(sTiles_selinv(0, 0, &h));       /* after the Cholesky */
    sTiles_unbind(0, 0, &h);

    /* All n variances in one call. */
    double *d = malloc(n * sizeof *d);
    sTiles_get_selinv_diag(0, 0, &h, n, d);
    printf("diag(Qinv)[0..2]  = %.6f %.6f %.6f\n", d[0], d[1], d[2]);

    /* One entry. Indices are 0-based, in the original ordering of Q. */
    printf("Qinv[10, 11]      = %.6f\n", sTiles_get_selinv_elm(0, 0, 10, 11, &h));

    /* Entries are exact inside the pattern of the Cholesky factor, which
     * sTiles stores in whole tiles, and 0 outside it. Ask only for entries
     * you know are in the pattern: the diagonal and the non-zeros of Q are. */
    printf("Qinv[10, 150]     = %.1f   (outside the pattern)\n",
           sTiles_get_selinv_elm(0, 0, 10, 150, &h));

    sTiles_quit();
    free(d);
    free_coo(&Q);
    return 0;
}
