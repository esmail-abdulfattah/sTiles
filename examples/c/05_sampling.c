/* Draw samples from N(0, Q^-1), the Gaussian with precision matrix Q.
 *
 * If Q = C C^T, then x = C^-T z with z ~ N(0, I) has covariance Q^-1.
 * sTiles_solve_LT computes exactly that, in the original ordering of Q.
 *
 *   make 05_sampling && ./05_sampling
 */
#include "stiles.h"
#include "common.h"

/* A standard normal draw (Box-Muller). Use your own generator in practice. */
static double randn(void)
{
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0), u2 = rand() / (RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.141592653589793 * u2);
}

int main(void)
{
    lower_coo Q = tridiagonal(1000, 4.0, -1.0);
    int n = Q.n, draws = 10000;

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
    CHECK(sTiles_selinv(0, 0, &h));

    /* n x draws standard normals, column by column; one call turns every
     * column into a draw from N(0, Q^-1), in place. */
    srand(1);
    double *X = malloc((size_t)n * draws * sizeof *X);
    for (size_t i = 0; i < (size_t)n * draws; i++) X[i] = randn();
    CHECK(sTiles_solve_LT(0, 0, &h, X, draws));
    sTiles_unbind(0, 0, &h);

    /* Check: the sample variance of x[0] matches the exact one. */
    double sum = 0.0, sum2 = 0.0;
    for (int k = 0; k < draws; k++) {
        double x0 = X[(size_t)k * n];
        sum += x0; sum2 += x0 * x0;
    }
    double mean = sum / draws;
    printf("exact variance of x[0]   = %.4f\n", sTiles_get_selinv_elm(0, 0, 0, 0, &h));
    printf("sample variance of x[0]  = %.4f\n", (sum2 - draws * mean * mean) / (draws - 1));

    sTiles_quit();
    free(X);
    free_coo(&Q);
    return 0;
}
