/* Factorize your own matrix, read from a Matrix Market (.mtx) file.
 *
 *   make 06_matrix_market && ./06_matrix_market              # ../data/grid40.mtx
 *   ./06_matrix_market my_matrix.mtx
 */
#include <string.h>
#include "stiles.h"
#include "common.h"

/* Read a "coordinate real" .mtx file and keep its lower triangle, 0-based.
 * A "symmetric" file stores one triangle already; a general one stores both,
 * and the upper entries are dropped. */
static lower_coo read_mtx(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    char line[1024];
    do {
        if (!fgets(line, sizeof line, f)) { fprintf(stderr, "%s: empty\n", path); exit(1); }
    } while (line[0] == '%');

    int m, entries;
    lower_coo A = {0};
    sscanf(line, "%d %d %d", &A.n, &m, &entries);
    A.rows = malloc(entries * sizeof *A.rows);
    A.cols = malloc(entries * sizeof *A.cols);
    A.vals = malloc(entries * sizeof *A.vals);
    for (int e = 0; e < entries; e++) {
        int i, j; double v;
        if (fscanf(f, "%d %d %lf", &i, &j, &v) != 3) { fprintf(stderr, "%s: bad entry\n", path); exit(1); }
        if (i < j) continue;
        A.rows[A.nnz] = i - 1; A.cols[A.nnz] = j - 1; A.vals[A.nnz] = v; A.nnz++;
    }
    fclose(f);
    return A;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "../data/grid40.mtx";
    lower_coo Q = read_mtx(path);
    printf("%s: %d x %d, %d entries in the lower triangle\n", path, Q.n, Q.n, Q.nnz);

    sTiles_set_log_level(-1);
    void *h = NULL;
    int  calls[] = {1}, cores[] = {2}, variant[] = {0};
    bool inverse[] = {false};
    CHECK(sTiles_create(&h, 1, calls, cores, variant, inverse));

    double t0 = now();
    CHECK(sTiles_assign_graph_one_call(0, 0, &h, Q.n, Q.nnz, Q.rows, Q.cols));
    CHECK(sTiles_init_group(0, &h));
    double analysis = now() - t0;

    CHECK(sTiles_assign_values(0, 0, &h, Q.vals));
    sTiles_bind(0, 0, &h);
    CHECK(sTiles_chol(0, 0, &h));
    sTiles_unbind(0, 0, &h);

    printf("log|Q| = %.6f\n", sTiles_get_logdet(0, 0, &h));
    printf("non-zeros in L = %lld, analysis %.3f s, cholesky %.4f s\n",
           sTiles_get_nnz_factor(0, 0, &h), analysis, sTiles_get_chol_timing(0, 0, &h));

    sTiles_quit();
    free_coo(&Q);
    return 0;
}
